"""Fail-closed planning for replacing unfinished tail simulations.

The replacement controller deliberately runs outside the immutable generator
runtime.  It audits one or more stopped source campaigns, freezes the exact
deficit and a non-overlapping tail seed, then delegates prepare/generate/finalize
to the original generator checkout and Python interpreter.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import Config, validate_config
from .dataset_state import (
    DONE_CONFIG_STATE_KEY,
    FINGERPRINT_VERSION,
    RUNTIME_CONFIG_FIELDS,
    SAMPLE_GENERATOR_SCHEMA_VERSION,
    generation_config_state,
    sample_generation_fingerprint,
)


SCHEMA_VERSION = 1
PLAN_NAME = "REPLACEMENT_PLAN.json"
DONE_NAME = "REPLACEMENT_DONE.json"
SEED_STRIDE = 7919
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FINAL_RE = re.compile(r"^shard_(\d{5})\.done\.json$")
_FRAGMENT_RE = re.compile(r"^fragment_(\d{8})\.done\.json$")


class ReplacementPlanError(ValueError):
    """Raised when a replacement cannot be proven safe."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplacementPlanError(
            f"{description} at {path} is unreadable: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReplacementPlanError(f"{description} at {path} is not a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_sha256(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReplacementPlanError(f"{description} is not a lowercase SHA-256")
    return value


def _source_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "samples" / "PLAN.json").is_file():
        return root
    if root.name == "samples" and (root / "PLAN.json").is_file():
        return root.parent
    raise ReplacementPlanError(
        f"source {root} has no samples/PLAN.json"
    )


def _absolute_executable(path: str | os.PathLike[str]) -> Path:
    """Return an absolute executable path without dereferencing venv symlinks."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.absolute()


def _config_from_source_plan(root: Path, plan: Mapping[str, Any]) -> Config:
    state = plan.get(DONE_CONFIG_STATE_KEY)
    layout = plan.get("layout")
    if not isinstance(state, Mapping) or not isinstance(layout, Mapping):
        raise ReplacementPlanError(f"{root}: PLAN config/layout is malformed")
    stored = state.get("config")
    if not isinstance(stored, Mapping):
        raise ReplacementPlanError(f"{root}: PLAN config snapshot is malformed")
    if (
        state.get("version") != FINGERPRINT_VERSION
        or state.get("generator_schema_version") != SAMPLE_GENERATOR_SCHEMA_VERSION
        or state.get("algorithm") != "sha256"
    ):
        raise ReplacementPlanError(f"{root}: unsupported PLAN config fingerprint schema")

    expected_scientific = {
        field.name
        for field in dataclasses.fields(Config)
        if field.name not in RUNTIME_CONFIG_FIELDS
    }
    if set(stored) != expected_scientific:
        missing = sorted(expected_scientific - set(stored))
        extra = sorted(set(stored) - expected_scientific)
        raise ReplacementPlanError(
            f"{root}: PLAN scientific config keys differ from this schema "
            f"(missing={missing}, extra={extra})"
        )
    stored_dict = dict(stored)
    if sample_generation_fingerprint(stored_dict) != state.get("sha256"):
        raise ReplacementPlanError(f"{root}: PLAN config fingerprint is inconsistent")

    compute_chunk_size = layout.get("compute_chunk_size")
    if isinstance(compute_chunk_size, bool) or not isinstance(compute_chunk_size, int):
        raise ReplacementPlanError(f"{root}: PLAN compute_chunk_size is malformed")
    values = {
        **stored_dict,
        "dataset_name": "source-audit",
        "out_dir": str(root),
        "workers": 1,
        "force": False,
        "compute_chunk_size": compute_chunk_size,
    }
    try:
        cfg = Config(**values)
        validate_config(cfg)
    except (TypeError, ValueError) as exc:
        raise ReplacementPlanError(f"{root}: PLAN config is invalid: {exc}") from exc
    return cfg


def _validate_source_plan(root: Path) -> tuple[dict[str, Any], Config]:
    plan_path = root / "samples" / "PLAN.json"
    plan = _read_json(plan_path, "source PLAN")
    persisted = dict(plan)
    recorded_sha = persisted.pop("plan_sha256", None)
    if _require_sha256(recorded_sha, f"{root}: plan_sha256") != _canonical_sha256(
        persisted
    ):
        raise ReplacementPlanError(f"{root}: PLAN fingerprint is inconsistent")
    if plan.get("schema_version") != 1 or plan.get("kind") != "dlcoalsim_generation_plan":
        raise ReplacementPlanError(f"{root}: unsupported generation PLAN schema")

    cfg = _config_from_source_plan(root, plan)
    layout = plan["layout"]
    expected_layout = {
        "n_samples": cfg.n_samples,
        "shard_size": cfg.shard_size,
        "compute_chunk_size": cfg.compute_chunk_size,
        "n_shards": (cfg.n_samples + cfg.shard_size - 1) // cfg.shard_size,
        "n_chunks": sum(
            (
                min(cfg.shard_size, cfg.n_samples - start)
                + cfg.compute_chunk_size
                - 1
            )
            // cfg.compute_chunk_size
            for start in range(0, cfg.n_samples, cfg.shard_size)
        ),
        "sample_id_rule": "sim_{global_sample_index:08d}",
        "seed_rule": "config.seed + global_sample_index * 7919",
        "partition_rule": "chunk_idx_mod_world_size",
        "array_schema_version": SAMPLE_GENERATOR_SCHEMA_VERSION,
    }
    if layout != expected_layout or plan.get("layout_sha256") != _canonical_sha256(
        expected_layout
    ):
        raise ReplacementPlanError(f"{root}: PLAN fragment layout is inconsistent")

    for name in (
        "empirical_maps_sha256",
        "empirical_ne_templates_sha256",
        "runtime_environment_sha256",
    ):
        _require_sha256(plan.get(name), f"{root}: {name}")
    return plan, cfg


def _shard_tasks(n_samples: int, shard_size: int) -> list[tuple[int, int, int]]:
    return [
        (index, start, min(shard_size, n_samples - start))
        for index, start in enumerate(range(0, n_samples, shard_size))
    ]


def _chunk_tasks(
    n_samples: int, shard_size: int, chunk_size: int
) -> list[tuple[int, int, int, int]]:
    tasks: list[tuple[int, int, int, int]] = []
    chunk_index = 0
    for shard_index, shard_start, shard_count in _shard_tasks(
        n_samples, shard_size
    ):
        for start in range(
            shard_start, shard_start + shard_count, chunk_size
        ):
            count = min(chunk_size, shard_start + shard_count - start)
            tasks.append((chunk_index, shard_index, start, count))
            chunk_index += 1
    return tasks


def _artifact_record(
    root: Path,
    relative_path: str,
    record: object,
    description: str,
) -> dict[str, Any]:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReplacementPlanError(f"{description} has an unsafe artifact path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReplacementPlanError(
            f"{description} artifact escapes its source root"
        ) from exc
    if not isinstance(record, Mapping):
        raise ReplacementPlanError(f"{description} has a malformed file record")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not path.is_file()
        or path.stat().st_size != size
    ):
        raise ReplacementPlanError(
            f"{description} artifact {relative_path} is missing or has the wrong size"
        )
    return {
        "path": relative.as_posix(),
        "size_bytes": size,
        "sha256": _require_sha256(digest, f"{description} artifact sha256"),
    }


def _receipt_record(
    *,
    root: Path,
    receipt_path: Path,
    plan_sha256: str,
    kind: str,
    expected_index: int,
    expected_shard: int,
    expected_start: int,
    expected_count: int,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path, f"{kind} receipt")
    persisted = dict(receipt)
    recorded = persisted.pop("receipt_sha256", None)
    if _require_sha256(
        recorded, f"{receipt_path}: receipt_sha256"
    ) != _canonical_sha256(persisted):
        raise ReplacementPlanError(f"{receipt_path}: receipt fingerprint is inconsistent")
    if receipt.get("schema_version") != 1 or receipt.get("plan_sha256") != plan_sha256:
        raise ReplacementPlanError(f"{receipt_path}: receipt schema/PLAN mismatch")
    result = receipt.get("result")
    files = receipt.get("files")
    if not isinstance(result, Mapping) or not isinstance(files, Mapping):
        raise ReplacementPlanError(f"{receipt_path}: receipt payload is malformed")

    if kind == "final":
        if (
            result.get("shard_idx") != expected_shard
            or result.get("start") != expected_start
            or result.get("n_samples") != expected_count
        ):
            raise ReplacementPlanError(f"{receipt_path}: final range is inconsistent")
        relative_npz = f"samples/shard_{expected_shard:05d}.npz"
        if result.get("shard_path") != relative_npz:
            raise ReplacementPlanError(f"{receipt_path}: final artifact path is inconsistent")
    else:
        if (
            result.get("chunk_idx") != expected_index
            or result.get("shard_idx") != expected_shard
            or result.get("start") != expected_start
            or result.get("n_samples") != expected_count
        ):
            raise ReplacementPlanError(f"{receipt_path}: fragment range is inconsistent")
        relative_npz = f"samples/.fragments/fragment_{expected_index:08d}.npz"
        if result.get("fragment_path") != relative_npz:
            raise ReplacementPlanError(
                f"{receipt_path}: fragment artifact path is inconsistent"
            )

    npz = _artifact_record(
        root, relative_npz, files.get("npz"), str(receipt_path)
    )
    metadata = _artifact_record(
        root,
        str(Path(relative_npz).with_suffix(".jsonl.gz")),
        files.get("metadata_jsonl_gz"),
        str(receipt_path),
    )
    return {
        "kind": kind,
        "receipt_path": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": recorded,
        "index": expected_index,
        "shard_idx": expected_shard,
        "start": expected_start,
        "n_samples": expected_count,
        "files": {"npz": npz, "metadata_jsonl_gz": metadata},
    }


def _scan_source(root: Path) -> dict[str, Any]:
    plan, cfg = _validate_source_plan(root)
    samples = root / "samples"
    final_tasks = {
        index: (index, start, count)
        for index, start, count in _shard_tasks(cfg.n_samples, cfg.shard_size)
    }
    chunk_tasks = {
        index: (index, shard, start, count)
        for index, shard, start, count in _chunk_tasks(
            cfg.n_samples, cfg.shard_size, cfg.compute_chunk_size
        )
    }

    final_paths: dict[int, Path] = {}
    final_dir = samples / ".receipts"
    if final_dir.exists():
        for path in final_dir.glob("*.done.json"):
            match = _FINAL_RE.fullmatch(path.name)
            if match is None:
                raise ReplacementPlanError(f"{root}: unexpected final receipt {path.name}")
            index = int(match.group(1))
            if index not in final_tasks or index in final_paths:
                raise ReplacementPlanError(f"{root}: duplicate/unexpected final receipt {path.name}")
            final_paths[index] = path

    fragment_paths: dict[int, Path] = {}
    fragment_dir = samples / ".fragments"
    if fragment_dir.exists():
        for path in fragment_dir.glob("*.done.json"):
            match = _FRAGMENT_RE.fullmatch(path.name)
            if match is None:
                raise ReplacementPlanError(
                    f"{root}: unexpected fragment receipt {path.name}"
                )
            index = int(match.group(1))
            if index not in chunk_tasks or index in fragment_paths:
                raise ReplacementPlanError(
                    f"{root}: duplicate/unexpected fragment receipt {path.name}"
                )
            fragment_paths[index] = path

    selected: list[dict[str, Any]] = []
    finalized_shards = set(final_paths)
    for index, path in sorted(final_paths.items()):
        _, start, count = final_tasks[index]
        selected.append(
            _receipt_record(
                root=root,
                receipt_path=path,
                plan_sha256=str(plan["plan_sha256"]),
                kind="final",
                expected_index=index,
                expected_shard=index,
                expected_start=start,
                expected_count=count,
            )
        )
    ignored_fragments = 0
    for index, path in sorted(fragment_paths.items()):
        _, shard, start, count = chunk_tasks[index]
        if shard in finalized_shards:
            ignored_fragments += 1
            continue
        selected.append(
            _receipt_record(
                root=root,
                receipt_path=path,
                plan_sha256=str(plan["plan_sha256"]),
                kind="fragment",
                expected_index=index,
                expected_shard=shard,
                expected_start=start,
                expected_count=count,
            )
        )

    ordered = sorted(selected, key=lambda item: (item["start"], item["n_samples"]))
    previous_end = 0
    for index, item in enumerate(ordered):
        start = int(item["start"])
        end = start + int(item["n_samples"])
        if index and start < previous_end:
            raise ReplacementPlanError(f"{root}: selected receipt ranges overlap")
        previous_end = end

    completed = sum(int(item["n_samples"]) for item in ordered)
    return {
        "root": str(root),
        "plan_file_sha256": _file_sha256(samples / "PLAN.json"),
        "plan_sha256": plan["plan_sha256"],
        "planned_n_samples": cfg.n_samples,
        "seed": cfg.seed,
        "seed_stride": SEED_STRIDE,
        "planned_seed_start": cfg.seed,
        "planned_seed_stop_exclusive": cfg.seed + cfg.n_samples * SEED_STRIDE,
        "completed_samples": completed,
        "selected_final_receipts": len(final_paths),
        "selected_fragment_receipts": len(ordered) - len(final_paths),
        "ignored_fragments_covered_by_final": ignored_fragments,
        "selected_receipts_sha256": _canonical_sha256(ordered),
        "selected_receipts": ordered,
        "scientific_config": plan[DONE_CONFIG_STATE_KEY]["config"],
        "empirical_maps_sha256": plan["empirical_maps_sha256"],
        "empirical_ne_templates_sha256": plan[
            "empirical_ne_templates_sha256"
        ],
        "runtime_environment_sha256": plan["runtime_environment_sha256"],
    }


def _scientific_without_size_and_seed(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    config = snapshot.get("scientific_config")
    if not isinstance(config, Mapping):
        raise ReplacementPlanError("source snapshot lacks a scientific config")
    return {
        str(key): value
        for key, value in config.items()
        if key not in {"n_samples", "seed"}
    }


def _seed_sequences_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_seed = int(left["seed"])
    right_seed = int(right["seed"])
    if (right_seed - left_seed) % SEED_STRIDE != 0:
        return False
    offset = (right_seed - left_seed) // SEED_STRIDE
    return -int(right["planned_n_samples"]) < offset < int(left["planned_n_samples"])


def _audit_sources(source_roots: Sequence[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    if not source_roots:
        raise ReplacementPlanError("at least one --source is required")
    roots = [_source_root(path) for path in source_roots]
    if len(set(roots)) != len(roots):
        raise ReplacementPlanError("the same source root was supplied more than once")
    snapshots = [_scan_source(root) for root in roots]
    snapshots.sort(key=lambda item: (int(item["seed"]), str(item["root"])))

    reference_science = _scientific_without_size_and_seed(snapshots[0])
    reference_hashes = {
        key: snapshots[0][key]
        for key in (
            "empirical_maps_sha256",
            "empirical_ne_templates_sha256",
            "runtime_environment_sha256",
        )
    }
    for snapshot in snapshots[1:]:
        if _scientific_without_size_and_seed(snapshot) != reference_science:
            raise ReplacementPlanError(
                "source scientific configs differ in fields other than n_samples/seed"
            )
        for key, value in reference_hashes.items():
            if snapshot[key] != value:
                raise ReplacementPlanError(f"source {key} values differ")
    for index, left in enumerate(snapshots):
        for right in snapshots[index + 1 :]:
            if _seed_sequences_overlap(left, right):
                raise ReplacementPlanError(
                    f"source planned seed sequences overlap: {left['root']} and {right['root']}"
                )
    return snapshots


def _runtime_probe(
    generator_root: Path, generator_python: Path
) -> dict[str, str]:
    if not generator_root.is_dir():
        raise ReplacementPlanError(f"generator root is not a directory: {generator_root}")
    if not generator_python.is_file():
        raise ReplacementPlanError(f"generator Python does not exist: {generator_python}")
    code = (
        "import json, pathlib, simulator;"
        "from simulator.shards import (_canonical_json_sha256,"
        "_generation_runtime_state,_runtime_fingerprint_payload);"
        "state=_generation_runtime_state();"
        "print(json.dumps({'simulator_file':str(pathlib.Path(simulator.__file__).resolve()),"
        "'runtime_environment_sha256':_canonical_json_sha256("
        "_runtime_fingerprint_payload(state))},sort_keys=True))"
    )
    completed = _run_generator_subprocess(
        generator_root, generator_python, ["-c", code], module=False
    )
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError, TypeError) as exc:
        raise ReplacementPlanError("generator runtime probe returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise ReplacementPlanError("generator runtime probe returned a non-object")
    simulator_file = Path(str(value.get("simulator_file", ""))).resolve()
    try:
        simulator_file.relative_to(generator_root)
    except ValueError as exc:
        raise ReplacementPlanError(
            f"generator Python imported simulator outside --generator-root: {simulator_file}"
        ) from exc
    return {
        "simulator_file": str(simulator_file),
        "runtime_environment_sha256": _require_sha256(
            value.get("runtime_environment_sha256"),
            "probed runtime_environment_sha256",
        ),
    }


def _run_generator_subprocess(
    generator_root: Path,
    generator_python: Path,
    arguments: Sequence[str],
    *,
    module: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [str(generator_python)]
    if module:
        command.extend(["-m", "simulator"])
    command.extend(str(value) for value in arguments)
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(generator_root)
        if not old_pythonpath
        else os.pathsep.join((str(generator_root), old_pythonpath))
    )
    try:
        return subprocess.run(
            command,
            cwd=generator_root,
            env=environment,
            check=True,
            capture_output=capture_output,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise ReplacementPlanError(
            f"immutable generator command failed ({' '.join(command)}): {detail}"
        ) from exc
    except OSError as exc:
        raise ReplacementPlanError(f"cannot execute immutable generator: {exc}") from exc


def _config_argv(config: Mapping[str, Any]) -> list[str]:
    values = dict(config)
    argv: list[str] = []
    bool_fields = {
        "force",
        "compression",
        "include_truth",
        "enable_stdpopsim",
        "mut_map_scale_to_baseline",
    }
    for field in dataclasses.fields(Config):
        name = field.name
        value = values[name]
        option = "--" + name.replace("_", "-")
        if name in bool_fields:
            continue
        argv.extend((option, str(value)))
    if values["force"]:
        raise ReplacementPlanError("replacement Config must never set force=True")
    if values["include_truth"]:
        argv.append("--include-truth")
    if values["enable_stdpopsim"]:
        argv.append("--enable-stdpopsim")
    argv.append(
        "--mut-map-scale-to-baseline"
        if values["mut_map_scale_to_baseline"]
        else "--no-mut-map-scale-to-baseline"
    )
    if not values["compression"]:
        argv.append("--no-compression")
    return argv


def _replacement_config(
    source: Mapping[str, Any],
    *,
    deficit: int,
    tail_seed: int,
    out_dir: Path,
    dataset_name: str,
    workers: int,
) -> Config:
    scientific = dict(source["scientific_config"])
    scientific["n_samples"] = deficit
    scientific["seed"] = tail_seed
    values = {
        **scientific,
        "dataset_name": dataset_name,
        "out_dir": str(out_dir),
        "workers": workers,
        "force": False,
        "compute_chunk_size": 1,
    }
    try:
        config = Config(**values)
        validate_config(config)
    except (TypeError, ValueError) as exc:
        raise ReplacementPlanError(f"replacement Config is invalid: {exc}") from exc
    return config


def _validate_replacement_generation_plan(
    out_dir: Path,
    replacement_config: Mapping[str, Any],
    source_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    plan, config = _validate_source_plan(out_dir)
    if dataclasses.asdict(config) != {
        **dict(replacement_config),
        "dataset_name": "source-audit",
        "out_dir": str(out_dir),
        "workers": 1,
        "force": False,
        "compute_chunk_size": 1,
    }:
        # _validate_source_plan reconstructs ignored runtime values; compare
        # only the generation state plus the separately checked layout.
        expected_state = generation_config_state(replacement_config)
        if plan.get(DONE_CONFIG_STATE_KEY) != expected_state:
            raise ReplacementPlanError(
                "replacement generation PLAN has a different Config fingerprint"
            )
    for key in (
        "empirical_maps_sha256",
        "empirical_ne_templates_sha256",
        "runtime_environment_sha256",
    ):
        if plan.get(key) != source_snapshot.get(key):
            raise ReplacementPlanError(
                f"replacement generation PLAN {key} differs from sources"
            )
    return plan


def _load_replacement_plan(out_dir: Path) -> dict[str, Any]:
    plan = _read_json(out_dir / PLAN_NAME, "replacement plan")
    persisted = dict(plan)
    recorded = persisted.pop("replacement_plan_sha256", None)
    if _require_sha256(
        recorded, "replacement_plan_sha256"
    ) != _canonical_sha256(persisted):
        raise ReplacementPlanError("replacement plan fingerprint is inconsistent")
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("kind") != "dlcoalsim_tail_replacement":
        raise ReplacementPlanError("unsupported replacement plan schema")
    if Path(str(plan.get("out_dir", ""))).resolve() != out_dir:
        raise ReplacementPlanError("replacement plan out_dir does not match its location")
    deficit = plan.get("deficit_samples")
    if isinstance(deficit, bool) or not isinstance(deficit, int) or deficit < 0:
        raise ReplacementPlanError("replacement plan deficit is malformed")
    if deficit == 0 and plan.get("replacement_config") is not None:
        raise ReplacementPlanError("zero-deficit plan unexpectedly contains a Config")
    if deficit > 0 and not isinstance(plan.get("replacement_config"), Mapping):
        raise ReplacementPlanError("positive-deficit plan has no Config")
    return plan


def prepare_replacement(
    source_roots: Sequence[str | os.PathLike[str]],
    *,
    target_total: int,
    out_dir: str | os.PathLike[str],
    generator_root: str | os.PathLike[str],
    generator_python: str | os.PathLike[str] = sys.executable,
    dataset_name: str = "DLCoalSim-Tail-Replacement",
    workers: int = -1,
) -> dict[str, Any]:
    """Freeze a replacement deficit and prepare it in the immutable runtime."""

    if isinstance(target_total, bool) or not isinstance(target_total, int) or target_total < 0:
        raise ReplacementPlanError("target_total must be a non-negative integer")
    output = Path(out_dir).expanduser().resolve()
    roots = [_source_root(path) for path in source_roots]
    if output in roots:
        raise ReplacementPlanError("replacement out_dir must not be a source root")
    existing_plan_path = output / PLAN_NAME
    if existing_plan_path.exists():
        existing = _load_replacement_plan(output)
        expected_roots = sorted(str(root) for root in roots)
        recorded_roots = sorted(str(item["root"]) for item in existing["sources"])
        immutable = existing["immutable_generator"]
        if (
            existing["target_total"] != target_total
            or expected_roots != recorded_roots
            or Path(immutable["root"]) != Path(generator_root).expanduser().resolve()
            or Path(immutable["python"])
            != _absolute_executable(generator_python)
        ):
            raise ReplacementPlanError(
                f"{existing_plan_path} already freezes different prepare arguments"
            )
        if existing["deficit_samples"]:
            expected_config = existing["replacement_config"]
            if (
                expected_config["dataset_name"] != dataset_name
                or expected_config["workers"] != workers
            ):
                raise ReplacementPlanError(
                    f"{existing_plan_path} already freezes different runtime fields"
                )
            _validate_replacement_generation_plan(
                output, expected_config, existing["source_provenance"]
            )
        return existing

    snapshots = _audit_sources([str(root) for root in roots])
    for left, right in zip(snapshots, snapshots[1:], strict=False):
        if int(left["planned_seed_stop_exclusive"]) != int(right["seed"]):
            raise ReplacementPlanError(
                "source planned seed ranges must be strictly contiguous"
            )
    planned = sum(int(snapshot["planned_n_samples"]) for snapshot in snapshots)
    if planned != target_total:
        raise ReplacementPlanError(
            f"source PLANs cover {planned} planned samples, expected target_total "
            f"{target_total}"
        )
    completed = sum(int(snapshot["completed_samples"]) for snapshot in snapshots)
    if completed > target_total:
        raise ReplacementPlanError(
            f"completed source samples ({completed}) exceed target_total ({target_total})"
        )
    deficit = target_total - completed
    tail_seed = max(
        int(snapshot["planned_seed_stop_exclusive"]) for snapshot in snapshots
    )

    immutable_root = Path(generator_root).expanduser().resolve()
    immutable_python = _absolute_executable(generator_python)
    probe = _runtime_probe(immutable_root, immutable_python)
    source_runtime = str(snapshots[0]["runtime_environment_sha256"])
    if probe["runtime_environment_sha256"] != source_runtime:
        raise ReplacementPlanError(
            "immutable generator runtime hash differs from source PLANs"
        )

    replacement_config: dict[str, Any] | None = None
    generation_plan: dict[str, Any] | None = None
    if deficit:
        config = _replacement_config(
            snapshots[0],
            deficit=deficit,
            tail_seed=tail_seed,
            out_dir=output,
            dataset_name=dataset_name,
            workers=workers,
        )
        replacement_config = dataclasses.asdict(config)
        _run_generator_subprocess(
            immutable_root,
            immutable_python,
            [*_config_argv(replacement_config), "--prepare-only"],
        )
        generation_plan = _validate_replacement_generation_plan(
            output, replacement_config, snapshots[0]
        )

    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "dlcoalsim_tail_replacement",
        "out_dir": str(output),
        "target_total": target_total,
        "completed_source_samples": completed,
        "deficit_samples": deficit,
        "tail_seed": tail_seed,
        "seed_stride": SEED_STRIDE,
        "source_snapshot_sha256": _canonical_sha256(snapshots),
        "sources": snapshots,
        "source_provenance": {
            key: snapshots[0][key]
            for key in (
                "empirical_maps_sha256",
                "empirical_ne_templates_sha256",
                "runtime_environment_sha256",
            )
        },
        "immutable_generator": {
            "root": str(immutable_root),
            "python": str(immutable_python),
            **probe,
        },
        "replacement_config": replacement_config,
        "generation_plan_sha256": (
            None if generation_plan is None else generation_plan["plan_sha256"]
        ),
    }
    plan["replacement_plan_sha256"] = _canonical_sha256(plan)
    _write_json_atomic(existing_plan_path, plan)
    return plan


def generate_replacement(
    out_dir: str | os.PathLike[str],
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Run one immutable-generator partition, or safely no-op for K=0."""

    output = Path(out_dir).expanduser().resolve()
    plan = _load_replacement_plan(output)
    if plan["deficit_samples"] == 0:
        return {"no_op": True, "deficit_samples": 0}
    if (rank is None) != (world_size is None):
        raise ReplacementPlanError("rank and world_size must be supplied together")
    if rank is None:
        try:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplacementPlanError(
                "generate requires --rank/--world-size or SLURM_PROCID/SLURM_NTASKS"
            ) from exc
    assert world_size is not None
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ReplacementPlanError("rank/world_size are out of range")

    config = plan["replacement_config"]
    immutable = plan["immutable_generator"]
    completed = _run_generator_subprocess(
        Path(immutable["root"]),
        Path(immutable["python"]),
        [
            *_config_argv(config),
            "--generate-only",
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
        ],
        capture_output=False,
    )
    return {
        "no_op": False,
        "rank": rank,
        "world_size": world_size,
        "returncode": completed.returncode,
    }


def finalize_replacement(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Finalize the frozen replacement dataset, or safely no-op for K=0."""

    output = Path(out_dir).expanduser().resolve()
    plan = _load_replacement_plan(output)
    done_path = output / DONE_NAME
    if plan["deficit_samples"] == 0:
        return {"no_op": True, "deficit_samples": 0}

    immutable = plan["immutable_generator"]
    _run_generator_subprocess(
        Path(immutable["root"]),
        Path(immutable["python"]),
        [*_config_argv(plan["replacement_config"]), "--finalize-only"],
    )
    generation_plan = _validate_replacement_generation_plan(
        output, plan["replacement_config"], plan["source_provenance"]
    )
    if generation_plan["plan_sha256"] != plan["generation_plan_sha256"]:
        raise ReplacementPlanError("generation PLAN changed after replacement prepare")
    samples_done = output / "samples" / "DONE.json"
    done = _read_json(samples_done, "replacement samples DONE")
    if done.get("n_samples") != plan["deficit_samples"]:
        raise ReplacementPlanError("replacement DONE has the wrong sample count")
    if done.get("generation_plan_sha256") != plan["generation_plan_sha256"]:
        raise ReplacementPlanError("replacement DONE references a different generation PLAN")
    completion = {
        "schema_version": SCHEMA_VERSION,
        "kind": "dlcoalsim_tail_replacement_done",
        "replacement_plan_sha256": plan["replacement_plan_sha256"],
        "deficit_samples": plan["deficit_samples"],
        "samples_done_path": "samples/DONE.json",
        "samples_done_file_sha256": _file_sha256(samples_done),
    }
    completion["replacement_done_sha256"] = _canonical_sha256(completion)
    if done_path.exists():
        existing = _read_json(done_path, "replacement completion")
        if existing != completion:
            raise ReplacementPlanError(f"{done_path} contains a different completion")
        return existing
    _write_json_atomic(done_path, completion)
    return completion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze and run exact tail replacements for stopped simulations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", action="append", required=True)
    prepare.add_argument("--target-total", type=int, required=True)
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--generator-root", required=True)
    prepare.add_argument("--generator-python", default=sys.executable)
    prepare.add_argument("--dataset-name", default="DLCoalSim-Tail-Replacement")
    prepare.add_argument("--workers", type=int, default=-1)

    for command in ("generate", "finalize"):
        child = subparsers.add_parser(command)
        child.add_argument("--out-dir", required=True)
        if command == "generate":
            child.add_argument("--rank", type=int)
            child.add_argument("--world-size", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_replacement(
                args.source,
                target_total=args.target_total,
                out_dir=args.out_dir,
                generator_root=args.generator_root,
                generator_python=args.generator_python,
                dataset_name=args.dataset_name,
                workers=args.workers,
            )
        elif args.command == "generate":
            result = generate_replacement(
                args.out_dir, rank=args.rank, world_size=args.world_size
            )
        else:
            result = finalize_replacement(args.out_dir)
    except ReplacementPlanError as exc:
        raise SystemExit(f"[error] {exc}") from None
    if args.command == "prepare":
        result = {
            "plan": str(Path(args.out_dir).expanduser().resolve() / PLAN_NAME),
            "target_total": result["target_total"],
            "completed_source_samples": result["completed_source_samples"],
            "deficit_samples": result["deficit_samples"],
            "tail_seed": result["tail_seed"],
            "no_op": result["deficit_samples"] == 0,
        }
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
