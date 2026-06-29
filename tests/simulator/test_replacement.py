from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from necast.simulator import replacement
from necast.simulator.config import Config
from necast.simulator.dataset_state import SAMPLE_GENERATOR_SCHEMA_VERSION, generation_config_state


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _file_record(path: Path) -> dict:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _source_plan(root: Path, *, n: int, seed: int, marker: str = "same") -> dict:
    cfg = Config(
        preset="full_v01",
        n_samples=n,
        seed=seed,
        shard_size=2,
        compute_chunk_size=1,
        out_dir=str(root),
        workers=4,
    )
    state = generation_config_state(cfg)
    layout = {
        "n_samples": n,
        "shard_size": 2,
        "compute_chunk_size": 1,
        "n_shards": (n + 1) // 2,
        "n_chunks": n,
        "sample_id_rule": "sim_{global_sample_index:08d}",
        "seed_rule": "config.seed + global_sample_index * 7919",
        "partition_rule": "chunk_idx_mod_world_size",
        "array_schema_version": SAMPLE_GENERATOR_SCHEMA_VERSION,
    }
    plan = {
        "schema_version": 1,
        "kind": "dlcoalsim_generation_plan",
        "sample_generation_config": state,
        "layout": layout,
        "layout_sha256": _canonical(layout),
        "empirical_maps_sha256": hashlib.sha256(f"map-{marker}".encode()).hexdigest(),
        "empirical_ne_templates_sha256": hashlib.sha256(
            f"template-{marker}".encode()
        ).hexdigest(),
        "runtime_environment_sha256": hashlib.sha256(
            f"runtime-{marker}".encode()
        ).hexdigest(),
    }
    plan["plan_sha256"] = _canonical(plan)
    _write_json(root / "samples" / "PLAN.json", plan)
    return plan


def _receipt(
    root: Path,
    plan: dict,
    *,
    kind: str,
    index: int,
    shard: int,
    start: int,
    count: int,
) -> None:
    if kind == "final":
        relative = f"samples/shard_{shard:05d}.npz"
        receipt_path = root / "samples" / ".receipts" / f"shard_{shard:05d}.done.json"
        result = {
            "shard_idx": shard,
            "start": start,
            "n_samples": count,
            "shard_path": relative,
        }
    else:
        relative = f"samples/.fragments/fragment_{index:08d}.npz"
        receipt_path = (
            root
            / "samples"
            / ".fragments"
            / f"fragment_{index:08d}.done.json"
        )
        result = {
            "chunk_idx": index,
            "shard_idx": shard,
            "start": start,
            "n_samples": count,
            "fragment_path": relative,
        }
    npz = root / relative
    metadata = npz.with_suffix(".jsonl.gz")
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(f"npz-{kind}-{index}".encode())
    metadata.write_bytes(f"meta-{kind}-{index}".encode())
    receipt = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "result": result,
        "files": {
            "npz": _file_record(npz),
            "metadata_jsonl_gz": _file_record(metadata),
        },
    }
    receipt["receipt_sha256"] = _canonical(receipt)
    _write_json(receipt_path, receipt)


def _sources(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_plan = _source_plan(left, n=4, seed=100)
    right_plan = _source_plan(
        right, n=3, seed=100 + 4 * replacement.SEED_STRIDE
    )
    return left, right, left_plan, right_plan


def test_scan_prefers_final_receipt_and_never_double_counts(tmp_path: Path) -> None:
    left, _, plan, _ = _sources(tmp_path)
    _receipt(left, plan, kind="final", index=0, shard=0, start=0, count=2)
    _receipt(left, plan, kind="fragment", index=0, shard=0, start=0, count=1)
    _receipt(left, plan, kind="fragment", index=1, shard=0, start=1, count=1)
    _receipt(left, plan, kind="fragment", index=2, shard=1, start=2, count=1)

    snapshot = replacement._scan_source(left)

    assert snapshot["completed_samples"] == 3
    assert snapshot["selected_final_receipts"] == 1
    assert snapshot["selected_fragment_receipts"] == 1
    assert snapshot["ignored_fragments_covered_by_final"] == 2
    assert [(row["kind"], row["start"]) for row in snapshot["selected_receipts"]] == [
        ("final", 0),
        ("fragment", 2),
    ]


def test_audit_rejects_config_hash_and_seed_overlap(tmp_path: Path) -> None:
    left, right, left_plan, right_plan = _sources(tmp_path)
    _receipt(left, left_plan, kind="fragment", index=0, shard=0, start=0, count=1)
    _receipt(right, right_plan, kind="fragment", index=0, shard=0, start=0, count=1)
    assert len(replacement._audit_sources([left, right])) == 2

    changed = json.loads((right / "samples" / "PLAN.json").read_text())
    changed["empirical_maps_sha256"] = "0" * 64
    changed.pop("plan_sha256")
    changed["plan_sha256"] = _canonical(changed)
    _write_json(right / "samples" / "PLAN.json", changed)
    _receipt(
        right,
        changed,
        kind="fragment",
        index=0,
        shard=0,
        start=0,
        count=1,
    )
    with pytest.raises(replacement.ReplacementPlanError, match="empirical_maps"):
        replacement._audit_sources([left, right])

    overlapping = tmp_path / "overlap"
    overlap_plan = _source_plan(
        overlapping, n=2, seed=100 + replacement.SEED_STRIDE
    )
    _receipt(
        overlapping,
        overlap_plan,
        kind="fragment",
        index=0,
        shard=0,
        start=0,
        count=1,
    )
    with pytest.raises(replacement.ReplacementPlanError, match="seed sequences overlap"):
        replacement._audit_sources([left, overlapping])


def test_corrupt_receipt_or_missing_artifact_fails_closed(tmp_path: Path) -> None:
    left, _, plan, _ = _sources(tmp_path)
    _receipt(left, plan, kind="fragment", index=0, shard=0, start=0, count=1)
    receipt_path = (
        left / "samples" / ".fragments" / "fragment_00000000.done.json"
    )
    receipt = json.loads(receipt_path.read_text())
    receipt["result"]["start"] = 1
    _write_json(receipt_path, receipt)
    with pytest.raises(replacement.ReplacementPlanError, match="fingerprint"):
        replacement._scan_source(left)

    _receipt(left, plan, kind="fragment", index=0, shard=0, start=0, count=1)
    (left / "samples" / ".fragments" / "fragment_00000000.npz").unlink()
    with pytest.raises(replacement.ReplacementPlanError, match="missing"):
        replacement._scan_source(left)


def test_prepare_freezes_deficit_tail_and_immutable_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left, right, left_plan, right_plan = _sources(tmp_path)
    for index in range(3):
        _receipt(
            left,
            left_plan,
            kind="fragment",
            index=index,
            shard=index // 2,
            start=index,
            count=1,
        )
    for index in range(2):
        _receipt(
            right,
            right_plan,
            kind="fragment",
            index=index,
            shard=index // 2,
            start=index,
            count=1,
        )
    runtime = left_plan["runtime_environment_sha256"]
    monkeypatch.setattr(
        replacement,
        "_runtime_probe",
        lambda root, python: {
            "simulator_file": str(root / "necast" / "simulator" / "__init__.py"),
            "runtime_environment_sha256": runtime,
        },
    )

    prepared_config: dict | None = None

    def fake_run(root, python, arguments, *, module=True):
        nonlocal prepared_config
        assert module
        assert arguments[-1] == "--prepare-only"
        values = {}
        iterator = iter(arguments[:-1])
        for token in iterator:
            if token.startswith("--") and token not in {
                "--include-truth",
                "--enable-stdpopsim",
                "--mut-map-scale-to-baseline",
                "--no-mut-map-scale-to-baseline",
                "--no-compression",
            }:
                values[token] = next(iterator)
        prepared_config = values
        return type("Completed", (), {"stdout": "", "stderr": ""})()

    monkeypatch.setattr(replacement, "_run_generator_subprocess", fake_run)

    expected_generation_plan = {
        "plan_sha256": "9" * 64,
        "empirical_maps_sha256": left_plan["empirical_maps_sha256"],
        "empirical_ne_templates_sha256": left_plan[
            "empirical_ne_templates_sha256"
        ],
        "runtime_environment_sha256": runtime,
    }
    monkeypatch.setattr(
        replacement,
        "_validate_replacement_generation_plan",
        lambda out, config, source: expected_generation_plan,
    )
    generator_root = tmp_path / "deploy"
    generator_root.mkdir()
    generator_python = tmp_path / "python"
    generator_python.write_text("")
    output = tmp_path / "replacement"

    plan = replacement.prepare_replacement(
        [left, right],
        target_total=7,
        out_dir=output,
        generator_root=generator_root,
        generator_python=generator_python,
        workers=8,
    )

    assert plan["completed_source_samples"] == 5
    assert plan["deficit_samples"] == 2
    assert plan["tail_seed"] == 100 + 7 * replacement.SEED_STRIDE
    assert plan["replacement_config"]["n_samples"] == 2
    assert plan["replacement_config"]["seed"] == plan["tail_seed"]
    assert plan["replacement_config"]["compute_chunk_size"] == 1
    assert plan["replacement_config"]["workers"] == 8
    assert prepared_config["--n-samples"] == "2"
    assert prepared_config["--seed"] == str(plan["tail_seed"])
    assert (output / replacement.PLAN_NAME).is_file()


def test_zero_deficit_is_a_safe_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left"
    source_plan = _source_plan(left, n=2, seed=10)
    _receipt(left, source_plan, kind="final", index=0, shard=0, start=0, count=2)
    runtime = source_plan["runtime_environment_sha256"]
    monkeypatch.setattr(
        replacement,
        "_runtime_probe",
        lambda root, python: {
            "simulator_file": str(root / "necast" / "simulator" / "__init__.py"),
            "runtime_environment_sha256": runtime,
        },
    )
    monkeypatch.setattr(
        replacement,
        "_run_generator_subprocess",
        lambda *args, **kwargs: pytest.fail("zero deficit invoked generator"),
    )
    generator_root = tmp_path / "deploy"
    generator_root.mkdir()
    generator_python = tmp_path / "python"
    generator_python.write_text("")
    output = tmp_path / "replacement"

    plan = replacement.prepare_replacement(
        [left],
        target_total=2,
        out_dir=output,
        generator_root=generator_root,
        generator_python=generator_python,
    )
    assert plan["deficit_samples"] == 0
    assert plan["replacement_config"] is None
    assert replacement.generate_replacement(output) == {
        "no_op": True,
        "deficit_samples": 0,
    }
    assert replacement.finalize_replacement(output) == {
        "no_op": True,
        "deficit_samples": 0,
    }
    assert not (output / "samples").exists()
    assert not (output / replacement.DONE_NAME).exists()


def test_prepare_rejects_wrong_generator_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left"
    source_plan = _source_plan(left, n=2, seed=10)
    _receipt(left, source_plan, kind="fragment", index=0, shard=0, start=0, count=1)
    monkeypatch.setattr(
        replacement,
        "_runtime_probe",
        lambda root, python: {
            "simulator_file": str(root / "necast" / "simulator" / "__init__.py"),
            "runtime_environment_sha256": "0" * 64,
        },
    )
    root = tmp_path / "deploy"
    root.mkdir()
    python = tmp_path / "python"
    python.write_text("")
    with pytest.raises(replacement.ReplacementPlanError, match="runtime hash differs"):
        replacement.prepare_replacement(
            [left],
            target_total=2,
            out_dir=tmp_path / "replacement",
            generator_root=root,
            generator_python=python,
        )


def test_prepare_rejects_noncontiguous_or_wrong_planned_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left_plan = _source_plan(left, n=2, seed=10)
    right_plan = _source_plan(
        right, n=2, seed=10 + 3 * replacement.SEED_STRIDE
    )
    _receipt(left, left_plan, kind="fragment", index=0, shard=0, start=0, count=1)
    _receipt(right, right_plan, kind="fragment", index=0, shard=0, start=0, count=1)
    monkeypatch.setattr(
        replacement,
        "_runtime_probe",
        lambda root, python: pytest.fail("coverage audit must precede runtime probe"),
    )
    root = tmp_path / "deploy"
    root.mkdir()
    python = tmp_path / "python"
    python.write_text("")
    with pytest.raises(replacement.ReplacementPlanError, match="strictly contiguous"):
        replacement.prepare_replacement(
            [left, right],
            target_total=4,
            out_dir=tmp_path / "replacement",
            generator_root=root,
            generator_python=python,
        )

    with pytest.raises(replacement.ReplacementPlanError, match="cover 2 planned"):
        replacement.prepare_replacement(
            [left],
            target_total=3,
            out_dir=tmp_path / "other",
            generator_root=root,
            generator_python=python,
        )


def test_config_argv_roundtrips_boolean_intent() -> None:
    cfg = dataclasses.asdict(Config())
    cfg["force"] = False
    cfg["include_truth"] = True
    cfg["enable_stdpopsim"] = False
    cfg["mut_map_scale_to_baseline"] = False
    cfg["compression"] = False
    argv = replacement._config_argv(cfg)
    assert "--force" not in argv
    assert "--include-truth" in argv
    assert "--enable-stdpopsim" not in argv
    assert "--no-mut-map-scale-to-baseline" in argv
    assert "--no-compression" in argv


def test_generate_streams_subprocess_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "replacement"
    cfg = dataclasses.asdict(Config(n_samples=1, out_dir=str(output)))
    plan = {
        "schema_version": replacement.SCHEMA_VERSION,
        "kind": "dlcoalsim_tail_replacement",
        "out_dir": str(output.resolve()),
        "target_total": 1,
        "completed_source_samples": 0,
        "deficit_samples": 1,
        "tail_seed": 99,
        "seed_stride": replacement.SEED_STRIDE,
        "source_snapshot_sha256": "0" * 64,
        "sources": [],
        "source_provenance": {},
        "immutable_generator": {
            "root": str(tmp_path),
            "python": str(tmp_path / "python"),
        },
        "replacement_config": cfg,
        "generation_plan_sha256": "1" * 64,
    }
    plan["replacement_plan_sha256"] = _canonical(plan)
    _write_json(output / replacement.PLAN_NAME, plan)
    calls = []

    def fake_run(root, python, arguments, *, module=True, capture_output=True):
        calls.append((module, capture_output))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(replacement, "_run_generator_subprocess", fake_run)
    result = replacement.generate_replacement(output, rank=0, world_size=1)
    assert result["returncode"] == 0
    assert calls == [(True, False)]
