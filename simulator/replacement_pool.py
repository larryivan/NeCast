"""Fail-closed completion-ordered replacement candidate pools.

This controller is deliberately distinct from :mod:`simulator.replacement`.
The v1 controller prepares exactly K replacement simulations and publishes an
ordinary completed generator root.  This v2 controller instead prepares a
fixed pool of C >= K one-sample fragments, lets independent array ranks finish
out of order, and freezes the first K fully validated registrations.

Selection is based only on generator-plan identity, receipt integrity, file
hashes, and the expected sample identity/index/seed.  Target values, variant
counts, demographic labels, and other scientific payload fields are never
used for eligibility or ordering.  Completion ordering is nevertheless a
runtime-censoring mechanism and is declared as such in ``SELECTION.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import gzip
import hashlib
import json
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import replacement as v1
from .dataset_state import generation_config_state


SCHEMA_VERSION = 2
POOL_PLAN_NAME = "POOL_PLAN.json"
REGISTRY_NAME = ".REGISTRATIONS.json"
SELECTION_NAME = "SELECTION.json"
LOCK_NAME = ".registration.lock"
SEALED_DIR_NAME = "sealed"

POOL_PLAN_SCHEMA = "dlcoalsim-replacement-candidate-pool-plan-v2"
POOL_PLAN_KIND = "dlcoalsim_replacement_candidate_pool_plan"
REGISTRY_SCHEMA = "dlcoalsim-replacement-registration-registry-v2"
REGISTRY_KIND = "dlcoalsim_replacement_registration_registry"
SELECTION_SCHEMA = "dlcoalsim-replacement-completion-selection-v2"
SELECTION_KIND = "dlcoalsim_replacement_completion_selection"

SEED_STRIDE = v1.SEED_STRIDE
_PROCESS_REGISTRATION_LOCK = threading.RLock()


class ReplacementPoolError(ValueError):
    """Raised when a candidate pool cannot be proven safe."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplacementPoolError(f"{label} at {path} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplacementPoolError(f"{label} at {path} is not a JSON object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _require_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplacementPoolError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(relative_to.resolve()).as_posix()
    except ValueError as exc:
        raise ReplacementPoolError(f"artifact escapes pool root: {resolved}") from exc
    if not resolved.is_file():
        raise ReplacementPoolError(f"artifact is missing: {resolved}")
    return {
        "path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def _candidate_groups(required_k: int, candidate_count: int) -> dict[str, Any]:
    return {
        "primary": {
            "count": required_k,
            "candidate_indices": list(range(required_k)),
        },
        "standby": {
            "count": candidate_count - required_k,
            "candidate_indices": list(range(required_k, candidate_count)),
        },
    }


def _candidate_records(
    *,
    required_k: int,
    candidate_count: int,
    tail_seed: int,
    shard_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(candidate_count):
        base = f"samples/.fragments/fragment_{index:08d}"
        records.append(
            {
                "candidate_index": index,
                "candidate_group": "primary" if index < required_k else "standby",
                "sample_index": index,
                "chunk_index": index,
                "shard_index": index // shard_size,
                "seed": tail_seed + index * SEED_STRIDE,
                "expected_sample_id": f"sim_{index:08d}",
                "receipt_path": f"{base}.done.json",
                "npz_path": f"{base}.npz",
                "metadata_path": f"{base}.jsonl.gz",
            }
        )
    return records


def _selection_contract(required_k: int) -> dict[str, Any]:
    return {
        "required_selected_candidates": required_k,
        "eligibility": (
            "candidate generator fragment receipt, NPZ, and metadata must pass "
            "plan/index/seed/size/SHA256 validation"
        ),
        "ordering": (
            "first K eligible candidates to acquire the serialized cross-node "
            "registration lock receive registration ordinals 1..K"
        ),
        "late_registration_policy": (
            "after SELECTION.json is frozen, late and duplicate registrations "
            "cannot change the selected prefix"
        ),
        "scientific_payload_fields_used_for_selection": [],
        "target_variant_demography_fields_read_for_selection": False,
        "completion_order_runtime_censoring": required_k > 0,
        "selected_candidate_seeds_may_be_noncontiguous": True,
    }


def _candidate_seed_interval(
    tail_seed: int, candidate_count: int
) -> dict[str, int]:
    return {
        "seed_start": tail_seed,
        "seed_stop_exclusive": tail_seed + candidate_count * SEED_STRIDE,
        "seed_stride": SEED_STRIDE,
        "n_candidates": candidate_count,
    }


def _validate_pool_plan_shape(plan: Mapping[str, Any], output: Path) -> None:
    if (
        plan.get("schema") != POOL_PLAN_SCHEMA
        or plan.get("schema_version") != SCHEMA_VERSION
        or plan.get("kind") != POOL_PLAN_KIND
        or plan.get("status") != "prepared"
    ):
        raise ReplacementPoolError("unsupported replacement pool plan")
    if Path(str(plan.get("out_dir", ""))).resolve() != output:
        raise ReplacementPoolError("pool plan out_dir does not match its location")

    target_total = _require_int(plan.get("target_total"), "target_total")
    completed = _require_int(
        plan.get("completed_source_samples"), "completed_source_samples"
    )
    required_k = _require_int(plan.get("required_k"), "required_k")
    candidate_count = _require_int(plan.get("candidate_count"), "candidate_count")
    if completed + required_k != target_total:
        raise ReplacementPoolError("pool plan target/deficit accounting is invalid")
    if candidate_count < required_k:
        raise ReplacementPoolError("candidate_count is smaller than required_k")
    if required_k == 0 and candidate_count != 0:
        raise ReplacementPoolError("K=0 pool must not contain candidates")

    tail_seed = _require_int(plan.get("tail_seed"), "tail_seed")
    if plan.get("seed_stride") != SEED_STRIDE:
        raise ReplacementPoolError("pool plan has an unsupported seed stride")
    if plan.get("candidate_groups") != _candidate_groups(
        required_k, candidate_count
    ):
        raise ReplacementPoolError("pool plan primary/standby groups are inconsistent")
    if plan.get("candidate_seed_interval") != _candidate_seed_interval(
        tail_seed, candidate_count
    ):
        raise ReplacementPoolError("pool plan candidate seed interval is inconsistent")
    if plan.get("selection_contract") != _selection_contract(required_k):
        raise ReplacementPoolError("pool plan selection contract is inconsistent")

    sources = plan.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ReplacementPoolError("pool plan has no source snapshots")
    if plan.get("source_snapshot_sha256") != _canonical_sha256(sources):
        raise ReplacementPoolError("pool source snapshot fingerprint is inconsistent")

    candidate_config = plan.get("candidate_config")
    candidate_state = plan.get("candidate_config_state")
    if candidate_count == 0:
        if any(
            value is not None
            for value in (
                candidate_config,
                candidate_state,
                plan.get("generation_plan_file_sha256"),
                plan.get("generation_plan_sha256"),
            )
        ):
            raise ReplacementPoolError("K=0 pool unexpectedly has a generator config")
        if plan.get("candidates") != []:
            raise ReplacementPoolError("K=0 pool unexpectedly has candidate records")
        return

    if not isinstance(candidate_config, Mapping):
        raise ReplacementPoolError("candidate generator config is missing")
    if (
        candidate_config.get("n_samples") != candidate_count
        or candidate_config.get("compute_chunk_size") != 1
        or candidate_config.get("workers") != 1
        or candidate_config.get("seed") != tail_seed
    ):
        raise ReplacementPoolError("candidate generator config violates the pool contract")
    if candidate_state != generation_config_state(candidate_config):
        raise ReplacementPoolError("candidate generator config fingerprint is inconsistent")
    shard_size = _require_int(
        candidate_config.get("shard_size"), "candidate shard_size", minimum=1
    )
    if plan.get("candidates") != _candidate_records(
        required_k=required_k,
        candidate_count=candidate_count,
        tail_seed=tail_seed,
        shard_size=shard_size,
    ):
        raise ReplacementPoolError("pool candidate records are inconsistent")
    for name in ("generation_plan_file_sha256", "generation_plan_sha256"):
        value = plan.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ReplacementPoolError(f"pool plan {name} is malformed")


def _load_pool_plan(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    output = Path(out_dir).expanduser().resolve()
    path = output / POOL_PLAN_NAME
    plan = _read_json(path, "replacement pool plan")
    unsigned = dict(plan)
    recorded = unsigned.pop("pool_plan_sha256", None)
    if not isinstance(recorded, str) or recorded != _canonical_sha256(unsigned):
        raise ReplacementPoolError("pool plan fingerprint is inconsistent")
    _validate_pool_plan_shape(plan, output)
    return plan


def _validate_generation_plan(
    output: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    if int(plan["candidate_count"]) == 0:
        raise ReplacementPoolError("K=0 pool has no generation plan")
    try:
        generation_plan = v1._validate_replacement_generation_plan(
            output,
            plan["candidate_config"],
            plan["source_provenance"],
        )
    except (v1.ReplacementPlanError, OSError, ValueError) as exc:
        raise ReplacementPoolError(
            f"candidate generation plan is invalid: {exc}"
        ) from exc
    path = output / str(plan["generation_plan_path"])
    if (
        generation_plan.get("plan_sha256") != plan.get("generation_plan_sha256")
        or _file_sha256(path) != plan.get("generation_plan_file_sha256")
    ):
        raise ReplacementPoolError("candidate generation PLAN changed after prepare")
    return generation_plan


def _empty_registry(plan: Mapping[str, Any]) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "schema": REGISTRY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": REGISTRY_KIND,
        "status": "open" if int(plan["required_k"]) else "selection_frozen",
        "pool_plan_sha256": plan["pool_plan_sha256"],
        "required_k": plan["required_k"],
        "candidate_count": plan["candidate_count"],
        "registration_count": 0,
        "registrations": [],
    }
    registry["registry_sha256"] = _canonical_sha256(registry)
    return registry


def _validate_registration(
    registration: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    expected_ordinal: int,
) -> None:
    index = _require_int(
        registration.get("candidate_index"), "registered candidate_index"
    )
    if index >= int(plan["candidate_count"]):
        raise ReplacementPoolError("registered candidate index is out of range")
    expected = plan["candidates"][index]
    expected_fields = {
        "registration_ordinal": expected_ordinal,
        "candidate_index": index,
        "candidate_group": expected["candidate_group"],
        "sample_index": expected["sample_index"],
        "sample_id": expected["expected_sample_id"],
        "seed": expected["seed"],
        "pool_plan_sha256": plan["pool_plan_sha256"],
        "generation_plan_sha256": plan["generation_plan_sha256"],
    }
    for key, value in expected_fields.items():
        if registration.get(key) != value:
            raise ReplacementPoolError(
                f"registration {expected_ordinal} has inconsistent {key}"
            )
    if not isinstance(registration.get("source_artifacts"), Mapping):
        raise ReplacementPoolError("registration source artifacts are malformed")
    receipt_sha256 = registration.get("receipt_sha256")
    if not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64:
        raise ReplacementPoolError("registration receipt fingerprint is malformed")
    identity = {
        "sample_id": expected["expected_sample_id"],
        "sample_index": index,
        "seed": expected["seed"],
    }
    if registration.get("metadata_identity_sha256") != _canonical_sha256(identity):
        raise ReplacementPoolError("registration metadata identity hash is invalid")


def _load_registry(
    output: Path, plan: Mapping[str, Any]
) -> dict[str, Any]:
    path = output / REGISTRY_NAME
    if not path.exists():
        return _empty_registry(plan)
    registry = _read_json(path, "replacement registration registry")
    unsigned = dict(registry)
    recorded = unsigned.pop("registry_sha256", None)
    if not isinstance(recorded, str) or recorded != _canonical_sha256(unsigned):
        raise ReplacementPoolError("registration registry fingerprint is inconsistent")
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("schema_version") != SCHEMA_VERSION
        or registry.get("kind") != REGISTRY_KIND
        or registry.get("pool_plan_sha256") != plan.get("pool_plan_sha256")
        or registry.get("required_k") != plan.get("required_k")
        or registry.get("candidate_count") != plan.get("candidate_count")
    ):
        raise ReplacementPoolError("registration registry contract is inconsistent")
    registrations = registry.get("registrations")
    if not isinstance(registrations, list):
        raise ReplacementPoolError("registration registry payload is malformed")
    if (
        registry.get("registration_count") != len(registrations)
        or len(registrations) > int(plan["required_k"])
    ):
        raise ReplacementPoolError("registration registry count is invalid")
    indices: set[int] = set()
    for ordinal, registration in enumerate(registrations, start=1):
        if not isinstance(registration, Mapping):
            raise ReplacementPoolError("registration entry is not an object")
        _validate_registration(
            registration, plan=plan, expected_ordinal=ordinal
        )
        index = int(registration["candidate_index"])
        if index in indices:
            raise ReplacementPoolError("candidate registered more than once")
        indices.add(index)
    expected_status = (
        "selection_frozen"
        if len(registrations) == int(plan["required_k"])
        else "open"
    )
    if registry.get("status") != expected_status:
        raise ReplacementPoolError("registration registry status is inconsistent")
    return registry


def _write_registry(path: Path, registry: dict[str, Any]) -> None:
    unsigned = dict(registry)
    unsigned.pop("registry_sha256", None)
    registry = unsigned
    registry["registry_sha256"] = _canonical_sha256(registry)
    _write_json_atomic(path, registry)


@contextmanager
def _registration_lock(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with _PROCESS_REGISTRATION_LOCK:
        with (output / LOCK_NAME).open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _metadata_identity(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplacementPoolError(f"candidate metadata is unreadable: {path}") from exc
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ReplacementPoolError("candidate metadata must contain exactly one row")
    row = rows[0]
    identity = {
        "sample_id": row.get("sample_id"),
        "sample_index": row.get("sample_index"),
        "seed": row.get("seed"),
    }
    wanted = {
        "sample_id": expected["expected_sample_id"],
        "sample_index": expected["sample_index"],
        "seed": expected["seed"],
    }
    if identity != wanted:
        raise ReplacementPoolError("candidate metadata identity/index/seed is invalid")
    return wanted


def _validate_candidate_artifacts(
    output: Path,
    plan: Mapping[str, Any],
    candidate_index: int,
) -> dict[str, Any]:
    if candidate_index < 0 or candidate_index >= int(plan["candidate_count"]):
        raise ReplacementPoolError("candidate_index is out of range")
    _validate_generation_plan(output, plan)
    expected = plan["candidates"][candidate_index]
    receipt_path = output / str(expected["receipt_path"])
    try:
        receipt = v1._receipt_record(
            root=output,
            receipt_path=receipt_path,
            plan_sha256=str(plan["generation_plan_sha256"]),
            kind="fragment",
            expected_index=candidate_index,
            expected_shard=int(expected["shard_index"]),
            expected_start=candidate_index,
            expected_count=1,
        )
    except (v1.ReplacementPlanError, OSError, ValueError) as exc:
        raise ReplacementPoolError(
            f"candidate {candidate_index} receipt is invalid: {exc}"
        ) from exc

    source_artifacts = {
        "receipt": _artifact_record(receipt_path, relative_to=output),
        "npz": dict(receipt["files"]["npz"]),
        "metadata_jsonl_gz": dict(receipt["files"]["metadata_jsonl_gz"]),
    }
    for name in ("npz", "metadata_jsonl_gz"):
        artifact = source_artifacts[name]
        path = output / str(artifact["path"])
        if _file_sha256(path) != artifact["sha256"]:
            raise ReplacementPoolError(
                f"candidate {candidate_index} {name} SHA256 mismatch"
            )
    metadata_path = output / str(source_artifacts["metadata_jsonl_gz"]["path"])
    identity = _metadata_identity(metadata_path, expected)
    return {
        "candidate_index": candidate_index,
        "candidate_group": expected["candidate_group"],
        "sample_index": candidate_index,
        "sample_id": expected["expected_sample_id"],
        "seed": expected["seed"],
        "pool_plan_sha256": plan["pool_plan_sha256"],
        "generation_plan_sha256": plan["generation_plan_sha256"],
        "receipt_sha256": str(receipt["receipt_sha256"]),
        "source_artifacts": source_artifacts,
        "metadata_identity_sha256": _canonical_sha256(identity),
    }


def _link_verified(
    source: Path,
    destination: Path,
    expected: Mapping[str, Any],
    *,
    output: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            destination.stat().st_size != int(expected["size_bytes"])
            or _file_sha256(destination) != expected["sha256"]
        ):
            raise ReplacementPoolError(
                f"existing sealed artifact differs: {destination}"
            )
    else:
        try:
            os.link(source, destination)
        except OSError as exc:
            raise ReplacementPoolError(
                f"cannot hardlink selected artifact on the same filesystem: {exc}"
            ) from exc
    if (
        destination.stat().st_size != int(expected["size_bytes"])
        or _file_sha256(destination) != expected["sha256"]
    ):
        raise ReplacementPoolError(f"sealed artifact verification failed: {destination}")
    return {
        "path": destination.resolve().relative_to(output).as_posix(),
        "size_bytes": int(expected["size_bytes"]),
        "sha256": str(expected["sha256"]),
    }


def _seal_registration(
    output: Path,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    index = int(registration["candidate_index"])
    destination_root = output / SEALED_DIR_NAME / f"candidate_{index:08d}"
    sealed: dict[str, Any] = {}
    for name, suffix in (
        ("receipt", ".done.json"),
        ("npz", ".npz"),
        ("metadata_jsonl_gz", ".jsonl.gz"),
    ):
        source_record = registration["source_artifacts"][name]
        source = output / str(source_record["path"])
        destination = destination_root / f"fragment_{index:08d}{suffix}"
        sealed[name] = _link_verified(
            source, destination, source_record, output=output
        )
    return sealed


def _selection_rule(required_k: int) -> dict[str, Any]:
    return {
        "selected_prefix_length": required_k,
        "eligibility": (
            "valid generator PLAN and matching one-sample fragment receipt, NPZ "
            "SHA256, metadata SHA256, sample identity, sample index, and seed"
        ),
        "ordering_authority": "serialized fcntl.flock registration commit order",
        "ordering_key": "registration_ordinal",
        "tie_policy": "the cross-node exclusive lock creates a total order",
        "late_and_duplicate_policy": "cannot mutate the frozen selected prefix",
        "scientific_payload_fields_used": [],
        "target_variant_demography_fields_read": False,
    }


def _runtime_censoring(required_k: int, candidate_count: int) -> dict[str, Any]:
    return {
        "declared": True,
        "applied": required_k > 0,
        "mechanism": (
            "the first K integrity-valid candidates in serialized registration "
            "completion order are selected from a predeclared C-candidate pool"
            if required_k
            else "not applied because the frozen source deficit K is zero"
        ),
        "required_k": required_k,
        "candidate_count": candidate_count,
        "standby_candidates": candidate_count - required_k,
        "selection_may_depend_on_runtime": required_k > 0,
        "ordinary_uncensored_iid_completion": False if required_k else True,
        "scientific_payload_fields_used_for_selection": [],
        "statistical_consequence": (
            "simulation and validation completion time may depend on sample "
            "properties; the selected replacement subset is runtime-censored"
            if required_k
            else "none from replacement completion ordering"
        ),
    }


def _freeze_selection(
    output: Path,
    plan: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any] | None:
    required_k = int(plan["required_k"])
    registrations = registry["registrations"]
    if len(registrations) < required_k:
        return None
    if len(registrations) != required_k:
        raise ReplacementPoolError("registry exceeds the frozen selected prefix")

    selected: list[dict[str, Any]] = []
    for registration in registrations:
        item = dict(registration)
        item["sealed_artifacts"] = _seal_registration(output, registration)
        selected.append(item)

    sealed_root = output / SEALED_DIR_NAME
    pool_plan_record = _artifact_record(
        output / POOL_PLAN_NAME, relative_to=output
    )
    sealed_pool_plan = _link_verified(
        output / POOL_PLAN_NAME,
        sealed_root / POOL_PLAN_NAME,
        pool_plan_record,
        output=output,
    )
    generation_record: dict[str, Any] | None = None
    if int(plan["candidate_count"]):
        source_generation_plan = output / str(plan["generation_plan_path"])
        source_record = _artifact_record(source_generation_plan, relative_to=output)
        generation_record = _link_verified(
            source_generation_plan,
            sealed_root / "GENERATION_PLAN.json",
            source_record,
            output=output,
        )

    selected_indices = [int(item["candidate_index"]) for item in selected]
    selection: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": SELECTION_KIND,
        "status": "complete",
        "pool_plan_path": POOL_PLAN_NAME,
        "pool_plan_sha256": plan["pool_plan_sha256"],
        "pool_plan_file_sha256": pool_plan_record["sha256"],
        "generation_plan_sha256": plan["generation_plan_sha256"],
        "required_k": required_k,
        "candidate_count": plan["candidate_count"],
        "registration_cutoff_ordinal": required_k,
        "selection_rule": _selection_rule(required_k),
        "runtime_censoring": _runtime_censoring(
            required_k, int(plan["candidate_count"])
        ),
        "candidate_groups": plan["candidate_groups"],
        "selected_candidate_indices": selected_indices,
        "selected_seeds": [int(item["seed"]) for item in selected],
        "unselected_candidate_indices_at_freeze": sorted(
            set(range(int(plan["candidate_count"]))) - set(selected_indices)
        ),
        "selected_candidates": selected,
        "selected_candidates_sha256": _canonical_sha256(selected),
        "sealed_view": {
            "root": SEALED_DIR_NAME,
            "pool_plan": sealed_pool_plan,
            "generation_plan": generation_record,
            "selected_artifact_count": 3 * len(selected),
        },
    }
    selection["selection_sha256"] = _canonical_sha256(selection)
    path = output / SELECTION_NAME
    if path.exists():
        existing = _load_selection(output, plan, registry=registry)
        if existing != selection:
            raise ReplacementPoolError("existing selection differs from frozen prefix")
        return existing
    _write_json_atomic(path, selection)
    return _load_selection(output, plan, registry=registry)


def _validate_sealed_artifact(
    output: Path,
    source: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> None:
    if (
        sealed.get("size_bytes") != source.get("size_bytes")
        or sealed.get("sha256") != source.get("sha256")
    ):
        raise ReplacementPoolError("sealed and source artifact records differ")
    path = output / str(sealed.get("path", ""))
    if (
        not path.is_file()
        or path.stat().st_size != int(sealed["size_bytes"])
        or _file_sha256(path) != sealed["sha256"]
    ):
        raise ReplacementPoolError(f"sealed artifact is missing or corrupt: {path}")


def _load_selection(
    output: Path,
    plan: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = output / SELECTION_NAME
    selection = _read_json(path, "replacement pool selection")
    unsigned = dict(selection)
    recorded = unsigned.pop("selection_sha256", None)
    if not isinstance(recorded, str) or recorded != _canonical_sha256(unsigned):
        raise ReplacementPoolError("selection fingerprint is inconsistent")
    required_k = int(plan["required_k"])
    candidate_count = int(plan["candidate_count"])
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("schema_version") != SCHEMA_VERSION
        or selection.get("kind") != SELECTION_KIND
        or selection.get("status") != "complete"
        or selection.get("pool_plan_sha256") != plan.get("pool_plan_sha256")
        or selection.get("generation_plan_sha256")
        != plan.get("generation_plan_sha256")
        or selection.get("required_k") != required_k
        or selection.get("candidate_count") != candidate_count
        or selection.get("registration_cutoff_ordinal") != required_k
        or selection.get("selection_rule") != _selection_rule(required_k)
        or selection.get("runtime_censoring")
        != _runtime_censoring(required_k, candidate_count)
        or selection.get("candidate_groups") != plan.get("candidate_groups")
    ):
        raise ReplacementPoolError("selection contract is inconsistent")
    if selection.get("pool_plan_file_sha256") != _file_sha256(
        output / POOL_PLAN_NAME
    ):
        raise ReplacementPoolError("selection pool plan file hash changed")
    selected = selection.get("selected_candidates")
    if not isinstance(selected, list) or len(selected) != required_k:
        raise ReplacementPoolError("selection has the wrong selected count")
    if selection.get("selected_candidates_sha256") != _canonical_sha256(selected):
        raise ReplacementPoolError("selected candidate fingerprint is inconsistent")
    indices: list[int] = []
    seeds: list[int] = []
    for ordinal, item in enumerate(selected, start=1):
        if not isinstance(item, Mapping):
            raise ReplacementPoolError("selected candidate is not an object")
        _validate_registration(item, plan=plan, expected_ordinal=ordinal)
        sealed = item.get("sealed_artifacts")
        if not isinstance(sealed, Mapping):
            raise ReplacementPoolError("selected sealed artifacts are malformed")
        for name in ("receipt", "npz", "metadata_jsonl_gz"):
            if not isinstance(sealed.get(name), Mapping):
                raise ReplacementPoolError("selected sealed artifact is malformed")
            _validate_sealed_artifact(
                output, item["source_artifacts"][name], sealed[name]
            )
        indices.append(int(item["candidate_index"]))
        seeds.append(int(item["seed"]))
    if len(set(indices)) != len(indices) or len(set(seeds)) != len(seeds):
        raise ReplacementPoolError("selected candidate indices/seeds are not unique")
    if (
        selection.get("selected_candidate_indices") != indices
        or selection.get("selected_seeds") != seeds
        or selection.get("unselected_candidate_indices_at_freeze")
        != sorted(set(range(candidate_count)) - set(indices))
    ):
        raise ReplacementPoolError("selection candidate summaries are inconsistent")
    if registry is not None and selected != registry["registrations"]:
        # selected records add sealed_artifacts; compare the registration prefix.
        stripped = []
        for item in selected:
            value = dict(item)
            value.pop("sealed_artifacts", None)
            stripped.append(value)
        if stripped != registry["registrations"]:
            raise ReplacementPoolError("selection differs from registration prefix")
    return selection


def prepare_pool(
    source_roots: Sequence[str | os.PathLike[str]],
    *,
    target_total: int,
    candidate_count: int,
    out_dir: str | os.PathLike[str],
    generator_root: str | os.PathLike[str],
    generator_python: str | os.PathLike[str] = sys.executable,
    dataset_name: str = "DLCoalSim-Replacement-Candidate-Pool",
) -> dict[str, Any]:
    """Freeze source receipts and prepare one immutable C-candidate pool."""

    target_total = _require_int(target_total, "target_total")
    candidate_count = _require_int(candidate_count, "candidate_count")
    output = Path(out_dir).expanduser().resolve()
    roots = [v1._source_root(value) for value in source_roots]
    if not roots:
        raise ReplacementPoolError("at least one source root is required")
    if output in roots:
        raise ReplacementPoolError("pool out_dir must not be a source root")

    existing_path = output / POOL_PLAN_NAME
    immutable_root = Path(generator_root).expanduser().resolve()
    immutable_python = v1._absolute_executable(generator_python)
    if existing_path.exists():
        existing = _load_pool_plan(output)
        if (
            existing["target_total"] != target_total
            or existing["candidate_count"] != candidate_count
            or sorted(str(root) for root in roots)
            != sorted(str(item["root"]) for item in existing["sources"])
            or Path(existing["immutable_generator"]["root"]) != immutable_root
            or Path(existing["immutable_generator"]["python"]) != immutable_python
            or (
                existing["candidate_config"] is not None
                and existing["candidate_config"]["dataset_name"] != dataset_name
            )
        ):
            raise ReplacementPoolError(
                "POOL_PLAN.json already freezes different prepare arguments"
            )
        if int(existing["candidate_count"]):
            _validate_generation_plan(output, existing)
        elif not (output / SELECTION_NAME).exists():
            with _registration_lock(output):
                registry = _load_registry(output, existing)
                _write_registry(output / REGISTRY_NAME, registry)
                _freeze_selection(output, existing, registry)
        return existing
    if (output / REGISTRY_NAME).exists() or (output / SELECTION_NAME).exists():
        raise ReplacementPoolError("pool state exists without POOL_PLAN.json")

    try:
        snapshots = v1._audit_sources([str(root) for root in roots])
    except (v1.ReplacementPlanError, OSError, ValueError) as exc:
        raise ReplacementPoolError(f"source audit failed: {exc}") from exc
    for left, right in zip(snapshots, snapshots[1:], strict=False):
        if int(left["planned_seed_stop_exclusive"]) != int(right["seed"]):
            raise ReplacementPoolError(
                "source planned seed ranges must be strictly contiguous"
            )
    planned = sum(int(item["planned_n_samples"]) for item in snapshots)
    if planned != target_total:
        raise ReplacementPoolError(
            f"source PLANs cover {planned} samples, expected {target_total}"
        )
    completed = sum(int(item["completed_samples"]) for item in snapshots)
    if completed > target_total:
        raise ReplacementPoolError("completed source samples exceed target_total")
    required_k = target_total - completed
    if candidate_count < required_k:
        raise ReplacementPoolError("candidate_count must be at least the frozen K")
    if required_k == 0 and candidate_count != 0:
        raise ReplacementPoolError("K=0 requires candidate_count=0")
    tail_seed = max(int(item["planned_seed_stop_exclusive"]) for item in snapshots)
    source_provenance = {
        key: snapshots[0][key]
        for key in (
            "empirical_maps_sha256",
            "empirical_ne_templates_sha256",
            "runtime_environment_sha256",
        )
    }

    try:
        probe = v1._runtime_probe(immutable_root, immutable_python)
    except (v1.ReplacementPlanError, OSError, ValueError) as exc:
        raise ReplacementPoolError(f"immutable generator probe failed: {exc}") from exc
    if probe["runtime_environment_sha256"] != snapshots[0][
        "runtime_environment_sha256"
    ]:
        raise ReplacementPoolError(
            "immutable generator runtime hash differs from source PLANs"
        )

    candidate_config: dict[str, Any] | None = None
    candidate_state: dict[str, Any] | None = None
    generation_plan_sha256: str | None = None
    generation_plan_file_sha256: str | None = None
    if candidate_count:
        config = v1._replacement_config(
            snapshots[0],
            deficit=candidate_count,
            tail_seed=tail_seed,
            out_dir=output,
            dataset_name=dataset_name,
            workers=1,
        )
        candidate_config = dataclasses.asdict(config)
        candidate_state = generation_config_state(candidate_config)
        try:
            v1._run_generator_subprocess(
                immutable_root,
                immutable_python,
                [*v1._config_argv(candidate_config), "--prepare-only"],
            )
            generation_plan = v1._validate_replacement_generation_plan(
                output, candidate_config, source_provenance
            )
        except (v1.ReplacementPlanError, OSError, ValueError) as exc:
            raise ReplacementPoolError(
                f"candidate generator prepare failed: {exc}"
            ) from exc
        generation_plan_sha256 = str(generation_plan["plan_sha256"])
        generation_plan_file_sha256 = _file_sha256(
            output / "samples" / "PLAN.json"
        )

    shard_size = (
        int(candidate_config["shard_size"]) if candidate_config is not None else 1
    )
    plan: dict[str, Any] = {
        "schema": POOL_PLAN_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": POOL_PLAN_KIND,
        "status": "prepared",
        "out_dir": str(output),
        "target_total": target_total,
        "completed_source_samples": completed,
        "required_k": required_k,
        "candidate_count": candidate_count,
        "candidate_groups": _candidate_groups(required_k, candidate_count),
        "tail_seed": tail_seed,
        "seed_stride": SEED_STRIDE,
        "candidate_seed_interval": _candidate_seed_interval(
            tail_seed, candidate_count
        ),
        "candidates": _candidate_records(
            required_k=required_k,
            candidate_count=candidate_count,
            tail_seed=tail_seed,
            shard_size=shard_size,
        ),
        "selection_contract": _selection_contract(required_k),
        "source_snapshot_sha256": _canonical_sha256(snapshots),
        "sources": snapshots,
        "source_provenance": source_provenance,
        "immutable_generator": {
            "root": str(immutable_root),
            "python": str(immutable_python),
            **probe,
        },
        "candidate_config": candidate_config,
        "candidate_config_state": candidate_state,
        "generation_plan_path": (
            "samples/PLAN.json" if candidate_count else None
        ),
        "generation_plan_file_sha256": generation_plan_file_sha256,
        "generation_plan_sha256": generation_plan_sha256,
    }
    plan["pool_plan_sha256"] = _canonical_sha256(plan)
    _write_json_atomic(existing_path, plan)
    plan = _load_pool_plan(output)

    if required_k == 0:
        with _registration_lock(output):
            registry = _empty_registry(plan)
            _write_registry(output / REGISTRY_NAME, registry)
            registry = _load_registry(output, plan)
            _freeze_selection(output, plan, registry)
    return plan


def generate_candidate(
    out_dir: str | os.PathLike[str],
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Generate exactly the candidate whose index equals this array rank."""

    output = Path(out_dir).expanduser().resolve()
    plan = _load_pool_plan(output)
    if int(plan["required_k"]) == 0:
        return {
            "no_op": True,
            "required_k": 0,
            "candidate_count": 0,
        }
    if (rank is None) != (world_size is None):
        raise ReplacementPoolError("rank and world_size must be supplied together")
    if rank is None:
        try:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplacementPoolError(
                "generate requires --rank/--world-size or SLURM_PROCID/SLURM_NTASKS"
            ) from exc
    assert world_size is not None
    candidate_count = int(plan["candidate_count"])
    if world_size != candidate_count:
        raise ReplacementPoolError("world_size must exactly equal candidate_count")
    if rank < 0 or rank >= candidate_count:
        raise ReplacementPoolError("rank/candidate_index is out of range")
    _validate_generation_plan(output, plan)
    immutable = plan["immutable_generator"]
    try:
        completed = v1._run_generator_subprocess(
            Path(immutable["root"]),
            Path(immutable["python"]),
            [
                *v1._config_argv(plan["candidate_config"]),
                "--generate-only",
                "--rank",
                str(rank),
                "--world-size",
                str(world_size),
            ],
            capture_output=False,
        )
    except (v1.ReplacementPlanError, OSError, ValueError) as exc:
        raise ReplacementPoolError(f"candidate generation failed: {exc}") from exc
    return {
        "no_op": False,
        "candidate_index": rank,
        "candidate_group": plan["candidates"][rank]["candidate_group"],
        "seed": plan["candidates"][rank]["seed"],
        "rank": rank,
        "world_size": world_size,
        "returncode": completed.returncode,
    }


def register_candidate(
    out_dir: str | os.PathLike[str],
    candidate_index: int,
) -> dict[str, Any]:
    """Register one validated completion and freeze the first K prefix."""

    output = Path(out_dir).expanduser().resolve()
    plan = _load_pool_plan(output)
    candidate_index = _require_int(candidate_index, "candidate_index")
    if int(plan["required_k"]) == 0:
        raise ReplacementPoolError("K=0 pool has no candidates to register")
    if (output / SELECTION_NAME).exists():
        selection = _load_selection(output, plan)
        return {
            "status": (
                "already_selected"
                if candidate_index in selection["selected_candidate_indices"]
                else "selection_frozen"
            ),
            "candidate_index": candidate_index,
            "selection_frozen": True,
            "selected_candidate_indices": selection[
                "selected_candidate_indices"
            ],
        }

    validated = _validate_candidate_artifacts(output, plan, candidate_index)
    with _registration_lock(output):
        plan = _load_pool_plan(output)
        if (output / SELECTION_NAME).exists():
            selection = _load_selection(output, plan)
            return {
                "status": "selection_frozen",
                "candidate_index": candidate_index,
                "selection_frozen": True,
                "selected_candidate_indices": selection[
                    "selected_candidate_indices"
                ],
            }
        registry = _load_registry(output, plan)
        existing = next(
            (
                item
                for item in registry["registrations"]
                if int(item["candidate_index"]) == candidate_index
            ),
            None,
        )
        if existing is not None:
            comparable = dict(existing)
            comparable.pop("registration_ordinal", None)
            if comparable != validated:
                raise ReplacementPoolError(
                    "duplicate registration no longer matches validated artifacts"
                )
            selection = _freeze_selection(output, plan, registry)
            return {
                "status": "already_registered",
                "candidate_index": candidate_index,
                "registration_ordinal": existing["registration_ordinal"],
                "selection_frozen": selection is not None,
            }
        if len(registry["registrations"]) >= int(plan["required_k"]):
            selection = _freeze_selection(output, plan, registry)
            if selection is None:  # pragma: no cover - guarded by the count
                raise ReplacementPoolError("full registry did not freeze selection")
            return {
                "status": "selection_frozen",
                "candidate_index": candidate_index,
                "selection_frozen": True,
            }

        registration = {
            "registration_ordinal": len(registry["registrations"]) + 1,
            **validated,
        }
        registry["registrations"].append(registration)
        registry["registration_count"] = len(registry["registrations"])
        registry["status"] = (
            "selection_frozen"
            if registry["registration_count"] == int(plan["required_k"])
            else "open"
        )
        _write_registry(output / REGISTRY_NAME, registry)
        registry = _load_registry(output, plan)
        selection = _freeze_selection(output, plan, registry)
        return {
            "status": "registered",
            "candidate_index": candidate_index,
            "candidate_group": registration["candidate_group"],
            "seed": registration["seed"],
            "registration_ordinal": registration["registration_ordinal"],
            "registered_count": registry["registration_count"],
            "required_k": plan["required_k"],
            "selection_frozen": selection is not None,
            "selected_candidate_indices": (
                None
                if selection is None
                else selection["selected_candidate_indices"]
            ),
        }


def pool_status(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a fully validated, read-only pool status summary."""

    output = Path(out_dir).expanduser().resolve()
    plan = _load_pool_plan(output)
    registry = _load_registry(output, plan)
    selection = (
        _load_selection(output, plan, registry=registry)
        if (output / SELECTION_NAME).exists()
        else None
    )
    registered = int(registry["registration_count"])
    required_k = int(plan["required_k"])
    return {
        "status": "complete" if selection is not None else "collecting",
        "pool_plan": str(output / POOL_PLAN_NAME),
        "pool_plan_sha256": plan["pool_plan_sha256"],
        "required_k": required_k,
        "candidate_count": plan["candidate_count"],
        "primary_count": plan["candidate_groups"]["primary"]["count"],
        "standby_count": plan["candidate_groups"]["standby"]["count"],
        "registered_count": registered,
        "remaining_to_freeze": max(0, required_k - registered),
        "selection_frozen": selection is not None,
        "selection": (
            None if selection is None else str(output / SELECTION_NAME)
        ),
        "selected_candidate_indices": (
            [] if selection is None else selection["selected_candidate_indices"]
        ),
        "selected_seeds": [] if selection is None else selection["selected_seeds"],
        "runtime_censoring_applied": required_k > 0,
        "no_op": required_k == 0,
    }


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and freeze completion-ordered replacement pools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", action="append", required=True)
    prepare.add_argument("--target-total", type=_non_negative_int, required=True)
    prepare.add_argument("--candidate-count", type=_non_negative_int, required=True)
    prepare.add_argument("--out-dir", required=True)
    prepare.add_argument("--generator-root", required=True)
    prepare.add_argument("--generator-python", default=sys.executable)
    prepare.add_argument(
        "--dataset-name",
        default="DLCoalSim-Replacement-Candidate-Pool",
    )

    generate = subparsers.add_parser("generate")
    generate.add_argument("--out-dir", required=True)
    generate.add_argument("--rank", type=int)
    generate.add_argument("--world-size", type=int)

    register = subparsers.add_parser("register")
    register.add_argument("--out-dir", required=True)
    register.add_argument("--candidate-index", type=_non_negative_int, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--out-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_pool(
                args.source,
                target_total=args.target_total,
                candidate_count=args.candidate_count,
                out_dir=args.out_dir,
                generator_root=args.generator_root,
                generator_python=args.generator_python,
                dataset_name=args.dataset_name,
            )
            result = {
                "pool_plan": str(
                    Path(args.out_dir).expanduser().resolve() / POOL_PLAN_NAME
                ),
                "pool_plan_sha256": result["pool_plan_sha256"],
                "required_k": result["required_k"],
                "candidate_count": result["candidate_count"],
                "primary_count": result["candidate_groups"]["primary"]["count"],
                "standby_count": result["candidate_groups"]["standby"]["count"],
                "tail_seed": result["tail_seed"],
                "no_op": result["required_k"] == 0,
            }
        elif args.command == "generate":
            result = generate_candidate(
                args.out_dir,
                rank=args.rank,
                world_size=args.world_size,
            )
        elif args.command == "register":
            result = register_candidate(
                args.out_dir, args.candidate_index
            )
        else:
            result = pool_status(args.out_dir)
    except (ReplacementPoolError, v1.ReplacementPlanError) as exc:
        raise SystemExit(f"[error] {exc}") from None
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
