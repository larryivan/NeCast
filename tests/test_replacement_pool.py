from __future__ import annotations

import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from simulator.config import Config
from simulator.dataset_state import SAMPLE_GENERATOR_SCHEMA_VERSION, generation_config_state
from simulator import replacement as v1
from simulator import replacement_pool as pool


GENERATION_PLAN_SHA = "9" * 64


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _file_record(path: Path) -> dict:
    return {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _source_plan(root: Path, *, n_samples: int, seed: int) -> dict:
    config = Config(
        preset="full_v01",
        n_samples=n_samples,
        seed=seed,
        shard_size=2,
        compute_chunk_size=1,
        out_dir=str(root),
        workers=1,
    )
    state = generation_config_state(config)
    layout = {
        "n_samples": n_samples,
        "shard_size": 2,
        "compute_chunk_size": 1,
        "n_shards": (n_samples + 1) // 2,
        "n_chunks": n_samples,
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
        "empirical_maps_sha256": hashlib.sha256(b"maps").hexdigest(),
        "empirical_ne_templates_sha256": hashlib.sha256(b"templates").hexdigest(),
        "runtime_environment_sha256": hashlib.sha256(b"runtime").hexdigest(),
    }
    plan["plan_sha256"] = _canonical(plan)
    _write_json(root / "samples" / "PLAN.json", plan)
    return plan


def _source_fragment(root: Path, plan: dict, index: int) -> None:
    shard = index // 2
    relative = f"samples/.fragments/fragment_{index:08d}.npz"
    npz = root / relative
    metadata = npz.with_suffix(".jsonl.gz")
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(f"source-npz-{index}".encode())
    metadata.write_bytes(f"source-meta-{index}".encode())
    receipt = {
        "schema_version": 1,
        "plan_sha256": plan["plan_sha256"],
        "result": {
            "chunk_idx": index,
            "shard_idx": shard,
            "start": index,
            "n_samples": 1,
            "fragment_path": relative,
        },
        "files": {
            "npz": _file_record(npz),
            "metadata_jsonl_gz": _file_record(metadata),
        },
    }
    receipt["receipt_sha256"] = _canonical(receipt)
    _write_json(
        root
        / "samples"
        / ".fragments"
        / f"fragment_{index:08d}.done.json",
        receipt,
    )


def _patch_generator(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    runtime_sha256: str,
) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        v1,
        "_runtime_probe",
        lambda root, python: {
            "simulator_file": str(root / "simulator" / "__init__.py"),
            "runtime_environment_sha256": runtime_sha256,
        },
    )

    def fake_run(
        root,
        python,
        arguments,
        *,
        module=True,
        capture_output=True,
    ):
        calls.append(list(arguments))
        if "--prepare-only" in arguments:
            _write_json(output / "samples" / "PLAN.json", {"test": True})
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "", "stderr": ""},
        )()

    monkeypatch.setattr(v1, "_run_generator_subprocess", fake_run)
    monkeypatch.setattr(
        v1,
        "_validate_replacement_generation_plan",
        lambda out, config, source: {
            "plan_sha256": GENERATION_PLAN_SHA,
            **dict(source),
        },
    )
    return calls


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "case",
    n_samples: int = 5,
    completed: int = 2,
    candidate_count: int = 5,
) -> tuple[Path, Path, dict, list[list[str]]]:
    source = tmp_path / f"{name}-source"
    source_plan = _source_plan(source, n_samples=n_samples, seed=100)
    for index in range(completed):
        _source_fragment(source, source_plan, index)
    output = tmp_path / f"{name}-pool"
    calls = _patch_generator(
        monkeypatch, output, source_plan["runtime_environment_sha256"]
    )
    generator_root = tmp_path / f"{name}-generator"
    (generator_root / "simulator").mkdir(parents=True)
    (generator_root / "simulator" / "__init__.py").write_text("")
    generator_python = tmp_path / f"{name}-python"
    generator_python.write_text("")
    plan = pool.prepare_pool(
        [source],
        target_total=n_samples,
        candidate_count=candidate_count,
        out_dir=output,
        generator_root=generator_root,
        generator_python=generator_python,
    )
    return output, source, plan, calls


def _candidate_fragment(
    output: Path,
    plan: dict,
    index: int,
    *,
    seed: int | None = None,
) -> None:
    expected = plan["candidates"][index]
    npz = output / expected["npz_path"]
    metadata = output / expected["metadata_path"]
    receipt_path = output / expected["receipt_path"]
    npz.parent.mkdir(parents=True, exist_ok=True)
    npz.write_bytes(f"candidate-npz-{index}".encode())
    with gzip.open(metadata, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_id": expected["expected_sample_id"],
                    "sample_index": index,
                    "seed": expected["seed"] if seed is None else seed,
                    # These fields must never participate in pool selection.
                    "target": [999],
                    "n_variants": 999,
                    "demography_type": "ignored",
                }
            )
            + "\n"
        )
    receipt = {
        "schema_version": 1,
        "plan_sha256": plan["generation_plan_sha256"],
        "result": {
            "chunk_idx": index,
            "shard_idx": expected["shard_index"],
            "start": index,
            "n_samples": 1,
            "fragment_path": expected["npz_path"],
        },
        "files": {
            "npz": _file_record(npz),
            "metadata_jsonl_gz": _file_record(metadata),
        },
    }
    receipt["receipt_sha256"] = _canonical(receipt)
    _write_json(receipt_path, receipt)


def test_k3_c5_freezes_out_of_order_prefix_and_late_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, plan, _ = _prepare(tmp_path, monkeypatch)
    assert plan["required_k"] == 3
    assert plan["candidate_count"] == 5
    assert plan["candidate_config"]["n_samples"] == 5
    assert plan["candidate_config"]["compute_chunk_size"] == 1
    assert plan["candidate_config"]["workers"] == 1
    assert plan["candidate_groups"]["primary"]["candidate_indices"] == [0, 1, 2]
    assert plan["candidate_groups"]["standby"]["candidate_indices"] == [3, 4]

    for index in (4, 1, 3):
        _candidate_fragment(output, plan, index)
        result = pool.register_candidate(output, index)
    assert result["selection_frozen"] is True

    selection_path = output / pool.SELECTION_NAME
    before = selection_path.read_bytes()
    selection = json.loads(before)
    assert selection["selected_candidate_indices"] == [4, 1, 3]
    assert selection["registration_cutoff_ordinal"] == 3
    assert [item["registration_ordinal"] for item in selection["selected_candidates"]] == [
        1,
        2,
        3,
    ]
    assert selection["runtime_censoring"]["applied"] is True
    assert (
        selection["selection_rule"]["target_variant_demography_fields_read"]
        is False
    )
    for item in selection["selected_candidates"]:
        for key in ("receipt", "npz", "metadata_jsonl_gz"):
            source_record = item["source_artifacts"][key]
            sealed_record = item["sealed_artifacts"][key]
            assert sealed_record["size_bytes"] == source_record["size_bytes"]
            assert sealed_record["sha256"] == source_record["sha256"]
            assert (output / sealed_record["path"]).samefile(
                output / source_record["path"]
            )

    _candidate_fragment(output, plan, 0)
    assert pool.register_candidate(output, 0)["status"] == "selection_frozen"
    assert pool.register_candidate(output, 4)["status"] == "already_selected"
    assert selection_path.read_bytes() == before
    assert pool.pool_status(output)["selected_candidate_indices"] == [4, 1, 3]


def test_concurrent_registration_has_one_unique_k_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, plan, _ = _prepare(tmp_path, monkeypatch, name="concurrent")
    for index in range(5):
        _candidate_fragment(output, plan, index)

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(
            executor.map(lambda index: pool.register_candidate(output, index), range(5))
        )
    assert all(result["status"] in {"registered", "selection_frozen"} for result in results)
    selection = json.loads((output / pool.SELECTION_NAME).read_text())
    selected = selection["selected_candidate_indices"]
    assert len(selected) == 3
    assert len(set(selected)) == 3
    assert sorted(
        item["registration_ordinal"]
        for item in selection["selected_candidates"]
    ) == [1, 2, 3]
    registry_before = (output / pool.REGISTRY_NAME).read_bytes()
    pool.register_candidate(output, selected[0])
    assert (output / pool.REGISTRY_NAME).read_bytes() == registry_before


@pytest.mark.parametrize("failure", ["receipt", "payload", "seed"])
def test_bad_receipt_hash_or_seed_fails_before_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    output, _, plan, _ = _prepare(
        tmp_path, monkeypatch, name=f"bad-{failure}"
    )
    _candidate_fragment(output, plan, 0)
    expected = plan["candidates"][0]
    if failure == "receipt":
        receipt_path = output / expected["receipt_path"]
        receipt = json.loads(receipt_path.read_text())
        receipt["result"]["start"] = 1
        _write_json(receipt_path, receipt)
    elif failure == "payload":
        (output / expected["npz_path"]).write_bytes(b"tampered")
    else:
        _candidate_fragment(output, plan, 0, seed=expected["seed"] + 1)

    with pytest.raises(pool.ReplacementPoolError):
        pool.register_candidate(output, 0)
    assert not (output / pool.SELECTION_NAME).exists()
    assert pool.pool_status(output)["registered_count"] == 0


def test_less_than_k_does_not_publish_and_missing_candidate_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, plan, _ = _prepare(tmp_path, monkeypatch, name="missing")
    for index in (1, 2):
        _candidate_fragment(output, plan, index)
        pool.register_candidate(output, index)
    status = pool.pool_status(output)
    assert status["registered_count"] == 2
    assert status["remaining_to_freeze"] == 1
    assert not (output / pool.SELECTION_NAME).exists()

    # Candidate zero never produces a receipt. Candidate three is enough to
    # freeze the exact K=3 prefix without waiting for it.
    _candidate_fragment(output, plan, 3)
    pool.register_candidate(output, 3)
    assert pool.pool_status(output)["selected_candidate_indices"] == [1, 2, 3]
    assert not (output / plan["candidates"][0]["receipt_path"]).exists()


def test_selected_payload_tamper_is_detected_through_sealed_hardlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, plan, _ = _prepare(tmp_path, monkeypatch, name="tamper")
    for index in (0, 1, 2):
        _candidate_fragment(output, plan, index)
        pool.register_candidate(output, index)
    (output / plan["candidates"][1]["npz_path"]).write_bytes(b"changed")
    with pytest.raises(pool.ReplacementPoolError, match="sealed artifact"):
        pool.pool_status(output)


def test_generate_requires_one_rank_per_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, plan, calls = _prepare(tmp_path, monkeypatch, name="generate")
    with pytest.raises(pool.ReplacementPoolError, match="world_size"):
        pool.generate_candidate(output, rank=0, world_size=4)
    result = pool.generate_candidate(output, rank=4, world_size=5)
    assert result["candidate_index"] == 4
    assert result["seed"] == plan["candidates"][4]["seed"]
    assert calls[-1][-5:] == [
        "--generate-only",
        "--rank",
        "4",
        "--world-size",
        "5",
    ]


def test_k_zero_is_an_empty_frozen_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, plan, calls = _prepare(
        tmp_path,
        monkeypatch,
        name="zero",
        n_samples=2,
        completed=2,
        candidate_count=0,
    )
    assert plan["required_k"] == 0
    assert plan["candidate_config"] is None
    assert not any("--prepare-only" in call for call in calls)
    selection = json.loads((output / pool.SELECTION_NAME).read_text())
    assert selection["selected_candidates"] == []
    assert selection["runtime_censoring"]["applied"] is False
    assert pool.generate_candidate(output) == {
        "no_op": True,
        "required_k": 0,
        "candidate_count": 0,
    }
    status = pool.pool_status(output)
    assert status["status"] == "complete"
    assert status["no_op"] is True
