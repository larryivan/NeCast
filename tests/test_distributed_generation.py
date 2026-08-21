from __future__ import annotations

import dataclasses
import gzip
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from simulator import shards
from simulator.config import DEMOGRAPHY_PROB_FIELDS, Config, validate_config
from simulator.dataset_state import DatasetConfigMismatchError
from simulator.empirical_maps import EmpiricalMapStore, PiecewiseRateMap
from simulator.shards import (
    DatasetGenerationIncompleteError,
    finalize_dataset,
    generate_dataset,
    generate_dataset_partition,
    prepare_dataset,
)


def _config(out_dir: Path) -> Config:
    cfg = Config(
        dataset_name="distributed-generation-test",
        out_dir=str(out_dir),
        seed=424242,
        n_samples=4,
        shard_size=2,
        compute_chunk_size=1,
        seq_len=1_000,
        n_haplotypes=4,
        time_bins=4,
        min_time=50.0,
        max_time=5_000.0,
        p_clean_constant=1.0,
        p_random_global=0.0,
        p_custom_hetero=0.0,
        p_empirical_slice=0.0,
        p_stdpopsim_anchor=0.0,
        recomb_map="",
        mut_map="",
        workers=1,
        compression=False,
        include_truth=True,
    )
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_constant = 1.0
    validate_config(cfg)
    return cfg


def _assert_logical_dataset_equal(left: Path, right: Path) -> None:
    left_shards = sorted((left / "samples").glob("shard_*.npz"))
    right_shards = sorted((right / "samples").glob("shard_*.npz"))
    assert [path.name for path in left_shards] == [path.name for path in right_shards]
    for left_path, right_path in zip(left_shards, right_shards, strict=True):
        with np.load(left_path, allow_pickle=False) as left_data:
            with np.load(right_path, allow_pickle=False) as right_data:
                assert left_data.files == right_data.files
                for key in left_data.files:
                    np.testing.assert_array_equal(left_data[key], right_data[key])
        with gzip.open(left_path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as handle:
            left_rows = [json.loads(line) for line in handle]
        with gzip.open(right_path.with_suffix(".jsonl.gz"), "rt", encoding="utf-8") as handle:
            right_rows = [json.loads(line) for line in handle]
        assert left_rows == right_rows
    assert (left / "metadata" / "samples.csv").read_bytes() == (
        right / "metadata" / "samples.csv"
    ).read_bytes()


def _run_partition_in_process(
    cfg_values: dict,
    rank: int,
    world_size: int,
) -> dict:
    return generate_dataset_partition(
        Config(**cfg_values),
        EmpiricalMapStore(),
        rank=rank,
        world_size=world_size,
    )


def _hold_shared_campaign_lease(
    out_dir: str,
    ready_path: str,
    release_path: str,
) -> None:
    with shards._campaign_activity_lock(out_dir, shared=True):
        Path(ready_path).write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 30.0
        while not Path(release_path).exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("test campaign lease was not released")
            time.sleep(0.01)


def test_force_waits_for_active_generation_campaign(tmp_path: Path) -> None:
    out_dir = tmp_path / "active-campaign"
    cfg = _config(out_dir)
    prepare_dataset(cfg, EmpiricalMapStore())
    ready = tmp_path / "rank-ready"
    release = tmp_path / "release-rank"

    with ProcessPoolExecutor(max_workers=1) as process_executor:
        rank_future = process_executor.submit(
            _hold_shared_campaign_lease,
            str(out_dir),
            str(ready),
            str(release),
        )
        deadline = time.monotonic() + 10.0
        while not ready.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("active rank did not acquire its campaign lease")
            time.sleep(0.01)

        with ThreadPoolExecutor(max_workers=1) as thread_executor:
            force_future = thread_executor.submit(
                prepare_dataset,
                dataclasses.replace(cfg, force=True),
                EmpiricalMapStore(),
            )
            time.sleep(0.15)
            assert not force_future.done()
            release.write_text("release", encoding="utf-8")
            rank_future.result(timeout=10)
            assert force_future.result(timeout=10)["kind"] == (
                "dlcoalsim_generation_plan"
            )


def test_duplicate_rank_processes_commit_each_chunk_once(tmp_path: Path) -> None:
    out_dir = tmp_path / "concurrent"
    cfg = _config(out_dir)
    cfg_values = dataclasses.asdict(cfg)

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_run_partition_in_process, cfg_values, 0, 1)
            for _ in range(2)
        ]
        results = [future.result(timeout=30) for future in futures]

    assert sum(result["generated_chunks"] for result in results) == 4
    assert sum(result["reused_chunks"] for result in results) == 4
    assert finalize_dataset(cfg)["n_samples"] == 4


def test_world_size_can_change_while_committed_chunks_are_reused(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    distributed_dir = tmp_path / "distributed"
    baseline = generate_dataset(_config(baseline_dir), EmpiricalMapStore())

    cfg = _config(distributed_dir)
    first = generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=2,
    )
    assert first["generated_chunks"] == 2
    assert not (distributed_dir / "samples" / "DONE.json").exists()

    rank_zero = generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=3,
    )
    rank_one = generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=1,
        world_size=3,
    )
    rank_two = generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=2,
        world_size=3,
    )
    assert (rank_zero["generated_chunks"], rank_zero["reused_chunks"]) == (1, 1)
    assert (rank_one["generated_chunks"], rank_one["reused_chunks"]) == (1, 0)
    assert (rank_two["generated_chunks"], rank_two["reused_chunks"]) == (0, 1)

    manifest = finalize_dataset(cfg)
    assert manifest["n_samples"] == baseline["n_samples"] == 4
    assert manifest["source_counts"] == baseline["source_counts"]
    assert manifest["demography_counts"] == baseline["demography_counts"]
    assert not (distributed_dir / "samples" / ".fragments").exists()
    _assert_logical_dataset_equal(baseline_dir, distributed_dir)


def test_finalize_rejects_missing_chunks_without_deleting_valid_work(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "distributed"
    cfg = _config(out_dir)
    generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=2,
    )
    fragment_dir = out_dir / "samples" / ".fragments"
    before = {
        path.name: path.read_bytes()
        for path in fragment_dir.glob("fragment_*.done.json")
    }

    with pytest.raises(DatasetGenerationIncompleteError, match="2 committed chunk"):
        finalize_dataset(cfg)

    assert not (out_dir / "samples" / "DONE.json").exists()
    assert {
        path.name: path.read_bytes()
        for path in fragment_dir.glob("fragment_*.done.json")
    } == before

    generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=1,
        world_size=2,
    )
    assert finalize_dataset(cfg)["n_samples"] == 4


def test_corrupt_fragment_recomputes_only_that_chunk(tmp_path: Path) -> None:
    out_dir = tmp_path / "distributed"
    cfg = _config(out_dir)
    generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=1,
    )
    fragment_dir = out_dir / "samples" / ".fragments"
    corrupt = fragment_dir / "fragment_00000001.npz"
    corrupt.write_bytes(corrupt.read_bytes() + b"corrupt")

    with pytest.raises(DatasetGenerationIncompleteError, match="1 committed chunk"):
        finalize_dataset(cfg)
    assert (fragment_dir / "fragment_00000000.done.json").is_file()
    assert not (fragment_dir / "fragment_00000001.done.json").exists()

    resumed = generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=1,
    )
    assert resumed["generated_chunks"] == 1
    assert resumed["reused_chunks"] == 3
    assert finalize_dataset(cfg)["n_samples"] == 4


def test_plan_rejects_changed_compute_chunk_layout(tmp_path: Path) -> None:
    out_dir = tmp_path / "distributed"
    cfg = _config(out_dir)
    prepare_dataset(cfg, EmpiricalMapStore())

    changed = dataclasses.replace(cfg, compute_chunk_size=2)
    with pytest.raises(DatasetConfigMismatchError, match="fragment layout"):
        generate_dataset_partition(
            changed,
            EmpiricalMapStore(),
            rank=0,
            world_size=1,
        )


def test_plan_rejects_changed_generator_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "distributed"
    cfg = _config(out_dir)
    prepare_dataset(cfg, EmpiricalMapStore())
    changed_runtime = shards._generation_runtime_state()
    changed_runtime["python_version"] = "different-runtime"
    monkeypatch.setattr(
        shards,
        "_generation_runtime_state",
        lambda: changed_runtime,
    )

    with pytest.raises(
        DatasetConfigMismatchError,
        match="generator code or runtime versions differ",
    ):
        generate_dataset_partition(
            cfg,
            EmpiricalMapStore(),
            rank=0,
            world_size=1,
        )


def test_plan_rejects_changed_empirical_input_provenance(tmp_path: Path) -> None:
    out_dir = tmp_path / "distributed"
    cfg = _config(out_dir)
    first = EmpiricalMapStore(
        metadata={
            "recombination": {
                "path": "/first/location/map.bed",
                "source_size_bytes": 12,
                "source_sha256": "a" * 64,
            }
        }
    )
    prepare_dataset(cfg, first)

    same_content_new_path = EmpiricalMapStore(
        metadata={
            "recombination": {
                "path": "/another/location/map.bed",
                "source_size_bytes": 12,
                "source_sha256": "a" * 64,
            }
        }
    )
    generate_dataset_partition(
        cfg,
        same_content_new_path,
        rank=0,
        world_size=1,
    )

    changed_content = EmpiricalMapStore(
        metadata={
            "recombination": {
                "path": "/another/location/map.bed",
                "source_size_bytes": 12,
                "source_sha256": "b" * 64,
            }
        }
    )
    with pytest.raises(DatasetConfigMismatchError, match="empirical maps differ"):
        generate_dataset_partition(
            cfg,
            changed_content,
            rank=0,
            world_size=1,
        )


def test_plan_hashes_exact_parsed_empirical_map_arrays(tmp_path: Path) -> None:
    out_dir = tmp_path / "distributed"
    cfg = _config(out_dir)

    def store(rates: list[float]) -> EmpiricalMapStore:
        return EmpiricalMapStore(
            rec_maps={
                "chr1": PiecewiseRateMap(
                    chrom="chr1",
                    position=np.array([0.0, 500.0, 1_000.0]),
                    rate=np.asarray(rates, dtype=np.float64),
                )
            },
            metadata={"fixture": {"same_summary": True}},
        )

    prepare_dataset(cfg, store([1.0e-8, 2.0e-8]))
    with pytest.raises(DatasetConfigMismatchError, match="empirical maps differ"):
        generate_dataset_partition(
            cfg,
            store([2.0e-8, 1.0e-8]),
            rank=0,
            world_size=1,
        )


def test_plan_rejects_changed_empirical_ne_template_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "distributed"
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_path = (
        template_dir / "phlash_unified_populations_normalized.csv"
    )
    template_path.write_text(
        "population,generations_ago,Ne\n"
        "YRI,100,10000\n"
        "YRI,1000,20000\n",
        encoding="utf-8",
    )
    cfg = _config(out_dir)
    cfg.p_empirical_human_ne_template = 1.0
    cfg.empirical_ne_template_dir = str(template_dir)
    validate_config(cfg)
    prepare_dataset(cfg, EmpiricalMapStore())

    template_path.write_text(
        "population,generations_ago,Ne\n"
        "YRI,100,10000\n"
        "YRI,1000,25000\n",
        encoding="utf-8",
    )
    with pytest.raises(
        DatasetConfigMismatchError,
        match="empirical Ne templates differ",
    ):
        generate_dataset_partition(
            cfg,
            EmpiricalMapStore(),
            rank=0,
            world_size=1,
        )


def test_finalizer_recovers_orphan_final_shard_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "distributed"
    cfg = _config(out_dir)
    generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=1,
    )
    original_write_json = shards.write_json_atomic
    interrupted = False

    def fail_second_shard_receipt(path: Path, value: dict) -> None:
        nonlocal interrupted
        if path.name == "shard_00001.done.json" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated finalizer interruption")
        original_write_json(path, value)

    monkeypatch.setattr(shards, "write_json_atomic", fail_second_shard_receipt)
    with pytest.raises(RuntimeError, match="finalizer interruption"):
        finalize_dataset(cfg)
    first_receipt = out_dir / "samples" / ".receipts" / "shard_00000.done.json"
    first_receipt_bytes = first_receipt.read_bytes()
    assert not (out_dir / "samples" / "DONE.json").exists()

    monkeypatch.setattr(shards, "write_json_atomic", original_write_json)
    assert finalize_dataset(cfg)["n_samples"] == 4
    assert first_receipt.read_bytes() == first_receipt_bytes


def test_single_fragment_finalize_never_consumes_checkpoint_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "single-fragment-shards"
    cfg = dataclasses.replace(_config(out_dir), compute_chunk_size=2)
    generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=1,
    )
    original_write_json = shards.write_json_atomic
    interrupted = False

    def fail_first_shard_receipt(path: Path, value: dict) -> None:
        nonlocal interrupted
        if path.name == "shard_00000.done.json" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated pair-commit interruption")
        original_write_json(path, value)

    monkeypatch.setattr(shards, "write_json_atomic", fail_first_shard_receipt)
    with pytest.raises(RuntimeError, match="pair-commit interruption"):
        finalize_dataset(cfg)

    fragment = (
        out_dir / "samples" / ".fragments" / "fragment_00000000.npz"
    )
    assert fragment.is_file()
    assert fragment.with_suffix(".jsonl.gz").is_file()
    assert fragment.with_name("fragment_00000000.done.json").is_file()

    monkeypatch.setattr(shards, "write_json_atomic", original_write_json)
    assert finalize_dataset(cfg)["n_samples"] == 4


def _modify_target_without_changing_shard_schema(
    shard_path: Path,
    cfg: Config,
) -> None:
    with np.load(shard_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["target_log10_ne"] = arrays["target_log10_ne"].copy()
    arrays["target_log10_ne"][0, 0] += np.float32(0.25)
    shards._write_npz_arrays(shard_path, arrays, cfg)


def test_invalid_existing_final_receipt_is_never_resigned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "distributed"
    cfg = _config(out_dir)
    generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=1,
    )
    original_write_json = shards.write_json_atomic

    def fail_done(path: Path, value: dict) -> None:
        if path.name == "DONE.json":
            raise RuntimeError("simulated DONE interruption")
        original_write_json(path, value)

    monkeypatch.setattr(shards, "write_json_atomic", fail_done)
    with pytest.raises(RuntimeError, match="DONE interruption"):
        finalize_dataset(cfg)
    monkeypatch.setattr(shards, "write_json_atomic", original_write_json)

    shard_path = out_dir / "samples" / "shard_00000.npz"
    receipt_path = (
        out_dir / "samples" / ".receipts" / "shard_00000.done.json"
    )
    assert receipt_path.is_file()
    _modify_target_without_changing_shard_schema(shard_path, cfg)

    with pytest.raises(DatasetGenerationIncompleteError, match="2 committed chunk"):
        finalize_dataset(cfg)
    assert not shard_path.exists()
    assert not receipt_path.exists()
    assert not (out_dir / "samples" / "DONE.json").exists()


def test_final_shard_without_receipt_is_never_trusted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "missing-final-receipt"
    cfg = _config(out_dir)
    generate_dataset_partition(
        cfg,
        EmpiricalMapStore(),
        rank=0,
        world_size=1,
    )
    original_write_json = shards.write_json_atomic

    def fail_done(path: Path, value: dict) -> None:
        if path.name == "DONE.json":
            raise RuntimeError("simulated DONE interruption")
        original_write_json(path, value)

    monkeypatch.setattr(shards, "write_json_atomic", fail_done)
    with pytest.raises(RuntimeError, match="DONE interruption"):
        finalize_dataset(cfg)
    monkeypatch.setattr(shards, "write_json_atomic", original_write_json)

    shard_path = out_dir / "samples" / "shard_00000.npz"
    receipt_path = (
        out_dir / "samples" / ".receipts" / "shard_00000.done.json"
    )
    receipt_path.unlink()
    _modify_target_without_changing_shard_schema(shard_path, cfg)

    with pytest.raises(DatasetGenerationIncompleteError, match="2 committed chunk"):
        finalize_dataset(cfg)
    assert not shard_path.exists()
    assert not (out_dir / "samples" / "DONE.json").exists()


def test_completed_dataset_hash_receipts_detect_same_schema_tampering(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "complete"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())
    _modify_target_without_changing_shard_schema(
        out_dir / "samples" / "shard_00000.npz",
        cfg,
    )

    with pytest.raises(
        DatasetConfigMismatchError,
        match="final shard receipt",
    ):
        generate_dataset(cfg, EmpiricalMapStore())


def test_completed_dataset_detects_aggregate_metadata_tampering(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "complete"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())
    csv_path = out_dir / "metadata" / "samples.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")

    with pytest.raises(
        DatasetConfigMismatchError,
        match="aggregate metadata",
    ):
        generate_dataset(cfg, EmpiricalMapStore())


def test_completed_dataset_detects_done_manifest_tampering(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "complete"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())
    done_path = out_dir / "samples" / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["source_counts"]["constant"] = 999
    done_path.write_text(json.dumps(done), encoding="utf-8")

    with pytest.raises(
        DatasetConfigMismatchError,
        match="DONE.json content",
    ):
        generate_dataset(cfg, EmpiricalMapStore())
