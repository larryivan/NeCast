from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from simulator.config import Config, DEMOGRAPHY_PROB_FIELDS, validate_config
from simulator.empirical_maps import EmpiricalMapStore, PiecewiseRateMap
from simulator.shards import generate_dataset


def _empirical_store() -> EmpiricalMapStore:
    rec_map = PiecewiseRateMap(
        chrom="chr1",
        position=np.array([0.0, 500.0, 1_500.0, 3_000.0], dtype=np.float64),
        rate=np.array([1.0e-8, 2.0e-8, 1.5e-8], dtype=np.float64),
    )
    mut_map = PiecewiseRateMap(
        chrom="chr1",
        position=np.array([0.0, 750.0, 2_000.0, 3_000.0], dtype=np.float64),
        rate=np.array([1.2e-8, 0.8e-8, 1.6e-8], dtype=np.float64),
    )
    return EmpiricalMapStore(
        rec_maps={"chr1": rec_map},
        mut_maps={"chr1": mut_map},
        metadata={"fixture": {"shared": True}},
    )


def _config(out_dir: Path, workers: int) -> Config:
    cfg = Config(
        dataset_name="parallel-shared-map-test",
        out_dir=str(out_dir),
        n_samples=2,
        shard_size=1,
        seq_len=1_000,
        n_haplotypes=4,
        time_bins=4,
        min_time=50.0,
        max_time=5_000.0,
        p_clean_constant=0.0,
        p_random_global=0.0,
        p_custom_hetero=0.0,
        p_empirical_slice=1.0,
        p_stdpopsim_anchor=0.0,
        recomb_map="must-not-be-loaded-in-workers.bed",
        mut_map="must-not-be-loaded-in-workers.bed",
        observed_map_noise=0.0,
        workers=workers,
        compression=False,
    )
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_constant = 1.0
    validate_config(cfg)
    return cfg


def _assert_shards_equal(left: Path, right: Path) -> None:
    left_shards = sorted((left / "samples").glob("shard_*.npz"))
    right_shards = sorted((right / "samples").glob("shard_*.npz"))
    assert [path.name for path in left_shards] == [path.name for path in right_shards]
    for left_path, right_path in zip(left_shards, right_shards, strict=True):
        with np.load(left_path, allow_pickle=False) as left_data:
            with np.load(right_path, allow_pickle=False) as right_data:
                assert set(left_data.files) == set(right_data.files)
                for key in left_data.files:
                    np.testing.assert_array_equal(left_data[key], right_data[key], err_msg=key)


def _read_shard_metadata(dataset: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((dataset / "samples").glob("shard_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    return rows


def test_parallel_workers_share_maps_and_match_single_worker(tmp_path: Path) -> None:
    parallel_dir = tmp_path / "parallel"
    single_dir = tmp_path / "single"
    store = _empirical_store()

    parallel_manifest = generate_dataset(_config(parallel_dir, workers=2), store)
    single_manifest = generate_dataset(_config(single_dir, workers=1), store)

    assert parallel_manifest["source_counts"] == {"empirical_map_slice": 2}
    assert single_manifest["source_counts"] == {"empirical_map_slice": 2}
    _assert_shards_equal(parallel_dir, single_dir)

    metadata = _read_shard_metadata(parallel_dir)
    assert len(metadata) == 2
    for row in metadata:
        assert row["slice_alignment"] == "shared_chromosomal_interval"
        assert row["rec_slice"]["slice_start_bp"] == row["mut_slice"]["slice_start_bp"]
        assert row["rec_slice"]["slice_end_bp"] == row["mut_slice"]["slice_end_bp"]
