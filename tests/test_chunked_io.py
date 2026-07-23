from __future__ import annotations

import dataclasses
import gzip
import zipfile
from pathlib import Path

import numpy as np

from simulator.config import Config, DEMOGRAPHY_PROB_FIELDS, validate_config
from simulator.empirical_maps import EmpiricalMapStore
from simulator.shards import _write_npz_arrays, generate_dataset


def _config(
    out_dir: Path,
    *,
    workers: int,
    compute_chunk_size: int,
) -> Config:
    cfg = Config(
        dataset_name="chunked-io-test",
        out_dir=str(out_dir),
        seed=271828,
        n_samples=4,
        shard_size=2,
        compute_chunk_size=compute_chunk_size,
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
        workers=workers,
        compression=True,
        compression_level=1,
        include_truth=True,
    )
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_constant = 1.0
    validate_config(cfg)
    return cfg


def _assert_logical_outputs_equal(left: Path, right: Path) -> None:
    left_shards = sorted((left / "samples").glob("shard_*.npz"))
    right_shards = sorted((right / "samples").glob("shard_*.npz"))
    assert [path.name for path in left_shards] == [path.name for path in right_shards]
    for left_path, right_path in zip(left_shards, right_shards, strict=True):
        with np.load(left_path, allow_pickle=False) as left_data:
            with np.load(right_path, allow_pickle=False) as right_data:
                assert left_data.files == right_data.files
                for key in left_data.files:
                    assert left_data[key].dtype == right_data[key].dtype, key
                    assert left_data[key].shape == right_data[key].shape, key
                    np.testing.assert_array_equal(left_data[key], right_data[key], err_msg=key)
                    assert (
                        left_data[key].tobytes(order="C")
                        == right_data[key].tobytes(order="C")
                    ), key
        with gzip.open(left_path.with_suffix(".jsonl.gz"), "rb") as handle:
            left_metadata = handle.read()
        with gzip.open(right_path.with_suffix(".jsonl.gz"), "rb") as handle:
            right_metadata = handle.read()
        assert left_metadata == right_metadata

    with gzip.open(left / "metadata" / "samples.jsonl.gz", "rb") as handle:
        left_aggregate = handle.read()
    with gzip.open(right / "metadata" / "samples.jsonl.gz", "rb") as handle:
        right_aggregate = handle.read()
    assert left_aggregate == right_aggregate
    assert (left / "metadata" / "samples.csv").read_bytes() == (
        right / "metadata" / "samples.csv"
    ).read_bytes()


def _sidecar_bytes(dataset: Path) -> bytes:
    parts = []
    for path in sorted((dataset / "samples").glob("shard_*.jsonl.gz")):
        with gzip.open(path, "rb") as handle:
            parts.append(handle.read())
    return b"".join(parts)


def test_npz_writer_honors_compression_level_and_preserves_arrays(tmp_path: Path) -> None:
    arrays = {
        "zeros": np.zeros(256_000, dtype=np.uint8),
        "values": np.arange(128, dtype=np.float32),
        "labels": np.array(["alpha", "beta"], dtype="U8"),
    }
    base = Config(out_dir=str(tmp_path / "unused"))
    level_zero = tmp_path / "level-zero.npz"
    level_nine = tmp_path / "level-nine.npz"
    stored = tmp_path / "stored.npz"

    _write_npz_arrays(
        level_zero,
        arrays,
        dataclasses.replace(base, compression=True, compression_level=0),
    )
    _write_npz_arrays(
        level_nine,
        arrays,
        dataclasses.replace(base, compression=True, compression_level=9),
    )
    _write_npz_arrays(
        stored,
        arrays,
        dataclasses.replace(base, compression=False, compression_level=1),
    )

    for path in (level_zero, level_nine, stored):
        with np.load(path, allow_pickle=False) as data:
            assert data.files == list(arrays)
            for key, expected in arrays.items():
                np.testing.assert_array_equal(data[key], expected)

    with zipfile.ZipFile(level_zero) as archive:
        assert {entry.compress_type for entry in archive.infolist()} == {
            zipfile.ZIP_DEFLATED
        }
    with zipfile.ZipFile(level_nine) as archive:
        assert {entry.compress_type for entry in archive.infolist()} == {
            zipfile.ZIP_DEFLATED
        }
    with zipfile.ZipFile(stored) as archive:
        assert {entry.compress_type for entry in archive.infolist()} == {
            zipfile.ZIP_STORED
        }
    assert level_nine.stat().st_size < level_zero.stat().st_size


def test_compute_chunks_and_worker_count_are_lossless(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    chunked_dir = tmp_path / "chunked"
    parallel_dir = tmp_path / "parallel"

    baseline = generate_dataset(
        _config(baseline_dir, workers=1, compute_chunk_size=2),
        EmpiricalMapStore(),
    )
    chunked = generate_dataset(
        _config(chunked_dir, workers=1, compute_chunk_size=1),
        EmpiricalMapStore(),
    )
    parallel = generate_dataset(
        _config(parallel_dir, workers=2, compute_chunk_size=1),
        EmpiricalMapStore(),
    )

    assert baseline["shards"] == chunked["shards"] == parallel["shards"]
    _assert_logical_outputs_equal(baseline_dir, chunked_dir)
    _assert_logical_outputs_equal(chunked_dir, parallel_dir)
    assert not (baseline_dir / "samples" / ".fragments").exists()
    assert not (chunked_dir / "samples" / ".fragments").exists()
    assert not (parallel_dir / "samples" / ".fragments").exists()

    for dataset in (baseline_dir, chunked_dir, parallel_dir):
        with gzip.open(dataset / "metadata" / "samples.jsonl.gz", "rb") as handle:
            aggregate = handle.read()
        assert aggregate == _sidecar_bytes(dataset)


def test_fragment_metadata_honors_compression_level(tmp_path: Path) -> None:
    level_zero_dir = tmp_path / "metadata-level-zero"
    level_nine_dir = tmp_path / "metadata-level-nine"
    level_zero = dataclasses.replace(
        _config(level_zero_dir, workers=2, compute_chunk_size=1),
        compression_level=0,
    )
    level_nine = dataclasses.replace(
        _config(level_nine_dir, workers=2, compute_chunk_size=1),
        compression_level=9,
    )

    generate_dataset(level_zero, EmpiricalMapStore())
    generate_dataset(level_nine, EmpiricalMapStore())

    zero_sidecars = sorted((level_zero_dir / "samples").glob("shard_*.jsonl.gz"))
    nine_sidecars = sorted((level_nine_dir / "samples").glob("shard_*.jsonl.gz"))
    assert [path.name for path in zero_sidecars] == [
        path.name for path in nine_sidecars
    ]
    for zero_path, nine_path in zip(zero_sidecars, nine_sidecars, strict=True):
        with gzip.open(zero_path, "rb") as handle:
            zero_metadata = handle.read()
        with gzip.open(nine_path, "rb") as handle:
            nine_metadata = handle.read()
        assert zero_metadata == nine_metadata
    assert sum(path.stat().st_size for path in nine_sidecars) < sum(
        path.stat().st_size for path in zero_sidecars
    )
