from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from simulator import shards
from simulator.config import DEMOGRAPHY_PROB_FIELDS, Config, validate_config
from simulator.dataset_state import DatasetConfigMismatchError
from simulator.empirical_maps import EmpiricalMapStore
from simulator.shards import (
    DatasetGenerationLockedError,
    DatasetSourceUnavailableError,
    UnsafeOutputDirectoryError,
    dataset_output_lock,
    generate_dataset,
    reuse_completed_dataset_if_compatible,
)
from simulator.stdpopsim_support import StdpopsimSimulationError


def _config(out_dir: Path, *, force: bool = False) -> Config:
    cfg = Config(
        dataset_name="recovery-test",
        out_dir=str(out_dir),
        force=force,
        n_samples=1,
        shard_size=1,
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
    )
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_constant = 1.0
    validate_config(cfg)
    return cfg


def test_no_done_cleans_stale_shards_before_regeneration(tmp_path: Path) -> None:
    out_dir = tmp_path / "dataset"
    samples = out_dir / "samples"
    metadata = out_dir / "metadata"
    samples.mkdir(parents=True)
    metadata.mkdir()
    (samples / "shard_99999.npz").write_bytes(b"stale")
    (samples / "shard_99999.jsonl.gz").write_bytes(b"stale")
    (samples / "shard_00000.tmp.npz").write_bytes(b"partial")
    (metadata / "samples.csv").write_text("stale", encoding="utf-8")
    (out_dir / "manifest.json").write_text("stale", encoding="utf-8")

    manifest = generate_dataset(_config(out_dir), EmpiricalMapStore())

    assert manifest["n_samples"] == 1
    assert [path.name for path in samples.glob("shard_*.npz")] == ["shard_00000.npz"]
    assert [path.name for path in samples.glob("shard_*.jsonl.gz")] == [
        "shard_00000.jsonl.gz"
    ]
    assert not (out_dir / "manifest.json").exists()
    assert reuse_completed_dataset_if_compatible(_config(out_dir))["n_samples"] == 1


def test_completed_dataset_with_missing_or_extra_shards_is_rejected(tmp_path: Path) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())

    missing = out_dir / "samples" / "shard_00000.jsonl.gz"
    missing.unlink()
    with pytest.raises(DatasetConfigMismatchError, match=r"metadata files.*--force"):
        generate_dataset(cfg, EmpiricalMapStore())

    missing.write_bytes(b"not-a-valid-gzip")
    (out_dir / "samples" / "shard_99999.npz").write_bytes(b"stale")
    with pytest.raises(DatasetConfigMismatchError, match=r"NPZ shard files.*--force"):
        generate_dataset(cfg, EmpiricalMapStore())


def test_completed_dataset_missing_recipe_ids_is_rejected(tmp_path: Path) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())
    shard = out_dir / "samples" / "shard_00000.npz"
    with np.load(shard, allow_pickle=False) as data:
        arrays = {
            key: np.asarray(data[key])
            for key in data.files
            if key != "generator_recipe_id"
        }
    np.savez(shard, **arrays)

    with pytest.raises(
        DatasetConfigMismatchError,
        match=r"missing arrays: generator_recipe_id.*--force",
    ):
        generate_dataset(cfg, EmpiricalMapStore())


def test_completed_dataset_with_misaligned_valid_type_id_is_rejected(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())
    shard = out_dir / "samples" / "shard_00000.npz"
    with np.load(shard, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    arrays["demography_type_id"] = np.ones_like(arrays["demography_type_id"])
    np.savez(shard, **arrays)

    with pytest.raises(
        DatasetConfigMismatchError,
        match=r"metadata demography_type_id differs.*--force",
    ):
        generate_dataset(cfg, EmpiricalMapStore())


def test_force_invalidates_done_before_failed_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())
    (out_dir / "manifest.json").write_text("previous complete dataset", encoding="utf-8")

    def fail_generation(_task):
        raise RuntimeError("simulated worker failure")

    monkeypatch.setattr(shards, "_generate_shard", fail_generation)
    with pytest.raises(RuntimeError, match="simulated worker failure"):
        generate_dataset(dataclasses.replace(cfg, force=True), EmpiricalMapStore())

    assert not (out_dir / "samples" / "DONE.json").exists()
    assert not (out_dir / "manifest.json").exists()
    assert not list((out_dir / "samples").glob("shard_*"))


def test_failed_chunk_generation_retains_fragments_until_safe_retry_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "dataset"
    cfg = dataclasses.replace(
        _config(out_dir),
        n_samples=2,
        shard_size=1,
        compute_chunk_size=1,
    )
    original_generate_shard = shards._generate_shard
    calls = 0

    def fail_after_first_fragment(task):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated chunk failure")
        return original_generate_shard(task)

    monkeypatch.setattr(shards, "_generate_shard", fail_after_first_fragment)
    with pytest.raises(RuntimeError, match="simulated chunk failure"):
        generate_dataset(cfg, EmpiricalMapStore())

    fragment_dir = out_dir / "samples" / ".fragments"
    assert not (out_dir / "samples" / "DONE.json").exists()
    assert [path.name for path in fragment_dir.glob("fragment_*.npz")] == [
        "fragment_00000000.npz"
    ]
    assert [path.name for path in fragment_dir.glob("fragment_*.jsonl.gz")] == [
        "fragment_00000000.jsonl.gz"
    ]

    resumed_chunk_ids: list[int] = []

    def record_resumed_generation(task):
        resumed_chunk_ids.append(task[0])
        return original_generate_shard(task)

    monkeypatch.setattr(shards, "_generate_shard", record_resumed_generation)
    manifest = generate_dataset(cfg, EmpiricalMapStore())

    assert resumed_chunk_ids == [1]
    assert manifest["n_samples"] == 2
    assert manifest["shards"] == [
        "samples/shard_00000.npz",
        "samples/shard_00001.npz",
    ]
    assert not fragment_dir.exists()


def test_invalid_force_config_is_rejected_before_existing_done_is_touched(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())
    done = out_dir / "samples" / "DONE.json"
    original = done.read_bytes()

    with pytest.raises(ValueError, match="n_samples"):
        generate_dataset(
            dataclasses.replace(cfg, force=True, n_samples=0),
            EmpiricalMapStore(),
        )

    assert done.read_bytes() == original


def test_output_lock_rejects_a_concurrent_writer(tmp_path: Path) -> None:
    with dataset_output_lock(tmp_path):
        with pytest.raises(DatasetGenerationLockedError, match="already writing"):
            with dataset_output_lock(tmp_path):
                pass


def test_nonempty_unmarked_output_directory_is_rejected(tmp_path: Path) -> None:
    out_dir = tmp_path / "unrelated"
    out_dir.mkdir()
    note = out_dir / "keep-me.txt"
    note.write_text("user data", encoding="utf-8")

    with pytest.raises(UnsafeOutputDirectoryError, match="non-empty directory"):
        generate_dataset(_config(out_dir), EmpiricalMapStore())

    assert note.read_text(encoding="utf-8") == "user data"
    assert not (out_dir / ".dlcoalsim.lock").exists()


def test_empirical_only_demography_requires_available_templates(tmp_path: Path) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_empirical_human_ne_template = 1.0
    cfg.empirical_ne_template_dir = str(tmp_path / "missing-templates")
    validate_config(cfg)

    with pytest.raises(DatasetSourceUnavailableError, match="only enabled demography"):
        generate_dataset(cfg, EmpiricalMapStore())

    assert not (out_dir / "samples" / "DONE.json").exists()


def test_enabled_empirical_map_source_fails_closed_when_maps_are_unavailable(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    cfg.p_empirical_slice = 0.1
    validate_config(cfg)

    with pytest.raises(
        DatasetSourceUnavailableError,
        match="refusing to disable the requested empirical source",
    ):
        generate_dataset(cfg, EmpiricalMapStore())

    assert not (out_dir / "samples" / "DONE.json").exists()


def test_unavailable_optional_demography_is_disabled_with_a_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    cfg.p_constant = 1.0
    cfg.p_empirical_human_ne_template = 1.0
    cfg.empirical_ne_template_dir = str(tmp_path / "missing-templates")
    validate_config(cfg)

    manifest = generate_dataset(cfg, EmpiricalMapStore())

    assert manifest["demography_counts"] == {"constant": 1}
    assert manifest["generator_recipe_counts"] == {"constant": 1}
    assert "templates are unavailable" in capsys.readouterr().out


def test_requested_stdpopsim_setup_failure_preserves_existing_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "dataset"
    cfg = _config(out_dir)
    generate_dataset(cfg, EmpiricalMapStore())
    done = out_dir / "samples" / "DONE.json"
    original = done.read_bytes()

    requested = dataclasses.replace(
        cfg,
        force=True,
        enable_stdpopsim=True,
        p_stdpopsim_anchor=1.0,
    )

    def fail_setup(_cfg: Config) -> None:
        raise StdpopsimSimulationError("dependency missing")

    monkeypatch.setattr(shards, "validate_stdpopsim_setup", fail_setup)
    with pytest.raises(DatasetSourceUnavailableError, match="dependency missing"):
        generate_dataset(requested, EmpiricalMapStore())

    assert done.read_bytes() == original
