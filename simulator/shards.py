from __future__ import annotations

import dataclasses
import gzip
import json
import math
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import DEMOGRAPHY_PROB_FIELDS, Config, validate_config
from .dataset_state import (
    DatasetConfigMismatchError,
    stamp_done_manifest,
    validate_done_for_reuse,
)
from .empirical_maps import EmpiricalMapStore
from .ne_templates import empirical_ne_templates_available
from .sample import init_worker, simulate_one_with_retries
from .shared_maps import create_shared_empirical_map_store, release_shared_memory
from .stdpopsim_support import StdpopsimSimulationError, validate_stdpopsim_setup
from .utils import resolve_workers, safe_json_dumps


class DatasetGenerationLockedError(RuntimeError):
    """Raised when another process is already writing the same dataset."""


class DatasetSourceUnavailableError(RuntimeError):
    """Raised when no configured sample source can actually run."""


class UnsafeOutputDirectoryError(RuntimeError):
    """Raised when an output path is not a dedicated simulator directory."""


def _assert_safe_output_directory(output: Path) -> None:
    resolved = output.expanduser().resolve()
    protected = {
        Path(resolved.anchor).resolve(),
        Path.home().resolve(),
        Path(__file__).resolve().parents[1],
    }
    if resolved in protected:
        raise UnsafeOutputDirectoryError(
            f"Refusing to use protected path {resolved}; choose a dedicated dataset directory."
        )
    if not resolved.exists():
        return

    children = list(resolved.iterdir())
    if not children:
        return
    known_output = (
        (resolved / ".dlcoalsim.lock").exists()
        or (resolved / "config.json").is_file()
        or (resolved / "manifest.json").is_file()
        or (resolved / "samples" / "DONE.json").is_file()
        or any((resolved / "samples").glob("shard_*"))
        or any((resolved / "metadata").glob("samples.*"))
    )
    if not known_output:
        raise UnsafeOutputDirectoryError(
            f"Refusing to write into non-empty directory {resolved} because it has no "
            "DLCoalSim marker; choose an empty or dedicated dataset directory."
        )


@contextmanager
def dataset_output_lock(out_dir: str | Path) -> Iterator[None]:
    """Hold an inter-process lock for one output-directory transaction."""

    output = Path(out_dir)
    _assert_safe_output_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".dlcoalsim.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    windows_lock = os.name == "nt"
    acquired = False
    try:
        if windows_lock:
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.seek(0)
                handle.write("\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise DatasetGenerationLockedError(
                    f"Another DLCoalSim process is already writing {output}."
                ) from exc
            acquired = True
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DatasetGenerationLockedError(
                    f"Another DLCoalSim process is already writing {output}."
                ) from exc
            acquired = True

        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            if acquired and windows_lock:
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif acquired:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def concat_with_offsets(arrs: list[np.ndarray], dtype=None) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    total = 0
    for a in arrs:
        total += len(a)
        offsets.append(total)
    if total == 0:
        data = np.array([], dtype=dtype or np.float32)
    else:
        data = np.concatenate(arrs).astype(dtype or arrs[0].dtype, copy=False)
    return data, np.array(offsets, dtype=np.int64)


def write_shard(samples: list[dict], out_path: Path, cfg: Config) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array([s["sample_id"] for s in samples], dtype="U64")
    source_ids = np.array([s["source_id"] for s in samples], dtype=np.int16)
    demo_ids = np.array([s["demo_id"] for s in samples], dtype=np.int16)
    y = np.stack([s["target_log10_ne"] for s in samples]).astype(np.float32)

    positions, variant_offsets = concat_with_offsets([s["positions"] for s in samples], dtype=np.uint32)
    packed_bytes = int(math.ceil(cfg.n_haplotypes / 8))
    geno_list = [s["geno_packed"].reshape(-1, packed_bytes) for s in samples]
    geno_n = sum(x.shape[0] for x in geno_list)
    geno = np.concatenate(geno_list, axis=0).astype(np.uint8) if geno_n else np.zeros((0, packed_bytes), dtype=np.uint8)
    missing_flat_idx, missing_offsets = concat_with_offsets([s["missing_flat_idx"] for s in samples], dtype=np.uint32)

    obs_rec_pos, obs_rec_pos_offsets = concat_with_offsets([s["obs_rec_pos"] for s in samples], dtype=np.float32)
    obs_rec_rate, obs_rec_rate_offsets = concat_with_offsets([s["obs_rec_rate"] for s in samples], dtype=np.float32)
    obs_mut_pos, obs_mut_pos_offsets = concat_with_offsets([s["obs_mut_pos"] for s in samples], dtype=np.float32)
    obs_mut_rate, obs_mut_rate_offsets = concat_with_offsets([s["obs_mut_rate"] for s in samples], dtype=np.float32)

    save_kwargs = dict(
        sample_id=ids,
        source_id=source_ids,
        demography_id=demo_ids,
        target_log10_ne=y,
        variant_positions_bp=positions,
        variant_offsets=variant_offsets,
        genotype_packed=geno,
        missing_flat_idx=missing_flat_idx,
        missing_offsets=missing_offsets,
        missing_storage=np.array(["sparse_flat_index_uint32"], dtype="U32"),
        packed_hap_bytes=np.array([packed_bytes], dtype=np.int16),
        n_haplotypes=np.array([cfg.n_haplotypes], dtype=np.int16),
        sequence_length=np.array([s["metadata"]["sequence_length"] for s in samples], dtype=np.int64),
        obs_rec_pos=obs_rec_pos,
        obs_rec_pos_offsets=obs_rec_pos_offsets,
        obs_rec_rate=obs_rec_rate,
        obs_rec_rate_offsets=obs_rec_rate_offsets,
        obs_mut_pos=obs_mut_pos,
        obs_mut_pos_offsets=obs_mut_pos_offsets,
        obs_mut_rate=obs_mut_rate,
        obs_mut_rate_offsets=obs_mut_rate_offsets,
    )
    if cfg.include_truth:
        true_rec_pos, true_rec_pos_offsets = concat_with_offsets([s["true_rec_pos"] for s in samples], dtype=np.float32)
        true_rec_rate, true_rec_rate_offsets = concat_with_offsets([s["true_rec_rate"] for s in samples], dtype=np.float32)
        true_mut_pos, true_mut_pos_offsets = concat_with_offsets([s["true_mut_pos"] for s in samples], dtype=np.float32)
        true_mut_rate, true_mut_rate_offsets = concat_with_offsets([s["true_mut_rate"] for s in samples], dtype=np.float32)
        save_kwargs.update(
            true_rec_pos=true_rec_pos,
            true_rec_pos_offsets=true_rec_pos_offsets,
            true_rec_rate=true_rec_rate,
            true_rec_rate_offsets=true_rec_rate_offsets,
            true_mut_pos=true_mut_pos,
            true_mut_pos_offsets=true_mut_pos_offsets,
            true_mut_rate=true_mut_rate,
            true_mut_rate_offsets=true_mut_rate_offsets,
        )
    if cfg.compression:
        np.savez_compressed(out_path, **save_kwargs)
    else:
        np.savez(out_path, **save_kwargs)

    meta_path = out_path.with_suffix(".jsonl.gz")
    with gzip.open(meta_path, "wt", encoding="utf-8") as f:
        for s in samples:
            f.write(safe_json_dumps(s["metadata"]) + "\n")


def _shard_npz_path(samples_dir: Path, shard_idx: int) -> Path:
    return samples_dir / f"shard_{shard_idx:05d}.npz"


def _tmp_npz_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")


def _atomic_write_shard(samples: list[dict], out_path: Path, cfg: Config) -> None:
    tmp_path = _tmp_npz_path(out_path)
    tmp_meta = tmp_path.with_suffix(".jsonl.gz")
    final_meta = out_path.with_suffix(".jsonl.gz")
    for path in [tmp_path, tmp_meta]:
        if path.exists():
            path.unlink()
    write_shard(samples, tmp_path, cfg)
    tmp_path.replace(out_path)
    tmp_meta.replace(final_meta)


def _seed_for_index(cfg: Config, index: int) -> int:
    return cfg.seed + int(index) * 7919


def _shard_tasks(n_samples: int, shard_size: int) -> list[tuple[int, int, int]]:
    tasks: list[tuple[int, int, int]] = []
    for shard_idx, start in enumerate(range(0, n_samples, shard_size)):
        count = min(shard_size, n_samples - start)
        tasks.append((shard_idx, start, count))
    return tasks


def _count_key(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)


def _shard_file_bytes(path: Path) -> int:
    meta_path = path.with_suffix(".jsonl.gz")
    total = path.stat().st_size
    if meta_path.exists():
        total += meta_path.stat().st_size
    return total


def _generate_shard(task: tuple[int, int, int, dict]) -> dict:
    shard_idx, start, count, cfg_dict = task
    cfg = Config(**cfg_dict)
    samples: list[dict] = []
    metadata_rows: list[dict] = []
    full_metadata: list[dict] = []
    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    map_counts: dict[str, int] = {}
    shard_t0 = time.perf_counter()
    sample_time_sum = 0.0
    max_sample_sec = 0.0
    slowest_meta: dict = {}

    for index in range(start, start + count):
        sample_t0 = time.perf_counter()
        sample = simulate_one_with_retries((index, _seed_for_index(cfg, index)))
        sample_sec = time.perf_counter() - sample_t0
        sample_time_sum += sample_sec
        samples.append(sample)
        meta = sample["metadata"]
        if sample_sec > max_sample_sec:
            max_sample_sec = sample_sec
            slowest_meta = meta
        source = meta.get("source_type", "unknown")
        demo = meta.get("demography_type", "unknown")
        map_mode = meta.get("map_mode", meta.get("map_type", "unknown"))
        _count_key(source_counts, source)
        _count_key(demo_counts, demo)
        _count_key(map_counts, map_mode)
        metadata_rows.append(_metadata_row(meta))
        full_metadata.append(meta)

    out_path = _shard_npz_path(Path(cfg.out_dir) / "samples", shard_idx)
    _atomic_write_shard(samples, out_path, cfg)
    shard_sec = time.perf_counter() - shard_t0
    return {
        "shard_idx": int(shard_idx),
        "start": int(start),
        "n_samples": int(count),
        "shard_path": (Path("samples") / out_path.name).as_posix(),
        "shard_bytes": int(_shard_file_bytes(out_path)),
        "shard_sec": float(shard_sec),
        "mean_sample_sec": float(sample_time_sum / max(1, count)),
        "max_sample_sec": float(max_sample_sec),
        "slowest_demography_type": str(slowest_meta.get("demography_type", "unknown")),
        "slowest_source_type": str(slowest_meta.get("source_type", "unknown")),
        "slowest_n_variants": int(slowest_meta.get("n_variants", 0)),
        "slowest_max_Ne": float(slowest_meta.get("max_Ne", 0.0)),
        "metadata_rows": metadata_rows,
        "full_metadata": full_metadata,
        "source_counts": source_counts,
        "demography_counts": demo_counts,
        "map_counts": map_counts,
    }


def _metadata_row(meta: dict) -> dict:
    rec_slice = meta.get("rec_slice", {})
    mut_slice = meta.get("mut_slice", {})
    return {
        "sample_id": meta["sample_id"],
        "sample_index": meta.get("sample_index", 0),
        "source_type": meta.get("source_type", ""),
        "scenario": meta.get("scenario", ""),
        "scenario_key": meta.get("scenario_key", ""),
        "demography_type": meta.get("demography_type", ""),
        "demography_mixture_version": meta.get("demography_mixture_version", ""),
        "target_quality": meta.get("target_quality", "strict_single_population"),
        "empirical_ne_template": meta.get("empirical_ne_template", False),
        "template_source_key": meta.get("template_source_key", ""),
        "template_method": meta.get("template_method", ""),
        "template_population": meta.get("template_population", ""),
        "n_epochs": meta.get("n_epochs", 0),
        "n_change_points": meta.get("n_change_points", 0),
        "has_recent_event": meta.get("has_recent_event", False),
        "has_ancient_event": meta.get("has_ancient_event", False),
        "min_Ne": meta.get("min_Ne", 0.0),
        "max_Ne": meta.get("max_Ne", 0.0),
        "Ne_ratio_max_min": meta.get("Ne_ratio_max_min", 0.0),
        "recent_min_Ne": meta.get("recent_min_Ne", 0.0),
        "ancient_mean_Ne": meta.get("ancient_mean_Ne", 0.0),
        "event_severity": meta.get("event_severity", 0.0),
        "event_duration": meta.get("event_duration", 0.0),
        "map_type": meta.get("map_type", ""),
        "map_mode": meta.get("map_mode", ""),
        "recomb_mode": meta.get("recomb_mode", ""),
        "mu_mode": meta.get("mu_mode", ""),
        "noise_profile": meta.get("noise_profile", ""),
        "n_variants": meta.get("n_variants", 0),
        "variant_density_per_mb": meta.get("variant_density_per_mb", 0.0),
        "genotype_error": meta.get("genotype_error", 0.0),
        "missing_rate": meta.get("missing_rate", 0.0),
        "n_missing_genotypes": meta.get("n_missing_genotypes", 0),
        "missing_storage": meta.get("missing_storage", ""),
        "phase_switch_pair_count": meta.get("phase_switch_pair_count", 0),
        "phaseable_pair_count": meta.get("phaseable_pair_count", 0),
        "mean_obs_rec_rate": meta.get("mean_obs_rec_rate", 0.0),
        "mean_obs_mut_rate": meta.get("mean_obs_mut_rate", 0.0),
        "std_obs_rec_rate": meta.get("std_obs_rec_rate", 0.0),
        "std_obs_mut_rate": meta.get("std_obs_mut_rate", 0.0),
        "obs_rec_rate_min": meta.get("obs_rec_rate_min", 0.0),
        "obs_rec_rate_max": meta.get("obs_rec_rate_max", 0.0),
        "obs_mut_rate_min": meta.get("obs_mut_rate_min", 0.0),
        "obs_mut_rate_max": meta.get("obs_mut_rate_max", 0.0),
        "num_trees": meta.get("num_trees", 0),
        "num_sites": meta.get("num_sites", 0),
        "sequence_length": meta.get("sequence_length", 0),
        "n_haplotypes": meta.get("n_haplotypes", 0),
        "n_diploid_individuals": meta.get("n_diploid_individuals", 0),
        "sample_ploidy": meta.get("sample_ploidy", 0),
        "seed": meta.get("seed", 0),
        "phase_switch_rate_per_mb": meta.get("noise", {}).get("phase_switch_rate_per_mb", 0.0),
        "phase_switch_count": meta.get("noise", {}).get("phase_switch_count", 0),
        "phase_pairing": meta.get("phase_pairing", ""),
        "recomb_rate_unit_before": rec_slice.get("rate_unit_before", ""),
        "recomb_rate_unit_after": rec_slice.get("rate_unit_after", ""),
        "recomb_scaling_method": rec_slice.get("scaling_method", ""),
        "mu_rate_unit_before": mut_slice.get("rate_unit_before", ""),
        "mu_rate_unit_after": mut_slice.get("rate_unit_after", ""),
        "mu_scaling_method": mut_slice.get("scaling_method", ""),
    }


def _write_metadata(metadata_rows: list[dict], full_metadata: list[dict], out_dir: Path) -> dict:
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(metadata_rows)
    df.to_csv(meta_dir / "samples.csv", index=False)
    paths = {
        "summary_csv": "metadata/samples.csv",
        "full_jsonl_gz": "metadata/samples.jsonl.gz",
    }
    try:
        df.to_parquet(meta_dir / "samples.parquet", index=False)
        paths["summary_parquet"] = "metadata/samples.parquet"
    except Exception:
        pass
    with gzip.open(meta_dir / "samples.jsonl.gz", "wt", encoding="utf-8") as f:
        for meta in full_metadata:
            f.write(safe_json_dumps(meta) + "\n")
    return paths


def _load_json_object(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetConfigMismatchError(
            f"Cannot safely reuse the existing shards because {description} at {path} "
            f"is unreadable: {exc}. Rerun with --force to regenerate them."
        ) from exc
    if not isinstance(value, dict):
        raise DatasetConfigMismatchError(
            f"Cannot safely reuse the existing shards because {description} at {path} "
            "is not a JSON object. Rerun with --force to regenerate them."
        )
    return value


def _write_json_atomic(path: Path, value: dict) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _artifact_reuse_error(message: str) -> DatasetConfigMismatchError:
    return DatasetConfigMismatchError(
        f"Cannot safely reuse the existing shards because {message}. "
        "Rerun with --force to regenerate them."
    )


def _validate_completed_artifacts(manifest: dict, cfg: Config) -> None:
    """Verify that DONE describes a complete, structurally compatible corpus."""

    out_dir = Path(cfg.out_dir)
    samples_dir = out_dir / "samples"
    tasks = _shard_tasks(cfg.n_samples, cfg.shard_size)
    expected_shards = [
        (
            (Path("samples") / _shard_npz_path(samples_dir, shard_idx).name).as_posix(),
            start,
            count,
        )
        for shard_idx, start, count in tasks
    ]
    expected_paths = [path for path, _, _ in expected_shards]

    if manifest.get("n_samples") != cfg.n_samples:
        raise _artifact_reuse_error(
            f"DONE.json reports n_samples={manifest.get('n_samples')!r}, expected {cfg.n_samples}"
        )
    if manifest.get("n_shards") != len(expected_shards):
        raise _artifact_reuse_error(
            f"DONE.json reports n_shards={manifest.get('n_shards')!r}, "
            f"expected {len(expected_shards)}"
        )
    if manifest.get("shards") != expected_paths:
        raise _artifact_reuse_error("the shard list in DONE.json does not match the requested layout")

    actual_npz = sorted(
        path.relative_to(out_dir).as_posix()
        for path in samples_dir.glob("shard_*.npz")
        if path.is_file()
    )
    actual_meta = sorted(
        path.relative_to(out_dir).as_posix()
        for path in samples_dir.glob("shard_*.jsonl.gz")
        if path.is_file()
    )
    expected_meta = sorted(
        Path(path).with_suffix(".jsonl.gz").as_posix()
        for path in expected_paths
    )
    if actual_npz != sorted(expected_paths):
        raise _artifact_reuse_error("the NPZ shard files on disk do not exactly match DONE.json")
    if actual_meta != expected_meta:
        raise _artifact_reuse_error("the per-shard metadata files on disk do not exactly match DONE.json")

    required_arrays = {
        "sample_id",
        "target_log10_ne",
        "n_haplotypes",
        "sequence_length",
    }
    truth_arrays = {
        "true_rec_pos",
        "true_rec_pos_offsets",
        "true_rec_rate",
        "true_rec_rate_offsets",
        "true_mut_pos",
        "true_mut_pos_offsets",
        "true_mut_rate",
        "true_mut_rate_offsets",
    }
    total_bytes = 0
    for relative_path, start, count in expected_shards:
        shard_path = out_dir / relative_path
        meta_path = shard_path.with_suffix(".jsonl.gz")
        if shard_path.stat().st_size <= 0 or meta_path.stat().st_size <= 0:
            raise _artifact_reuse_error(f"{relative_path} or its metadata sidecar is empty")
        total_bytes += _shard_file_bytes(shard_path)

        try:
            with np.load(shard_path, allow_pickle=False) as data:
                missing = required_arrays - set(data.files)
                if missing:
                    raise ValueError(f"missing arrays: {', '.join(sorted(missing))}")
                sample_ids = np.asarray(data["sample_id"])
                expected_ids = [f"sim_{index:08d}" for index in range(start, start + count)]
                if sample_ids.ndim != 1 or sample_ids.astype(str).tolist() != expected_ids:
                    raise ValueError("sample IDs do not match the shard range")
                target = np.asarray(data["target_log10_ne"])
                if target.shape != (count, cfg.time_bins):
                    raise ValueError(
                        f"target shape is {target.shape}, expected {(count, cfg.time_bins)}"
                    )
                stored_h = np.asarray(data["n_haplotypes"]).reshape(-1)
                if stored_h.size != 1 or int(stored_h[0]) != cfg.n_haplotypes:
                    raise ValueError("stored haplotype count does not match the configuration")
                sequence_lengths = np.asarray(data["sequence_length"]).reshape(-1)
                if sequence_lengths.size != count or np.any(sequence_lengths <= 0):
                    raise ValueError("stored sequence lengths are missing or invalid")
                if not cfg.enable_stdpopsim and np.any(sequence_lengths != cfg.seq_len):
                    raise ValueError("stored sequence lengths do not match the configuration")
                present_truth = truth_arrays & set(data.files)
                if cfg.include_truth and present_truth != truth_arrays:
                    raise ValueError("truth arrays requested by the configuration are incomplete")
                if not cfg.include_truth and present_truth:
                    raise ValueError("truth arrays are present but include_truth is disabled")
        except Exception as exc:
            raise _artifact_reuse_error(f"{relative_path} is unreadable or incompatible: {exc}") from exc

        try:
            with gzip.open(meta_path, "rb") as handle:
                if not handle.read(1):
                    raise ValueError("metadata sidecar is empty")
        except Exception as exc:
            raise _artifact_reuse_error(f"{meta_path.relative_to(out_dir)} is unreadable: {exc}") from exc

    recorded_bytes = manifest.get("samples_bytes")
    if recorded_bytes is not None and recorded_bytes != total_bytes:
        raise _artifact_reuse_error(
            f"shard byte total is {total_bytes}, but DONE.json records {recorded_bytes!r}"
        )

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise _artifact_reuse_error("DONE.json has no valid aggregate metadata manifest")
    for key in ("summary_csv", "full_jsonl_gz"):
        relative = metadata.get(key)
        if not isinstance(relative, str):
            raise _artifact_reuse_error(f"DONE.json is missing metadata path {key!r}")
        path = out_dir / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise _artifact_reuse_error(f"aggregate metadata file {relative!r} is missing or empty")


def _prepare_generation_output(out_dir: Path, samples_dir: Path, done: Path) -> None:
    """Invalidate completion first, then remove artifacts from incomplete runs."""

    removed = 0

    def unlink_file(path: Path) -> None:
        nonlocal removed
        if path.is_file():
            path.unlink()
            removed += 1

    # Remove the public root manifest first; then invalidate the sample commit
    # marker before touching any shard or aggregate-metadata file.
    unlink_file(out_dir / "manifest.json")
    unlink_file(out_dir / "manifest.json.tmp")
    unlink_file(done)
    unlink_file(done.with_name(f"{done.name}.tmp"))

    for path in samples_dir.glob("shard_*"):
        unlink_file(path)
    metadata_dir = out_dir / "metadata"
    if metadata_dir.exists():
        for path in metadata_dir.glob("samples.*"):
            unlink_file(path)
    if removed:
        print(f"[cleanup] removed {removed} generated artifact(s) from an incomplete or forced run")


def _reuse_completed_dataset(done: Path, cfg: Config) -> dict:
    manifest = _load_json_object(done, "DONE.json")
    validate_done_for_reuse(manifest, cfg)
    _validate_completed_artifacts(manifest, cfg)
    print(f"[skip] samples: compatible DONE exists ({done})")
    return manifest


def generate_dataset(
    cfg: Config,
    empirical: EmpiricalMapStore,
    *,
    _lock_already_held: bool = False,
) -> dict:
    validate_config(cfg)
    if _lock_already_held:
        return _generate_dataset_locked(cfg, empirical)
    with dataset_output_lock(cfg.out_dir):
        return _generate_dataset_locked(cfg, empirical)


def reuse_completed_dataset_if_compatible(
    cfg: Config,
    *,
    _lock_already_held: bool = False,
) -> dict | None:
    """Return a verified completed manifest without loading generation inputs."""

    validate_config(cfg)

    def reuse() -> dict | None:
        done = Path(cfg.out_dir) / "samples" / "DONE.json"
        if done.exists() and not cfg.force:
            return _reuse_completed_dataset(done, cfg)
        return None

    if _lock_already_held:
        return reuse()
    with dataset_output_lock(cfg.out_dir):
        return reuse()


def _generate_dataset_locked(cfg: Config, empirical: EmpiricalMapStore) -> dict:
    n = cfg.n_samples
    out_dir = Path(cfg.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    done = samples_dir / "DONE.json"
    if done.exists() and not cfg.force:
        return _reuse_completed_dataset(done, cfg)
    empirical_available = empirical.is_available(cfg.seq_len)
    non_empirical_weight = (
        cfg.p_clean_constant
        + cfg.p_random_global
        + cfg.p_custom_hetero
        + (cfg.p_stdpopsim_anchor if cfg.enable_stdpopsim else 0.0)
    )
    if cfg.p_empirical_slice > 0.0 and not empirical_available and non_empirical_weight <= 0.0:
        raise DatasetSourceUnavailableError(
            "empirical_map_slice is the only enabled source, but no colocated "
            "recombination/mutation map interval can satisfy seq_len"
        )
    if cfg.p_empirical_slice > 0.0 and not empirical_available:
        print(
            "[warn] empirical maps unavailable, incomplete, or shorter than seq_len; "
            "empirical_map_slice weight will be disabled"
        )

    if cfg.p_empirical_human_ne_template > 0.0:
        try:
            empirical_demo_available = empirical_ne_templates_available(cfg)
        except Exception as exc:
            raise DatasetSourceUnavailableError(
                f"failed to load empirical human Ne templates: {exc}"
            ) from exc
        if not empirical_demo_available:
            other_demo_weight = sum(
                float(getattr(cfg, field))
                for field in DEMOGRAPHY_PROB_FIELDS
                if field != "p_empirical_human_ne_template"
            )
            if other_demo_weight <= 0.0:
                raise DatasetSourceUnavailableError(
                    "empirical_human_ne_template is the only enabled demography, "
                    "but no templates are available"
                )
            print(
                "[warn] empirical human Ne templates are unavailable; "
                "their demography weight will be disabled"
            )

    try:
        validate_stdpopsim_setup(cfg)
    except StdpopsimSimulationError as exc:
        raise DatasetSourceUnavailableError(str(exc)) from exc
    _prepare_generation_output(out_dir, samples_dir, done)

    shard_jobs = _shard_tasks(n, cfg.shard_size)
    workers = min(resolve_workers(cfg.workers), max(1, len(shard_jobs)))
    print(f"[generate] samples n={n} shard_size={cfg.shard_size} shards={len(shard_jobs)} workers={workers}")
    cfg_dict = dataclasses.asdict(cfg)

    shard_results: dict[int, dict] = {}
    metadata_rows: list[dict] = []
    full_metadata: list[dict] = []
    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    map_counts: dict[str, int] = {}
    completed_samples = 0
    completed_bytes = 0

    def handle_shard_result(result: dict, progress: tqdm) -> None:
        nonlocal completed_samples, completed_bytes
        shard_results[int(result["shard_idx"])] = result
        completed_samples += int(result["n_samples"])
        completed_bytes += int(result["shard_bytes"])
        avg_bytes = completed_bytes / max(1, completed_samples)
        est_total_bytes = avg_bytes * n
        progress.update(int(result["n_samples"]))
        progress.set_postfix(
            {
                "disk_GB": f"{completed_bytes / 1e9:.1f}",
                "est_GB": f"{est_total_bytes / 1e9:.1f}",
                "shard_s": f"{float(result.get('shard_sec', 0.0)):.0f}",
                "mean_s": f"{float(result.get('mean_sample_sec', 0.0)):.1f}",
                "max_s": f"{float(result.get('max_sample_sec', 0.0)):.1f}",
                "slow": str(result.get("slowest_demography_type", ""))[:18],
                "vars": int(result.get("slowest_n_variants", 0)),
            }
        )

    if workers > 1:
        shared_spec, shared_handles = create_shared_empirical_map_store(empirical)
        shared_bytes = sum(shm.size for shm in shared_handles)
        if shared_bytes:
            print(f"[maps] shared read-only empirical arrays: {shared_bytes / 1e6:.1f} MB")
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_worker,
                initargs=(cfg_dict, None, shared_spec),
            ) as ex:
                futures = {
                    ex.submit(_generate_shard, (shard_idx, start, count, cfg_dict)): shard_idx
                    for shard_idx, start, count in shard_jobs
                }
                with tqdm(total=n, desc="simulate samples", unit="sample") as progress:
                    for fut in as_completed(futures):
                        handle_shard_result(fut.result(), progress)
        finally:
            release_shared_memory(shared_handles, unlink=True)
    else:
        init_worker(cfg_dict, empirical=empirical)
        with tqdm(total=n, desc="simulate samples", unit="sample") as progress:
            for shard_idx, start, count in shard_jobs:
                handle_shard_result(_generate_shard((shard_idx, start, count, cfg_dict)), progress)

    shard_paths: list[str] = []
    for shard_idx in sorted(shard_results):
        result = shard_results[shard_idx]
        shard_paths.append(result["shard_path"])
        metadata_rows.extend(result["metadata_rows"])
        full_metadata.extend(result["full_metadata"])
        _merge_counts(source_counts, result["source_counts"])
        _merge_counts(demo_counts, result["demography_counts"])
        _merge_counts(map_counts, result["map_counts"])
    metadata_rows.sort(key=lambda row: int(row.get("sample_index", 0)))
    full_metadata.sort(key=lambda meta: int(meta.get("sample_index", 0)))

    if len(metadata_rows) != n:
        raise RuntimeError(f"generated {len(metadata_rows)} samples, expected {n}")

    metadata_paths = _write_metadata(metadata_rows, full_metadata, out_dir)
    avg_bytes_per_sample = completed_bytes / max(1, len(metadata_rows))
    samples_gb = completed_bytes / 1e9
    avg_mb_per_sample = avg_bytes_per_sample / 1e6
    print(f"[disk] samples={samples_gb:.2f} GB avg={avg_mb_per_sample:.2f} MB/sample")
    manifest = {
        "n_samples": len(metadata_rows),
        "n_shards": len(shard_paths),
        "shards": shard_paths,
        "metadata": metadata_paths,
        "empirical_maps": empirical.metadata,
        "source_counts": source_counts,
        "demography_counts": demo_counts,
        "map_counts": map_counts,
        "samples_bytes": int(completed_bytes),
        "samples_gb": float(samples_gb),
        "avg_bytes_per_sample": float(avg_bytes_per_sample),
        "avg_mb_per_sample": float(avg_mb_per_sample),
        "estimated_total_samples_gb": float((avg_bytes_per_sample * len(metadata_rows)) / 1e9),
    }
    manifest = stamp_done_manifest(manifest, cfg)
    _write_json_atomic(done, manifest)
    return manifest
