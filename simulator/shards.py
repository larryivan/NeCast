from __future__ import annotations

import dataclasses
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import time
import zipfile
from collections.abc import Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .atomic_io import (
    canonical_sha256,
    file_sha256,
    fsync_directory,
    read_json,
    write_json_atomic,
)
from .config import (
    DEMOGRAPHY_PROB_FIELDS,
    DEMOGRAPHY_TYPE_IDS,
    GENERATOR_RECIPE_IDS,
    SOURCE_IDS,
    Config,
    validate_config,
)
from .dataset_state import (
    DONE_CONFIG_STATE_KEY,
    SAMPLE_GENERATOR_SCHEMA_VERSION,
    DatasetConfigMismatchError,
    generation_config_state,
    stamp_done_manifest,
    validate_done_for_reuse,
)
from .empirical_maps import EmpiricalMapStore
from .ne_templates import (
    empirical_ne_template_provenance,
    empirical_ne_templates_available,
)
from .sample import init_worker, simulate_one_with_retries
from .shared_maps import create_shared_empirical_map_store, release_shared_memory
from .stdpopsim_support import StdpopsimSimulationError, validate_stdpopsim_setup
from .utils import resolve_workers, safe_json_dumps


class DatasetGenerationLockedError(RuntimeError):
    """Raised when another process is already writing the same dataset."""


class DatasetSourceUnavailableError(RuntimeError):
    """Raised when no configured sample source can actually run."""


class DatasetGenerationIncompleteError(RuntimeError):
    """Raised when finalization is requested before every chunk is committed."""


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
def dataset_output_lock(
    out_dir: str | Path,
    *,
    blocking: bool = False,
) -> Iterator[None]:
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
                operation = fcntl.LOCK_EX
                if not blocking:
                    operation |= fcntl.LOCK_NB
                fcntl.flock(handle.fileno(), operation)
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


@contextmanager
def _campaign_activity_lock(
    out_dir: str | Path,
    *,
    shared: bool,
) -> Iterator[None]:
    """Coordinate long-running ranks with destructive/finalizing operations."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".dlcoalsim.generation.lock"
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            # Windows' byte-range API has no shared mode; serialize there.
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(handle.fileno(), operation)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def map_rate_offsets(offsets: np.ndarray) -> np.ndarray:
    """Where each row's rates begin, derived from where its positions do.

    A map is stored with one offsets array rather than two, because the second
    carries no information the first does not: a rate lives *between* two
    breakpoints, so a row holds one fewer rate than it holds positions.  The
    count is clamped at zero because a row with no map at all holds none of
    either, not minus one rate -- which is the case a subtraction alone gets
    wrong, and the reason this is a function rather than a note in the docs.
    """

    lengths = np.diff(np.asarray(offsets, dtype=np.int64))
    return np.concatenate(([0], np.cumsum(np.maximum(lengths - 1, 0))))


def map_slice(
    positions: np.ndarray,
    rates: np.ndarray,
    offsets: np.ndarray,
    row: int,
) -> tuple[np.ndarray, np.ndarray]:
    """One row's piecewise map: its breakpoints and the rates between them."""

    rate_offsets = map_rate_offsets(offsets)
    return (
        positions[int(offsets[row]) : int(offsets[row + 1])],
        rates[int(rate_offsets[row]) : int(rate_offsets[row + 1])],
    )


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


_PER_SAMPLE_ARRAYS = (
    "sample_id",
    "source_id",
    "demography_type_id",
    "generator_recipe_id",
    "target_log10_ne",
    "sequence_length",
)
_GLOBAL_ARRAYS = (
    "generator_schema_version",
    "missing_storage",
    "packed_hap_bytes",
    "n_haplotypes",
)
_OFFSET_ARRAY_PAIRS = (
    ("variant_positions_bp", "variant_offsets"),
    ("missing_flat_idx", "missing_offsets"),
    ("obs_rec_pos", "obs_rec_offsets"),
    ("obs_mut_pos", "obs_mut_offsets"),
    ("true_rec_pos", "true_rec_offsets"),
    ("true_mut_pos", "true_mut_offsets"),
)
# Arrays whose row layout is decided by another array's offsets rather than by
# their own.  A rate sits between two breakpoints, so its rows are read through
# the positions' offsets; concatenating it is all the merge has to do.
_DERIVED_LAYOUT_ARRAYS = (
    "obs_rec_rate",
    "obs_mut_rate",
    "true_rec_rate",
    "true_mut_rate",
)
_BASE_ARRAY_ORDER = (
    "sample_id",
    "source_id",
    "demography_type_id",
    "generator_recipe_id",
    "target_log10_ne",
    "variant_positions_bp",
    "variant_offsets",
    "genotype_packed",
    "missing_flat_idx",
    "missing_offsets",
    "missing_storage",
    "generator_schema_version",
    "packed_hap_bytes",
    "n_haplotypes",
    "sequence_length",
    "obs_rec_pos",
    "obs_rec_offsets",
    "obs_rec_rate",
    "obs_mut_pos",
    "obs_mut_offsets",
    "obs_mut_rate",
)
_TRUTH_ARRAY_ORDER = (
    "true_rec_pos",
    "true_rec_offsets",
    "true_rec_rate",
    "true_mut_pos",
    "true_mut_offsets",
    "true_mut_rate",
)
# Present when ``arg_detail`` is not "none".  The shard layout has to depend on
# the setting rather than on which rows happened to be selected, so an
# unselected row carries zeros and says so in its ``arg_detail`` field.
_GENEALOGY_ARRAY_ORDER = (
    "pair_coalescences_per_bin",
    "branch_length_per_bin",
)
_FRAGMENT_DIRNAME = ".fragments"
_FRAGMENT_RECEIPT_SCHEMA_VERSION = 1
_FINAL_SHARD_RECEIPT_SCHEMA_VERSION = 1
_GENERATION_PLAN_SCHEMA_VERSION = 1
_GENERATION_PLAN_NAME = "PLAN.json"
_FINAL_RECEIPT_DIRNAME = ".receipts"


def _shard_arrays(samples: list[dict], cfg: Config) -> dict[str, np.ndarray]:
    ids = np.array([s["sample_id"] for s in samples], dtype="U64")
    source_ids = np.array([s["source_id"] for s in samples], dtype=np.int16)
    demography_type_ids = np.array(
        [s["demography_type_id"] for s in samples],
        dtype=np.int16,
    )
    generator_recipe_ids = np.array(
        [s["generator_recipe_id"] for s in samples],
        dtype=np.int16,
    )
    y = np.stack([s["target_log10_ne"] for s in samples]).astype(np.float32)

    positions, variant_offsets = concat_with_offsets([s["positions"] for s in samples], dtype=np.uint32)
    packed_bytes = int(math.ceil(cfg.n_haplotypes / 8))
    geno_list = [s["geno_packed"].reshape(-1, packed_bytes) for s in samples]
    geno_n = sum(x.shape[0] for x in geno_list)
    geno = np.concatenate(geno_list, axis=0).astype(np.uint8) if geno_n else np.zeros((0, packed_bytes), dtype=np.uint8)
    missing_flat_idx, missing_offsets = concat_with_offsets([s["missing_flat_idx"] for s in samples], dtype=np.uint32)

    # Positions float64, rates float32; see the note in ``sample.py``.
    obs_rec_pos, obs_rec_offsets = concat_with_offsets([s["obs_rec_pos"] for s in samples], dtype=np.float64)
    obs_rec_rate, _ = concat_with_offsets([s["obs_rec_rate"] for s in samples], dtype=np.float32)
    obs_mut_pos, obs_mut_offsets = concat_with_offsets([s["obs_mut_pos"] for s in samples], dtype=np.float64)
    obs_mut_rate, _ = concat_with_offsets([s["obs_mut_rate"] for s in samples], dtype=np.float32)

    arrays = dict(
        sample_id=ids,
        source_id=source_ids,
        demography_type_id=demography_type_ids,
        generator_recipe_id=generator_recipe_ids,
        target_log10_ne=y,
        variant_positions_bp=positions,
        variant_offsets=variant_offsets,
        genotype_packed=geno,
        missing_flat_idx=missing_flat_idx,
        missing_offsets=missing_offsets,
        missing_storage=np.array(["sparse_flat_index_uint32"], dtype="U32"),
        generator_schema_version=np.array(
            [SAMPLE_GENERATOR_SCHEMA_VERSION],
            dtype=np.int16,
        ),
        packed_hap_bytes=np.array([packed_bytes], dtype=np.int16),
        n_haplotypes=np.array([cfg.n_haplotypes], dtype=np.int16),
        sequence_length=np.array([s["metadata"]["sequence_length"] for s in samples], dtype=np.int64),
        obs_rec_pos=obs_rec_pos,
        obs_rec_offsets=obs_rec_offsets,
        obs_rec_rate=obs_rec_rate,
        obs_mut_pos=obs_mut_pos,
        obs_mut_offsets=obs_mut_offsets,
        obs_mut_rate=obs_mut_rate,
    )
    if cfg.include_truth:
        true_rec_pos, true_rec_offsets = concat_with_offsets([s["true_rec_pos"] for s in samples], dtype=np.float64)
        true_rec_rate, _ = concat_with_offsets([s["true_rec_rate"] for s in samples], dtype=np.float32)
        true_mut_pos, true_mut_offsets = concat_with_offsets([s["true_mut_pos"] for s in samples], dtype=np.float64)
        true_mut_rate, _ = concat_with_offsets([s["true_mut_rate"] for s in samples], dtype=np.float32)
        arrays.update(
            true_rec_pos=true_rec_pos,
            true_rec_offsets=true_rec_offsets,
            true_rec_rate=true_rec_rate,
                true_mut_pos=true_mut_pos,
            true_mut_offsets=true_mut_offsets,
            true_mut_rate=true_mut_rate,
            )
    if cfg.arg_detail != "none":
        # The genealogy summary: how many coalescences fell in each target time
        # bin, and how much branch length spanned it.  Two vectors of 64 floats
        # per row, 512 bytes, against a genotype block of one to twenty
        # megabytes -- and 0.08% of the simulation's own runtime to compute.
        #
        # They are the sufficient statistic for Ne(t): the whole inference is
        # a coalescence rate divided by an exposure.  Storing them makes three
        # things possible that are not possible from the sequence alone -- an
        # oracle bound on how well any model could do on this data, a per-bin
        # statement of what was resolvable in a given row, and an intermediate
        # supervision target.  Recorded, not used: what to do with them is a
        # separate decision from whether to keep them.
        arrays.update(
            pair_coalescences_per_bin=np.stack(
                [s["pair_coalescences_per_bin"] for s in samples]
            ).astype(np.float32),
            branch_length_per_bin=np.stack(
                [s["branch_length_per_bin"] for s in samples]
            ).astype(np.float32),
        )
    return arrays


def _write_npz_arrays(out_path: Path, arrays: dict[str, np.ndarray], cfg: Config) -> None:
    """Write an NPZ with an explicit, low-overhead DEFLATE level."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if cfg.compression else zipfile.ZIP_STORED
    zip_kwargs = {
        "mode": "w",
        "compression": compression,
        "allowZip64": True,
    }
    if cfg.compression:
        zip_kwargs["compresslevel"] = cfg.compression_level
    with zipfile.ZipFile(out_path, **zip_kwargs) as archive:
        for name, array in arrays.items():
            with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                np.lib.format.write_array(
                    member,
                    np.asanyarray(array),
                    allow_pickle=False,
                )


def write_shard(samples: list[dict], out_path: Path, cfg: Config) -> None:
    _write_npz_arrays(out_path, _shard_arrays(samples, cfg), cfg)

    meta_path = out_path.with_suffix(".jsonl.gz")
    with gzip.open(
        meta_path,
        "wt",
        encoding="utf-8",
        compresslevel=cfg.compression_level,
    ) as f:
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


def _fragment_dir(samples_dir: Path) -> Path:
    return samples_dir / _FRAGMENT_DIRNAME


def _fragment_npz_path(samples_dir: Path, chunk_idx: int) -> Path:
    return _fragment_dir(samples_dir) / f"fragment_{chunk_idx:08d}.npz"


def _fragment_receipt_path(samples_dir: Path, chunk_idx: int) -> Path:
    return _fragment_dir(samples_dir) / f"fragment_{chunk_idx:08d}.done.json"


def _fragment_lock_path(samples_dir: Path, chunk_idx: int) -> Path:
    return (
        _fragment_dir(samples_dir)
        / ".locks"
        / f"fragment_{chunk_idx:08d}.lock"
    )


def _final_receipt_path(samples_dir: Path, shard_idx: int) -> Path:
    return (
        samples_dir
        / _FINAL_RECEIPT_DIRNAME
        / f"shard_{shard_idx:05d}.done.json"
    )


def _generation_plan_path(samples_dir: Path) -> Path:
    return samples_dir / _GENERATION_PLAN_NAME


def _compute_tasks(
    n_samples: int,
    shard_size: int,
    compute_chunk_size: int,
) -> list[tuple[int, int, int, int]]:
    """Split every final shard into independently scheduled contiguous chunks."""

    tasks: list[tuple[int, int, int, int]] = []
    chunk_idx = 0
    for shard_idx, shard_start, shard_count in _shard_tasks(n_samples, shard_size):
        shard_end = shard_start + shard_count
        for start in range(shard_start, shard_end, compute_chunk_size):
            count = min(compute_chunk_size, shard_end - start)
            tasks.append((chunk_idx, shard_idx, start, count))
            chunk_idx += 1
    return tasks


_GENERATION_WORKER_CFG: Config | None = None
_GENERATION_PLAN_SHA256: str | None = None


def _init_generation_worker(
    cfg_dict: dict,
    empirical: EmpiricalMapStore | None = None,
    empirical_shared_spec=None,
    plan_sha256: str | None = None,
    empirical_ne_templates_parsed_sha256: str | None = None,
) -> None:
    global _GENERATION_PLAN_SHA256, _GENERATION_WORKER_CFG
    _GENERATION_WORKER_CFG = Config(**cfg_dict)
    _GENERATION_PLAN_SHA256 = plan_sha256
    if empirical_ne_templates_parsed_sha256 is not None:
        current_parsed_sha256 = empirical_ne_template_provenance(
            _GENERATION_WORKER_CFG,
            include_source_files=False,
        )["parsed_templates_sha256"]
        if current_parsed_sha256 != empirical_ne_templates_parsed_sha256:
            raise RuntimeError(
                "worker empirical Ne templates differ from the immutable "
                "generation plan"
            )
    init_worker(cfg_dict, empirical, empirical_shared_spec)


def _generation_worker_cfg() -> Config:
    if _GENERATION_WORKER_CFG is None:
        raise RuntimeError("shard worker was not initialized")
    return _GENERATION_WORKER_CFG


def _generation_plan_sha256() -> str:
    if _GENERATION_PLAN_SHA256 is None:
        raise RuntimeError("generation worker has no immutable plan fingerprint")
    return _GENERATION_PLAN_SHA256


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


def _generate_shard(task: tuple[int, int, int, int]) -> dict:
    chunk_idx, shard_idx, start, count = task
    cfg = _generation_worker_cfg()
    samples: list[dict] = []
    metadata_rows: list[dict] = []
    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    generator_recipe_counts: dict[str, int] = {}
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
        generator_recipe = meta.get("generator_recipe", "unknown")
        map_mode = meta.get("map_mode", meta.get("map_type", "unknown"))
        _count_key(source_counts, source)
        _count_key(demo_counts, demo)
        _count_key(generator_recipe_counts, generator_recipe)
        _count_key(map_counts, map_mode)
        metadata_rows.append(_metadata_row(meta))

    out_path = _fragment_npz_path(Path(cfg.out_dir) / "samples", chunk_idx)
    _atomic_write_shard(samples, out_path, cfg)
    shard_sec = time.perf_counter() - shard_t0
    return {
        "chunk_idx": int(chunk_idx),
        "shard_idx": int(shard_idx),
        "start": int(start),
        "n_samples": int(count),
        "fragment_path": out_path.relative_to(Path(cfg.out_dir)).as_posix(),
        "fragment_bytes": int(_shard_file_bytes(out_path)),
        "shard_sec": float(shard_sec),
        "mean_sample_sec": float(sample_time_sum / max(1, count)),
        "max_sample_sec": float(max_sample_sec),
        "slowest_demography_type": str(slowest_meta.get("demography_type", "unknown")),
        "slowest_source_type": str(slowest_meta.get("source_type", "unknown")),
        "slowest_n_variants": int(slowest_meta.get("n_variants", 0)),
        "slowest_max_Ne": float(slowest_meta.get("max_Ne", 0.0)),
        "metadata_rows": metadata_rows,
        "source_counts": source_counts,
        "demography_counts": demo_counts,
        "generator_recipe_counts": generator_recipe_counts,
        "map_counts": map_counts,
    }


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()



def _file_record(path: Path) -> dict[str, int | str]:
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
    }


def _file_record_matches(
    path: Path,
    record: object,
    *,
    full_hash: bool,
) -> bool:
    if not isinstance(record, Mapping) or not path.is_file():
        return False
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or path.stat().st_size != size
    ):
        return False
    return not full_hash or file_sha256(path) == digest


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())



@contextmanager
def _fragment_lock(samples_dir: Path, chunk_idx: int) -> Iterator[None]:
    lock_path = _fragment_lock_path(samples_dir, chunk_idx)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            if lock_path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _validate_sample_artifact_pair(
    npz_path: Path,
    metadata_path: Path,
    cfg: Config,
    *,
    expected_start: int,
    expected_count: int,
) -> None:
    array_order = (
        _BASE_ARRAY_ORDER
        + (_TRUTH_ARRAY_ORDER if cfg.include_truth else ())
        + (_GENEALOGY_ARRAY_ORDER if cfg.arg_detail != "none" else ())
    )
    expected_keys = set(array_order)
    expected_ids = [
        f"sim_{index:08d}"
        for index in range(expected_start, expected_start + expected_count)
    ]
    with np.load(npz_path, allow_pickle=False) as data:
        if set(data.files) != expected_keys:
            raise ValueError("stored arrays do not match the current shard schema")
        sample_ids = np.asarray(data["sample_id"])
        if sample_ids.shape != (expected_count,) or sample_ids.astype(str).tolist() != expected_ids:
            raise ValueError("sample IDs do not match the committed range")
        per_sample = _PER_SAMPLE_ARRAYS + (
            _GENEALOGY_ARRAY_ORDER if cfg.arg_detail != "none" else ()
        )
        for key in per_sample:
            if np.asarray(data[key]).shape[0] != expected_count:
                raise ValueError(f"{key} does not contain one value per sample")
        target = np.asarray(data["target_log10_ne"])
        if target.shape != (expected_count, cfg.time_bins) or target.dtype != np.float32:
            raise ValueError("target array has an incompatible shape or dtype")
        schema = np.asarray(data["generator_schema_version"]).reshape(-1)
        if schema.size != 1 or int(schema[0]) != SAMPLE_GENERATOR_SCHEMA_VERSION:
            raise ValueError("generator schema version mismatch")
        haplotypes = np.asarray(data["n_haplotypes"]).reshape(-1)
        if haplotypes.size != 1 or int(haplotypes[0]) != cfg.n_haplotypes:
            raise ValueError("haplotype count mismatch")
        packed_bytes = int(math.ceil(cfg.n_haplotypes / 8))
        packed = np.asarray(data["genotype_packed"])
        positions = np.asarray(data["variant_positions_bp"])
        if packed.ndim != 2 or packed.shape != (len(positions), packed_bytes):
            raise ValueError("packed genotypes are not aligned with variant positions")
        for data_key, offsets_key in _OFFSET_ARRAY_PAIRS:
            if data_key not in expected_keys:
                continue
            values = np.asarray(data[data_key])
            offsets = np.asarray(data[offsets_key], dtype=np.int64)
            if (
                offsets.shape != (expected_count + 1,)
                or int(offsets[0]) != 0
                or int(offsets[-1]) != len(values)
                or np.any(np.diff(offsets) < 0)
            ):
                raise ValueError(f"{offsets_key} is incompatible with {data_key}")

    row_count = 0
    with gzip.open(metadata_path, "rt", encoding="utf-8") as handle:
        for row_count, line in enumerate(handle, start=1):
            row = json.loads(line)
            index = row_count - 1
            if index >= expected_count:
                raise ValueError("metadata contains too many rows")
            if str(row.get("sample_id", "")) != expected_ids[index]:
                raise ValueError("metadata sample IDs do not match the committed range")
            if int(row.get("generator_schema_version", -1)) != SAMPLE_GENERATOR_SCHEMA_VERSION:
                raise ValueError("metadata generator schema version mismatch")
            if int(row.get("target_time_bins", -1)) != cfg.time_bins:
                raise ValueError("metadata target bin count mismatch")
            if (
                float(row.get("demography_resolution_log10", -1.0))
                != cfg.demography_resolution_log10
            ):
                raise ValueError("metadata history bin count mismatch")
    if row_count != expected_count:
        raise ValueError(
            f"metadata contains {row_count} rows, expected {expected_count}"
        )


def _fragment_receipt(
    result: Mapping[str, object],
    cfg: Config,
    plan_sha256: str,
) -> dict:
    samples_dir = Path(cfg.out_dir) / "samples"
    chunk_idx = int(result["chunk_idx"])
    npz_path = _fragment_npz_path(samples_dir, chunk_idx)
    metadata_path = npz_path.with_suffix(".jsonl.gz")
    persisted = {
        key: result[key]
        for key in (
            "chunk_idx",
            "shard_idx",
            "start",
            "n_samples",
            "fragment_path",
            "shard_sec",
            "mean_sample_sec",
            "max_sample_sec",
            "slowest_demography_type",
            "slowest_source_type",
            "slowest_n_variants",
            "slowest_max_Ne",
            "source_counts",
            "demography_counts",
            "generator_recipe_counts",
            "map_counts",
        )
    }
    persisted["fragment_bytes"] = int(
        npz_path.stat().st_size + metadata_path.stat().st_size
    )
    receipt = {
        "schema_version": _FRAGMENT_RECEIPT_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "result": persisted,
        "files": {
            "npz": _file_record(npz_path),
            "metadata_jsonl_gz": _file_record(metadata_path),
        },
    }
    receipt["receipt_sha256"] = _canonical_json_sha256(receipt)
    return receipt


def _discard_fragment_job(
    samples_dir: Path,
    task: tuple[int, int, int, int],
) -> None:
    chunk_idx = task[0]
    npz_path = _fragment_npz_path(samples_dir, chunk_idx)
    receipt_path = _fragment_receipt_path(samples_dir, chunk_idx)
    for path in (
        npz_path,
        npz_path.with_suffix(".jsonl.gz"),
        _tmp_npz_path(npz_path),
        _tmp_npz_path(npz_path).with_suffix(".jsonl.gz"),
        receipt_path,
        receipt_path.with_name(f"{receipt_path.name}.tmp"),
    ):
        path.unlink(missing_ok=True)
    for path in receipt_path.parent.glob(f".{receipt_path.name}.*.tmp"):
        path.unlink(missing_ok=True)


def _load_committed_fragment_result(
    task: tuple[int, int, int, int],
    cfg: Config,
    plan_sha256: str,
    *,
    full_hash: bool,
    structural: bool,
) -> dict | None:
    chunk_idx, shard_idx, start, count = task
    samples_dir = Path(cfg.out_dir) / "samples"
    receipt_path = _fragment_receipt_path(samples_dir, chunk_idx)
    npz_path = _fragment_npz_path(samples_dir, chunk_idx)
    metadata_path = npz_path.with_suffix(".jsonl.gz")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_content = dict(receipt) if isinstance(receipt, dict) else {}
        recorded_receipt_sha = receipt_content.pop("receipt_sha256", None)
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != _FRAGMENT_RECEIPT_SCHEMA_VERSION
            or receipt.get("plan_sha256") != plan_sha256
            or recorded_receipt_sha != _canonical_json_sha256(receipt_content)
        ):
            return None
        result = receipt.get("result")
        files = receipt.get("files")
        if not isinstance(result, dict) or not isinstance(files, dict):
            return None
        expected_path = npz_path.relative_to(Path(cfg.out_dir)).as_posix()
        if (
            int(result.get("chunk_idx", -1)) != chunk_idx
            or int(result.get("shard_idx", -1)) != shard_idx
            or int(result.get("start", -1)) != start
            or int(result.get("n_samples", -1)) != count
            or result.get("fragment_path") != expected_path
            or not _file_record_matches(
                npz_path,
                files.get("npz"),
                full_hash=full_hash,
            )
            or not _file_record_matches(
                metadata_path,
                files.get("metadata_jsonl_gz"),
                full_hash=full_hash,
            )
        ):
            return None
        if structural:
            _validate_sample_artifact_pair(
                npz_path,
                metadata_path,
                cfg,
                expected_start=start,
                expected_count=count,
            )
        loaded = dict(result)
        loaded["fragment_bytes"] = int(
            npz_path.stat().st_size + metadata_path.stat().st_size
        )
        return loaded
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _generate_or_resume_fragment(task: tuple[int, int, int, int]) -> dict:
    cfg = _generation_worker_cfg()
    plan_sha256 = _generation_plan_sha256()
    samples_dir = Path(cfg.out_dir) / "samples"
    with _fragment_lock(samples_dir, task[0]):
        committed = _load_committed_fragment_result(
            task,
            cfg,
            plan_sha256,
            full_hash=False,
            structural=False,
        )
        if committed is not None:
            committed["reused"] = True
            return committed

        _discard_fragment_job(samples_dir, task)
        result = _generate_shard(task)
        npz_path = _fragment_npz_path(samples_dir, task[0])
        metadata_path = npz_path.with_suffix(".jsonl.gz")
        _fsync_file(npz_path)
        _fsync_file(metadata_path)
        fsync_directory(npz_path.parent)
        receipt = _fragment_receipt(result, cfg, plan_sha256)
        write_json_atomic(_fragment_receipt_path(samples_dir, task[0]), receipt)
        committed = _load_committed_fragment_result(
            task,
            cfg,
            plan_sha256,
            full_hash=True,
            structural=True,
        )
        if committed is None:
            raise RuntimeError(f"chunk {task[0]} failed post-write validation")
        committed["reused"] = False
        return committed


def _concatenate_offset_array(
    datasets: list,
    data_key: str,
    offsets_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    pieces: list[np.ndarray] = []
    offsets = [0]
    total = 0
    for data in datasets:
        piece = np.asarray(data[data_key])
        source_offsets = np.asarray(data[offsets_key], dtype=np.int64)
        if (
            source_offsets.ndim != 1
            or len(source_offsets) == 0
            or int(source_offsets[0]) != 0
            or int(source_offsets[-1]) != len(piece)
        ):
            raise ValueError(f"fragment has invalid {offsets_key}")
        pieces.append(piece)
        offsets.extend((source_offsets[1:] + total).tolist())
        total += len(piece)
    return np.concatenate(pieces, axis=0), np.asarray(offsets, dtype=np.int64)


def _arrays_from_fragments(
    fragment_paths: list[Path],
    cfg: Config,
    expected_start: int,
    expected_count: int,
) -> dict[str, np.ndarray]:
    array_order = (
        _BASE_ARRAY_ORDER
        + (_TRUTH_ARRAY_ORDER if cfg.include_truth else ())
        + (_GENEALOGY_ARRAY_ORDER if cfg.arg_detail != "none" else ())
    )
    expected_keys = set(array_order)
    with ExitStack() as stack:
        datasets = [
            stack.enter_context(np.load(path, allow_pickle=False))
            for path in fragment_paths
        ]
        for path, data in zip(fragment_paths, datasets, strict=True):
            if set(data.files) != expected_keys:
                missing = expected_keys - set(data.files)
                extra = set(data.files) - expected_keys
                raise ValueError(
                    f"fragment {path} has incompatible arrays "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})"
                )

        values: dict[str, np.ndarray] = {}
        per_sample = _PER_SAMPLE_ARRAYS + (
            _GENEALOGY_ARRAY_ORDER if cfg.arg_detail != "none" else ()
        )
        for key in per_sample:
            values[key] = np.concatenate(
                [np.asarray(data[key]) for data in datasets],
                axis=0,
            )
        for key in _GLOBAL_ARRAYS:
            reference = np.asarray(datasets[0][key])
            for data in datasets[1:]:
                candidate = np.asarray(data[key])
                if (
                    reference.dtype != candidate.dtype
                    or reference.shape != candidate.shape
                    or reference.tobytes() != candidate.tobytes()
                ):
                    raise ValueError(f"fragments disagree on global array {key}")
            values[key] = reference

        for data_key, offsets_key in _OFFSET_ARRAY_PAIRS:
            if data_key not in expected_keys:
                continue
            values[data_key], values[offsets_key] = _concatenate_offset_array(
                datasets,
                data_key,
                offsets_key,
            )
        for key in ("genotype_packed", *_DERIVED_LAYOUT_ARRAYS):
            if key not in expected_keys:
                continue
            values[key] = np.concatenate(
                [np.asarray(data[key]) for data in datasets],
                axis=0,
            )

        expected_ids = [
            f"sim_{index:08d}"
            for index in range(expected_start, expected_start + expected_count)
        ]
        actual_ids = values["sample_id"].astype(str).tolist()
        if actual_ids != expected_ids:
            raise ValueError("fragment sample IDs do not cover the final shard range")
        if values["target_log10_ne"].shape[0] != expected_count:
            raise ValueError("fragments do not contain the expected number of samples")
        if len(values["genotype_packed"]) != len(values["variant_positions_bp"]):
            raise ValueError("fragment genotype and variant arrays are misaligned")
        return {key: values[key] for key in array_order}


def _stream_gzip_members(
    source_paths: list[Path],
    out_path: Path,
    *,
    compression_level: int,
) -> int:
    """Transcode ordered gzip members into one bounded-memory gzip stream."""

    line_count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wb", compresslevel=compression_level) as target:
        for source_path in source_paths:
            with gzip.open(source_path, "rb") as source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    line_count += chunk.count(b"\n")
                    target.write(chunk)
    return line_count


def _merge_fragment_group(
    results: list[dict],
    final_task: tuple[int, int, int],
    cfg: Config,
) -> dict:
    shard_idx, start, count = final_task
    ordered = sorted(results, key=lambda result: int(result["start"]))
    cursor = start
    for result in ordered:
        if int(result["shard_idx"]) != shard_idx or int(result["start"]) != cursor:
            raise ValueError(f"fragment coverage for shard {shard_idx} is not contiguous")
        cursor += int(result["n_samples"])
    if cursor != start + count:
        raise ValueError(f"fragment coverage for shard {shard_idx} is incomplete")

    out_dir = Path(cfg.out_dir)
    fragment_paths = [out_dir / str(result["fragment_path"]) for result in ordered]
    out_path = _shard_npz_path(out_dir / "samples", shard_idx)
    final_meta = out_path.with_suffix(".jsonl.gz")
    if len(fragment_paths) == 1:
        fragment_path = fragment_paths[0]
        fragment_meta = fragment_path.with_suffix(".jsonl.gz")
        tmp_path = _tmp_npz_path(out_path)
        tmp_meta = tmp_path.with_suffix(".jsonl.gz")
        for path in (tmp_path, tmp_meta):
            path.unlink(missing_ok=True)
        try:
            os.link(fragment_path, tmp_path)
            os.link(fragment_meta, tmp_meta)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)
            shutil.copyfile(fragment_path, tmp_path)
            shutil.copyfile(fragment_meta, tmp_meta)
        tmp_path.replace(out_path)
        tmp_meta.replace(final_meta)
        return {
            "shard_idx": int(shard_idx),
            "shard_path": (Path("samples") / out_path.name).as_posix(),
            "shard_bytes": int(_shard_file_bytes(out_path)),
        }

    arrays = _arrays_from_fragments(fragment_paths, cfg, start, count)
    tmp_path = _tmp_npz_path(out_path)
    tmp_meta = tmp_path.with_suffix(".jsonl.gz")
    for path in (tmp_path, tmp_meta):
        if path.exists():
            path.unlink()
    _write_npz_arrays(tmp_path, arrays, cfg)
    metadata_rows = _stream_gzip_members(
        [path.with_suffix(".jsonl.gz") for path in fragment_paths],
        tmp_meta,
        compression_level=cfg.compression_level,
    )
    if metadata_rows != count:
        raise ValueError(
            f"fragment metadata for shard {shard_idx} has {metadata_rows} rows, expected {count}"
        )
    tmp_path.replace(out_path)
    tmp_meta.replace(final_meta)
    return {
        "shard_idx": int(shard_idx),
        "shard_path": (Path("samples") / out_path.name).as_posix(),
        "shard_bytes": int(_shard_file_bytes(out_path)),
    }


def _bounded_process_results(
    executor: ProcessPoolExecutor,
    jobs: list[tuple[int, int, int, int]],
    *,
    max_in_flight: int,
) -> Iterator[dict]:
    iterator = iter(jobs)
    pending = {}

    def fill() -> None:
        while len(pending) < max_in_flight:
            try:
                job = next(iterator)
            except StopIteration:
                return
            pending[executor.submit(_generate_or_resume_fragment, job)] = job

    fill()
    try:
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                yield future.result()
            fill()
    except BaseException:
        for future in pending:
            future.cancel()
        raise


def _metadata_row(meta: dict) -> dict:
    rec_slice = meta.get("rec_slice", {})
    mut_slice = meta.get("mut_slice", {})
    return {
        "sample_id": meta["sample_id"],
        "sample_index": meta.get("sample_index", 0),
        "source_type": meta.get("source_type", ""),
        "scenario": meta.get("scenario", ""),
        "scenario_key": meta.get("scenario_key", ""),
        "generator_recipe_key": meta.get("generator_recipe_key", ""),
        "generator_recipe": meta.get("generator_recipe", ""),
        "generator_recipe_id": meta.get("generator_recipe_id", -1),
        "generator_schema_version": meta.get("generator_schema_version", -1),
        "demography_type": meta.get("demography_type", ""),
        "demography_type_id": meta.get("demography_type_id", -1),
        "history_family": meta.get("history_family", ""),
        "demography_mixture_version": meta.get("demography_mixture_version", ""),
        "demography_function": meta.get("demography_function", "piecewise_constant"),
        "trajectory_form": meta.get("trajectory_form", ""),
        "curve_origin": meta.get("curve_origin", ""),
        "dataset_regime": meta.get("dataset_regime", ""),
        "direction": meta.get("direction", ""),
        "time_regime": meta.get("time_regime", ""),
        "native_continuous_growth": meta.get("native_continuous_growth", False),
        "literature_demography_anchor": meta.get("literature_demography_anchor", False),
        "literature_model_id": meta.get("literature_model_id", ""),
        "growth_rate_per_generation": meta.get("growth_rate_per_generation", 0.0),
        "target_quality": meta.get("target_quality", "strict_single_population"),
        "target_time_bins": meta.get("target_time_bins", 0),
        "demography_resolution_log10": meta.get("demography_resolution_log10", 0.0),
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
        "log10_ne_range": meta.get("log10_ne_range", 0.0),
        "total_variation_log10": meta.get("total_variation_log10", 0.0),
        "mean_abs_log10_change": meta.get("mean_abs_log10_change", 0.0),
        "turn_count": meta.get("turn_count", 0),
        # The spectral walk is a parametric curve, so these six scalars are the
        # history itself rather than a summary of it: they replay it exactly
        # onto any grid, which is what lets a row generated against a 64-bin
        # target be re-scored against 96 or 128 without regenerating it.  Zero
        # on every other recipe.
        "spectral_seed": meta.get("spectral_seed", 0),
        "spectral_n_modes": meta.get("spectral_n_modes", 0),
        "spectral_slope": meta.get("spectral_slope", 0.0),
        "spectral_tilt": meta.get("spectral_tilt", 0.0),
        "spectral_ne_range": meta.get("spectral_ne_range", 0.0),
        "spectral_level": meta.get("spectral_level", 0.0),
        # The rest of what the samplers record.  This whitelist is the only
        # path from a sampler to the disk, and every field a recipe draws has
        # to be on it or the claim that grouping can be done at analysis time
        # on realised parameters -- which is what replaced 27 recipe names --
        # is false.  ``held_out_evaluation`` is the one that mattered most: it
        # is how a split keeps the stdpopsim catalogue out of the training
        # gradient, and without it here that flag existed only in memory.
        "held_out_evaluation": meta.get("held_out_evaluation", False),
        "arg_detail": meta.get("arg_detail", "none"),
        "arg_path": meta.get("arg_path", ""),
        "pair_coalescences_total": meta.get("pair_coalescences_total", 0.0),
        "resolvable_bins": meta.get("resolvable_bins", 0),
        "stress_scenario": meta.get("stress_scenario", False),
        "curve_demography": meta.get("curve_demography", False),
        "knot_count": meta.get("knot_count", 0),
        "event_time": meta.get("event_time", 0.0),
        "event_count": meta.get("event_count", 0),
        "event_permanent": meta.get("event_permanent", False),
        # A row can hold several events, so the list itself does not fit a
        # fixed-width row; the deepest and the shallowest do, and between them
        # they carry the sign, the magnitude and whether the events agree.
        "event_log_depth_min": (
            float(min(meta["event_log_depths"])) if meta.get("event_log_depths") else 0.0
        ),
        "event_log_depth_max": (
            float(max(meta["event_log_depths"])) if meta.get("event_log_depths") else 0.0
        ),
        "event_family": meta.get("event_family", ""),
        "bottleneck_recovery_log10": meta.get("bottleneck_recovery_log10", 0.0),
        "exponential_log10_factor": meta.get("exponential_log10_factor", 0.0),
        "exponential_direction": meta.get("exponential_direction", ""),
        "ooa_ancestral_ne": meta.get("ooa_ancestral_ne", 0.0),
        "ooa_expansion_factor": meta.get("ooa_expansion_factor", 0.0),
        "ooa_bottleneck_factor": meta.get("ooa_bottleneck_factor", 0.0),
        "ooa_growth_factor": meta.get("ooa_growth_factor", 0.0),
        "ooa_bottleneck_duration": meta.get("ooa_bottleneck_duration", 0.0),
        "oscillation_strength_fraction": meta.get("oscillation_strength_fraction", 0.0),
        "oscillation_onset": meta.get("oscillation_onset", 0.0),
        "oscillation_end": meta.get("oscillation_end", 0.0),
        "oscillation_regularity": meta.get("oscillation_regularity", 0.0),
        "oscillation_swing_uniformity": meta.get("oscillation_swing_uniformity", 0.0),
        "oscillation_amplitude_drift": meta.get("oscillation_amplitude_drift", 0.0),
        "zigzag_rate_change_count": meta.get("zigzag_rate_change_count", 0),
        "zigzag_internal_extrema_count": meta.get("zigzag_internal_extrema_count", 0),
        "sawtooth_half_cycles": meta.get("sawtooth_half_cycles", 0),
        "sawtooth_onset": meta.get("sawtooth_onset", 0.0),
        "sawtooth_log_ne_contrast": meta.get("sawtooth_log_ne_contrast", 0.0),
        "literature_shaped_family": meta.get("literature_shaped_family", False),
        "source_literature_model_id": meta.get("source_literature_model_id", ""),
        "stdpopsim_species": meta.get("stdpopsim_species", ""),
        "stdpopsim_model_id": meta.get("stdpopsim_model_id", ""),
        "stdpopsim_ne_rescale": meta.get("stdpopsim_ne_rescale", 0.0),
        "template_time_scale": meta.get("template_time_scale", 0.0),
        "template_log10_shift": meta.get("template_log10_shift", 0.0),
        "template_contrast_scale": meta.get("template_contrast_scale", 0.0),
        "template_min_Ne": meta.get("template_min_Ne", 0.0),
        "template_max_Ne": meta.get("template_max_Ne", 0.0),
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
        "phase_switch_rate_per_het": meta.get("noise", {}).get("phase_switch_rate_per_het", 0.0),
        "phase_switch_count": meta.get("noise", {}).get("phase_switch_count", 0),
        "false_variant_rate_per_bp": meta.get("noise", {}).get("false_variant_rate_per_bp", 0.0),
        "false_variant_count": meta.get("noise", {}).get("false_variant_count", 0),
        "phase_pairing": meta.get("phase_pairing", ""),
        "recomb_rate_unit_before": rec_slice.get("rate_unit_before", ""),
        "recomb_rate_unit_after": rec_slice.get("rate_unit_after", ""),
        "recomb_scaling_method": rec_slice.get("scaling_method", ""),
        "mu_rate_unit_before": mut_slice.get("rate_unit_before", ""),
        "mu_rate_unit_after": mut_slice.get("rate_unit_after", ""),
        "mu_scaling_method": mut_slice.get("scaling_method", ""),
    }



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
        "source_id",
        "demography_type_id",
        "generator_recipe_id",
        "generator_schema_version",
        "target_log10_ne",
        "n_haplotypes",
        "sequence_length",
    }
    truth_arrays = {
        "true_rec_pos",
        "true_rec_offsets",
        "true_rec_rate",
            "true_mut_pos",
        "true_mut_offsets",
        "true_mut_rate",
        }
    plan_sha256 = manifest.get("generation_plan_sha256")
    if plan_sha256 is not None:
        if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
            raise _artifact_reuse_error(
                "DONE.json has a malformed generation-plan fingerprint"
            )
        plan_path = _generation_plan_path(samples_dir)
        if not plan_path.is_file():
            raise _artifact_reuse_error(
                "DONE.json references a missing generation PLAN.json"
            )
        plan = _validate_generation_plan(
            _load_json_object(plan_path, "PLAN.json"),
            cfg,
            empirical=None,
        )
        if plan.get("plan_sha256") != plan_sha256:
            raise _artifact_reuse_error(
                "DONE.json and PLAN.json have different fingerprints"
            )
    total_bytes = 0
    for shard_idx, (relative_path, start, count) in enumerate(expected_shards):
        shard_path = out_dir / relative_path
        meta_path = shard_path.with_suffix(".jsonl.gz")
        if shard_path.stat().st_size <= 0 or meta_path.stat().st_size <= 0:
            raise _artifact_reuse_error(f"{relative_path} or its metadata sidecar is empty")
        total_bytes += _shard_file_bytes(shard_path)

        stored_ids_for_metadata: dict[str, list[int]] = {}
        try:
            with np.load(shard_path, allow_pickle=False) as data:
                missing = required_arrays - set(data.files)
                if missing:
                    raise ValueError(f"missing arrays: {', '.join(sorted(missing))}")
                sample_ids = np.asarray(data["sample_id"])
                expected_ids = [f"sim_{index:08d}" for index in range(start, start + count)]
                if sample_ids.ndim != 1 or sample_ids.astype(str).tolist() != expected_ids:
                    raise ValueError("sample IDs do not match the shard range")
                for id_key in (
                    "source_id",
                    "demography_type_id",
                    "generator_recipe_id",
                ):
                    ids = np.asarray(data[id_key])
                    if ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer):
                        raise ValueError(
                            f"{id_key} shape/dtype is {ids.shape}/{ids.dtype}, "
                            f"expected {(count,)}/integer"
                        )
                valid_ids = {
                    "source_id": set(SOURCE_IDS.values()),
                    "demography_type_id": set(DEMOGRAPHY_TYPE_IDS.values()),
                    "generator_recipe_id": set(GENERATOR_RECIPE_IDS.values()),
                }
                for id_key, allowed in valid_ids.items():
                    if not set(np.asarray(data[id_key], dtype=np.int64)).issubset(allowed):
                        raise ValueError(f"{id_key} contains an unknown current-schema ID")
                    stored_ids_for_metadata[id_key] = np.asarray(
                        data[id_key],
                        dtype=np.int64,
                    ).tolist()
                schema_version = np.asarray(
                    data["generator_schema_version"],
                ).reshape(-1)
                if (
                    schema_version.size != 1
                    or int(schema_version[0]) != SAMPLE_GENERATOR_SCHEMA_VERSION
                ):
                    raise ValueError(
                        "stored generator schema version does not match this release"
                    )
                target = np.asarray(data["target_log10_ne"])
                if target.shape != (count, cfg.time_bins):
                    raise ValueError(
                        f"target shape is {target.shape}, expected {(count, cfg.time_bins)}"
                    )
                if target.dtype != np.float32:
                    raise ValueError(
                        f"target dtype is {target.dtype}, expected float32"
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
            with gzip.open(meta_path, "rt", encoding="utf-8") as handle:
                metadata_rows = [
                    json.loads(line)
                    for line in handle
                    if line.strip()
                ]
            if len(metadata_rows) != count:
                raise ValueError(
                    f"metadata sidecar has {len(metadata_rows)} rows, expected {count}"
                )
            for index, row in enumerate(metadata_rows):
                if str(row.get("sample_id", "")) != expected_ids[index]:
                    raise ValueError("metadata sample IDs do not match the shard")
                if (
                    int(row.get("generator_schema_version", -1))
                    != SAMPLE_GENERATOR_SCHEMA_VERSION
                ):
                    raise ValueError("metadata generator schema version mismatch")
                if int(row.get("target_time_bins", -1)) != cfg.time_bins:
                    raise ValueError("metadata target bin count mismatch")
                if (
                    float(row.get("demography_resolution_log10", -1.0))
                    != cfg.demography_resolution_log10
                ):
                    raise ValueError(
                        "metadata demography generation bin count mismatch"
                    )
                for id_key in (
                    "source_id",
                    "demography_type_id",
                    "generator_recipe_id",
                ):
                    if int(row.get(id_key, -1)) != stored_ids_for_metadata[id_key][index]:
                        raise ValueError(f"metadata {id_key} differs from the shard array")
                demography_type = str(row.get("demography_type", ""))
                generator_recipe = str(row.get("generator_recipe", ""))
                if (
                    DEMOGRAPHY_TYPE_IDS.get(demography_type)
                    != stored_ids_for_metadata["demography_type_id"][index]
                ):
                    raise ValueError("metadata demography type string/ID mismatch")
                if (
                    GENERATOR_RECIPE_IDS.get(generator_recipe)
                    != stored_ids_for_metadata["generator_recipe_id"][index]
                ):
                    raise ValueError("metadata generator recipe string/ID mismatch")
        except Exception as exc:
            raise _artifact_reuse_error(f"{meta_path.relative_to(out_dir)} is unreadable: {exc}") from exc

        if plan_sha256 is not None:
            committed = _load_committed_final_result(
                (shard_idx, start, count),
                cfg,
                plan_sha256,
                full_hash=True,
                structural=False,
            )
            if committed is None:
                raise _artifact_reuse_error(
                    f"{relative_path} does not match its final shard receipt"
                )

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

    metadata_integrity = manifest.get("metadata_integrity")
    if plan_sha256 is not None and not isinstance(metadata_integrity, Mapping):
        raise _artifact_reuse_error(
            "DONE.json has no aggregate-metadata integrity records"
        )
    if isinstance(metadata_integrity, Mapping):
        if set(metadata_integrity) != set(metadata):
            raise _artifact_reuse_error(
                "aggregate-metadata integrity records do not match the metadata manifest"
            )
        for key, relative in metadata.items():
            record = metadata_integrity.get(key)
            if (
                not isinstance(record, Mapping)
                or record.get("path") != relative
                or record.get("n_rows") != cfg.n_samples
                or not _file_record_matches(
                    out_dir / str(relative),
                    record,
                    full_hash=True,
                )
            ):
                raise _artifact_reuse_error(
                    f"aggregate metadata {relative!r} does not match its integrity record"
                )


def _remove_fragment_artifacts(path: Path) -> int:
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_file() or path.is_symlink():
        path.unlink()
        return 1

    removed = 0
    for child in path.iterdir():
        removed += _remove_fragment_artifacts(child)
    path.rmdir()
    return removed


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
    unlink_file(_generation_plan_path(samples_dir))
    unlink_file(
        _generation_plan_path(samples_dir).with_name(
            f"{_generation_plan_path(samples_dir).name}.tmp"
        )
    )

    for path in samples_dir.glob("shard_*"):
        unlink_file(path)
    removed += _remove_fragment_artifacts(_fragment_dir(samples_dir))
    removed += _remove_fragment_artifacts(samples_dir / _FINAL_RECEIPT_DIRNAME)
    removed += _remove_fragment_artifacts(samples_dir / "ranks")
    metadata_dir = out_dir / "metadata"
    if metadata_dir.exists():
        for path in metadata_dir.glob("samples.*"):
            unlink_file(path)
    if removed:
        print(f"[cleanup] removed {removed} generated artifact(s) from an incomplete or forced run")


def _reuse_completed_dataset(done: Path, cfg: Config) -> dict:
    manifest = _load_json_object(done, "DONE.json")
    recorded_manifest_sha = manifest.get("done_manifest_sha256")
    if recorded_manifest_sha is not None:
        content = dict(manifest)
        content.pop("done_manifest_sha256", None)
        if recorded_manifest_sha != _canonical_json_sha256(content):
            raise _artifact_reuse_error(
                "DONE.json content does not match its recorded fingerprint"
            )
    elif manifest.get("generation_plan_sha256") is not None:
        raise _artifact_reuse_error(
            "DONE.json is missing its content fingerprint"
        )
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
    """Generate and finalize a dataset on one node with durable chunk resume."""

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


def _normalized_empirical_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_empirical_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "path"
        }
    if isinstance(value, (list, tuple)):
        return [_normalized_empirical_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_compatible_metadata(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parsed_empirical_map_state(
    empirical: EmpiricalMapStore,
) -> dict[str, object]:
    digest = hashlib.sha256()
    counts: dict[str, int] = {}
    for kind, maps in (
        ("recombination", empirical.rec_maps),
        ("mutation", empirical.mut_maps),
    ):
        counts[kind] = len(maps)
        for chrom, rate_map in sorted(maps.items()):
            semantics = {
                "kind": kind,
                "chrom": str(chrom),
                "map_chrom": str(rate_map.chrom),
                "rate_unit": rate_map.rate_unit,
                "rate_unit_before": rate_map.rate_unit_before,
                "rate_unit_after": rate_map.rate_unit_after,
                "mean_rate_before_scaling": float(
                    rate_map.mean_rate_before_scaling
                ),
                "mean_rate_after_scaling": float(
                    rate_map.mean_rate_after_scaling
                ),
                "scaling_method": rate_map.scaling_method,
                "source_format": rate_map.source_format,
            }
            encoded = json.dumps(
                semantics,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            for array in (rate_map.position, rate_map.rate):
                original = np.asarray(array)
                dtype_text = original.dtype.str.encode("ascii")
                digest.update(len(dtype_text).to_bytes(2, "little"))
                digest.update(dtype_text)
                values = np.asarray(original, dtype="<f8")
                digest.update(values.ndim.to_bytes(1, "little"))
                for dimension in values.shape:
                    digest.update(int(dimension).to_bytes(8, "little"))
                digest.update(values.tobytes(order="C"))
    return {
        "parsed_maps_sha256": digest.hexdigest(),
        "n_recombination_chromosomes": counts["recombination"],
        "n_mutation_chromosomes": counts["mutation"],
    }


def _fragment_layout(cfg: Config) -> dict[str, object]:
    shard_jobs = _shard_tasks(cfg.n_samples, cfg.shard_size)
    compute_jobs = _compute_tasks(
        cfg.n_samples,
        cfg.shard_size,
        cfg.compute_chunk_size,
    )
    return {
        "n_samples": int(cfg.n_samples),
        "shard_size": int(cfg.shard_size),
        "compute_chunk_size": int(cfg.compute_chunk_size),
        "n_shards": len(shard_jobs),
        "n_chunks": len(compute_jobs),
        "sample_id_rule": "sim_{global_sample_index:08d}",
        "seed_rule": "config.seed + global_sample_index * 7919",
        "partition_rule": "chunk_idx_mod_world_size",
        "array_schema_version": int(SAMPLE_GENERATOR_SCHEMA_VERSION),
    }


def _current_ne_template_provenance(cfg: Config) -> dict[str, object]:
    enabled = cfg.p_empirical_human_ne_template > 0.0
    return {
        "enabled": enabled,
        "templates": (
            empirical_ne_template_provenance(cfg)
            if enabled
            else {
                "n_templates": 0,
                "source_counts": {},
                "parsed_templates_sha256": hashlib.sha256(b"").hexdigest(),
                "source_files": [],
            }
        ),
    }


def _generation_runtime_state() -> dict[str, object]:
    package_versions: dict[str, str | None] = {}
    for package in (
        "msprime",
        "tskit",
        "numpy",
        "pandas",
        "demes",
        "stdpopsim",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    source_root = Path(__file__).resolve().parent
    source_digest = hashlib.sha256()
    source_files: list[str] = []
    for path in sorted(source_root.glob("*.py")):
        before = path.stat()
        relative = path.relative_to(source_root.parent).as_posix()
        encoded = relative.encode("utf-8")
        source_digest.update(len(encoded).to_bytes(4, "little"))
        source_digest.update(encoded)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                source_digest.update(chunk)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(
                f"Generator source changed while recording provenance: {path}"
            )
        source_files.append(relative)

    repo_root = source_root.parent
    git_commit: str | None = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        candidate = completed.stdout.strip()
        if len(candidate) == 40:
            git_commit = candidate
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "packages": package_versions,
        "generator_source_sha256": source_digest.hexdigest(),
        "generator_source_files": source_files,
        "git_commit": git_commit,
    }


def _runtime_fingerprint_payload(
    runtime_state: Mapping[str, object],
) -> dict[str, object]:
    # The commit is valuable provenance, while the exact source-tree digest is
    # the scientific identity. Committing unchanged dirty code should not make
    # a resumable campaign incompatible.
    return {
        str(key): value
        for key, value in runtime_state.items()
        if key != "git_commit"
    }


def _build_generation_plan(cfg: Config, empirical: EmpiricalMapStore) -> dict:
    empirical_metadata = _json_compatible_metadata(empirical.metadata)
    empirical_fingerprint_payload = _normalized_empirical_metadata(
        empirical_metadata
    )
    parsed_empirical_maps = _parsed_empirical_map_state(empirical)
    template_provenance = _current_ne_template_provenance(cfg)
    runtime_state = _generation_runtime_state()
    layout = _fragment_layout(cfg)
    plan = {
        "schema_version": _GENERATION_PLAN_SCHEMA_VERSION,
        "kind": "dlcoalsim_generation_plan",
        DONE_CONFIG_STATE_KEY: generation_config_state(cfg),
        "layout": layout,
        "layout_sha256": _canonical_json_sha256(layout),
        "empirical_maps": empirical_metadata,
        "parsed_empirical_maps": parsed_empirical_maps,
        "empirical_maps_sha256": _canonical_json_sha256(
            {
                "metadata": empirical_fingerprint_payload,
                "parsed_maps": parsed_empirical_maps,
            }
        ),
        "empirical_ne_templates": template_provenance,
        "empirical_ne_templates_sha256": _canonical_json_sha256(
            _normalized_empirical_metadata(template_provenance)
        ),
        "runtime_environment": runtime_state,
        "runtime_environment_sha256": _canonical_json_sha256(
            _runtime_fingerprint_payload(runtime_state)
        ),
    }
    plan["plan_sha256"] = _canonical_json_sha256(plan)
    return plan


def _validate_generation_plan(
    plan: Mapping[str, object],
    cfg: Config,
    empirical: EmpiricalMapStore | None,
    *,
    check_runtime: bool | None = None,
) -> dict:
    if (
        plan.get("schema_version") != _GENERATION_PLAN_SCHEMA_VERSION
        or plan.get("kind") != "dlcoalsim_generation_plan"
    ):
        raise _artifact_reuse_error("PLAN.json uses an unsupported schema")
    try:
        validate_done_for_reuse(plan, cfg)
    except DatasetConfigMismatchError as exc:
        raise DatasetConfigMismatchError(
            str(exc).replace("DONE.json", "PLAN.json")
        ) from exc
    expected_layout = _fragment_layout(cfg)
    if (
        plan.get("layout") != expected_layout
        or plan.get("layout_sha256") != _canonical_json_sha256(expected_layout)
    ):
        raise _artifact_reuse_error(
            "PLAN.json has a different fragment layout; compute_chunk_size, "
            "shard_size, or n_samples changed"
        )
    persisted = dict(plan)
    recorded_plan_sha = persisted.pop("plan_sha256", None)
    if (
        not isinstance(recorded_plan_sha, str)
        or recorded_plan_sha != _canonical_json_sha256(persisted)
    ):
        raise _artifact_reuse_error("PLAN.json fingerprint is malformed or inconsistent")
    empirical_metadata = plan.get("empirical_maps")
    parsed_empirical_maps = plan.get("parsed_empirical_maps")
    if (
        not isinstance(empirical_metadata, Mapping)
        or not isinstance(parsed_empirical_maps, Mapping)
        or plan.get("empirical_maps_sha256")
        != _canonical_json_sha256(
            {
                "metadata": _normalized_empirical_metadata(
                    empirical_metadata
                ),
                "parsed_maps": parsed_empirical_maps,
            }
        )
    ):
        raise _artifact_reuse_error("PLAN.json empirical-map provenance is malformed")
    if empirical is not None:
        current = _normalized_empirical_metadata(empirical.metadata)
        current_parsed = _parsed_empirical_map_state(empirical)
        if _canonical_json_sha256(
            {
                "metadata": current,
                "parsed_maps": current_parsed,
            }
        ) != plan.get("empirical_maps_sha256"):
            raise _artifact_reuse_error(
                "the currently loaded empirical maps differ from PLAN.json"
            )
    template_provenance = plan.get("empirical_ne_templates")
    if (
        not isinstance(template_provenance, Mapping)
        or plan.get("empirical_ne_templates_sha256")
        != _canonical_json_sha256(
            _normalized_empirical_metadata(template_provenance)
        )
    ):
        raise _artifact_reuse_error(
            "PLAN.json empirical-Ne-template provenance is malformed"
        )
    if empirical is not None:
        current_templates = _current_ne_template_provenance(cfg)
        if _canonical_json_sha256(
            _normalized_empirical_metadata(current_templates)
        ) != plan.get("empirical_ne_templates_sha256"):
            raise _artifact_reuse_error(
                "the currently loaded empirical Ne templates differ from PLAN.json"
            )
    runtime_state = plan.get("runtime_environment")
    if (
        not isinstance(runtime_state, Mapping)
        or plan.get("runtime_environment_sha256")
        != _canonical_json_sha256(
            _runtime_fingerprint_payload(runtime_state)
        )
    ):
        raise _artifact_reuse_error(
            "PLAN.json runtime-environment provenance is malformed"
        )
    if check_runtime is None:
        check_runtime = empirical is not None
    if (
        check_runtime
        and _canonical_json_sha256(
            _runtime_fingerprint_payload(_generation_runtime_state())
        )
        != plan.get("runtime_environment_sha256")
    ):
        raise _artifact_reuse_error(
            "the current generator code or runtime versions differ from PLAN.json"
        )
    return dict(plan)


def _validate_generation_sources(
    cfg: Config,
    empirical: EmpiricalMapStore,
) -> None:
    out_dir = Path(cfg.out_dir)
    empirical_available = empirical.is_available(cfg.seq_len)
    if cfg.p_empirical_slice > 0.0 and not empirical_available:
        raise DatasetSourceUnavailableError(
            "empirical_map_slice is enabled, but no colocated recombination/mutation "
            "map interval can satisfy seq_len; refusing to disable the requested "
            "empirical source"
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



def _has_legacy_incomplete_artifacts(out_dir: Path, samples_dir: Path) -> bool:
    metadata_dir = out_dir / "metadata"
    return (
        (out_dir / "manifest.json").exists()
        or any(samples_dir.glob("shard_*"))
        or _fragment_dir(samples_dir).exists()
        or (samples_dir / _FINAL_RECEIPT_DIRNAME).exists()
        or (samples_dir / "ranks").exists()
        or (
            metadata_dir.exists()
            and any(metadata_dir.glob("samples.*"))
        )
    )


def _prepare_or_load_plan_locked(
    cfg: Config,
    empirical: EmpiricalMapStore,
    *,
    allow_force: bool,
) -> tuple[str, dict]:
    out_dir = Path(cfg.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    done = samples_dir / "DONE.json"
    if done.exists() and not cfg.force:
        return "complete", _reuse_completed_dataset(done, cfg)

    # Resolve every requested source before invalidating any completed corpus.
    _validate_generation_sources(cfg, empirical)
    # DONE is the only sample-level commit marker. If it is absent, a public
    # root manifest must not continue advertising an older complete corpus.
    (out_dir / "manifest.json").unlink(missing_ok=True)
    (out_dir / "manifest.json.tmp").unlink(missing_ok=True)
    if cfg.force:
        if not allow_force:
            raise ValueError(
                "--force is not allowed in a distributed generation rank; run one "
                "--prepare-only --force command before launching the ranks"
            )
        with _campaign_activity_lock(out_dir, shared=False):
            _prepare_generation_output(out_dir, samples_dir, done)

    plan_path = _generation_plan_path(samples_dir)
    for orphan in samples_dir.glob(f".{plan_path.name}.*.tmp"):
        orphan.unlink(missing_ok=True)
    if plan_path.exists():
        plan = _load_json_object(plan_path, "PLAN.json")
        return "plan", _validate_generation_plan(plan, cfg, empirical)

    # A pre-resume-format interrupted run has no trustworthy config or pair
    # commit markers. Clean it once during migration; all new runs write PLAN
    # before their first fragment and are resumed in place thereafter.
    if _has_legacy_incomplete_artifacts(out_dir, samples_dir):
        _prepare_generation_output(out_dir, samples_dir, done)
        samples_dir.mkdir(parents=True, exist_ok=True)

    plan = _build_generation_plan(cfg, empirical)
    write_json_atomic(plan_path, plan)
    return "plan", plan


def prepare_dataset(
    cfg: Config,
    empirical: EmpiricalMapStore,
) -> dict:
    """Atomically create or validate the immutable generation plan."""

    validate_config(cfg)
    with dataset_output_lock(cfg.out_dir, blocking=True):
        _, payload = _prepare_or_load_plan_locked(
            cfg,
            empirical,
            allow_force=True,
        )
        return payload


def _rank_receipt_path(samples_dir: Path, world_size: int, rank: int) -> Path:
    return (
        samples_dir
        / "ranks"
        / f"world_{world_size:06d}_rank_{rank:06d}.done.json"
    )


def _load_committed_final_result(
    final_task: tuple[int, int, int],
    cfg: Config,
    plan_sha256: str,
    *,
    full_hash: bool,
    structural: bool,
) -> dict | None:
    shard_idx, start, count = final_task
    samples_dir = Path(cfg.out_dir) / "samples"
    npz_path = _shard_npz_path(samples_dir, shard_idx)
    metadata_path = npz_path.with_suffix(".jsonl.gz")
    receipt_path = _final_receipt_path(samples_dir, shard_idx)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_content = dict(receipt) if isinstance(receipt, dict) else {}
        recorded_receipt_sha = receipt_content.pop("receipt_sha256", None)
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != _FINAL_SHARD_RECEIPT_SCHEMA_VERSION
            or receipt.get("plan_sha256") != plan_sha256
            or recorded_receipt_sha != _canonical_json_sha256(receipt_content)
        ):
            return None
        result = receipt.get("result")
        files = receipt.get("files")
        if not isinstance(result, dict) or not isinstance(files, dict):
            return None
        expected_path = (
            Path("samples") / _shard_npz_path(samples_dir, shard_idx).name
        ).as_posix()
        if (
            int(result.get("shard_idx", -1)) != shard_idx
            or int(result.get("start", -1)) != start
            or int(result.get("n_samples", -1)) != count
            or result.get("shard_path") != expected_path
            or not _file_record_matches(
                npz_path,
                files.get("npz"),
                full_hash=full_hash,
            )
            or not _file_record_matches(
                metadata_path,
                files.get("metadata_jsonl_gz"),
                full_hash=full_hash,
            )
        ):
            return None
        if structural:
            _validate_sample_artifact_pair(
                npz_path,
                metadata_path,
                cfg,
                expected_start=start,
                expected_count=count,
            )
        loaded = dict(result)
        loaded["shard_bytes"] = int(
            npz_path.stat().st_size + metadata_path.stat().st_size
        )
        return loaded
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _generate_rank_from_plan(
    cfg: Config,
    empirical: EmpiricalMapStore,
    plan: Mapping[str, object],
    *,
    rank: int,
    world_size: int,
) -> dict:
    with _campaign_activity_lock(cfg.out_dir, shared=True):
        return _generate_rank_while_active(
            cfg,
            empirical,
            plan,
            rank=rank,
            world_size=world_size,
        )


def _generate_rank_while_active(
    cfg: Config,
    empirical: EmpiricalMapStore,
    plan: Mapping[str, object],
    *,
    rank: int,
    world_size: int,
) -> dict:
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or rank < 0
        or rank >= world_size
    ):
        raise ValueError("distributed generation requires 0 <= rank < world_size")

    n = cfg.n_samples
    out_dir = Path(cfg.out_dir)
    samples_dir = out_dir / "samples"
    plan_sha256 = str(plan["plan_sha256"])
    template_plan = plan["empirical_ne_templates"]
    template_parsed_sha256 = (
        str(template_plan["templates"]["parsed_templates_sha256"])
        if template_plan["enabled"]
        else None
    )
    all_jobs = _compute_tasks(n, cfg.shard_size, cfg.compute_chunk_size)
    shard_jobs = _shard_tasks(n, cfg.shard_size)
    committed_shards = {
        final_task[0]
        for final_task in shard_jobs
        if _load_committed_final_result(
            final_task,
            cfg,
            plan_sha256,
            full_hash=False,
            structural=False,
        )
        is not None
    }
    assigned_jobs = [
        task
        for task in all_jobs
        if task[0] % world_size == rank
        and task[1] not in committed_shards
    ]
    committed_results: list[dict] = []
    pending_jobs: list[tuple[int, int, int, int]] = []
    for task in assigned_jobs:
        committed = _load_committed_fragment_result(
            task,
            cfg,
            plan_sha256,
            full_hash=False,
            structural=False,
        )
        if committed is None:
            pending_jobs.append(task)
        else:
            committed["reused"] = True
            committed_results.append(committed)
    resolved_workers = resolve_workers(cfg.workers)
    workers = min(resolved_workers, max(1, len(pending_jobs)))
    print(
        f"[generate] rank={rank}/{world_size} samples={n} "
        f"shard_size={cfg.shard_size} compute_chunk_size={cfg.compute_chunk_size} "
        f"assigned_chunks={len(assigned_jobs)}/{len(all_jobs)} "
        f"pending_chunks={len(pending_jobs)} workers={workers}"
    )
    cfg_dict = dataclasses.asdict(cfg)
    _fragment_dir(samples_dir).mkdir(parents=True, exist_ok=True)

    completed_samples = 0
    completed_fragment_bytes = 0
    generated_chunks = 0
    reused_chunks = 0

    def handle_chunk_result(result: dict, progress: tqdm) -> None:
        nonlocal completed_samples, completed_fragment_bytes
        nonlocal generated_chunks, reused_chunks
        completed_samples += int(result["n_samples"])
        completed_fragment_bytes += int(result["fragment_bytes"])
        avg_bytes = completed_fragment_bytes / max(1, completed_samples)
        est_total_bytes = avg_bytes * n
        if result.get("reused"):
            reused_chunks += 1
        else:
            generated_chunks += 1
        progress.update(int(result["n_samples"]))
        progress.set_postfix(
            {
                "new": generated_chunks,
                "reuse": reused_chunks,
                "disk_GB": f"{completed_fragment_bytes / 1e9:.1f}",
                "est_GB": f"{est_total_bytes / 1e9:.1f}",
                "chunk_s": f"{float(result.get('shard_sec', 0.0)):.0f}",
                "mean_s": f"{float(result.get('mean_sample_sec', 0.0)):.1f}",
                "max_s": f"{float(result.get('max_sample_sec', 0.0)):.1f}",
                "slow": str(result.get("slowest_demography_type", ""))[:18],
                "vars": int(result.get("slowest_n_variants", 0)),
            }
        )

    assigned_samples = sum(task[3] for task in assigned_jobs)
    with tqdm(
        total=assigned_samples,
        desc=f"simulate rank {rank}",
        unit="sample",
    ) as progress:
        for result in committed_results:
            handle_chunk_result(result, progress)
        if workers > 1 and pending_jobs:
            shared_spec, shared_handles = create_shared_empirical_map_store(empirical)
            shared_bytes = sum(shm.size for shm in shared_handles)
            if shared_bytes:
                print(
                    "[maps] shared read-only empirical arrays: "
                    f"{shared_bytes / 1e6:.1f} MB"
                )
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_generation_worker,
                    initargs=(
                        cfg_dict,
                        None,
                        shared_spec,
                        plan_sha256,
                        template_parsed_sha256,
                    ),
                ) as ex:
                    for result in _bounded_process_results(
                        ex,
                        pending_jobs,
                        max_in_flight=min(len(pending_jobs), 4 * workers),
                    ):
                        handle_chunk_result(result, progress)
            finally:
                release_shared_memory(shared_handles, unlink=True)
        elif pending_jobs:
            _init_generation_worker(
                cfg_dict,
                empirical=empirical,
                plan_sha256=plan_sha256,
                empirical_ne_templates_parsed_sha256=(
                    template_parsed_sha256
                ),
            )
            for job in pending_jobs:
                handle_chunk_result(_generate_or_resume_fragment(job), progress)

    rank_result = {
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "rank": rank,
        "world_size": world_size,
        "assigned_chunks": len(assigned_jobs),
        "assigned_samples": int(assigned_samples),
        "generated_chunks": generated_chunks,
        "reused_chunks": reused_chunks,
    }
    write_json_atomic(
        _rank_receipt_path(samples_dir, world_size, rank),
        rank_result,
    )
    return rank_result



def generate_dataset_partition(
    cfg: Config,
    empirical: EmpiricalMapStore,
    *,
    rank: int,
    world_size: int,
) -> dict:
    """Generate one deterministic subset of chunks without a global writer lock."""

    validate_config(cfg)
    if cfg.force:
        raise ValueError(
            "--force cannot be used by generation ranks; prepare the output once "
            "with --prepare-only --force"
        )
    activity = ExitStack()
    try:
        with dataset_output_lock(cfg.out_dir, blocking=True):
            status, payload = _prepare_or_load_plan_locked(
                cfg,
                empirical,
                allow_force=False,
            )
            if status == "plan":
                # Acquire the shared campaign lease before releasing the plan
                # lock. This closes the prepare→generate gap in which a force
                # operation could otherwise replace PLAN beneath this rank.
                activity.enter_context(
                    _campaign_activity_lock(cfg.out_dir, shared=True)
                )
        if status == "complete":
            return {
                "rank": rank,
                "world_size": world_size,
                "complete": True,
                "generated_chunks": 0,
                "reused_chunks": 0,
            }
        return _generate_rank_while_active(
            cfg,
            empirical,
            payload,
            rank=rank,
            world_size=world_size,
        )
    finally:
        activity.close()


def _final_shard_receipt(
    final_task: tuple[int, int, int],
    cfg: Config,
    plan_sha256: str,
    counts: Mapping[str, Mapping[str, int]],
) -> dict:
    shard_idx, start, count = final_task
    samples_dir = Path(cfg.out_dir) / "samples"
    npz_path = _shard_npz_path(samples_dir, shard_idx)
    metadata_path = npz_path.with_suffix(".jsonl.gz")
    result = {
        "shard_idx": int(shard_idx),
        "start": int(start),
        "n_samples": int(count),
        "shard_path": (Path("samples") / npz_path.name).as_posix(),
        "shard_bytes": int(npz_path.stat().st_size + metadata_path.stat().st_size),
        **{
            key: {str(name): int(value) for name, value in values.items()}
            for key, values in counts.items()
        },
    }
    receipt = {
        "schema_version": _FINAL_SHARD_RECEIPT_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "result": result,
        "files": {
            "npz": _file_record(npz_path),
            "metadata_jsonl_gz": _file_record(metadata_path),
        },
    }
    receipt["receipt_sha256"] = _canonical_json_sha256(receipt)
    return receipt


def _discard_final_shard(samples_dir: Path, shard_idx: int) -> None:
    npz_path = _shard_npz_path(samples_dir, shard_idx)
    receipt_path = _final_receipt_path(samples_dir, shard_idx)
    for path in (
        npz_path,
        npz_path.with_suffix(".jsonl.gz"),
        _tmp_npz_path(npz_path),
        _tmp_npz_path(npz_path).with_suffix(".jsonl.gz"),
        receipt_path,
        receipt_path.with_name(f"{receipt_path.name}.tmp"),
    ):
        path.unlink(missing_ok=True)
    for path in receipt_path.parent.glob(f".{receipt_path.name}.*.tmp"):
        path.unlink(missing_ok=True)


def _committed_fragments_for_shard(
    final_task: tuple[int, int, int],
    compute_jobs: list[tuple[int, int, int, int]],
    cfg: Config,
    plan_sha256: str,
) -> tuple[list[dict], list[int]]:
    shard_idx = final_task[0]
    results: list[dict] = []
    missing: list[int] = []
    samples_dir = Path(cfg.out_dir) / "samples"
    for task in compute_jobs:
        if task[1] != shard_idx:
            continue
        result = _load_committed_fragment_result(
            task,
            cfg,
            plan_sha256,
            full_hash=True,
            structural=True,
        )
        if result is None:
            with _fragment_lock(samples_dir, task[0]):
                result = _load_committed_fragment_result(
                    task,
                    cfg,
                    plan_sha256,
                    full_hash=True,
                    structural=True,
                )
                if result is None:
                    _discard_fragment_job(samples_dir, task)
            missing.append(task[0])
        else:
            results.append(result)
    return results, missing


def _merge_committed_shard(
    final_task: tuple[int, int, int],
    fragment_results: list[dict],
    cfg: Config,
    plan_sha256: str,
) -> dict:
    shard_idx, start, count = final_task
    samples_dir = Path(cfg.out_dir) / "samples"
    chunk_ids = sorted(int(result["chunk_idx"]) for result in fragment_results)
    with ExitStack() as locks:
        for chunk_idx in chunk_ids:
            locks.enter_context(_fragment_lock(samples_dir, chunk_idx))
        existing = _load_committed_final_result(
            final_task,
            cfg,
            plan_sha256,
            full_hash=True,
            structural=True,
        )
        if existing is not None:
            return existing
        _discard_final_shard(samples_dir, shard_idx)
        merged = _merge_fragment_group(fragment_results, final_task, cfg)
        npz_path = Path(cfg.out_dir) / str(merged["shard_path"])
        metadata_path = npz_path.with_suffix(".jsonl.gz")
        _fsync_file(npz_path)
        _fsync_file(metadata_path)
        fsync_directory(samples_dir)
        _validate_sample_artifact_pair(
            npz_path,
            metadata_path,
            cfg,
            expected_start=start,
            expected_count=count,
        )
        counts = {
            "source_counts": {},
            "demography_counts": {},
            "generator_recipe_counts": {},
            "map_counts": {},
        }
        for result in fragment_results:
            for key in counts:
                _merge_counts(counts[key], result.get(key, {}))
        receipt = _final_shard_receipt(
            final_task,
            cfg,
            plan_sha256,
            counts,
        )
        write_json_atomic(_final_receipt_path(samples_dir, shard_idx), receipt)
        committed = _load_committed_final_result(
            final_task,
            cfg,
            plan_sha256,
            full_hash=True,
            structural=False,
        )
        if committed is None:
            raise RuntimeError(f"final shard {shard_idx} failed post-write validation")
        for result in fragment_results:
            task = (
                int(result["chunk_idx"]),
                int(result["shard_idx"]),
                int(result["start"]),
                int(result["n_samples"]),
            )
            _discard_fragment_job(samples_dir, task)
        return committed


def _write_metadata_from_shards(
    shard_paths: list[str],
    out_dir: Path,
    cfg: Config,
    *,
    expected_count: int,
) -> dict:
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    csv_path = meta_dir / "samples.csv"
    csv_tmp = csv_path.with_name(f"{csv_path.name}.tmp")
    summary_rows: list[dict] | None = [] if expected_count <= 100_000 else None
    row_count = 0
    with csv_tmp.open("w", encoding="utf-8", newline="") as csv_handle:
        writer: csv.DictWriter | None = None
        for relative_path in shard_paths:
            metadata_path = (out_dir / relative_path).with_suffix(".jsonl.gz")
            with gzip.open(metadata_path, "rt", encoding="utf-8") as source:
                for line in source:
                    row = _metadata_row(json.loads(line))
                    if writer is None:
                        writer = csv.DictWriter(
                            csv_handle,
                            fieldnames=list(row),
                            lineterminator="\n",
                        )
                        writer.writeheader()
                    writer.writerow(row)
                    if summary_rows is not None:
                        summary_rows.append(row)
                    row_count += 1
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
    if row_count != expected_count:
        csv_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"aggregate metadata contains {row_count} rows, expected {expected_count}"
        )
    csv_tmp.replace(csv_path)
    fsync_directory(meta_dir)
    paths = {
        "summary_csv": "metadata/samples.csv",
        "full_jsonl_gz": "metadata/samples.jsonl.gz",
    }

    parquet_path = meta_dir / "samples.parquet"
    parquet_tmp = parquet_path.with_name(f"{parquet_path.name}.tmp")
    parquet_path.unlink(missing_ok=True)
    parquet_tmp.unlink(missing_ok=True)
    if summary_rows is not None:
        try:
            pd.DataFrame(summary_rows).to_parquet(parquet_tmp, index=False)
            _fsync_file(parquet_tmp)
            parquet_tmp.replace(parquet_path)
            fsync_directory(meta_dir)
            paths["summary_parquet"] = "metadata/samples.parquet"
        except Exception:
            parquet_tmp.unlink(missing_ok=True)

    full_path = meta_dir / "samples.jsonl.gz"
    full_tmp = full_path.with_name(f"{full_path.name}.tmp")
    full_count = _stream_gzip_members(
        [
            (out_dir / relative_path).with_suffix(".jsonl.gz")
            for relative_path in shard_paths
        ],
        full_tmp,
        compression_level=cfg.compression_level,
    )
    if full_count != expected_count:
        full_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"full aggregate metadata contains {full_count} rows, "
            f"expected {expected_count}"
        )
    _fsync_file(full_tmp)
    full_tmp.replace(full_path)
    fsync_directory(meta_dir)
    return paths


def _finalize_dataset_locked(cfg: Config) -> dict:
    with _campaign_activity_lock(cfg.out_dir, shared=False):
        return _finalize_dataset_while_quiescent(cfg)


def _finalize_dataset_while_quiescent(cfg: Config) -> dict:
    out_dir = Path(cfg.out_dir)
    samples_dir = out_dir / "samples"
    done = samples_dir / "DONE.json"
    if done.exists():
        return _reuse_completed_dataset(done, cfg)
    plan_path = _generation_plan_path(samples_dir)
    if not plan_path.is_file():
        raise DatasetGenerationIncompleteError(
            f"No generation plan exists at {plan_path}; run generation first."
        )
    plan = _validate_generation_plan(
        _load_json_object(plan_path, "PLAN.json"),
        cfg,
        empirical=None,
        check_runtime=True,
    )
    plan_sha256 = str(plan["plan_sha256"])
    shard_jobs = _shard_tasks(cfg.n_samples, cfg.shard_size)
    compute_jobs = _compute_tasks(
        cfg.n_samples,
        cfg.shard_size,
        cfg.compute_chunk_size,
    )

    ready_fragments: dict[int, list[dict]] = {}
    final_results: dict[int, dict] = {}
    missing_chunks: list[int] = []
    # Preflight the entire corpus before consuming any fragment. This makes an
    # early finalize attempt non-destructive and easy to resume.
    for final_task in shard_jobs:
        shard_idx = final_task[0]
        committed = _load_committed_final_result(
            final_task,
            cfg,
            plan_sha256,
            full_hash=True,
            structural=True,
        )
        if committed is not None:
            final_results[shard_idx] = committed
            continue
        # Final files are never a checkpoint by themselves. The fragment
        # receipts remain intact until the final receipt is committed, so an
        # absent/invalid final receipt is always rebuilt from that trusted
        # chain instead of re-signing merely shape-compatible output.
        _discard_final_shard(samples_dir, shard_idx)
        fragments, missing = _committed_fragments_for_shard(
            final_task,
            compute_jobs,
            cfg,
            plan_sha256,
        )
        ready_fragments[shard_idx] = fragments
        missing_chunks.extend(missing)
    if missing_chunks:
        preview = ", ".join(str(index) for index in missing_chunks[:16])
        suffix = "" if len(missing_chunks) <= 16 else ", ..."
        raise DatasetGenerationIncompleteError(
            f"Cannot finalize: {len(missing_chunks)} committed chunk(s) are missing "
            f"or invalid ({preview}{suffix}). Rerun the generation ranks; valid "
            "fragments and final shards were preserved."
        )

    for final_task in shard_jobs:
        shard_idx = final_task[0]
        if shard_idx not in final_results:
            final_results[shard_idx] = _merge_committed_shard(
                final_task,
                ready_fragments[shard_idx],
                cfg,
                plan_sha256,
            )

    ordered_results = [final_results[index] for index in range(len(shard_jobs))]
    shard_paths = [str(result["shard_path"]) for result in ordered_results]
    completed_bytes = sum(int(result["shard_bytes"]) for result in ordered_results)
    source_counts: dict[str, int] = {}
    demo_counts: dict[str, int] = {}
    generator_recipe_counts: dict[str, int] = {}
    map_counts: dict[str, int] = {}
    for result in ordered_results:
        _merge_counts(source_counts, result.get("source_counts", {}))
        _merge_counts(demo_counts, result.get("demography_counts", {}))
        _merge_counts(
            generator_recipe_counts,
            result.get("generator_recipe_counts", {}),
        )
        _merge_counts(map_counts, result.get("map_counts", {}))

    metadata_paths = _write_metadata_from_shards(
        shard_paths,
        out_dir,
        cfg,
        expected_count=cfg.n_samples,
    )
    metadata_integrity = {
        key: {
            "path": relative,
            "n_rows": int(cfg.n_samples),
            **_file_record(out_dir / relative),
        }
        for key, relative in metadata_paths.items()
    }
    avg_bytes_per_sample = completed_bytes / max(1, cfg.n_samples)
    samples_gb = completed_bytes / 1e9
    avg_mb_per_sample = avg_bytes_per_sample / 1e6
    print(f"[disk] samples={samples_gb:.2f} GB avg={avg_mb_per_sample:.2f} MB/sample")
    manifest = {
        "n_samples": cfg.n_samples,
        "n_shards": len(shard_paths),
        "shards": shard_paths,
        "metadata": metadata_paths,
        "metadata_integrity": metadata_integrity,
        "empirical_maps": plan["empirical_maps"],
        "parsed_empirical_maps": plan["parsed_empirical_maps"],
        "empirical_ne_templates": plan["empirical_ne_templates"],
        "runtime_environment": plan["runtime_environment"],
        "generation_plan_sha256": plan_sha256,
        "fragment_layout": plan["layout"],
        "source_counts": source_counts,
        "demography_counts": demo_counts,
        "generator_recipe_counts": generator_recipe_counts,
        "map_counts": map_counts,
        "samples_bytes": int(completed_bytes),
        "samples_gb": float(samples_gb),
        "avg_bytes_per_sample": float(avg_bytes_per_sample),
        "avg_mb_per_sample": float(avg_mb_per_sample),
        "estimated_total_samples_gb": float(
            (avg_bytes_per_sample * cfg.n_samples) / 1e9
        ),
    }
    manifest = stamp_done_manifest(manifest, cfg)
    manifest["done_manifest_sha256"] = _canonical_json_sha256(manifest)
    _validate_completed_artifacts(manifest, cfg)
    write_json_atomic(done, manifest)
    _remove_fragment_artifacts(_fragment_dir(samples_dir))
    return manifest


def finalize_dataset(
    cfg: Config,
    *,
    _lock_already_held: bool = False,
) -> dict:
    """Merge all committed chunks and publish DONE under one global lock."""

    validate_config(cfg)
    if _lock_already_held:
        return _finalize_dataset_locked(cfg)
    with dataset_output_lock(cfg.out_dir, blocking=True):
        return _finalize_dataset_locked(cfg)


def _generate_dataset_locked(cfg: Config, empirical: EmpiricalMapStore) -> dict:
    status, payload = _prepare_or_load_plan_locked(
        cfg,
        empirical,
        allow_force=True,
    )
    if status == "complete":
        return payload
    _generate_rank_from_plan(
        cfg,
        empirical,
        payload,
        rank=0,
        world_size=1,
    )
    return _finalize_dataset_locked(cfg)
