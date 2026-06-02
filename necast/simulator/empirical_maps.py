from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.io.common import infer_compression

from .assets import asset_path
from .config import MAX_MSPRIME_RATE, Config
from .utils import loguniform


# Gaps in a published map are filled with a constant so the rate map covers the
# whole chromosome, which msprime requires.  A slice must not be taken from that
# fill: the rate there was never measured, and the largest of these gaps are the
# acrocentric short arms and the centromeres, which carry no callable sequence in
# real data either.  Gaps below this are bridged instead of splitting the run --
# at a hundred kilobases every autosome still offers a measured span of at least
# 32 Mb (chr21), and below it chr21 falls to 20 Mb and drops out of any 30 Mb
# slice, taking with it the highest recombination rate in the genome.
MEASURED_GAP_TOLERANCE_BP = 100_000.0


def _merge_runs(runs: list[tuple[float, float]], tolerance: float) -> np.ndarray:
    """Join spans separated by less than ``tolerance``.  Returns an (n, 2) array."""

    if not runs:
        return np.zeros((0, 2), dtype=np.float64)
    ordered = sorted(runs)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start - merged[-1][1] < tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=np.float64)


@dataclass
class PiecewiseRateMap:
    chrom: str
    position: np.ndarray
    rate: np.ndarray
    rate_unit: str = "per_bp_per_generation"
    rate_unit_before: str = "per_bp"
    rate_unit_after: str = "per_bp_per_generation"
    mean_rate_before_scaling: float = 0.0
    mean_rate_after_scaling: float = 0.0
    scaling_method: str = "identity"
    source_format: str = "unknown"
    # (n, 2) spans the source file actually measured, merged across small gaps.
    # Empty means the map makes no claim and its whole extent is used.
    measured_runs: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float64)
    )

    @property
    def length(self) -> float:
        return float(self.position[-1]) if len(self.position) else 0.0


def _validate_piecewise_rate_map(rate_map: PiecewiseRateMap) -> None:
    position = np.asarray(rate_map.position)
    rate = np.asarray(rate_map.rate)
    if position.ndim != 1 or rate.ndim != 1:
        raise ValueError(f"Empirical map on {rate_map.chrom} must use one-dimensional arrays")
    if len(position) != len(rate) + 1 or len(rate) == 0:
        raise ValueError(
            f"Empirical map on {rate_map.chrom} must have one more position than rate"
        )
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rate)):
        raise ValueError(f"Empirical map on {rate_map.chrom} contains non-finite values")
    if np.any(np.diff(position) <= 0):
        raise ValueError(f"Empirical map positions on {rate_map.chrom} must be strictly increasing")


@dataclass
class EmpiricalMapStore:
    rec_maps: dict[str, PiecewiseRateMap] = field(default_factory=dict)
    mut_maps: dict[str, PiecewiseRateMap] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)
    _common_span_cache: dict[str, tuple[float, float]] = field(
        default_factory=dict,
        repr=False,
    )
    _usable_chrom_cache: dict[float, tuple[str, ...]] = field(
        default_factory=dict,
        repr=False,
    )

    def common_chroms(self, min_len: int) -> list[str]:
        cache_key = float(min_len)
        cached = self._usable_chrom_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        chroms = sorted(set(self.rec_maps) & set(self.mut_maps), key=_chrom_sort_key)
        usable: list[str] = []
        for chrom in chroms:
            try:
                self.common_slice_start_bounds(chrom, min_len)
            except ValueError:
                continue
            usable.append(chrom)
        self._usable_chrom_cache[cache_key] = tuple(usable)
        return usable

    def common_slice_start_bounds(self, chrom: str, length: int) -> tuple[float, float]:
        """Return the inclusive start-coordinate range measured by both maps.

        The bound is the longest span both maps measured, not the span they
        cover.  A rate map covers its whole chromosome because msprime requires
        it to, and the difference is filled with a constant that was never
        observed -- on chr21 that fill is the first 14.1 Mb, the acrocentric
        short arm, which is in no map and is not callable in real data either.
        Slicing from it puts a fabricated flat landscape into a fifth of the
        rows drawn from that chromosome.
        """

        if chrom not in self.rec_maps or chrom not in self.mut_maps:
            raise ValueError(f"Chromosome {chrom!r} is not present in both empirical maps")
        if length <= 0:
            raise ValueError("Slice length must be positive")

        cached_span = self._common_span_cache.get(chrom)
        if cached_span is None:
            rec_map = self.rec_maps[chrom]
            mut_map = self.mut_maps[chrom]
            _validate_piecewise_rate_map(rec_map)
            _validate_piecewise_rate_map(mut_map)
            cached_span = self._longest_measured_overlap(rec_map, mut_map)
            self._common_span_cache[chrom] = cached_span
        common_left, common_right = cached_span
        latest_start = common_right - float(length)
        if latest_start < common_left:
            raise ValueError(
                f"Chromosome {chrom!r} has only {common_right - common_left:g} bp "
                f"measured by both empirical maps, shorter than requested slice {length:g} bp"
            )
        return common_left, latest_start

    @staticmethod
    def _longest_measured_overlap(
        rec_map: "PiecewiseRateMap", mut_map: "PiecewiseRateMap"
    ) -> tuple[float, float]:
        """The longest span both maps measured, or their shared extent if neither says."""

        def runs(rate_map: "PiecewiseRateMap") -> np.ndarray:
            declared = np.asarray(rate_map.measured_runs, dtype=np.float64).reshape(-1, 2)
            if declared.size:
                return declared
            return np.asarray(
                [[float(rate_map.position[0]), float(rate_map.position[-1])]],
                dtype=np.float64,
            )

        best = (0.0, 0.0)
        left_runs, right_runs = runs(rec_map), runs(mut_map)
        i = j = 0
        while i < len(left_runs) and j < len(right_runs):
            start = max(float(left_runs[i][0]), float(right_runs[j][0]))
            end = min(float(left_runs[i][1]), float(right_runs[j][1]))
            if end - start > best[1] - best[0]:
                best = (start, end)
            if left_runs[i][1] < right_runs[j][1]:
                i += 1
            else:
                j += 1
        if best[1] <= best[0]:
            raise ValueError(
                f"Chromosome {rec_map.chrom!r} has no span measured by both empirical maps"
            )
        return best

    def is_available(self, min_len: int) -> bool:
        return bool(self.common_chroms(min_len))


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    clean = chrom.lower().removeprefix("chr")
    return (int(clean), clean) if clean.isdigit() else (10_000, clean)


def _canonical_chrom(value) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return f"chr{text}"


def _read_table_maybe_gzip(
    path: str,
    *,
    compression: str | None = None,
) -> pd.DataFrame:
    kwargs = dict(sep=r"\s+", comment="#", header=None, dtype=str)
    if compression is None:
        if path.endswith(".gz"):
            kwargs["compression"] = "gzip"
    elif not isinstance(compression, str) or not compression:
        raise ValueError(f"Unsupported empirical-map compression: {compression}")
    else:
        kwargs["compression"] = None if compression == "none" else compression
    return pd.read_csv(path, **kwargs)


def _inferred_map_compression(path: Path) -> str:
    """Match the legacy filename-based pandas decompression semantics."""

    path_text = str(path)
    if path_text.endswith(".gz"):
        return "gzip"
    inferred = infer_compression(path_text, "infer")
    return str(inferred) if inferred is not None else "none"


ROULETTE_RAW_TO_PER_GEN = 1.015e-7
RECOMB_CM_PER_MB_TO_PER_BP = 1.0e-8
# A published map has a resolution of its own, coarser than the truth.
OBSERVED_CELL_BP = (1_000.0, 20_000.0)
RECOMB_MAP_UNITS = {"per_bp", "cM_per_Mb"}
MUT_MAP_UNITS = {"per_bp", "relative", "roulette_raw"}


def _finite_mean(rates: np.ndarray) -> float:
    finite = np.asarray(rates, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if len(finite) else 0.0


def _scale_rates(
    rates: np.ndarray,
    baseline_rate: float | None,
    map_kind: str,
    input_unit: str,
    scale_to_baseline: bool,
) -> tuple[np.ndarray, dict]:
    if map_kind == "recombination" and input_unit not in RECOMB_MAP_UNITS:
        raise ValueError(f"Unsupported recombination map unit: {input_unit}")
    if map_kind == "mutation" and input_unit not in MUT_MAP_UNITS:
        raise ValueError(f"Unsupported mutation map unit: {input_unit}")

    before_mean = _finite_mean(rates)
    fill = baseline_rate if baseline_rate is not None else 1e-8
    rates = np.nan_to_num(rates.astype(np.float64, copy=False), nan=fill, posinf=fill, neginf=0.0)

    method = "identity"
    if map_kind == "recombination":
        if input_unit == "cM_per_Mb":
            rates = rates * RECOMB_CM_PER_MB_TO_PER_BP
            method = "cM_per_Mb_to_per_bp_per_generation"
    elif input_unit == "roulette_raw":
        rates = rates * ROULETTE_RAW_TO_PER_GEN
        method = "roulette_raw_to_per_bp_per_generation"
        if scale_to_baseline and baseline_rate is not None:
            mean_after_conversion = _finite_mean(rates)
            if mean_after_conversion > 0:
                rates = rates * (float(baseline_rate) / mean_after_conversion)
                method += "+mean_to_baseline"
    elif input_unit == "relative":
        if baseline_rate is None:
            raise ValueError("relative mutation maps require a baseline mutation rate")
        positive_mean = _finite_mean(rates[rates > 0])
        if scale_to_baseline:
            if positive_mean <= 0:
                raise ValueError("relative mutation map has no positive rates to normalize")
            rates = rates * (float(baseline_rate) / positive_mean)
            method = "relative_mean_to_baseline"
        else:
            rates = rates * float(baseline_rate)
            method = "relative_multiplier_to_baseline"

    rates = np.clip(rates, 0.0, None)
    if len(rates) and np.nanmax(rates) > MAX_MSPRIME_RATE:
        raise ValueError(
            f"{map_kind} map rates look too large for msprime per-bp per-generation rates; "
            "set --recomb-map-unit/--mut-map-unit explicitly so rates can be converted"
        )
    after_mean = _finite_mean(rates)
    info = {
        "rate_unit_before": input_unit,
        "rate_unit_after": "per_bp_per_generation",
        "mean_rate_before_scaling": before_mean,
        "mean_rate_after_scaling": after_mean,
        "scaling_method": method,
    }
    return rates, info


def _piecewise_from_points(
    chrom: str,
    pos: np.ndarray,
    rate_points: np.ndarray,
    baseline_rate: float | None,
    map_kind: str,
    input_unit: str,
    scale_to_baseline: bool,
) -> PiecewiseRateMap:
    pos = np.asarray(pos)
    rate_points = np.asarray(rate_points)
    if pos.ndim != 1 or rate_points.ndim != 1 or len(pos) != len(rate_points):
        raise ValueError(f"Point map on {chrom} has incompatible position/rate arrays")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rate_points)):
        raise ValueError(f"Point map on {chrom} contains non-finite values")
    order = np.argsort(pos)
    pos = pos[order].astype(np.float64, copy=False)
    rate_points = rate_points[order].astype(np.float64, copy=False)
    keep = np.concatenate([[True], np.diff(pos) > 0])
    pos = pos[keep]
    rate_points = rate_points[keep]
    if len(pos) == 0:
        raise ValueError(f"No numeric positions for empirical map chromosome {chrom}")
    if len(pos) > 1:
        left_step = pos[1] - pos[0]
        right_step = pos[-1] - pos[-2]
        boundaries = np.concatenate(
            [
                [max(0.0, pos[0] - left_step / 2.0)],
                (pos[:-1] + pos[1:]) / 2.0,
                [pos[-1] + max(1.0, right_step / 2.0)],
            ]
        )
    else:
        step = 1_000.0
        boundaries = np.array([max(0.0, pos[0] - step / 2.0), pos[0] + step / 2.0], dtype=np.float64)
    rates, scaling = _scale_rates(rate_points, baseline_rate, map_kind, input_unit, scale_to_baseline)
    # A point map measures at its centres; a wide space between two of them is
    # interpolation, not measurement, so it splits the measured run exactly as a
    # filled interval gap does.
    measured = [
        (float(boundaries[i]), float(boundaries[i + 1])) for i in range(len(rates))
    ]
    return PiecewiseRateMap(
        chrom=chrom,
        position=boundaries.astype(np.float64),
        rate=rates.astype(np.float64),
        rate_unit=scaling["rate_unit_after"],
        **scaling,
        source_format="point_centers",
        measured_runs=_merge_runs(measured, MEASURED_GAP_TOLERANCE_BP),
    )


def _piecewise_from_intervals(
    chrom: str,
    starts: np.ndarray,
    ends: np.ndarray,
    rates: np.ndarray,
    baseline_rate: float | None,
    map_kind: str,
    input_unit: str,
    scale_to_baseline: bool,
) -> PiecewiseRateMap:
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    rates = np.asarray(rates)
    if (
        starts.ndim != 1
        or ends.ndim != 1
        or rates.ndim != 1
        or len(starts) != len(ends)
        or len(starts) != len(rates)
    ):
        raise ValueError(f"Interval map on {chrom} has incompatible start/end/rate arrays")
    if (
        not np.all(np.isfinite(starts))
        or not np.all(np.isfinite(ends))
        or not np.all(np.isfinite(rates))
    ):
        raise ValueError(f"Interval map on {chrom} contains non-finite values")
    order = np.argsort(starts)
    starts = starts[order].astype(np.float64, copy=False)
    ends = ends[order].astype(np.float64, copy=False)
    rates = rates[order].astype(np.float64, copy=False)
    keep = ends > starts
    starts = starts[keep]
    ends = ends[keep]
    rates = rates[keep]
    if len(starts) == 0:
        raise ValueError(f"No valid intervals for empirical map chromosome {chrom}")

    rates, scaling = _scale_rates(rates, baseline_rate, map_kind, input_unit, scale_to_baseline)
    filler = baseline_rate if baseline_rate is not None else float(np.nanmedian(rates))
    positions = [0.0]
    out_rates: list[float] = []
    measured: list[tuple[float, float]] = []
    current = 0.0
    for start, end, rate in zip(starts, ends, rates):
        start = float(start)
        end = float(end)
        if end <= current:
            continue
        if start > current:
            out_rates.append(float(filler))
            positions.append(start)
            current = start
        if end > current:
            out_rates.append(float(rate))
            positions.append(end)
            measured.append((start if start > current else current, end))
            current = end

    positions_arr = np.asarray(positions, dtype=np.float64)
    rates_arr = np.asarray(out_rates, dtype=np.float64)
    keep = np.concatenate([[True], np.diff(positions_arr) > 0])
    positions_arr = positions_arr[keep]
    if len(rates_arr) != len(positions_arr) - 1:
        rates_arr = rates_arr[: len(positions_arr) - 1]
    return PiecewiseRateMap(
        chrom=chrom,
        position=positions_arr,
        rate=rates_arr.astype(np.float64),
        rate_unit=scaling["rate_unit_after"],
        **scaling,
        source_format="bed_intervals",
        measured_runs=_merge_runs(measured, MEASURED_GAP_TOLERANCE_BP),
    )


def load_empirical_rate_table(
    path: str,
    baseline_rate: float | None = None,
    map_kind: str = "generic",
    rate_unit: str = "per_bp",
    scale_to_baseline: bool = True,
    *,
    compression: str | None = None,
) -> dict[str, PiecewiseRateMap]:
    """Load BED-like or position-rate maps, preserving chromosome boundaries.

    Canonical input is a BED-like interval table with columns:
    chrom, start, end, rate. Coordinates are 0-based half-open intervals and
    rate is per bp per generation unless rate_unit explicitly says otherwise.

    Legacy point-center maps are accepted and converted to interval boundaries
    with midpoint cuts. Missing interval gaps are filled with baseline_rate so
    the returned map can be passed to msprime after slicing.
    """
    df = _read_table_maybe_gzip(path, compression=compression)
    maps: dict[str, PiecewiseRateMap] = {}

    if df.shape[1] >= 4:
        chrom_col = df.iloc[:, 0].map(_canonical_chrom)
        starts = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        ends = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        rates = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        parsed = pd.DataFrame({"chrom": chrom_col, "start": starts, "end": ends, "rate": rates}).dropna()
        for chrom, sub in parsed.groupby("chrom", sort=False):
            maps[str(chrom)] = _piecewise_from_intervals(
                str(chrom),
                sub["start"].to_numpy(),
                sub["end"].to_numpy(),
                sub["rate"].to_numpy(),
                baseline_rate,
                map_kind,
                rate_unit,
                scale_to_baseline,
            )
    elif df.shape[1] == 3:
        chrom_col = df.iloc[:, 0].map(_canonical_chrom)
        pos = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        rates = pd.to_numeric(df.iloc[:, 2], errors="coerce")
        parsed = pd.DataFrame({"chrom": chrom_col, "pos": pos, "rate": rates}).dropna()
        for chrom, sub in parsed.groupby("chrom", sort=False):
            maps[str(chrom)] = _piecewise_from_points(
                str(chrom),
                sub["pos"].to_numpy(),
                sub["rate"].to_numpy(),
                baseline_rate,
                map_kind,
                rate_unit,
                scale_to_baseline,
            )
    elif df.shape[1] == 2:
        pos = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        rates = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        parsed = pd.DataFrame({"pos": pos, "rate": rates}).dropna()
        maps["__single__"] = _piecewise_from_points(
            "__single__",
            parsed["pos"].to_numpy(),
            parsed["rate"].to_numpy(),
            baseline_rate,
            map_kind,
            rate_unit,
            scale_to_baseline,
        )
    else:
        raise ValueError(f"Cannot parse empirical map: {path}")

    if not maps:
        raise ValueError(f"Empirical map has no usable numeric rows: {path}")
    return maps


def _load_empirical_rate_table_cached(
    path: str | Path,
    baseline_rate: float | None,
    map_kind: str,
    rate_unit: str,
    scale_to_baseline: bool,
) -> dict[str, PiecewiseRateMap]:
    from . import config as config_module
    from .map_cache import load_or_build_rate_maps

    parser_path = Path(path).expanduser()
    if not parser_path.is_absolute():
        parser_path = Path.cwd() / parser_path
    resolved_source = parser_path.resolve(strict=True)
    compression = _inferred_map_compression(parser_path)
    parse_parameters = {
        "parser": "load_empirical_rate_table",
        "compression": compression,
        "baseline_rate": baseline_rate,
        "map_kind": map_kind,
        "rate_unit": rate_unit,
        "scale_to_baseline": scale_to_baseline,
    }
    return load_or_build_rate_maps(
        source_path=resolved_source,
        parse_parameters=parse_parameters,
        related_code_paths={
            "empirical_maps_parser": Path(__file__),
            "rate_validation_config": Path(config_module.__file__),
        },
        builder=lambda snapshot: load_empirical_rate_table(
            str(snapshot),
            baseline_rate,
            map_kind=map_kind,
            rate_unit=rate_unit,
            scale_to_baseline=scale_to_baseline,
            compression=compression,
        ),
        map_class=PiecewiseRateMap,
    )


def _source_file_provenance(path: str | Path) -> dict[str, int | str]:
    source = Path(path).expanduser().resolve(strict=True)
    before = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = source.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ValueError(f"Empirical map changed while recording provenance: {source}")
    return {
        "source_size_bytes": int(after.st_size),
        "source_sha256": digest.hexdigest(),
    }


def _summarize_maps(path: str, maps: dict[str, PiecewiseRateMap], requested_unit: str) -> dict:
    methods = sorted({m.scaling_method for m in maps.values()})
    formats = sorted({m.source_format for m in maps.values()})
    before = [m.mean_rate_before_scaling for m in maps.values()]
    after = [m.mean_rate_after_scaling for m in maps.values()]
    return {
        "path": path,
        **_source_file_provenance(path),
        "requested_rate_unit": requested_unit,
        "rate_unit_before": requested_unit,
        "rate_unit_after": "per_bp_per_generation",
        "mean_rate_before_scaling": float(np.mean(before)) if before else 0.0,
        "mean_rate_after_scaling": float(np.mean(after)) if after else 0.0,
        "scaling_method": "+".join(methods) if methods else "unknown",
        "source_formats": formats,
        "n_chromosomes": len(maps),
        "chromosomes": sorted(maps, key=_chrom_sort_key),
    }


def _resolve_map_path(path: str) -> Path | None:
    if not path:
        return None
    resolved = asset_path(path)
    return resolved if resolved.exists() else None


def load_empirical_maps(cfg: Config) -> EmpiricalMapStore:
    store = EmpiricalMapStore()
    recomb_path = _resolve_map_path(cfg.recomb_map)
    mut_path = _resolve_map_path(cfg.mut_map)
    if recomb_path is not None:
        print(f"[maps] loading empirical recombination map: {recomb_path}")
        store.rec_maps = _load_empirical_rate_table_cached(
            recomb_path,
            cfg.baseline_rec,
            map_kind="recombination",
            rate_unit=cfg.recomb_map_unit,
            scale_to_baseline=False,
        )
        store.metadata["recombination"] = _summarize_maps(str(recomb_path), store.rec_maps, cfg.recomb_map_unit)
    if mut_path is not None:
        print(f"[maps] loading empirical mutation map: {mut_path}")
        store.mut_maps = _load_empirical_rate_table_cached(
            mut_path,
            cfg.baseline_mu,
            map_kind="mutation",
            rate_unit=cfg.mut_map_unit,
            scale_to_baseline=cfg.mut_map_scale_to_baseline,
        )
        store.metadata["mutation"] = _summarize_maps(str(mut_path), store.mut_maps, cfg.mut_map_unit)
    return store


def slice_piecewise_map(
    rng: np.random.Generator,
    rate_map: PiecewiseRateMap,
    length: int,
    obs_noise_sigma: float,
    baseline_rate: float,
    start: float | None = None,
    *,
    validate_input: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Slice a rate map, optionally at an explicit source-map coordinate.

    Omitting ``start`` preserves the original random-slice API. Passing it lets
    callers use the exact same genomic interval for multiple empirical maps.
    """
    if validate_input:
        _validate_piecewise_rate_map(rate_map)
    global_pos = rate_map.position
    global_rate = rate_map.rate
    if length <= 0:
        raise ValueError("Slice length must be positive")

    min_start = float(global_pos[0])
    max_start = float(global_pos[-1]) - float(length)
    if max_start < min_start:
        raise ValueError(
            f"Empirical map on {rate_map.chrom} spans only "
            f"{float(global_pos[-1]) - min_start:g} bp, shorter than requested slice {length:g} bp"
        )
    if start is None:
        start = min_start if max_start == min_start else float(rng.uniform(min_start, max_start))
    else:
        start = float(start)
        if not np.isfinite(start):
            raise ValueError("Slice start must be finite")
        if start < min_start or start > max_start:
            raise ValueError(
                f"Slice start {start:g} is outside the valid range "
                f"[{min_start:g}, {max_start:g}] for {rate_map.chrom}"
            )

    end = start + length

    internal_boundaries = global_pos[(global_pos > start) & (global_pos < end)]
    absolute_interval_starts = np.concatenate(
        [np.array([start], dtype=np.float64), internal_boundaries.astype(np.float64, copy=False)]
    )
    rate_indices = np.searchsorted(global_pos, absolute_interval_starts, side="right") - 1
    rate_indices = np.clip(rate_indices, 0, len(global_rate) - 1)
    rate = global_rate[rate_indices].copy()
    pos = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            internal_boundaries.astype(np.float64, copy=False) - start,
            np.array([float(length)], dtype=np.float64),
        ]
    )

    true_rate = np.clip(rate, baseline_rate * 1e-4, baseline_rate * 500)

    # The observed map is what the model is handed and is expected to trust.
    # It is re-binned onto a grid of its own before the noise is applied, for
    # two reasons.  Returning it on the true breakpoints handed over every
    # boundary of the true map exactly, and noised only the intensities, when
    # a published map is uncertain about where the hotspots are as much as
    # about how strong they are.  And a real map reports the average rate in
    # its own bins, which also smooths a hotspot it did not resolve.
    obs_cell = float(loguniform(rng, *OBSERVED_CELL_BP))
    obs_pos = np.arange(0.0, float(length) + obs_cell, obs_cell)
    if len(obs_pos) < 2:
        obs_pos = np.array([0.0, float(length)], dtype=np.float64)
    obs_pos[-1] = float(length)
    cumulative = np.concatenate(([0.0], np.cumsum(true_rate * np.diff(pos))))
    obs_rate = np.diff(np.interp(obs_pos, pos, cumulative)) / np.diff(obs_pos)

    # Centred so the multiplier has mean one.  With a zero location a lognormal
    # has mean exp(sigma^2 / 2), so the observed map integral sat above the
    # truth in every row, and nw_log10 divides the segregating count by exactly
    # that integral.
    from .maps import observed_noise_multiplier

    obs_rate = obs_rate * observed_noise_multiplier(
        rng, obs_rate.shape[0], obs_cell, obs_noise_sigma, true_rates=obs_rate
    )
    obs_rate = np.clip(obs_rate, baseline_rate * 1e-5, baseline_rate * 1000)
    meta = {
        "chrom": rate_map.chrom,
        "slice_start_bp": start,
        "slice_end_bp": end,
        "rate_unit": rate_map.rate_unit,
        "rate_unit_before": rate_map.rate_unit_before,
        "rate_unit_after": rate_map.rate_unit_after,
        "mean_rate_before_scaling": rate_map.mean_rate_before_scaling,
        "mean_rate_after_scaling": rate_map.mean_rate_after_scaling,
        "scaling_method": rate_map.scaling_method,
        "source_format": rate_map.source_format,
        "observed_cell_bp": obs_cell,
        "observed_noise_sigma": float(obs_noise_sigma),
    }
    return (
        pos.astype(np.float64),
        true_rate.astype(np.float64),
        obs_pos.astype(np.float64),
        obs_rate.astype(np.float64),
        meta,
    )
