from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .utils import loguniform


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

    @property
    def length(self) -> float:
        return float(self.position[-1]) if len(self.position) else 0.0


@dataclass
class EmpiricalMapStore:
    rec_maps: dict[str, PiecewiseRateMap] = field(default_factory=dict)
    mut_maps: dict[str, PiecewiseRateMap] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)

    def common_chroms(self, min_len: int) -> list[str]:
        chroms = sorted(set(self.rec_maps) & set(self.mut_maps), key=_chrom_sort_key)
        return [c for c in chroms if self.rec_maps[c].length >= min_len and self.mut_maps[c].length >= min_len]

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


def _read_table_maybe_gzip(path: str) -> pd.DataFrame:
    kwargs = dict(sep=r"\s+", comment="#", header=None, dtype=str)
    if path.endswith(".gz"):
        kwargs["compression"] = "gzip"
    return pd.read_csv(path, **kwargs)


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
    if len(rates) and np.nanmax(rates) > 1e-4:
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
    return PiecewiseRateMap(
        chrom=chrom,
        position=boundaries.astype(np.float64),
        rate=rates.astype(np.float64),
        rate_unit=scaling["rate_unit_after"],
        **scaling,
        source_format="point_centers",
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
    )


def load_empirical_rate_table(
    path: str,
    baseline_rate: float | None = None,
    map_kind: str = "generic",
    rate_unit: str = "per_bp",
    scale_to_baseline: bool = True,
) -> dict[str, PiecewiseRateMap]:
    """Load BED-like or position-rate maps, preserving chromosome boundaries.

    Canonical input is a BED-like interval table with columns:
    chrom, start, end, rate. Coordinates are 0-based half-open intervals and
    rate is per bp per generation unless rate_unit explicitly says otherwise.

    Legacy point-center maps are accepted and converted to interval boundaries
    with midpoint cuts. Missing interval gaps are filled with baseline_rate so
    the returned map can be passed to msprime after slicing.
    """
    df = _read_table_maybe_gzip(path)
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


def _summarize_maps(path: str, maps: dict[str, PiecewiseRateMap], requested_unit: str) -> dict:
    methods = sorted({m.scaling_method for m in maps.values()})
    formats = sorted({m.source_format for m in maps.values()})
    before = [m.mean_rate_before_scaling for m in maps.values()]
    after = [m.mean_rate_after_scaling for m in maps.values()]
    return {
        "path": path,
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
    raw = Path(path).expanduser()
    if raw.exists():
        return raw
    if raw.is_absolute():
        return None
    repo_root = Path(__file__).resolve().parents[1]
    repo_path = repo_root / raw
    return repo_path if repo_path.exists() else None


def load_empirical_maps(cfg: Config) -> EmpiricalMapStore:
    store = EmpiricalMapStore()
    recomb_path = _resolve_map_path(cfg.recomb_map)
    mut_path = _resolve_map_path(cfg.mut_map)
    if recomb_path is not None:
        print(f"[maps] loading empirical recombination map: {recomb_path}")
        store.rec_maps = load_empirical_rate_table(
            str(recomb_path),
            cfg.baseline_rec,
            map_kind="recombination",
            rate_unit=cfg.recomb_map_unit,
            scale_to_baseline=False,
        )
        store.metadata["recombination"] = _summarize_maps(str(recomb_path), store.rec_maps, cfg.recomb_map_unit)
    if mut_path is not None:
        print(f"[maps] loading empirical mutation map: {mut_path}")
        store.mut_maps = load_empirical_rate_table(
            str(mut_path),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    global_pos = rate_map.position
    global_rate = rate_map.rate
    max_start = max(1.0, global_pos[-1] - length - 1)
    start = float(rng.uniform(0, max_start))
    end = start + length
    idx_start = max(0, np.searchsorted(global_pos, start, side="right") - 1)
    idx_end = min(len(global_rate), np.searchsorted(global_pos, end, side="right"))
    pos = global_pos[idx_start : idx_end + 1].copy()
    rate = global_rate[idx_start:idx_end].copy()

    if len(rate) == 0:
        pos = np.array([0.0, float(length)], dtype=np.float64)
        rate = np.array([baseline_rate], dtype=np.float64)
    else:
        pos = pos - start
        pos[0] = 0.0
        if pos[-1] < length:
            pos = np.append(pos, float(length))
            rate = np.append(rate, rate[-1])
        else:
            pos[-1] = float(length)
        if len(pos) != len(rate) + 1:
            pos = pos[: len(rate) + 1]
            pos[-1] = float(length)

    true_rate = np.clip(rate, baseline_rate * 1e-4, baseline_rate * 500)

    # The observed map is what the model is handed and is expected to trust.
    # It is re-binned onto a grid of its own before the noise is applied, for
    # two reasons.  Returning it on the true breakpoints handed over every
    # boundary of the real map exactly -- every hotspot edge included -- and
    # noised only the intensities, when a published map is uncertain about
    # where the hotspots are as much as about how strong they are.  And a real
    # map reports the average rate in its own bins, which also smooths a
    # hotspot it did not resolve.
    obs_cell = float(loguniform(rng, *OBSERVED_CELL_BP))
    obs_pos = np.arange(0.0, float(length) + obs_cell, obs_cell)
    if len(obs_pos) < 2:
        obs_pos = np.array([0.0, float(length)], dtype=np.float64)
    obs_pos[-1] = float(length)
    cumulative = np.concatenate(([0.0], np.cumsum(true_rate * np.diff(pos))))
    integrated = np.interp(obs_pos, pos, cumulative)
    obs_rate = np.diff(integrated) / np.diff(obs_pos)

    # Centred so the multiplier has mean one.  With a zero location a lognormal
    # has mean exp(sigma^2 / 2), so the observed map integral sat above the
    # truth in every row, and nw_log10 divides the segregating count by exactly
    # that integral.
    obs_rate = obs_rate * rng.lognormal(
        -0.5 * obs_noise_sigma**2, obs_noise_sigma, size=obs_rate.shape
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
