from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np

from .config import Config
from .slurm import allocated_cpu_count


def resolve_workers(
    workers: int,
    *,
    environ: Mapping[str, str] | None = None,
    cpu_count: int | None = None,
) -> int:
    if workers == -1:
        return max(
            1,
            allocated_cpu_count(environ=environ, override=cpu_count) - 1,
        )
    return max(1, workers)


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) % (2**32 - 1))


def time_grid(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    return log_time_grid(cfg.min_time, cfg.max_time, cfg.time_bins)


def log_time_grid(
    min_time: float,
    max_time: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.geomspace(min_time, max_time, n_bins + 1).astype(np.float64)
    mids = np.sqrt(edges[:-1] * edges[1:]).astype(np.float64)
    return edges, mids


def _epoch_growth_rates(values: Sequence[float], growth_rates: Sequence[float] | None) -> np.ndarray:
    if growth_rates is None:
        return np.zeros(len(values), dtype=np.float64)
    rates = np.asarray(growth_rates, dtype=np.float64)
    if rates.shape != (len(values),):
        raise ValueError("growth_rates must contain one rate per demographic epoch")
    if not np.all(np.isfinite(rates)):
        raise ValueError("growth_rates must be finite")
    return rates


def piecewise_eval(
    times: np.ndarray,
    breaks: Sequence[float],
    values: Sequence[float],
    growth_rates: Sequence[float] | None = None,
) -> np.ndarray:
    """Evaluate a piecewise-exponential population-size history.

    ``values[i]`` is the population size at the start of epoch ``i`` and
    ``growth_rates[i]`` follows the msprime backward-time convention:

    ``N(t) = values[i] * exp(-growth_rates[i] * (t - epoch_start[i]))``.

    Omitting ``growth_rates`` preserves the previous piecewise-constant
    behaviour.
    """
    times = np.asarray(times, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 1 or values_arr.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    breaks_arr = np.asarray(breaks, dtype=np.float64)
    if breaks_arr.shape != (values_arr.size - 1,):
        raise ValueError("values must contain exactly one more item than breaks")
    if np.any(np.diff(breaks_arr) <= 0):
        raise ValueError("breaks must be strictly increasing")
    rates = _epoch_growth_rates(values_arr, growth_rates)

    epoch = np.searchsorted(breaks_arr, times, side="right")
    starts = np.concatenate(([0.0], breaks_arr))
    elapsed = times - starts[epoch]
    return values_arr[epoch] * np.exp(-rates[epoch] * elapsed)


def bin_average_log10_ne(
    edges: np.ndarray,
    breaks: Sequence[float],
    values: Sequence[float],
    growth_rates: Sequence[float] | None = None,
) -> np.ndarray:
    """Average piecewise-exponential log10 Ne over each logarithmic time bin.

    The bins are logarithmic, so the average is taken uniformly in log time.
    Breakpoints inside a bin are included explicitly, which prevents short
    bottlenecks from disappearing just because they miss a bin midpoint. Within
    a native exponential epoch, the integral is evaluated analytically rather
    than sampled numerically.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("time edges must be positive and strictly increasing")

    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim != 1 or values_arr.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    rates_arr = _epoch_growth_rates(values_arr, growth_rates)
    if len(breaks) != values_arr.size - 1:
        raise ValueError("values must contain exactly one more item than breaks")
    pairs = sorted(
        (float(b), float(v), float(r))
        for b, v, r in zip(breaks, values_arr[1:], rates_arr[1:])
        if np.isfinite(b) and b > 0 and np.isfinite(v)
    )
    cleaned_breaks = [b for b, _, _ in pairs]
    cleaned_values = np.asarray([float(values_arr[0]), *(v for _, v, _ in pairs)], dtype=np.float64)
    cleaned_rates = np.asarray([float(rates_arr[0]), *(r for _, _, r in pairs)], dtype=np.float64)
    epoch_starts = np.asarray([0.0, *cleaned_breaks], dtype=np.float64)
    out: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        cuts = [float(left)]
        cuts.extend(b for b in cleaned_breaks if left < b < right)
        cuts.append(float(right))

        total = 0.0
        weight_sum = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b <= a:
                continue
            log_width = float(np.log(b) - np.log(a))
            if log_width <= 0.0:
                # ``b > a`` is not enough: two breaks close enough together in
                # relative terms make log(b) - log(a) underflow to exactly zero,
                # and the sub-interval then carries no weight anyway.  Dense
                # epochs make this reachable -- a piecewise-exponential curve
                # with a few hundred knots hits it -- where a handful of epochs
                # never would.
                continue
            probe = float(np.sqrt(a * b))
            epoch = int(np.searchsorted(cleaned_breaks, probe, side="right"))
            mean_elapsed = float((b - a) / log_width - epoch_starts[epoch])
            mean_log10_ne = float(
                np.log10(max(cleaned_values[epoch], 1.0))
                - cleaned_rates[epoch] * mean_elapsed / np.log(10.0)
            )
            weight = float(np.log(b) - np.log(a))
            total += weight * mean_log10_ne
            weight_sum += weight
        out.append(total / weight_sum if weight_sum else float("nan"))
    return np.asarray(out, dtype=np.float32)


def loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def safe_json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
