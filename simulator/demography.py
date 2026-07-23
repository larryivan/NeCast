from __future__ import annotations

from collections.abc import Callable
from typing import Sequence

import msprime
import numpy as np

from .config import (
    DEMOGRAPHY_TYPE_IDS,
    GENERATOR_RECIPE_IDS,
    GENERATOR_RECIPE_HISTORY_FAMILY_OVERRIDES,
    GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE,
    HISTORY_FAMILY_BY_DEMOGRAPHY_TYPE,
    ULTRA_STRESS_TYPE_TO_DEMOGRAPHY_TYPE,
    Config,
)
from .ne_templates import empirical_ne_templates_available, load_empirical_ne_templates
from .utils import bin_average_log10_ne, loguniform, piecewise_eval, time_grid


SamplerResult = tuple[list[float], list[float], dict]
DemoSampler = Callable[[np.random.Generator, Config, np.ndarray, np.ndarray], SamplerResult]

STRESS_NE_CAP = 500_000.0
ULTRA_STRESS_NE_CAP = 1_000_000.0


DEMO_PROB_FIELDS = [
    ("constant", "p_constant"),
    ("near_constant", "p_near_constant"),
    ("smooth_random_walk", "p_smooth_random_walk"),
    ("dense_random_walk", "p_dense_random_walk"),
    ("dense_multiscale_random_walk", "p_dense_multiscale_random_walk"),
    ("dense_recent_wiggle", "p_dense_recent_wiggle"),
    ("classic_sawtooth", "p_classic_sawtooth"),
    ("canonical_zigzag_1s14", "p_canonical_zigzag_1s14"),
    ("canonical_smcpp_sawtooth", "p_canonical_smcpp_sawtooth"),
    ("empirical_human_ne_template", "p_empirical_human_ne_template"),
    ("smooth_random_walk_stress", "p_smooth_random_walk_stress"),
    ("dense_random_walk_stress", "p_dense_random_walk_stress"),
    ("ultra_stress", "p_ultra_stress"),
    ("single_bottleneck", "p_single_bottleneck"),
    ("recent_bottleneck", "p_recent_bottleneck"),
    ("recent_bottleneck_extreme", "p_recent_bottleneck_extreme"),
    ("recent_founder_recovery", "p_recent_founder_recovery"),
    ("founder_recovery_extreme", "p_founder_recovery_extreme"),
    ("continuous_exponential_growth", "p_continuous_exponential_growth"),
    ("rapid_recent_growth_extreme", "p_rapid_recent_growth_extreme"),
    ("continuous_exponential_decline", "p_continuous_exponential_decline"),
    ("three_epoch", "p_three_epoch"),
    ("serial_founder", "p_serial_founder"),
    ("serial_founder_extreme", "p_serial_founder_extreme"),
    ("ancient_event", "p_ancient_event"),
    ("ancient_recent_compound", "p_ancient_recent_compound"),
    ("ancient_recent_conflict", "p_ancient_recent_conflict"),
    ("oscillating_mild", "p_oscillating_mild"),
    ("zigzag_strong", "p_zigzag_strong"),
    # Legacy configuration keys remain accepted for backwards-compatible configs.
    ("expansion", "p_expansion"),
    ("contraction", "p_contraction"),
    ("zigzag", "p_zigzag"),
]


def build_demography(
    breaks: Sequence[float],
    values: Sequence[float],
    growth_rates: Sequence[float] | None = None,
) -> msprime.Demography:
    if growth_rates is None:
        growth_rates = [0.0] * len(values)
    if len(growth_rates) != len(values):
        raise ValueError("growth_rates must contain one rate per demographic epoch")
    dem = msprime.Demography()
    dem.add_population(
        name="pop",
        initial_size=float(values[0]),
        growth_rate=float(growth_rates[0]),
    )
    for t, n, growth_rate in zip(breaks, values[1:], growth_rates[1:]):
        dem.add_population_parameters_change(
            time=float(t),
            initial_size=float(n),
            growth_rate=float(growth_rate),
            population="pop",
        )
    dem.sort_events()
    return dem


def _clean_demography(
    breaks: Sequence[float],
    values: Sequence[float],
    growth_rates: Sequence[float] | None,
) -> tuple[list[float], list[float], list[float]]:
    if len(values) != len(breaks) + 1:
        raise ValueError("demography values must have exactly one more item than breaks")
    if growth_rates is None:
        growth_rates = [0.0] * len(values)
    if len(growth_rates) != len(values):
        raise ValueError("growth_rates must contain one rate per demographic epoch")
    pairs = sorted(
        (float(t), float(v), float(g))
        for t, v, g in zip(breaks, values[1:], growth_rates[1:])
        if np.isfinite(t) and float(t) > 0
    )
    out_breaks: list[float] = []
    out_values = [float(values[0])]
    out_growth_rates = [float(growth_rates[0])]
    last = 0.0
    for t, v, growth_rate in pairs:
        if t <= last + 1e-6:
            continue
        out_breaks.append(t)
        out_values.append(float(np.clip(v, 50.0, 2_000_000.0)))
        out_growth_rates.append(growth_rate)
        last = t
    out_values[0] = float(np.clip(out_values[0], 50.0, 2_000_000.0))
    if not np.all(np.isfinite(out_growth_rates)):
        raise ValueError("demography growth rates must be finite")
    return out_breaks, out_values, out_growth_rates


def resolve_demography_type(
    generator_recipe: str,
    values: Sequence[float],
    meta: dict,
) -> str:
    """Map a fine-grained generator recipe to its public scientific type."""
    if generator_recipe == "ultra_stress":
        stress_type = str(meta.get("stress_type", ""))
        try:
            return ULTRA_STRESS_TYPE_TO_DEMOGRAPHY_TYPE[stress_type]
        except KeyError as exc:
            raise ValueError(f"Unknown ultra-stress recipe subtype: {stress_type!r}") from exc
    if generator_recipe == "ancient_event":
        present_ne = float(values[0])
        ancient_ne = float(values[-1])
        if present_ne > ancient_ne:
            return "expansion"
        if present_ne < ancient_ne:
            return "contraction"
        return "constant"
    try:
        demography_type = GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE[generator_recipe]
    except KeyError as exc:
        raise ValueError(f"Unknown demography generator recipe: {generator_recipe!r}") from exc
    if demography_type is None:
        raise ValueError(f"Generator recipe {generator_recipe!r} requires dynamic classification")
    return demography_type


def _trajectory_form(
    generator_recipe: str,
    breaks: Sequence[float],
    growth_rates: Sequence[float],
    meta: dict,
) -> str:
    if generator_recipe == "constant":
        return "constant"
    if generator_recipe == "empirical_human_ne_template":
        return "empirical_template"
    if meta.get("bin_aligned_demography", False):
        return "bin_aligned_piecewise_constant"
    if any(abs(float(rate)) > 0.0 for rate in growth_rates):
        if meta.get("demography_function") == "exponential_linear_time" and len(breaks) == 1:
            return "exponential_linear_time"
        return "piecewise_exponential"
    if len(breaks) == 1:
        return "step"
    return "piecewise_step"


def trajectory_shape_values(
    edges: np.ndarray,
    breaks: Sequence[float],
    values: Sequence[float],
    growth_rates: Sequence[float],
) -> np.ndarray:
    """Return ordered left/right-limit values from the present to target max."""
    values_arr = np.asarray(values, dtype=np.float64)
    rates_arr = np.asarray(growth_rates, dtype=np.float64)
    breaks_arr = np.asarray(breaks, dtype=np.float64)
    if len(values_arr) != len(breaks_arr) + 1 or len(rates_arr) != len(values_arr):
        raise ValueError("invalid demographic epoch arrays")

    left = 0.0
    right = float(edges[-1])
    starts = np.concatenate(([0.0], breaks_arr))

    def epoch_value(epoch: int, time: float) -> float:
        elapsed = time - float(starts[epoch])
        with np.errstate(over="raise", invalid="raise"):
            result = float(values_arr[epoch] * np.exp(-rates_arr[epoch] * elapsed))
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError("demographic trajectory has a non-finite or non-positive size")
        return result

    left_epoch = int(np.searchsorted(breaks_arr, left, side="right"))
    path = [epoch_value(left_epoch, left)]
    for epoch, boundary in enumerate(breaks_arr):
        if left < float(boundary) < right:
            # Preserve both limits when an instantaneous size reset accompanies
            # a rate change. Smooth transitions simply contribute a duplicate.
            path.append(epoch_value(epoch, float(boundary)))
            post_change = float(values_arr[epoch + 1])
            if not np.isfinite(post_change) or post_change <= 0.0:
                raise ValueError(
                    "demographic trajectory has a non-finite or non-positive size"
                )
            path.append(post_change)
    right_epoch = int(np.searchsorted(breaks_arr, right, side="left"))
    path.append(epoch_value(right_epoch, right))
    return np.asarray(path, dtype=np.float64)


def summarize_demography(
    edges: np.ndarray,
    breaks: Sequence[float],
    values: Sequence[float],
    growth_rates: Sequence[float],
    meta: dict,
    *,
    generator_recipe: str,
    demography_type: str,
) -> dict:
    t_max = float(edges[-1])
    recent_grid = np.geomspace(float(edges[0]), min(5_000.0, t_max), 96)
    ancient_left = min(max(50_000.0, float(edges[0])), t_max)
    ancient_grid = np.geomspace(ancient_left, t_max, 96) if ancient_left < t_max else np.array([t_max])
    recent_ne = piecewise_eval(recent_grid, breaks, values, growth_rates)
    ancient_ne = piecewise_eval(ancient_grid, breaks, values, growth_rates)
    if (
        not np.all(np.isfinite(recent_ne))
        or not np.all(np.isfinite(ancient_ne))
        or np.any(recent_ne <= 0.0)
        or np.any(ancient_ne <= 0.0)
    ):
        raise ValueError("demographic trajectory summary is non-finite or non-positive")
    shape_values = trajectory_shape_values(
        edges,
        breaks,
        values,
        growth_rates,
    )
    min_ne = float(np.min(shape_values))
    max_ne = float(np.max(shape_values))
    log10_values = np.log10(shape_values)
    log10_diffs = np.diff(log10_values)
    nonzero_diffs = log10_diffs[np.abs(log10_diffs) > 1e-12]
    signs = np.sign(nonzero_diffs)
    turn_count = int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0
    log10_ne_range = float(np.max(log10_values) - np.min(log10_values))
    total_variation_log10 = float(np.sum(np.abs(log10_diffs)))
    if log10_ne_range <= 1e-12:
        direction = "stable"
    elif turn_count > 0:
        direction = "mixed"
    elif shape_values[0] > shape_values[-1]:
        direction = "toward_present_growth"
    else:
        direction = "toward_present_decline"

    if not breaks:
        time_regime = "none"
    else:
        min_break = float(min(breaks))
        max_break = float(max(breaks))
        if min_break <= 5_000.0 and max_break >= 50_000.0:
            time_regime = "multiscale"
        elif max_break <= 5_000.0:
            time_regime = "recent"
        elif min_break >= 50_000.0:
            time_regime = "ancient"
        else:
            time_regime = "intermediate"

    event_duration = float(meta.get("event_duration", 0.0))
    recipe_severity = meta.get("event_severity")
    if recipe_severity is not None:
        recipe_severity = float(recipe_severity)
    severity = float(min_ne / max(max_ne, 1.0))
    has_recent = bool(meta.get("has_recent_event", any(float(t) <= 5_000.0 for t in breaks)))
    has_ancient = bool(meta.get("has_ancient_event", any(float(t) >= 50_000.0 for t in breaks)))
    if generator_recipe == "stdpopsim":
        curve_origin = "catalog_model"
        dataset_regime = "anchor"
    elif meta.get("literature_demography_anchor", False):
        curve_origin = "literature_anchor"
        dataset_regime = "benchmark"
    elif meta.get("empirical_ne_template", False):
        curve_origin = "empirical_derived"
        dataset_regime = "normal"
    elif meta.get("ultra_stress_scenario", False):
        curve_origin = "generic_synthetic"
        dataset_regime = "ultra"
    elif meta.get("stress_scenario", False):
        curve_origin = "generic_synthetic"
        dataset_regime = "extreme"
    else:
        curve_origin = "generic_synthetic"
        dataset_regime = "normal"
    return {
        "n_epochs": int(len(values)),
        "n_change_points": int(len(breaks)),
        "has_recent_event": has_recent,
        "has_ancient_event": has_ancient,
        "min_Ne": min_ne,
        "max_Ne": max_ne,
        "Ne_ratio_max_min": float(max_ne / max(min_ne, 1.0)),
        "recent_min_Ne": float(np.min(recent_ne)),
        "ancient_mean_Ne": float(np.mean(ancient_ne)),
        "event_severity": severity,
        "recipe_event_severity": recipe_severity,
        "event_duration": event_duration,
        "log10_ne_range": log10_ne_range,
        "total_variation_log10": total_variation_log10,
        "mean_abs_log10_change": float(
            total_variation_log10 / max(1, len(nonzero_diffs))
        ),
        "turn_count": turn_count,
        "direction": direction,
        "time_regime": time_regime,
        "trajectory_form": _trajectory_form(
            generator_recipe,
            breaks,
            growth_rates,
            meta,
        ),
        "curve_origin": curve_origin,
        "dataset_regime": dataset_regime,
        "history_family": GENERATOR_RECIPE_HISTORY_FAMILY_OVERRIDES.get(
            generator_recipe,
            HISTORY_FAMILY_BY_DEMOGRAPHY_TYPE[demography_type],
        ),
        "time_span_min": 0.0,
        "target_time_span_min": float(edges[0]),
        "time_span_max": t_max,
    }


def _native_exponential_transition(
    present_ne: float,
    older_ne: float,
    start: float,
) -> tuple[list[float], list[float], list[float]]:
    """Return an exact linear-time exponential transition for msprime.

    Positive growth rates mean growth towards the present, following msprime's
    backward-time convention ``N(t) = N(0) exp(-g t)``.
    """
    if start <= 0:
        raise ValueError("exponential transition time must be positive")
    growth_rate = float(np.log(present_ne / older_ne) / start)
    return [float(start)], [float(present_ne), float(older_ne)], [growth_rate, 0.0]


def _boundary_values(
    initial_size: float,
    breaks: Sequence[float],
    growth_rates: Sequence[float],
) -> list[float]:
    """Calculate continuous epoch-boundary sizes under msprime semantics."""
    if len(growth_rates) != len(breaks) + 1:
        raise ValueError("growth_rates must contain one rate per demographic epoch")
    values = [float(initial_size)]
    previous_time = 0.0
    for index, boundary in enumerate(breaks):
        elapsed = float(boundary) - previous_time
        values.append(float(values[-1] * np.exp(-float(growth_rates[index]) * elapsed)))
        previous_time = float(boundary)
    return values


def _smoothed_array(arr: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return arr
    kernel = np.ones(win) / win
    sm = np.convolve(arr, kernel, mode="same")
    half = win // 2
    if half:
        sm[:half] = arr[:half]
        sm[-half:] = arr[-half:]
    return sm


def _bin_aligned_values(
    log10_values: Sequence[float],
    edges: np.ndarray,
    idx: Sequence[int] | None = None,
    lo: float = 50.0,
    hi: float = 2_000_000.0,
) -> tuple[list[float], list[float]]:
    arr = np.asarray(log10_values, dtype=np.float64)
    if idx is None:
        use_idx = np.arange(arr.size, dtype=int)
    else:
        use_idx = np.unique(np.asarray(idx, dtype=int))
        use_idx = use_idx[(0 <= use_idx) & (use_idx < arr.size)]
    if use_idx.size == 0 or use_idx[0] != 0:
        use_idx = np.insert(use_idx, 0, 0)
    breaks = [float(edges[i]) for i in use_idx[1:]]
    values = [float(np.clip(10 ** float(arr[i]), lo, hi)) for i in use_idx]
    return breaks, values


def _constant(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    n0 = loguniform(rng, 3_000, 200_000)
    return [], [n0], {
        "control_type": "constant",
        "event_severity": 1.0,
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _near_constant(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    n0 = loguniform(rng, 5_000, 150_000)
    k = min(8, cfg.time_bins)
    idx = np.linspace(0, cfg.time_bins - 1, num=k, dtype=int)
    step_sd = rng.uniform(0.02, 0.05)
    logn = np.log(n0)
    vals = [logn]
    for _ in range(1, k):
        logn = float(logn + rng.normal(0.0, step_sd))
        vals.append(logn)
    breaks = [float(mids[i]) for i in idx[1:]]
    values = [float(np.exp(v)) for v in vals]
    return breaks, values, {
        "control_type": "near_constant",
        "near_constant_step_sd": float(step_sd),
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _smooth_random_walk(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    logn = rng.uniform(3.6, 5.2)
    vals_log = [logn]
    step_sd = rng.uniform(0.08, 0.22)
    for _ in range(1, cfg.time_bins):
        logn = float(np.clip(logn + rng.normal(0, step_sd), 2.7, 5.7))
        vals_log.append(logn)
    arr = np.array(vals_log)
    win = int(rng.integers(3, 7))
    sm = _smoothed_array(arr, win)
    knot_count = min(cfg.time_bins, max(12, int(round(cfg.time_bins * rng.uniform(0.35, 0.55)))))
    idx = np.linspace(0, cfg.time_bins - 1, num=knot_count, dtype=int)
    breaks, values = _bin_aligned_values(sm, edges, idx)
    return breaks, values, {
        "random_walk_step_sd": float(step_sd),
        "smoothing_window": int(win),
        "knot_count": int(len(values)),
        "bin_aligned_demography": True,
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _smooth_random_walk_stress(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    log_min = np.log10(100.0)
    log_max = np.log10(STRESS_NE_CAP)
    logn = rng.uniform(2.2, 5.55)
    vals_log = [logn]
    step_sd = rng.uniform(0.22, 0.55)
    jump_prob = rng.uniform(0.05, 0.12)
    for _ in range(1, cfg.time_bins):
        step = rng.normal(0.0, step_sd)
        if rng.random() < jump_prob:
            step += rng.normal(0.0, step_sd * 2.5)
        logn = float(np.clip(logn + step, log_min, log_max))
        vals_log.append(logn)
    arr = np.array(vals_log)
    win = int(rng.integers(1, 4))
    sm = _smoothed_array(arr, win)
    knot_count = min(cfg.time_bins, max(16, int(round(cfg.time_bins * rng.uniform(0.55, 0.85)))))
    idx = np.linspace(0, cfg.time_bins - 1, num=knot_count, dtype=int)
    breaks, values = _bin_aligned_values(sm, edges, idx, lo=100.0, hi=STRESS_NE_CAP)
    return breaks, values, {
        "stress_scenario": True,
        "stress_type": "smooth_random_walk_stress",
        "stress_ne_cap": float(STRESS_NE_CAP),
        "random_walk_step_sd": float(step_sd),
        "random_walk_jump_prob": float(jump_prob),
        "smoothing_window": int(win),
        "knot_count": int(len(values)),
        "bin_aligned_demography": True,
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _dense_random_walk(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    logn = rng.uniform(3.5, 5.25)
    anchor = logn
    vals_log = [logn]
    step_sd = rng.uniform(0.035, 0.09)
    mean_reversion = rng.uniform(0.03, 0.08)
    for _ in range(1, cfg.time_bins):
        step = rng.normal(0.0, step_sd) - mean_reversion * (logn - anchor)
        logn = float(np.clip(logn + step, 2.7, 5.85))
        vals_log.append(logn)
    arr = _smoothed_array(np.asarray(vals_log), int(rng.integers(1, 3)))
    breaks, values = _bin_aligned_values(arr, edges)
    return breaks, values, {
        "dense_demography": True,
        "dense_type": "correlated_random_walk",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "random_walk_step_sd": float(step_sd),
        "mean_reversion": float(mean_reversion),
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _dense_multiscale_random_walk(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t = np.arange(cfg.time_bins, dtype=np.float64)
    coarse_k = int(rng.integers(5, 10))
    coarse_x = np.linspace(0, cfg.time_bins - 1, coarse_k)
    trend_sd = rng.uniform(0.12, 0.30)
    coarse = [rng.uniform(3.4, 5.25)]
    for _ in range(1, coarse_k):
        coarse.append(float(np.clip(coarse[-1] + rng.normal(0.0, trend_sd), 2.7, 5.9)))
    trend = np.interp(t, coarse_x, np.asarray(coarse))

    wiggle_sd = rng.uniform(0.015, 0.055)
    phi = rng.uniform(0.65, 0.90)
    wiggle = np.zeros(cfg.time_bins, dtype=np.float64)
    for i in range(1, cfg.time_bins):
        wiggle[i] = phi * wiggle[i - 1] + rng.normal(0.0, wiggle_sd)

    arr = np.clip(trend + wiggle, 2.7, 5.9)
    breaks, values = _bin_aligned_values(arr, edges)
    return breaks, values, {
        "dense_demography": True,
        "dense_type": "multiscale_random_walk",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "coarse_knot_count": int(coarse_k),
        "trend_step_sd": float(trend_sd),
        "wiggle_step_sd": float(wiggle_sd),
        "wiggle_phi": float(phi),
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _dense_recent_wiggle(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    logn = rng.uniform(3.5, 5.25)
    vals_log = [logn]
    ancient_sd = rng.uniform(0.018, 0.055)
    recent_boost = rng.uniform(0.08, 0.18)
    recent_scale = rng.uniform(6.0, 16.0)
    for i in range(1, cfg.time_bins):
        step_sd = ancient_sd + recent_boost * np.exp(-i / recent_scale)
        logn = float(np.clip(logn + rng.normal(0.0, step_sd), 2.7, 5.85))
        vals_log.append(logn)
    breaks, values = _bin_aligned_values(vals_log, edges)
    return breaks, values, {
        "dense_demography": True,
        "dense_type": "recent_weighted_wiggle",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "ancient_step_sd": float(ancient_sd),
        "recent_step_boost": float(recent_boost),
        "recent_wiggle_scale_bins": float(recent_scale),
        "has_recent_event": True,
        "has_ancient_event": False,
    }


def _dense_random_walk_stress(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    log_min = np.log10(100.0)
    log_max = np.log10(STRESS_NE_CAP)
    logn = rng.uniform(2.2, 5.55)
    vals_log = [logn]
    step_sd = rng.uniform(0.12, 0.32)
    jump_prob = rng.uniform(0.03, 0.08)
    for _ in range(1, cfg.time_bins):
        step = rng.normal(0.0, step_sd)
        if rng.random() < jump_prob:
            step += rng.normal(0.0, step_sd * rng.uniform(2.0, 4.0))
        logn = float(np.clip(logn + step, log_min, log_max))
        vals_log.append(logn)
    breaks, values = _bin_aligned_values(vals_log, edges, lo=100.0, hi=STRESS_NE_CAP)
    return breaks, values, {
        "stress_scenario": True,
        "stress_type": "dense_random_walk_stress",
        "stress_ne_cap": float(STRESS_NE_CAP),
        "dense_demography": True,
        "dense_type": "stress_random_walk",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "random_walk_step_sd": float(step_sd),
        "random_walk_jump_prob": float(jump_prob),
        "has_recent_event": False,
        "has_ancient_event": False,
    }


def _single_bottleneck(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    n0 = loguniform(rng, 5_000, 150_000)
    start = loguniform(rng, 500, t_max / 3)
    end = min(t_max * 0.95, start * rng.uniform(1.5, 20.0))
    severity = loguniform(rng, 0.02, 0.5)
    nbot = max(200.0, n0 * severity)
    nanc = n0 * loguniform(rng, 0.5, 3.0)
    return [start, end], [n0, nbot, nanc], {
        "event_time": float(start),
        "event_duration": float(end - start),
        "event_severity": float(severity),
    }


def _canonical_zigzag_1s14(
    rng: np.random.Generator,
    cfg: Config,
    edges: np.ndarray,
    mids: np.ndarray,
) -> SamplerResult:
    # Exact stdpopsim Zigzag_1S14 port of the validation model introduced by
    # Schiffels & Durbin (2014), doi:10.1038/ng.3015. The fixed parameters are
    # intentionally independent of the output label grid and random seed.
    n0 = 5.0 * 14_312.0
    breaks = [
        33.333334976,
        133.3334544,
        533.33324512,
        2_133.3352704,
        8_533.329632,
        34_133.318528,
    ]
    growth_rates = [
        0.0,
        0.02302578256,
        -0.0057564631,
        0.00143911578,
        -0.0003597785,
        8.9944801565e-5,
        0.0,
    ]
    values = _boundary_values(n0, breaks, growth_rates)
    # The catalog model explicitly resets the ancient population size. The
    # rounded published rates otherwise differ by about 0.0006%.
    values[-1] = n0 / 10.0
    return breaks, values, {
        "classic_demography_model": True,
        "literature_demography_anchor": True,
        "literature_model_id": "Zigzag_1S14",
        "literature_reference": "Schiffels_and_Durbin_2014",
        "demography_function": "piecewise_exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "dense_type": "canonical_zigzag_1s14",
        "knot_count": int(len(values)),
        "sawtooth_cycles": 2.5,
        "sawtooth_exponential_segment_count": 5,
        "sawtooth_internal_extrema_count": 4,
        "sawtooth_randomized": False,
        "has_recent_event": True,
        "has_ancient_event": False,
        "event_severity": float(min(values) / max(values)),
    }


def _canonical_smcpp_sawtooth(
    rng: np.random.Generator,
    cfg: Config,
    edges: np.ndarray,
    mids: np.ndarray,
) -> SamplerResult:
    # Exact scrm command from Terhorst, Kamm & Song (2017) Online Methods,
    # doi:10.1038/ng.3748. This differs slightly from Zigzag_1S14 and is kept
    # as a separate provenance-preserving benchmark anchor.
    nref = 14_312.0
    breaks = [
        33.333334976,
        133.3334544,
        533.50498912,
        2_133.3352704,
        8_533.329632,
        34_133.318528,
    ]
    values = [
        5.0 * nref,
        5.0 * nref,
        0.500002043581 * nref,
        5.00496081234 * nref,
        0.500618264601 * nref,
        5.0061592508 * nref,
        0.500615510407 * nref,
    ]
    scrm_scaled_growth_rates = [1_318.18, -329.546, 82.3865, -20.5966, 5.14916]
    growth_rates = [
        0.0,
        *(float(rate / (4.0 * nref)) for rate in scrm_scaled_growth_rates),
        0.0,
    ]
    return breaks, values, {
        "classic_demography_model": True,
        "literature_demography_anchor": True,
        "literature_model_id": "Terhorst2017_sawtooth",
        "literature_reference": "Terhorst_Kamm_and_Song_2017",
        "demography_function": "piecewise_exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "dense_type": "canonical_smcpp_sawtooth",
        "knot_count": int(len(values)),
        "sawtooth_cycles": 2.5,
        "sawtooth_exponential_segment_count": 5,
        "sawtooth_internal_extrema_count": 4,
        "sawtooth_randomized": False,
        "has_recent_event": True,
        "has_ancient_event": False,
        "event_severity": float(min(values) / max(values)),
    }


def _classic_sawtooth(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    # Literature-shaped training family: preserve the dimensionless
    # Zigzag_1S14 trajectory while randomizing its time, Ne scale, and log-Ne
    # contrast. The two fixed canonical versions remain separate samplers.
    base_breaks, base_values, base_meta = _canonical_zigzag_1s14(rng, cfg, edges, mids)
    time_scale = loguniform(rng, 0.55, 1.80)
    size_scale = loguniform(rng, 0.55, 1.80)
    contrast = rng.uniform(0.75, 1.25)

    center = float(np.sqrt(min(base_values) * max(base_values)))
    breaks = [float(t * time_scale) for t in base_breaks]
    values = [
        float(size_scale * center * (float(value) / center) ** contrast)
        for value in base_values
    ]
    growth_rates = [
        float(rate * contrast / time_scale)
        for rate in base_meta["demography_growth_rates"]
    ]
    return breaks, values, {
        "classic_demography_model": True,
        "literature_shaped_family": True,
        "source_literature_model_id": "Zigzag_1S14",
        "demography_function": "piecewise_exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "dense_type": "classic_sawtooth",
        "knot_count": int(len(values)),
        "sawtooth_cycles": 2.5,
        "sawtooth_exponential_segment_count": 5,
        "sawtooth_internal_extrema_count": 4,
        "sawtooth_randomized": True,
        "sawtooth_time_scale": float(time_scale),
        "sawtooth_size_scale": float(size_scale),
        "sawtooth_log_ne_contrast": float(contrast),
        "has_recent_event": True,
        "has_ancient_event": False,
        "event_severity": float(min(values) / max(values)),
    }


def _template_log10_bin_average(template, edges: np.ndarray, time_scale: float, n_grid: int = 8) -> np.ndarray:
    t = np.maximum(np.asarray(template.generations_ago, dtype=np.float64), 1e-6)
    log_t = np.log(t)
    log_ne = np.log10(np.asarray(template.ne, dtype=np.float64))
    out: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        grid = np.geomspace(float(left), float(right), n_grid) / max(time_scale, 1e-6)
        vals = np.interp(np.log(np.maximum(grid, 1e-6)), log_t, log_ne, left=log_ne[0], right=log_ne[-1])
        out.append(float(np.mean(vals)))
    return np.asarray(out, dtype=np.float64)


def _empirical_human_ne_template(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    store = load_empirical_ne_templates(cfg)
    template = store.choose(rng, cfg)

    time_scale = float(np.exp(rng.normal(0.0, cfg.empirical_ne_template_time_warp_sd)))
    time_scale = float(np.clip(time_scale, 0.45, 2.25))
    log_shift = float(rng.normal(0.0, cfg.empirical_ne_template_log10_shift_sd))
    contrast = float(np.exp(rng.normal(0.0, cfg.empirical_ne_template_contrast_sd)))
    contrast = float(np.clip(contrast, 0.70, 1.45))
    bin_noise_sd = float(cfg.empirical_ne_template_bin_noise_sd)

    raw = _template_log10_bin_average(template, edges, time_scale=time_scale)
    center = float(np.median(raw))
    arr = center + contrast * (raw - center) + log_shift
    if bin_noise_sd > 0:
        noise = rng.normal(0.0, bin_noise_sd, size=cfg.time_bins)
        noise = _smoothed_array(noise, int(rng.integers(2, 5)))
        arr = arr + noise
    arr = np.clip(arr, np.log10(100.0), np.log10(2_000_000.0))

    breaks, values = _bin_aligned_values(arr, edges, lo=100.0, hi=2_000_000.0)
    recent = arr[mids <= 5_000.0]
    ancient = arr[mids >= 50_000.0]
    return breaks, values, {
        "empirical_ne_template": True,
        "target_quality": "empirical_inference_template_perturbed",
        "template_source_key": template.source_key,
        "template_method": template.method,
        "template_population": template.population,
        "template_path": template.path,
        "template_time_unit": template.time_unit_note,
        "template_generation_time_years": 29.0,
        "template_curve_point_count": int(template.ne.size),
        "template_time_min": float(np.min(template.generations_ago)),
        "template_time_max": float(np.max(template.generations_ago)),
        "template_min_Ne": float(np.min(template.ne)),
        "template_max_Ne": float(np.max(template.ne)),
        "template_time_scale": time_scale,
        "template_log10_shift": log_shift,
        "template_contrast_scale": contrast,
        "template_bin_noise_sd": bin_noise_sd,
        "dense_demography": True,
        "dense_type": "empirical_human_ne_template",
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "has_recent_event": bool(recent.size and np.ptp(recent) >= 0.15),
        "has_ancient_event": bool(ancient.size and np.ptp(ancient) >= 0.15),
        "event_severity": float(10 ** (np.min(arr) - np.max(arr))),
    }


def _recent_bottleneck(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    n0 = loguniform(rng, 5_000, 200_000)
    start = loguniform(rng, 50, 5_000)
    end = min(t_max * 0.8, start + loguniform(rng, 50, 3_000))
    severity = loguniform(rng, 0.005, 0.2)
    nbot = max(100.0, n0 * severity)
    nanc = n0 * loguniform(rng, 0.5, 2.0)
    end = max(start + 1.0, end)
    return [start, end], [n0, nbot, nanc], {
        "event_time": float(start),
        "event_duration": float(end - start),
        "event_severity": float(severity),
        "has_recent_event": True,
    }


def _recent_bottleneck_extreme(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    n0 = loguniform(rng, 10_000, 300_000)
    start = loguniform(rng, 50, 3_000)
    duration = loguniform(rng, 10, 1_500)
    end = min(t_max * 0.6, start + duration)
    severity = loguniform(rng, 0.0005, 0.02)
    nbot = max(100.0, n0 * severity)
    nanc = loguniform(rng, 1_000, STRESS_NE_CAP)
    end = max(start + 1.0, end)
    return [start, end], [n0, nbot, nanc], {
        "stress_scenario": True,
        "stress_type": "recent_bottleneck_extreme",
        "stress_ne_cap": float(STRESS_NE_CAP),
        "event_time": float(start),
        "event_duration": float(end - start),
        "event_severity": float(severity),
        "has_recent_event": True,
    }


def _recent_founder_recovery(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    founder_time = loguniform(rng, 100, 5_000)
    duration = loguniform(rng, 50, 2_000)
    founder_end = min(t_max * 0.8, founder_time + duration)
    severity = loguniform(rng, 0.005, 0.1)
    recovery_factor = loguniform(rng, 5.0, 100.0)
    founder_ne = loguniform(rng, 200, 5_000)
    present_ne = min(1_000_000.0, founder_ne * recovery_factor)
    ancient_ne = min(1_000_000.0, founder_ne / severity * loguniform(rng, 0.5, 2.0))
    return [founder_time, max(founder_time + 1.0, founder_end)], [present_ne, founder_ne, ancient_ne], {
        "event_time": float(founder_time),
        "event_duration": float(founder_end - founder_time),
        "event_severity": float(severity),
        "recovery_factor": float(recovery_factor),
        "has_recent_event": True,
    }


def _founder_recovery_extreme(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    founder_time = loguniform(rng, 50, 3_000)
    duration = loguniform(rng, 10, 1_000)
    founder_end = min(t_max * 0.6, founder_time + duration)
    founder_ne = loguniform(rng, 100, 2_000)
    recovery_factor = loguniform(rng, 20.0, 250.0)
    present_ne = min(STRESS_NE_CAP, founder_ne * recovery_factor)
    ancient_ne = loguniform(rng, 5_000, STRESS_NE_CAP)
    return [founder_time, max(founder_time + 1.0, founder_end)], [present_ne, founder_ne, ancient_ne], {
        "stress_scenario": True,
        "stress_type": "founder_recovery_extreme",
        "stress_ne_cap": float(STRESS_NE_CAP),
        "event_time": float(founder_time),
        "event_duration": float(founder_end - founder_time),
        "event_severity": float(founder_ne / max(present_ne, 1.0)),
        "recovery_factor": float(recovery_factor),
        "has_recent_event": True,
    }


def _continuous_exponential_growth(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    start = loguniform(rng, 100, 30_000)
    sampled_factor = loguniform(rng, 2.0, 100.0)
    older_ne = loguniform(rng, 2_000, 80_000)
    present_ne = min(1_000_000.0, older_ne * sampled_factor)
    realized_factor = present_ne / older_ne
    breaks, values, growth_rates = _native_exponential_transition(present_ne, older_ne, start)
    mode = "rapid_recent_growth" if start <= 5_000 and realized_factor >= 10 else "mild_growth"
    return breaks, values, {
        "growth_start": float(start),
        "growth_factor": float(realized_factor),
        "sampled_growth_factor": float(sampled_factor),
        "growth_rate_per_generation": float(growth_rates[0]),
        "growth_mode": mode,
        "demography_function": "exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "has_recent_event": bool(start <= 5_000),
        "event_time": float(start),
        "event_severity": float(min(present_ne, older_ne) / max(present_ne, older_ne)),
    }


def _rapid_recent_growth_extreme(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    start = loguniform(rng, 50, 8_000)
    sampled_factor = loguniform(rng, 20.0, 250.0)
    older_ne = loguniform(rng, 500, 20_000)
    present_ne = min(STRESS_NE_CAP, older_ne * sampled_factor)
    realized_factor = present_ne / older_ne
    breaks, values, growth_rates = _native_exponential_transition(present_ne, older_ne, start)
    return breaks, values, {
        "stress_scenario": True,
        "stress_type": "rapid_recent_growth_extreme",
        "stress_ne_cap": float(STRESS_NE_CAP),
        "growth_start": float(start),
        "growth_factor": float(realized_factor),
        "sampled_growth_factor": float(sampled_factor),
        "growth_rate_per_generation": float(growth_rates[0]),
        "growth_mode": "extreme_rapid_recent_growth",
        "demography_function": "exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "has_recent_event": True,
        "event_time": float(start),
        "event_severity": float(min(present_ne, older_ne) / max(present_ne, older_ne)),
    }


def _continuous_exponential_decline(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    start = loguniform(rng, 100, 30_000)
    sampled_factor = loguniform(rng, 2.0, 50.0)
    older_ne = loguniform(rng, 20_000, 300_000)
    present_ne = max(200.0, older_ne / sampled_factor)
    realized_factor = older_ne / present_ne
    breaks, values, growth_rates = _native_exponential_transition(present_ne, older_ne, start)
    return breaks, values, {
        "decline_start": float(start),
        "decline_factor": float(realized_factor),
        "sampled_decline_factor": float(sampled_factor),
        "growth_rate_per_generation": float(growth_rates[0]),
        "demography_function": "exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "has_recent_event": bool(start <= 5_000),
        "event_time": float(start),
        "event_severity": float(min(present_ne, older_ne) / max(present_ne, older_ne)),
    }


def _three_epoch(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    t1 = loguniform(rng, 100, 20_000)
    t2 = loguniform(rng, max(t1 * 1.5, 1_000), t_max * 0.8)
    n0 = loguniform(rng, 3_000, 200_000)
    n1 = float(np.clip(n0 * loguniform(rng, 0.1, 8.0), 200, 500_000))
    n2 = float(np.clip(n1 * loguniform(rng, 0.1, 8.0), 200, 500_000))
    return [t1, t2], [n0, n1, n2], {"event_time": float(t1), "event_duration": float(t2 - t1)}


def _serial_founder(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    n_events = int(rng.integers(2, 5))
    base_times = np.geomspace(100.0, 50_000.0, n_events + 2)[1:-1]
    times = sorted((base_times * rng.lognormal(0.0, 0.25, size=n_events)).tolist())
    n = loguniform(rng, 20_000, 300_000)
    values = [n]
    breaks: list[float] = []
    severities: list[float] = []
    durations: list[float] = []
    for idx, t in enumerate(times):
        severity = loguniform(rng, 0.02, 0.5)
        recovery = loguniform(rng, 1.5, 10.0)
        next_t = times[idx + 1] if idx + 1 < len(times) else 50_000.0
        max_duration = max(50.0, min(2_000.0, (next_t - t) * 0.35))
        duration = loguniform(rng, 50, max_duration)
        founder_ne = max(100.0, n * severity)
        recovered_ne = min(1_000_000.0, founder_ne * recovery)
        breaks.extend([float(t), float(t + duration)])
        values.extend([founder_ne, recovered_ne])
        n = recovered_ne
        severities.append(severity)
        durations.append(duration)
    return breaks, values, {
        "n_founder_events": int(n_events),
        "event_severity": float(min(severities)),
        "event_duration": float(sum(durations)),
        "has_recent_event": bool(min(times) <= 5_000),
    }


def _serial_founder_extreme(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    n_events = int(rng.integers(3, 6))
    base_times = np.geomspace(50.0, 70_000.0, n_events + 2)[1:-1]
    times = sorted((base_times * rng.lognormal(0.0, 0.35, size=n_events)).tolist())
    n = loguniform(rng, 50_000, STRESS_NE_CAP)
    values = [n]
    breaks: list[float] = []
    severities: list[float] = []
    durations: list[float] = []
    for idx, t in enumerate(times):
        next_t = times[idx + 1] if idx + 1 < len(times) else 80_000.0
        max_duration = max(20.0, min(1_500.0, (next_t - t) * 0.25))
        duration = loguniform(rng, 20, max_duration)
        severity = loguniform(rng, 0.001, 0.05)
        recovery = loguniform(rng, 5.0, 50.0)
        founder_ne = max(100.0, n * severity)
        recovered_ne = min(STRESS_NE_CAP, founder_ne * recovery)
        breaks.extend([float(t), float(t + duration)])
        values.extend([founder_ne, recovered_ne])
        n = recovered_ne
        severities.append(severity)
        durations.append(duration)
    return breaks, values, {
        "stress_scenario": True,
        "stress_type": "serial_founder_extreme",
        "stress_ne_cap": float(STRESS_NE_CAP),
        "n_founder_events": int(n_events),
        "event_severity": float(min(severities)),
        "event_duration": float(sum(durations)),
        "has_recent_event": bool(min(times) <= 5_000),
    }


def _ancient_event(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    n0 = loguniform(rng, 3_000, 150_000)
    t = loguniform(rng, 50_000, t_max * 0.9)
    n1 = float(np.clip(n0 * loguniform(rng, 0.1, 10.0), 200, 500_000))
    return [t], [n0, n1], {"event_time": float(t), "has_ancient_event": True}


def _ancient_recent_compound(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    recent_t = loguniform(rng, 100, 5_000)
    ancient_t = loguniform(rng, 50_000, min(300_000.0, t_max * 0.9))
    n0 = loguniform(rng, 5_000, 200_000)
    recent_factor = loguniform(rng, 0.02, 0.5) if rng.random() < 0.5 else loguniform(rng, 2.0, 20.0)
    n_recent = float(np.clip(n0 * recent_factor, 100, 1_000_000))
    n_ancient = float(np.clip(n_recent * loguniform(rng, 0.1, 10.0), 200, 1_000_000))
    return [recent_t, ancient_t], [n0, n_recent, n_ancient], {
        "recent_event_time": float(recent_t),
        "ancient_event_time": float(ancient_t),
        "event_severity": float(min(n_recent, n0) / max(n_recent, n0)),
        "has_recent_event": True,
        "has_ancient_event": True,
    }


def _ancient_recent_conflict(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    recent_t = loguniform(rng, 50, 5_000)
    ancient_t = loguniform(rng, 50_000, min(300_000.0, t_max * 0.9))
    n0 = loguniform(rng, 5_000, 300_000)
    if rng.random() < 0.5:
        recent_factor = loguniform(rng, 0.001, 0.05)
        ancient_factor = loguniform(rng, 8.0, 100.0)
        conflict_mode = "recent_collapse_ancient_expansion"
    else:
        recent_factor = loguniform(rng, 8.0, 100.0)
        ancient_factor = loguniform(rng, 0.001, 0.05)
        conflict_mode = "recent_expansion_ancient_collapse"
    n_recent = float(np.clip(n0 * recent_factor, 100, STRESS_NE_CAP))
    n_ancient = float(np.clip(n_recent * ancient_factor, 100, STRESS_NE_CAP))
    return [recent_t, ancient_t], [n0, n_recent, n_ancient], {
        "stress_scenario": True,
        "stress_type": "ancient_recent_conflict",
        "stress_ne_cap": float(STRESS_NE_CAP),
        "conflict_mode": conflict_mode,
        "recent_event_time": float(recent_t),
        "ancient_event_time": float(ancient_t),
        "event_severity": float(min(n_recent, n0) / max(n_recent, n0)),
        "has_recent_event": True,
        "has_ancient_event": True,
    }


def _ultra_stress(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    cap = ULTRA_STRESS_NE_CAP
    mode = str(
        rng.choice(
            [
                "ultra_random_walk",
                "ultra_recent_bottleneck",
                "ultra_founder_recovery",
                "ultra_rapid_recent_growth",
                "ultra_serial_founder",
                "ultra_ancient_recent_conflict",
            ],
            p=[0.24, 0.20, 0.18, 0.16, 0.12, 0.10],
        )
    )
    common = {
        "stress_scenario": True,
        "ultra_stress_scenario": True,
        "stress_type": mode,
        "stress_ne_cap": float(cap),
    }

    if mode == "ultra_random_walk":
        log_min = np.log10(100.0)
        log_max = np.log10(cap)
        logn = rng.uniform(2.1, 5.85)
        step_sd = rng.uniform(0.18, 0.45)
        jump_prob = rng.uniform(0.05, 0.10)
        vals_log = [logn]
        for _ in range(1, cfg.time_bins):
            step = rng.normal(0.0, step_sd)
            if rng.random() < jump_prob:
                step += rng.normal(0.0, step_sd * rng.uniform(2.0, 4.0))
            logn = float(np.clip(logn + step, log_min, log_max))
            vals_log.append(logn)
        arr = _smoothed_array(np.asarray(vals_log), int(rng.integers(1, 4)))
        breaks, values = _bin_aligned_values(arr, edges, lo=100.0, hi=cap)
        return breaks, values, {
            **common,
            "dense_demography": True,
            "dense_type": mode,
            "bin_aligned_demography": True,
            "knot_count": int(len(values)),
            "random_walk_step_sd": float(step_sd),
            "random_walk_jump_prob": float(jump_prob),
            "has_recent_event": False,
            "has_ancient_event": False,
        }

    if mode == "ultra_recent_bottleneck":
        t_max = float(edges[-1])
        n0 = loguniform(rng, 20_000, 500_000)
        start = loguniform(rng, 50, 3_000)
        duration = loguniform(rng, 10, 1_500)
        end = max(start + 1.0, min(t_max * 0.6, start + duration))
        severity = loguniform(rng, 0.0005, 0.015)
        nbot = max(100.0, n0 * severity)
        nanc = loguniform(rng, 1_000, cap)
        return [start, end], [n0, nbot, nanc], {
            **common,
            "event_time": float(start),
            "event_duration": float(end - start),
            "event_severity": float(severity),
            "has_recent_event": True,
        }

    if mode == "ultra_founder_recovery":
        t_max = float(edges[-1])
        founder_time = loguniform(rng, 50, 3_000)
        duration = loguniform(rng, 10, 1_000)
        founder_end = max(founder_time + 1.0, min(t_max * 0.6, founder_time + duration))
        founder_ne = loguniform(rng, 100, 2_000)
        recovery_factor = loguniform(rng, 40.0, 500.0)
        present_ne = min(cap, founder_ne * recovery_factor)
        ancient_ne = loguniform(rng, 5_000, cap)
        return [founder_time, founder_end], [present_ne, founder_ne, ancient_ne], {
            **common,
            "event_time": float(founder_time),
            "event_duration": float(founder_end - founder_time),
            "event_severity": float(founder_ne / max(present_ne, 1.0)),
            "recovery_factor": float(recovery_factor),
            "has_recent_event": True,
        }

    if mode == "ultra_rapid_recent_growth":
        start = loguniform(rng, 50, 8_000)
        sampled_factor = loguniform(rng, 40.0, 500.0)
        older_ne = loguniform(rng, 500, 20_000)
        present_ne = min(cap, older_ne * sampled_factor)
        realized_factor = present_ne / older_ne
        breaks, values, growth_rates = _native_exponential_transition(present_ne, older_ne, start)
        return breaks, values, {
            **common,
            "growth_start": float(start),
            "growth_factor": float(realized_factor),
            "sampled_growth_factor": float(sampled_factor),
            "growth_rate_per_generation": float(growth_rates[0]),
            "growth_mode": mode,
            "demography_function": "exponential_linear_time",
            "native_continuous_growth": True,
            "demography_growth_rates": growth_rates,
            "has_recent_event": True,
            "event_time": float(start),
            "event_severity": float(min(present_ne, older_ne) / max(present_ne, older_ne)),
        }

    if mode == "ultra_serial_founder":
        n_events = int(rng.integers(3, 6))
        base_times = np.geomspace(50.0, 70_000.0, n_events + 2)[1:-1]
        times = sorted((base_times * rng.lognormal(0.0, 0.35, size=n_events)).tolist())
        n = loguniform(rng, 50_000, cap)
        values = [n]
        breaks: list[float] = []
        severities: list[float] = []
        durations: list[float] = []
        for idx, t in enumerate(times):
            next_t = times[idx + 1] if idx + 1 < len(times) else 80_000.0
            max_duration = max(20.0, min(1_500.0, (next_t - t) * 0.25))
            duration = loguniform(rng, 20, max_duration)
            severity = loguniform(rng, 0.001, 0.05)
            recovery = loguniform(rng, 5.0, 50.0)
            founder_ne = max(100.0, n * severity)
            recovered_ne = min(cap, founder_ne * recovery)
            breaks.extend([float(t), float(t + duration)])
            values.extend([founder_ne, recovered_ne])
            n = recovered_ne
            severities.append(severity)
            durations.append(duration)
        return breaks, values, {
            **common,
            "n_founder_events": int(n_events),
            "event_severity": float(min(severities)),
            "event_duration": float(sum(durations)),
            "has_recent_event": bool(min(times) <= 5_000),
        }

    t_max = float(edges[-1])
    recent_t = loguniform(rng, 50, 5_000)
    ancient_t = loguniform(rng, 50_000, min(300_000.0, t_max * 0.9))
    n0 = loguniform(rng, 5_000, 500_000)
    if rng.random() < 0.5:
        recent_factor = loguniform(rng, 0.001, 0.05)
        ancient_factor = loguniform(rng, 20.0, 200.0)
        conflict_mode = "recent_collapse_ancient_expansion"
    else:
        recent_factor = loguniform(rng, 20.0, 200.0)
        ancient_factor = loguniform(rng, 0.001, 0.05)
        conflict_mode = "recent_expansion_ancient_collapse"
    n_recent = float(np.clip(n0 * recent_factor, 100, cap))
    n_ancient = float(np.clip(n_recent * ancient_factor, 100, cap))
    return [recent_t, ancient_t], [n0, n_recent, n_ancient], {
        **common,
        "conflict_mode": conflict_mode,
        "recent_event_time": float(recent_t),
        "ancient_event_time": float(ancient_t),
        "event_severity": float(min(n_recent, n0) / max(n_recent, n0)),
        "has_recent_event": True,
        "has_ancient_event": True,
    }


def _oscillating(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray, strong: bool) -> SamplerResult:
    t_max = float(edges[-1])
    n_turns = int(rng.integers(8, 15) if strong else rng.integers(10, 19))
    base_times = np.geomspace(max(100.0, cfg.min_time), t_max * 0.8, n_turns)
    jitter = rng.lognormal(0.0, 0.12, size=n_turns)
    breaks = np.maximum.accumulate(base_times * jitter)
    breaks = np.clip(breaks, max(50.0, cfg.min_time), t_max * 0.9)
    # Clipping can collapse the oldest knots; keep only genuinely distinct
    # turning times so every reported cycle is scientifically meaningful.
    breaks = np.unique(breaks).astype(np.float64).tolist()

    present_ne = loguniform(rng, 3_000, 120_000)
    boundary_values = [present_ne]
    direction = 1 if rng.random() < 0.5 else -1
    lo, hi = (2.0, 8.0) if strong else (1.2, 2.5)
    for _ in range(1, len(breaks)):
        factor = loguniform(rng, lo, hi)
        candidate = boundary_values[-1] * factor if direction > 0 else boundary_values[-1] / factor
        next_ne = float(np.clip(candidate, 200.0, 500_000.0))
        # If a bound would flatten the intended turn, reverse it instead.
        if np.isclose(next_ne, boundary_values[-1], rtol=1e-12, atol=1e-9):
            candidate = boundary_values[-1] / factor if direction > 0 else boundary_values[-1] * factor
            next_ne = float(np.clip(candidate, 200.0, 500_000.0))
        direction = -1 if next_ne > boundary_values[-1] else 1
        boundary_values.append(next_ne)

    values = [present_ne, *boundary_values]
    growth_rates = [0.0]
    for left_t, right_t, left_ne, right_ne in zip(
        breaks[:-1],
        breaks[1:],
        boundary_values[:-1],
        boundary_values[1:],
    ):
        growth_rates.append(float(np.log(left_ne / right_ne) / (right_t - left_t)))
    growth_rates.append(0.0)
    return breaks, values, {
        "oscillation_strength": "strong" if strong else "mild",
        "demography_function": "piecewise_exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "zigzag_rate_change_count": int(len(breaks)),
        "zigzag_internal_extrema_count": int(max(0, len(breaks) - 2)),
        "knot_count": int(len(values)),
        "event_severity": float(min(values) / max(values)),
    }


def _oscillating_mild(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    return _oscillating(rng, cfg, edges, mids, strong=False)


def _zigzag_strong(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    return _oscillating(rng, cfg, edges, mids, strong=True)


def _zigzag_legacy(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    breaks, values, meta = _oscillating(rng, cfg, edges, mids, strong=True)
    return breaks, values, {
        **meta,
        "legacy_demography_model": False,
        "legacy_sampler_alias": True,
        "oscillation_strength": "strong",
    }


def _expansion(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    ncur = loguniform(rng, 20_000, 300_000)
    factor = loguniform(rng, 2.0, 50.0)
    nanc = ncur / factor
    t = loguniform(rng, 100, 50_000)
    return [t], [ncur, max(200.0, nanc)], {"legacy_step_change": True, "event_severity": float(1.0 / factor)}


def _contraction(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    ncur = loguniform(rng, 1_000, 50_000)
    factor = loguniform(rng, 2.0, 30.0)
    nanc = ncur * factor
    t = loguniform(rng, 500, 100_000)
    return [t], [ncur, min(500_000, nanc)], {"legacy_step_change": True, "event_severity": float(1.0 / factor)}


DEMO_SAMPLERS: dict[str, DemoSampler] = {
    "constant": _constant,
    "near_constant": _near_constant,
    "smooth_random_walk": _smooth_random_walk,
    "dense_random_walk": _dense_random_walk,
    "dense_multiscale_random_walk": _dense_multiscale_random_walk,
    "dense_recent_wiggle": _dense_recent_wiggle,
    "classic_sawtooth": _classic_sawtooth,
    "canonical_zigzag_1s14": _canonical_zigzag_1s14,
    "canonical_smcpp_sawtooth": _canonical_smcpp_sawtooth,
    "empirical_human_ne_template": _empirical_human_ne_template,
    "smooth_random_walk_stress": _smooth_random_walk_stress,
    "dense_random_walk_stress": _dense_random_walk_stress,
    "ultra_stress": _ultra_stress,
    "single_bottleneck": _single_bottleneck,
    "recent_bottleneck": _recent_bottleneck,
    "recent_bottleneck_extreme": _recent_bottleneck_extreme,
    "recent_founder_recovery": _recent_founder_recovery,
    "founder_recovery_extreme": _founder_recovery_extreme,
    "continuous_exponential_growth": _continuous_exponential_growth,
    "rapid_recent_growth_extreme": _rapid_recent_growth_extreme,
    "continuous_exponential_decline": _continuous_exponential_decline,
    "three_epoch": _three_epoch,
    "serial_founder": _serial_founder,
    "serial_founder_extreme": _serial_founder_extreme,
    "ancient_event": _ancient_event,
    "ancient_recent_compound": _ancient_recent_compound,
    "ancient_recent_conflict": _ancient_recent_conflict,
    "oscillating_mild": _oscillating_mild,
    "zigzag_strong": _zigzag_strong,
    "expansion": _expansion,
    "contraction": _contraction,
    "zigzag": _zigzag_legacy,
}


def choose_generator_recipe(rng: np.random.Generator, cfg: Config) -> str:
    choices: list[str] = []
    probs: list[float] = []
    for demo_type, field in DEMO_PROB_FIELDS:
        weight = float(getattr(cfg, field, 0.0))
        if demo_type == "empirical_human_ne_template" and weight > 0 and not empirical_ne_templates_available(cfg):
            continue
        if weight > 0:
            choices.append(demo_type)
            probs.append(weight)
    if not choices:
        return "constant"
    p = np.asarray(probs, dtype=np.float64)
    return str(rng.choice(choices, p=p / p.sum()))


def choose_demo_type(rng: np.random.Generator, cfg: Config) -> str:
    """Compatibility alias for the pre-v3 recipe-selection API."""
    return choose_generator_recipe(rng, cfg)


def sample_custom_demography(rng: np.random.Generator, cfg: Config) -> tuple[msprime.Demography, np.ndarray, dict]:
    edges, mids = time_grid(cfg)
    generator_recipe = choose_generator_recipe(rng, cfg)
    sampler = DEMO_SAMPLERS.get(generator_recipe)
    if sampler is None:
        raise ValueError(f"Unknown demography sampler: {generator_recipe}")

    breaks, values, meta = sampler(rng, cfg, edges, mids)
    breaks, values, growth_rates = _clean_demography(
        breaks,
        values,
        meta.get("demography_growth_rates"),
    )
    y = bin_average_log10_ne(edges, breaks, values, growth_rates)
    demography_type = resolve_demography_type(generator_recipe, values, meta)
    meta = {
        **meta,
        **summarize_demography(
            edges,
            breaks,
            values,
            growth_rates,
            meta,
            generator_recipe=generator_recipe,
            demography_type=demography_type,
        ),
        "scenario": generator_recipe,
        "generator_recipe": generator_recipe,
        "generator_recipe_id": GENERATOR_RECIPE_IDS[generator_recipe],
        "demography_id": GENERATOR_RECIPE_IDS[generator_recipe],
        "demography_type": demography_type,
        "demography_type_id": DEMOGRAPHY_TYPE_IDS[demography_type],
        "demography_mixture_version": cfg.demography_mixture_version,
        "demography_breaks": [float(x) for x in breaks],
        "demography_values": [float(x) for x in values],
        "demography_growth_rates": [float(x) for x in growth_rates],
        "target_scale": "log10_Ne",
        "target_aggregation": "log_time_bin_average",
    }
    return build_demography(breaks, values, growth_rates), y, meta
