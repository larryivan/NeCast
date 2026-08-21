from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Sequence

import msprime
import numpy as np

from .config import (
    DEMOGRAPHY_TYPE_IDS,
    NE_BOUNDS,
    NE_LOG10_BOUNDS,
    NE_LOG10_STRESS_BOUNDS,
    NE_STRESS_BOUNDS,
    GENERATOR_RECIPE_IDS,
    GENERATOR_RECIPE_HISTORY_FAMILY_OVERRIDES,
    GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE,
    HISTORY_FAMILY_BY_DEMOGRAPHY_TYPE,
    Config,
)
from .ne_templates import empirical_ne_templates_available, load_empirical_ne_templates
from .stdpopsim_templates import (
    choose_stdpopsim_template,
    stdpopsim_templates_available,
)
from .utils import (
    bin_average_log10_ne,
    demography_generation_grid,
    loguniform,
    piecewise_eval,
    time_grid,
)


SamplerResult = tuple[list[float], list[float], dict]
DemoSampler = Callable[[np.random.Generator, Config, np.ndarray, np.ndarray], SamplerResult]


# Where a recipe puts the trajectory as a whole, in log10 effective size.
#
# Every recipe used to carry its own hard-coded range and they had all been
# chosen the same way: 3,000 to 200,000, 2,000 to 200,000, 3,000 to 120,000.
# Weighted over the mixture that left essentially nothing below ten thousand --
# the lowest four 0.25-decade bands of the mean level held 0.000, 0.000, 0.003
# and 0.008 of the weight -- while the prediction head runs from 100 to a
# million.  The bottom decade of the model's own output range was never
# trained.  Real populations live there: the vaquita's effective size is under
# three thousand, and so is that of most managed and island populations.
#
# 2.4 to 5.6 leaves four tenths of a decade of headroom at each end for the
# trajectory's own excursion before it reaches the declared bounds.
LEVEL_LOG10_RANGE = (2.4, 5.6)
LEVEL_NE_RANGE = (10.0 ** LEVEL_LOG10_RANGE[0], 10.0 ** LEVEL_LOG10_RANGE[1])

# Recipes whose public type depends on the trajectory they happened to draw.
# A recipe whose public label is a property of the drawn trajectory: whether a
# run of size changes adds up to an expansion or a contraction is not
# something the sampler decides.
_DYNAMIC_DIRECTION_RECIPES = frozenset(
    {"size_event", "multi_event", "exponential", "event_on_background"}
)


DEMO_PROB_FIELDS = [
    # Background trajectory
    ("constant", "p_constant"),
    ("spectral_walk", "p_spectral_walk"),
    # Events, parameterised continuously rather than named
    ("size_event", "p_size_event"),
    ("multi_event", "p_multi_event"),
    ("exponential", "p_exponential"),
    ("sawtooth", "p_sawtooth"),
    ("literature_sawtooth", "p_literature_sawtooth"),
    ("bottleneck", "p_bottleneck"),
    ("event_on_background", "p_event_on_background"),
    ("ooa_chain", "p_ooa_chain"),
    # Anchored on histories somebody else published
    ("empirical_human_ne_template", "p_empirical_human_ne_template"),
    ("stdpopsim_single_pop", "p_stdpopsim_single_pop"),
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
    stress: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    """Final tidy-up, clipped to the band the sampler declared.

    Applying the stress band here would open it for every sampler.  That
    happened once while consolidating these bounds: the regular samplers hold
    [2.0, 6.0] inside their own construction, and a stress-band clip here
    reopened the door to [1.7, 6.3], after which 2,000 draws reported the same
    support for both groups -- exactly what two bands exist to prevent.
    """

    bounds = NE_STRESS_BOUNDS if stress else NE_BOUNDS
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
        out_values.append(float(np.clip(v, *bounds)))
        out_growth_rates.append(growth_rate)
        last = t
    out_values[0] = float(np.clip(out_values[0], *bounds))
    if not np.all(np.isfinite(out_growth_rates)):
        raise ValueError("demography growth rates must be finite")
    return out_breaks, out_values, out_growth_rates


def resolve_demography_type(
    generator_recipe: str,
    values: Sequence[float],
    meta: dict,
) -> str:
    """Map a fine-grained generator recipe to its public scientific type."""
    if generator_recipe in _DYNAMIC_DIRECTION_RECIPES:
        # These have no fixed public type: whether a run of events adds up to
        # an expansion, a contraction or neither is a property of the drawn
        # trajectory, not of the sampler.
        present_ne = float(values[0])
        ancient_ne = float(values[-1])
        if present_ne > ancient_ne * 1.05:
            return "expansion"
        if present_ne * 1.05 < ancient_ne:
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
    # ``event_severity`` has exactly one definition -- the smallest effective
    # size the trajectory reaches over the largest -- and it is computed here,
    # once, for every recipe.  It used to be computable here and also
    # overridable by the sampler, and five samplers took the override under
    # three different conventions: bottleneck-over-present in three of them,
    # founder-over-ancient in a fourth, founder-over-present in a fifth.
    # Worse, they wrote the value they drew rather than the one they realised,
    # so wherever the effective size hit its floor the number was wrong -- 37%
    # of ``recent_bottleneck_extreme`` rows, at a declared median depth of
    # 0.0032 against a realised 0.0076.
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
    n0 = loguniform(rng, *LEVEL_NE_RANGE)
    return [], [n0], {
        "control_type": "constant",
        "has_recent_event": False,
        "has_ancient_event": False,
    }



# ---------------------------------------------------------------------------
# The spectral walk
# ---------------------------------------------------------------------------
#
# One sampler in place of the four random walks that used to carry 34.5% of
# rows, plus ``near_constant`` and the two stress walks.
#
# Measured against ``empirical_human_ne_template``, the one anchor in the
# mixture built from inferred human histories -- turning points 6/9/13, total
# variation 1.68/2.65/3.46, range 0.78/1.27/1.78 in log10 Ne -- those four
# produced a bimodal union.  Fifteen percent of rows sat near 6 turning points
# and 19.5% near 19-23, with nothing at 9, which is where the human anchor is;
# the union's range median was 0.68 against the anchor's 1.27.  A ridge
# classifier separates them (0.733 against 0.25 chance), so they were not
# duplicates -- but they occupied two corners of the (roughness x amplitude)
# plane and left the other two empty, and the crowded corner, three code paths
# and a fifth of all rows, was high-frequency and low-amplitude.  That is the
# regime where a curve wiggling fast around a level puts almost nothing into
# the coalescent length spectrum, so predicting the level and ignoring the
# structure is the winning strategy.  The mixture was teaching the shortcut.
#
# All of them were also defined in bin index -- they stepped ``cfg.time_bins``
# times and placed knots at a fixed fraction of that count -- so changing the
# generation grid changed their distribution instead of resolving it.
#
# This one is a band-limited Fourier series on continuous log time.  Roughness
# is the spectral slope, amplitude is an explicit range in log10 Ne, and the
# recent/ancient asymmetry the fourth sampler encoded in a decaying step size
# is a tilt on the envelope -- three continuous axes spanning everything the
# discrete samplers did, and the corners they missed.  Because the curve exists
# before the grid does, the same parameters mean the same history at 64, 96 or
# 128 bins, and the scalars recorded in the metadata reconstruct it exactly at
# any of them.

SPECTRAL_MODE_RANGE = (6, 49)
SPECTRAL_SLOPE_RANGE = (0.6, 2.0)
SPECTRAL_TILT_RANGE = (-1.2, 1.2)
SPECTRAL_LEVEL_RANGE = LEVEL_LOG10_RANGE
# Range in log10 Ne, lognormal on the anchor's own median and spread.  The cap
# sits below the width of ``NE_LOG10_BOUNDS`` so the curve never reaches the
# clip: a clipped curve grows a flat plateau, which is a shape the sampler did
# not choose and the model would have to learn as if it had.
SPECTRAL_RANGE_MEDIAN = 1.27
SPECTRAL_RANGE_LOG_SD = 0.36
SPECTRAL_RANGE_CAP = 3.2
# A minority of draws are nearly flat.  The lognormal body reaches 0.55 at its
# first percentile and no further, so without this the mixture would have no
# "almost constant but not constant" histories at all -- which is what the
# retired ``near_constant`` sampler supplied, at a range median of 0.04.
# Keeping it as a mixture component rather than a separate recipe leaves the
# amplitude axis unbroken.
SPECTRAL_QUIET_PROBABILITY = 0.08
SPECTRAL_QUIET_RANGE = (0.02, 0.55)
# The stress band.  One switch rather than the two stress random-walk samplers
# it replaces, which differed from each other only in roughness -- the same
# axis the spectral slope already spans, at 12 turning points against 23.
SPECTRAL_STRESS_RANGE_CAP = 4.2
SPECTRAL_STRESS_PROBABILITY = 0.035
# Safety valve on the one combination the two independent axes can produce and
# nature does not: fully rough at full amplitude.  It binds above the 97th
# percentile, so it shapes the tail without moving the body.
SPECTRAL_TV_CAP = 9.0
# Samples per bin used to turn the analytic curve into bin averages.  Averaging
# is the anti-aliasing: at 48 generation bins the Nyquist limit is 24 modes and
# the series carries up to 48, so the modes above it have to average away
# inside a bin rather than fold back onto the ones below.
SPECTRAL_OVERSAMPLE = 16


def spectral_walk_curve(
    edges: np.ndarray,
    seed: int,
    slope: float,
    n_modes: int,
    tilt: float,
    ne_range: float,
    level: float,
) -> np.ndarray:
    """Bin-averaged log10 Ne for one spectral walk.

    Separate from the sampler because these arguments are exactly what the
    metadata stores: a row can be replayed onto any grid without the dataset
    that produced it.
    """

    log_edges = np.log(np.asarray(edges, dtype=np.float64))
    widths = np.diff(log_edges)
    offsets = (np.arange(SPECTRAL_OVERSAMPLE, dtype=np.float64) + 0.5) / SPECTRAL_OVERSAMPLE
    dense = log_edges[:-1, None] + offsets[None, :] * widths[:, None]
    u = (dense - log_edges[0]) / (log_edges[-1] - log_edges[0])

    coeff_rng = np.random.default_rng(int(seed))
    modes = np.arange(1, int(n_modes) + 1, dtype=np.float64)[:, None, None]
    cos_c = coeff_rng.normal(size=int(n_modes))[:, None, None]
    sin_c = coeff_rng.normal(size=int(n_modes))[:, None, None]
    phase = modes * np.pi * u[None, :, :]
    curve = (modes ** -float(slope) * (cos_c * np.cos(phase) + sin_c * np.sin(phase))).sum(0)

    # Tilt before scaling: the envelope decides how the variation is split
    # between recent and ancient time, the scaling decides how much there is.
    curve = (curve - curve.mean()) * np.exp(float(tilt) * (u - 0.5))
    realised = float(curve.max() - curve.min())
    curve = curve * (float(ne_range) / max(realised, 1e-12))
    total_variation = float(np.abs(np.diff(curve.reshape(-1))).sum())
    if total_variation > SPECTRAL_TV_CAP:
        curve = curve * (SPECTRAL_TV_CAP / total_variation)
    # Centre on the midrange rather than the mean so the level below is the
    # midpoint of the realised curve and the bounds check is exact.
    curve = curve - 0.5 * (curve.max() + curve.min())
    return curve.mean(axis=1) + float(level)


def spectral_walk_level(
    rng: np.random.Generator, ne_range: float, stress: bool = False
) -> float:
    """Draw a level that keeps the whole curve inside ``NE_LOG10_BOUNDS``.

    Moving the level is the alternative to clipping the curve.  At the widest
    ranges the bounds leave no freedom at all and the level is forced to the
    middle, which is the honest answer: a history spanning 3.2 decades inside a
    4.0-decade window has nowhere else to sit.
    """

    bounds = NE_LOG10_STRESS_BOUNDS if stress else NE_LOG10_BOUNDS
    half = 0.5 * float(ne_range)
    low, high = bounds[0] + half, bounds[1] - half
    drawn = float(rng.uniform(*SPECTRAL_LEVEL_RANGE))
    if low > high:
        return 0.5 * (bounds[0] + bounds[1])
    return float(np.clip(drawn, low, high))


def _spectral_walk(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    seed = int(rng.integers(0, 2**62))
    n_modes = int(rng.integers(*SPECTRAL_MODE_RANGE))
    slope = float(rng.uniform(*SPECTRAL_SLOPE_RANGE))
    tilt = float(rng.uniform(*SPECTRAL_TILT_RANGE))
    stress = bool(rng.random() < SPECTRAL_STRESS_PROBABILITY)
    cap = SPECTRAL_STRESS_RANGE_CAP if stress else SPECTRAL_RANGE_CAP
    if rng.random() < SPECTRAL_QUIET_PROBABILITY:
        ne_range = float(loguniform(rng, *SPECTRAL_QUIET_RANGE))
    else:
        ne_range = float(
            min(SPECTRAL_RANGE_MEDIAN * np.exp(rng.normal(0.0, SPECTRAL_RANGE_LOG_SD)), cap)
        )
    level = spectral_walk_level(rng, ne_range, stress=stress)
    log10_ne = spectral_walk_curve(edges, seed, slope, n_modes, tilt, ne_range, level)
    breaks, values = _bin_aligned_values(log10_ne, edges)
    return breaks, values, {
        "spectral_seed": int(seed),
        "spectral_n_modes": int(n_modes),
        "spectral_slope": float(slope),
        "spectral_tilt": float(tilt),
        "spectral_ne_range": float(ne_range),
        "spectral_level": float(level),
        "stress_scenario": bool(stress),
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "has_recent_event": False,
        "has_ancient_event": False,
    }





# The Zigzag_1S14 character rather than its arithmetic: alternations spaced
# about fourfold apart, each about a decade deep.  The version this replaces
# perturbed the published array of six breakpoints, so the cycle count never
# moved -- measured on the label, the turning points came out 3/4/5 on every
# draw.  Building the ladder from a free number of half cycles keeps the
# character and frees the count.
LITERATURE_SAWTOOTH_HALF_CYCLES = (4, 13)
LITERATURE_SAWTOOTH_ONSET = (20.0, 3_000.0)
LITERATURE_SAWTOOTH_STEP_RATIO = 4.0        # the published spacing
LITERATURE_SAWTOOTH_STEP_JITTER = 0.40      # log-normal sigma, per step
LITERATURE_SAWTOOTH_SWING_DECADES = 1.0     # the published half-cycle depth
LITERATURE_SAWTOOTH_SWING_JITTER = 0.32     # log-normal sigma, per half cycle
LITERATURE_SAWTOOTH_CONTRAST = (0.45, 1.9)


def _literature_sawtooth(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    """The Zigzag_1S14 character, redrawn.

    ``sawtooth`` draws a free zigzag and covers a wide shape space; this draws
    the published one and covers its neighbourhood.  Those are different jobs,
    and folding the second into the first was a mistake: measured over six
    thousand free draws the closest came to 0.199 RMSE of the published shape
    while two ordinary draws sit 0.31 apart, so the literature shape lived in
    the prior's tail.  Widening the free sawtooth made it worse rather than
    better -- more axes buy coverage at the cost of density anywhere in
    particular -- which is the argument for a second recipe rather than a wider
    first one.

    What is preserved is the character: a run of fast alternations, spaced
    about fourfold apart in time, each about a decade deep, ancient ones slower
    than recent ones.  Everything else is drawn -- how many cycles, where they
    start, and every individual spacing and depth.

    Freeing the cycle count costs density on the published curve itself, and
    there is no setting that recovers it: with the count free, only about one
    draw in nine has the six breakpoints that curve has, so no amount of
    tightening the jitter brings the hit rate back.  That is the right trade
    anyway.  ``canonical_zigzag_1s14`` already puts six hundred rows of the
    exact curve in front of the model; what this recipe is for is the shape
    class around it, and a classifier separates it from a free sawtooth at
    0.820 against a 0.500 chance rate, so the class is a real one.
    """

    t_max = float(edges[-1])
    n_half = int(rng.integers(*LITERATURE_SAWTOOTH_HALF_CYCLES))
    onset = float(
        loguniform(rng, max(LITERATURE_SAWTOOTH_ONSET[0], cfg.min_time), LITERATURE_SAWTOOTH_ONSET[1])
    )
    steps = np.log(LITERATURE_SAWTOOTH_STEP_RATIO) + rng.normal(
        0.0, LITERATURE_SAWTOOTH_STEP_JITTER, size=n_half
    )
    ladder = np.log(onset) + np.concatenate(([0.0], np.cumsum(np.abs(steps))))
    breaks = np.unique(
        np.clip(np.exp(ladder), max(float(cfg.min_time), 1.0), t_max * 0.9)
    ).astype(np.float64)
    if breaks.size < 2:
        breaks = np.asarray([onset, min(onset * 4.0, t_max * 0.9)], dtype=np.float64)

    contrast = float(rng.uniform(*LITERATURE_SAWTOOTH_CONTRAST))
    swings = LITERATURE_SAWTOOTH_SWING_DECADES * contrast * np.exp(
        rng.normal(0.0, LITERATURE_SAWTOOTH_SWING_JITTER, size=breaks.size)
    )
    # Alternate, starting either way.
    signs = np.where(np.arange(breaks.size) % 2 == 0, 1.0, -1.0)
    if rng.random() < 0.5:
        signs = -signs
    level = float(np.log10(loguniform(rng, *LEVEL_NE_RANGE)))
    log_profile = np.clip(level + signs * swings * 0.5, *NE_LOG10_BOUNDS)

    boundary_values = [float(10.0**value) for value in log_profile]
    present_ne = float(
        np.clip(10.0 ** (level - signs[0] * swings[0] * 0.5), *NE_BOUNDS)
    )
    values = [present_ne, *boundary_values]
    growth_rates = [0.0]
    for left_t, right_t, left_ne, right_ne in zip(
        breaks[:-1], breaks[1:], boundary_values[:-1], boundary_values[1:]
    ):
        growth_rates.append(float(np.log(left_ne / right_ne) / (right_t - left_t)))
    growth_rates.append(0.0)

    return breaks.tolist(), values, {
        "literature_shaped_family": True,
        "source_literature_model_id": "Zigzag_1S14",
        "demography_function": "piecewise_exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "sawtooth_half_cycles": int(breaks.size),
        "sawtooth_onset": float(onset),
        "sawtooth_log_ne_contrast": float(contrast),
        "zigzag_rate_change_count": int(breaks.size),
        "knot_count": int(len(values)),
        "has_recent_event": bool(breaks[0] <= 5_000.0),
        "has_ancient_event": bool(breaks[-1] >= 50_000.0),
    }


# ---------------------------------------------------------------------------
# An event on a moving background
# ---------------------------------------------------------------------------
#
# The one hole left in the mixture, and it is a hole in the joint rather than
# in any margin.  Measured across the other eleven recipes, 28.4% of rows carry
# a sharp event -- a single step of more than half a decade -- and 41.0% carry
# a background that is genuinely moving, six or more turns after that step is
# removed.  Only 2.7% carry both, against the 11.6% independence would give.
# Every recipe is one or the other by construction: ``bottleneck`` and
# ``size_event`` leave a background with a median of one turn, and
# ``spectral_walk`` and the templates leave a largest step of 0.14 to 0.19.
#
# Nature does not separate them.  A population does not sit at a constant size
# waiting for a bottleneck; it drifts, and then something happens.  This is
# also the row a shortcut fails on from either side: a model that has learned
# to predict the level misses the event, and one that has learned to find the
# single event misses the background.

EVENT_BACKGROUND_RANGE = (0.30, 1.60)
EVENT_BACKGROUND_EVENTS = (1, 3)
EVENT_BACKGROUND_ONSET = (50.0, 0.6)       # absolute floor, fraction of the grid
EVENT_BACKGROUND_DURATION_RATIO = (0.05, 8.0)
EVENT_BACKGROUND_DEPTH = (0.4, 2.5)        # decades, sign drawn


def _event_on_background(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    seed = int(rng.integers(0, 2**62))
    n_modes = int(rng.integers(*SPECTRAL_MODE_RANGE))
    slope = float(rng.uniform(*SPECTRAL_SLOPE_RANGE))
    tilt = float(rng.uniform(*SPECTRAL_TILT_RANGE))
    # A quieter background than ``spectral_walk`` draws, so the event stays
    # legible on top of it rather than being one wiggle among many.
    background = float(loguniform(rng, *EVENT_BACKGROUND_RANGE))
    level = spectral_walk_level(rng, background)
    curve = spectral_walk_curve(edges, seed, slope, n_modes, tilt, background, level)

    bin_starts = np.asarray(edges[:-1], dtype=np.float64)
    depths: list[float] = []
    times: list[float] = []
    for _ in range(int(rng.integers(*EVENT_BACKGROUND_EVENTS))):
        start = float(
            loguniform(rng, EVENT_BACKGROUND_ONSET[0], t_max * EVENT_BACKGROUND_ONSET[1])
        )
        end = start * (1.0 + float(loguniform(rng, *EVENT_BACKGROUND_DURATION_RATIO)))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        depth = sign * float(loguniform(rng, *EVENT_BACKGROUND_DEPTH))
        covered = (bin_starts >= start) & (bin_starts < end)
        if not covered.any():
            # Shorter than the bin it lands in.  Give it the one bin it starts
            # in rather than dropping it: an event too brief for the grid to
            # resolve is a real history, and the label is what the grid can say
            # about it.
            index = int(np.searchsorted(bin_starts, start, side="right") - 1)
            covered[max(index, 0)] = True
        curve = curve + np.where(covered, depth, 0.0)
        depths.append(float(depth))
        times.append(float(start))

    curve = np.clip(curve, *NE_LOG10_BOUNDS)
    breaks, values = _bin_aligned_values(curve, edges)
    return breaks, values, {
        "spectral_seed": int(seed),
        "spectral_n_modes": int(n_modes),
        "spectral_slope": float(slope),
        "spectral_tilt": float(tilt),
        "spectral_ne_range": float(background),
        "spectral_level": float(level),
        "event_count": int(len(depths)),
        "event_time": float(min(times)),
        "event_log_depths": depths,
        "event_permanent": False,
        "bin_aligned_demography": True,
        "knot_count": int(len(values)),
        "has_recent_event": bool(min(times) <= 5_000.0),
        "has_ancient_event": bool(max(times) >= 50_000.0),
    }


# ---------------------------------------------------------------------------
# Bottlenecks
# ---------------------------------------------------------------------------
#
# ``size_event`` already spans this: a bottleneck is a negative depth with a
# finite duration recovering near where it started.  But spanning a shape and
# training on it are different things.  Measured over four thousand draws, only
# 12.8% of ``size_event`` rows are a drop of half a decade or more that ends
# and comes back -- 2.05% of the whole mixture -- and 0.64% are a drop of more
# than a decade and a half that does.  The bottleneck is the most studied event
# in the field and the one a method is most often asked about, and it was
# getting one row in fifty.
#
# This draws from the same space as ``size_event``, tilted onto that corner.
# The overlap is deliberate; the alternative is a broader prior, which by
# construction puts less density everywhere rather than more density here.

BOTTLENECK_ONSET = (50.0, 0.5)             # absolute floor, fraction of the grid
BOTTLENECK_DURATION_RATIO = (0.02, 3.0)    # of the onset time
BOTTLENECK_DEPTH_DECADES = (0.4, 3.0)
BOTTLENECK_RECOVERY_LOG = 0.6              # how far the recovery may miss by


def _bottleneck(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    present = loguniform(rng, *LEVEL_NE_RANGE)
    start = float(loguniform(rng, BOTTLENECK_ONSET[0], t_max * BOTTLENECK_ONSET[1]))
    duration = start * float(loguniform(rng, *BOTTLENECK_DURATION_RATIO))
    end = float(min(start + duration, t_max * 0.92))
    if end <= start * 1.02:
        end = min(start * 1.02, t_max * 0.92)

    depth = float(loguniform(rng, *BOTTLENECK_DEPTH_DECADES))
    bottom = float(np.clip(present * 10.0**-depth, *NE_BOUNDS))
    # The ancient size sits near the present one: that is what makes it a
    # bottleneck rather than a step change, and it is the axis the retired
    # recipes pinned -- one to within a factor of two, another far above.
    ancient = float(
        np.clip(
            present * 10.0 ** rng.uniform(-BOTTLENECK_RECOVERY_LOG, BOTTLENECK_RECOVERY_LOG),
            *NE_BOUNDS,
        )
    )
    return [start, end], [present, bottom, ancient], {
        "event_count": 1,
        "event_time": float(start),
        "event_duration": float(end - start),
        "event_log_depths": [float(np.log10(bottom / present))],
        "event_permanent": False,
        "bottleneck_recovery_log10": float(np.log10(ancient / present)),
        "has_recent_event": bool(start <= 5_000.0),
        "has_ancient_event": bool(end >= 50_000.0),
    }


# Zigzag_1S14, written out by hand from Schiffels & Durbin (2014).  The mixture
# carried three encodings of this one history: this table, the scrm command from
# the SMC++ paper, and the frozen stdpopsim catalogue -- which this table is
# bit-identical to, and which the scrm command differs from by 5.4e-4 at most on
# the 64-bin label, not one bin of it above the 0.0019 noise floor.  Three
# encodings of one history is the same defect as ``zigzag`` aliasing
# ``zigzag_strong``, one level up.
#
# One survives as a recipe, and it is this one: it is the id the SMC literature
# reports against, so a tight error bar on it is what makes our numbers
# comparable to published ones, and that needs more replicates than a share of
# the catalogue's weight buys.  The catalogue draw excludes it in return, so
# exactly one code path produces this history.
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
    }


# Not a sampler.  The scrm command from the SMC++ Online Methods is the same
# model as the table above, published twice: identical growth rates in all six
# epochs, one breakpoint apart by 0.03%, and sizes differing only by where the
# rounding accumulates.  It is kept because a second independent statement of
# the published values is what makes the first one checkable.
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
    arr = np.clip(arr, NE_LOG10_BOUNDS[0], NE_LOG10_BOUNDS[1])

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
    }























# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
#
# One sampler in place of eight.  ``recent_bottleneck``, ``single_bottleneck``,
# ``recent_founder_recovery``, ``recent_bottleneck_extreme``,
# ``founder_recovery_extreme``, ``expansion``, ``contraction`` and
# ``ancient_event`` all describe the same object -- a size change at some time,
# lasting some while, of some depth, after which the population settles at some
# level -- and differed only in which corner of that four-dimensional space
# they were pinned to.  Measured pairwise on the 64-bin label they were barely
# separable: 0.619 for the first two against a 0.500 chance rate, and 0.667 to
# 0.786 across the rest, on 14.4% of all rows.
#
# The axis that had been hardest hidden is the last one.  ``recent_bottleneck``
# fixed the post-event level at 0.5 to 2 times the pre-event level, so the
# population always came back to roughly where it started; ``founder_recovery``
# fixed it far above; ``expansion`` and ``contraction`` never came back at all,
# by giving the event no end.  All three are one continuous parameter, and a
# duration that runs past the end of the grid turns the event into a step
# change without needing a separate sampler for it.
#
# Two of the four measured defects in the retired samplers are also fixed here.
# ``event_severity`` now records what the trajectory actually does rather than
# what was drawn: the old code wrote the drawn value while clipping the
# realised size at a floor, which on ``recent_bottleneck_extreme`` misreported
# 37% of rows -- a declared median depth of 0.0032 arriving in the label as
# 0.0076.  And the field means the same thing in every recipe now; it used to
# mean bottleneck-over-present in three samplers, founder-over-ancient in a
# fourth and founder-over-present in a fifth.

SIZE_EVENT_TIME = (50.0, 0.6)          # absolute floor, fraction of the grid
SIZE_EVENT_DURATION_RATIO = (0.05, 30.0)
SIZE_EVENT_LOG_DEPTH = (0.15, 3.0)     # magnitude in log10 Ne; the sign is drawn
SIZE_EVENT_LOG_ASYMMETRY = 1.0         # post-event level, log10, either way
SIZE_EVENT_PRESENT_NE = LEVEL_NE_RANGE


def _event_epochs(
    rng: np.random.Generator,
    edges: np.ndarray,
    n_events: int,
) -> tuple[list[float], list[float], dict]:
    """Lay out ``n_events`` size changes on the log-time axis.

    Times are drawn on a log scale and sorted, because the target grid is
    logarithmic: drawn uniformly, almost every event would land in the ancient
    half where a single bin is tens of thousands of generations wide.
    """

    t_max = float(edges[-1])
    present = loguniform(rng, *SIZE_EVENT_PRESENT_NE)
    times = np.sort(
        np.array(
            [
                loguniform(rng, SIZE_EVENT_TIME[0], t_max * SIZE_EVENT_TIME[1])
                for _ in range(n_events)
            ]
        )
    )

    breaks: list[float] = []
    values: list[float] = [present]
    depths: list[float] = []
    durations: list[float] = []
    level = float(present)
    last = 0.0
    for start in times:
        start = float(max(start, last * 1.05, SIZE_EVENT_TIME[0]))
        if start >= t_max * 0.95:
            break
        sign = 1.0 if rng.random() < 0.5 else -1.0
        depth = sign * float(loguniform(rng, *SIZE_EVENT_LOG_DEPTH))
        asymmetry = float(rng.uniform(-SIZE_EVENT_LOG_ASYMMETRY, SIZE_EVENT_LOG_ASYMMETRY))
        duration = start * float(loguniform(rng, *SIZE_EVENT_DURATION_RATIO))
        end = start + duration

        event_level = float(np.clip(level * 10.0**depth, *NE_BOUNDS))
        breaks.append(start)
        values.append(event_level)
        depths.append(float(np.log10(event_level / level)))
        last = start

        if end >= t_max * 0.95:
            # The event never ends inside the observable window, which is what
            # a permanent size change is.  Recording it this way rather than as
            # a separate recipe keeps the duration axis unbroken.
            durations.append(float("inf"))
            level = event_level
            break
        settled = float(np.clip(level * 10.0**asymmetry, *NE_BOUNDS))
        breaks.append(float(max(end, start * 1.05)))
        values.append(settled)
        durations.append(float(end - start))
        level = settled
        last = float(max(end, start * 1.05))

    realised = [
        float(np.log10(values[index + 1] / values[index]))
        for index in range(len(values) - 1)
    ]
    meta = {
        "event_count": int(len(depths)),
        "event_time": float(breaks[0]) if breaks else 0.0,
        "event_duration": float(durations[0]) if durations and np.isfinite(durations[0]) else 0.0,
        "event_log_depths": [float(x) for x in depths],
        "event_permanent": bool(durations and not np.isfinite(durations[-1])),
        "has_recent_event": bool(breaks and breaks[0] <= 5_000.0),
        "has_ancient_event": bool(breaks and breaks[-1] >= 50_000.0),
    }
    return breaks, values, meta


def _size_event(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    breaks, values, meta = _event_epochs(rng, edges, 1)
    return breaks, values, {**meta, "event_family": "single"}


def _multi_event(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    """Two to four events, each with its own independent sign.

    This replaces ``three_epoch``, ``ancient_recent_compound``,
    ``ancient_recent_conflict``, ``serial_founder`` and
    ``serial_founder_extreme``.  The conflict case -- a recent and an ancient
    event pulling opposite ways, which had its own sampler at 0.3% -- is what
    independent signs produce on their own about half the time, and the serial
    founder is what a run of same-signed events with finite durations is.
    """

    n_events = int(rng.integers(2, 5))
    breaks, values, meta = _event_epochs(rng, edges, n_events)
    return breaks, values, {**meta, "event_family": "multiple"}


# ---------------------------------------------------------------------------
# Exponential
# ---------------------------------------------------------------------------
#
# ``continuous_exponential_growth``, ``continuous_exponential_decline`` and
# ``rapid_recent_growth_extreme`` are one sampler with a signed rate.  What is
# worth keeping is not the three names but the mechanism: this is the only
# family that uses msprime's own ``growth_rate``, so the trajectory is
# genuinely exponential in linear time rather than a staircase approximating
# one.  That is a real difference in ``demography_function``, not a redundancy,
# and the bin-averaged label is different from any piecewise-constant history.

EXPONENTIAL_ONSET = (100.0, 100_000.0)
EXPONENTIAL_LOG_FACTOR = (0.3, 2.2)
EXPONENTIAL_OLDER_NE = LEVEL_NE_RANGE


def _exponential(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    start = float(min(loguniform(rng, *EXPONENTIAL_ONSET), t_max * 0.8))
    older = loguniform(rng, *EXPONENTIAL_OLDER_NE)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    log_factor = sign * float(loguniform(rng, *EXPONENTIAL_LOG_FACTOR))
    present = float(np.clip(older * 10.0**log_factor, *NE_BOUNDS))
    breaks, values, growth_rates = _native_exponential_transition(present, older, start)
    return breaks, values, {
        "demography_function": "exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "event_time": float(start),
        "exponential_log10_factor": float(np.log10(present / older)),
        "exponential_direction": "growth" if present > older else "decline",
        "has_recent_event": bool(start <= 5_000.0),
        "has_ancient_event": bool(start >= 50_000.0),
    }


# ---------------------------------------------------------------------------
# Sawtooth
# ---------------------------------------------------------------------------
#
# ``oscillating_mild``, ``zigzag_strong`` and ``zigzag`` were one function with
# a boolean.  ``zigzag`` was worse than that: ``_zigzag_legacy`` returned
# ``_oscillating(..., strong=True)`` unchanged, so it was bit-identical to
# ``zigzag_strong`` on all 200 seeds tested, and a classifier separating the
# two scored 0.283 -- below the 0.500 chance rate, which is what happens when a
# model memorises a curve in training and meets the same curve labelled the
# other way in test.
#
# The boolean is now a continuous strength, so the mixture fills the range
# between mild and strong instead of sitting on its two ends.

SAWTOOTH_TURNS = (4, 19)
SAWTOOTH_MILD_FACTOR = (1.2, 2.5)
# The ceiling reaches past a tenfold swing rather than stopping just short of
# it.  Zigzag_1S14 alternates by exactly a decade every half cycle, so with a
# ceiling of eight that shape sat at the very edge of the prior.
SAWTOOTH_STRONG_FACTOR = (2.5, 16.0)
# Where the oscillation lives.  The version this replaces laid its turning
# points on a geomspace from a hundred generations to eight tenths of the grid,
# every single time: measured over two thousand draws the first turn fell
# between 86 and 117 generations and the last between 341,000 and 450,000, so
# every sawtooth spanned the same three and a half decades and only the number
# of turns inside it varied.  A real population that oscillated for a while and
# then settled, or that has only been oscillating recently, was not in the
# support at all.
SAWTOOTH_ONSET = (50.0, 0.03)          # absolute floor, fraction of the grid
SAWTOOTH_EXTENT = (15.0, 12_000.0)     # end over start; Zigzag_1S14 spans 1,024x
# How evenly the turns are spaced.  Gap lengths in log time come from a
# Dirichlet whose concentration this sets: high concentration is the periodic
# sawtooth the literature models are, low concentration is irregular.  The old
# spacing was a fixed 12% lognormal jitter on a geometric ladder, which measured
# a coefficient of variation of 0.19 and could not reach either end.
SAWTOOTH_REGULARITY_CONCENTRATION = (1.0, 60.0)
# How alike the swings are.  Regularity above governs *when* the turns happen;
# this governs whether they are all the same size.  The published sawtooths are
# uniform in both -- and needing regular gaps and uniform swings and near-peak
# strength and no drift all at once made the literature shape a one-in-a-
# thousand draw: over six thousand samples the closest came to 0.199 RMSE of
# it, against 0.312 between two ordinary draws.  A prior that reaches a shape
# only in its tail has not trained on that shape.
SAWTOOTH_SWING_UNIFORMITY = (0.0, 1.0)
# A log-linear trend across the swings, so the oscillation can grow or decay
# through time instead of holding one amplitude for the whole history.
SAWTOOTH_AMPLITUDE_DRIFT = 1.6


def _sawtooth(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    strength = float(rng.uniform(0.0, 1.0))
    factor_lo = SAWTOOTH_MILD_FACTOR[0] + strength * (SAWTOOTH_STRONG_FACTOR[0] - SAWTOOTH_MILD_FACTOR[0])
    factor_hi = SAWTOOTH_MILD_FACTOR[1] + strength * (SAWTOOTH_STRONG_FACTOR[1] - SAWTOOTH_MILD_FACTOR[1])

    n_turns = int(rng.integers(*SAWTOOTH_TURNS))
    onset = float(
        loguniform(rng, max(SAWTOOTH_ONSET[0], cfg.min_time), t_max * SAWTOOTH_ONSET[1])
    )
    end = float(min(onset * loguniform(rng, *SAWTOOTH_EXTENT), t_max * 0.9))
    if end <= onset * 1.5:
        end = min(onset * 1.5, t_max * 0.9)
    # Gap lengths in log time from a Dirichlet: one concentration parameter
    # takes the ladder from irregular to periodic without changing anything
    # else about the curve.
    concentration = float(loguniform(rng, *SAWTOOTH_REGULARITY_CONCENTRATION))
    uniformity = float(rng.uniform(*SAWTOOTH_SWING_UNIFORMITY))
    gaps = rng.dirichlet(np.full(max(n_turns - 1, 1), concentration))
    ladder = np.log(onset) + np.concatenate(([0.0], np.cumsum(gaps))) * (
        np.log(end) - np.log(onset)
    )
    breaks = np.unique(np.exp(ladder)).astype(np.float64).tolist()

    present_ne = loguniform(rng, *LEVEL_NE_RANGE)
    boundary_values = [present_ne]
    direction = 1 if rng.random() < 0.5 else -1
    # Grow or decay the swings across the history rather than holding one
    # amplitude throughout.
    drift = float(rng.uniform(-SAWTOOTH_AMPLITUDE_DRIFT, SAWTOOTH_AMPLITUDE_DRIFT))
    # Draw every swing first, then pull them toward their common mean by the
    # uniformity: at one the curve alternates by a single size throughout, as
    # the published models do, and at zero each swing is independent.
    drawn = np.array(
        [loguniform(rng, factor_lo, factor_hi) for _ in range(max(len(breaks) - 1, 1))]
    )
    log_drawn = np.log(drawn)
    log_swings = (1.0 - uniformity) * log_drawn + uniformity * log_drawn.mean()
    for index in range(1, len(breaks)):
        position = index / max(len(breaks) - 1, 1)
        factor = float(
            np.exp(log_swings[index - 1] * np.exp(drift * (position - 0.5)))
        )
        candidate = boundary_values[-1] * factor if direction > 0 else boundary_values[-1] / factor
        next_ne = float(np.clip(candidate, NE_BOUNDS[0], NE_BOUNDS[1]))
        if np.isclose(next_ne, boundary_values[-1], rtol=1e-12, atol=1e-9):
            candidate = boundary_values[-1] / factor if direction > 0 else boundary_values[-1] * factor
            next_ne = float(np.clip(candidate, NE_BOUNDS[0], NE_BOUNDS[1]))
        direction = -1 if next_ne > boundary_values[-1] else 1
        boundary_values.append(next_ne)

    values = [present_ne, *boundary_values]
    growth_rates = [0.0]
    for left_t, right_t, left_ne, right_ne in zip(
        breaks[:-1], breaks[1:], boundary_values[:-1], boundary_values[1:]
    ):
        growth_rates.append(float(np.log(left_ne / right_ne) / (right_t - left_t)))
    growth_rates.append(0.0)
    return breaks, values, {
        "oscillation_strength_fraction": float(strength),
        "oscillation_onset": float(onset),
        "oscillation_end": float(end),
        "oscillation_regularity": float(concentration),
        "oscillation_swing_uniformity": float(uniformity),
        "oscillation_amplitude_drift": float(drift),
        "demography_function": "piecewise_exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "zigzag_rate_change_count": int(len(breaks)),
        "zigzag_internal_extrema_count": int(max(0, len(breaks) - 2)),
        "knot_count": int(len(values)),
    }


# ---------------------------------------------------------------------------
# The out-of-Africa chain
# ---------------------------------------------------------------------------
#
# The one shape the mixture could not previously produce.  A serial founder
# history is a chain: an ancestral population expands, a subset leaves and is
# held small for thousands of generations, and only then grows -- and the
# phases are ordered and their sizes related, which is exactly what
# ``ancient_recent_compound`` threw away by drawing its two events
# independently.  Real inferred human histories all have this shape, and it is
# the shape a demographic inference method is most often asked about.
#
# The ranges bracket the published single-population human models rather than
# reproducing any one of them: ancestral sizes near ten thousand, an ancient
# expansion some thousands of generations back, a bottleneck one to a few
# thousand generations back holding at a few percent to a third of the
# pre-bottleneck size, and recent exponential growth over the last few hundred
# to fifteen hundred generations.  Widening every one of them well past the
# human values is deliberate: the point is the chain, not the species.

OOA_ANCESTRAL_NE = LEVEL_NE_RANGE
OOA_EXPANSION_FACTOR = (1.2, 6.0)
OOA_BOTTLENECK_FACTOR = (0.02, 0.5)
OOA_GROWTH_FACTOR = (2.0, 400.0)
# The chain is an ordering of phases, not a date.  Ranges bracketing only the
# human values pinned the trajectory's minimum to bins 6 through 24 of 64 on
# nine draws out of ten, which teaches a model where to expect a bottleneck
# rather than how to find one.  These span most of the grid; the ordering
# constraint below is what makes it the chain.
OOA_EXPANSION_TIME = (2_000.0, 250_000.0)
OOA_BOTTLENECK_TIME = (400.0, 40_000.0)
OOA_GROWTH_ONSET = (60.0, 6_000.0)


def _ooa_chain(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    t_max = float(edges[-1])
    ancestral = loguniform(rng, *OOA_ANCESTRAL_NE)
    expansion_factor = loguniform(rng, *OOA_EXPANSION_FACTOR)
    pre_bottleneck = float(np.clip(ancestral * expansion_factor, *NE_BOUNDS))
    # The bottleneck has to end up below the ancestral level, not merely below
    # the expanded one.  Without this the ancient expansion can outrun a mild
    # bottleneck and the trajectory's lowest point lands at the ancient end,
    # which is a decline, not a founder event -- and the chain is the only
    # reason this recipe exists.  Human values are far inside the constraint:
    # an ancestral size near 7,300 against an out-of-Africa bottleneck near
    # 1,860.
    bottleneck_ceiling = min(OOA_BOTTLENECK_FACTOR[1], 0.9 / expansion_factor)
    bottleneck_factor = loguniform(
        rng, min(OOA_BOTTLENECK_FACTOR[0], bottleneck_ceiling * 0.5), bottleneck_ceiling
    )
    bottleneck_ne = float(np.clip(pre_bottleneck * bottleneck_factor, *NE_BOUNDS))
    growth_factor = loguniform(rng, *OOA_GROWTH_FACTOR)
    present = float(np.clip(bottleneck_ne * growth_factor, *NE_BOUNDS))

    # The chain only means anything if the phases stay in order.
    growth_onset = float(loguniform(rng, *OOA_GROWTH_ONSET))
    bottleneck_time = float(max(loguniform(rng, *OOA_BOTTLENECK_TIME), growth_onset * 1.6))
    expansion_time = float(
        min(max(loguniform(rng, *OOA_EXPANSION_TIME), bottleneck_time * 1.6), t_max * 0.85)
    )
    if not (growth_onset < bottleneck_time < expansion_time):
        raise ValueError("out-of-Africa phases are out of order")

    # Recent growth is native exponential; everything older is piecewise.
    growth_rate = float(np.log(present / bottleneck_ne) / growth_onset)
    breaks = [growth_onset, bottleneck_time, expansion_time]
    values = [present, bottleneck_ne, pre_bottleneck, ancestral]
    growth_rates = [growth_rate, 0.0, 0.0, 0.0]
    return breaks, values, {
        "demography_function": "piecewise_exponential_linear_time",
        "native_continuous_growth": True,
        "demography_growth_rates": growth_rates,
        "ooa_ancestral_ne": float(ancestral),
        "ooa_expansion_factor": float(expansion_factor),
        "ooa_bottleneck_factor": float(bottleneck_ne / pre_bottleneck),
        "ooa_growth_factor": float(present / bottleneck_ne),
        "ooa_bottleneck_duration": float(bottleneck_time - growth_onset),
        "event_time": float(bottleneck_time),
        "event_duration": float(bottleneck_time - growth_onset),
        "knot_count": int(len(values)),
        "has_recent_event": True,
        "has_ancient_event": bool(expansion_time >= 50_000.0),
    }


# ---------------------------------------------------------------------------
# The held-out evaluation catalogue
# ---------------------------------------------------------------------------


def _stdpopsim_single_pop(rng: np.random.Generator, cfg: Config, edges: np.ndarray, mids: np.ndarray) -> SamplerResult:
    """One of the eighteen frozen stdpopsim single-population models, verbatim.

    No randomisation of the trajectory at all: the map, the mutation rate and
    the sequencing noise still differ from row to row, so a run of these is a
    set of data replicates of a fixed known history.  That is what makes them
    usable as evidence -- the shape was chosen by other people for other
    reasons, so error on it is not error on our own prior redrawn.

    These rows carry ``held_out_evaluation`` so the split can route them out of
    the training gradient.  Training on them would not be wrong so much as
    pointless: it would destroy the only measurement they exist to provide.
    """

    template = choose_stdpopsim_template(rng, cfg)
    if template is None:
        raise ValueError(
            "stdpopsim single-population templates are unavailable; run "
            "scripts/build_stdpopsim_single_pop.py or set p_stdpopsim_single_pop to 0"
        )
    return list(template.breaks), list(template.values), {
        "demography_function": "piecewise_exponential_linear_time",
        "native_continuous_growth": any(float(r) != 0.0 for r in template.growth_rates),
        "demography_growth_rates": list(template.growth_rates),
        "held_out_evaluation": True,
        "literature_demography_anchor": True,
        "literature_model_id": template.model_id,
        "stdpopsim_species": template.species,
        "stdpopsim_model_id": template.model_id,
        "stdpopsim_population": template.population,
        "stdpopsim_generation_time_years": float(template.generation_time_years),
        "stdpopsim_ne_rescale": float(template.ne_rescale),
        "knot_count": int(len(template.values)),
        "has_recent_event": bool(template.breaks and min(template.breaks) <= 5_000.0),
        "has_ancient_event": bool(template.breaks and max(template.breaks) >= 50_000.0),
    }


DEMO_SAMPLERS: dict[str, DemoSampler] = {
    "constant": _constant,
    "spectral_walk": _spectral_walk,
    "size_event": _size_event,
    "multi_event": _multi_event,
    "exponential": _exponential,
    "sawtooth": _sawtooth,
    "literature_sawtooth": _literature_sawtooth,
    "bottleneck": _bottleneck,
    "event_on_background": _event_on_background,
    "ooa_chain": _ooa_chain,
    "empirical_human_ne_template": _empirical_human_ne_template,
    "stdpopsim_single_pop": _stdpopsim_single_pop,
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


def sample_custom_demography(rng: np.random.Generator, cfg: Config) -> tuple[msprime.Demography, np.ndarray, dict]:
    target_edges, _ = time_grid(cfg)
    generation_edges, generation_mids = demography_generation_grid(cfg)
    # Existing samplers use ``cfg.time_bins`` for their internal array lengths.
    # Give them a view whose time_bins matches the fixed generation grid so the
    # v3 latent histories and RNG stream remain bit-for-bit unchanged while the
    # supervised target moves from 48/32 to 64 bins.
    sampler_cfg = dataclasses.replace(
        cfg,
        time_bins=cfg.demography_generation_bins,
    )
    generator_recipe = choose_generator_recipe(rng, sampler_cfg)
    sampler = DEMO_SAMPLERS.get(generator_recipe)
    if sampler is None:
        raise ValueError(f"Unknown demography sampler: {generator_recipe}")

    breaks, values, meta = sampler(
        rng,
        sampler_cfg,
        generation_edges,
        generation_mids,
    )
    is_stress = "stress" in str(generator_recipe)
    breaks, values, growth_rates = _clean_demography(
        breaks,
        values,
        meta.get("demography_growth_rates"),
        stress=is_stress,
    )
    y = bin_average_log10_ne(target_edges, breaks, values, growth_rates)
    demography_type = resolve_demography_type(generator_recipe, values, meta)
    meta = {
        **meta,
        **summarize_demography(
            generation_edges,
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
        "demography_type": demography_type,
        "demography_type_id": DEMOGRAPHY_TYPE_IDS[demography_type],
        "demography_mixture_version": cfg.demography_mixture_version,
        "demography_breaks": [float(x) for x in breaks],
        "demography_values": [float(x) for x in values],
        "demography_growth_rates": [float(x) for x in growth_rates],
        "target_scale": "log10_Ne",
        "target_aggregation": "log_time_bin_average",
        "target_time_bins": int(cfg.time_bins),
        "demography_generation_bins": int(cfg.demography_generation_bins),
    }
    return build_demography(breaks, values, growth_rates), y, meta
