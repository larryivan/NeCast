from __future__ import annotations

import msprime
import numpy as np

from .config import Config
from .empirical_maps import EmpiricalMapStore, slice_piecewise_map
from .utils import loguniform


def constant_map(length: int, rate: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([0.0, float(length)], dtype=np.float64), np.array([float(rate)], dtype=np.float64)


def make_rate_map(pos: np.ndarray, rates: np.ndarray) -> msprime.RateMap:
    pos = np.asarray(pos, dtype=np.float64).copy()
    rates = np.asarray(rates, dtype=np.float64)
    if len(pos) == 0 or len(rates) == 0:
        raise ValueError("RateMap requires at least one interval")
    pos[0] = 0.0
    if len(pos) != len(rates) + 1:
        pos = pos[: len(rates) + 1]
        if len(pos) < len(rates) + 1:
            pos = np.concatenate([pos, [pos[-1] + 1.0]])
    rates = np.clip(rates, 0.0, None)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rates)):
        raise ValueError("RateMap contains non-finite positions or rates")
    if np.any(np.diff(pos) <= 0):
        raise ValueError("RateMap positions must be strictly increasing")
    if np.nanmax(rates) > 1e-4:
        raise ValueError("RateMap rates are too large; expected per-bp per-generation probabilities")
    return msprime.RateMap(position=pos, rate=rates)


def _segment_range(cfg: Config, is_mut: bool) -> tuple[int, int]:
    lo = max(1, cfg.map_segments_min)
    hi = max(lo, cfg.map_segments_max)
    if is_mut:
        lo = max(16, lo // 2)
        hi = max(lo, hi // 2)
    return lo, hi


def _sample_bp_width(rng: np.random.Generator, length: int, lo: float, hi: float) -> float:
    upper = max(1.0, min(float(hi), float(length)))
    lower = max(1.0, min(float(lo), upper))
    return loguniform(rng, lower, upper)


def _overlapping_intervals(pos: np.ndarray, center: float, width: float) -> np.ndarray:
    left = max(0.0, float(center) - float(width) / 2.0)
    right = min(float(pos[-1]), float(center) + float(width) / 2.0)
    mask = (pos[:-1] < right) & (pos[1:] > left)
    if not mask.any():
        idx = int(np.clip(np.searchsorted(pos, center, side="right") - 1, 0, len(pos) - 2))
        mask[idx] = True
    return mask


# Calibration targets, measured rather than assumed.
#
# Recombination: deCODE 2019 on chr20 between 10 and 26 Mb carries 393 runs
# above ten times the mean rate, a median run width of 466 bp, and 1.76% of
# sequence in them.  Scaled to a twenty-megabase row that is roughly 490
# hotspots.  The generator previously drew 3 to 15 "hotspots" 1 to 20 kb wide,
# and because a multiplier was applied to whole segments their realised width
# was the segment they landed in -- 39 to 156 kb, twenty to seventy times what
# was declared.
HOTSPOT_PER_MB = 24.5
HOTSPOT_WIDTH_BP = (300.0, 2_000.0)
HOTSPOT_FACTOR = (10.0, 100.0)
# Coldspots exist -- recombination deserts are real -- but a hundredfold
# suppression across two megabases is not.  Softened accordingly.
COLDSPOT_COUNT_RANGE = (0, 4)   # rng.integers is half-open
COLDSPOT_WIDTH_BP = (50_000.0, 500_000.0)
COLDSPOT_FACTOR = (0.1, 0.5)
# Mutation: Roulette on chr20 between 10 and 26 Mb gives a megabase-scale
# p90/p10 of 1.12 and a kilobase-scale p90/p10 of 1.34.  Real mutation rate has
# no hotspot architecture at all: its fine-scale variation is sequence context
# at single sites (CpG), which a binary mutation model cannot represent, and
# its regional variation is a smooth 1.2-fold drift.  So the mutation map gets
# the broad walk and nothing else, at a standard deviation calibrated to that
# megabase target.
MUTATION_BROAD_SD = 0.06
RECOMBINATION_BROAD_SD = 0.55
# Retains the ceiling the previous generator applied as a literal.
MAX_SYNTHETIC_RATE_MULTIPLIER = 200.0


def _refined_grid(pos: np.ndarray, features: list[tuple[float, float, float]]) -> np.ndarray:
    """Add every feature edge to the breakpoints.

    Without this a multiplier lands on whole segments, so a feature is as wide
    as whatever segment contains it rather than as wide as it was drawn.  That
    was the single largest source of unrealistic structure in this generator.
    """

    edges = [pos]
    for left, right, _ in features:
        edges.append(np.array([left, right], dtype=np.float64))
    grid = np.unique(np.concatenate(edges))
    return grid[(grid >= pos[0]) & (grid <= pos[-1])]


def _length_weighted_mean(rates: np.ndarray, pos: np.ndarray) -> float:
    """The mean that decides how many events actually occur.

    Normalising by the unweighted segment mean instead leaves the realised
    rate off by however much rate and segment width happen to correlate, and
    they do correlate: a feature applied by interval overlap preferentially
    hits wide segments.  Measured on the previous generator, the realised
    recombination rate averaged 1.655 times the configured baseline, with a
    tenth-to-ninetieth percentile spread of 1.06 to 2.36.
    """

    widths = np.diff(pos)
    total = widths.sum()
    if total <= 0:
        raise ValueError("a rate map must span a positive length")
    return float((np.asarray(rates) * widths).sum() / total)


def synthetic_heterogeneous_map(
    rng: np.random.Generator,
    cfg: Config,
    baseline: float,
    is_mut: bool,
    obs_noise_sigma: float,
    hotspot_missing_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    length = cfg.seq_len
    seg_min, seg_max = _segment_range(cfg, is_mut)
    n_segments = int(rng.integers(seg_min, seg_max + 1))
    internal = np.sort(rng.uniform(0, length, size=n_segments - 1))
    base_pos = np.unique(
        np.concatenate([[0.0], internal, [float(length)]])
    ).astype(np.float64)

    broad_sd = MUTATION_BROAD_SD if is_mut else RECOMBINATION_BROAD_SD
    walk = np.cumsum(rng.normal(0, broad_sd, size=len(base_pos) - 1))
    walk = (walk - walk.mean()) / (walk.std() + 1e-8)
    base_rates = baseline * np.exp(broad_sd * walk)

    # Features are drawn as spans, then the grid is refined so each applies to
    # exactly the span it was drawn with.  Mutation gets none: see the
    # calibration note above.
    features: list[tuple[float, float, float]] = []
    hot_widths: list[float] = []
    cold_widths: list[float] = []
    if not is_mut:
        n_hot = int(rng.poisson(HOTSPOT_PER_MB * length / 1_000_000.0))
        for _ in range(n_hot):
            width = _sample_bp_width(rng, length, *HOTSPOT_WIDTH_BP)
            centre = float(rng.uniform(0, length))
            hot_widths.append(width)
            features.append(
                (
                    max(0.0, centre - width / 2.0),
                    min(float(length), centre + width / 2.0),
                    float(rng.uniform(*HOTSPOT_FACTOR)),
                )
            )
        for _ in range(int(rng.integers(*COLDSPOT_COUNT_RANGE))):
            width = _sample_bp_width(rng, length, *COLDSPOT_WIDTH_BP)
            centre = float(rng.uniform(0, length))
            cold_widths.append(width)
            features.append(
                (
                    max(0.0, centre - width / 2.0),
                    min(float(length), centre + width / 2.0),
                    float(rng.uniform(*COLDSPOT_FACTOR)),
                )
            )

    pos = _refined_grid(base_pos, features)
    # Carry the broad walk onto the refined grid by locating each refined
    # segment inside the base grid it came from.
    midpoints = 0.5 * (pos[:-1] + pos[1:])
    owner = np.clip(
        np.searchsorted(base_pos, midpoints, side="right") - 1,
        0,
        len(base_rates) - 1,
    )
    rates = base_rates[owner]
    for left, right, factor in features:
        covered = (pos[:-1] >= left) & (pos[1:] <= right)
        rates = np.where(covered, rates * factor, rates)

    # Clip first, then normalise.  The other order lets the ceiling remove
    # rate that the normalisation has already counted, so the realised mean
    # lands below the baseline -- measured at 0.810 with the ceiling applied
    # afterwards.  Normalising last makes the realised mean exactly the
    # baseline and can lift the maximum slightly past the nominal ceiling,
    # which is a guard against msprime's 1e-4 limit rather than a bound with
    # meaning of its own: two hundred times the baseline is 2e-6.
    rates = np.clip(
        rates,
        baseline * 1e-5,
        baseline * MAX_SYNTHETIC_RATE_MULTIPLIER,
    )
    rates = rates * (baseline / _length_weighted_mean(rates, pos))

    # A lognormal with zero location has mean exp(sigma^2 / 2), so the observed
    # map integral would sit systematically above the truth -- 3.2% at the
    # default sigma of 0.25.  ``nw_log10`` divides the segregating count by
    # that integral, so the bias lands directly on the baseline every model
    # predicts a residual around.  Shifting the location makes the multiplier
    # mean exactly one.
    obs = rates * rng.lognormal(
        -0.5 * obs_noise_sigma**2, obs_noise_sigma, size=rates.shape
    )
    if hotspot_missing_prob > 0:
        high = obs > np.quantile(obs, 0.95)
        miss = high & (rng.random(obs.shape) < hotspot_missing_prob)
        obs[miss] = baseline * rng.uniform(0.5, 2.0, size=miss.sum())
    obs = np.clip(obs, baseline * 1e-6, baseline * 500)
    meta = {
        "synthetic_map_segments": int(len(pos) - 1),
        "synthetic_base_segments": int(len(base_pos) - 1),
        "map_is_mutation": bool(is_mut),
        "hotspot_model": "refined_grid_exact_span",
        "hotspot_count": int(len(hot_widths)),
        "coldspot_count": int(len(cold_widths)),
        "hotspot_width_bp_mean": float(np.mean(hot_widths)) if hot_widths else 0.0,
        "coldspot_width_bp_mean": float(np.mean(cold_widths)) if cold_widths else 0.0,
        "observed_noise_sigma": float(obs_noise_sigma),
        "broad_sd": float(broad_sd),
        "realised_rate_over_baseline": float(
            _length_weighted_mean(rates, pos) / baseline
        ),
    }
    return pos, rates.astype(np.float64), pos.copy(), obs.astype(np.float64), meta


def choose_source(rng: np.random.Generator, cfg: Config, empirical_available: bool) -> str:
    probs = {
        "clean_constant_map": cfg.p_clean_constant,
        "random_global_mu_r": cfg.p_random_global,
        "custom_heterogeneous": cfg.p_custom_hetero,
        "empirical_map_slice": cfg.p_empirical_slice if empirical_available else 0.0,
        "stdpopsim_anchor": cfg.p_stdpopsim_anchor if cfg.enable_stdpopsim else 0.0,
    }
    total = sum(probs.values())
    if total <= 0:
        return "custom_heterogeneous"
    keys = list(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


def sample_maps_for_source(
    rng: np.random.Generator,
    cfg: Config,
    source: str,
    empirical: EmpiricalMapStore,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    obs_noise = cfg.observed_map_noise
    miss_prob = cfg.hotspot_missing_prob

    meta: dict = {"source_type": source}
    if source == "clean_constant_map":
        rec_pos, rec_rate = constant_map(cfg.seq_len, cfg.baseline_rec)
        mut_pos, mut_rate = constant_map(cfg.seq_len, cfg.baseline_mu)
        obs_rec_pos, obs_rec_rate = rec_pos.copy(), rec_rate.copy()
        obs_mut_pos, obs_mut_rate = mut_pos.copy(), mut_rate.copy()
        meta.update({
            "map_type": "constant_clean",
            "map_mode": "constant_clean",
            "recomb_mode": "constant",
            "mu_mode": "constant",
            "global_rec_rate": cfg.baseline_rec,
            "global_mu_rate": cfg.baseline_mu,
        })
    elif source == "random_global_mu_r":
        rec = cfg.baseline_rec * loguniform(rng, 0.3, 3.0)
        mu = cfg.baseline_mu * loguniform(rng, 0.5, 2.0)
        rec_pos, rec_rate = constant_map(cfg.seq_len, rec)
        mut_pos, mut_rate = constant_map(cfg.seq_len, mu)
        obs_rec_pos, obs_rec_rate = rec_pos.copy(), rec_rate.copy()
        obs_mut_pos, obs_mut_rate = mut_pos.copy(), mut_rate.copy()
        meta.update({
            "map_type": "constant_random_global",
            "map_mode": "constant_random_global",
            "recomb_mode": "constant_random_global",
            "mu_mode": "constant_random_global",
            "global_rec_rate": rec,
            "global_mu_rate": mu,
        })
    elif source == "empirical_map_slice" and empirical.is_available(cfg.seq_len):
        chrom = str(rng.choice(empirical.common_chroms(cfg.seq_len)))
        rec_pos, rec_rate, obs_rec_pos, obs_rec_rate, rec_meta = slice_piecewise_map(
            rng, empirical.rec_maps[chrom], cfg.seq_len, obs_noise, cfg.baseline_rec
        )
        mut_pos, mut_rate, obs_mut_pos, obs_mut_rate, mut_meta = slice_piecewise_map(
            rng, empirical.mut_maps[chrom], cfg.seq_len, obs_noise * 0.6, cfg.baseline_mu
        )
        meta.update({
            "map_type": "empirical_slice",
            "map_mode": "empirical_slice",
            "recomb_mode": "empirical_slice",
            "mu_mode": "empirical_slice",
            "observed_noise_sigma": obs_noise,
            "rec_slice": rec_meta,
            "mut_slice": mut_meta,
        })
    else:
        source = "custom_heterogeneous"
        rec_pos, rec_rate, obs_rec_pos, obs_rec_rate, rec_meta = synthetic_heterogeneous_map(
            rng, cfg, cfg.baseline_rec, is_mut=False, obs_noise_sigma=obs_noise, hotspot_missing_prob=miss_prob
        )
        mut_pos, mut_rate, obs_mut_pos, obs_mut_rate, mut_meta = synthetic_heterogeneous_map(
            rng, cfg, cfg.baseline_mu, is_mut=True, obs_noise_sigma=obs_noise * 0.6, hotspot_missing_prob=miss_prob * 0.5
        )
        meta.update({
            "source_type": source,
            "map_type": "synthetic_heterogeneous",
            "map_mode": "synthetic_heterogeneous",
            "recomb_mode": "synthetic_heterogeneous",
            "mu_mode": "synthetic_heterogeneous",
            "rec_map_meta": rec_meta,
            "mut_map_meta": mut_meta,
        })

    return rec_pos, rec_rate, obs_rec_pos, obs_rec_rate, mut_pos, mut_rate, obs_mut_pos, obs_mut_rate, meta
