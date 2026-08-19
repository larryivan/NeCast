from __future__ import annotations

import math

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


# Calibration targets, all measured from the maps in ``maps/`` rather than
# chosen.  Reference windows: chr1 20-40 Mb and chr20 10-26 Mb, both clear of
# a centromere.
#
# Recombination, log-rate standard deviation of the length-weighted mean on a
# regular grid:
#
#                    1 kb    10 kb   100 kb    1 Mb
#   deCODE 2019      3.821   3.472    1.898   0.712
#   HapMap II        2.088   1.964    1.381   0.615
#   pyrho CEU        1.321   1.371    1.137   0.688
#
# deCODE runs high at fine scale because 18.4% of its 1 kb cells are exactly
# zero -- no crossover was observed there in 4.5M meioses -- which is counting
# sparsity rather than biology, so the fine-scale target comes from the two LD
# maps and the megabase target from all three, which agree there.
#
# Autocorrelation of log rate on 10 kb cells is 0.65 at one cell, 0.50 at two,
# 0.27 at ten, and then flattens near 0.08 out to megabases.  That plateau is
# the reason the background here is a separate slow component rather than a
# single decaying one: a map is a megabase-scale landscape with kilobase-scale
# hotspots on it, not one process.
#
# Hotspots: chr20 between 10 and 26 Mb carries 393 runs above ten times the
# mean, a median run width of 466 bp, and 1.76% of sequence in them.
BACKGROUND_CELL_BP = 2_000
# 24.6 runs per megabase were measured, but a run merges hotspots that abut,
# so the generating density is higher than the counted one.
HOTSPOT_PER_MB = 60.0
HOTSPOT_WIDTH_BP = (200.0, 1_200.0)
HOTSPOT_FACTOR = (15.0, 200.0)
COLDSPOT_COUNT_RANGE = (0, 4)   # rng.integers is half-open
COLDSPOT_WIDTH_BP = (50_000.0, 500_000.0)
COLDSPOT_FACTOR = (0.1, 0.5)
# Drawn per row rather than fixed.  The measured amplitude differs more
# between maps (1.30 for pyrho against 1.76 for HapMap II, at 10 kb) than
# between windows of one map (1.72 to 2.02 within HapMap II), so a single
# constant would be both wrong and a fingerprint: every synthetic row would
# carry the same log-rate spread to nine decimal places, which is a label the
# model can condition on.  These brackets straddle the measured range.
RECOMBINATION_BROAD_SD = (0.50, 1.50)
RECOMBINATION_CORRELATION_BP = (200_000.0, 3_000_000.0)
#
# Mutation, Roulette on chr1 20-60 Mb in 1 kb bins: log-rate sd 0.132, with a
# megabase p90/p10 of 1.12, and autocorrelation 0.39 at one kilobase falling
# to a 0.08 plateau beyond 200 kb.  So the variance is mostly local -- that is
# sequence context, which is uncorrelated between kilobases -- sitting on a
# very mild megabase drift.  There are no mutation hotspots: fine-scale
# mutation variation is a property of single sites, which a binary mutation
# model cannot represent at all.
MUTATION_BROAD_SD = (0.03, 0.07)
MUTATION_CORRELATION_BP = (1_000_000.0, 10_000_000.0)
MUTATION_LOCAL_SD = (0.08, 0.16)
# Not white noise: Roulette's autocorrelation is 0.39 at one kilobase and
# 0.25 at ten, so the local term carries a correlation length of its own.
MUTATION_LOCAL_CORRELATION_BP = (1_500.0, 12_000.0)
MUTATION_CELL_BP = 1_000
# Retains the ceiling the previous generator applied as a literal.
MAX_SYNTHETIC_RATE_MULTIPLIER = 200.0
# A published map reports the average rate in bins of its own, which both
# hides the true breakpoints and partly smooths a hotspot it did not resolve.
OBSERVED_CELL_BP = (1_000.0, 20_000.0)


def _ornstein_uhlenbeck(
    rng: np.random.Generator, n_cells: int, cell_bp: float, correlation_bp: float
) -> np.ndarray:
    """A unit-variance process whose correlation decays over a real distance.

    The previous background was a random walk indexed by segment number, and
    segment widths were spacings of uniform draws.  Measured on that version,
    the log rate moved by 0.0105 across a segment at the 25th width percentile
    and 0.0097 across one at the 75th -- the same jump over four times the
    sequence -- so the correlation length in base pairs was an artefact of the
    segmentation rather than a property of the map.

    Nothing standardises the result: the realised amplitude has to vary from
    row to row, because forcing it constant is what turned the old generator's
    spread into a fingerprint.
    """

    if n_cells <= 0:
        raise ValueError("a background needs at least one cell")
    retention = math.exp(-float(cell_bp) / float(correlation_bp))
    innovation = math.sqrt(max(1.0 - retention * retention, 0.0))
    noise = rng.normal(0.0, 1.0, size=n_cells)
    out = np.empty(n_cells, dtype=np.float64)
    out[0] = noise[0]
    for index in range(1, n_cells):
        out[index] = retention * out[index - 1] + innovation * noise[index]
    return out


def _apply_spans(
    edges: np.ndarray, rates: np.ndarray, spans: list[tuple[float, float, float]]
) -> np.ndarray:
    """Multiply exactly the sequence each span covers.

    The edges are already in the grid, so a span multiplies the cells it was
    drawn over and no others.  Applying a multiplier to whole pre-existing
    segments instead made a feature as wide as the segment containing it: 39
    to 156 kb for one drawn at 1 to 20 kb.
    """

    out = rates.copy()
    for left, right, factor in spans:
        covered = (edges[:-1] >= left) & (edges[1:] <= right)
        out = np.where(covered, out * factor, out)
    return out


def _observed_view(
    rng: np.random.Generator,
    edges: np.ndarray,
    rates: np.ndarray,
    baseline: float,
    obs_noise_sigma: float,
    hotspot_missing_prob: float,
    has_hotspots: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """What a published map of this region would look like.

    Re-binned onto a grid of its own, so the true breakpoints are not visible
    in it.  That mattered once the feature edges entered the true grid: the
    observed map was returned on the same edges, which handed over every
    hotspot position exactly and noised only the intensities.  A real
    published map is uncertain about where the hotspots are -- PRDM9 alleles
    differ between populations and move them -- not just about how strong they
    are.

    The lognormal multiplier is centred so its mean is one.  With a zero
    location it has mean ``exp(sigma^2/2)``, so the observed integral sat 3.2%
    above the truth in every row at the default sigma, and ``nw_log10``
    divides the segregating count by exactly that integral.
    """

    length = float(edges[-1])
    cell = float(loguniform(rng, *OBSERVED_CELL_BP))
    obs_edges = np.arange(0.0, length + cell, cell)
    obs_edges[-1] = length
    if len(obs_edges) < 2:
        obs_edges = np.array([0.0, length])

    cumulative = np.concatenate(([0.0], np.cumsum(rates * np.diff(edges))))
    integrated = np.interp(obs_edges, edges, cumulative)
    obs = np.diff(integrated) / np.diff(obs_edges)

    obs = obs * rng.lognormal(
        -0.5 * obs_noise_sigma**2, obs_noise_sigma, size=obs.shape
    )
    # Only a map with hotspots can miss one.  Applying this to the mutation
    # map, which has none, rewrote the top ventile of a smooth drift to model
    # a phenomenon that does not exist there.
    if has_hotspots and hotspot_missing_prob > 0 and len(obs) > 20:
        high = obs > np.quantile(obs, 0.95)
        miss = high & (rng.random(obs.shape) < hotspot_missing_prob)
        if miss.any():
            obs[miss] = baseline * rng.uniform(0.5, 2.0, size=int(miss.sum()))
            # Replacing the top ventile with something near the baseline
            # removes rate, so without this the observed integral lands
            # systematically low -- 0.9679 of the truth at the default
            # dropout, against 1.0016 with it switched off.  The lognormal
            # above was centred for exactly this reason; missing a hotspot
            # should change where a published map puts its rate, not how much
            # of it there is in total.
            widths = np.diff(obs_edges)
            scale = (rates * np.diff(edges)).sum() / (obs * widths).sum()
            obs = obs * scale
    obs = np.clip(obs, baseline * 1e-6, baseline * 500)
    return obs_edges, obs


def _length_weighted_mean(rates: np.ndarray, pos: np.ndarray) -> float:
    """The mean that decides how many events actually occur.

    Normalising by the unweighted segment mean instead leaves the realised
    rate off by however much rate and segment width correlate, and they do:
    a feature applied by interval overlap preferentially hits wide segments.
    Measured on the previous generator, the realised recombination rate
    averaged 1.655 times the configured baseline, spread 1.06 to 2.36.
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
    length = int(cfg.seq_len)
    cell = MUTATION_CELL_BP if is_mut else BACKGROUND_CELL_BP
    cell = float(min(cell, max(1.0, length)))

    broad_sd = float(loguniform(rng, *(MUTATION_BROAD_SD if is_mut else RECOMBINATION_BROAD_SD)))
    correlation_bp = float(
        loguniform(
            rng,
            *(MUTATION_CORRELATION_BP if is_mut else RECOMBINATION_CORRELATION_BP),
        )
    )

    n_cells = max(1, int(math.ceil(length / cell)))
    edges = np.concatenate((np.arange(n_cells, dtype=np.float64) * cell, [float(length)]))
    log_rate = broad_sd * _ornstein_uhlenbeck(rng, n_cells, cell, correlation_bp)

    local_sd = 0.0
    if is_mut:
        # Sequence context: real at the scale of single sites, and between
        # kilobases it is nearly independent -- Roulette's autocorrelation is
        # 0.39 at one kilobase.  A binary mutation model has no sequence to
        # condition on, so it enters as an uncorrelated cell-level term.
        local_sd = float(loguniform(rng, *MUTATION_LOCAL_SD))
        local_correlation = float(loguniform(rng, *MUTATION_LOCAL_CORRELATION_BP))
        log_rate = log_rate + local_sd * _ornstein_uhlenbeck(
            rng, n_cells, cell, local_correlation
        )

    spans: list[tuple[float, float, float]] = []
    hot_widths: list[float] = []
    cold_widths: list[float] = []
    if not is_mut:
        for _ in range(int(rng.poisson(HOTSPOT_PER_MB * length / 1_000_000.0))):
            width = _sample_bp_width(rng, length, *HOTSPOT_WIDTH_BP)
            centre = float(rng.uniform(0, length))
            hot_widths.append(width)
            spans.append(
                (
                    max(0.0, centre - width / 2.0),
                    min(float(length), centre + width / 2.0),
                    float(loguniform(rng, *HOTSPOT_FACTOR)),
                )
            )
        for _ in range(int(rng.integers(*COLDSPOT_COUNT_RANGE))):
            width = _sample_bp_width(rng, length, *COLDSPOT_WIDTH_BP)
            centre = float(rng.uniform(0, length))
            cold_widths.append(width)
            spans.append(
                (
                    max(0.0, centre - width / 2.0),
                    min(float(length), centre + width / 2.0),
                    float(rng.uniform(*COLDSPOT_FACTOR)),
                )
            )

    # Feature edges join the grid so each span multiplies exactly what it
    # covers.  The background is carried across by locating each refined cell
    # in the regular grid it came from, which leaves it a function of physical
    # position rather than of the refinement.
    pos = np.unique(
        np.concatenate([edges] + [np.array([a, b]) for a, b, _ in spans])
    ) if spans else edges
    pos = pos[(pos >= 0.0) & (pos <= float(length))]
    owner = np.clip(
        np.searchsorted(edges, 0.5 * (pos[:-1] + pos[1:]), side="right") - 1,
        0,
        n_cells - 1,
    )
    rates = baseline * np.exp(log_rate[owner])
    rates = _apply_spans(pos, rates, spans)

    # Clip first, then normalise.  The other order lets the ceiling remove
    # rate the normalisation has already counted, which left the realised mean
    # at 0.810 of the baseline.
    rates = np.clip(rates, baseline * 1e-5, baseline * MAX_SYNTHETIC_RATE_MULTIPLIER)
    rates = rates * (baseline / _length_weighted_mean(rates, pos))

    obs_pos, obs = _observed_view(
        rng,
        pos,
        rates,
        baseline,
        obs_noise_sigma,
        hotspot_missing_prob,
        has_hotspots=bool(hot_widths),
    )
    meta = {
        "synthetic_map_segments": int(len(rates)),
        "background_cell_bp": float(cell),
        "map_is_mutation": bool(is_mut),
        "hotspot_model": "physical_grid_exact_span",
        "hotspot_count": int(len(hot_widths)),
        "coldspot_count": int(len(cold_widths)),
        "hotspot_width_bp_mean": float(np.mean(hot_widths)) if hot_widths else 0.0,
        "coldspot_width_bp_mean": float(np.mean(cold_widths)) if cold_widths else 0.0,
        "observed_noise_sigma": float(obs_noise_sigma),
        "broad_sd": broad_sd,
        "local_sd": local_sd,
        "correlation_bp": correlation_bp,
        "observed_segments": int(len(obs)),
        "realised_rate_over_baseline": float(
            _length_weighted_mean(rates, pos) / baseline
        ),
    }
    return pos, rates.astype(np.float64), obs_pos, obs.astype(np.float64), meta


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
    empirical_noise = getattr(cfg, "empirical_map_noise", obs_noise)
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
            rng, empirical.rec_maps[chrom], cfg.seq_len, empirical_noise, cfg.baseline_rec
        )
        mut_pos, mut_rate, obs_mut_pos, obs_mut_rate, mut_meta = slice_piecewise_map(
            rng, empirical.mut_maps[chrom], cfg.seq_len, empirical_noise * 0.6, cfg.baseline_mu
        )
        meta.update({
            "map_type": "empirical_slice",
            "map_mode": "empirical_slice",
            "recomb_mode": "empirical_slice",
            "mu_mode": "empirical_slice",
            "observed_noise_sigma": empirical_noise,
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
