from __future__ import annotations

import msprime
import numpy as np

from .config import MAX_MSPRIME_RATE, MAX_SYNTHETIC_RATE_MULTIPLIER, Config
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
    if np.nanmax(rates) > MAX_MSPRIME_RATE:
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
    pos = np.unique(np.concatenate([[0.0], internal, [float(length)]])).astype(np.float64)
    n = len(pos) - 1

    broad_sd = 0.25 if is_mut else 0.55
    rw = np.cumsum(rng.normal(0, broad_sd, size=n))
    rw = (rw - rw.mean()) / (rw.std() + 1e-8)
    rates = baseline * np.exp((0.25 if is_mut else 0.55) * rw)

    hot_widths: list[float] = []
    cold_widths: list[float] = []
    if is_mut:
        n_hot = int(rng.integers(1, 5))
        hot_factor = (2, 12)
    else:
        n_hot = int(rng.integers(3, 16))
        hot_factor = (5, 80)
    for _ in range(n_hot):
        center = float(rng.uniform(0, length))
        width = _sample_bp_width(rng, length, 1_000.0, 20_000.0)
        hot_widths.append(width)
        rates[_overlapping_intervals(pos, center, width)] *= rng.uniform(*hot_factor)

    n_cold = int(rng.integers(0, 4))
    for _ in range(n_cold):
        center = float(rng.uniform(0, length))
        width = _sample_bp_width(rng, length, 50_000.0, 2_000_000.0)
        cold_widths.append(width)
        rates[_overlapping_intervals(pos, center, width)] *= rng.uniform(0.01, 0.3)

    rates = rates * (baseline / (rates.mean() + 1e-30))
    rates = np.clip(
        rates,
        baseline * 1e-5,
        baseline * MAX_SYNTHETIC_RATE_MULTIPLIER,
    )

    obs = rates * rng.lognormal(0, obs_noise_sigma, size=rates.shape)
    if hotspot_missing_prob > 0:
        high = obs > np.quantile(obs, 0.95)
        miss = high & (rng.random(obs.shape) < hotspot_missing_prob)
        obs[miss] = baseline * rng.uniform(0.5, 2.0, size=miss.sum())
    obs = np.clip(obs, baseline * 1e-6, baseline * 500)
    meta = {
        "synthetic_map_segments": int(n),
        "map_is_mutation": bool(is_mut),
        "hotspot_model": "bp_interval_overlap",
        "hotspot_count": int(n_hot),
        "coldspot_count": int(n_cold),
        "hotspot_width_bp_mean": float(np.mean(hot_widths)) if hot_widths else 0.0,
        "coldspot_width_bp_mean": float(np.mean(cold_widths)) if cold_widths else 0.0,
        "observed_noise_sigma": float(obs_noise_sigma),
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
        start_min, start_max = empirical.common_slice_start_bounds(chrom, cfg.seq_len)
        shared_start = start_min if start_min == start_max else float(rng.uniform(start_min, start_max))
        shared_end = shared_start + cfg.seq_len
        rec_pos, rec_rate, obs_rec_pos, obs_rec_rate, rec_meta = slice_piecewise_map(
            rng,
            empirical.rec_maps[chrom],
            cfg.seq_len,
            obs_noise,
            cfg.baseline_rec,
            start=shared_start,
            validate_input=False,
        )
        mut_pos, mut_rate, obs_mut_pos, obs_mut_rate, mut_meta = slice_piecewise_map(
            rng,
            empirical.mut_maps[chrom],
            cfg.seq_len,
            obs_noise * 0.6,
            cfg.baseline_mu,
            start=shared_start,
            validate_input=False,
        )
        meta.update({
            "map_type": "empirical_slice",
            "map_mode": "empirical_slice",
            "recomb_mode": "empirical_slice",
            "mu_mode": "empirical_slice",
            "observed_noise_sigma": obs_noise,
            "slice_alignment": "shared_chromosomal_interval",
            "slice_chrom": chrom,
            "slice_start_bp": shared_start,
            "slice_end_bp": shared_end,
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
