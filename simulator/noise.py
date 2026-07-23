from __future__ import annotations

import numpy as np

from .config import Config


def noise_params_for_sample(rng: np.random.Generator, cfg: Config, source: str) -> dict:
    if source == "clean_constant_map":
        return {
            "profile": "clean",
            "genotype_error": 0.0,
            "missing_rate": 0.0,
            "phase_switch_rate_per_mb": 0.0,
            "phase_switch_rate_per_bp": 0.0,
            "phase_switch_model": "diploid_pair_switch",
        }
    ge = rng.uniform(cfg.genotype_error_min, cfg.genotype_error_max)
    mr = rng.uniform(cfg.missing_rate_min, cfg.missing_rate_max)
    ps_mb = rng.uniform(cfg.phase_switch_rate_per_mb_min, cfg.phase_switch_rate_per_mb_max)
    return {
        "profile": "mild_noise",
        "genotype_error": float(ge),
        "missing_rate": float(mr),
        "phase_switch_rate_per_mb": float(ps_mb),
        "phase_switch_rate_per_bp": float(ps_mb) / 1_000_000.0,
        "phase_switch_model": "diploid_pair_switch",
    }


def apply_phase_switch_errors(
    G: np.ndarray,
    positions_bp: np.ndarray,
    rng: np.random.Generator,
    switch_rate_per_bp: float,
) -> tuple[np.ndarray, dict]:
    X = np.empty_like(G, dtype=np.uint8, order="K", subok=True)
    np.greater(G, 0, out=X)
    stats = {
        "phase_switch_count": 0,
        "phase_switch_pair_count": 0,
        "phaseable_pair_count": int(X.shape[1] // 2),
    }
    if X.shape[0] < 2 or X.shape[1] < 2 or switch_rate_per_bp <= 0:
        return X, stats

    positions = np.asarray(positions_bp, dtype=np.float64)
    if len(positions) != X.shape[0]:
        raise ValueError("positions_bp length must match genotype variant count")
    deltas = np.diff(positions, prepend=positions[0])
    switch_prob = 1.0 - np.exp(-float(switch_rate_per_bp) * np.clip(deltas, 0.0, None))
    switch_prob[0] = 0.0

    for left in range(0, X.shape[1] - 1, 2):
        switches = rng.random(X.shape[0]) < switch_prob
        states = (np.cumsum(switches) % 2).astype(bool)
        if not states.any():
            continue
        stats["phase_switch_count"] += int(switches.sum())
        stats["phase_switch_pair_count"] += 1
        a = X[states, left].copy()
        X[states, left] = X[states, left + 1]
        X[states, left + 1] = a
    return X, stats


def _sample_flat_indices(rng: np.random.Generator, n: int, p: float) -> np.ndarray:
    if n <= 0 or p <= 0:
        return np.zeros(0, dtype=np.uint32)
    if n > np.iinfo(np.uint32).max:
        raise ValueError("flat genotype index exceeds uint32 range")
    k = int(rng.binomial(int(n), float(np.clip(p, 0.0, 1.0))))
    if k == 0:
        return np.zeros(0, dtype=np.uint32)
    idx = rng.choice(int(n), size=k, replace=False)
    idx.sort()
    return idx.astype(np.uint32)


def apply_genotype_noise(
    G: np.ndarray,
    positions_bp: np.ndarray,
    rng: np.random.Generator,
    noise: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if G.size == 0:
        stats = {"phase_switch_count": 0, "phase_switch_pair_count": 0, "phaseable_pair_count": 0}
        return G.astype(np.uint8), np.zeros(0, dtype=np.uint32), stats

    ge = float(noise.get("genotype_error", 0.0))
    mr = float(noise.get("missing_rate", 0.0))
    phase_rate = float(noise.get("phase_switch_rate_per_bp", 0.0))
    X, stats = apply_phase_switch_errors(G, positions_bp, rng, phase_rate)
    flat = X.reshape(-1)
    if ge > 0:
        flip_idx = _sample_flat_indices(rng, flat.size, ge)
        flat[flip_idx] = 1 - flat[flip_idx]
    missing_flat_idx = _sample_flat_indices(rng, flat.size, mr)
    if len(missing_flat_idx):
        flat[missing_flat_idx] = 0
    return X, missing_flat_idx, stats


def pack_haplotypes(G: np.ndarray, n_haplotypes: int) -> np.ndarray:
    if G.shape[1] != n_haplotypes:
        raise ValueError(f"Expected H={n_haplotypes}, got {G.shape[1]}")
    packed_input = G if G.dtype == np.uint8 else G.astype(np.uint8)
    return np.packbits(packed_input, axis=1, bitorder="little")
