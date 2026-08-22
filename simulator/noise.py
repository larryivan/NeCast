from __future__ import annotations

import numpy as np

from .config import Config
from .utils import loguniform


def noise_params_for_sample(rng: np.random.Generator, cfg: Config, source: str) -> dict:
    if source == "clean_constant_map":
        return {
            "profile": "clean",
            "genotype_error": 0.0,
            "false_variant_rate_per_bp": 0.0,
            "missing_rate": 0.0,
            "phase_switch_rate_per_het": 0.0,
            "phase_switch_model": "diploid_pair_switch_per_heterozygous_site",
        }
    ge = rng.uniform(cfg.genotype_error_min, cfg.genotype_error_max)
    fv = loguniform(
        rng, cfg.false_variant_rate_per_bp_min, cfg.false_variant_rate_per_bp_max
    )
    mr = rng.uniform(cfg.missing_rate_min, cfg.missing_rate_max)
    ps_het = rng.uniform(
        cfg.phase_switch_rate_per_het_min, cfg.phase_switch_rate_per_het_max
    )
    return {
        "profile": "mild_noise",
        "genotype_error": float(ge),
        "false_variant_rate_per_bp": float(fv),
        "missing_rate": float(mr),
        "phase_switch_rate_per_het": float(ps_het),
        "phase_switch_model": "diploid_pair_switch_per_heterozygous_site",
    }


def apply_phase_switch_errors(
    G: np.ndarray,
    positions_bp: np.ndarray,
    rng: np.random.Generator,
    switch_rate_per_het: float,
) -> tuple[np.ndarray, dict]:
    """Swap the two haplotypes of a diploid downstream of a switch point.

    Switches are drawn per heterozygous site rather than per base pair.  That
    is where the error actually arises: a phasing algorithm decides the
    relative phase of consecutive heterozygous sites, and at a homozygous site
    there is no decision to get wrong.  Published switch-error rates for
    statistical phasing are quoted the same way, 0.5% to 2% of adjacent
    heterozygous pairs.

    The previous version drew from a Poisson process in physical distance at
    up to 0.05 per megabase, which gave 1.7 switches per diploid over twenty
    megabases where the literature implies about a hundred, and which spent
    switches on sequence with no heterozygous site to express them.

    This is not dead code even though the six-state encoding the current model
    reads is a dosage and dosage is invariant under the swap.  What is stored
    is sixty-four packed haplotypes, so the swap changes the data on disk --
    56,414 haplotype calls at the old rate -- and any phased consumer sees it.
    """

    X = np.empty_like(G, dtype=np.uint8, order="K", subok=True)
    np.greater(G, 0, out=X)
    stats = {
        "phase_switch_count": 0,
        "phase_switch_pair_count": 0,
        "phaseable_pair_count": int(X.shape[1] // 2),
        "phase_switch_heterozygous_sites": 0,
    }
    if X.shape[0] < 2 or X.shape[1] < 2 or switch_rate_per_het <= 0:
        return X, stats
    if len(np.asarray(positions_bp)) != X.shape[0]:
        raise ValueError("positions_bp length must match genotype variant count")

    probability = float(np.clip(switch_rate_per_het, 0.0, 1.0))
    for left in range(0, X.shape[1] - 1, 2):
        heterozygous = X[:, left] != X[:, left + 1]
        n_het = int(heterozygous.sum())
        stats["phase_switch_heterozygous_sites"] += n_het
        if n_het < 2:
            continue
        # A switch can only occur between two heterozygous sites, so the draws
        # live on those sites and the state is carried forward to every site
        # after the first one that flipped -- homozygous sites included, since
        # the haplotypes are swapped there too even though it is unobservable.
        het_index = np.flatnonzero(heterozygous)
        switches = rng.random(n_het) < probability
        switches[0] = False
        if not switches.any():
            continue
        parity_at_het = np.cumsum(switches) % 2 == 1
        # Carry each heterozygous site's parity forward to the sites after it.
        # Homozygous sites are swapped too: the swap is unobservable there, but
        # the haplotypes really are exchanged from the switch point onward.
        preceding = np.searchsorted(het_index, np.arange(X.shape[0]), side="right") - 1
        forward = np.where(preceding >= 0, parity_at_het[np.clip(preceding, 0, None)], False)
        stats["phase_switch_count"] += int(switches.sum())
        stats["phase_switch_pair_count"] += 1
        a = X[forward, left].copy()
        X[forward, left] = X[forward, left + 1]
        X[forward, left + 1] = a
    return X, stats


def flat_index_dtype(n_cells: int) -> np.dtype:
    """The narrowest unsigned type that can address ``n_cells`` genotype calls.

    A row's missing calls are stored as flat offsets into its (variants,
    haplotypes) matrix.  That product is small for a short sequence and a small
    sample and passes four billion for a long one with many haplotypes, so the
    width is a property of the row rather than of the format.  Which width was
    used travels with the data in ``missing_storage``.
    """

    return np.dtype(np.uint32 if n_cells <= np.iinfo(np.uint32).max else np.uint64)


def _sample_flat_indices(rng: np.random.Generator, n: int, p: float) -> np.ndarray:
    dtype = flat_index_dtype(max(int(n), 0))
    if n <= 0 or p <= 0:
        return np.zeros(0, dtype=dtype)
    k = int(rng.binomial(int(n), float(np.clip(p, 0.0, 1.0))))
    if k == 0:
        return np.zeros(0, dtype=dtype)
    idx = rng.choice(int(n), size=k, replace=False)
    idx.sort()
    return idx.astype(dtype, copy=False)


def add_false_variants(
    G: np.ndarray,
    positions_bp: np.ndarray,
    rng: np.random.Generator,
    sequence_length: int,
    rate_per_bp: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Insert sites that are not real, the way a caller does.

    Every other noise channel here rearranges variants that exist.  Real
    sequencing manufactures them: a miscalled base at a position where nobody
    actually varies produces a site that looks like a variant and is carried by
    exactly one sample, because two independent errors at one position are rare.

    Without this the mixture gets the low end of the frequency spectrum
    backwards.  Flipping a call at a singleton site moves it up far more often
    than down -- sixty-three zeros against one one -- so the existing noise
    *depletes* singletons, by a measured 12.2%, while real data is enriched in
    them.  Singletons sit on the shortest terminal branches, which is the most
    recent slice of the genealogy, so a model calibrated on a depleted spectrum
    reads a real dataset's surplus as a larger recent effective size.

    The rate spans four orders of magnitude on purpose.  What survives calling
    is a property of the pipeline, not of biology -- a hard-filtered callset and
    a raw one differ by orders of magnitude -- so the point is to cover the
    range and record which was drawn, not to pick one.
    """

    if rate_per_bp <= 0.0 or sequence_length <= 0:
        return G, positions_bp, {"false_variant_count": 0}
    expected = float(rate_per_bp) * float(sequence_length)
    count = int(rng.poisson(expected))
    if count <= 0:
        return G, positions_bp, {"false_variant_count": 0}

    n_haplotypes = G.shape[1] if G.size else 0
    if n_haplotypes <= 0:
        return G, positions_bp, {"false_variant_count": 0}
    new_positions = rng.uniform(0.0, float(sequence_length), size=count)
    rows = np.zeros((count, n_haplotypes), dtype=G.dtype if G.size else np.uint8)
    rows[np.arange(count), rng.integers(0, n_haplotypes, size=count)] = 1

    positions = np.concatenate([np.asarray(positions_bp, dtype=np.float64), new_positions])
    genotypes = np.concatenate(
        [G if G.size else np.zeros((0, n_haplotypes), dtype=rows.dtype), rows]
    )
    order = np.argsort(positions, kind="stable")
    return genotypes[order], positions[order], {"false_variant_count": int(count)}


def apply_genotype_noise(
    G: np.ndarray,
    positions_bp: np.ndarray,
    rng: np.random.Generator,
    noise: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if G.size == 0:
        stats = {"phase_switch_count": 0, "phase_switch_pair_count": 0, "phaseable_pair_count": 0}
        return G.astype(np.uint8), np.zeros(0, dtype=flat_index_dtype(G.size)), stats

    ge = float(noise.get("genotype_error", 0.0))
    mr = float(noise.get("missing_rate", 0.0))
    phase_rate = float(noise.get("phase_switch_rate_per_het", 0.0))
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
