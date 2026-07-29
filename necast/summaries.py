"""Summary statistics of one window: site-frequency spectrum, Watterson's
estimate, linkage disequilibrium and identity-by-state tracts.

Genotypes are a `states` array with one row per site and one column per
individual:

    0, 1, 2   number of derived alleles, both alleles called
    3         one allele missing, the other ancestral
    4         one allele missing, the other derived
    5         both alleles missing

Positions are 0-based coordinates inside the window, in ascending order.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache

import numpy as np
from numba import njit

PROJECTED = 32                                   # haplotypes the spectrum is projected to
LD_LAGS = np.array([1, 2, 4, 8, 16, 32, 64])     # site pairs this many sites apart
LD_EDGES = np.logspace(-6.0, -2.0, 25)           # genetic-distance classes, in Morgan
TRACT_CLASSES = 27                               # 0, 1, [2, 4), [4, 8), ..., [2^25, inf) bp
LEAVES = 16                                      # finest regions; coarser ones are their unions
REGION_SIZES = (1, 2, 4, 8, 16)                  # 31 nested regions in all


@dataclass
class Summaries:
    sfs: np.ndarray            # (16,) folded spectrum, projected to 32 haplotypes
    ld: np.ndarray             # (5, 26) r^2 summaries per genetic-distance class
    ibs: np.ndarray            # (7, 31, 27) tract counts per kind, region and length class
    ibs_exposure: np.ndarray   # (31,) callable pair-sites per region
    log10_watterson: float     # Watterson's estimate of log10 Ne


def summarise(positions, states, rec_map, mut_map, length, threads=1):
    """Every summary the network reads, before normalisation."""
    seg = segregating(states)
    pair = pairwise(states)
    pair_positions, pair_states = positions[pair], np.ascontiguousarray(states[pair])
    ibs, exposure = identity_by_state(pair_positions, pair_states, length, threads)
    return Summaries(
        sfs=folded_sfs(states),
        ld=linkage_disequilibrium(pair_positions, pair_states, rec_map),
        ibs=ibs,
        ibs_exposure=exposure,
        log10_watterson=watterson(int(seg.sum()), mut_map),
    )


# --- sites -------------------------------------------------------------------

def allele_counts(states):
    """Called and derived alleles per site."""
    full = states <= 2
    half = (states == 3) | (states == 4)
    called = 2 * full.sum(1) + half.sum(1)
    derived = np.where(full, states, 0).sum(1, dtype=np.int64) + (states == 4).sum(1)
    return called, derived


def segregating(states):
    """Sites where the called alleles are not all the same."""
    called, derived = allele_counts(states)
    return (derived > 0) & (derived < called)


def pairwise(states):
    """Sites that vary among individuals with both alleles called.

    Only these can tell two individuals apart, so LD and IBS read them."""
    full = states <= 2
    derived = np.where(full, states, 0).sum(1, dtype=np.int64)
    return (derived > 0) & (derived < 2 * full.sum(1))


# --- site-frequency spectrum and Watterson's estimate ------------------------

@cache
def _projection(observed=65, projected=PROJECTED):
    """P(k of `projected` haplotypes derived | j of n derived), for every n >= projected."""
    matrix = np.zeros((observed, observed, projected + 1))
    for n in range(projected, observed):
        for j in range(n + 1):
            for k in range(max(0, projected - (n - j)), min(projected, j) + 1):
                matrix[n, j, k] = math.comb(j, k) * math.comb(n - j, projected - k) / math.comb(n, projected)
    return matrix


def folded_sfs(states):
    """Folded spectrum after projecting every site to 32 haplotypes.

    Projection absorbs missing calls: a site with fewer than 32 called alleles
    contributes nothing, one with more is counted by its expected spectrum in
    a random subsample of 32."""
    called, derived = allele_counts(states)
    width = 2 * states.shape[1] + 1
    joint = np.bincount(called * width + derived, minlength=width * width).reshape(width, width)
    p = np.einsum("nj,njk->k", joint.astype(np.float64), _projection(width), optimize=True)
    return np.append(p[1:16] + p[31:16:-1], p[16])


def watterson(n_segregating, mut_map, haplotypes=64):
    """log10 of Watterson's estimate S / (4 a mu L), with mu L from the mutation map."""
    a = math.fsum(1.0 / i for i in range(1, haplotypes))
    breaks, rates = mut_map
    exposure = float(np.sum(np.diff(breaks) * rates))
    return float(np.float32(np.log10(max(n_segregating / (4.0 * a * exposure), np.finfo(float).tiny))))


# --- linkage disequilibrium --------------------------------------------------

def cumulative(breaks, rates, x):
    """Integral of a piecewise-constant map from 0 to each x."""
    area = np.concatenate(([0.0], np.cumsum(np.diff(breaks) * rates)))
    i = np.clip(np.searchsorted(breaks, x, side="right") - 1, 0, len(rates) - 1)
    return area[i] + (x - breaks[i]) * rates[i]


def linkage_disequilibrium(positions, states, rec_map):
    """Mean genotype r^2 of site pairs at fixed site lags, by genetic distance.

    Rows: mean r^2, mean 1/n, mean r^2 - 1/(n - 1), mean n / 32, log(1 + pairs),
    where n is the number of individuals called at both sites."""
    morgans = cumulative(*rec_map, positions.astype(np.float64))
    pairs, called, r2, inverse, centred = _ld_sums(states, morgans, LD_EDGES, LD_LAGS)
    out = np.zeros((5, len(LD_EDGES) + 1))
    seen = pairs > 0
    out[0, seen] = r2[seen] / pairs[seen]
    out[1, seen] = inverse[seen] / pairs[seen]
    out[2, seen] = centred[seen] / pairs[seen]
    out[3, seen] = called[seen] / pairs[seen] / states.shape[1]
    out[4] = np.log1p(pairs)
    return out.astype(np.float32)


@njit(nogil=True, cache=True)
def _ld_sums(states, morgans, edges, lags, block=65536):
    sites, n = states.shape
    classes = len(edges) + 1
    pairs = np.zeros(classes, np.int64)
    called = np.zeros(classes, np.int64)
    r2 = np.zeros(classes)
    inverse = np.zeros(classes)
    centred = np.zeros(classes)
    part = np.zeros((3, classes))
    for lag in lags:
        # sums are added up block by block, the order the original code used
        for start in range(0, sites - lag, block):
            part[:] = 0.0
            for i in range(start, min(start + block, sites - lag)):
                j = i + lag
                m = sx = sy = sxx = syy = sxy = 0
                for k in range(n):
                    x = np.int64(states[i, k])
                    y = np.int64(states[j, k])
                    if x <= 2 and y <= 2:
                        m += 1
                        sx += x
                        sy += y
                        sxx += x * x
                        syy += y * y
                        sxy += x * y
                if m < 4:
                    continue
                cx = sxx - sx * sx / m
                cy = syy - sy * sy / m
                if cx <= 0.0 or cy <= 0.0:
                    continue
                cov = sxy - sx * sy / m
                value = min(max(cov * cov / (cx * cy), 0.0), 1.0)
                c = np.searchsorted(edges, morgans[j] - morgans[i], side="right")
                pairs[c] += 1
                called[c] += m
                part[0, c] += value
                part[1, c] += 1.0 / m
                part[2, c] += value - 1.0 / (m - 1.0)
            r2 += part[0]
            inverse += part[1]
            centred += part[2]
    return pairs, called, r2, inverse, centred


# --- identity-by-state tracts ------------------------------------------------

def identity_by_state(positions, states, length, threads=1):
    """Tract-length spectra over all pairs of individuals.

    Kinds 0-3 are stretches cut at sites where two individuals share no
    allele (IBS0), at missing calls and at the window ends: from one IBS0 site
    to the next (0), between an IBS0 site and a window end (1), between an
    IBS0 site and a missing call (2), and stretches with called sites but no
    IBS0 site, between window ends or missing calls (3).
    Kinds 4-6: runs of sites with IBS0, IBS1 and IBS2. Each tract is counted
    in the region holding its midpoint, at all five region sizes."""
    n = states.shape[1]
    pairs = np.array([(a, b) for a in range(n) for b in range(a + 1, n)], dtype=np.int64)
    positions = positions.astype(np.int64)
    cuts = np.linspace(0, len(pairs), 4 * threads + 1).astype(np.int64)
    with ThreadPoolExecutor(threads) as pool:
        parts = pool.map(lambda c: _tracts(states, positions, length, pairs[cuts[c]:cuts[c + 1]]), range(len(cuts) - 1))
        leaf_counts = sum(parts)
    full = (states <= 2).sum(1).astype(np.int64)
    leaf_exposure = np.bincount(np.minimum(positions * LEAVES // length, LEAVES - 1),
                                weights=full * (full - 1) // 2, minlength=LEAVES)
    return _regions(leaf_counts.astype(np.uint64), axis=1), _regions(leaf_exposure, axis=0)


def _regions(leaves, axis):
    """Sum the 16 leaves into the 31 nested regions (1, 2, 4, 8 and 16 parts)."""
    leaves = np.moveaxis(leaves, axis, 0)
    out = [leaves.reshape(size, LEAVES // size, *leaves.shape[1:]).sum(1) for size in REGION_SIZES]
    return np.moveaxis(np.concatenate(out), 0, axis)


@njit(nogil=True, cache=True)
def _tracts(states, positions, length, pairs):
    counts = np.zeros((7, LEAVES, TRACT_CLASSES), np.int64)
    for q in range(len(pairs)):
        a = pairs[q, 0]
        b = pairs[q, 1]
        segment_start = 0      # where the current callable segment began
        first_segment = True   # no missing call seen yet
        seen = 0               # callable sites in this segment
        last_ibs0 = -1         # position of the last IBS0 site in this segment
        run_kind = -1          # IBS state of the current run, -1 if none
        run_start = run_end = 0
        for s in range(len(positions)):
            x = np.int64(states[s, a])
            y = np.int64(states[s, b])
            p = positions[s]
            if x > 2 or y > 2:                      # a missing call cuts every tract
                if run_kind >= 0:
                    _count(counts, 4 + run_kind, run_start, run_end, length)
                    run_kind = -1
                if last_ibs0 >= 0:
                    _count(counts, 2, last_ibs0, p, length)
                elif seen > 0:
                    _count(counts, 3, segment_start, p, length)
                segment_start, first_segment, seen, last_ibs0 = p, False, 0, -1
                continue
            seen += 1
            kind = 2 - abs(x - y)                   # 0, 1 or 2 alleles in common
            if kind != run_kind:
                if run_kind >= 0:
                    _count(counts, 4 + run_kind, run_start, run_end, length)
                run_kind, run_start = kind, p
            run_end = p
            if kind == 0:
                if last_ibs0 >= 0:
                    _count(counts, 0, last_ibs0, p, length)
                else:
                    _count(counts, 1 if first_segment else 2, segment_start, p, length)
                last_ibs0 = p
        if run_kind >= 0:
            _count(counts, 4 + run_kind, run_start, run_end, length)
        if last_ibs0 >= 0:
            _count(counts, 1, last_ibs0, length, length)
        elif seen > 0:
            _count(counts, 3, segment_start, length, length)
    return counts


@njit(nogil=True, cache=True)
def _count(counts, kind, start, end, length):
    size = end - start
    length_class = 0
    while size > 0:        # 0 -> 0, 1 -> 1, 2-3 -> 2, 4-7 -> 3, ...
        length_class += 1
        size >>= 1
    leaf = min((start + (end - start) // 2) * LEAVES // length, LEAVES - 1)
    counts[kind, leaf, min(length_class, TRACT_CLASSES - 1)] += 1
