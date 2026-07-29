"""PSMC+ posterior of coalescence time, pooled over the 32 individuals.

PSMC+ (Cousins et al. 2024) is a hidden Markov model along one diploid genome.
Its hidden state is the coalescence time of the two alleles, binned into 65
intervals: [0, 50) generations, then the network's 64 intervals. Observations
are heterozygous sites counted in bins of about 100 bp; bins with missing calls
are shortened accordingly. The recombination and mutation maps scale the
transition and emission probabilities of each bin.

Each individual is decoded twice: first with a constant population size, then
with the size history that the pooled first pass implies. The network reads,
for both passes, the mean and the standard deviation over individuals of the
posterior averaged along the genome: 4 x 64 = 256 numbers.

One step of the recursions is O(D), not O(D^2). Leaving interval i happens
with probability p_i, and the new interval j is drawn with weight K[i, j],
which is a_i * b_j above the diagonal and c_i * d_j below it. Products with the
transition matrix therefore reduce to running sums.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from necast.third_party.psmcplus.transition_matrix import Transition_Matrix

GENERATIONS = np.geomspace(50.0, 500_000.0, 65)   # edges of the network's 64 intervals
MIN_SITES = 500                                   # fewer sites in the window: no posterior
MIN_HETEROZYGOTES = 8                             # fewer in an individual: not decoded
MEMORY = 4e9                                      # bytes the decoding threads may use together


def summarise(positions, states, rec_map, mut_map, length, threads=1):
    """The 256 posterior summaries of one window, or None if it cannot be decoded."""
    if len(positions) < MIN_SITES:
        return None
    het = states == 1
    missing = states > 2
    theta = float(np.mean(het.sum(0) / np.maximum(length - missing.sum(0), 1)))
    if not theta > 0:
        return None
    bin_bp = 100 if theta * 100 <= 1 else max(10, int(1 / theta))
    mu, mut_values = relative_map(*mut_map, length, "%.3f")
    r, rec_values = relative_map(*rec_map, length, "%.2f", floor=0.02)
    size = theta / (4 * mu)                       # the constant size of the first pass
    maps = Maps(binned(mut_values, length, bin_bp),
                binned(rec_values, length, bin_bp, truncate=True), bin_bp, theta)

    sequences = []
    for i in range(states.shape[1]):
        site = positions[het[:, i]]
        if len(site) >= MIN_HETEROZYGOTES:
            sequences.append(observations(site, positions[missing[:, i]], bin_bp))
    if not sequences:
        return None

    edges = np.concatenate(([0.0], GENERATIONS / (2 * size)))   # coalescent units
    rho = 4 * size * r * bin_bp                                 # per bin
    first = decode_all(sequences, maps, edges, rho, np.ones(65), threads)
    finite = np.isfinite(first).all(1)
    if not finite.any():
        return None
    first, sequences = first[finite], [s for s, ok in zip(sequences, finite) if ok]
    log10_ne = occupancy_to_log10_ne(first.mean(0))
    second = decode_all(sequences, maps, edges, rho, relative_rates(size, log10_ne), threads)
    second = second[np.isfinite(second).all(1)]
    if not len(second):
        return None
    return np.concatenate([first.mean(0), first.std(0), second.mean(0), second.std(0)]).astype(np.float32)


# --- maps --------------------------------------------------------------------

class Maps:
    """Per-bin map values the decoder reads, as indices into short tables."""

    def __init__(self, mutation, recombination, bin_bp, theta):
        # a sequence can run up to two bins past the window: repeat the last bin
        self.mutation, self.mutation_index = np.unique(np.pad(mutation, (0, 2), "edge"), return_inverse=True)
        if self.mutation[0] == 0:
            self.mutation[0] = self.mutation[1] / 2
        self.recombination, self.recombination_index = np.unique(np.pad(recombination, (0, 2), "edge"),
                                                                 return_inverse=True)
        self.bin_bp = bin_bp
        self.theta = theta


def relative_map(breaks, rates, length, text_format, floor=0.0):
    """Mean rate of a map, and its rate relative to that mean per base.

    The relative rates are rounded as PSMC+ reads them from a text file
    written with `text_format`, such as '%.3f'."""
    ends = np.rint(breaks[1:]).astype(np.int64)
    ends[-1] = length
    starts = np.concatenate(([0], ends[:-1]))
    keep = ends > starts
    starts, ends, rates = starts[keep], ends[keep], np.asarray(rates, float)[keep]
    mean = float(np.sum(rates * (ends - starts)) / np.sum(ends - starts))
    relative = np.array([float(text_format % max(x, floor)) for x in rates / mean])
    return mean, np.repeat(relative, ends - starts)


def binned(per_base, length, bin_bp, truncate=False):
    """Mean of a per-base map over consecutive bins; PSMC+ truncates the
    recombination map to two decimals and sets zeros to 1e-5."""
    n = length // bin_bp
    means = per_base[: n * bin_bp].reshape(n, bin_bp).mean(1)
    if truncate:
        means = np.floor(means * 100) / 100
        means[means == 0] = 1e-5
    return means


# --- one individual ----------------------------------------------------------

def observations(heterozygous, missing, bin_bp):
    """Per-bin observation codes, as PSMC+ builds them from its input file.

    A code is (heterozygotes) + (missing sites) x (bin size). The missing
    sites between two heterozygotes are placed right after the first one,
    and the sequence ends two bins after the last heterozygote."""
    previous = np.concatenate(([-1], heterozygous[:-1]))
    gaps = np.searchsorted(missing, heterozygous, "right") - np.searchsorted(missing, previous, "right")
    gaps = np.minimum(gaps, np.maximum(heterozygous - previous - 2, 0))
    return _codes(heterozygous.astype(np.int64), gaps.astype(np.int64), bin_bp)


@njit(cache=True)
def _codes(heterozygous, gaps, bin_bp):
    bins = (heterozygous[-1] + 2 * bin_bp) // bin_bp
    hets = np.zeros(bins, np.int64)
    masked = np.zeros(bins, np.int64)
    for k in range(len(heterozygous)):
        hets[heterozygous[k] // bin_bp] += 1
        start = 0 if k == 0 else heterozygous[k - 1] + 1
        for site in range(start, start + gaps[k]):
            masked[site // bin_bp] += 1
    most = hets.max()
    return hets + masked * ((most + 1) if bin_bp <= most else bin_bp)


# --- decoding ----------------------------------------------------------------

def decode_all(sequences, maps, edges, rho, relative_inverse_size, threads):
    """Posterior occupancy of the 64 intervals for every individual."""
    model = Transitions(edges, rho, relative_inverse_size, maps.recombination)
    mids = (edges[:-1] + edges[1:]) / 2
    no_event = np.exp(-maps.bin_bp * maps.theta * np.outer(maps.mutation, mids))
    factorial = np.array([math.factorial(j) for j in range(maps.bin_bp)], float)

    def one(codes):
        n = len(codes)
        occupancy = _decode(codes, maps.mutation_index[:n], maps.recombination_index[:n], maps.bin_bp, maps.theta,
                            mids, maps.mutation, no_event, model.start, model.leave, model.stay_rate,
                            model.up, model.down, model.next_up, model.next_down, model.row_up, model.row_down,
                            factorial, np.empty((n, len(mids))))
        occupancy = occupancy[1:]                         # drop the [0, 50) generations state
        return occupancy / occupancy.sum()

    longest = max(len(s) for s in sequences)
    threads = max(1, min(threads, int(MEMORY // (longest * len(mids) * 8))))
    with ThreadPoolExecutor(threads) as pool:
        return np.array(list(pool.map(one, sequences)))


class Transitions:
    """PSMC+'s transition matrix in the factored form the recursions use."""

    def __init__(self, edges, rho, relative_inverse_size, recombination):
        d = len(edges) - 1
        matrix = Transition_Matrix(D=d, midpoint_transitions=False)
        matrix.T = edges
        q = matrix.write_tm(lambda_A=np.clip(relative_inverse_size, 0.05, 50.0), lambda_B=None, T_S_index=None,
                            T_E_index=None, gamma=None, check=False, rho=rho, exponential=True)
        expected_time = np.asarray(matrix.e_betas, float)
        jumps = q.copy()
        np.fill_diagonal(jumps, 0.0)
        jumps /= (1 - np.exp(-rho * expected_time))[:, None]   # K: where a recombination lands
        self.start = stationary(q)
        self.stay_rate = jumps.sum(1)
        self.leave = 1 - np.exp(-np.outer(recombination, rho * expected_time))   # p_i for each map value
        self.up = np.append(np.diag(jumps, 1), 0.0)              # K[i, i + 1]
        self.down = np.insert(np.diag(jumps, -1), 0, 0.0)        # K[i, i - 1]
        # ratios of neighbouring entries, which is all the rank-one structure needs
        self.next_up = np.zeros(d)      # b[j + 1] / b[j]
        self.next_down = np.zeros(d)    # d[j - 1] / d[j]
        self.row_up = np.zeros(d)       # a[i] / a[i + 1]
        self.row_down = np.zeros(d)     # c[i] / c[i - 1]
        for j in range(1, d - 1):
            self.next_up[j] = _ratio(jumps[j - 1, j + 1], jumps[j - 1, j])
            self.next_down[j] = _ratio(jumps[j + 1, j - 1], jumps[j + 1, j])
        for i in range(d - 2):
            self.row_up[i] = _ratio(jumps[i, i + 2], jumps[i + 1, i + 2])
        for i in range(2, d):
            self.row_down[i] = _ratio(jumps[i, i - 2], jumps[i - 1, i - 2])


def _ratio(x, y):
    return x / y if y != 0 else 0.0


def stationary(q):
    """The distribution that the transition matrix leaves unchanged."""
    d = len(q)
    a = q.T - np.eye(d)
    a[-1] = 1.0
    return np.linalg.solve(a, np.eye(d)[-1])


@njit(nogil=True, cache=True)
def _decode(codes, mutation_index, recombination_index, bin_bp, theta, mids, mutation, no_event,
            start, leave, stay_rate, up, down, next_up, next_down, row_up, row_down, factorial, forward):
    """Scaled forward-backward pass; returns the posterior averaged over bins."""
    n, d = forward.shape
    scale = np.empty(n)
    emission = np.empty(d)
    g = np.empty(d)
    h = np.empty(d)
    w = np.empty(d)
    back = np.empty(d)
    occupancy = np.zeros(d)

    for i in range(n):                               # forward
        _emission(emission, codes[i], mutation[mutation_index[i]], no_event[mutation_index[i]],
                  bin_bp, theta, mids, factorial)
        if i == 0:
            for s in range(d):
                g[s] = emission[s] * start[s]
        else:
            p = leave[recombination_index[i]]
            for s in range(d):
                w[s] = forward[i - 1, s] * p[s]
            u = 0.0                                  # sum of w[k] K[k, s] over k < s
            for s in range(d):
                g[s] = forward[i - 1, s] - w[s] * stay_rate[s] + u
                u = u * next_up[s] + w[s] * up[s]
            v = 0.0                                  # sum of w[k] K[k, s] over k > s
            for s in range(d - 1, -1, -1):
                g[s] += v
                v = v * next_down[s] + w[s] * down[s]
            for s in range(d):
                g[s] *= emission[s]
        total = 0.0
        for s in range(d):
            total += g[s]
        scale[i] = total
        for s in range(d):
            forward[i, s] = g[s] / total

    for s in range(d):                               # backward, adding up the posterior
        back[s] = 1.0 / scale[n - 1]
    for i in range(n - 1, -1, -1):
        total = 0.0
        for s in range(d):
            h[s] = max(forward[i, s] * back[s], 0.0)
            total += h[s]
        total = max(total, 1e-300)
        for s in range(d):
            occupancy[s] += h[s] / total
        if i == 0:
            break
        _emission(emission, codes[i], mutation[mutation_index[i]], no_event[mutation_index[i]],
                  bin_bp, theta, mids, factorial)
        p = leave[recombination_index[i]]
        for s in range(d):
            g[s] = emission[s] * back[s]
        x_up = 0.0                                   # sum of K[s, k] g[k] over k > s
        h[d - 1] = 0.0
        for s in range(d - 2, -1, -1):
            x_up = x_up * row_up[s] + up[s] * g[s + 1]
            h[s] = x_up
        x_down = 0.0                                 # sum of K[s, k] g[k] over k < s
        for s in range(1, d):
            x_down = x_down * row_down[s] + down[s] * g[s - 1]
            h[s] += x_down
        for s in range(d):
            back[s] = (g[s] + p[s] * (h[s] - stay_rate[s] * g[s])) / scale[i - 1]
    return occupancy / n


@njit(nogil=True, cache=True)
def _emission(out, code, mutation, no_event, bin_bp, theta, mids, factorial):
    """P(observation | interval): Poisson heterozygotes over the called bases."""
    if code == 0:
        out[:] = no_event
        return
    hets, masked = code % bin_bp, code // bin_bp
    for s in range(len(out)):
        x = (bin_bp - masked) * theta * (mids[s] * mutation)
        out[s] = (x ** hets) * np.exp(-x) / factorial[hets]


# --- the second pass ---------------------------------------------------------

def occupancy_to_log10_ne(occupancy):
    """Size history implied by a posterior occupancy: the inverse of its hazard."""
    f = np.maximum(occupancy, 1e-12)
    f = f / f.sum()
    survival = 1 - np.concatenate(([0.0], np.cumsum(f)[:-1]))
    hazard = f / (np.diff(GENERATIONS) * np.maximum(survival, 1e-6))
    return np.log10(1 / (2 * np.maximum(hazard, 1e-12)))


def relative_rates(size, log10_ne):
    """Coalescence rate of each state relative to the first pass; the extra
    [0, 50) generations state takes the rate of the first interval."""
    rates = size / 10.0 ** log10_ne
    return np.concatenate(([rates[0]], rates))
