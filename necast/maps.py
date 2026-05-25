"""Recombination and mutation maps of a window.

A map is piecewise constant: breaks b_0 < b_1 < ... < b_n and one rate per bp
for each [b_i, b_i+1). Maps are read on chromosome coordinates and cut to a
window's accessible intervals, which are joined end to end, so the map of a
window runs from 0 to its accessible length.
"""

import math

import numpy as np

from necast.window import open_text, plain_name

RATE_RANGE = (1e-13, 1e-5)           # recombination per bp: the range of the training maps
SPIKE_WIDTH = (1_000, 1_000_000)     # bp over which a removed stretch's recombination is spread
MUTATION_FLOOR = 1e-3                # lowest mutation rate, relative to the window mean


def read(path, scale=1.0):
    """{chromosome: (breaks, rates)} from a BED file of rates: chromosome, start,
    end, rate, with an optional header line. Rates are multiplied by `scale`;
    a stretch between rows takes the chromosome's mean rate."""
    rows = {}
    with open_text(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 4 and fields[1].isdigit():
                rows.setdefault(plain_name(fields[0]), []).append((int(fields[1]), int(fields[2]), float(fields[3])))
    return {chromosome: _contiguous(np.array(sorted(r)), scale) for chromosome, r in rows.items()}


def _contiguous(rows, scale):
    starts, ends, rates = rows[:, 0], rows[:, 1], rows[:, 2] * scale
    breaks = np.union1d(starts, ends)
    i = np.searchsorted(starts, breaks[:-1], side="right") - 1
    mean = np.sum((ends - starts) * rates) / np.sum(ends - starts)
    return breaks, np.where(breaks[:-1] < ends[i], rates[i], mean)


def flat(rate, length):
    """A constant map over a window."""
    return np.array([0.0, float(length)]), np.array([rate], np.float32)


def mutation(physical, intervals, mu):
    """The mutation map of a window: the shape of the physical map, floored at
    MUTATION_FLOOR of its mean, with the mean over the window set to `mu`."""
    breaks, rates = _collapse(*physical, intervals)
    widths, length = np.diff(breaks), breaks[-1]
    floored = np.maximum(rates, float(np.sum(rates * widths) / length) * MUTATION_FLOOR)
    scaled = floored * (mu / float(np.sum(floored * widths) / length))
    keep = np.append(True, scaled[1:] != scaled[:-1])         # merge equal neighbours
    return np.append(breaks[:-1][keep], length), scaled[keep].astype(np.float32)


def recombination(physical, intervals):
    """The recombination map of a window, within RATE_RANGE.

    A masked stretch between two accessible intervals is cut out of the window,
    but its recombination is kept: it is placed at the junction, spread over the
    narrowest span (at least 1 kb) that keeps the rate under the cap. Spans that
    overlap or touch are merged."""
    breaks, rates = _collapse(*physical, intervals)
    base = np.clip(rates, *RATE_RANGE)
    area = np.append(0.0, np.cumsum(base * np.diff(breaks)))
    full = _extend(*physical, intervals[0, 0], intervals[-1, 1])
    full_area = np.append(0.0, np.cumsum(full[1] * np.diff(full[0])))
    length = int(breaks[-1])
    offsets = np.append(0, np.cumsum(intervals[:, 1] - intervals[:, 0]))
    clusters, spans = [], []                                  # clusters: [junctions, genetic length]
    for k in np.flatnonzero(intervals[1:, 0] > intervals[:-1, 1]):
        genetic = _integral(full[0], full_area, intervals[k, 1], intervals[k + 1, 0])
        junction = int(offsets[k + 1])
        clusters.append([[junction], genetic])
        spans.append(_spike(junction, junction, genetic, breaks, area, length))
        while len(spans) >= 2 and spans[-1][0] <= spans[-2][1]:
            junctions, genetic = clusters.pop()
            spans.pop()
            clusters[-1][0] += junctions
            clusters[-1][1] += genetic
            spans[-1] = _spike(min(clusters[-1][0]), max(clusters[-1][0]), clusters[-1][1], breaks, area, length)
    if not spans:
        return breaks, base.astype(np.float32)
    left, right, total = (np.array(v, np.float64) for v in zip(*spans))
    i = np.searchsorted(left, breaks, side="left") - 1
    covered = (i >= 0) & (breaks < right[np.maximum(i, 0)])  # strictly inside a span
    out = np.unique(np.concatenate((breaks[~covered], left, right)))
    starts = out[:-1]
    out_rates = base[np.searchsorted(breaks, starts, side="right") - 1].astype(np.float64)
    j = np.searchsorted(left, starts, side="right") - 1
    spiked = (j >= 0) & (starts < right[np.maximum(j, 0)])
    out_rates[spiked] = (total / (right - left))[j[spiked]]
    return out, np.clip(out_rates, *RATE_RANGE).astype(np.float32)


def _spike(c_lo, c_hi, genetic, breaks, area, length):
    """(left, right, genetic length) of the narrowest span around the junctions
    c_lo..c_hi whose mean rate, with `genetic` added, stays under the cap."""
    cap = RATE_RANGE[1]

    def place(width):
        left = min(max((c_lo + c_hi) // 2 - width // 2, 0), length - width)
        return left, left + width

    def fits(width):
        return genetic + _integral(breaks, area, *place(width)) <= cap * width * (1.0 + 1e-12)

    width = max(SPIKE_WIDTH[0], c_hi - c_lo + SPIKE_WIDTH[0], math.ceil(genetic / cap))
    if not fits(width):
        lo = hi = width
        while not fits(hi) and hi < SPIKE_WIDTH[1]:
            hi = min(SPIKE_WIDTH[1], 2 * hi)
        while hi - lo > 1:                                    # the smallest width that fits, in (lo, hi]
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if fits(mid) else (mid, hi)
        width = hi
    left, right = place(width)
    return left, right, genetic + _integral(breaks, area, left, right)


def _integral(breaks, area, a, b):
    return float(np.interp(b, breaks, area) - np.interp(a, breaks, area))


def _extend(breaks, rates, start, end):
    """Repeat the first or last rate where the span [start, end) reaches past the map."""
    if start < breaks[0]:
        breaks, rates = np.append(float(start), breaks), np.append(rates[0], rates)
    if end > breaks[-1]:
        breaks, rates = np.append(breaks, float(end)), np.append(rates, rates[-1])
    return breaks, rates


def _collapse(breaks, rates, intervals):
    """A physical map cut to the accessible intervals, joined end to end: a
    piece starts at each interval start and at each break inside an interval."""
    breaks, rates = _extend(breaks, rates, intervals[0, 0], intervals[-1, 1])
    starts, ends = intervals[:, 0], intervals[:, 1]
    offsets = np.append(0, np.cumsum(ends - starts)[:-1])
    j = np.searchsorted(starts, breaks, side="right") - 1
    inside = (j >= 0) & (breaks > starts[np.maximum(j, 0)]) & (breaks < ends[np.maximum(j, 0)])
    physical = np.sort(np.concatenate((starts.astype(np.float64), breaks[inside])))
    k = np.searchsorted(starts, physical, side="right") - 1
    collapsed = physical - starts[k] + offsets[k]
    piece_rates = rates[np.clip(np.searchsorted(breaks, physical, side="right") - 1, 0, len(rates) - 1)]
    return np.append(collapsed, float(np.sum(ends - starts))), piece_rates
