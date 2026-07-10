"""Windows: 30 Mb of accessible sequence cut from a mask, with genotypes from a VCF.

A window is exactly 30 Mb of accessible sequence on one chromosome. Its
accessible intervals are joined end to end, so positions inside a window run
from 0 to 30,000,000 whatever the gaps between them.
"""

import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore", "no intervals found")   # cyvcf2, for a region without records; necast logs it

WINDOW_BP = 30_000_000
MAX_GAP_BP = 1_000_000       # a window never spans a masked stretch longer than this
INDIVIDUALS = 32             # the network reads 32 individuals at a time
BASES = {"A", "C", "G", "T"}


@dataclass
class Window:
    chromosome: str
    intervals: np.ndarray    # (k, 2) accessible intervals [start, end), 0-based, on the chromosome
    positions: np.ndarray    # sites, in window coordinates
    states: np.ndarray       # (sites, individuals) genotypes, coded as in necast.summaries
    rec_map: tuple           # (breaks, rates per bp per generation), breaks from 0 to WINDOW_BP
    mut_map: tuple

    @property
    def start(self):
        return int(self.intervals[0, 0])

    @property
    def end(self):
        return int(self.intervals[-1, 1])


# --- the mask and the windows ------------------------------------------------

def read_bed(path):
    """{chromosome: (n, 2) array of merged intervals} from a BED file. Header
    lines are skipped, and 'chr1' and '1' name the same chromosome."""
    rows = {}
    with open_text(path) as f:
        for line in f:
            fields = line.split()
            if len(fields) >= 3 and fields[1].isdigit():
                rows.setdefault(plain_name(fields[0]), []).append((int(fields[1]), int(fields[2])))
    return {chrom: _merge(np.array(sorted(r), dtype=np.int64)) for chrom, r in rows.items()}


def subtract(intervals, removed):
    """Intervals minus the parts covered by `removed` (both sorted)."""
    out = []
    for start, end in intervals:
        for cut_start, cut_end in removed[(removed[:, 1] > start) & (removed[:, 0] < end)]:
            if cut_start > start:
                out.append((start, cut_start))
            start = max(start, cut_end)
        if start < end:
            out.append((start, end))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def _merge(intervals):
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.array(merged, dtype=np.int64)


def cut_windows(intervals):
    """Windows of exactly WINDOW_BP accessible bases, as lists of intervals.

    Accessible intervals are grouped into blocks that no gap longer than
    MAX_GAP_BP interrupts; each block is cut from left to right, splitting an
    interval where a window fills up. A remainder shorter than a window is
    left out."""
    windows = []
    for block in _blocks(intervals):
        current, filled = [], 0
        for start, end in block:
            while start < end:
                take = min(end - start, WINDOW_BP - filled)
                current.append((start, start + take))
                filled += take
                start += take
                if filled == WINDOW_BP:
                    windows.append(np.array(current, dtype=np.int64))
                    current, filled = [], 0
    return windows


def _blocks(intervals):
    cuts = np.flatnonzero(intervals[1:, 0] - intervals[:-1, 1] > MAX_GAP_BP) + 1
    return np.split(intervals, cuts)


def collapse(physical, intervals):
    """Window coordinates of physical positions; -1 outside the accessible intervals."""
    offsets = np.concatenate(([0], np.cumsum(intervals[:, 1] - intervals[:, 0])[:-1]))
    i = np.searchsorted(intervals[:, 0], physical, side="right") - 1
    inside = (i >= 0) & (physical < intervals[np.maximum(i, 0), 1])
    return np.where(inside, physical - intervals[np.maximum(i, 0), 0] + offsets[np.maximum(i, 0)], -1)


# --- genotypes ---------------------------------------------------------------

def read_genotypes(vcf, chromosome, intervals, columns):
    """Positions (window coordinates) and states of the chosen samples.

    Kept: biallelic SNVs that passed filters, inside the accessible
    intervals. Positions with more than one such record are dropped, since
    they cannot be told apart. REF is taken as the ancestral allele."""
    positions, rows = [], []
    region = f"{chromosome}:{intervals[0, 0] + 1}-{intervals[-1, 1]}"
    for record in vcf(region):
        if record.FILTER is None and len(record.ALT) == 1 and record.REF in BASES and record.ALT[0] in BASES:
            positions.append(record.POS - 1)
            rows.append(record.genotype.array()[columns, :2])
    if not positions:
        return np.zeros(0, np.int64), np.zeros((0, len(columns)), np.uint8)
    physical = np.array(positions, dtype=np.int64)
    _, index, count = np.unique(physical, return_inverse=True, return_counts=True)
    window = collapse(physical, intervals)
    keep = (count[index] == 1) & (window >= 0)
    return window[keep], states_from_alleles(np.array(rows)[keep])


def states_from_alleles(alleles):
    """Six genotype states from allele pairs, with -1 for a missing allele."""
    if np.any(alleles < -1):
        raise ValueError("found haploid genotypes; NeCast reads diploid autosomes only")
    a, b = alleles[..., 0], alleles[..., 1]
    states = np.where((a >= 0) & (b >= 0), a + b, 5)
    one = (a < 0) ^ (b < 0)
    states = np.where(one, 3 + np.maximum(a, b), states)
    return states.astype(np.uint8)


def plain_name(chromosome):
    """'chr1' and '1' name the same chromosome."""
    return chromosome[3:] if chromosome.lower().startswith("chr") else chromosome


def open_text(path):
    import gzip
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)
