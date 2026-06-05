"""Build the deCODE 2019 recombination map in the BED form the ingest reads.

Why this map.  The simulator needs recombination structure at the scale that
breaks haplotype tracts, and that scale is roughly a kilobase: measured on
chr20, the fine maps resolve hotspots with a median width near 500 bp and
carry about 1.8% of sequence above ten times the mean rate.  The map this
replaces -- Palsson et al. 2025 -- is published in 1 Mb windows, so below a
megabase it is constant and no hotspot exists in it at all.  That is not a
download mistake: 1 Mb is the resolution that paper released, because its
subject is non-crossover events and sex differences at megabase scale.  Its
crossover column is itself Halldorsson 2019 re-binned to 1 Mb.

So the finer product of the same lineage is the one to use directly.
Halldorsson et al. (Science 363:eaau1043, 2019) localise 4,531,535 crossovers
and report an average map resolution of 682 bp, natively on GRCh38.  Measured
here on chr20 between 10 and 26 Mb it gives 7,710 intervals, a median width of
972 bp, and 393 hot runs with a median width of 466 bp.

Being a pedigree map it also carries no demographic assumptions, which the
LD-based alternatives (HapMap II, pyrho) cannot claim -- they are inferred
from linkage disequilibrium under a demographic model, and this project infers
demography from linkage.

Source: https://hgdownload.soe.ucsc.edu/gbdb/hg38/recombRate/recombAvg.bw
Units in the bigWig are cM/Mb, which is what this script writes; the ingest
converts with ``RECOMB_CM_PER_MB_TO_PER_BP`` when ``recomb_map_unit`` is
``cM_per_Mb``.  Writing the source units rather than pre-converting keeps the
file checkable against UCSC.

Usage:
    python -m necast.third_party.maps.build_decode2019 --bigwig recombAvg.bw \
        --out necast/third_party/maps/decode2019_avg_hg38.bed
"""

from __future__ import annotations

import argparse
from pathlib import Path

AUTOSOMES = tuple("chr%d" % index for index in range(1, 23))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bigwig", required=True, help="recombAvg.bw from UCSC")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--max-cm-per-mb",
        type=float,
        default=1000.0,
        help=(
            "ceiling on the rate, in cM/Mb.  The map carries sub-resolution "
            "spikes: genome-wide, 34 intervals exceed 10,000 cM/Mb and their "
            "median width is one base pair, 58 bp in total.  They are an "
            "artefact of localising a crossover to a point, not a hotspot, "
            "and both the ingest and msprime's own RateMap reject rates above "
            "1e-4 per bp.  A thousand cM/Mb is eight hundred times the "
            "genome-wide mean of 1.2315 and far above any characterised human "
            "hotspot, and capping there touches 979 intervals covering "
            "0.0008% of the sequence"
        ),
    )
    parser.add_argument(
        "--min-cm-per-mb",
        type=float,
        default=0.1,
        help=(
            "floor on the rate, in cM/Mb.  A pedigree map reports zero where "
            "no crossover was observed, which is sparsity rather than an "
            "absence of recombination: 18.4%% of 1 kb cells on chr1 between 20 "
            "and 40 Mb are exactly zero, and in those same cells HapMap II "
            "reports a median of 6.6%% of its mean and pyrho CEU 17.0%%, "
            "neither of them ever zero.  A true zero would mean the interval "
            "never recombines and its whole span shares one genealogy, which "
            "at 18.4%% of cells is 3.7 Mb of a twenty-megabase row.  A tenth "
            "of a cM/Mb is 8%% of the genome-wide mean of 1.2315"
        ),
    )
    parser.add_argument(
        "--merge-tolerance",
        type=float,
        default=0.0,
        help=(
            "merge adjacent intervals whose rates agree to this relative "
            "tolerance.  Zero merges only exactly equal neighbours, which is "
            "lossless and already removes the long constant stretches the "
            "bigWig stores one interval at a time"
        ),
    )
    return parser.parse_args(argv)


def merged_intervals(intervals, tolerance: float):
    """Join neighbours that are contiguous and agree in rate.

    Lossless at the default tolerance.  It exists because the source stores a
    constant stretch as many equal intervals, and carrying those through would
    triple the file for no information.
    """

    out = []
    for start, stop, rate in intervals:
        if out:
            last_start, last_stop, last_rate = out[-1]
            close = (
                rate == last_rate
                if tolerance <= 0.0
                else abs(rate - last_rate) <= tolerance * max(abs(last_rate), 1e-30)
            )
            if last_stop == start and close:
                out[-1] = (last_start, stop, last_rate)
                continue
        out.append((start, stop, rate))
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import pyBigWig

    handle = pyBigWig.open(args.bigwig)
    available = set(handle.chroms())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("w") as stream:
        stream.write("Chromosome\tStart\tEnd\tRate\n")
        for name in AUTOSOMES:
            if name not in available:
                raise SystemExit("%s missing from %s" % (name, args.bigwig))
            raw = handle.intervals(name, 0, handle.chroms()[name])
            if raw is None:
                raise SystemExit("no intervals for %s" % name)
            # Drop the unmeasured stretches rather than writing them as zero:
            # a zero rate and an absent measurement are different claims, and
            # the slicer already treats gaps as the baseline.
            usable = [
                (int(a), int(b), min(max(float(v), args.min_cm_per_mb), args.max_cm_per_mb))
                for a, b, v in raw
                if v == v and b > a
            ]
            for start, stop, rate in merged_intervals(usable, args.merge_tolerance):
                stream.write(
                    "%s\t%d\t%d\t%.6f\n" % (name[3:], start, stop, rate)
                )
                total += 1
    handle.close()
    print("wrote %d intervals to %s" % (total, out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
