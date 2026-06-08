"""The necast command.

    necast run data.vcf.gz --mask accessible.bed --human --out results
    necast run data.vcf.gz --mask accessible.bed --mu 1.25e-8 --rec-rate 1e-8 --out results
    necast example
    necast simulate --preset mini --out-dir simulations
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from necast import __version__, maps, network, report, window

EXAMPLE = Path(__file__).parent / "example"
EXAMPLE_MU = 8.34e-9                      # mean mutation rate of the simulated example window
NOT_AUTOSOMES = {"X", "Y", "M", "MT"}
THIRD_PARTY = Path(__file__).parent / "third_party"
HUMAN = {"mu": 1.29e-8,                   # what --human uses unless told otherwise
         "rec_map": THIRD_PARTY / "maps" / "decode2019_avg_hg38.bed",
         "mut_map": THIRD_PARTY / "maps" / "roulette_2023_autosomes_hg38_1000bp.bed",
         "gaps": THIRD_PARTY / "reference" / "grch38_gaps.bed"}


class InputError(Exception):
    """A problem with the input, reported as a message without a traceback."""


def main(argv=None):
    parser = argparse.ArgumentParser(prog="necast", description="Population size history from unphased genotypes.")
    parser.add_argument("--version", action="version", version=f"necast {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = commands.add_parser("run", help="estimate a population's size history from a VCF",
                            description="Estimate the size history of one population from the genotypes "
                                        "of its individuals, read in groups of 32.")
    p.add_argument("vcf", help="VCF or BCF file, with an index (.tbi or .csi)")
    p.add_argument("--mask", required=True, help="BED file of the accessible sequence")
    p.add_argument("--out", required=True, help="folder for the results")
    p.add_argument("--human", action="store_true",
                   help="human data on GRCh38: leave out assembly gaps and centromeres, and unless given otherwise "
                        "use the deCODE recombination map, the Roulette mutation map and --mu 1.29e-8")
    p.add_argument("--mu", type=float, help="mutation rate per bp per generation")
    p.add_argument("--rec-rate", type=float, help="recombination rate per bp per generation, one for the whole genome")
    p.add_argument("--rec-map", help="BED file of recombination rates in cM/Mb, instead of --rec-rate")
    p.add_argument("--mut-map", help="BED file of relative mutation rates; their mean in each window is set to --mu")
    p.add_argument("--samples", help="file of the samples to read, one per line (default: all in the VCF)")
    p.add_argument("--chromosomes", help="chromosomes to read, separated by commas (default: all autosomes)")
    p.add_argument("--generation-time", type=float, help="years per generation, to give times in years as well")
    p.add_argument("--threads", type=int, default=cpus(), help=f"CPU threads (default: all, here {cpus()})")

    p = commands.add_parser("example", help="run NeCast on the example data that comes with it",
                            description="Run NeCast on one simulated window of 32 individuals whose history is known.")
    p.add_argument("--out", default="necast-example", help="folder for the results (default: necast-example)")
    p.add_argument("--threads", type=int, default=cpus(), help=f"CPU threads (default: all, here {cpus()})")

    commands.add_parser("simulate", help="simulate windows to train or test NeCast (see necast simulate --help)")

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["simulate"]:              # the simulator reads its own options
        command, args = "simulate", {"args": argv[1:]}
    else:
        args = vars(parser.parse_args(argv))
        command = args.pop("command")
    try:
        {"run": run, "example": example, "simulate": simulate}[command](**args)
    except InputError as e:
        sys.exit(f"necast: {e}")


def run(vcf, mask, out, human=False, mu=None, rec_rate=None, rec_map=None, mut_map=None, samples=None,
        chromosomes=None, generation_time=None, threads=None, truth=None):
    """Estimate the size history from a VCF and write the results into the folder `out`.

    Returns log10 Ne on the 64 intervals of necast.posterior.GENERATIONS. `truth`,
    the true log10 Ne of simulated data, is only drawn in the figure."""
    if human:
        mu = HUMAN["mu"] if mu is None else mu
        rec_map = HUMAN["rec_map"] if rec_map is None and rec_rate is None else rec_map
        mut_map = HUMAN["mut_map"] if mut_map is None else mut_map
    if mu is None:
        raise InputError("give the mutation rate per bp per generation with --mu")
    if rec_map is None and rec_rate is None:
        raise InputError("give a recombination map with --rec-map, or one rate for the whole genome with --rec-rate")
    threads = threads or cpus()
    vcf, groups, sample_count = open_vcf(vcf, samples)
    accessible = window.read_bed(mask)
    names = pick_chromosomes(vcf, accessible, chromosomes)
    if human:
        gaps = window.read_bed(HUMAN["gaps"])
        check_grch38(vcf, names, {name: int(g[-1, 1]) for name, g in gaps.items()})
        accessible = {name: window.subtract(accessible[name], gaps[name]) for name in names}
    rec = read_map(rec_map, names, "recombination", scale=1e-8) if rec_map else None
    mut = read_map(mut_map, names, "mutation") if mut_map else None
    contig = {window.plain_name(c): c for c in vcf.seqnames}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = logger(out / "log.txt")
    log(f"necast {__version__}: {plural(len(groups), 'group')} of 32 individuals from {sample_count} samples, "
        f"{'chromosome' if len(names) == 1 else 'chromosomes'} {', '.join(names)}, {threads} threads")
    if sample_count % 32:
        log(f"  the last {sample_count % 32} samples do not fill a group of 32 and are not read")
    torch.set_num_threads(min(threads, 8))
    net = network.load()
    results = []
    for chromosome in names:
        windows = window.cut_windows(accessible[chromosome])
        if not windows:
            log(f"chromosome {chromosome}: no window (no 30 Mb of accessible sequence without a gap over 1 Mb)")
        for intervals in windows:
            started = time.perf_counter()
            positions, states = window.read_genotypes(vcf, contig[chromosome], intervals, groups.ravel())
            rec_w = maps.recombination(rec[chromosome], intervals) if rec else maps.flat(rec_rate, window.WINDOW_BP)
            mut_w = maps.mutation(mut[chromosome], intervals, mu) if mut else maps.flat(mu, window.WINDOW_BP)
            for g in range(len(groups)):
                w = window.Window(chromosome, intervals, positions, np.ascontiguousarray(states[:, 32 * g: 32 * g + 32]),
                                  rec_w, mut_w)
                results.append((chromosome, w.start, w.end, g + 1, net.estimate(w, threads)))
            log(f"chromosome {chromosome}, {intervals[0, 0] + 1:,}-{intervals[-1, 1]:,} bp: "
                f"{len(positions):,} sites, {time.perf_counter() - started:.0f} s")
    if not results:
        raise InputError("found no window: a window needs 30 Mb of accessible sequence on one chromosome, "
                         "with no masked stretch longer than 1 Mb. Check the mask.")
    mean = report.write(out, results, generation_time, truth)
    log(f"done: {plural(len(results) // len(groups), 'window')}; results in {out}: necast.tsv, windows.tsv, necast.png")
    return mean


def example(out="necast-example", threads=None):
    """Run NeCast on the example data: one simulated window of 32 individuals."""
    print(f"The example is a simulated window of 32 individuals with a known history. Running it is the same\n"
          f"as running, in {EXAMPLE}:\n\n"
          f"  necast run example.vcf.gz --mask mask.bed --rec-map recombination.bed.gz \\\n"
          f"      --mut-map mutation.bed.gz --mu {EXAMPLE_MU} --out {Path(out).resolve()}\n")
    truth = np.log10(np.loadtxt(EXAMPLE / "truth.tsv", skiprows=1, usecols=1))
    mean = run(EXAMPLE / "example.vcf.gz", EXAMPLE / "mask.bed", out, mu=EXAMPLE_MU,
               rec_map=EXAMPLE / "recombination.bed.gz", mut_map=EXAMPLE / "mutation.bed.gz", threads=threads,
               truth=truth)
    expected = np.log10(np.loadtxt(EXAMPLE / "expected.tsv", skiprows=1, usecols=3))
    print(f"\nError against the true history: {np.sqrt(np.mean((mean - truth) ** 2)):.3f} "
          "(root mean square difference of log10 Ne over the 64 intervals).")
    difference = np.abs(mean - expected).max()
    if difference < 1e-4:
        print("NeCast works: the estimate matches the one in necast/example/expected.tsv.")
    else:
        print(f"Warning: the estimate differs from necast/example/expected.tsv by up to {difference:.2g} in log10 Ne.")


def simulate(args):
    """Run the simulator with its own options (see docs/simulate.md)."""
    try:
        from necast.simulator.cli import main as simulator
    except ModuleNotFoundError as e:
        raise InputError(f"necast simulate needs {e.name}; install it with:  pip install 'necast[simulate]'")
    sys.argv = ["necast simulate", *args]
    simulator()


def open_vcf(path, samples):
    """The VCF, reading only the chosen samples, with their columns in groups of 32."""
    from cyvcf2 import VCF
    if not Path(path).exists():
        raise InputError(f"cannot find {path}")
    if not (Path(f"{path}.tbi").exists() or Path(f"{path}.csi").exists()):
        raise InputError(f"{path} has no index (.tbi or .csi). Make one with:  bcftools index {path}")
    vcf = VCF(str(path))
    chosen = [line.strip() for line in open(samples) if line.strip()] if samples else vcf.samples
    unknown = sorted(set(chosen) - set(vcf.samples))
    if unknown:
        raise InputError(f"{len(unknown)} of the samples are not in the VCF, for example {unknown[0]}")
    if len(set(chosen)) < len(chosen):
        raise InputError("a sample is listed twice")
    if len(chosen) < 32:
        raise InputError(f"NeCast reads 32 diploid individuals at a time, and there are only {len(chosen)} samples")
    vcf.set_samples(chosen)
    column = {name: i for i, name in enumerate(vcf.samples)}
    return vcf, np.array([column[s] for s in chosen[: len(chosen) // 32 * 32]]).reshape(-1, 32), len(chosen)


def pick_chromosomes(vcf, mask, requested):
    contigs = [window.plain_name(c) for c in vcf.seqnames]
    if requested:
        names = [window.plain_name(c.strip()) for c in requested.split(",")]
        for name in names:
            if name not in contigs or name not in mask:
                raise InputError(f"chromosome {name} is not in {'the VCF' if name not in contigs else 'the mask'}")
        return names
    names = [c for c in contigs if c in mask and c.upper() not in NOT_AUTOSOMES]
    if not names:
        raise InputError("the VCF and the mask have no autosome in common; check the chromosome names")
    return names


def check_grch38(vcf, names, lengths):
    """Stop if a chromosome is not a GRCh38 autosome, or the VCF gives it another length."""
    for name in names:
        if name not in lengths:
            raise InputError(f"--human reads autosomes 1-22, and chromosome {name} is not one")
    try:
        found = dict(zip((window.plain_name(c) for c in vcf.seqnames), vcf.seqlens))
    except AttributeError:                # the VCF header gives no lengths
        return
    for name in names:
        if name in found and found[name] != lengths[name]:
            raise InputError(f"--human expects GRCh38 coordinates, but chromosome {name} is {found[name]:,} bp long "
                             f"in the VCF and {lengths[name]:,} bp in GRCh38")


def read_map(path, names, kind, scale=1.0):
    rates = maps.read(path, scale)
    missing = [name for name in names if name not in rates]
    if missing:
        raise InputError(f"the {kind} map {path} has no rates for chromosome {missing[0]}")
    return rates


def logger(path):
    """A function that prints a message and adds it to the file `path`."""
    path.write_text("")

    def log(message):
        print(message, flush=True)
        with open(path, "a") as f:
            f.write(message + "\n")
    return log


def cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def plural(n, word):
    return f"{n} {word}{'s' if n != 1 else ''}"
