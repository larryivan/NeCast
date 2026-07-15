"""The necast command.

    necast run data.vcf.gz --mask accessible.bed --human --out results
    necast run data.vcf.gz --mask accessible.bed --mu 1.25e-8 --rec-rate 1e-8 --out results
    necast example
    necast simulate --preset mini --out-dir simulations
    necast train simulations --out trained
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from necast import __version__, maps, network, report, window
from necast.window import InputError

EXAMPLE = Path(__file__).parent / "example"
EXAMPLE_MU = 8.34e-9                      # mean mutation rate of the simulated example window
NOT_AUTOSOMES = {"X", "Y", "M", "MT"}
THIRD_PARTY = Path(__file__).parent / "third_party"
HUMAN = {"mu": 1.29e-8,                   # what --human uses unless told otherwise
         "rec_map": THIRD_PARTY / "maps" / "decode2019_avg_hg38.bed",
         "mut_map": THIRD_PARTY / "maps" / "roulette_2023_autosomes_hg38_1000bp.bed",
         "gaps": THIRD_PARTY / "reference" / "grch38_gaps.bed"}
TRAINED = {"mu": (6.25e-9, 2.5e-8),        # ranges of the training simulations: mutation and
           "rec": (1.7e-9, 3e-8),          # recombination per bp per generation (window means),
           "missing": 0.02}                # missing allele calls


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
    p.add_argument("--chromosomes", help="chromosomes to read, separated by commas (default: all in both the VCF "
                                         "and the mask except X, Y, M and MT; with --human, 1-22)")
    p.add_argument("--generation-time", type=float, help="years per generation, to give times in years as well")
    p.add_argument("--weights", help="a network trained with necast train (default: the published one)")
    p.add_argument("--threads", type=int, default=cpus(), help=f"CPU threads (default: all, here {cpus()})")

    p = commands.add_parser("example", help="run NeCast on the example data that comes with it",
                            description="Run NeCast on one simulated window of 32 individuals whose history is known.")
    p.add_argument("--out", default="necast-example", help="folder for the results (default: necast-example)")
    p.add_argument("--threads", type=int, default=cpus(), help=f"CPU threads (default: all, here {cpus()})")

    commands.add_parser("simulate", help="simulate windows to train or test NeCast (see necast simulate --help)")

    p = commands.add_parser("train", help="train the network on simulated windows",
                            description="Train the network on windows from necast simulate (see docs/train.md).")
    p.add_argument("simulations", help="folder written by necast simulate")
    p.add_argument("--out", required=True, help="folder for the prepared windows and the trained network")
    p.add_argument("--epochs", type=int, default=12, help="passes over the windows (default: 12)")
    p.add_argument("--device", help="cuda or cpu (default: cuda if there is a GPU)")
    p.add_argument("--prepare-only", action="store_true", help="only prepare the windows, which needs no GPU")
    p.add_argument("--threads", type=int, default=cpus(), help=f"CPU threads (default: all, here {cpus()})")

    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["simulate"]:              # the simulator reads its own options
        command, args = "simulate", {"args": argv[1:]}
    else:
        args = vars(parser.parse_args(argv))
        command = args.pop("command")
    try:
        {"run": run, "example": example, "simulate": simulate, "train": train}[command](**args)
    except InputError as e:
        sys.exit(f"necast: {e}")


def run(vcf, mask, out, human=False, mu=None, rec_rate=None, rec_map=None, mut_map=None, samples=None,
        chromosomes=None, generation_time=None, weights=None, threads=None, truth=None):
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
    if not 0 < mu < 1e-6:
        raise InputError(f"--mu {mu:g} is not a mutation rate per bp per generation (humans: about 1.29e-8)")
    if rec_rate is not None and not 0 < rec_rate < 1e-4:
        raise InputError(f"--rec-rate {rec_rate:g} is not a rate per bp per generation; 1 cM/Mb is 1e-8")
    for path in (vcf, mask, samples, rec_map, mut_map, weights):
        if path and not Path(path).exists():
            raise InputError(f"cannot find {path}")
    threads = threads or cpus()
    vcf, groups, sample_count = open_vcf(vcf, samples)
    accessible = window.read_bed(mask)
    gaps = window.read_bed(HUMAN["gaps"]) if human else None          # GRCh38 chromosomes 1-22
    names = pick_chromosomes(vcf, accessible, chromosomes, autosomes=gaps)
    if human:
        check_grch38(vcf, names, {name: int(g[-1, 1]) for name, g in gaps.items()})
        accessible = {name: window.subtract(accessible[name], gaps[name]) for name in names}
    rec = read_map(rec_map, names, "recombination", scale=1e-8) if rec_map else None
    if rec and np.mean([np.sum(np.diff(b) * r) / (b[-1] - b[0]) for b, r in (rec[n] for n in names)]) < 1e-11:
        raise InputError(f"the rates in {rec_map} are below 0.001 cM/Mb on average; the map should be in cM/Mb "
                         "(the human average is about 1.3). If it is per bp per generation, multiply it by 1e8.")
    mut = read_map(mut_map, names, "mutation") if mut_map else None
    contig = {window.plain_name(c): c for c in vcf.seqnames}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = logger(out / "log.txt")
    log(f"necast {__version__}: {plural(len(groups), 'group')} of 32 individuals from {sample_count} samples, "
        f"{'chromosome' if len(names) == 1 else 'chromosomes'} {', '.join(names)}, {threads} threads")
    if sample_count % 32:
        log(f"  the last {sample_count % 32} samples do not fill a group of 32 and are not read")
    if not TRAINED["mu"][0] <= mu <= TRAINED["mu"][1]:
        log(f"  warning: --mu {mu:g} is outside the training range ({TRAINED['mu'][0]:g} to {TRAINED['mu'][1]:g}); "
            "the estimate may be biased")
    if rec is None:
        log("  note: without a recombination map, NeCast underestimated sizes 50-500 generations ago about "
            "1.7-fold in the paper's simulations")
    torch.set_num_threads(min(threads, 8))
    net = network.load(weights or network.WEIGHTS)
    started, results = time.perf_counter(), []
    for chromosome in names:
        windows = window.cut_windows(accessible[chromosome])
        if not windows:
            log(f"chromosome {chromosome}: no window (no 30 Mb of accessible sequence without a gap over 1 Mb)")
        for intervals in windows:
            positions, states = window.read_genotypes(vcf, contig[chromosome], intervals, groups.ravel())
            span = f"chromosome {chromosome}, {intervals[0, 0] + 1:,}-{intervals[-1, 1]:,} bp"
            if not len(positions):                # e.g. a chromosome the VCF header names but holds no records for
                log(f"{span}: no sites in the VCF; left out")
                continue
            physical = rec[chromosome] if rec else maps.flat(rec_rate, intervals[-1, 1])   # on the chromosome
            rec_w = maps.recombination(physical, intervals)
            mut_w = maps.mutation(mut[chromosome], intervals, mu) if mut else maps.flat(mu, window.WINDOW_BP)
            log(f"{span}: {len(positions):,} sites")
            check_map(rec_w, log)
            for g in range(len(groups)):
                w = window.Window(positions, np.ascontiguousarray(states[:, 32 * g: 32 * g + 32]), rec_w, mut_w)
                check_calls(w.states, f" of group {g + 1}" if len(groups) > 1 else "", log)
                results.append((chromosome, int(intervals[0, 0]), int(intervals[-1, 1]), g + 1,
                                net.estimate(w, threads)))
    if not results:
        raise InputError("found no window: a window needs 30 Mb of accessible sequence on one chromosome, "
                         "with no masked stretch longer than 1 Mb, and sites in the VCF. Check the mask.")
    mean = report.write(out, results, generation_time, truth)
    log(f"done in {time.perf_counter() - started:.0f} s: {plural(len(results) // len(groups), 'window')}; "
        f"results in {out}: necast.tsv, windows.tsv, necast.png")
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


def train(simulations, out, epochs=12, device=None, prepare_only=False, threads=None):
    """Prepare the simulated windows, then train a network on them (see necast.train)."""
    from necast import train as training
    if epochs < 1:
        raise InputError("--epochs must be at least 1")
    torch.set_num_threads(min(threads or cpus(), 8))
    Path(out).mkdir(parents=True, exist_ok=True)
    log = logger(Path(out) / "log.txt")
    training.prepare(simulations, out, threads or cpus(), log)
    if not prepare_only:
        training.train(out, epochs, device, log)


def open_vcf(path, samples):
    """The VCF, reading only the chosen samples, with their columns in groups of 32."""
    from cyvcf2 import VCF
    if not (Path(f"{path}.tbi").exists() or Path(f"{path}.csi").exists()):
        raise InputError(f"{path} has no index (.tbi or .csi). Make one with:  bcftools index {path}  "
                         "(compress a plain .vcf with bgzip first)")
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


def pick_chromosomes(vcf, mask, requested, autosomes=None):
    """The requested chromosomes, or else those in both the VCF and the mask
    that are `autosomes` or, when these are not known, not X, Y, M or MT."""
    contigs = [window.plain_name(c) for c in vcf.seqnames]
    if requested:
        names = [window.plain_name(c.strip()) for c in requested.split(",")]
        for name in names:
            if name not in contigs or name not in mask:
                raise InputError(f"chromosome {name} is not in {'the VCF' if name not in contigs else 'the mask'}")
        return names
    names = [c for c in contigs if c in mask and (c in autosomes if autosomes else c.upper() not in NOT_AUTOSOMES)]
    if not names:
        raise InputError("the VCF and the mask have no autosome in common; check the chromosome names")
    return names


def check_map(rec_map, log):
    """Warn when a window's recombination rate is outside the range of the training windows."""
    rate = float(np.sum(np.diff(rec_map[0]) * rec_map[1])) / window.WINDOW_BP
    low, high = TRAINED["rec"]
    if not low <= rate <= high:
        log(f"  warning: the mean recombination rate of this window, {rate:.2g} per bp, is outside the "
            f"training range ({low:g} to {high:g})")


def check_calls(states, label, log):
    """Warn when more allele calls are missing than in any training window."""
    missing = (np.isin(states, (3, 4)).sum() + 2 * (states == 5).sum()) / (2 * states.size)
    if missing > TRAINED["missing"]:
        log(f"  warning: {missing:.1%} of the allele calls{label} are missing; the training windows had at most "
            f"{TRAINED['missing']:.0%}")


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
    try:
        rates = maps.read(path, scale)
    except ValueError:                    # e.g. a map in HapMap format: position, rate, cM
        rates = {}
    if not rates:
        raise InputError(f"cannot read the {kind} map {path}: a map is a BED file of chromosome, start, end and "
                         "rate (see docs/guide.md)")
    missing = [name for name in names if name not in rates]
    if missing:
        raise InputError(f"the {kind} map {path} has no rates for chromosome {missing[0]}")
    return rates


def logger(path):
    """A function that prints a message and adds it to the file `path`."""
    def log(message):
        print(message, flush=True)
        with open(path, "a") as f:
            f.write(message + "\n")
    return log


def cpus():
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def plural(n, word):
    return f"{n} {word}{'s' if n != 1 else ''}"
