"""Build the example data and the test window from one simulated benchmark window.

The window is the first of history EMP_CEU (a published CEU size history) in
setting b (slices of the deCODE and Roulette maps, with sequencing errors) of
https://huggingface.co/datasets/Larrivhan/NeCast-benchmark:

    windows/b/EMP_CEU/samples/fragments/fragment_00000000.npz    sha256 38bf1c3c...
    estimates/necast/b/EMP_CEU/final/predictions.npz              sha256 212894f1...

    python scripts/make_example.py windows/b/EMP_CEU predictions.npz

writes necast/example/ (a VCF, a mask, two maps and the true history, as a user
would have them) and tests/data/benchmark_window.npz (the window's arrays as
the network read them, and its published prediction). Needs bgzip and tabix.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from necast.train import read_simulations
from necast.window import WINDOW_BP

CHROMOSOME = "1"
GENOTYPES = np.array(["0/0", "0/1", "1/1", "0/.", "1/.", "./."])


def write_bed(path, breaks, rates):
    ends = np.rint(breaks[1:]).astype(np.int64)
    starts = np.append(0, ends[:-1])
    with open(path, "w") as f:
        f.write("chromosome\tstart\tend\trate\n")
        for s, e, r in zip(starts, ends, rates):
            if e > s:
                f.write(f"{CHROMOSOME}\t{s}\t{e}\t{r:.8g}\n")


def write_vcf(path, positions, chosen, length):
    rng = np.random.default_rng(1)
    bases = np.array(list("ACGT"))
    ref = rng.integers(0, 4, len(positions))
    alt = (ref + rng.integers(1, 4, len(positions))) % 4
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write(f"##contig=<ID={CHROMOSOME},length={length}>\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        f.write("\t".join(f"person{i + 1}" for i in range(32)) + "\n")
        for i, pos in enumerate(positions):
            f.write(f"{CHROMOSOME}\t{pos + 1}\t.\t{bases[ref[i]]}\t{bases[alt[i]]}\t.\tPASS\t.\tGT\t")
            f.write("\t".join(GENOTYPES[chosen[i]]) + "\n")
    subprocess.run(["bgzip", "-f", str(path)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", f"{path}.gz"], check=True)


def main(simulations, predictions):
    name, w, target = next(read_simulations(simulations))      # the first window of the first fragment
    published = np.load(predictions)
    assert published["row_index"][0] == 0

    out = Path("necast/example")
    out.mkdir(parents=True, exist_ok=True)
    write_vcf(out / "example.vcf", w.positions, w.states, WINDOW_BP)
    (out / "mask.bed").write_text(f"{CHROMOSOME}\t0\t{WINDOW_BP}\n")
    write_bed(out / "recombination.bed", w.rec_map[0], w.rec_map[1] * 1e8)  # cM/Mb
    write_bed(out / "mutation.bed", *w.mut_map)                              # per bp
    for file in ("recombination.bed", "mutation.bed"):
        subprocess.run(["gzip", "-n", "-f", str(out / file)], check=True)
    with open(out / "truth.tsv", "w") as f:
        f.write("interval\tne\n")
        for i, value in enumerate(target):
            f.write(f"{i + 1}\t{10 ** float(value):.6g}\n")

    Path("tests/data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed("tests/data/benchmark_window.npz", positions=w.positions, states=w.states,
                        rec_breaks=w.rec_map[0], rec_rates=w.rec_map[1], mut_breaks=w.mut_map[0],
                        mut_rates=w.mut_map[1], prediction=published["prediction"][0])
    mean_mu = float(np.sum(np.diff(w.mut_map[0]) * w.mut_map[1]) / WINDOW_BP)
    print(f"window {name}: {len(w.positions)} sites, mean mutation rate {mean_mu:.4g}")


if __name__ == "__main__":
    main(*sys.argv[1:])
