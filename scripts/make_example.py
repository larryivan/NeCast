"""Build the example data and the test window from one simulated benchmark window.

The window is the first of history EMP_CEU (a published CEU size history) in
setting b (slices of the deCODE and Roulette maps, with sequencing errors) of
https://huggingface.co/datasets/Larrivhan/NeCast-benchmark:

    windows/b/EMP_CEU/samples/fragments/fragment_00000000.npz    sha256 38bf1c3c...
    estimates/necast/b/EMP_CEU/final/predictions.npz              sha256 212894f1...

    python scripts/make_example.py fragment_00000000.npz predictions.npz

writes necast/example/ (a VCF, a mask, two maps and the true history, as a user
would have them) and tests/data/benchmark_window.npz (the window's arrays as
the network read them, and its published prediction). Needs bgzip and tabix.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

ROW = 0
CHROMOSOME = "1"
GENOTYPES = np.array(["0/0", "0/1", "1/1", "0/.", "1/.", "./."])


def paper_individuals(window_id, n=64):
    """The 32 of the window's 64 individuals that the network read."""
    digest = hashlib.blake2b(f"dlcoalsim-ladder-v1:0:{window_id}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big")).permutation(n)[:32]


def states(packed, missing, n_haplotypes):
    """Six genotype states of every individual (see necast.summaries)."""
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :n_haplotypes]
    lost = np.zeros(bits.shape, bool)
    lost.reshape(-1)[missing] = True
    called = 2 - lost[:, 0::2] - lost[:, 1::2].astype(int)
    derived = (bits[:, 0::2] & ~lost[:, 0::2]) + (bits[:, 1::2] & ~lost[:, 1::2])
    out = np.where(called == 2, derived, 5)
    return np.where(called == 1, 3 + derived, out).astype(np.uint8)


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


def main(fragment, predictions):
    z = np.load(fragment)
    k = ROW
    v0, v1 = z["variant_offsets"][k: k + 2]
    m0, m1 = z["missing_offsets"][k: k + 2]
    r0, r1 = z["obs_rec_offsets"][k: k + 2]
    u0, u1 = z["obs_mut_offsets"][k: k + 2]
    length = int(z["sequence_length"][k])
    window_id = str(z["sample_id"][k])
    positions = z["variant_positions_bp"][v0:v1]
    chosen = states(z["genotype_packed"][v0:v1], z["missing_flat_idx"][m0:m1].astype(np.int64),
                    int(z["n_haplotypes"][0]))[:, paper_individuals(window_id)]
    rec = z["obs_rec_pos"][r0:r1], z["obs_rec_rate"][r0 - k: r1 - k - 1]
    mut = z["obs_mut_pos"][u0:u1], z["obs_mut_rate"][u0 - k: u1 - k - 1]
    published = np.load(predictions)
    assert published["row_index"][0] == k and length == 30_000_000

    out = Path("necast/example")
    out.mkdir(parents=True, exist_ok=True)
    write_vcf(out / "example.vcf", positions, chosen, length)
    (out / "mask.bed").write_text(f"{CHROMOSOME}\t0\t{length}\n")
    write_bed(out / "recombination.bed", rec[0], rec[1] * 1e8)              # cM/Mb
    write_bed(out / "mutation.bed", *mut)                                    # per bp
    for name in ("recombination.bed", "mutation.bed"):
        subprocess.run(["gzip", "-n", "-f", str(out / name)], check=True)
    with open(out / "truth.tsv", "w") as f:
        f.write("interval\tne\n")
        for i, value in enumerate(z["target_log10_ne"][k]):
            f.write(f"{i + 1}\t{10 ** float(value):.6g}\n")

    Path("tests/data").mkdir(parents=True, exist_ok=True)
    np.savez_compressed("tests/data/benchmark_window.npz", positions=positions, states=chosen,
                        rec_breaks=rec[0], rec_rates=rec[1], mut_breaks=mut[0], mut_rates=mut[1],
                        prediction=published["prediction"][0])
    mean_mu = float(np.sum(np.diff(mut[0]) * mut[1]) / length)
    print(f"window {window_id}: {len(positions)} sites, mean mutation rate {mean_mu:.4g}")


if __name__ == "__main__":
    main(*sys.argv[1:])
