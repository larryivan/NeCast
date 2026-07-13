import shutil
import subprocess

import numpy as np
import pytest

from necast import window
from necast.window import WINDOW_BP

M = 1_000_000


def test_read_bed(tmp_path):
    path = tmp_path / "mask.bed"
    path.write_text("track name=mask\n#chrom\tstart\tend\nchr1\t50\t100\nchr1\t0\t60\n1\t200\t300\nchr2\t5\t10\n")
    mask = window.read_bed(path)
    assert sorted(mask) == ["1", "2"]
    np.testing.assert_array_equal(mask["1"], [[0, 100], [200, 300]])


def test_subtract():
    intervals = np.array([[0, 100], [200, 300]])
    removed = np.array([[50, 60], [90, 250]])
    np.testing.assert_array_equal(window.subtract(intervals, removed), [[0, 50], [60, 90], [250, 300]])


def test_cut_windows():
    # a 1-Mb gap joins two intervals, a longer gap starts a new block; remainders are dropped
    intervals = np.array([[0, 20 * M], [21 * M, 45 * M], [46 * M + 1, 80 * M]])
    windows = window.cut_windows(intervals)
    assert len(windows) == 2
    np.testing.assert_array_equal(windows[0], [[0, 20 * M], [21 * M, 31 * M]])
    np.testing.assert_array_equal(windows[1], [[46 * M + 1, 76 * M + 1]])
    assert all(np.sum(w[:, 1] - w[:, 0]) == WINDOW_BP for w in windows)


def test_collapse():
    intervals = np.array([[100, 200], [300, 400]])
    np.testing.assert_array_equal(window.collapse(np.array([99, 100, 199, 200, 300, 399, 400]), intervals),
                                  [-1, 0, 99, -1, 100, 199, -1])


def test_states_from_alleles():
    alleles = np.array([[[0, 0], [0, 1], [1, 1], [0, -1], [-1, 1], [-1, -1], [-1, -2]]])   # [-1, -2] is '.'
    np.testing.assert_array_equal(window.states_from_alleles(alleles), [[0, 1, 2, 3, 4, 5, 5]])
    with pytest.raises(window.InputError, match="haploid"):
        window.states_from_alleles(np.array([[[1, -2]]]))


needs_tabix = pytest.mark.skipif(not (shutil.which("bgzip") and shutil.which("tabix")), reason="needs bgzip and tabix")


def write_vcf(folder, records):
    """An indexed VCF of samples a, b and c on chr1 from (pos, ref, alt, filter, genotypes) records."""
    cyvcf2 = pytest.importorskip("cyvcf2")
    lines = ["##fileformat=VCFv4.2", "##contig=<ID=chr1,length=1000>",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc"]
    lines += ["\t".join(["chr1", pos, ".", ref, alt, ".", filt, ".", "GT", *gts]) for pos, ref, alt, filt, gts in records]
    path = folder / "test.vcf"
    path.write_text("\n".join(lines) + "\n")
    subprocess.run(["bgzip", "-f", str(path)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", f"{path}.gz"], check=True)
    return cyvcf2.VCF(f"{path}.gz")


@needs_tabix
def test_read_genotypes(tmp_path):
    vcf = write_vcf(tmp_path, [
        ("101", "A", "G", "PASS", ["0/1", "1/1", "./."]),     # kept
        ("151", "C", "T", ".", ["0|0", "1/.", "0/0"]),         # kept: '.' filter, a half call
        ("171", "c", "t", "PASS", ["0/1", ".", "1/1"]),        # kept: lowercase bases, a call written '.'
        ("201", "A", "G", "LowQual", ["0/1", "0/1", "0/1"]),   # failed filter
        ("251", "A", "G,T", "PASS", ["0/1", "0/2", "0/0"]),    # multiallelic
        ("301", "AT", "A", "PASS", ["0/1", "0/1", "0/0"]),     # indel
        ("351", "A", "C", "PASS", ["0/1", "0/0", "0/0"]),      # two records at one position:
        ("351", "A", "G", "PASS", ["0/0", "0/1", "0/0"]),      # both dropped
        ("451", "G", "A", "PASS", ["0/1", "0/0", "0/0"]),      # outside the mask
    ])
    positions, states = window.read_genotypes(vcf, "chr1", np.array([[50, 400]]), np.array([2, 0, 1]))
    np.testing.assert_array_equal(positions, [50, 100, 120])
    np.testing.assert_array_equal(states, [[5, 1, 2], [0, 0, 4], [2, 1, 5]])


@needs_tabix
def test_haploid_calls_stop_the_reading(tmp_path):
    for genotypes in (["1", "0/1", "0/0"], ["1", "0", "1"]):     # some haploid calls; only haploid calls
        vcf = write_vcf(tmp_path, [("101", "A", "G", "PASS", genotypes)])
        with pytest.raises(window.InputError, match="haploid"):
            window.read_genotypes(vcf, "chr1", np.array([[50, 400]]), np.array([0, 1, 2]))
