import gzip
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from necast import cli

EXAMPLE = cli.EXAMPLE
VCF, MASK = EXAMPLE / "example.vcf.gz", EXAMPLE / "mask.bed"
needs_tabix = pytest.mark.skipif(not (shutil.which("bgzip") and shutil.which("tabix")), reason="needs bgzip and tabix")


def example_vcf(folder, edit):
    """The example VCF with its lines changed by `edit`, compressed and indexed."""
    path = folder / "edited.vcf"
    path.write_text("\n".join(edit(gzip.open(VCF, "rt").read().splitlines())) + "\n")
    subprocess.run(["bgzip", "-f", str(path)], check=True)
    subprocess.run(["tabix", "-f", "-p", "vcf", f"{path}.gz"], check=True)
    return f"{path}.gz"


def test_example(tmp_path):
    cli.example(out=tmp_path, threads=4)
    got = np.loadtxt(tmp_path / "necast.tsv", skiprows=1, usecols=3)
    expected = np.loadtxt(EXAMPLE / "expected.tsv", skiprows=1, usecols=3)
    assert np.abs(np.log10(got / expected)).max() < 1e-4
    assert (tmp_path / "necast.png").exists() and (tmp_path / "log.txt").exists()
    assert len((tmp_path / "windows.tsv").read_text().splitlines()) == 2


def test_input_errors(tmp_path):
    with pytest.raises(cli.InputError, match="--mu"):
        cli.run(VCF, MASK, tmp_path, rec_rate=1e-8)
    with pytest.raises(cli.InputError, match="--rec-map"):
        cli.run(VCF, MASK, tmp_path, mu=1e-8)
    unindexed = tmp_path / "copy.vcf.gz"
    unindexed.write_bytes(VCF.read_bytes())
    with pytest.raises(cli.InputError, match="bcftools index"):
        cli.run(unindexed, MASK, tmp_path, mu=1e-8, rec_rate=1e-8)
    samples = tmp_path / "samples.txt"
    samples.write_text("\n".join(f"person{i}" for i in range(1, 11)))
    with pytest.raises(cli.InputError, match="only 10"):
        cli.run(VCF, MASK, tmp_path, mu=1e-8, rec_rate=1e-8, samples=samples)
    short = tmp_path / "short.bed"
    short.write_text("1\t0\t10000000\n")
    with pytest.raises(cli.InputError, match="no window"):
        cli.run(VCF, short, tmp_path, mu=1e-8, rec_rate=1e-8)
    with pytest.raises(cli.InputError, match="cannot find"):
        cli.run(VCF, tmp_path / "typo.bed", tmp_path, mu=1e-8, rec_rate=1e-8)
    hapmap = tmp_path / "hapmap.txt"
    hapmap.write_text("Chromosome Position(bp) Rate(cM/Mb) Map(cM)\nchr1 55550 2.98 0\nchr1 82571 0.08 0.08\n")
    with pytest.raises(cli.InputError, match="a map is a BED file"):
        cli.run(VCF, MASK, tmp_path, mu=1e-8, rec_map=hapmap)
    with pytest.raises(cli.InputError, match="--epochs"):
        cli.train(tmp_path, tmp_path / "trained", epochs=0)
    with pytest.raises(cli.InputError, match="no simulated windows"):
        cli.train(tmp_path, tmp_path / "trained", threads=1)


@needs_tabix
def test_a_window_without_sites_is_left_out(tmp_path):
    # chromosome 2 is named in the header but has no records, as in a VCF split by chromosome
    vcf = example_vcf(tmp_path, lambda lines: [lines[0], "##contig=<ID=2,length=30000000>", *lines[1:]])
    mask = tmp_path / "mask.bed"
    mask.write_text("2\t0\t30000000\n")
    with pytest.raises(cli.InputError, match="no window"):
        cli.run(vcf, mask, tmp_path / "out", mu=1e-8, rec_rate=1e-8)
    assert "no sites in the VCF; left out" in (tmp_path / "out" / "log.txt").read_text()


def test_errors_are_reported_without_a_traceback():
    with pytest.raises(SystemExit, match="^necast: give the mutation rate"):
        cli.main(["run", str(VCF), "--mask", str(MASK), "--out", "unused", "--rec-rate", "1e-8"])


def test_default_chromosomes():
    vcf = SimpleNamespace(seqnames=["chr1", "chr1_KI270706v1_random", "chrEBV", "chrX", "chr2"])
    mask = {name: None for name in ("1", "1_KI270706v1_random", "EBV", "X")}
    assert cli.pick_chromosomes(vcf, mask, None) == ["1", "1_KI270706v1_random", "EBV"]
    assert cli.pick_chromosomes(vcf, mask, None, autosomes={"1": None, "2": None}) == ["1"]    # --human
    assert cli.pick_chromosomes(vcf, mask, "chr1,EBV", autosomes={"1": None}) == ["1", "EBV"]
    with pytest.raises(cli.InputError, match="not in the mask"):
        cli.pick_chromosomes(vcf, mask, "2")


def test_human_needs_grch38(tmp_path):
    with pytest.raises(cli.InputError, match="GRCh38"):
        cli.run(VCF, MASK, tmp_path, human=True)          # the example's chromosome 1 is 30 Mb long


def test_rates_in_the_wrong_units(tmp_path):
    with pytest.raises(cli.InputError, match="--mu"):
        cli.run(VCF, MASK, tmp_path, mu=1.29, rec_rate=1e-8)
    with pytest.raises(cli.InputError, match="1 cM/Mb is 1e-8"):
        cli.run(VCF, MASK, tmp_path, mu=1.29e-8, rec_rate=1.2)
    per_bp = tmp_path / "per_bp.bed"
    per_bp.write_text("1\t0\t30000000\t1.2e-8\n")
    with pytest.raises(cli.InputError, match="multiply it by 1e8"):
        cli.run(VCF, MASK, tmp_path, mu=1.29e-8, rec_map=per_bp)
