import numpy as np
import pytest

from necast import cli

EXAMPLE = cli.EXAMPLE
VCF, MASK = EXAMPLE / "example.vcf.gz", EXAMPLE / "mask.bed"


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


def test_errors_are_reported_without_a_traceback():
    with pytest.raises(SystemExit, match="^necast: give the mutation rate"):
        cli.main(["run", str(VCF), "--mask", str(MASK), "--out", "unused", "--rec-rate", "1e-8"])


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
