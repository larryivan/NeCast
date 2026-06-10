import numpy as np
import pytest

from necast import network, summaries, train
from necast.window import WINDOW_BP


def test_six_states():
    haplotypes = np.array([[0, 1, 1, 1], [1, 0, 0, 0]], np.uint8)       # 2 sites x 2 individuals
    packed = np.packbits(haplotypes, axis=1, bitorder="little")
    missing = np.array([1 * 4 + 0])                                        # site 1, first allele of individual 0
    np.testing.assert_array_equal(train.six_states(packed, missing, 4), [[1, 2], [3, 0]])


def test_individuals_are_the_ones_the_paper_read():
    # the first 8 of the 32 the published network read in benchmark window sim_00000000
    np.testing.assert_array_equal(train.individuals("sim_00000000", 64)[:8], [39, 36, 24, 55, 28, 14, 34, 4])


def test_simulate_and_prepare(tmp_path):
    pytest.importorskip("msprime")
    from necast.cli import main
    main(["simulate", "--preset", "mini", "--seq-len", str(WINDOW_BP), "--n-haplotypes", "64",
          "--n-samples", "1", "--seed", "3", "--out-dir", str(tmp_path / "sim")])
    (name, w, target), = train.read_simulations(tmp_path / "sim")
    assert w.states.shape[1] == 32 and len(target) == 64 and w.rec_map[0][-1] == WINDOW_BP
    train.prepare(tmp_path / "sim", tmp_path / "out", threads=4, log=lambda message: None)
    with np.load(tmp_path / "out" / "windows" / f"{name}.npz") as z:
        assert len(z["positions"]) == summaries.segregating(w.states).sum()
        assert z["sfs"].shape == (16,) and z["posterior"].shape == (256,)


def test_train_one_epoch(tmp_path):
    rng = np.random.default_rng(0)
    folder = tmp_path / "windows"
    folder.mkdir()
    flat = np.array([0.0, WINDOW_BP])
    for i in range(12):
        positions = np.sort(rng.choice(WINDOW_BP, 500, replace=False))
        states = rng.choice(3, (500, 32), p=[0.6, 0.3, 0.1]).astype(np.uint8)
        s = summaries.summarise(positions, states, (flat, np.array([1e-8], np.float32)),
                                (flat, np.array([1.25e-8], np.float32)), WINDOW_BP)
        np.savez_compressed(folder / f"sim_{i:08d}.npz", positions=positions, states=states,
                            rec_breaks=flat, rec_rates=np.array([1e-8], np.float32), mut_breaks=flat,
                            mut_rates=np.array([1.25e-8], np.float32), target=rng.normal(4, 0.3, 64).astype(np.float32),
                            sfs=s.sfs, ld=s.ld, ibs=s.ibs, ibs_exposure=s.ibs_exposure,
                            log10_watterson=np.float32(s.log10_watterson),
                            posterior=rng.random(256).astype(np.float32) if i % 2 else np.full(256, np.nan, np.float32))
    train.train(tmp_path, epochs=1, device="cpu", log=lambda message: None)
    assert len((tmp_path / "training.tsv").read_text().splitlines()) == 2
    trained = network.load(tmp_path / "necast.pt")
    assert trained.offset.abs().sum() > 0 and trained.sfs_std.min() > 0
