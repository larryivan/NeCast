import math

import numpy as np
import pytest

from necast import summaries

L = 1_000_000
FLAT = (np.array([0.0, L]), np.array([1e-8]))


def sites(*rows):
    """States of 32 individuals, one row per site; unlisted individuals are 0."""
    out = np.zeros((len(rows), 32), np.uint8)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def test_a_singleton_is_half_a_singleton_after_projection():
    # one derived allele among 64: a subsample of 32 keeps it with probability 1/2
    sfs = summaries.folded_sfs(sites([1]))
    assert sfs[0] == 0.5 and sfs[1:].sum() == 0


def test_sites_with_too_few_calls_are_dropped():
    # 20 individuals with both alleles missing leave 24 called alleles, fewer than 32
    assert summaries.folded_sfs(sites([1] + [5] * 20)).sum() == 0


def test_folding_joins_complementary_classes():
    common = summaries.folded_sfs(sites([2] * 32))          # all derived: monomorphic
    assert common.sum() == 0
    half = summaries.folded_sfs(sites([1] * 32))            # 32 of 64 derived
    assert half.sum() == pytest.approx(1 - 2 * math.comb(32, 32) * math.comb(32, 0) / math.comb(64, 32))


def test_watterson():
    a = sum(1 / i for i in range(1, 64))
    expected = math.log10(10 / (4 * a * 1e-8 * L))
    assert summaries.watterson(10, FLAT) == pytest.approx(expected, abs=1e-6)


def test_segregating_and_pairwise_sites():
    states = sites([1], [4], [3], [2, 2], [4, 0])
    # [4]: the only derived allele is half-called; it varies among called alleles
    # but not among individuals with both alleles called
    assert summaries.segregating(states).tolist() == [True, True, False, True, True]
    assert summaries.pairwise(states).tolist() == [True, False, False, True, False]


def test_linkage_disequilibrium_of_identical_sites():
    row = [1] * 8 + [0] * 24
    positions = np.array([100, 200])
    ld = summaries.linkage_disequilibrium(positions, sites(row, row), FLAT)
    distance_class = np.searchsorted(summaries.LD_EDGES, 100 * 1e-8, side="right")
    assert ld[0, distance_class] == 1.0                       # r^2 of the one pair
    assert ld[4].sum() == pytest.approx(math.log(2))          # log(1 + one pair)


def test_ibs_tracts_of_two_individuals():
    # individuals 0 and 1 share no allele at 300 and 700 (IBS0)
    positions = np.array([300, 500, 700])
    states = np.zeros((3, 2), np.uint8)
    states[[0, 2], 0] = 2
    counts, exposure = summaries.identity_by_state(positions, states, L)
    whole = counts[:, 0]                                      # the region covering the window
    assert whole[0].sum() == 1                                # the 400-bp gap between them
    assert whole[0, 9] == 1                                   # 400 bp lies in [256, 512)
    assert whole[1].sum() == 2                                # 0..300 and 700..end, cut by the window
    assert whole[2:4].sum() == 0
    assert whole[4].sum() == 2 and whole[6].sum() == 1        # runs: IBS0 at 300 and 700, IBS2 at 500
    assert exposure[0] == 3                                   # one pair called at three sites


def test_missing_calls_cut_tracts():
    positions = np.array([100, 200, 300])
    states = np.array([[2, 0], [5, 0], [2, 0]], np.uint8)
    counts, _ = summaries.identity_by_state(positions, states, L)
    whole = counts[:, 0]
    assert whole[0].sum() == 0          # no complete gap: the missing call at 200 cuts it
    assert whole[1].sum() == 2          # the window start and end
    assert whole[2].sum() == 2          # both sides of the missing call


def test_threads_do_not_change_tract_counts():
    rng = np.random.default_rng(1)
    states = rng.choice([0, 1, 2, 5], p=[0.7, 0.2, 0.08, 0.02], size=(500, 32)).astype(np.uint8)
    positions = np.sort(rng.choice(L, 500, replace=False))
    one = summaries.identity_by_state(positions, states, L, threads=1)
    four = summaries.identity_by_state(positions, states, L, threads=4)
    assert np.array_equal(one[0], four[0]) and np.array_equal(one[1], four[1])
