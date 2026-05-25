import numpy as np
import torch

from necast import network
from necast.window import WINDOW_BP

NET = network.load()


def random_sites(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    positions = torch.from_numpy(np.sort(rng.choice(WINDOW_BP // 50, n, replace=False)))
    states = torch.from_numpy(rng.choice(6, (n, 32), p=[0.5, 0.3, 0.1, 0.04, 0.04, 0.02]).astype(np.uint8))
    return positions, states


def flat(rate):
    return np.array([0.0, WINDOW_BP]), np.array([rate], np.float32)


def test_order_of_individuals_does_not_matter():
    positions, states = random_sites()
    exposure = network.map_exposure(flat(1e-8), flat(1.25e-8))
    with torch.no_grad():
        a = NET.read_genotypes(positions, states, exposure)
        b = NET.read_genotypes(positions, states[:, torch.randperm(32)], exposure)
    assert torch.allclose(a, b, atol=1e-5)


def test_regions_without_sites_are_zero():
    positions, states = random_sites()               # all in the first fiftieth of the window
    with torch.no_grad():
        regions, sites, calls = NET.regions(positions, states)
    empty = sites == 0
    assert empty.sum() > 2000 and torch.all(regions[empty] == 0) and torch.all(calls[empty] == 0)
    assert sites.sum() == len(positions)


def test_map_exposure_of_flat_maps():
    e = network.map_exposure(flat(1e-8), flat(2e-8)).numpy()
    assert e.shape == (network.REGIONS, 8)
    np.testing.assert_allclose(e[:, 3], e[:, 0], rtol=1e-5)                 # relative position = centre
    # rate inside a region: a difference of float32 running sums, as in training
    np.testing.assert_allclose(e[:-1, 4], np.log1p(1e6 * 1e-8 * network.REGION), rtol=1e-3)
    np.testing.assert_allclose(e[:-1, 7], np.log1p(1e6 * 2e-8 * network.REGION), rtol=1e-3)
