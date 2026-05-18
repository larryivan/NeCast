import math

import numpy as np

from necast import posterior
from necast.third_party.psmcplus.transition_matrix import Transition_Matrix

EDGES = np.concatenate(([0.0], posterior.GENERATIONS / 2e4))
RHO = 4e-3


def psmcplus_matrix(relative_inverse_size):
    matrix = Transition_Matrix(D=65, midpoint_transitions=False)
    matrix.T = EDGES
    q = matrix.write_tm(lambda_A=relative_inverse_size, lambda_B=None, T_S_index=None, T_E_index=None,
                        gamma=None, check=False, rho=RHO, exponential=True)
    return q, np.asarray(matrix.e_betas)


def test_linear_recursion_equals_the_dense_one():
    rng = np.random.default_rng(3)
    bin_bp, theta, n = 100, 1e-3, 400
    size = np.exp(rng.normal(0, 0.5, 65))
    maps = posterior.Maps(rng.choice([0.8, 1.0, 1.3], n), rng.choice([0.01, 0.5, 2.0], n), bin_bp, theta)
    model = posterior.Transitions(EDGES, RHO, size, maps.recombination)
    mids = (EDGES[:-1] + EDGES[1:]) / 2
    no_event = np.exp(-bin_bp * theta * np.outer(maps.mutation, mids))
    factorial = np.array([math.factorial(j) for j in range(bin_bp)], float)
    codes = rng.choice([0, 1, 2, 30 * bin_bp + 1, 100 * bin_bp], p=[0.85, 0.08, 0.02, 0.03, 0.02], size=n)

    fast = posterior._decode(codes, maps.mutation_index, maps.recombination_index, bin_bp, theta, mids,
                             maps.mutation, no_event, model.start, model.leave, model.stay_rate, model.up, model.down,
                             model.next_up, model.next_down, model.row_up, model.row_down, factorial,
                             np.empty((n, 65)))

    # textbook forward-backward with full matrices built from PSMC+'s own matrix
    q, expected_time = psmcplus_matrix(size)
    jumps = q - np.diag(np.diag(q))
    jumps /= (1 - np.exp(-RHO * expected_time))[:, None]

    def transition(i):
        p = 1 - np.exp(-maps.recombination[maps.recombination_index[i]] * RHO * expected_time)
        return np.diag(1 - p * jumps.sum(1)) + p[:, None] * jumps

    def emission(i):
        out = np.empty(65)
        b = maps.mutation_index[i]
        posterior._emission(out, codes[i], maps.mutation[b], no_event[b], bin_bp, theta, mids, factorial)
        return out

    forward, scale = np.zeros((n, 65)), np.zeros(n)
    f = emission(0) * model.start
    scale[0], forward[0] = f.sum(), f / f.sum()
    for i in range(1, n):
        f = (forward[i - 1] @ transition(i)) * emission(i)
        scale[i], forward[i] = f.sum(), f / f.sum()
    back = np.full(65, 1 / scale[-1])
    post = [forward[-1] * back]
    for i in range(n - 1, 0, -1):
        back = transition(i) @ (emission(i) * back) / scale[i - 1]
        post.append(forward[i - 1] * back)
    slow = np.mean([p / p.sum() for p in post], axis=0)

    assert np.allclose(fast, slow, rtol=1e-9, atol=1e-15)


def test_start_is_stationary():
    q, _ = psmcplus_matrix(np.ones(65))
    start = posterior.stationary(q)
    assert np.isclose(start.sum(), 1) and np.allclose(start @ q, start, rtol=1e-12, atol=1e-15)


def test_observation_codes():
    # hets at 50, 120 and 450 (bins 0, 1, 4); three missing sites between 120 and 450
    codes = posterior.observations(np.array([50, 120, 450]), np.array([200, 210, 300]), 100)
    assert len(codes) == (450 + 200) // 100
    # the missing sites are placed right after 120, in bin 1
    assert codes.tolist() == [1, 1 + 3 * 100, 0, 0, 1, 0]


def test_too_few_sites_give_no_posterior():
    states = np.zeros((100, 32), np.uint8)
    states[:, 0] = 1
    flat = (np.array([0.0, 30e6]), np.array([1e-8]))
    assert posterior.summarise(np.arange(100) * 1000, states, flat, flat, 30_000_000) is None
