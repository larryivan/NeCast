import numpy as np

from necast import maps

M = 1_000_000


def integral(m):
    breaks, rates = m
    return float(np.sum(np.diff(breaks) * rates.astype(np.float64)))


def test_read(tmp_path):
    path = tmp_path / "rates.bed"
    path.write_text("Chromosome\tStart\tEnd\tRate\nchr1\t100\t200\t1.0\nchr1\t300\t400\t3.0\n2\t0\t10\t5.0\n")
    rates = maps.read(path, scale=1e-8)
    breaks, values = rates["1"]
    np.testing.assert_array_equal(breaks, [100, 200, 300, 400])
    np.testing.assert_allclose(values, [1e-8, 2e-8, 3e-8])     # the hole takes the mean
    assert sorted(rates) == ["1", "2"]


def test_mutation_keeps_the_shape_and_sets_the_mean():
    physical = (np.array([0.0, 10 * M, 20 * M, 40 * M]), np.array([1e-8, 1e-12, 2e-8]))
    intervals = np.array([[0, 15 * M], [25 * M, 40 * M]])
    breaks, rates = maps.mutation(physical, intervals, 1.29e-8)
    np.testing.assert_array_equal(breaks, [0, 10 * M, 15 * M, 30 * M])
    assert abs(integral((breaks, rates)) / (30 * M) - 1.29e-8) < 1e-15
    assert rates.dtype == np.float32
    assert rates[1] / rates[0] > 1e-3 * 0.99                     # floored at 1e-3 of the mean


def test_recombination_keeps_the_genetic_length_of_masked_stretches():
    physical = (np.array([0.0, 40 * M]), np.array([1e-8]))
    intervals = np.array([[0, 15 * M], [15 * M + 500_000, 30 * M + 500_000]])
    rec = maps.recombination(physical, intervals)
    assert rec[0][-1] == 30 * M and rec[1].dtype == np.float32
    assert abs(integral(rec) - 1e-8 * (30 * M + 500_000)) < 1e-6   # the 0.5 Mb stretch is kept
    assert rec[1].max() <= maps.RATE_RANGE[1]


def test_recombination_clips_to_the_training_range():
    physical = (np.array([0.0, 10 * M, 30 * M]), np.array([0.0, 1e-3]))
    rec = maps.recombination(physical, np.array([[0, 30 * M]]))
    np.testing.assert_allclose(rec[1], np.float32(maps.RATE_RANGE))
