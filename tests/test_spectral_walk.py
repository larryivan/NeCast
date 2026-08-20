"""Calibration and invariance tests for the spectral walk.

Every number here was measured, and the comment on each test says against
what.  The reference throughout is ``empirical_human_ne_template``, the one
recipe in the mixture built from histories inferred on real human data: over
its own draws it turns 6/9/13 times, has total variation 1.68/2.65/3.46 in
log10 Ne, and spans 0.78/1.27/1.78.  Those are the only numbers in the
generator that were not chosen by us, so they are what the synthetic walk is
held to.

The tolerances are wide on purpose.  A sampler that matched the anchor exactly
would train a model that only works on histories as tame as the anchor, so the
tails here deliberately exceed it; what the tests pin is that the *body* of the
distribution sits on the anchor and the tails go outward rather than the
distribution sitting somewhere else entirely.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from simulator.config import NE_LOG10_BOUNDS, RETIRED_GENERATOR_RECIPES, Config
from simulator.demography import (
    DEMO_PROB_FIELDS,
    SPECTRAL_RANGE_CAP,
    SPECTRAL_TV_CAP,
    sample_custom_demography,
    spectral_walk_curve,
)
from simulator.utils import time_grid


ANCHOR_TURNS = (6, 9, 13)
ANCHOR_TOTAL_VARIATION = (1.68, 2.65, 3.46)
ANCHOR_RANGE = (0.78, 1.27, 1.78)


def _only_spectral(**overrides) -> Config:
    cfg = dataclasses.replace(Config(), **overrides)
    for _, field in DEMO_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_spectral_walk = 1.0
    return cfg


def _turning_points(curve: np.ndarray) -> int:
    """Count sign changes in the slope, ignoring flats.

    A piecewise-constant curve has zero slope wherever it holds, and counting
    ``sign`` transitions through those zeros turns every plateau boundary into
    a spurious turning point -- which inflates a 22-knot curve to 39 turns
    against a true 6.  Dropping the zero differences first is what makes the
    count comparable to the anchor's.
    """

    steps = np.diff(np.asarray(curve, dtype=np.float64))
    steps = steps[steps != 0.0]
    if steps.size < 2:
        return 0
    return int((np.diff(np.sign(steps)) != 0).sum())


def _profile(n: int = 1_200, **overrides) -> dict[str, np.ndarray]:
    cfg = _only_spectral(**overrides)
    turns, variation, span = [], [], []
    recent, ancient = [], []
    for seed in range(n):
        _, target, _ = sample_custom_demography(np.random.default_rng(seed), cfg)
        target = np.asarray(target, dtype=np.float64)
        half = target.size // 2
        turns.append(_turning_points(target))
        variation.append(float(np.abs(np.diff(target)).sum()))
        span.append(float(target.max() - target.min()))
        recent.append(float(np.abs(np.diff(target[:half])).sum()))
        ancient.append(float(np.abs(np.diff(target[half:])).sum()))
    return {
        "turns": np.array(turns, dtype=np.float64),
        "total_variation": np.array(variation),
        "range": np.array(span),
        "recent": np.array(recent),
        "ancient": np.array(ancient),
    }


@pytest.fixture(scope="module")
def profile() -> dict[str, np.ndarray]:
    return _profile()


# --------------------------------------------------------------------------
# Where the body of the distribution sits
# --------------------------------------------------------------------------


def test_the_median_curve_turns_as_often_as_the_human_anchor(profile) -> None:
    """The statistic the four retired samplers missed.

    Their union was bimodal -- 15% of rows near 6 turning points and 19.5%
    near 19 to 23 -- with nothing at 9, which is where the anchor is.  Median
    9 is the whole point of the replacement, so it is pinned tightly.
    """

    assert np.median(profile["turns"]) == pytest.approx(ANCHOR_TURNS[1], abs=1.0)


def test_the_median_curve_spans_as_far_as_the_human_anchor(profile) -> None:
    """The retired union's range median was 0.68 against the anchor's 1.27.

    Half the amplitude at twice the frequency is the regime where a history
    puts almost nothing into the coalescent length spectrum, so predicting the
    mean level and ignoring the structure wins.  This is the test that stops
    the mixture drifting back into it.
    """

    assert np.median(profile["range"]) == pytest.approx(ANCHOR_RANGE[1], rel=0.15)


def test_total_variation_sits_on_the_anchor_and_the_upper_tail_exceeds_it(
    profile,
) -> None:
    variation = profile["total_variation"]
    assert np.median(variation) == pytest.approx(ANCHOR_TOTAL_VARIATION[1], rel=0.25)
    # Deliberate headroom: the model has to survive histories rougher than the
    # one anchor we happen to have.
    assert np.percentile(variation, 90) > ANCHOR_TOTAL_VARIATION[2]
    assert float(variation.max()) <= SPECTRAL_TV_CAP + 1e-6


# --------------------------------------------------------------------------
# The two axes are independent, which is what one sampler buys over four
# --------------------------------------------------------------------------


def test_all_four_corners_of_roughness_by_amplitude_are_populated(profile) -> None:
    """The four retired samplers occupied two corners and left two empty.

    ``smooth_random_walk`` was smooth and large, the three dense ones were
    rough and small; no sampler produced a rough curve at full amplitude or a
    smooth one at low amplitude.  A model trained on that can read amplitude
    off roughness, which is a correlation of our construction and not of
    population genetics.
    """

    rough = profile["turns"] >= 12
    large = profile["range"] >= 1.0
    for name, mask in (
        ("smooth+large", ~rough & large),
        ("smooth+small", ~rough & ~large),
        ("rough+large", rough & large),
        ("rough+small", rough & ~large),
    ):
        assert mask.mean() > 0.05, f"{name} corner is nearly empty"


def test_the_tilt_reaches_both_directions_of_recent_versus_ancient(profile) -> None:
    """``dense_recent_wiggle`` only ever made recent time the noisier half.

    Its measured recent/ancient variation ratio was 1.93 and the other three
    samplers were flat at 1.03, 0.96 and 0.98, so no row in the mixture was
    ever quieter recently than anciently.  Real inferred histories go both
    ways, and the envelope tilt is symmetric so the mixture does too.
    """

    ratio = profile["recent"] / np.maximum(profile["ancient"], 1e-9)
    assert np.percentile(ratio, 10) < 0.7
    assert np.percentile(ratio, 90) > 1.5
    assert np.median(ratio) == pytest.approx(1.0, abs=0.25)


# --------------------------------------------------------------------------
# The curve exists before the grid does
# --------------------------------------------------------------------------


@pytest.mark.parametrize("target_bins", [64, 96, 128])
def test_the_distribution_does_not_move_with_the_target_resolution(
    target_bins: int, profile
) -> None:
    """The reason for the rewrite as much as the calibration was.

    All four retired samplers stepped ``cfg.time_bins`` times and placed knots
    at a fixed fraction of that count, so asking for 96 bins instead of 64 did
    not resolve the same history better -- it drew a different distribution of
    histories.  Here the series is defined on continuous log time, so the grid
    only decides how finely an existing curve is read.
    """

    measured = _profile(n=400, time_bins=target_bins)
    assert np.median(measured["turns"]) == pytest.approx(
        np.median(profile["turns"]), abs=1.0
    )
    assert np.median(measured["range"]) == pytest.approx(
        np.median(profile["range"]), rel=0.1
    )


def test_a_finer_generation_grid_converges_rather_than_drifts() -> None:
    """48 generation bins has to be enough to resolve up to 48 modes.

    Averaging over the bin is what makes that true: it is a low-pass filter, so
    the modes above the grid's Nyquist limit average away inside a bin instead
    of folding back onto the ones below.  Doubling and quadrupling the grid
    then changes the answer by a few percent rather than revealing that 48 was
    reading a different distribution.
    """

    coarse = _profile(n=400, demography_generation_bins=48)
    fine = _profile(n=400, demography_generation_bins=192)
    assert np.median(fine["total_variation"]) == pytest.approx(
        np.median(coarse["total_variation"]), rel=0.15
    )
    assert np.median(fine["range"]) == pytest.approx(
        np.median(coarse["range"]), rel=0.1
    )


def test_the_stored_scalars_reproduce_the_curve_at_any_resolution() -> None:
    """The six numbers in the metadata are the whole history.

    Not a convenience: it is what lets a row generated at 64 bins be re-scored
    against a 96- or 128-bin target without regenerating the dataset, and what
    makes the label auditable after the fact.
    """

    cfg = _only_spectral()
    _, target, meta = sample_custom_demography(np.random.default_rng(7), cfg)
    generation_edges = np.geomspace(
        cfg.min_time, cfg.max_time, cfg.demography_generation_bins + 1
    )
    replayed = spectral_walk_curve(
        generation_edges,
        meta["spectral_seed"],
        meta["spectral_slope"],
        meta["spectral_n_modes"],
        meta["spectral_tilt"],
        meta["spectral_ne_range"],
        meta["spectral_level"],
    )
    np.testing.assert_allclose(
        replayed, np.log10(meta["demography_values"]), rtol=0.0, atol=1e-9
    )
    # And the same parameters on a grid the sample never saw.
    dense = spectral_walk_curve(
        np.geomspace(cfg.min_time, cfg.max_time, 4 * cfg.demography_generation_bins + 1),
        meta["spectral_seed"],
        meta["spectral_slope"],
        meta["spectral_n_modes"],
        meta["spectral_tilt"],
        meta["spectral_ne_range"],
        meta["spectral_level"],
    )
    assert dense.size == 4 * cfg.demography_generation_bins
    assert float(dense.max() - dense.min()) == pytest.approx(
        float(np.ptp(np.log10(meta["demography_values"]))), rel=0.25
    )


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------


def test_no_curve_is_clipped_into_a_plateau(profile) -> None:
    """Clipping would grow a flat stretch the sampler never chose.

    The level is moved to fit the range instead, so at the widest ranges the
    curve is centred in the band rather than pressed against its edge.
    """

    assert SPECTRAL_RANGE_CAP < NE_LOG10_BOUNDS[1] - NE_LOG10_BOUNDS[0]
    assert float(profile["range"].max()) <= SPECTRAL_RANGE_CAP + 1e-6


def test_every_value_stays_inside_the_declared_effective_size_band() -> None:
    cfg = _only_spectral()
    for seed in range(200):
        _, target, _ = sample_custom_demography(np.random.default_rng(seed), cfg)
        assert float(np.min(target)) >= NE_LOG10_BOUNDS[0] - 1e-9
        assert float(np.max(target)) <= NE_LOG10_BOUNDS[1] + 1e-9


def test_the_retired_samplers_are_gone_but_still_decode() -> None:
    from simulator.config import GENERATOR_RECIPE_IDS
    from simulator.demography import DEMO_SAMPLERS

    for name in RETIRED_GENERATOR_RECIPES:
        assert name in GENERATOR_RECIPE_IDS
        assert name not in DEMO_SAMPLERS
    assert Config().p_spectral_walk == pytest.approx(0.20)


def test_a_generated_row_replays_exactly_from_its_metadata_alone(tmp_path) -> None:
    """End to end, through the writer rather than the sampler's return value.

    The shard's array schema is fixed, so the six scalars ride in the metadata
    sidecar; this is the test that they actually arrive there and are enough.
    Without it the claim that a dataset can be re-scored at a different
    resolution is untested at the only layer that matters.
    """

    import gzip
    import json

    import numpy as np

    from simulator.config import SOURCE_PROB_FIELDS, validate_config
    from simulator.empirical_maps import EmpiricalMapStore
    from simulator.shards import generate_dataset
    from simulator.utils import bin_average_log10_ne, time_grid

    cfg = _only_spectral(
        dataset_name="spectral-replay",
        out_dir=str(tmp_path / "ds"),
        seed=4_242,
        n_samples=4,
        shard_size=4,
        seq_len=20_000,
        n_haplotypes=8,
        baseline_mu=1.0e-7,
        map_segments_min=8,
        map_segments_max=8,
    )
    for field in SOURCE_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_custom_hetero = 1.0
    validate_config(cfg)
    generate_dataset(cfg, EmpiricalMapStore())

    shard = np.load(
        next((tmp_path / "ds").rglob("*.npz")), allow_pickle=True
    )
    with gzip.open(tmp_path / "ds" / "metadata" / "samples.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]

    generation_edges = np.geomspace(
        cfg.min_time, cfg.max_time, cfg.demography_generation_bins + 1
    )
    target_edges, _ = time_grid(cfg)
    for index, row in enumerate(rows):
        curve = spectral_walk_curve(
            generation_edges,
            row["spectral_seed"],
            row["spectral_slope"],
            row["spectral_n_modes"],
            row["spectral_tilt"],
            row["spectral_ne_range"],
            row["spectral_level"],
        )
        replayed = bin_average_log10_ne(
            target_edges,
            list(generation_edges[1 : cfg.demography_generation_bins]),
            list(10.0**curve),
            [0.0] * cfg.demography_generation_bins,
        )
        np.testing.assert_allclose(
            replayed,
            shard["target_log10_ne"][index],
            rtol=0.0,
            atol=1e-6,
        )
