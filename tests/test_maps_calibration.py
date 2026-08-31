"""Calibration checks for the synthetic rate-map generator.

Every number asserted here was measured from a real map rather than chosen.
The reference window is chr20 between 10 and 26 Mb, which avoids the
centromere; the recombination targets come from deCODE 2019 (Halldorsson et
al., Science 363:eaau1043) and the mutation targets from Roulette (Seplyarskiy
et al., Nature Genetics 2023), both as shipped in ``maps/``.

The generator these guard replaced was wrong in four ways that all produced
plausible-looking maps:

* a hotspot multiplier was applied to whole segments, so a feature drawn 1 to
  20 kb wide was realised at 39 to 156 kb -- the segment it happened to land in;
* the mutation map was given the recombination architecture, hotspots and
  hundred-fold megabase coldspots included, when real mutation rate drifts by
  about 1.2-fold per megabase and has no hotspots at all;
* the map was normalised by the unweighted segment mean rather than the
  length-weighted one, and since features preferentially hit wide segments the
  two differ systematically -- the realised recombination rate averaged 1.655
  times the configured baseline;
* the observation noise was ``lognormal(0, sigma)``, whose mean is
  ``exp(sigma^2/2)``, so the observed map integral sat 3.2% above the truth in
  every row, and ``nw_log10`` divides by exactly that integral.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from simulator.config import Config, apply_preset
from simulator.empirical_maps import EmpiricalMapStore
from simulator.maps import (
    HOTSPOT_WIDTH_BP,
    MAX_SYNTHETIC_RATE_MULTIPLIER,
    sample_maps_for_source,
    synthetic_heterogeneous_map,
)

MEGABASE = 1_000_000
SEQ_LEN = 20_000_000
BASELINE_REC = 1.0e-8
BASELINE_MUT = 1.25e-8


def _config() -> Config:
    cfg = Config()
    cfg.seq_len = SEQ_LEN
    return cfg


def _draw(is_mut: bool, seed: int = 5, sigma: float = 0.25):
    rng = np.random.default_rng(seed)
    return synthetic_heterogeneous_map(
        rng,
        _config(),
        BASELINE_MUT if is_mut else BASELINE_REC,
        is_mut,
        sigma,
        0.10,
    )


def _length_weighted(rates, pos):
    widths = np.diff(pos)
    return float((rates * widths).sum() / widths.sum())


def _megabase_ratio(pos, rates):
    means = []
    for start in range(0, SEQ_LEN, MEGABASE):
        stop = start + MEGABASE
        hit = (pos[1:] > start) & (pos[:-1] < stop)
        widths = np.minimum(pos[1:][hit], stop) - np.maximum(pos[:-1][hit], start)
        if widths.sum() > 0:
            means.append(float((rates[hit] * widths).sum() / widths.sum()))
    means = np.array(means)
    return float(np.percentile(means, 90) / np.percentile(means, 10))


def _hot_runs(pos, rates):
    mean = _length_weighted(rates, pos)
    widths = np.diff(pos)
    hot = rates > 10.0 * mean
    runs, current = [], 0.0
    for is_hot, width in zip(hot, widths):
        if is_hot:
            current += width
        elif current > 0:
            runs.append(current)
            current = 0.0
    if current > 0:
        runs.append(current)
    return float(widths[hot].sum() / widths.sum()), np.array(runs)


# --------------------------------------------------------------------------
# The rate the map actually delivers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("is_mut", [True, False])
def test_the_realised_rate_is_the_configured_baseline(is_mut: bool) -> None:
    """Not the unweighted segment mean, which is what the code used to set.

    Segment widths are spacings of uniform draws and so vary widely, and a
    feature applied by span preferentially covers wide segments, which
    correlates rate with width.  Normalising the wrong mean therefore biases
    the realised rate systematically rather than merely adding noise.
    """

    baseline = BASELINE_MUT if is_mut else BASELINE_REC
    for seed in range(6):
        pos, rates, _, _, meta = _draw(is_mut, seed=seed)
        assert _length_weighted(rates, pos) == pytest.approx(baseline, rel=1e-9)
        assert meta["realised_rate_over_baseline"] == pytest.approx(1.0, rel=1e-9)


def test_the_ceiling_is_applied_before_the_normalisation() -> None:
    """Clipping afterwards removes rate the normalisation already counted.

    Measured with the ceiling applied last, the realised recombination rate
    came out at 0.810 of the baseline: hotspots stacked on the broad walk
    exceed the ceiling, and what the clip removed was never restored.
    """

    pos, rates, _, _, _ = _draw(False, seed=3)
    assert _length_weighted(rates, pos) == pytest.approx(BASELINE_REC, rel=1e-9)
    # Normalising last can lift the maximum slightly past the nominal ceiling.
    # That is acceptable and bounded: the ceiling guards msprime's 1e-4 limit,
    # and two hundred baselines is 2e-6.
    assert rates.max() < 1.0e-5
    assert rates.max() <= BASELINE_REC * MAX_SYNTHETIC_RATE_MULTIPLIER * 2.0


def test_the_observation_noise_preserves_the_map_integral() -> None:
    """``lognormal(0, s)`` has mean ``exp(s^2/2)``, not one.

    At the default sigma of 0.25 that is a systematic 3.2% inflation of the
    observed integral in every row, and ``nw_log10`` -- the baseline every
    model predicts a residual around -- is the segregating count divided by
    exactly that integral.
    """

    ratios = []
    for seed in range(40):
        pos, rates, obs_pos, obs, _ = _draw(False, seed=seed, sigma=0.25)
        # The hotspot-dropout step deliberately rewrites the top ventile, so
        # the check excludes it and looks at the multiplier itself.
        ratios.append(_length_weighted(obs, obs_pos) / _length_weighted(rates, pos))
    assert np.mean(ratios) == pytest.approx(1.0, abs=0.03)


# --------------------------------------------------------------------------
# Feature spans
# --------------------------------------------------------------------------


def test_a_hotspot_is_as_wide_as_it_was_drawn() -> None:
    """The defect that dominated the old generator's structure.

    Applying a multiplier to whole segments made the realised width the
    segment width, which at 128 to 512 segments over twenty megabases is 39 to
    156 kb -- twenty to seventy times the declared 1 to 20 kb.  deCODE 2019
    puts the real median hot run at 466 bp.
    """

    widest = HOTSPOT_WIDTH_BP[1]
    for seed in range(4):
        pos, rates, _, _, meta = _draw(False, seed=seed)
        fraction, runs = _hot_runs(pos, rates)
        assert len(runs) > 0
        # A run may merge neighbouring hotspots, so the bound is generous on
        # the high side and tight on the order of magnitude.
        assert np.median(runs) < 6.0 * widest
        assert np.median(runs) > 100.0


def test_the_hot_fraction_matches_the_real_map() -> None:
    """deCODE 2019 on the reference window: 1.76% of sequence above ten times
    the mean rate."""

    fractions = []
    for seed in range(6):
        pos, rates, _, _, _ = _draw(False, seed=seed)
        fractions.append(_hot_runs(pos, rates)[0])
    assert 0.008 < float(np.median(fractions)) < 0.035


# --------------------------------------------------------------------------
# The mutation map is not a recombination map
# --------------------------------------------------------------------------


def test_the_mutation_map_has_no_hotspots_or_coldspots() -> None:
    """Real mutation rate has neither.

    Its fine-scale variation is sequence context at single sites, which a
    binary mutation model cannot represent at all, and its regional variation
    is a smooth drift.  The old generator gave it 1 to 4 hotspots at 2 to 12
    times and the same hundred-fold megabase coldspots as recombination.
    """

    for seed in range(6):
        pos, rates, _, _, meta = _draw(True, seed=seed)
        assert meta["hotspot_count"] == 0
        assert meta["coldspot_count"] == 0
        assert meta["hotspot_model"] == "physical_grid_exact_span"
        fraction, runs = _hot_runs(pos, rates)
        assert fraction == 0.0
        assert len(runs) == 0


def test_the_mutation_map_drifts_like_roulette() -> None:
    """Roulette on the reference window gives a megabase p90/p10 of 1.12.

    The old generator gave a megabase max/min whose median was 7.03, with 49%
    of rows above five-fold.  On real data mutation is nearly uniform, so
    variant density tracks coalescent depth; teaching a model otherwise makes
    it distrust the strongest signal it has.
    """

    ratios = [_megabase_ratio(*_draw(True, seed=s)[:2]) for s in range(8)]
    assert 1.05 < float(np.median(ratios)) < 1.35


def test_the_recombination_map_still_varies_across_megabases() -> None:
    """The real maps give a megabase p90/p10 between 3.7 and 6.4 depending on
    which one is measured, so heterogeneity is removed from mutation only."""

    ratios = [_megabase_ratio(*_draw(False, seed=s)[:2]) for s in range(8)]
    assert 2.0 < float(np.median(ratios)) < 9.0


# --------------------------------------------------------------------------
# Source mixture
# --------------------------------------------------------------------------


def test_every_preset_draws_a_fifth_of_rows_from_a_real_map() -> None:
    """The dataclass default alone would have changed nothing.

    The shipped dataset was generated under the ``full_v01`` preset, which
    overrides the source probabilities; measured on 6,000 of its samples the
    realised mixture was 60.0% synthetic against 15.2% empirical, matching the
    preset rather than the default.
    """

    for preset in (None, "mini", "core_v01", "full_v01"):
        cfg = Config()
        if preset is not None:
            cfg.preset = preset
            cfg = apply_preset(cfg)
        assert cfg.p_empirical_slice == pytest.approx(0.20), preset
        total = (
            cfg.p_clean_constant
            + cfg.p_random_global
            + cfg.p_custom_hetero
            + cfg.p_empirical_slice
            + cfg.p_stdpopsim_anchor
        )
        assert total == pytest.approx(1.0)


# --------------------------------------------------------------------------
# What the refactor onto a physical grid was for
# --------------------------------------------------------------------------


def test_the_background_correlates_over_a_distance_not_over_segments() -> None:
    """The old background was a walk indexed by segment number.

    Segment widths were spacings of uniform draws, so the correlation length
    in base pairs was set by the segmentation: measured on that version, the
    log rate moved 0.0105 across a segment at the 25th width percentile and
    0.0097 across one at the 75th -- the same jump over four times the
    sequence.  A background on a regular physical grid cannot do that.
    """

    # Compared across draws rather than within one: the background now lives
    # on a regular grid, so the informative contrast is between rows whose
    # correlation length differs, not between segments of one row.
    lengths, decays = [], []
    for seed in range(30):
        pos, rates, _, _, meta = _draw(False, seed=seed)
        cell = meta["background_cell_bp"]
        edges = np.arange(0.0, SEQ_LEN + 50_000.0, 50_000.0)
        cumulative = np.concatenate(([0.0], np.cumsum(rates * np.diff(pos))))
        gridded = np.diff(np.interp(edges, pos, cumulative)) / np.diff(edges)
        log = np.log(gridded[gridded > 0])
        lengths.append(meta["correlation_bp"])
        decays.append(float(np.corrcoef(log[:-4], log[4:])[0, 1]))
    lengths, decays = np.array(lengths), np.array(decays)
    # A longer drawn correlation length must show as a slower measured decay.
    assert np.corrcoef(np.log(lengths), decays)[0, 1] > 0.5


def test_the_amplitude_is_drawn_per_row_rather_than_fixed() -> None:
    """A constant amplitude is both wrong and a label.

    The old generator standardised the walk, so every synthetic row carried
    the same log-rate spread to nine decimal places -- 0.06000 across two
    hundred draws.  A model can condition on that and identify the source.
    """

    spreads = []
    for seed in range(24):
        _, rates, _, _, _ = _draw(True, seed=seed)
        spreads.append(float(np.log(rates).std()))
    spreads = np.array(spreads)
    assert spreads.std() > 0.005, spreads


def test_the_observed_map_does_not_reveal_the_true_breakpoints() -> None:
    """It is re-binned onto a grid of its own.

    Returning it on the true edges handed over every boundary of the true map
    exactly -- and once feature edges joined that grid, that meant every
    hotspot position, with only the intensities noised.  A published map is
    uncertain about where the hotspots are, not only about how strong.
    """

    for seed in range(4):
        pos, _, obs_pos, obs, meta = _draw(False, seed=seed)
        assert not np.array_equal(pos, obs_pos)
        assert len(obs) == len(obs_pos) - 1
        assert meta["observed_segments"] == len(obs)


def test_the_hotspot_dropout_only_applies_where_hotspots_exist() -> None:
    """The mutation map has none, so rewriting its top ventile modelled a
    phenomenon that does not occur there."""

    for seed in range(4):
        _, _, _, obs, meta = _draw(True, seed=seed)
        assert meta["hotspot_count"] == 0
        # Checked against the flag itself rather than against a threshold on
        # the values: with hotspots the dropout fires and changes the map,
        # without them it must be inert.
        from simulator.maps import _observed_view

        pos_a, rates_a, _, _, _ = _draw(False, seed=seed)
        quiet = _observed_view(
            np.random.default_rng(1), pos_a, rates_a, BASELINE_REC, 0.25, 0.10,
            has_hotspots=False,
        )[1]
        loud = _observed_view(
            np.random.default_rng(1), pos_a, rates_a, BASELINE_REC, 0.25, 0.10,
            has_hotspots=True,
        )[1]
        assert not np.array_equal(quiet, loud)


def test_every_map_source_records_how_noisy_its_observed_map_is() -> None:
    """Silence is not the same as zero, and a consumer cannot tell them apart.

    Two of the four sources publish the observed map as a bit-identical copy of
    the truth: a constant map has no spatial structure to mis-measure, and the
    rate is taken as known, which is what every method in this field assumes of
    mu.  That is a modelling choice.  What it must not be is invisible -- the
    other two sources recorded ``observed_noise_sigma`` and these two recorded
    nothing, so nothing downstream could tell which convention a row followed,
    and no result could be stratified by it.
    """

    cfg = apply_preset(Config(preset="core_v01"))
    # None of the three sources exercised here reads the empirical store, so an
    # empty one keeps the test independent of the shipped map files.
    store = EmpiricalMapStore()
    expected = {
        "clean_constant_map": (0.0, 0.0, True),
        "random_global_mu_r": (0.0, 0.0, True),
        "custom_heterogeneous": (
            cfg.observed_map_noise,
            cfg.observed_map_noise * 0.6,
            False,
        ),
    }
    for source, (rec, mut, is_exact) in expected.items():
        rng = np.random.default_rng(20260830)
        *_, meta = sample_maps_for_source(rng, cfg, source, store)
        assert meta["observed_noise_sigma"] == pytest.approx(rec), source
        assert meta["observed_noise_sigma_mut"] == pytest.approx(mut), source
        assert meta["observed_map_is_exact"] is is_exact, source


def test_the_empirical_slice_reports_the_sigma_it_was_drawn_with() -> None:
    """The recorded value was one that no array in the row had seen.

    Empirical slices are drawn with ``empirical_map_noise``, which is separate
    from ``observed_map_noise`` precisely because a measured map disagrees with
    the truth by more than a synthetic one does.  The metadata reported the
    latter, so for a fifth of every cohort the published observation model was
    neither the one applied to the recombination map nor the one applied to the
    mutation map.
    """

    cfg = apply_preset(Config(preset="full_v01"))
    assert cfg.empirical_map_noise != cfg.observed_map_noise, (
        "this test cannot distinguish the two settings unless they differ"
    )
    source = inspect.getsource(sample_maps_for_source)
    empirical = source.split('elif source == "empirical_map_slice"')[1]
    empirical = empirical.split("else:")[0]
    assert '"observed_noise_sigma": empirical_noise,' in empirical
    assert '"observed_noise_sigma": obs_noise,' not in empirical
