"""Tests for the continuous scenario taxonomy.

The mixture went from 27 named recipes to 11 because the names were not
carrying information.  Measured pairwise on the 64-bin label with a ridge
classifier, ``zigzag`` and ``zigzag_strong`` scored 0.283 against a 0.500
chance rate -- they were the same function -- and the four recent-bottleneck
recipes, 14.4% of all rows between them, scored 0.619 to 0.786.  What replaced
them is not fewer scenarios but the same scenarios with their parameters
exposed: an axis that is continuous in nature is one sampler with a continuous
parameter here, and the parameter goes in the metadata so any grouping the old
names supported can still be made at analysis time, on realised values rather
than on which sampler happened to be called.

These tests pin the properties that made the consolidation worth doing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from simulator.config import (
    DEMOGRAPHY_PROB_FIELDS,
    NE_BOUNDS,
    NE_LOG10_BOUNDS,
    Config,
)
from simulator.demography import (
    DEMO_PROB_FIELDS,
    DEMO_SAMPLERS,
    _canonical_smcpp_sawtooth,
    _canonical_zigzag_1s14,
    sample_custom_demography,
    trajectory_shape_values,
)
from simulator.utils import bin_average_log10_ne, time_grid
from simulator.stdpopsim_templates import (
    drawable_stdpopsim_templates,
    load_stdpopsim_templates,
)


def _only(name: str, **overrides) -> Config:
    cfg = dataclasses.replace(Config(), **overrides)
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    setattr(cfg, dict(DEMO_PROB_FIELDS)[name], 1.0)
    return cfg


def _draw(name: str, n: int = 400, **overrides):
    cfg = _only(name, **overrides)
    for seed in range(n):
        demography, target, meta = sample_custom_demography(
            np.random.default_rng(seed), cfg
        )
        yield np.asarray(target, dtype=np.float64), meta


# --------------------------------------------------------------------------
# The axes that used to be separate samplers
# --------------------------------------------------------------------------


def test_a_size_event_reaches_both_directions_and_every_depth() -> None:
    """``expansion`` and ``contraction`` were separate recipes from the
    bottlenecks, which is a sign convention promoted to a taxonomy."""

    depths = [meta["event_log_depths"][0] for _, meta in _draw("size_event")]
    assert min(depths) < -1.0, "no deep bottleneck drawn"
    assert max(depths) > 1.0, "no large expansion drawn"
    # Neither sign may dominate: a taxonomy hidden in a sign would show here.
    downward = sum(1 for value in depths if value < 0) / len(depths)
    assert 0.35 < downward < 0.65


def test_a_size_event_spans_transient_to_permanent() -> None:
    """The duration axis is what separated ``recent_bottleneck`` from
    ``expansion``: one had an end and the other did not.

    An event whose end falls past the observable window is a permanent size
    change, and that is a value of the duration parameter rather than a
    different kind of history.
    """

    permanent = [meta["event_permanent"] for _, meta in _draw("size_event")]
    assert 0.05 < sum(permanent) / len(permanent) < 0.60


def test_the_recovery_level_is_not_pinned_to_the_starting_level() -> None:
    """The axis the retired samplers hid most completely.

    ``recent_bottleneck`` returned the population to 0.5-2x where it started,
    ``recent_founder_recovery`` left it far above, and nothing sat between --
    which is why a classifier separated those two at 0.619, barely above
    chance, while the space between them stayed empty.
    """

    ratios = []
    for target, meta in _draw("size_event"):
        if meta["event_permanent"]:
            continue
        ratios.append(float(target[0] - target[-1]))
    assert len(ratios) > 100
    assert np.percentile(ratios, 10) < -0.4
    assert np.percentile(ratios, 90) > 0.4
    # And the middle is populated, not just the two ends.
    middle = [value for value in ratios if -0.3 < value < 0.3]
    assert len(middle) > 0.15 * len(ratios)


def test_multi_event_produces_conflicting_directions_on_its_own() -> None:
    """``ancient_recent_conflict`` had its own recipe at 0.3% of rows.

    Two events with independent signs disagree about half the time, so the
    conflict case does not need to be named to be generated.
    """

    conflicts = 0
    total = 0
    for _, meta in _draw("multi_event"):
        depths = meta["event_log_depths"]
        if len(depths) < 2:
            continue
        total += 1
        conflicts += int(min(depths) < 0 < max(depths))
    assert total > 300
    assert 0.3 < conflicts / total < 0.9


def test_the_oscillation_is_placed_rather_than_spanning_the_whole_grid() -> None:
    """The sawtooth used to lay every curve on the same ladder.

    Measured over two thousand draws of the version this replaces, the first
    turning point fell between 86 and 117 generations and the last between
    341,000 and 450,000 -- so every sawtooth covered the same three and a half
    decades and only the number of turns inside it varied.  A population that
    oscillated for a while and then settled was not in the support.
    """

    onsets, ends = [], []
    for _, meta in _draw("sawtooth", n=600):
        onsets.append(meta["oscillation_onset"])
        ends.append(meta["oscillation_end"])
    # Both edges have to move by more than a decade across the prior.
    assert np.log10(np.percentile(onsets, 90) / np.percentile(onsets, 10)) > 1.0
    assert np.log10(np.percentile(ends, 90) / np.percentile(ends, 10)) > 1.0
    spans = np.log10(np.asarray(ends) / np.asarray(onsets))
    # And Zigzag_1S14, which spans 33 to 34,133 generations, has to sit inside
    # the support rather than at its edge.
    assert np.percentile(spans, 10) < 3.01 < np.percentile(spans, 98)


def test_the_turns_run_from_irregular_to_periodic() -> None:
    """The published sawtooths are periodic; inferred ones are not.

    The old spacing was a geometric ladder with a fixed 12% jitter, which could
    be neither: it measured a coefficient of variation of 0.19 on the log-time
    gaps every time, with no way to reach a regular ladder or an irregular one.
    """

    spreads = []
    for _, meta in _draw("sawtooth", n=800):
        breaks = np.asarray(meta["demography_breaks"], dtype=np.float64)
        if breaks.size < 4:
            continue
        gaps = np.diff(np.log(breaks))
        spreads.append(float(gaps.std() / max(gaps.mean(), 1e-9)))
    assert np.percentile(spreads, 10) < 0.18, "no near-periodic ladder drawn"
    assert np.percentile(spreads, 90) > 0.55, "no irregular ladder drawn"


def test_the_swing_amplitude_drifts_across_a_history() -> None:
    """An oscillation that grows or decays, not one that holds a single
    amplitude from the present to the root."""

    drifts = [meta["oscillation_amplitude_drift"] for _, meta in _draw("sawtooth", n=400)]
    assert min(drifts) < -0.8 and max(drifts) > 0.8
    assert abs(float(np.median(drifts))) < 0.3, "neither direction may dominate"


def test_the_exponential_rate_is_signed_rather_than_two_recipes() -> None:
    directions = [meta["exponential_direction"] for _, meta in _draw("exponential")]
    share = directions.count("growth") / len(directions)
    assert 0.35 < share < 0.65


# --------------------------------------------------------------------------
# The chain that could not be drawn before
# --------------------------------------------------------------------------


def test_the_out_of_africa_chain_keeps_its_phases_in_order() -> None:
    """The point of the recipe.

    ``ancient_recent_compound`` drew its two events independently, so it could
    put the recovery before the bottleneck as readily as after.  A serial
    founder history is a chain -- expansion, then departure and a long small
    period, then growth -- and if the order is not enforced it is not that
    history, it is two unrelated events.
    """

    for target, meta in _draw("ooa_chain", n=200):
        breaks = meta["demography_breaks"]
        assert breaks == sorted(breaks)
        assert len(breaks) == 3
        growth_onset, bottleneck_time, expansion_time = breaks
        assert growth_onset < bottleneck_time < expansion_time
        assert meta["ooa_bottleneck_factor"] < 1.0, "the bottleneck must shrink"
        assert meta["ooa_growth_factor"] > 1.0, "the recovery must grow"
        assert meta["ooa_expansion_factor"] > 1.0
        # The label has to dip in the middle.  A chain whose minimum sits at
        # either end is a monotone trajectory wearing the name.
        assert float(target.min()) < float(min(target[0], target[-1])) - 0.05


def test_the_out_of_africa_chain_holds_the_bottleneck_open() -> None:
    """A founder event that recovers immediately is not this history.

    The low period has to last long enough to be visible: the coalescent reads
    it as a raised coalescence rate over a stretch of time, and a pulse of a
    few generations leaves nothing to read.
    """

    durations = [meta["ooa_bottleneck_duration"] for _, meta in _draw("ooa_chain", n=300)]
    assert np.percentile(durations, 10) > 200.0
    assert np.median(durations) > 700.0


# --------------------------------------------------------------------------
# The held-out catalogue
# --------------------------------------------------------------------------


def test_the_literature_anchors_carry_weight() -> None:
    """At zero weight they are never drawn, and an evaluation set that was
    never generated cannot be held out of anything."""

    cfg = Config()
    assert cfg.p_stdpopsim_single_pop > 0.0
    models = drawable_stdpopsim_templates(cfg)
    # Enough replicates per model to read a spread, not just a mean.  At a
    # hundred thousand rows this is about 133 apiece.
    assert cfg.p_stdpopsim_single_pop / len(models) * 100_000 > 100


def test_the_frozen_catalogue_matches_the_published_event_table() -> None:
    """The freeze is checked against a source that is not itself.

    ``scripts/build_stdpopsim_single_pop.py`` reads whichever stdpopsim happens
    to be installed at build time, so nothing in the data file proves it is
    right.  ``_canonical_zigzag_1s14`` is the Schiffels & Durbin table typed in
    by hand from the paper, and it has to agree with the frozen entry exactly.
    """

    cfg = Config()
    generation_edges, generation_mids = time_grid(
        dataclasses.replace(cfg, time_bins=cfg.demography_generation_bins)
    )
    breaks, values, meta = _canonical_zigzag_1s14(
        np.random.default_rng(0), cfg, generation_edges, generation_mids
    )
    frozen = next(
        template
        for template in load_stdpopsim_templates(cfg)
        if template.model_id == "Zigzag_1S14"
    )
    np.testing.assert_array_equal(np.asarray(frozen.breaks), np.asarray(breaks))
    np.testing.assert_array_equal(np.asarray(frozen.values), np.asarray(values))
    np.testing.assert_array_equal(
        np.asarray(frozen.growth_rates),
        np.asarray(meta["demography_growth_rates"]),
    )


def test_the_smcpp_sawtooth_is_the_same_history_as_zigzag_1s14() -> None:
    """Why there is one catalogue entry and not two canonical recipes.

    The scrm command in the SMC++ Online Methods and the stdpopsim Zigzag_1S14
    catalogue entry are the same model published twice: identical growth rates
    in all six epochs, one breakpoint differing by 0.03%, and sizes differing
    only by where the rounding accumulates.  On the 64-bin label that is at
    most 5.4e-4, against a measurement noise floor of 0.0019 -- so carrying
    both cost 0.4% of every dataset and bought a difference this project cannot
    measure.
    """

    cfg = dataclasses.replace(Config(), time_bins=64)
    edges, _ = time_grid(cfg)
    generation_edges, generation_mids = time_grid(
        dataclasses.replace(cfg, time_bins=cfg.demography_generation_bins)
    )
    labels = []
    for sampler in (_canonical_zigzag_1s14, _canonical_smcpp_sawtooth):
        breaks, values, meta = sampler(
            np.random.default_rng(0), cfg, generation_edges, generation_mids
        )
        labels.append(
            bin_average_log10_ne(
                edges, breaks, values, meta["demography_growth_rates"]
            )
        )
    difference = float(np.abs(labels[0] - labels[1]).max())
    assert difference < 0.0019, "if they ever diverge, they are two models again"
    assert difference > 0.0, "and they are not literally the same numbers"


def test_the_stdpopsim_catalogue_is_broad_and_verbatim() -> None:
    templates = load_stdpopsim_templates(Config())
    assert len(templates) == 18
    assert len({template.species for template in templates}) == 8
    # Nothing is excluded from the draw at present.  Zigzag_1S14 was, while it
    # had a recipe of its own; the mechanism stays because the reason to use it
    # would recur, but every model is drawable now.
    drawable = drawable_stdpopsim_templates(Config())
    assert len(drawable) == 18
    assert len({template.species for template in drawable}) == 8
    # Every trajectory sits inside the declared band, by construction rather
    # than by clipping: clipping would flatten the very features these models
    # are held out to test.
    for template in templates:
        assert min(template.values) >= NE_BOUNDS[0] * (1 - 1e-9)
        assert max(template.values) <= NE_BOUNDS[1] * (1 + 1e-9)


def test_held_out_rows_are_labelled_so_a_split_can_find_them() -> None:
    seen = set()
    for _, meta in _draw("stdpopsim_single_pop", n=200):
        assert meta["held_out_evaluation"] is True
        assert meta["literature_demography_anchor"] is True
        seen.add(meta["stdpopsim_model_id"])
    assert len(seen) == 18, "every catalogue model has to be reachable"


def test_a_catalogue_model_draws_one_history_every_time() -> None:
    """These are data replicates of a known history, not training variety.

    Reading the model's bias and variance straight out of the dataset is the
    whole reason the catalogue carries weight, and that only works if each
    model's trajectory really is identical from row to row -- only the map, the
    mutation rate and the sequencing noise may differ.
    """

    by_model: dict[str, list] = {}
    for target, meta in _draw("stdpopsim_single_pop", n=300):
        by_model.setdefault(meta["stdpopsim_model_id"], []).append(target)
    assert len(by_model) > 10
    for model_id, targets in by_model.items():
        for target in targets[1:]:
            np.testing.assert_array_equal(target, targets[0], err_msg=model_id)


# --------------------------------------------------------------------------
# Contracts across the whole mixture
# --------------------------------------------------------------------------


def test_every_recipe_reports_severity_as_a_realised_ratio() -> None:
    """The old field meant three different things and was sometimes a lie.

    It was bottleneck-over-present in three samplers, founder-over-ancient in a
    fourth and founder-over-present in a fifth, and where the effective size
    hit its floor the drawn value was written rather than the realised one --
    on ``recent_bottleneck_extreme`` that misreported 37% of rows, a declared
    median depth of 0.0032 arriving in the label as 0.0076.
    """

    edges, _ = time_grid(Config())
    for name in sorted(DEMO_SAMPLERS):
        for target, meta in _draw(name, n=40):
            # Read off the trajectory msprime will actually run, not the
            # stored boundary values: where a published model rounds its
            # growth rates, integrating the rate and reading the stored size
            # differ in the sixth digit, and the integral is the one that
            # describes the simulation.
            shape = trajectory_shape_values(
                edges,
                meta["demography_breaks"],
                meta["demography_values"],
                meta["demography_growth_rates"],
            )
            severity = float(meta["event_severity"])
            assert 0.0 < severity <= 1.0 + 1e-9, name
            # One definition, everywhere: smallest size over largest.
            assert severity == pytest.approx(float(shape.min() / shape.max())), name


def test_the_mixture_is_normalised_and_every_recipe_is_reachable() -> None:
    cfg = Config()
    total = sum(float(getattr(cfg, field)) for field in DEMOGRAPHY_PROB_FIELDS)
    assert total == pytest.approx(1.0)
    for name, field in DEMO_PROB_FIELDS:
        assert float(getattr(cfg, field)) > 0.0, f"{name} is unreachable"
    assert len(DEMO_SAMPLERS) == 12


def test_every_recipe_id_names_a_live_sampler() -> None:
    """Nothing is carried for a dataset that predates a change.

    The tables used to hold thirty retired names so an older dataset could
    still decode its rows, but that decode was unreachable: the schema version
    is checked first and refuses the whole file.  Ids are dense and start at
    zero, and every one of them is a sampler this generator still has.
    """

    from simulator.config import GENERATOR_RECIPE_IDS

    assert set(GENERATOR_RECIPE_IDS) == set(DEMO_SAMPLERS) | {"stdpopsim"}
    assert sorted(GENERATOR_RECIPE_IDS.values()) == list(
        range(len(GENERATOR_RECIPE_IDS))
    )


def test_the_level_prior_covers_the_head_s_whole_output_range() -> None:
    """The bottom decade of the prediction range used to be untrained.

    Every recipe carried its own hard-coded effective-size range and they had
    all been picked the same way -- 3,000 to 200,000, 2,000 to 200,000, 3,000
    to 120,000.  Weighted over the mixture that left 0.000, 0.000, 0.003 and
    0.008 of the weight in the four lowest quarter-decade bands of the mean
    level, against a prediction head that runs from 100 to a million.  Real
    populations live down there: the vaquita's effective size is under three
    thousand, and so is that of most managed and island populations.
    """

    cfg = Config()
    levels, weights = [], []
    for name, field in DEMO_PROB_FIELDS:
        probability = float(getattr(cfg, field))
        drawn = [float(target.mean()) for target, _ in _draw(name, n=250)]
        levels.extend(drawn)
        weights.extend([probability / len(drawn)] * len(drawn))
    levels_arr = np.asarray(levels)
    weights_arr = np.asarray(weights)
    counts, _ = np.histogram(
        levels_arr, bins=np.linspace(*NE_LOG10_BOUNDS, 17), weights=weights_arr
    )
    share = counts / counts.sum()
    # No quarter-decade band anywhere in the head's range may be starved, and
    # none may hold so much that the prior is a spike.
    assert share.min() > 0.002, "part of the output range is untrained"
    assert share.max() < 0.20, "the level prior is concentrated"


def test_no_recipe_announces_itself_by_its_level() -> None:
    """A recipe whose level distribution is much tighter than the rest is
    partly identifiable from the level alone, which is a shortcut and not a
    fact about population genetics.

    The empirical templates are the case worth watching: 191 curves whose
    levels sit within about half a decade of each other, which left this recipe
    at a spread of 0.24 while every other recipe ran from 0.49 to 0.80.
    """

    spreads = {}
    for name, _ in DEMO_PROB_FIELDS:
        if name in ("canonical_zigzag_1s14", "stdpopsim_single_pop"):
            continue  # fixed published trajectories; their spread is not ours
        spreads[name] = float(
            np.std([float(target.mean()) for target, _ in _draw(name, n=300)])
        )
    widest = max(spreads.values())
    for name, spread in spreads.items():
        assert spread > 0.4 * widest, f"{name} is identifiable by its level alone"


def test_the_published_sawtooth_shape_is_a_class_the_mixture_trains_on() -> None:
    """Why the literature family exists, stated in the form that survived
    measurement.

    The first attempt aimed at density on the published curve itself, because
    ``sawtooth`` reaches it only in its tail: over six thousand free draws the
    closest came to 0.199 RMSE of it while two ordinary draws sit 0.31 apart.
    Perturbing the published six-breakpoint array did concentrate density there
    -- 23% of draws within 0.30 -- but at the cost of a cycle count that never
    moved, and freeing the count put the density back where it started.  There
    is no setting that has both: with the count free, about one draw in nine
    has six breakpoints at all.

    So the goal is the shape class, not the curve.  The exact curve already has
    six hundred rows of its own through ``canonical_zigzag_1s14``; what this
    recipe owes is a family that is genuinely a different family from the free
    sawtooth, and one the published curve belongs to.
    """

    def shapes(recipe: str, n: int) -> np.ndarray:
        return np.array([target - target.mean() for target, _ in _draw(recipe, n=n)])

    # The published curve itself, straight from the hand-written event table:
    # it no longer has a recipe, so there is nothing to draw it from.
    cfg = Config()
    generation_edges, generation_mids = time_grid(
        dataclasses.replace(cfg, time_bins=cfg.demography_generation_bins)
    )
    target_edges, _ = time_grid(cfg)
    breaks, values, meta = _canonical_zigzag_1s14(
        np.random.default_rng(0), cfg, generation_edges, generation_mids
    )
    published = bin_average_log10_ne(
        target_edges, breaks, values, meta["demography_growth_rates"]
    )
    reference = published - published.mean()
    family = shapes("literature_sawtooth", 900)
    free = shapes("sawtooth", 900)

    # The published curve sits inside the family rather than off its edge: its
    # distance to the nearest draw is well under how far two draws sit apart.
    to_reference = np.sqrt(((family - reference) ** 2).mean(axis=1))
    half = len(family) // 2
    internal = np.sqrt(
        ((family[:half] - family[half : 2 * half]) ** 2).mean(axis=1)
    )
    assert to_reference.min() < 0.5 * np.median(internal)

    # And the family is a family: a ridge classifier tells it from a free
    # sawtooth at about 0.82 against a 0.500 chance rate.  Anything near chance
    # would mean the recipe is the free sawtooth under another name.
    def separability(first: np.ndarray, second: np.ndarray) -> float:
        def features(block: np.ndarray) -> np.ndarray:
            steps = np.diff(block, axis=1)
            return np.concatenate([block, np.abs(steps), steps**2], axis=1)

        design = np.concatenate([features(first), features(second)])
        labels = np.array([0] * len(first) + [1] * len(second))
        order = np.random.default_rng(0).permutation(len(design))
        design, labels = design[order], labels[order]
        design = (design - design.mean(0)) / (design.std(0) + 1e-9)
        design = np.hstack([design, np.ones((len(design), 1))])
        targets = np.eye(2)[labels]
        scores = []
        for fold in range(4):
            test = np.arange(len(design)) % 4 == fold
            train = ~test
            weights = np.linalg.solve(
                design[train].T @ design[train] + np.eye(design.shape[1]),
                design[train].T @ targets[train],
            )
            scores.append(
                float((np.argmax(design[test] @ weights, 1) == labels[test]).mean())
            )
        return float(np.mean(scores))

    assert separability(family, free) > 0.70


def test_the_literature_family_is_a_family_and_not_a_copy() -> None:
    """Restoring it must not restore what it replaces.

    The version retired here scaled the published ladder by three scalars --
    time by 0.55 to 1.80, size by the same, contrast by 0.75 to 1.25 -- and
    left the fourfold breakpoint spacing and the tenfold swings otherwise
    exact.  Every breakpoint and every half cycle is perturbed on its own now,
    so two draws differ by more than two free sawtooths do.
    """

    shapes = np.array(
        [target - target.mean() for target, _ in _draw("literature_sawtooth", n=800)]
    )
    half = len(shapes) // 2
    spread = np.sqrt(((shapes[:half] - shapes[half : 2 * half]) ** 2).mean(axis=1))
    assert np.median(spread) > 0.35
    # The breakpoint ladder must not stay at its published 4x ratio.
    ratios = []
    for _, meta in _draw("literature_sawtooth", n=400):
        breaks = np.asarray(meta["demography_breaks"], dtype=np.float64)
        if breaks.size > 2:
            ratios.extend((breaks[1:] / breaks[:-1]).tolist())
    assert np.percentile(ratios, 10) < 2.5
    assert np.percentile(ratios, 90) > 6.0


def test_the_bottleneck_recipe_concentrates_on_bottlenecks() -> None:
    """``size_event`` spans this corner; it does not train on it.

    Measured over four thousand of its draws, 12.8% are a drop of half a decade
    or more that ends and comes back, and 4.0% are a drop of more than a decade
    and a half that does.  At its weight that put the most studied event in the
    field at one row in fifty of the whole mixture.
    """

    def textbook(draws) -> tuple[float, float]:
        proper = deep = 0
        for target, meta in draws:
            depth = meta["event_log_depths"][0] if meta["event_log_depths"] else 0.0
            recovered = abs(float(target[0] - target[-1])) < 0.5
            transient = not meta["event_permanent"]
            proper += int(depth < -0.5 and transient and recovered)
            deep += int(depth < -1.5 and transient and recovered)
        return proper / len(draws), deep / len(draws)

    proper, deep = textbook(list(_draw("bottleneck", n=800)))
    assert proper > 0.5, "the bottleneck recipe has to draw bottlenecks"
    assert deep > 0.10, "and deep ones, not only shallow dips"
    reference, _ = textbook(list(_draw("size_event", n=800)))
    assert proper > 3 * reference


def test_the_bottleneck_is_not_pinned_in_time() -> None:
    """The failure the founder chain had: a bottleneck recipe whose dip always
    lands in the same bins teaches where to look rather than how to look."""

    positions = [int(np.argmin(target)) for target, _ in _draw("bottleneck", n=600)]
    assert np.percentile(positions, 10) < 12
    assert np.percentile(positions, 90) > 45


def test_the_literature_family_varies_its_cycle_count() -> None:
    """The rigidity that remained after the first restoration.

    Perturbing the published array of six breakpoints left the cycle count
    fixed: the label came out with 3, 4 or 5 turning points on every draw
    against a free sawtooth's 3 to 15.  The ladder is built from a drawn number
    of half cycles now, so the count moves.
    """

    counts, turning = [], []
    for target, meta in _draw("literature_sawtooth", n=800):
        counts.append(len(meta["demography_breaks"]))
        steps = np.diff(target)
        steps = steps[steps != 0.0]
        turning.append(
            int((np.diff(np.sign(steps)) != 0).sum()) if steps.size > 1 else 0
        )
    assert np.percentile(counts, 10) < 6 < np.percentile(counts, 90)
    assert np.percentile(turning, 90) - np.percentile(turning, 10) >= 3


def test_an_event_can_land_on_a_moving_background() -> None:
    """The hole this recipe was added for, and it was a hole in the joint.

    Measured across the other eleven recipes, 28.4% of rows carried a sharp
    event -- a single step of more than half a decade -- and 41.0% carried a
    background still moving underneath it, six or more turns once that step is
    removed.  Only 2.7% carried both, against the 11.6% independence would
    give, because every recipe is one or the other by construction:
    ``bottleneck`` and ``size_event`` leave a background with a median of one
    turn, ``spectral_walk`` and the templates leave a largest step of 0.14 to
    0.19.  A population does not hold still waiting for a bottleneck.
    """

    def decompose(target: np.ndarray) -> tuple[float, int]:
        steps = np.diff(target)
        largest = int(np.argmax(np.abs(steps)))
        rest = np.delete(steps, largest)
        moving = rest[rest != 0.0]
        turns = (
            int((np.diff(np.sign(moving)) != 0).sum()) if moving.size > 1 else 0
        )
        return float(abs(steps[largest])), turns

    both = 0
    drawn = list(_draw("event_on_background", n=600))
    for target, _ in drawn:
        step, turns = decompose(target)
        both += int(step > 0.5 and turns >= 6)
    assert both / len(drawn) > 0.35, "the recipe has to deliver the joint"

    # And neither half on its own: it must not collapse into either parent.
    for parent in ("bottleneck", "spectral_walk"):
        reference = [decompose(target) for target, _ in _draw(parent, n=400)]
        share = np.mean([step > 0.5 and turns >= 6 for step, turns in reference])
        assert both / len(drawn) > 5 * max(share, 1e-9)


def test_the_catalogue_holds_every_model_again() -> None:
    """Zigzag_1S14 was excluded from the draw while it had a recipe of its own.

    That recipe is gone, so the model has to come back here or it disappears
    from the generator entirely -- and it is the one trajectory the whole SMC
    literature reports error against.
    """

    from simulator.stdpopsim_templates import CATALOGUE_EXCLUSIONS

    drawable = {t.model_id for t in drawable_stdpopsim_templates(Config())}
    assert "Zigzag_1S14" in drawable
    assert not CATALOGUE_EXCLUSIONS


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

# Fields a sampler records that deliberately do not reach the shard row: either
# the writer derives them itself, or they are variable-length and are persisted
# as fixed-width summaries instead, or they are provenance strings recoverable
# from an id that is persisted.
_NOT_PERSISTED = frozenset(
    {
        "source_id",
        "control_type",
        "time_span_min",
        "time_span_max",
        "target_time_span_min",
        "target_scale",
        "target_aggregation",
        "rec_map_meta",
        "mut_map_meta",
        "noise",
        "sample_id",
        "scenario_key",
        "generator_recipe_key",
        "truth_maps_written",
        # Variable length; ``event_log_depth_min`` and ``_max`` carry it.
        "event_log_depths",
        # Recoverable from the model id and the template store.
        "template_path",
        "template_time_unit",
        "template_generation_time_years",
        "template_curve_point_count",
        "template_time_min",
        "template_time_max",
        "template_bin_noise_sd",
        "dense_demography",
        "dense_type",
        "stdpopsim_population",
        "stdpopsim_generation_time_years",
    }
)


def test_every_sampler_parameter_reaches_the_shard_row() -> None:
    """The whitelist in ``_metadata_row`` is the only path from a sampler to
    the disk, and it does not fail loudly: a field it does not name is simply
    dropped.

    This is what replaced 27 recipe names -- the claim that any grouping the
    names supported can still be made at analysis time, on realised parameters
    rather than on which sampler was called.  That claim is only true if the
    parameters are actually written.  When this test was added they were not:
    ``held_out_evaluation`` never reached disk, which is the flag a split uses
    to keep the stdpopsim catalogue out of the training gradient, and with it
    went ``stdpopsim_model_id``, every ``event_``, ``ooa_``, ``oscillation_``
    and ``template_`` parameter, and ``knot_count``.
    """

    import dataclasses

    from simulator.config import SOURCE_PROB_FIELDS, apply_preset, validate_config
    from simulator.sample import simulate_one
    from simulator.shards import _metadata_row

    cfg = apply_preset(Config(preset="full_v01"))
    cfg = dataclasses.replace(
        cfg, seq_len=200_000, include_truth=True, truth_fraction=1.0
    )
    for field in SOURCE_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_custom_hetero = 1.0
    validate_config(cfg)

    dropped: dict[str, list[str]] = {}
    for name, field in DEMO_PROB_FIELDS:
        variant = dataclasses.replace(cfg)
        for _, other in DEMO_PROB_FIELDS:
            setattr(variant, other, 0.0)
        setattr(variant, field, 1.0)
        meta = simulate_one(
            (0, dataclasses.asdict(variant), {"rec_maps": {}, "mut_maps": {}}, 321)
        )["metadata"]
        row = _metadata_row(meta)
        missing = [
            key
            for key in meta
            if key not in row
            and key not in _NOT_PERSISTED
            and not key.startswith("demography_")
        ]
        if missing:
            dropped[name] = sorted(missing)
    assert not dropped, dropped


def test_the_held_out_flag_survives_to_disk() -> None:
    """Stated on its own because it is the one that breaks silently.

    Everything else being dropped costs an analysis; this one costs the
    evaluation design -- the catalogue rows would be trained on, and error
    measured on them would stop being evidence about generalisation.
    """

    import dataclasses

    from simulator.config import SOURCE_PROB_FIELDS, apply_preset, validate_config
    from simulator.sample import simulate_one
    from simulator.shards import _metadata_row

    def row_for(recipe: str, seed: int) -> dict:
        cfg = apply_preset(Config(preset="full_v01"))
        cfg = dataclasses.replace(cfg, seq_len=200_000)
        for field in SOURCE_PROB_FIELDS:
            setattr(cfg, field, 0.0)
        cfg.p_custom_hetero = 1.0
        for _, field in DEMO_PROB_FIELDS:
            setattr(cfg, field, 0.0)
        setattr(cfg, dict(DEMO_PROB_FIELDS)[recipe], 1.0)
        validate_config(cfg)
        sample = simulate_one(
            (0, dataclasses.asdict(cfg), {"rec_maps": {}, "mut_maps": {}}, seed)
        )
        return _metadata_row(sample["metadata"])

    for seed in range(4):
        row = row_for("stdpopsim_single_pop", 900 + seed)
        assert row["held_out_evaluation"] is True
        assert row["stdpopsim_model_id"], "and the model has to be identifiable"
    assert row_for("spectral_walk", 900)["held_out_evaluation"] is False
