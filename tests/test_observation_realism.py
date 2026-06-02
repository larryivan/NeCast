"""Tests for the two places the simulated data measurably left the real data.

Both are train/test mismatches rather than bugs: training is self-consistent
either way, the loss falls either way, and arms compare either way.  They only
bite when the model is pointed at a real dataset, and by then they are hard to
attribute -- a recent effective size that reads high looks like a model
failure, and a long-range feature that stops working looks like an
architecture failure.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from necast.simulator.config import (
    SOURCE_PROB_FIELDS,
    Config,
    apply_preset,
    validate_config,
)
from necast.simulator.noise import add_false_variants, noise_params_for_sample


def _config(**overrides) -> Config:
    cfg = apply_preset(Config(preset="full_v01"))
    settings = {"seq_len": 1_000_000, "include_truth": True, "truth_fraction": 1.0}
    settings.update(overrides)
    cfg = dataclasses.replace(cfg, **settings)
    validate_config(cfg)
    return cfg


# --------------------------------------------------------------------------
# Invented sites
# --------------------------------------------------------------------------


def test_a_false_variant_is_carried_by_exactly_one_haplotype() -> None:
    """Which is what makes it a singleton, and what the channel is for.

    Two independent errors landing on one position are rare enough to ignore,
    so a manufactured site is carried once.
    """

    rng = np.random.default_rng(0)
    genotypes = np.zeros((10, 64), dtype=np.uint8)
    positions = np.linspace(0, 999_999, 10)
    grown, grown_positions, stats = add_false_variants(
        genotypes, positions, rng, 1_000_000, 1.0e-3
    )
    assert stats["false_variant_count"] > 0
    assert grown.shape[0] == genotypes.shape[0] + stats["false_variant_count"]
    added = grown.sum(axis=1)
    # The ten real rows are empty here, so every non-empty row is invented.
    assert set(np.unique(added)).issubset({0, 1})
    assert int((added == 1).sum()) == stats["false_variant_count"]
    # Positions stay sorted, because everything downstream indexes by position.
    assert np.all(np.diff(grown_positions) >= 0)


def test_the_channel_is_off_where_the_source_is_declared_clean() -> None:
    cfg = _config()
    clean = noise_params_for_sample(
        np.random.default_rng(0), cfg, "clean_constant_map"
    )
    assert clean["false_variant_rate_per_bp"] == 0.0
    noisy = noise_params_for_sample(
        np.random.default_rng(0), cfg, "custom_heterogeneous"
    )
    assert noisy["false_variant_rate_per_bp"] > 0.0


def test_the_false_singleton_share_spans_what_callsets_report() -> None:
    """The rate is a property of the calling pipeline, not of the population.

    A hard-filtered callset and a raw one differ by orders of magnitude, so the
    prior has to cover the range rather than pick a point -- and the row has to
    record which was drawn, so the effect can be conditioned on or ablated.
    """

    cfg = _config()
    shares = []
    for seed in range(400):
        drawn = noise_params_for_sample(
            np.random.default_rng(seed), cfg, "custom_heterogeneous"
        )
        rate = drawn["false_variant_rate_per_bp"]
        # About sixteen hundred singletons a megabase at a typical effective
        # size; the constant only sets the scale this test reads in.
        shares.append(rate * cfg.seq_len / 1_600.0)
    assert np.percentile(shares, 10) < 0.05, "no near-clean callset drawn"
    assert np.percentile(shares, 90) > 0.15, "no raw callset drawn"


def test_invented_sites_are_recorded_on_the_row() -> None:
    """Recorded, not hidden: an ablation needs to be able to find them."""

    import dataclasses as dc

    from necast.simulator.sample import simulate_one
    from necast.simulator.shards import _metadata_row

    cfg = _config()
    sample = simulate_one(
        (0, dc.asdict(cfg), {"rec_maps": {}, "mut_maps": {}}, 4_242)
    )
    row = _metadata_row(sample["metadata"])
    assert "false_variant_rate_per_bp" in row
    assert "false_variant_count" in row


# --------------------------------------------------------------------------
# Map error that survives averaging
# --------------------------------------------------------------------------


def _correlation_by_scale(cfg: Config, seeds: int = 40) -> dict[int, float]:
    from necast.simulator.sample import simulate_one

    def rebin(pos, rate, edges):
        cumulative = np.concatenate(([0.0], np.cumsum(rate * np.diff(pos))))
        return np.diff(np.interp(edges, pos, cumulative)) / np.diff(edges)

    found: dict[int, list[float]] = {1_000: [], 10_000: [], 100_000: []}
    for seed in range(seeds):
        sample = simulate_one(
            (seed, dataclasses.asdict(cfg), {"rec_maps": {}, "mut_maps": {}}, 90_000 + seed)
        )
        true_pos = sample["true_rec_pos"]
        obs_pos = sample["obs_rec_pos"]
        if true_pos.size < 3 or obs_pos.size < 3:
            continue
        for scale in found:
            edges = np.arange(0, cfg.seq_len + scale, scale, dtype=float)
            truth = rebin(true_pos, sample["true_rec_rate"], edges)
            observed = rebin(obs_pos, sample["obs_rec_rate"], edges)
            usable = (truth > 0) & (observed > 0)
            if usable.sum() > 10:
                found[scale].append(
                    float(np.corrcoef(np.log(truth[usable]), np.log(observed[usable]))[0, 1])
                )
    return {scale: float(np.median(values)) for scale, values in found.items() if values}


def test_map_error_survives_being_averaged() -> None:
    """The property the per-cell draw could not have.

    Independent error cancels: with one lognormal per cell the map handed to
    the model correlated 0.720 with the truth at a kilobase and 0.984 at a
    hundred, against 0.687 and 0.890 for two real published maps of the same
    region.  Zoomed out our map was effectively perfect, so a model could learn
    to trust the coarse map completely -- true in training and false on real
    data, where the disagreement is regional and averaging does not remove it.

    The tolerances are loose because the target is a property of two particular
    real maps; what is pinned is that the error still exists at a hundred
    kilobases, which the previous draw could not deliver at any amplitude.
    """

    cfg = _config(seq_len=2_000_000)
    for field in SOURCE_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_custom_hetero = 1.0
    validate_config(cfg)

    measured = _correlation_by_scale(cfg)
    assert measured[1_000] == pytest.approx(0.687, abs=0.12)
    assert measured[100_000] == pytest.approx(0.890, abs=0.10)
    # And the decisive one: averaging a hundredfold must not restore the truth.
    assert measured[100_000] < 0.95, "the coarse map is too good to be real"
    assert measured[100_000] > measured[1_000], "error must still shrink with scale"


def test_the_mutation_map_survives_its_own_observation_noise() -> None:
    """The defect this test exists for was invisible to every other one.

    Observation error used to be an absolute log-scale amplitude, and the two
    maps are not on the same scale: across a row a recombination map's log rate
    has a standard deviation near 0.5 and a mutation map's near 0.04.  One
    number cannot serve both.  At an absolute 0.25 the observed recombination
    map correlated 0.851 with the truth -- fine -- while the observed mutation
    map correlated 0.163 on synthetic maps and 0.108 on real slices, with the
    level off by 0.07 to 0.09 in log10.  That is not a noisy map, it is no map:
    the model was handed pure noise and told it was the mutation rate.

    The error is a fraction of whatever spread the true map has now, so it
    lands in the same relative place on both.  What is pinned here is that
    neither map is drowned and neither is systematically rescaled -- the
    observed integral is what a segregating count gets divided by, so a level
    error is a direct scale error on the effective size read off the row.
    """

    cfg = _config(seq_len=2_000_000)
    for field in SOURCE_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_custom_hetero = 1.0
    validate_config(cfg)

    from necast.simulator.sample import simulate_one

    def rebin(pos, rate, edges):
        cumulative = np.concatenate(([0.0], np.cumsum(rate * np.diff(pos))))
        return np.diff(np.interp(edges, pos, cumulative)) / np.diff(edges)

    edges = np.arange(0, cfg.seq_len + 5_000, 5_000, dtype=float)
    found: dict[str, list[tuple[float, float]]] = {"rec": [], "mut": []}
    for seed in range(20):
        sample = simulate_one(
            (seed, dataclasses.asdict(cfg), {"rec_maps": {}, "mut_maps": {}}, 70_000 + seed)
        )
        for key in found:
            truth = rebin(
                sample[f"true_{key}_pos"], sample[f"true_{key}_rate"], edges
            )
            observed = rebin(
                sample[f"obs_{key}_pos"], sample[f"obs_{key}_rate"], edges
            )
            usable = (truth > 0) & (observed > 0)
            if usable.sum() < 20:
                continue
            log_truth = np.log10(truth[usable])
            log_observed = np.log10(observed[usable])
            if log_truth.std() < 1e-9 or log_observed.std() < 1e-9:
                continue
            found[key].append(
                (
                    float(np.corrcoef(log_truth, log_observed)[0, 1]),
                    float(log_observed.mean() - log_truth.mean()),
                )
            )

    for key in ("rec", "mut"):
        assert found[key], key
        correlation = float(np.median([value for value, _ in found[key]]))
        offset = float(np.median([value for _, value in found[key]]))
        assert correlation > 0.55, f"{key} map is drowned in its own noise: {correlation:.3f}"
        assert abs(offset) < 0.05, f"{key} map level is biased by {offset:+.3f} decades"
