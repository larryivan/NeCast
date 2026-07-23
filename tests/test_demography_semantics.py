from __future__ import annotations

import dataclasses
import json
import sys
from types import SimpleNamespace

import msprime
import numpy as np
import pytest

from simulator.config import (
    DEMOGRAPHY_PROB_FIELDS,
    DEMOGRAPHY_TYPE_IDS,
    DEMO_IDS,
    GENERATOR_RECIPE_IDS,
    GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE,
    ID_TO_DEMOGRAPHY_TYPE,
    ID_TO_DEMO,
    ID_TO_GENERATOR_RECIPE,
    SOURCE_PROB_FIELDS,
    ULTRA_STRESS_TYPE_TO_DEMOGRAPHY_TYPE,
    Config,
)
from simulator.demography import (
    DEMO_PROB_FIELDS,
    DEMO_SAMPLERS,
    _canonical_smcpp_sawtooth,
    _canonical_zigzag_1s14,
    _clean_demography,
    _classic_sawtooth,
    _native_exponential_transition,
    _zigzag_legacy,
    _zigzag_strong,
    build_demography,
    choose_generator_recipe,
    resolve_demography_type,
    sample_custom_demography,
    summarize_demography,
)
from simulator.dataset_state import SAMPLE_GENERATOR_SCHEMA_VERSION
from simulator.sample import simulate_one
from simulator.stdpopsim_support import (
    StdpopsimSimulationError,
    _extract_single_population_trajectory,
    try_stdpopsim_sample,
)
from simulator.utils import bin_average_log10_ne, time_grid


def _only_demography(name: str, *, time_bins: int = 48) -> Config:
    weights = {field: 0.0 for field in DEMOGRAPHY_PROB_FIELDS}
    field_by_name = dict(DEMO_PROB_FIELDS)
    weights[field_by_name[name]] = 1.0
    return dataclasses.replace(Config(), time_bins=time_bins, **weights)


EXPECTED_STATIC_RECIPE_TYPES = {
    "constant": "constant",
    "near_constant": "random_walk",
    "smooth_random_walk": "random_walk",
    "dense_random_walk": "random_walk",
    "dense_multiscale_random_walk": "random_walk",
    "dense_recent_wiggle": "random_walk",
    "smooth_random_walk_stress": "random_walk",
    "dense_random_walk_stress": "random_walk",
    "single_bottleneck": "bottleneck",
    "recent_bottleneck": "bottleneck",
    "recent_bottleneck_extreme": "bottleneck",
    "recent_founder_recovery": "bottleneck",
    "founder_recovery_extreme": "bottleneck",
    "continuous_exponential_growth": "exponential_growth",
    "rapid_recent_growth_extreme": "exponential_growth",
    "continuous_exponential_decline": "exponential_decline",
    "expansion": "expansion",
    "contraction": "contraction",
    "three_epoch": "three_epoch",
    "ancient_recent_compound": "three_epoch",
    "ancient_recent_conflict": "three_epoch",
    "serial_founder": "serial_founder",
    "serial_founder_extreme": "serial_founder",
    "classic_sawtooth": "zigzag",
    "canonical_zigzag_1s14": "zigzag",
    "canonical_smcpp_sawtooth": "zigzag",
    "oscillating_mild": "zigzag",
    "zigzag_strong": "zigzag",
    "zigzag": "zigzag",
    "empirical_human_ne_template": "empirical_human_ne_template",
}


def _log_time_average_oracle(
    edges: np.ndarray,
    breaks: list[float],
    values: list[float],
    growth_rates: list[float],
) -> np.ndarray:
    """Independent closed-form oracle for mean(log10 Ne) in log-time bins."""
    starts = [0.0, *breaks]
    result: list[float] = []
    for left, right in zip(edges[:-1], edges[1:]):
        cuts = [float(left), *(t for t in breaks if left < t < right), float(right)]
        integral = 0.0
        for a, b in zip(cuts[:-1], cuts[1:]):
            epoch = int(np.searchsorted(breaks, np.sqrt(a * b), side="right"))
            log_width = np.log(b / a)
            integral += np.log10(values[epoch]) * log_width
            integral -= growth_rates[epoch] / np.log(10.0) * (
                (b - a) - starts[epoch] * log_width
            )
        result.append(float(integral / np.log(right / left)))
    return np.asarray(result)


def _shape_path_oracle(
    edges: np.ndarray,
    breaks: list[float],
    values: list[float],
    growth_rates: list[float],
) -> np.ndarray:
    starts = [0.0, *breaks]

    def evaluate(epoch: int, time: float) -> float:
        return float(
            values[epoch]
            * np.exp(-growth_rates[epoch] * (time - starts[epoch]))
        )

    path = [float(values[0])]
    for epoch, boundary in enumerate(breaks):
        if 0.0 < boundary < float(edges[-1]):
            path.extend([evaluate(epoch, boundary), float(values[epoch + 1])])
    right_epoch = int(np.searchsorted(breaks, edges[-1], side="left"))
    path.append(evaluate(right_epoch, float(edges[-1])))
    return np.asarray(path, dtype=np.float64)


@pytest.mark.parametrize(
    ("present_ne", "older_ne", "expected_sign"),
    [
        (100_000.0, 10_000.0, 1),
        (10_000.0, 100_000.0, -1),
    ],
)
def test_native_exponential_matches_msprime_trajectory(
    present_ne: float,
    older_ne: float,
    expected_sign: int,
) -> None:
    end = 1_000.0
    breaks, values, rates = _native_exponential_transition(present_ne, older_ne, end)
    demography = build_demography(breaks, values, rates)
    demography.validate()

    times = np.array([0.0, 100.0, 500.0, end - 1e-6, end, 2 * end])
    observed = demography.debug().population_size_trajectory(times)[:, 0]
    expected = np.where(
        times < end,
        present_ne * np.exp(-rates[0] * times),
        older_ne,
    )

    assert np.sign(rates[0]) == expected_sign
    np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=1e-8)


def test_continuous_target_uses_exact_log_time_integral() -> None:
    breaks, values, rates = _native_exponential_transition(100_000.0, 10_000.0, 500.0)
    edges = np.array([50.0, 200.0, 1_000.0, 5_000.0])

    observed = bin_average_log10_ne(edges, breaks, values, rates)
    expected = _log_time_average_oracle(edges, breaks, values, rates)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-6)


def test_zero_growth_rates_preserve_piecewise_constant_targets() -> None:
    edges = np.geomspace(50.0, 500_000.0, 49)
    breaks = [200.0, 3_000.0, 70_000.0]
    values = [20_000.0, 2_000.0, 80_000.0, 10_000.0]

    previous_api = bin_average_log10_ne(edges, breaks, values)
    explicit_zero_rates = bin_average_log10_ne(edges, breaks, values, [0.0] * len(values))

    np.testing.assert_array_equal(previous_api, explicit_zero_rates)


@pytest.mark.parametrize(
    "scenario",
    [
        "continuous_exponential_growth",
        "rapid_recent_growth_extreme",
        "continuous_exponential_decline",
        "classic_sawtooth",
        "canonical_zigzag_1s14",
        "canonical_smcpp_sawtooth",
        "zigzag_strong",
    ],
)
def test_native_model_target_matches_msprime_debugger(scenario: str) -> None:
    cfg = _only_demography(scenario)
    demography, target, meta = sample_custom_demography(np.random.default_rng(20260723), cfg)
    edges, _ = time_grid(cfg)

    numeric: list[float] = []
    debugger = demography.debug()
    for left, right in zip(edges[:-1], edges[1:]):
        log_times = np.linspace(np.log(left), np.log(right), 513)
        times = np.exp(log_times)
        ne = debugger.population_size_trajectory(times)[:, 0]
        log_ne = np.log10(ne)
        integral = np.sum((log_ne[:-1] + log_ne[1:]) * np.diff(log_times) / 2.0)
        numeric.append(float(integral / (log_times[-1] - log_times[0])))

    assert meta["native_continuous_growth"] is True
    assert len(meta["demography_growth_rates"]) == len(meta["demography_values"])
    np.testing.assert_allclose(target, numeric, rtol=0.0, atol=2e-5)


@pytest.mark.parametrize(
    ("scenario", "factor_key", "factor_direction"),
    [
        ("continuous_exponential_growth", "growth_factor", "present_over_old"),
        ("rapid_recent_growth_extreme", "growth_factor", "present_over_old"),
        ("continuous_exponential_decline", "decline_factor", "old_over_present"),
    ],
)
def test_exponential_metadata_records_realized_model(
    scenario: str,
    factor_key: str,
    factor_direction: str,
) -> None:
    cfg = _only_demography(scenario)
    _, _, meta = sample_custom_demography(np.random.default_rng(19), cfg)
    present_ne, older_ne = meta["demography_values"]
    end = meta["demography_breaks"][0]
    rate = meta["demography_growth_rates"][0]

    expected_factor = (
        present_ne / older_ne
        if factor_direction == "present_over_old"
        else older_ne / present_ne
    )
    assert meta[factor_key] == pytest.approx(expected_factor)
    assert meta["growth_rate_per_generation"] == pytest.approx(
        np.log(present_ne / older_ne) / end
    )
    assert rate == pytest.approx(meta["growth_rate_per_generation"])


def test_canonical_zigzag_1s14_matches_published_event_table() -> None:
    cfg = Config()
    edges, mids = time_grid(cfg)
    breaks, values, meta = _canonical_zigzag_1s14(
        np.random.default_rng(1),
        cfg,
        edges,
        mids,
    )

    expected_breaks = [
        33.333334976,
        133.3334544,
        533.33324512,
        2_133.3352704,
        8_533.329632,
        34_133.318528,
    ]
    expected_values = [
        71_560.0,
        71_560.0,
        7_156.029247798124,
        71_560.21678822454,
        7_155.999712600628,
        71_559.65925132985,
        7_156.0,
    ]
    expected_rates = [
        0.0,
        0.02302578256,
        -0.0057564631,
        0.00143911578,
        -0.0003597785,
        8.9944801565e-5,
        0.0,
    ]

    assert breaks == expected_breaks
    np.testing.assert_allclose(values, expected_values, rtol=2e-14, atol=1e-9)
    assert meta["demography_growth_rates"] == expected_rates
    assert meta["literature_model_id"] == "Zigzag_1S14"

    demography = build_demography(breaks, values, expected_rates)
    ancient = demography.debug().population_size_trajectory([breaks[-1], 100_000.0])[:, 0]
    np.testing.assert_array_equal(ancient, np.array([7_156.0, 7_156.0]))


def test_smcpp_sawtooth_is_kept_distinct_from_zigzag_1s14() -> None:
    cfg = Config()
    edges, mids = time_grid(cfg)
    s_breaks, s_values, s_meta = _canonical_smcpp_sawtooth(
        np.random.default_rng(1),
        cfg,
        edges,
        mids,
    )
    z_breaks, z_values, _ = _canonical_zigzag_1s14(
        np.random.default_rng(1),
        cfg,
        edges,
        mids,
    )

    assert s_meta["literature_model_id"] == "Terhorst2017_sawtooth"
    assert s_meta["literature_demography_anchor"] is True
    assert s_breaks[2] == 533.50498912
    assert z_breaks[2] == 533.33324512
    assert s_values[-1] == pytest.approx(7_164.809184944985)
    assert z_values[-1] == 7_156.0
    expected_rates = [
        0.0,
        *(
            rate / (4 * 14_312)
            for rate in [1_318.18, -329.546, 82.3865, -20.5966, 5.14916]
        ),
        0.0,
    ]
    assert s_meta["demography_growth_rates"] == expected_rates


def test_continuous_metadata_round_trip_reconstructs_ground_truth() -> None:
    cfg = _only_demography("classic_sawtooth")
    demography, _, meta = sample_custom_demography(np.random.default_rng(81), cfg)
    restored = json.loads(json.dumps(meta))
    reconstructed = build_demography(
        restored["demography_breaks"],
        restored["demography_values"],
        restored["demography_growth_rates"],
    )
    times = np.geomspace(1.0, cfg.max_time * 2, 1_000)

    expected = demography.debug().population_size_trajectory(times)
    observed = reconstructed.debug().population_size_trajectory(times)
    np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-9)


def test_stdpopsim_proxy_extraction_preserves_first_population_growth() -> None:
    source = msprime.Demography()
    source.add_population(name="target", initial_size=100_000.0, growth_rate=0.001)
    source.add_population(name="other", initial_size=20_000.0)
    source.add_population_parameters_change(time=100.0, growth_rate=-0.002, population="target")
    source.add_population_parameters_change(time=200.0, initial_size=500.0, population="other")
    source.add_population_parameters_change(
        time=300.0,
        initial_size=40_000.0,
        growth_rate=0.0,
        population="target",
    )
    source.sort_events()

    breaks, values, rates = _extract_single_population_trajectory(source)
    reconstructed = build_demography(breaks, values, rates)
    times = np.array([0.0, 50.0, 100.0, 150.0, 299.0, 300.0, 1_000.0])

    expected = source.debug().population_size_trajectory(times)[:, 0]
    observed = reconstructed.debug().population_size_trajectory(times)[:, 0]
    assert breaks == [100.0, 300.0]
    assert rates == [0.001, -0.002, 0.0]
    np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=1e-8)


def test_stdpopsim_anchor_fails_instead_of_inventing_constant_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTreeSequence:
        sequence_length = 1_000

        def trim(self):
            return self

    class FakeEngine:
        def simulate(self, model, contig, samples, seed):
            return FakeTreeSequence()

    model = SimpleNamespace(
        id="BrokenModel",
        populations=[SimpleNamespace(id="pop")],
        model=None,
        get_samples=lambda *args: {"pop": int(args[0])},
    )
    species = SimpleNamespace(
        demographic_models=[model],
        get_demographic_model=lambda model_id: model,
        get_contig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    fake_stdpopsim = SimpleNamespace(
        get_species=lambda species_id: species,
        get_engine=lambda engine_id: FakeEngine(),
    )
    monkeypatch.setitem(sys.modules, "stdpopsim", fake_stdpopsim)
    cfg = dataclasses.replace(
        Config(),
        enable_stdpopsim=True,
        stdpopsim_models="BrokenModel",
        n_haplotypes=4,
        seq_len=1_000,
    )

    with pytest.raises(
        StdpopsimSimulationError,
        match="exposes no msprime demography",
    ):
        try_stdpopsim_sample(np.random.default_rng(123), cfg)


def test_native_literature_model_survives_full_sample_generation() -> None:
    cfg = _only_demography("canonical_zigzag_1s14")
    source_weights = {field: 0.0 for field in SOURCE_PROB_FIELDS}
    source_weights["p_clean_constant"] = 1.0
    cfg = dataclasses.replace(
        cfg,
        seq_len=5_000,
        n_haplotypes=4,
        **source_weights,
    )

    sample = simulate_one((0, dataclasses.asdict(cfg), {"rec_maps": {}, "mut_maps": {}}, 721))
    meta = sample["metadata"]

    assert sample["demo_id"] == GENERATOR_RECIPE_IDS["canonical_zigzag_1s14"]
    assert sample["demography_id"] == GENERATOR_RECIPE_IDS["canonical_zigzag_1s14"]
    assert sample["demography_type_id"] == DEMOGRAPHY_TYPE_IDS["zigzag"]
    assert sample["generator_recipe_id"] == GENERATOR_RECIPE_IDS["canonical_zigzag_1s14"]
    assert sample["target_log10_ne"].shape == (cfg.time_bins,)
    assert np.isfinite(sample["target_log10_ne"]).all()
    assert meta["demography_type"] == "zigzag"
    assert meta["generator_recipe"] == "canonical_zigzag_1s14"
    assert meta["scenario"] == "canonical_zigzag_1s14"
    assert meta["literature_model_id"] == "Zigzag_1S14"
    assert meta["demography_growth_rates"][-1] == 0.0
    assert meta["n_variants"] == sample["positions"].size


@pytest.mark.parametrize("seed", range(20))
def test_random_zigzag_is_continuous_alternating_and_terminally_constant(seed: int) -> None:
    cfg = Config()
    edges, mids = time_grid(cfg)
    breaks, values, meta = _zigzag_strong(np.random.default_rng(seed), cfg, edges, mids)
    rates = meta["demography_growth_rates"]

    starts = [0.0, *breaks]
    for epoch, boundary in enumerate(breaks):
        endpoint = values[epoch] * np.exp(-rates[epoch] * (boundary - starts[epoch]))
        assert endpoint == pytest.approx(values[epoch + 1], rel=2e-12)

    signs = np.sign(rates[1:-1])
    assert np.all(signs != 0)
    assert np.all(signs[1:] == -signs[:-1])
    assert rates[-1] == 0.0

    demography = build_demography(breaks, values, rates)
    terminal = demography.debug().population_size_trajectory(
        [breaks[-1], max(cfg.max_time * 2, breaks[-1] * 2)]
    )[:, 0]
    np.testing.assert_allclose(terminal[0], terminal[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("sampler", [_classic_sawtooth, _zigzag_strong])
def test_continuous_sawtooth_ground_truth_does_not_depend_on_label_bins(sampler) -> None:
    cfg32 = Config(time_bins=32)
    cfg64 = Config(time_bins=64)
    edges32, mids32 = time_grid(cfg32)
    edges64, mids64 = time_grid(cfg64)

    result32 = sampler(np.random.default_rng(991), cfg32, edges32, mids32)
    result64 = sampler(np.random.default_rng(991), cfg64, edges64, mids64)

    assert result32 == result64


def test_all_registered_demographies_have_probability_fields() -> None:
    assert {name for name, _ in DEMO_PROB_FIELDS} == set(DEMO_SAMPLERS)
    assert {field for _, field in DEMO_PROB_FIELDS} == set(DEMOGRAPHY_PROB_FIELDS)
    assert set(GENERATOR_RECIPE_IDS) == set(DEMO_SAMPLERS) | {"stdpopsim"}
    assert set(GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE) == set(GENERATOR_RECIPE_IDS)
    assert DEMO_IDS == GENERATOR_RECIPE_IDS
    assert len(set(DEMO_IDS.values())) == len(DEMO_IDS)
    assert len(set(DEMOGRAPHY_TYPE_IDS.values())) == len(DEMOGRAPHY_TYPE_IDS)
    assert len(set(GENERATOR_RECIPE_IDS.values())) == len(GENERATOR_RECIPE_IDS)
    assert {value: key for key, value in DEMO_IDS.items()} == ID_TO_DEMO
    assert {
        value: key for key, value in DEMOGRAPHY_TYPE_IDS.items()
    } == ID_TO_DEMOGRAPHY_TYPE
    assert {
        value: key for key, value in GENERATOR_RECIPE_IDS.items()
    } == ID_TO_GENERATOR_RECIPE


def test_public_demography_types_are_compact_and_explicit() -> None:
    assert set(DEMOGRAPHY_TYPE_IDS) == {
        "constant",
        "random_walk",
        "bottleneck",
        "exponential_growth",
        "exponential_decline",
        "expansion",
        "contraction",
        "three_epoch",
        "serial_founder",
        "zigzag",
        "empirical_human_ne_template",
        "stdpopsim",
    }
    assert {
        recipe: demography_type
        for recipe, demography_type in GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE.items()
        if demography_type is not None and recipe != "stdpopsim"
    } == EXPECTED_STATIC_RECIPE_TYPES
    assert {
        recipe
        for recipe, demography_type in GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE.items()
        if demography_type is None
    } == {"ancient_event", "ultra_stress"}


@pytest.mark.parametrize(
    ("stress_type", "expected"),
    [
        ("ultra_random_walk", "random_walk"),
        ("ultra_recent_bottleneck", "bottleneck"),
        ("ultra_founder_recovery", "bottleneck"),
        ("ultra_rapid_recent_growth", "exponential_growth"),
        ("ultra_serial_founder", "serial_founder"),
        ("ultra_ancient_recent_conflict", "three_epoch"),
    ],
)
def test_ultra_stress_is_classified_by_sampled_mechanism(
    stress_type: str,
    expected: str,
) -> None:
    assert ULTRA_STRESS_TYPE_TO_DEMOGRAPHY_TYPE[stress_type] == expected
    assert resolve_demography_type(
        "ultra_stress",
        [10_000.0, 20_000.0],
        {"stress_type": stress_type},
    ) == expected


def test_ancient_event_is_classified_by_forward_time_direction() -> None:
    assert resolve_demography_type("ancient_event", [20_000.0, 10_000.0], {}) == "expansion"
    assert resolve_demography_type("ancient_event", [10_000.0, 20_000.0], {}) == "contraction"
    assert resolve_demography_type("ancient_event", [10_000.0, 10_000.0], {}) == "constant"


@pytest.mark.parametrize("generator_recipe", sorted(DEMO_SAMPLERS))
def test_taxonomy_preserves_every_recipe_trajectory_and_adds_uniform_metrics(
    generator_recipe: str,
) -> None:
    cfg = _only_demography(generator_recipe)
    expected_rng = np.random.default_rng(830_001)
    assert choose_generator_recipe(expected_rng, cfg) == generator_recipe
    edges, mids = time_grid(cfg)
    raw_breaks, raw_values, raw_meta = DEMO_SAMPLERS[generator_recipe](
        expected_rng,
        cfg,
        edges,
        mids,
    )
    expected_breaks, expected_values, expected_rates = _clean_demography(
        raw_breaks,
        raw_values,
        raw_meta.get("demography_growth_rates"),
    )
    expected_target = bin_average_log10_ne(
        edges,
        expected_breaks,
        expected_values,
        expected_rates,
    )

    _, observed_target, meta = sample_custom_demography(
        np.random.default_rng(830_001),
        cfg,
    )

    assert meta["generator_recipe"] == generator_recipe
    assert meta["generator_recipe_id"] == GENERATOR_RECIPE_IDS[generator_recipe]
    assert meta["scenario"] == generator_recipe
    expected_type = EXPECTED_STATIC_RECIPE_TYPES.get(
        generator_recipe,
        resolve_demography_type(generator_recipe, expected_values, raw_meta),
    )
    assert meta["demography_type"] == expected_type
    assert meta["demography_type_id"] == DEMOGRAPHY_TYPE_IDS[expected_type]
    assert meta["demography_id"] == GENERATOR_RECIPE_IDS[generator_recipe]
    if generator_recipe == "near_constant":
        assert meta["history_family"] == "stable"
    np.testing.assert_array_equal(meta["demography_breaks"], expected_breaks)
    np.testing.assert_array_equal(meta["demography_values"], expected_values)
    np.testing.assert_array_equal(meta["demography_growth_rates"], expected_rates)
    np.testing.assert_array_equal(observed_target, expected_target)

    shape_values = _shape_path_oracle(
        edges,
        expected_breaks,
        expected_values,
        expected_rates,
    )
    log_values = np.log10(shape_values)
    differences = np.diff(log_values)
    nonzero_signs = np.sign(differences[np.abs(differences) > 1e-12])
    expected_turns = (
        int(np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1]))
        if nonzero_signs.size > 1
        else 0
    )
    assert meta["event_severity"] == pytest.approx(
        float(shape_values.min() / shape_values.max())
    )
    assert meta["log10_ne_range"] == pytest.approx(float(np.ptp(log_values)))
    assert meta["total_variation_log10"] == pytest.approx(float(np.abs(differences).sum()))
    assert meta["turn_count"] == expected_turns
    if "event_severity" in raw_meta:
        assert meta["recipe_event_severity"] == pytest.approx(raw_meta["event_severity"])
    else:
        assert meta["recipe_event_severity"] is None


def test_shape_metrics_include_growth_limits_and_instantaneous_resets() -> None:
    edges = np.array([1.0, 10.0, 100.0, 200.0])
    breaks = [100.0]
    values = [100.0, 100.0]
    growth_rates = [-np.log(10.0) / 100.0, 0.0]

    summary = summarize_demography(
        edges,
        breaks,
        values,
        growth_rates,
        {},
        generator_recipe="stdpopsim",
        demography_type="stdpopsim",
    )

    assert summary["min_Ne"] == pytest.approx(100.0)
    assert summary["max_Ne"] == pytest.approx(1_000.0)
    assert summary["event_severity"] == pytest.approx(0.1)
    assert summary["total_variation_log10"] == pytest.approx(2.0)
    assert summary["turn_count"] == 1
    assert summary["direction"] == "mixed"


def test_legacy_zigzag_keeps_seedwise_selection_and_matches_strong_recipe() -> None:
    cfg = Config()
    assert cfg.p_zigzag + cfg.p_zigzag_strong == pytest.approx(0.015)
    assert cfg.p_zigzag == pytest.approx(0.005)
    assert cfg.p_zigzag_strong == pytest.approx(0.01)
    assert sum(float(getattr(cfg, field)) for field in DEMOGRAPHY_PROB_FIELDS) == pytest.approx(1.0)
    edges, mids = time_grid(cfg)
    for seed in range(30):
        strong = _zigzag_strong(np.random.default_rng(seed), cfg, edges, mids)
        legacy = _zigzag_legacy(np.random.default_rng(seed), cfg, edges, mids)
        assert strong[0] == legacy[0]
        assert strong[1] == legacy[1]
        assert strong[2]["demography_growth_rates"] == legacy[2]["demography_growth_rates"]


def test_demography_semantic_version_was_bumped() -> None:
    cfg = Config()
    assert cfg.demography_mixture_version == "v0.9-merged-demography-taxonomy"
    assert cfg.p_canonical_zigzag_1s14 == 0.0
    assert cfg.p_canonical_smcpp_sawtooth == 0.0
    assert cfg.p_zigzag_strong == pytest.approx(0.01)
    assert cfg.p_zigzag == pytest.approx(0.005)
    assert SAMPLE_GENERATOR_SCHEMA_VERSION == 3
