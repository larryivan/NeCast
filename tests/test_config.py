from __future__ import annotations

import dataclasses

import pytest

from simulator.config import (
    DEMOGRAPHY_PROB_FIELDS,
    FLOAT32_MIN_POSITIVE,
    INT16_MAX,
    SOURCE_PROB_FIELDS,
    UINT32_MAX,
    Config,
    apply_preset,
    validate_config,
)


def valid_config() -> Config:
    return apply_preset(Config(preset="mini"))


@pytest.mark.parametrize("preset", ["mini", "core_v01", "full_v01"])
def test_builtin_presets_are_valid(preset: str) -> None:
    validate_config(apply_preset(Config(preset=preset)))


@pytest.mark.parametrize(
    ("preset", "generation_bins"),
    [
        ("mini", 32),
        ("core_v01", 48),
        ("full_v01", 48),
    ],
)
def test_builtin_presets_use_64_targets_without_changing_history_prior(
    preset: str,
    generation_bins: int,
) -> None:
    cfg = apply_preset(Config(preset=preset))

    assert cfg.time_bins == 64
    assert cfg.demography_generation_bins == generation_bins


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": True}, "n_samples"),
        ({"shard_size": 0}, "shard_size"),
        ({"compute_chunk_size": 0}, "compute_chunk_size"),
        ({"seq_len": 0}, "seq_len"),
        ({"time_bins": 0}, "time_bins"),
        ({"demography_generation_bins": 0}, "demography_generation_bins"),
        ({"n_haplotypes": 1}, "at least 2"),
        ({"n_haplotypes": 3}, "even"),
        ({"n_haplotypes": INT16_MAX + 1}, "int16"),
        ({"min_time": 0.0}, "min_time"),
        ({"min_time": 500.0, "max_time": 500.0}, "max_time"),
        ({"min_time": float("nan")}, "min_time"),
        ({"baseline_mu": 0.0}, "baseline_mu"),
        ({"baseline_rec": -1.0e-8}, "baseline_rec"),
        ({"baseline_mu": FLOAT32_MIN_POSITIVE / 2.0}, "float32"),
        ({"baseline_mu": 5.1e-7}, "synthetic maps"),
        ({"baseline_rec": 1.1e-4}, "baseline_rec"),
        ({"p_clean_constant": -0.01}, "p_clean_constant"),
        ({"genotype_error_min": -0.01}, "genotype_error_min"),
        ({"genotype_error_max": 1.01}, "genotype_error_max"),
        ({"genotype_error_min": 0.2, "genotype_error_max": 0.1}, "must not exceed"),
        ({"missing_rate_min": -0.01}, "missing_rate_min"),
        ({"missing_rate_max": 1.01}, "missing_rate_max"),
        ({"missing_rate_min": 0.2, "missing_rate_max": 0.1}, "must not exceed"),
        ({"phase_switch_rate_per_het_min": -0.1}, "phase_switch_rate_per_het_min"),
        (
            {"phase_switch_rate_per_het_min": 0.2, "phase_switch_rate_per_het_max": 0.1},
            "must not exceed",
        ),
        ({"hotspot_missing_prob": -0.1}, "hotspot_missing_prob"),
        ({"hotspot_missing_prob": 1.1}, "hotspot_missing_prob"),
        ({"observed_map_noise": -0.1}, "observed_map_noise"),
        ({"observed_map_noise": float("inf")}, "observed_map_noise"),
        ({"empirical_ne_template_log10_shift_sd": -0.1}, "log10_shift"),
        ({"empirical_ne_template_contrast_sd": -0.1}, "contrast"),
        ({"empirical_ne_template_time_warp_sd": -0.1}, "time_warp"),
        ({"empirical_ne_template_bin_noise_sd": -0.1}, "bin_noise"),
        ({"map_segments_min": 0}, "map_segments_min"),
        ({"map_segments_min": 10, "map_segments_max": 9}, "map_segments_min"),
        ({"max_retries": 0}, "max_retries"),
        ({"workers": 0}, "workers"),
        ({"workers": -2}, "workers"),
        ({"compression_level": -1}, "compression_level"),
        ({"compression_level": 10}, "compression_level"),
        ({"compression_level": True}, "compression_level"),
        ({"seed": 1.5}, "seed"),
        ({"dataset_name": ""}, "dataset_name"),
        ({"preset": "unknown"}, "preset"),
        ({"out_dir": ""}, "out_dir"),
        ({"out_dir": "."}, "dedicated dataset directory"),
        ({"demography_mixture_version": ""}, "demography_mixture_version"),
        ({"empirical_ne_template_dir": ""}, "empirical_ne_template_dir"),
        ({"stdpopsim_species": ""}, "stdpopsim_species"),
        ({"stdpopsim_contig_length": -1}, "stdpopsim_contig_length"),
        ({"recomb_map_unit": "centimorgans"}, "recomb_map_unit"),
        ({"mut_map_unit": "unknown"}, "mut_map_unit"),
    ],
)
def test_invalid_scalar_and_range_settings_are_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_config(dataclasses.replace(valid_config(), **changes))


def test_source_mixture_requires_a_positive_weight() -> None:
    cfg = dataclasses.replace(valid_config(), **{name: 0.0 for name in SOURCE_PROB_FIELDS})
    with pytest.raises(ValueError, match="source probability"):
        validate_config(cfg)


def test_disabled_stdpopsim_weight_is_not_an_enabled_source() -> None:
    source_weights = {name: 0.0 for name in SOURCE_PROB_FIELDS}
    source_weights["p_stdpopsim_anchor"] = 1.0
    cfg = dataclasses.replace(valid_config(), **source_weights, enable_stdpopsim=False)
    with pytest.raises(ValueError, match="enabled source probability"):
        validate_config(cfg)

    validate_config(dataclasses.replace(cfg, enable_stdpopsim=True))


def test_demography_mixture_requires_a_positive_weight() -> None:
    cfg = dataclasses.replace(valid_config(), **{name: 0.0 for name in DEMOGRAPHY_PROB_FIELDS})
    with pytest.raises(ValueError, match="demography probability"):
        validate_config(cfg)


def test_sequence_positions_must_fit_uint32_storage() -> None:
    with pytest.raises(ValueError, match="uint32 position"):
        validate_config(dataclasses.replace(valid_config(), seq_len=UINT32_MAX + 1))


def test_dense_flat_genotype_indices_must_fit_uint32_storage() -> None:
    cfg = dataclasses.replace(
        valid_config(),
        seq_len=UINT32_MAX // 2 + 1,
        n_haplotypes=2,
    )
    with pytest.raises(ValueError, match="sparse flat genotype"):
        validate_config(cfg)


def test_stdpopsim_contig_positions_must_fit_uint32_storage() -> None:
    cfg = dataclasses.replace(valid_config(), stdpopsim_contig_length=UINT32_MAX + 1)
    with pytest.raises(ValueError, match="uint32 position"):
        validate_config(cfg)


def test_stdpopsim_flat_indices_must_fit_uint32_storage() -> None:
    cfg = dataclasses.replace(
        valid_config(),
        n_haplotypes=2,
        stdpopsim_contig_length=UINT32_MAX // 2 + 1,
    )
    with pytest.raises(ValueError, match="sparse flat genotype"):
        validate_config(cfg)


def test_probability_fields_are_weights_and_need_not_sum_to_one() -> None:
    source_weights = {name: 0.0 for name in SOURCE_PROB_FIELDS}
    source_weights["p_custom_hetero"] = 4.0
    demography_weights = {name: 0.0 for name in DEMOGRAPHY_PROB_FIELDS}
    demography_weights["p_constant"] = 2.0
    validate_config(dataclasses.replace(valid_config(), **source_weights, **demography_weights))


@pytest.mark.parametrize(
    "source_mix",
    [
        "",
        ", ,",
        "smcpp_1kg:",
        ":1",
        "smcpp_1kg:not-a-number",
        "smcpp_1kg:-0.1",
        "smcpp_1kg:nan",
        "smcpp_1kg:0,phlash_unified:0",
        "smcpp_1kg:1,smcpp_1kg:0",
        "smcpp_1kg:1e308,phlash_unified:1e308",
        "smcpp_typo:1",
    ],
)
def test_invalid_empirical_ne_source_mix_is_rejected(source_mix: str) -> None:
    cfg = dataclasses.replace(valid_config(), empirical_ne_template_source_mix=source_mix)
    with pytest.raises(ValueError, match="empirical_ne_template_source_mix"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "name",
    [
        "force",
        "compression",
        "include_truth",
        "enable_stdpopsim",
        "mut_map_scale_to_baseline",
    ],
)
def test_boolean_settings_reject_integer_substitutes(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        validate_config(dataclasses.replace(valid_config(), **{name: 1}))
