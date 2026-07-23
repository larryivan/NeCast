from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path


SOURCE_IDS = {
    "clean_constant_map": 0,
    "random_global_mu_r": 1,
    "custom_heterogeneous": 2,
    "empirical_map_slice": 3,
    "stdpopsim_anchor": 4,
}
ID_TO_SOURCE = {v: k for k, v in SOURCE_IDS.items()}

DEMOGRAPHY_TYPE_IDS = {
    "constant": 0,
    "random_walk": 1,
    "bottleneck": 2,
    "exponential_growth": 3,
    "exponential_decline": 4,
    "expansion": 5,
    "contraction": 6,
    "three_epoch": 7,
    "serial_founder": 8,
    "zigzag": 9,
    "empirical_human_ne_template": 10,
    "stdpopsim": 11,
}
ID_TO_DEMOGRAPHY_TYPE = {v: k for k, v in DEMOGRAPHY_TYPE_IDS.items()}

# Append-only recipe IDs preserve the exact generator provenance used by older
# datasets, where these values were stored as ``demography_id``.
GENERATOR_RECIPE_IDS = {
    "smooth_random_walk": 0,
    "single_bottleneck": 1,
    "recent_bottleneck": 2,
    "expansion": 3,
    "contraction": 4,
    "three_epoch": 5,
    "zigzag": 6,
    "ancient_event": 7,
    "constant": 8,
    "stdpopsim": 9,
    "near_constant": 10,
    "continuous_exponential_growth": 11,
    "continuous_exponential_decline": 12,
    "recent_founder_recovery": 13,
    "serial_founder": 14,
    "ancient_recent_compound": 15,
    "oscillating_mild": 16,
    "zigzag_strong": 17,
    "smooth_random_walk_stress": 18,
    "recent_bottleneck_extreme": 19,
    "founder_recovery_extreme": 20,
    "rapid_recent_growth_extreme": 21,
    "serial_founder_extreme": 22,
    "ancient_recent_conflict": 23,
    "dense_random_walk": 24,
    "dense_multiscale_random_walk": 25,
    "dense_recent_wiggle": 26,
    "dense_random_walk_stress": 27,
    "classic_sawtooth": 28,
    "empirical_human_ne_template": 29,
    "ultra_stress": 30,
    "canonical_zigzag_1s14": 31,
    "canonical_smcpp_sawtooth": 32,
}
ID_TO_GENERATOR_RECIPE = {v: k for k, v in GENERATOR_RECIPE_IDS.items()}
# Preserve the original API and numeric namespace: ``DEMO_IDS`` and
# ``demography_id`` remain aliases for the fine-grained generator recipe.
DEMO_IDS = GENERATOR_RECIPE_IDS
ID_TO_DEMO = ID_TO_GENERATOR_RECIPE

# ``None`` marks recipes whose public type depends on the sampled trajectory.
GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE: dict[str, str | None] = {
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
    "stdpopsim": "stdpopsim",
    "ancient_event": None,
    "ultra_stress": None,
}

ULTRA_STRESS_TYPE_TO_DEMOGRAPHY_TYPE = {
    "ultra_random_walk": "random_walk",
    "ultra_recent_bottleneck": "bottleneck",
    "ultra_founder_recovery": "bottleneck",
    "ultra_rapid_recent_growth": "exponential_growth",
    "ultra_serial_founder": "serial_founder",
    "ultra_ancient_recent_conflict": "three_epoch",
}

HISTORY_FAMILY_BY_DEMOGRAPHY_TYPE = {
    "constant": "stable",
    "random_walk": "free_form",
    "bottleneck": "single_pulse",
    "exponential_growth": "monotonic_change",
    "exponential_decline": "monotonic_change",
    "expansion": "monotonic_change",
    "contraction": "monotonic_change",
    "three_epoch": "compound_events",
    "serial_founder": "compound_events",
    "zigzag": "oscillatory",
    "empirical_human_ne_template": "free_form",
    "stdpopsim": "external_model",
}
GENERATOR_RECIPE_HISTORY_FAMILY_OVERRIDES = {
    # Keep the strict ``constant`` public label uncontaminated while still
    # recognizing the near-constant control as part of the stable family.
    "near_constant": "stable",
}

DEFAULT_RECOMB_MAP = "maps/recomb_map/output/palsson_2025_autosomes_hg38.bed"
DEFAULT_MUT_MAP = "maps/mut_map/output/roulette_2023_autosomes_hg38_1000bp.bed"

SOURCE_PROB_FIELDS = (
    "p_clean_constant",
    "p_random_global",
    "p_custom_hetero",
    "p_empirical_slice",
    "p_stdpopsim_anchor",
)

EMPIRICAL_NE_SOURCE_KEYS = frozenset(
    {
        "smcpp_1kg",
        "phlash_unified",
        "phlash_super",
        "msmc2_example",
        "msmc_im_example",
    }
)

DEMOGRAPHY_PROB_FIELDS = (
    "p_constant",
    "p_near_constant",
    "p_smooth_random_walk",
    "p_dense_random_walk",
    "p_dense_multiscale_random_walk",
    "p_dense_recent_wiggle",
    "p_classic_sawtooth",
    "p_canonical_zigzag_1s14",
    "p_canonical_smcpp_sawtooth",
    "p_empirical_human_ne_template",
    "p_smooth_random_walk_stress",
    "p_dense_random_walk_stress",
    "p_ultra_stress",
    "p_single_bottleneck",
    "p_recent_bottleneck",
    "p_recent_bottleneck_extreme",
    "p_recent_founder_recovery",
    "p_founder_recovery_extreme",
    "p_continuous_exponential_growth",
    "p_rapid_recent_growth_extreme",
    "p_continuous_exponential_decline",
    "p_three_epoch",
    "p_serial_founder",
    "p_serial_founder_extreme",
    "p_ancient_event",
    "p_ancient_recent_compound",
    "p_ancient_recent_conflict",
    "p_oscillating_mild",
    "p_zigzag_strong",
    "p_expansion",
    "p_contraction",
    "p_zigzag",
)

UINT32_MAX = 2**32 - 1
INT16_MAX = 2**15 - 1
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
FLOAT32_MIN_POSITIVE = 2.0**-149
MAX_MSPRIME_RATE = 1.0e-4
MAX_SYNTHETIC_RATE_MULTIPLIER = 200.0
MAX_BASELINE_RATE = MAX_MSPRIME_RATE / MAX_SYNTHETIC_RATE_MULTIPLIER


@dataclass
class Config:
    dataset_name: str = "DLCoalSim-Core-v0.1"
    preset: str = "mini"
    out_dir: str = "./DLCoalSim-Out"
    seed: int = 12345
    force: bool = False
    demography_mixture_version: str = "v0.9-merged-demography-taxonomy"

    n_samples: int = 2000
    shard_size: int = 64
    compute_chunk_size: int = 16
    seq_len: int = 1_000_000
    n_haplotypes: int = 32
    # Supervised target resolution. Population-history samplers use the
    # separate generation grid below so changing label resolution cannot alter
    # the latent history or consume a different random-number sequence.
    time_bins: int = 64
    demography_generation_bins: int = 48
    min_time: float = 50.0
    max_time: float = 500_000.0

    baseline_mu: float = 1.25e-8
    baseline_rec: float = 1.0e-8

    p_clean_constant: float = 0.25
    p_random_global: float = 0.20
    p_custom_hetero: float = 0.45
    p_empirical_slice: float = 0.10
    p_stdpopsim_anchor: float = 0.00

    p_constant: float = 0.04
    p_near_constant: float = 0.05
    p_smooth_random_walk: float = 0.15
    p_dense_random_walk: float = 0.09
    p_dense_multiscale_random_walk: float = 0.06
    p_dense_recent_wiggle: float = 0.045
    p_classic_sawtooth: float = 0.03
    p_canonical_zigzag_1s14: float = 0.0
    p_canonical_smcpp_sawtooth: float = 0.0
    p_empirical_human_ne_template: float = 0.110
    p_smooth_random_walk_stress: float = 0.008
    p_dense_random_walk_stress: float = 0.004
    p_ultra_stress: float = 0.005
    p_single_bottleneck: float = 0.06
    p_recent_bottleneck: float = 0.08
    p_recent_bottleneck_extreme: float = 0.005
    p_recent_founder_recovery: float = 0.055
    p_founder_recovery_extreme: float = 0.004
    p_continuous_exponential_growth: float = 0.045
    p_rapid_recent_growth_extreme: float = 0.003
    p_continuous_exponential_decline: float = 0.025
    p_three_epoch: float = 0.035
    p_serial_founder: float = 0.025
    p_serial_founder_extreme: float = 0.003
    p_ancient_event: float = 0.015
    p_ancient_recent_compound: float = 0.02
    p_ancient_recent_conflict: float = 0.003
    p_oscillating_mild: float = 0.01
    p_zigzag_strong: float = 0.01
    p_expansion: float = 0.0025
    p_contraction: float = 0.0025
    p_zigzag: float = 0.005

    empirical_ne_template_dir: str = "data/templates/human_ne"
    empirical_ne_template_source_mix: str = "smcpp_1kg:0.25,phlash_unified:0.60,phlash_super:0.10,msmc2_example:0.04,msmc_im_example:0.01"
    empirical_ne_template_log10_shift_sd: float = 0.12
    empirical_ne_template_contrast_sd: float = 0.10
    empirical_ne_template_time_warp_sd: float = 0.15
    empirical_ne_template_bin_noise_sd: float = 0.025

    map_segments_min: int = 128
    map_segments_max: int = 512
    observed_map_noise: float = 0.25
    hotspot_missing_prob: float = 0.10

    recomb_map: str = DEFAULT_RECOMB_MAP
    mut_map: str = DEFAULT_MUT_MAP
    recomb_map_unit: str = "per_bp"
    mut_map_unit: str = "per_bp"
    mut_map_scale_to_baseline: bool = True

    genotype_error_min: float = 0.001
    genotype_error_max: float = 0.004
    missing_rate_min: float = 0.002
    missing_rate_max: float = 0.020
    phase_switch_rate_per_mb_min: float = 0.0
    phase_switch_rate_per_mb_max: float = 0.05

    enable_stdpopsim: bool = False
    stdpopsim_species: str = "HomSap"
    stdpopsim_models: str = ""
    stdpopsim_contig_length: int = 0

    workers: int = 1
    compression: bool = True
    compression_level: int = 1
    include_truth: bool = False
    max_retries: int = 3


def apply_preset(cfg: Config) -> Config:
    if cfg.preset == "mini":
        cfg.dataset_name = "DLCoalSim-Mini"
        cfg.n_samples = 512
        cfg.shard_size = 32
        cfg.seq_len = 1_000_000
        cfg.n_haplotypes = 24
        cfg.time_bins = 64
        cfg.demography_generation_bins = 32
        cfg.workers = 1
        cfg.enable_stdpopsim = False
        cfg.p_stdpopsim_anchor = 0.0
    elif cfg.preset == "core_v01":
        cfg.dataset_name = "DLCoalSim-Core-v0.1"
        cfg.n_samples = 20_000
        cfg.shard_size = 128
        cfg.seq_len = 10_000_000
        cfg.n_haplotypes = 64
        cfg.time_bins = 64
        cfg.demography_generation_bins = 48
        cfg.workers = -1
        cfg.p_clean_constant = 0.25
        cfg.p_random_global = 0.20
        cfg.p_custom_hetero = 0.45
        cfg.p_empirical_slice = 0.10
        cfg.p_stdpopsim_anchor = 0.0
    elif cfg.preset == "full_v01":
        cfg.dataset_name = "DLCoalSim-Full-v0.1"
        cfg.n_samples = 100_000
        cfg.shard_size = 128
        cfg.seq_len = 20_000_000
        cfg.n_haplotypes = 64
        cfg.time_bins = 64
        cfg.demography_generation_bins = 48
        cfg.workers = -1
        cfg.p_clean_constant = 0.10
        cfg.p_random_global = 0.15
        cfg.p_custom_hetero = 0.60
        cfg.p_empirical_slice = 0.15
        cfg.p_stdpopsim_anchor = 0.0
    else:
        raise ValueError(f"Unknown preset: {cfg.preset}")
    return cfg


def validate_config(cfg: Config) -> None:
    def nonempty_string(name: str) -> str:
        value = getattr(cfg, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    def integer(name: str) -> int:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    def finite(name: str) -> float:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be a finite number")
        return result

    def positive_integer(name: str) -> int:
        value = integer(name)
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0")
        return value

    def probability_range(min_name: str, max_name: str) -> None:
        minimum = finite(min_name)
        maximum = finite(max_name)
        if not 0.0 <= minimum <= 1.0:
            raise ValueError(f"{min_name} must be between 0 and 1")
        if not 0.0 <= maximum <= 1.0:
            raise ValueError(f"{max_name} must be between 0 and 1")
        if minimum > maximum:
            raise ValueError(f"{min_name} must not exceed {max_name}")

    def nonnegative_range(min_name: str, max_name: str) -> None:
        minimum = finite(min_name)
        maximum = finite(max_name)
        if minimum < 0.0:
            raise ValueError(f"{min_name} must be non-negative")
        if maximum < 0.0:
            raise ValueError(f"{max_name} must be non-negative")
        if minimum > maximum:
            raise ValueError(f"{min_name} must not exceed {max_name}")

    def boolean(name: str) -> bool:
        value = getattr(cfg, name)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        return value

    def weight_group(name: str, fields: tuple[str, ...]) -> float:
        total = 0.0
        for field in fields:
            value = finite(field)
            if value < 0.0:
                raise ValueError(f"{field} must be non-negative")
            total += value
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(f"at least one {name} probability must be greater than 0")
        return total

    def empirical_ne_source_mix() -> None:
        text = cfg.empirical_ne_template_source_mix
        if not isinstance(text, str):
            raise ValueError("empirical_ne_template_source_mix must be a string")

        total = 0.0
        found = False
        seen: set[str] = set()
        for raw_token in text.split(","):
            token = raw_token.strip()
            if not token:
                continue
            found = True
            if ":" in token:
                key, raw_weight = token.split(":", 1)
                key = key.strip()
                raw_weight = raw_weight.strip()
                if not key or not raw_weight:
                    raise ValueError(
                        "empirical_ne_template_source_mix entries must use 'source[:weight]'"
                    )
                try:
                    weight = float(raw_weight)
                except ValueError as exc:
                    raise ValueError(
                        f"empirical_ne_template_source_mix has a non-numeric weight for {key!r}"
                    ) from exc
            else:
                key = token
                weight = 1.0
            if key not in EMPIRICAL_NE_SOURCE_KEYS:
                supported = ", ".join(sorted(EMPIRICAL_NE_SOURCE_KEYS))
                raise ValueError(
                    f"empirical_ne_template_source_mix contains unknown source {key!r}; "
                    f"supported sources: {supported}"
                )
            if key in seen:
                raise ValueError(
                    f"empirical_ne_template_source_mix contains duplicate source {key!r}"
                )
            seen.add(key)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"empirical_ne_template_source_mix weight for {key!r} must be finite and non-negative"
                )
            total += weight
            if not math.isfinite(total):
                raise ValueError("empirical_ne_template_source_mix weights sum to a non-finite value")
        if not found or total <= 0.0:
            raise ValueError(
                "empirical_ne_template_source_mix must contain at least one positive source weight"
            )

    nonempty_string("dataset_name")
    if cfg.preset not in {"mini", "core_v01", "full_v01"}:
        raise ValueError("preset must be 'mini', 'core_v01', or 'full_v01'")
    out_dir_text = nonempty_string("out_dir")
    out_dir = Path(out_dir_text).expanduser().resolve()
    protected_output_dirs = {
        Path(out_dir.anchor).resolve(),
        Path.home().resolve(),
        Path(__file__).resolve().parents[1],
    }
    if out_dir in protected_output_dirs:
        raise ValueError(
            f"out_dir must be a dedicated dataset directory, not protected path {out_dir}"
        )
    nonempty_string("demography_mixture_version")
    nonempty_string("empirical_ne_template_dir")
    for name in ("recomb_map", "mut_map", "stdpopsim_models"):
        value = getattr(cfg, name)
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
    nonempty_string("stdpopsim_species")

    positive_integer("n_samples")
    positive_integer("shard_size")
    positive_integer("compute_chunk_size")
    seq_len = positive_integer("seq_len")
    n_haplotypes = positive_integer("n_haplotypes")
    positive_integer("time_bins")
    positive_integer("demography_generation_bins")

    if n_haplotypes < 2:
        raise ValueError("n_haplotypes must be at least 2")
    if n_haplotypes % 2 != 0:
        raise ValueError("n_haplotypes must be even because samples are simulated as diploid individuals")
    if n_haplotypes > INT16_MAX:
        raise ValueError(f"n_haplotypes must not exceed the int16 storage limit ({INT16_MAX})")
    packed_hap_bytes = math.ceil(n_haplotypes / 8)
    if packed_hap_bytes > INT16_MAX:
        raise ValueError(f"packed haplotype width must not exceed the int16 storage limit ({INT16_MAX})")

    if seq_len > UINT32_MAX:
        raise ValueError(f"seq_len must not exceed the uint32 position limit ({UINT32_MAX})")
    if seq_len * n_haplotypes > UINT32_MAX:
        raise ValueError(
            "seq_len * n_haplotypes must fit in uint32 so sparse flat genotype indices remain representable"
        )

    min_time = finite("min_time")
    max_time = finite("max_time")
    if min_time <= 0.0:
        raise ValueError("min_time must be greater than 0")
    if max_time <= min_time:
        raise ValueError("max_time must be greater than min_time")

    for name in ("baseline_mu", "baseline_rec"):
        value = finite(name)
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than 0")
        if value < FLOAT32_MIN_POSITIVE:
            raise ValueError(f"{name} is too small to remain positive in float32 storage")
        if value > MAX_BASELINE_RATE:
            raise ValueError(
                f"{name} must not exceed {MAX_BASELINE_RATE:g} per bp per generation "
                "because synthetic maps can amplify a baseline rate by up to "
                f"{MAX_SYNTHETIC_RATE_MULTIPLIER:g}x"
            )
        if value > FLOAT32_MAX:
            raise ValueError(f"{name} exceeds the float32 storage limit")

    for name in (
        "force",
        "compression",
        "include_truth",
        "enable_stdpopsim",
        "mut_map_scale_to_baseline",
    ):
        boolean(name)

    weight_group("source", SOURCE_PROB_FIELDS)
    enabled_source_total = sum(
        finite(name)
        for name in SOURCE_PROB_FIELDS
        if name != "p_stdpopsim_anchor"
    )
    if cfg.enable_stdpopsim:
        enabled_source_total += finite("p_stdpopsim_anchor")
    if enabled_source_total <= 0.0:
        raise ValueError("at least one enabled source probability must be greater than 0")
    weight_group("demography", DEMOGRAPHY_PROB_FIELDS)
    empirical_ne_source_mix()

    probability_range("genotype_error_min", "genotype_error_max")
    probability_range("missing_rate_min", "missing_rate_max")
    nonnegative_range("phase_switch_rate_per_mb_min", "phase_switch_rate_per_mb_max")

    hotspot_missing_prob = finite("hotspot_missing_prob")
    if not 0.0 <= hotspot_missing_prob <= 1.0:
        raise ValueError("hotspot_missing_prob must be between 0 and 1")

    for name in (
        "observed_map_noise",
        "empirical_ne_template_log10_shift_sd",
        "empirical_ne_template_contrast_sd",
        "empirical_ne_template_time_warp_sd",
        "empirical_ne_template_bin_noise_sd",
    ):
        if finite(name) < 0.0:
            raise ValueError(f"{name} must be non-negative")

    map_segments_min = positive_integer("map_segments_min")
    map_segments_max = positive_integer("map_segments_max")
    if map_segments_min > map_segments_max:
        raise ValueError("map_segments_min must not exceed map_segments_max")

    max_retries = integer("max_retries")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    workers = integer("workers")
    if workers != -1 and workers < 1:
        raise ValueError("workers must be -1 or an integer greater than 0")

    compression_level = integer("compression_level")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")

    integer("seed")
    stdpopsim_contig_length = integer("stdpopsim_contig_length")
    if stdpopsim_contig_length < 0:
        raise ValueError("stdpopsim_contig_length must be 0 or greater")
    if stdpopsim_contig_length > UINT32_MAX:
        raise ValueError(f"stdpopsim_contig_length must not exceed the uint32 position limit ({UINT32_MAX})")
    if stdpopsim_contig_length and stdpopsim_contig_length * n_haplotypes > UINT32_MAX:
        raise ValueError(
            "stdpopsim_contig_length * n_haplotypes must fit in uint32 for sparse flat genotype indices"
        )

    if cfg.recomb_map_unit not in {"per_bp", "cM_per_Mb"}:
        raise ValueError("recomb_map_unit must be 'per_bp' or 'cM_per_Mb'")
    if cfg.mut_map_unit not in {"per_bp", "relative", "roulette_raw"}:
        raise ValueError("mut_map_unit must be 'per_bp', 'relative', or 'roulette_raw'")
