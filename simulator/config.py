from __future__ import annotations

from dataclasses import dataclass


SOURCE_IDS = {
    "clean_constant_map": 0,
    "random_global_mu_r": 1,
    "custom_heterogeneous": 2,
    "empirical_map_slice": 3,
    "stdpopsim_anchor": 4,
}
ID_TO_SOURCE = {v: k for k, v in SOURCE_IDS.items()}

DEMO_IDS = {
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
}
ID_TO_DEMO = {v: k for k, v in DEMO_IDS.items()}

# deCODE 2019 (Halldorsson et al., Science 363:eaau1043): 4,531,535 localised
# crossovers, average map resolution 682 bp, native GRCh38.  It replaces the
# Palsson 2025 table, which is published in 1 Mb windows and therefore holds
# no structure at the scale that breaks haplotype tracts -- its crossover
# column is this same map re-binned.  Kept beside it for the megabase-scale
# non-crossover information, which is a different quantity.
DEFAULT_RECOMB_MAP = "maps/recomb_map/output/decode2019_avg_hg38.bed"
LEGACY_RECOMB_MAP = "maps/recomb_map/output/palsson_2025_autosomes_hg38.bed"
DEFAULT_MUT_MAP = "maps/mut_map/output/roulette_2023_autosomes_hg38_1000bp.bed"

@dataclass
class Config:
    dataset_name: str = "DLCoalSim-Core-v0.1"
    preset: str = "mini"
    out_dir: str = "./DLCoalSim-Out"
    seed: int = 12345
    force: bool = False
    demography_mixture_version: str = "v0.7-capped-stress-human-ne"

    n_samples: int = 2000
    shard_size: int = 64
    seq_len: int = 1_000_000
    n_haplotypes: int = 32
    time_bins: int = 48
    min_time: float = 50.0
    max_time: float = 500_000.0

    baseline_mu: float = 1.25e-8
    baseline_rec: float = 1.0e-8

    p_clean_constant: float = 0.25
    p_random_global: float = 0.20
    p_custom_hetero: float = 0.35
    p_empirical_slice: float = 0.20
    p_stdpopsim_anchor: float = 0.00

    p_constant: float = 0.04
    p_near_constant: float = 0.05
    p_smooth_random_walk: float = 0.15
    p_dense_random_walk: float = 0.09
    p_dense_multiscale_random_walk: float = 0.06
    p_dense_recent_wiggle: float = 0.045
    p_classic_sawtooth: float = 0.03
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
    # The deCODE bigWig is distributed in cM/Mb; writing the source units
    # rather than pre-converting keeps the file checkable against UCSC.
    recomb_map_unit: str = "cM_per_Mb"
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
    include_truth: bool = False
    max_retries: int = 3


def apply_preset(cfg: Config) -> Config:
    if cfg.preset == "mini":
        cfg.dataset_name = "DLCoalSim-Mini"
        cfg.n_samples = 512
        cfg.shard_size = 32
        cfg.seq_len = 1_000_000
        cfg.n_haplotypes = 24
        cfg.time_bins = 32
        cfg.workers = 1
        cfg.enable_stdpopsim = False
        cfg.p_stdpopsim_anchor = 0.0
    elif cfg.preset == "core_v01":
        cfg.dataset_name = "DLCoalSim-Core-v0.1"
        cfg.n_samples = 20_000
        cfg.shard_size = 128
        cfg.seq_len = 10_000_000
        cfg.n_haplotypes = 64
        cfg.time_bins = 48
        cfg.workers = -1
        cfg.p_clean_constant = 0.25
        cfg.p_random_global = 0.20
        cfg.p_custom_hetero = 0.35
        cfg.p_empirical_slice = 0.20
        cfg.p_stdpopsim_anchor = 0.0
    elif cfg.preset == "full_v01":
        cfg.dataset_name = "DLCoalSim-Full-v0.1"
        cfg.n_samples = 100_000
        cfg.shard_size = 128
        cfg.seq_len = 20_000_000
        cfg.n_haplotypes = 64
        cfg.time_bins = 48
        cfg.workers = -1
        cfg.p_clean_constant = 0.10
        cfg.p_random_global = 0.15
        cfg.p_custom_hetero = 0.55
        cfg.p_empirical_slice = 0.20
        cfg.p_stdpopsim_anchor = 0.0
    else:
        raise ValueError(f"Unknown preset: {cfg.preset}")
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.n_haplotypes < 2:
        raise ValueError("n_haplotypes must be at least 2")
    if cfg.n_haplotypes % 2 != 0:
        raise ValueError("n_haplotypes must be even because samples are simulated as diploid individuals")
