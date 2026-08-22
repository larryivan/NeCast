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

DEMOGRAPHY_TYPE_IDS = {
    # The public, scientific label for a row, coarser than the recipe that drew
    # it.  Four labels went when their recipes did -- exponential growth and
    # decline, three-epoch and serial founder are all things the continuous
    # event samplers produce, and which of them a row *is* depends on the
    # trajectory rather than on which sampler ran, so it is resolved per row.
    "constant": 0,
    "random_walk": 1,
    "bottleneck": 2,
    "expansion": 3,
    "contraction": 4,
    "zigzag": 5,
    "empirical_human_ne_template": 6,
    "stdpopsim": 7,
}
ID_TO_DEMOGRAPHY_TYPE = {v: k for k, v in DEMOGRAPHY_TYPE_IDS.items()}

# Append-only recipe IDs preserve the exact generator provenance used by older
# datasets, where these values were stored as ``demography_id``.
GENERATOR_RECIPE_IDS = {
    # The recipe a row was drawn from.  Ids are dense and start at zero: they
    # name a sampler in this generator, not a position in its history, and a
    # dataset written before a change is refused by its schema version long
    # before any id is read.
    "constant": 0,
    "spectral_walk": 1,
    "size_event": 2,
    "bottleneck": 3,
    "multi_event": 4,
    "event_on_background": 5,
    "ooa_chain": 6,
    "exponential": 7,
    "sawtooth": 8,
    "literature_sawtooth": 9,
    "empirical_human_ne_template": 10,
    "stdpopsim_single_pop": 11,
    "stdpopsim": 12,
}
ID_TO_GENERATOR_RECIPE = {v: k for k, v in GENERATOR_RECIPE_IDS.items()}
# ``None`` marks recipes whose public type depends on the sampled trajectory.
GENERATOR_RECIPE_TO_DEMOGRAPHY_TYPE: dict[str, str | None] = {
    # ``None`` marks a recipe whose public label is a property of the drawn
    # trajectory rather than of the sampler: a size change is an expansion or a
    # contraction depending on its sign.
    "constant": "constant",
    "spectral_walk": "random_walk",
    "size_event": None,
    "bottleneck": "bottleneck",
    "multi_event": None,
    "event_on_background": None,
    "ooa_chain": "bottleneck",
    "exponential": None,
    "sawtooth": "zigzag",
    "literature_sawtooth": "zigzag",
    "empirical_human_ne_template": "empirical_human_ne_template",
    "stdpopsim_single_pop": "stdpopsim",
    "stdpopsim": "stdpopsim",
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

# deCODE 2019 (Halldorsson et al., Science 363:eaau1043): 4,531,535 localised
# crossovers, average map resolution 682 bp, native GRCh38.  It replaces the
# Palsson 2025 table, which is published in 1 Mb windows and so holds nothing
# at the scale that breaks haplotype tracts -- its crossover column is this
# same map re-binned.  Kept beside it for the megabase-scale non-crossover
# information, which is a different quantity.
# Paths are resolved against ``simulator/assets`` first and the working
# directory second, so an installed copy finds its own data from anywhere.
DEFAULT_RECOMB_MAP = "maps/decode2019_avg_hg38.bed"
DEFAULT_MUT_MAP = "maps/roulette_2023_autosomes_hg38_1000bp.bed"

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
    "p_spectral_walk",
    "p_size_event",
    "p_multi_event",
    "p_exponential",
    "p_sawtooth",
    "p_literature_sawtooth",
    "p_bottleneck",
    "p_event_on_background",
    "p_ooa_chain",
    "p_empirical_human_ne_template",
    "p_stdpopsim_single_pop",
)

UINT32_MAX = 2**32 - 1
INT16_MAX = 2**15 - 1
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
FLOAT32_MIN_POSITIVE = 2.0**-149
MAX_MSPRIME_RATE = 1.0e-4
MAX_SYNTHETIC_RATE_MULTIPLIER = 200.0


# Effective size bounds, declared here instead of at twenty clip sites.
#
# The anchor is the project's own real data: ``_empirical_human_ne_template``
# draws its curves from smc++ and phlash run on 1000 Genomes, and measured over
# 1,500 draws its support is log10 Ne 2.963 to 5.711.  The regular bound leaves
# roughly an order of magnitude on each side of that -- a hundred individuals
# below, a million above -- which is also exactly the support the samplers
# already produced between them, so the distribution is unchanged and only the
# reason for it becomes explicit.
#
# Before this, twenty clip sites used at least five different pairs: 2.7 with
# an upper end of 5.7, 5.85 or 5.9 depending on which walk it was, and 200 to
# 500,000 or 100 to 1,000,000 for the event samplers.  Different samplers
# therefore had different supports, and since each has a characteristic curve
# shape that is a correspondence between shape and range which holds in this
# generator and nowhere in biology.
#
# Stress samplers keep a wider band on purpose.  That is a design choice rather
# than an accident, so it is named, and every sample records which band it used.
NE_LOG10_BOUNDS = (2.0, 6.0)
NE_LOG10_STRESS_BOUNDS = (1.7, 6.3)
NE_BOUNDS = (10.0 ** NE_LOG10_BOUNDS[0], 10.0 ** NE_LOG10_BOUNDS[1])
NE_STRESS_BOUNDS = (10.0 ** NE_LOG10_STRESS_BOUNDS[0], 10.0 ** NE_LOG10_STRESS_BOUNDS[1])
MAX_BASELINE_RATE = MAX_MSPRIME_RATE / MAX_SYNTHETIC_RATE_MULTIPLIER


@dataclass
class Config:
    dataset_name: str = "DLCoalSim-Core-v0.1"
    preset: str = "mini"
    out_dir: str = "./DLCoalSim-Out"
    seed: int = 12345
    force: bool = False
    demography_mixture_version: str = "v1"

    n_samples: int = 2000
    shard_size: int = 64
    compute_chunk_size: int = 16
    seq_len: int = 1_000_000
    n_haplotypes: int = 32
    # Supervised target resolution.  Nothing about the latent history depends
    # on it: histories are continuous functions of time, discretised only far
    # enough to represent them, so the label grid can move without moving the
    # ground truth or the random-number stream.
    time_bins: int = 64
    # How much log10 Ne error the piecewise-constant representation of a
    # continuous history may carry.  A flat history costs one epoch and a rough
    # one costs as many as it needs; msprime's cost is flat in epoch count, so
    # this buys fidelity against metadata size and simulation time.
    #
    # It is set by the finest statistic the label carries, not by its RMSE.  The
    # amplitude and total variation of the mixture have both converged by 0.02,
    # and the label RMSE against an exact representation is 0.0001 there -- but
    # half of all adjacent-bin steps in the label are smaller than 0.02, so a
    # tolerance that size erases turning points: the median spectral walk turns
    # 5 times at 0.05, 7 at 0.02, and 9 at 0.005 and below, and 9 is the human
    # anchor the mixture is built against.  Costs 70 ms and 12.6 kB per row
    # against 186 s of simulation, so the tail of the sweep is affordable and
    # the body of it is not worth saving.
    demography_resolution_log10: float = 0.005
    min_time: float = 50.0
    max_time: float = 500_000.0

    baseline_mu: float = 1.25e-8
    baseline_rec: float = 1.0e-8

    p_clean_constant: float = 0.25
    p_random_global: float = 0.20
    p_custom_hetero: float = 0.35
    p_empirical_slice: float = 0.20
    p_stdpopsim_anchor: float = 0.00

    # The demography mixture.  Two rules shape it.  First, an axis that is
    # continuous in nature is one sampler with a continuous parameter, not
    # several samplers pinned to points along it -- the previous mixture had
    # 27 named recipes and a measured pairwise separability as low as 0.283
    # against a 0.500 chance rate.  Second, training weight and evaluation
    # weight are different things: the last three entries are histories other
    # people published, and rows drawn from them carry
    # ``held_out_evaluation`` or ``literature_demography_anchor`` so a split
    # can keep them out of the training gradient.
    # Weights were set against a measured quantity rather than a feeling: for
    # each recipe, the share of its rows whose label standard deviation is
    # under 0.24 -- the level every architecture arm sits at, so on those rows
    # a per-row constant predictor already beats every arm we have run.  They
    # carry loss but not discrimination, and they are exactly the rows on which
    # the level-only shortcut is close to optimal.  Across the mixture they
    # were 31.6% of it.
    #
    # Two recipes are deliberately not cut for it.  ``constant`` is the control
    # and is trivial on purpose; without it the model invents structure in flat
    # data.  ``bottleneck`` is 65% trivial by that measure, but only because a
    # dip occupies a few of 64 log-time bins, so its cross-bin spread is small
    # however deep it is -- cutting the most studied event in the field to
    # improve an aggregate metric would be optimising the metric, not the
    # science.  What moved instead is the balance inside the background family,
    # towards the two recipes that are both realistic and structure-demanding:
    # the empirical templates, only 13% trivial, and the founder chain, 9%.
    p_constant: float = 0.04
    # Carries what the four random walks, ``near_constant`` and the two
    # stress walks carried between them.
    p_spectral_walk: float = 0.200
    p_size_event: float = 0.10
    p_multi_event: float = 0.08
    p_exponential: float = 0.04
    p_sawtooth: float = 0.03
    # The published sawtooth's own neighbourhood.  ``p_sawtooth`` covers the
    # shape space broadly and reaches this shape only in its tail -- of six
    # thousand free draws the closest sat 0.199 RMSE away, against 0.312
    # between two ordinary draws -- and widening it made that worse, because
    # more axes buy coverage at the cost of density anywhere in particular.
    p_literature_sawtooth: float = 0.03
    # The most studied event in the field, and ``size_event`` was giving it one
    # row in fifty: only 12.8% of its draws are a drop of half a decade or more
    # that ends and comes back.  This draws from the same space, tilted onto
    # that corner.  The overlap is deliberate -- the alternative is a broader
    # prior, which puts less density everywhere rather than more density here.
    p_bottleneck: float = 0.10
    # A moving background with sharp events on it.  Across the other recipes
    # 28.4% of rows carry a sharp event and 41.0% a moving background, but only
    # 2.7% carry both, against the 11.6% independence would give: every other
    # recipe is one or the other by construction.  This is also the row a
    # shortcut fails on from either side.
    p_event_on_background: float = 0.086
    p_ooa_chain: float = 0.08
    p_empirical_human_ne_template: float = 0.19
    # Eighteen single-population models across eight species, verbatim.  Error
    # measured on these is evidence about generalisation in a way that error on
    # our own prior redrawn can never be, and because each model draws one
    # fixed trajectory, a run of them is a set of data replicates of a known
    # history -- which is what makes bias and variance on it readable straight
    # out of the dataset.  phlash's simulation study is built the same way,
    # three replicates per scenario.
    #
    # All eighteen models, so at 2,400 rows in a hundred thousand each gets
    # about 133 replicates.
    p_stdpopsim_single_pop: float = 0.024


    empirical_ne_template_dir: str = "templates/human_ne"
    stdpopsim_template_path: str = (
        "templates/stdpopsim_single_pop/trajectories.csv"
    )
    empirical_ne_template_source_mix: str = "smcpp_1kg:0.25,phlash_unified:0.60,phlash_super:0.10,msmc2_example:0.04,msmc_im_example:0.01"
    # The templates are 191 curves whose levels all sit within about half a
    # decade of each other, so a small shift left this recipe with a level
    # spread of 0.24 against 0.49 to 0.80 for every other recipe -- enough
    # that the level alone half-announces which recipe a row came from.
    empirical_ne_template_log10_shift_sd: float = 0.70
    empirical_ne_template_contrast_sd: float = 0.10
    empirical_ne_template_time_warp_sd: float = 0.15
    empirical_ne_template_bin_noise_sd: float = 0.025

    map_segments_min: int = 128
    map_segments_max: int = 512
    # A ratio, not an amplitude: the observation error's log spread divided by
    # the true map's own log spread.  It used to be absolute, and that cannot
    # serve two maps at once -- a recombination map's log rate has a standard
    # deviation near 0.5 across a row and a mutation map's near 0.04, so an
    # absolute 0.25 left the observed recombination map correlating 0.851 with
    # the truth and the observed mutation map 0.163.  Five to eight times more
    # noise than signal is not a noisy map, it is no map.
    observed_map_noise: float = 0.55
    # Noise on a real map handed to the model, kept separate from the
    # synthetic one because it stands for the error in a published map
    # rather than in a map this code generated.  At 0.25 the observed map
    # correlates 0.993 in log rate with the truth, which is to say it is the
    # truth; two real maps of one region correlate 0.687 at a kilobase.
    empirical_map_noise: float = 0.80
    hotspot_missing_prob: float = 0.10

    recomb_map: str = DEFAULT_RECOMB_MAP
    mut_map: str = DEFAULT_MUT_MAP
    # The deCODE bigWig is distributed in cM/Mb; writing the source units
    # rather than pre-converting keeps the file checkable against UCSC.
    recomb_map_unit: str = "cM_per_Mb"
    mut_map_unit: str = "per_bp"
    mut_map_scale_to_baseline: bool = True

    # Sites a caller invents.  Drawn per row on a log scale across four orders
    # of magnitude, because what survives variant calling is a property of the
    # pipeline rather than of the population: a hard-filtered callset and a raw
    # one are orders of magnitude apart.  Covering the range and recording the
    # draw is the honest treatment; picking one number is not.
    #
    # The range is set from what is observable rather than from a per-base
    # error rate: the share of singletons that are not real.  Published
    # callsets put that anywhere from a few percent after hard filtering to
    # roughly a third in a raw one, and at a typical effective size a megabase
    # carries about sixteen hundred singletons, so the rate that spans it runs
    # from about 1.5e-5 to 6e-4 per base pair.
    false_variant_rate_per_bp_min: float = 1.5e-5
    false_variant_rate_per_bp_max: float = 6.0e-4
    genotype_error_min: float = 0.001
    genotype_error_max: float = 0.004
    missing_rate_min: float = 0.002
    missing_rate_max: float = 0.020
    # Phasing errors happen between consecutive heterozygous sites, which is
    # where a phasing algorithm makes a decision, not at a fixed rate per base
    # pair.  Published switch-error rates for statistical phasing are 0.5% to
    # 2% of adjacent heterozygous pairs.  The previous physical-distance rate
    # of at most 0.05 per Mb produced 1.7 switches per diploid over twenty
    # megabases where the literature implies roughly a hundred, and it also
    # mis-weighted the genome: at a fixed rate per base pair a sparse region
    # accumulates switches it has no heterozygous sites to express.
    phase_switch_rate_per_het_min: float = 0.002
    phase_switch_rate_per_het_max: float = 0.020

    enable_stdpopsim: bool = False
    stdpopsim_species: str = "HomSap"
    stdpopsim_models: str = ""
    stdpopsim_contig_length: int = 0

    workers: int = 1
    compression: bool = True
    compression_level: int = 1
    # The true maps are what makes an oracle ablation possible: hand the model
    # the map the data was simulated under instead of the one it is given, and
    # the difference separates "limited by map error" from "limited by not
    # using the map".  With the observed map now genuinely wrong -- it is
    # re-binned and noised rather than returned on the true breakpoints -- that
    # question has an answer worth measuring.
    #
    # They roughly double a sample on disk, so they are written for a fraction
    # of rows rather than all of them.  An ablation needs statistical power,
    # not every row.
    # How much of the genealogy to keep alongside the sequence.
    #
    #   "none"     nothing
    #   "summary"  coalescences and branch length per target time bin, two
    #              vectors of 64 floats: 512 bytes a row, 0.08% of the
    #              simulation's runtime, and the sufficient statistic for Ne(t)
    #   "full"     the tree sequence itself, as a .trees file per row
    #
    # "full" is off by default because it is not a storage trade-off, it is a
    # wall: measured at twenty megabases and sixty-four haplotypes a .trees
    # file runs 0.8 MB at an effective size of three thousand and 27.8 MB
    # gzipped at a hundred thousand, twenty times the sequence data it came
    # from.  A hundred thousand rows of it would be six to eight terabytes.
    # It exists so a dedicated, small dataset can be generated for a genealogy
    # or TMRCA model later without re-running everything -- set
    # ``arg_detail="full"`` with a few thousand rows.
    #
    # TMRCA is not a separate setting: it is derivable from "full" and its own
    # per-tree form costs one to three megabytes a row, the same order as the
    # genotypes, for information "summary" mostly already carries.
    arg_detail: str = "summary"
    # Reproducible subset, on the same index hash as ``truth_fraction`` so it
    # neither consumes the noise stream nor depends on row order.
    arg_fraction: float = 1.0
    include_truth: bool = True
    truth_fraction: float = 0.02
    max_retries: int = 3


def apply_preset(cfg: Config) -> Config:
    if cfg.preset == "mini":
        cfg.dataset_name = "DLCoalSim-Mini"
        cfg.n_samples = 512
        cfg.shard_size = 32
        cfg.seq_len = 1_000_000
        cfg.n_haplotypes = 24
        cfg.time_bins = 64
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
        # Thirty megabases because that is what every human autosome can supply
        # from measured map: the shortest measured span of the two empirical
        # maps together is chr21's 32.0 Mb, and a slice longer than that either
        # drops the small chromosomes -- which carry 1.7x the recombination rate
        # of the large ones, so losing them biases the training landscape -- or
        # is filled with a fabricated flat rate.  Reference length is the wrong
        # yardstick: chr21 measures 46.7 Mb but its acrocentric short arm is not
        # in any map and is not callable in real data either.
        cfg.seq_len = 30_000_000
        # A hundred and twenty-eight because the information gain is measured
        # and the alternative is not a superset.  On 128 paired histories at
        # 20 Mb the oracle improves 0.1482 -> 0.1276 from 64 to 128 haplotypes,
        # 4.2 standard errors, concentrated in the recent bins where the label
        # is hardest.  256 buys a further 0.0133 for 1.9x the disk and would
        # collide with the width ladder for VRAM.  Data at 128 can be trained
        # at 64 by dropping individuals; the reverse is not possible.
        cfg.n_haplotypes = 128
        cfg.time_bins = 64
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
    if cfg.arg_detail not in {"none", "summary", "full"}:
        raise ValueError("arg_detail must be 'none', 'summary', or 'full'")
    if not 0.0 <= cfg.arg_fraction <= 1.0:
        raise ValueError("arg_fraction must lie in [0, 1]")

    positive_integer("n_samples")
    positive_integer("shard_size")
    positive_integer("compute_chunk_size")
    seq_len = positive_integer("seq_len")
    n_haplotypes = positive_integer("n_haplotypes")
    positive_integer("time_bins")

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
    # There is no bound on ``seq_len * n_haplotypes`` here.  The sparse missing
    # index addresses the (variants, haplotypes) matrix, not the (bases,
    # haplotypes) one, and variants are three orders of magnitude rarer than
    # bases -- the old check rejected 60 Mb at 64 haplotypes, which stores an
    # index reaching about one billion.  The width is chosen per row from the
    # matrix actually built and declared in ``missing_storage``.

    resolution = finite("demography_resolution_log10")
    if resolution <= 0.0:
        raise ValueError("demography_resolution_log10 must be greater than 0")
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
    if not 0.0 <= float(cfg.truth_fraction) <= 1.0:
        raise ValueError("truth_fraction must lie in [0, 1]")
    probability_range("phase_switch_rate_per_het_min", "phase_switch_rate_per_het_max")

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

    if cfg.recomb_map_unit not in {"per_bp", "cM_per_Mb"}:
        raise ValueError("recomb_map_unit must be 'per_bp' or 'cM_per_Mb'")
    if cfg.mut_map_unit not in {"per_bp", "relative", "roulette_raw"}:
        raise ValueError("mut_map_unit must be 'per_bp', 'relative', or 'roulette_raw'")
