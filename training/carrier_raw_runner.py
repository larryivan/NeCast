"""Fail-closed CUDA runner for carrier-aware all-site raw mechanisms.

The runner has three deliberately separate stages:

``smoke``
    One optimiser step for every registered arm on the exact minimum- and
    maximum-variant development rows.
``pilot``
    One seed and one epoch on a deterministic, root-balanced 128-row subset.
    It measures all-site throughput and refuses to certify a full run whose
    extrapolated runtime exceeds the declared envelope.
``one_seed``
    The exact full-budget seed-0 prerequisite.  It uses the formal rows and
    budget, but can start from the pilot/performance receipts alone.  A
    separate module seals its outputs before any remaining formal seed starts.
``formal``
    One seed, all six arms in the same order on the same typed A100.  It uses
    the complete 5,664/629 architecture fit/inner split and writes predictions
    only for the repeatedly used 840-row development evaluation set.  Formal
    seeds 1--11 require and reverify the sealed full-budget seed-0 receipt.

The sealed 1,059-row selection set cannot be addressed by this module: the
input registry and summary cache must both be development-only and are bound
row-for-row before any raw payload is opened.  The learned head is direct64;
there is no hazard, tail, survival, dual-demography, or class output.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import dataclasses
from dataclasses import dataclass
import gc
import contextlib
import hashlib
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Sequence,
)

import numpy as np

from .diploid_ladder import CHAINS, ladder_levels, level_columns
from .distributed import (
    all_reduce_gradients,
    distributed_env,
    plan_epoch,
    predict_row_cost,
)
from .data import (
    _array_sha256,
    _cache_paths as _unphased_cache_paths,
    CACHE_SCHEMA as UNPHASED_CACHE_SCHEMA,
    CACHE_SCHEMA_VERSION as UNPHASED_CACHE_SCHEMA_VERSION,
    _environment,
    EXPECTED_DEVELOPMENT_ROWS,
    EXPECTED_EVALUATION_ROWS,
    EXPECTED_SELECTION_ROWS,
    _inner_split,
    _json_atomic,
    _load_jsonl as _load_unphased_jsonl,
    _metrics,
    _positive_integer,
    _seed_everything,
    _sha256,
    _training_deployment,
)
from .carrier_aware_raw_site_experiment import (
    ControlledRawArm,
    derange_map_rates,
    RAW_ARMS,
)
from .carrier_raw_packed import (
    apply_diploid_control_torch,
    prepare_carrier_control_shifts,
    transfer_control_shifts,
    transfer_packed_and_decode,
)
from .carrier_raw_screen_experiment import (
    screen_arm_spec,
    SCREEN_ARMS,
    SCREEN_EPOCHS,
    SCREEN_EVALUATION_ROWS,
    SCREEN_FIT_ROWS,
    SCREEN_INNER_ROWS,
    SCREEN_MODEL_CONFIG,
    SCREEN_MODEL_NAME,
    SCREEN_SCHEDULER_EPOCHS,
    SCREEN_SCOPE,
    SCREEN_SEEDS,
)
from .raw_site_gate import (
    _array_digest as _raw_array_digest,
    _raw_digest,
)


RUNNER_SCHEMA = "dlcoalsim-carrier-aware-raw-runner-v3"
EXPECTED_TARGETS = 64
DEFAULT_SEED = 20260728
MAXIMUM_FORMAL_HOURS = 73.0
RUNTIME_PROJECTION_GUARD_FRACTION = 0.03
MODEL_NAME = "carrier_aware_all_site_direct64_nw_residual_v1"

# One auditable host-memory envelope.  Persistent LRUs plus working prefetch
# and transient materialisation consume at most 87.5%; the remaining 12.5%
# is deliberately unallocated process/runtime headroom.
_HOST_BUDGET_FRACTIONS = {
    "artifact": 0.67,
    "shift": 0.05,
    "segment_plan": 0.025,
    "map": 0.02,
    "map_exposure": 0.02,
    "prefetch": 0.08,
    "transient": 0.01,
}
_MAX_ALLOCATED_HOST_FRACTION = 0.90
# Per-row counts of how many sample pairs coalesced in each of the 64 target
# bins.  The quantity the target is a transform of, and the auxiliary readout's
# supervision.
PAIR_COALESCENCE_LABEL_MEMBER = "pair_coalescences_per_bin"


def _resolved_host_budget_fractions(args: Any) -> dict[str, float]:
    """The host-budget split, with opt-in per-component overrides.

    The default split was proportioned for a 12.8 kb region grid, where map
    exposure -- charged per region -- fills 92% of its two-percent share.  A
    finer grid needs a different split of the same pot.  Re-proportioning is
    provisioning, not a relaxation: every projection still has to fit its cap,
    and the ceiling on the total allocated fraction still holds.
    """

    overrides = getattr(args, "host_budget_fractions", None) or {}
    if not overrides:
        return dict(_HOST_BUDGET_FRACTIONS)
    unknown = sorted(set(overrides).difference(_HOST_BUDGET_FRACTIONS))
    if unknown:
        raise ValueError(f"unknown host budget component(s): {unknown}")
    resolved = {**_HOST_BUDGET_FRACTIONS, **overrides}
    for name, value in sorted(resolved.items()):
        if not 0.0 < float(value) < 1.0:
            raise ValueError(f"host budget fraction for {name} must lie in (0, 1)")
    return {name: float(value) for name, value in resolved.items()}
_FROZEN_DEVELOPMENT_INVENTORY = {
    "schema": "carrier-raw-development-inventory-v2",
    "raw_index_sha256": (
        "088891f4a79f5f94ba87a7b5c7d080c44dcd0cc773a31dcb579c79d84c90d3cb"
    ),
    "development_rows": 7_133,
    "unique_source_artifacts": 1_536,
    "total_variant_sites": 1_727_411_933,
    "total_missing_calls": 1_083_180_773,
    "artifact_exact_bytes": 25_229_496_360,
    "map_both_kinds_interval_units": 41_943_252,
    "map_both_kinds_bytes": 671_320_288,
    # Map exposure is charged per region, so like the segment plan below it is
    # an execution projection and not a property of the frozen rows: it changes
    # with the model's region grid.  It was frozen as an inventory invariant
    # only because the grid had never varied.  The original 12.8 kb numbers are
    # retained as a regression reference, checked exactly when a run uses that
    # same grid and not treated as a data invariant for any other.
    "reference_map_exposure_macro_width_bp": 12_800,
    "reference_map_exposure_both_kinds_region_units": 22_297_758,
    "reference_map_exposure_both_kinds_bytes": 713_528_256,
    # Segment-plan size is an execution projection, not a property of the
    # frozen rows: it changes with chunk_sites.  Retain the original
    # 65,536-site projection as a regression reference without treating it as
    # a data-inventory invariant for other execution configurations.
    # The bound is a function of chunk_sites, leaf_width_bp and macro_levels
    # alike -- the leaf ladder sets how many reduction rows a plan carries --
    # so the reference only applies when all three match what it was measured
    # at, not when the chunk size alone does.
    "reference_segment_plan_chunk_sites": 65_536,
    "reference_segment_plan_leaf_width_bp": 100,
    "reference_segment_plan_macro_levels": 7,
    "reference_segment_plan_upper_bound_bytes": 862_727_664,
    "max_transient_bytes": 235_430_540,
    "p99_transient_bytes": 129_178_460,
    "max_transient_sites": 16_727_863,
    "max_compressed_npz_size_bytes": 55_854_448,
    "source_artifact_dtypes": {
        "variant_positions_bp": "uint32",
        "genotype_packed": "uint8",
        "missing_flat_idx": "uint32",
        "obs_rec_pos": "float32",
        "obs_rec_rate": "float32",
        "obs_mut_pos": "float32",
        "obs_mut_rate": "float32",
    },
    "artifacts_audited": 1_536,
    "development_slices_audited": 7_133,
    "dtype_shape_offset_violations": 0,
    "compressed_size_is_not_memory_accounting": True,
    "selection_rows_read": 0,
    "selection_access_count": 0,
}


def _parse_csv(value: str, *, allowed: Sequence[str]) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("CSV values must be nonempty and unique")
    unknown = sorted(set(result).difference(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown values: {unknown}")
    return result


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _short_budget_only(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "directional_only", False)
        or getattr(args, "screening_only", False)
        or getattr(args, "het_gap_screening_only", False)
    )


def _expected_arms(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(SCREEN_ARMS if getattr(args, "screening_only", False) else RAW_ARMS)


def _active_model_name(args: argparse.Namespace) -> str:
    return SCREEN_MODEL_NAME if getattr(args, "screening_only", False) else MODEL_NAME


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--unphased-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--stage", choices=("smoke", "pilot", "one_seed", "formal"), required=True
    )
    parser.add_argument(
        "--directional-only",
        action="store_true",
        help=(
            "run a short ordinary-GPU pilot for exploratory direction only; "
            "the resulting receipt can never authorize one-seed/formal work"
        ),
    )
    parser.add_argument(
        "--screening-only",
        action="store_true",
        help=(
            "run the frozen small-capacity all-site Raw/map architecture screen; "
            "its receipt is descriptive and can never authorize formal work"
        ),
    )
    parser.add_argument(
        "--het-gap-screening-only",
        action="store_true",
        help=(
            "run the frozen medium-scale sparse within-diploid het-gap screen; "
            "its receipt is development-only and cannot authorize formal work"
        ),
    )
    parser.add_argument(
        "--het-gap-gate-only",
        action="store_true",
        help=(
            "run the registered 96/16/16 one-epoch V100 real-row pilot for "
            "the sparse het-gap screen"
        ),
    )
    parser.add_argument(
        "--het-gap-cache",
        help="completed immutable 7,133-row sparse het-gap sidecar cache",
    )
    parser.add_argument(
        "--het-gap-gate-report",
        help="completed typed-V100 sparse het-gap gate required by full seeds",
    )
    parser.add_argument(
        "--required-gpu-class",
        choices=("A100", "V100", "P6000"),
        default="A100",
        help="fail-closed GPU family binding recorded in PLAN/report/DONE",
    )
    parser.add_argument(
        "--minimum-gpu-memory-gib",
        type=_positive_float,
        default=30.0,
        help="minimum observed device memory required before any raw row is loaded",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--arms")
    parser.add_argument("--epochs", type=_positive_integer, default=80)
    parser.add_argument(
        "--scheduler-epochs",
        type=_positive_integer,
        help=(
            "cosine-scheduler horizon; defaults to --epochs and is frozen to 80 "
            "for the short directional experiment"
        ),
    )
    parser.add_argument(
        "--formal-epochs",
        type=_positive_integer,
        default=80,
        help="full-budget epoch count used only by the pilot runtime extrapolation",
    )
    parser.add_argument("--accumulate-rows", type=_positive_integer, default=32)
    parser.add_argument("--learning-rate", type=_positive_float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=_positive_float, default=1.0)
    parser.add_argument("--inner-validation-basis-points", type=int, default=1000)
    parser.add_argument("--pilot-rows", type=_positive_integer, default=128)
    parser.add_argument("--pilot-fit-rows", type=_positive_integer, default=96)
    parser.add_argument("--pilot-inner-rows", type=_positive_integer, default=16)
    parser.add_argument("--pilot-evaluation-rows", type=_positive_integer, default=16)
    parser.add_argument(
        "--max-formal-hours",
        type=_positive_float,
        default=MAXIMUM_FORMAL_HOURS,
    )
    parser.add_argument(
        "--max-gpu-reserved-gib",
        type=_positive_float,
        default=60.0,
    )
    parser.add_argument("--pilot-report")
    parser.add_argument(
        "--performance-gate",
        help="completed performance-gate directory required by every full run",
    )
    parser.add_argument(
        "--one-seed-receipt",
        help="sealed full-budget seed-0 receipt required by formal seeds 1--11",
    )
    parser.add_argument("--verify-source-sha256", action="store_true")
    parser.add_argument("--site-width", type=_positive_integer, default=8)
    parser.add_argument("--individual-width", type=_positive_integer, default=32)
    parser.add_argument("--carrier-width", type=_positive_integer, default=32)
    parser.add_argument("--d-model", type=_positive_integer, default=64)
    parser.add_argument("--n-heads", type=_positive_integer, default=4)
    parser.add_argument("--leaf-width-bp", type=_positive_integer, default=100)
    parser.add_argument("--macro-levels", type=_positive_integer, default=7)
    parser.add_argument(
        "--tree-tap-levels",
        default="",
        help="comma-separated macro-tree levels to publish as memory groups",
    )
    parser.add_argument(
        "--region-sfs-bins",
        type=int,
        default=0,
        help="per-region allele-frequency histogram bins; 0 disables",
    )
    parser.add_argument(
        "--carrier-scale-levels",
        default="",
        help="comma-separated region scales that keep carrier identity",
    )
    parser.add_argument(
        "--scale-film",
        action="store_true",
        help="condition the scale readout on whole-sample statistics",
    )
    parser.add_argument("--use-site-frequency", action="store_true")
    parser.add_argument("--site-ld-lags", default="")
    parser.add_argument(
        "--site-lag-span-features",
        action="store_true",
        help=(
            "add one column per configured site lag giving the physical "
            "distance to its partner.  A lag spans k variants whatever the "
            "spacing, so the same column means 108 bp on the densest row here "
            "and 47,506 bp on the sparsest -- a factor of 440 in the time it "
            "speaks to, which the model otherwise has to infer"
        ),
    )
    parser.add_argument(
        "--site-genetic-span-features",
        action="store_true",
        help=(
            "add one column per configured site lag giving the genetic "
            "distance to its partner, read off the observed recombination "
            "map.  The coalescent runs in Morgans and the hand-computed LD "
            "feature is binned in Morgans; the graph has variant-count and "
            "base-pair coordinates and neither is that one"
        ),
    )
    parser.add_argument(
        "--site-pair-identity-lags",
        default="",
        help=(
            "site lags at which to add the fraction of diploid pairs identical "
            "at both sites.  A correlation is a second moment; this counts "
            "exact joint matches, which is the quantity an IBS spectrum is "
            "built from.  One shared column per lag, empty leaves the graph "
            "bit-identical"
        ),
    )
    parser.add_argument(
        "--pair-concordance-scales-bp",
        default="",
        help=(
            "base-pair scales at which to add two long-range pairwise "
            "concordance columns.  Carrier identity survives only as far as "
            "the region encoder, which pools the thirty-two diploids to a mean "
            "and a second moment and attends inside one region, so nothing "
            "above 12.8 kb can express which individuals match which.  Empty "
            "leaves the graph bit-identical"
        ),
    )
    parser.add_argument("--pair-concordance-width", type=int, default=8)
    parser.add_argument(
        "--pair-concordance-first-order",
        action="store_true",
        help=(
            "hand over the decayed first moment beside the pair statistic.  "
            "SGD learns a monomial on a timescale set by the largest leap in "
            "its support, and an intervening lower-order term in the same "
            "variables collapses that leap, so the quadratic is not reached "
            "from nothing"
        ),
    )
    parser.add_argument(
        "--tied-row-attention-heads",
        type=int,
        default=0,
        help=(
            "heads for a region-by-region attention whose logits are summed "
            "over the diploids.  The carrier attention is the column half of "
            "axial attention and runs inside one region; this is the row half, "
            "and it is the only operation here that keeps which individual is "
            "which across two regions.  Zero leaves the graph bit-identical"
        ),
    )
    parser.add_argument(
        "--long-block-dilation-cap",
        type=int,
        default=0,
        help=(
            "cap the doubling dilation of the long blocks.  Depth buys reach "
            "exponentially and parameters linearly, so this holds the "
            "parameter count exactly while collapsing the reach -- the control "
            "that says which of the two a depth win came from.  Zero leaves "
            "the doubling untouched"
        ),
    )
    parser.add_argument("--region-downsample-mode", choices=("conv", "pool"), default="conv")
    parser.add_argument("--bin-attention-temperature", action="store_true")
    parser.add_argument("--overlapping-regions", action="store_true")
    parser.add_argument(
        "--extreme-tap-levels",
        default="",
        help=(
            "tree levels read with a soft extremum instead of a mean; empty "
            "leaves the graph bit-identical"
        ),
    )
    parser.add_argument(
        "--carrier-tree-levels",
        default="",
        help=(
            "tree levels at which the 32 diploids attend to one another; "
            "empty leaves the graph bit-identical"
        ),
    )
    parser.add_argument(
        "--site-state-width",
        type=_nonnegative_integer,
        default=0,
        help=(
            "width of the site-level selective state space; 0 disables it and "
            "leaves the graph bit-identical"
        ),
    )
    parser.add_argument("--n-long-blocks", type=_nonnegative_integer, default=11)
    parser.add_argument("--n-regional-blocks", type=_positive_integer, default=3)
    parser.add_argument("--n-region-scales", type=_positive_integer, default=4)
    parser.add_argument("--chunk-sites", type=_positive_integer, default=65_536)
    parser.add_argument("--carrier-region-chunk", type=_positive_integer, default=512)
    parser.add_argument(
        "--segment-reduction-backend",
        choices=("padded", "segment_reduce"),
        default="padded",
        help=(
            "receipt-bound contiguous reducer; segment_reduce is an optional "
            "performance path and must pass the typed-A100 equivalence gate"
        ),
    )
    parser.add_argument(
        "--diploid-subsample",
        default="",
        help=(
            "comma-separated diploid counts to draw from, one draw per row "
            "per epoch (for example 4,8,16,32,64).  Empty keeps every "
            "individual the row carries, which is the frozen behaviour.  "
            "A draw also sets the context's log2 haplotype entry, so the "
            "model is told the size it is looking at rather than having to "
            "infer it"
        ),
    )
    parser.add_argument(
        "--diploid-subsample-seed",
        type=int,
        default=20260901,
        help="seed for the per-row individual draw and canonical ordering",
    )
    parser.add_argument(
        "--expected-selection-rows",
        type=int,
        default=EXPECTED_SELECTION_ROWS,
        help=(
            "how many rows this cohort seals for final selection.  The "
            "registry plan, the registry report, and the summary cache "
            "plan must all declare the same number or the run is refused. "
            "The default is the development cohort's seal, so a "
            "development run is unchanged and a production run that does "
            "not declare its own seal fails closed"
        ),
    )
    parser.add_argument(
        "--diploid-fixed",
        type=int,
        default=0,
        help=(
            "train on this many individuals from every row, taken as a "
            "nested prefix of the row's canonical ordering.  Exclusive "
            "with --diploid-subsample.  This is the mode a single summary "
            "evidence cache is consistent with, because every pass sees "
            "the same individuals the cache was built on"
        ),
    )
    parser.add_argument(
        "--measurement-diploids",
        type=int,
        default=0,
        help=(
            "individual count for inner validation and the single-row "
            "probe.  Defaults to the fixed training count, or the "
            "ladder's largest rung"
        ),
    )
    parser.add_argument(
        "--activation-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="trade compute for memory; disabled by default on the typed 80GB A100",
    )
    parser.add_argument(
        "--host-cache-gib",
        type=_positive_float,
        default=36.0,
        help="hard byte budget shared by packed dev cache, map cache, and prefetch",
    )
    parser.add_argument(
        "--host-cache-sites",
        type=_positive_integer,
        default=2_500_000_000,
        help="hard variant-site budget for cached/prefetched packed development rows",
    )
    parser.add_argument(
        "--prefetch-rows",
        type=_positive_integer,
        default=2,
        help="bounded producer/consumer lookahead; byte/site budgets remain authoritative",
    )
    args = parser.parse_args(argv)
    mode_count = sum(
        int(value)
        for value in (
            args.directional_only,
            args.screening_only,
            args.het_gap_screening_only,
        )
    )
    if mode_count > 1:
        parser.error("directional, map-screen, and het-gap-screen modes are exclusive")
    if args.scheduler_epochs is None:
        args.scheduler_epochs = args.epochs
    expected_arms = _expected_arms(args)
    try:
        args.arms = _parse_csv(
            args.arms or ",".join(expected_arms),
            allowed=expected_arms,
        )
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.arms != list(expected_arms):
        parser.error("each stage requires its registered arms in frozen order")
    if args.weight_decay < 0 or not math.isfinite(args.weight_decay):
        parser.error("weight decay must be finite and nonnegative")
    if not 1 <= args.inner_validation_basis_points <= 9999:
        parser.error("inner validation basis points must lie in [1,9999]")
    if args.stage == "pilot" and (
        args.pilot_fit_rows + args.pilot_inner_rows + args.pilot_evaluation_rows
        != args.pilot_rows
    ):
        parser.error("pilot fit/inner/evaluation counts must sum to pilot rows")
    if args.screening_only:
        model_contract = {
            "site_width": args.site_width,
            "individual_width": args.individual_width,
            "carrier_width": args.carrier_width,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "leaf_width_bp": args.leaf_width_bp,
            "macro_levels": args.macro_levels,
            "n_long_blocks": args.n_long_blocks,
            "n_regional_blocks": args.n_regional_blocks,
            "n_region_scales": args.n_region_scales,
        }
        screening_contract = (
            args.stage == "pilot"
            and args.required_gpu_class == "V100"
            and args.minimum_gpu_memory_gib >= 30.0
            and args.epochs == SCREEN_EPOCHS
            and args.scheduler_epochs == SCREEN_SCHEDULER_EPOCHS
            and args.formal_epochs == 80
            and args.pilot_rows
            == SCREEN_FIT_ROWS + SCREEN_INNER_ROWS + SCREEN_EVALUATION_ROWS
            and args.pilot_fit_rows == SCREEN_FIT_ROWS
            and args.pilot_inner_rows == SCREEN_INNER_ROWS
            and args.pilot_evaluation_rows == SCREEN_EVALUATION_ROWS
            and args.seed in SCREEN_SEEDS
            and args.max_formal_hours == MAXIMUM_FORMAL_HOURS
            and args.max_gpu_reserved_gib <= 25.5
            and args.chunk_sites == 262_144
            and args.carrier_region_chunk == 2_048
            and args.segment_reduction_backend == "segment_reduce"
            and args.activation_checkpoint is False
            and args.host_cache_gib == 36.0
            and args.host_cache_sites == 2_500_000_000
            and args.prefetch_rows == 4
            and args.accumulate_rows == 32
            and args.learning_rate == 3.0e-4
            and args.weight_decay == 1.0e-4
            and args.gradient_clip == 1.0
            and args.inner_validation_basis_points == 1000
            and model_contract
            == {
                key: value
                for key, value in SCREEN_MODEL_CONFIG.items()
                if key
                not in {
                    "use_site_recombination_features",
                    "site_recombination_film_scale",
                }
            }
        )
        if not screening_contract:
            parser.error(
                "screening-only is frozen to pilot/V100>=30GiB, three seeds, "
                "768/96/840 rows, 6 observed epochs with scheduler horizon 80, "
                "the registered small model, and the audited all-site loader"
            )
    elif args.directional_only:
        directional_contract = (
            args.stage == "pilot"
            and args.required_gpu_class == "V100"
            and args.minimum_gpu_memory_gib >= 30.0
            and args.epochs == 2
            and args.scheduler_epochs == 80
            and args.formal_epochs == 80
            and args.pilot_rows == 1_416
            and args.pilot_fit_rows == 512
            and args.pilot_inner_rows == 64
            and args.pilot_evaluation_rows == EXPECTED_EVALUATION_ROWS
            and args.seed in range(DEFAULT_SEED, DEFAULT_SEED + 3)
            and args.max_formal_hours == MAXIMUM_FORMAL_HOURS
            and args.max_gpu_reserved_gib <= 25.5
            and args.chunk_sites == 262_144
            and args.carrier_region_chunk == 2_048
            and args.segment_reduction_backend == "segment_reduce"
            and args.activation_checkpoint is False
            and args.host_cache_gib == 36.0
            and args.host_cache_sites == 2_500_000_000
            and args.prefetch_rows == 4
        )
        if not directional_contract:
            parser.error(
                "directional-only is frozen to pilot/V100>=30GiB, seeds "
                "20260728..20260730, 512/64/840 rows, 2 observed epochs, "
                "an 80-epoch scheduler horizon, 262144/2048 segment_reduce, "
                "36GiB/2.5B-sites cache, <=25.5GiB CUDA reserve, and the "
                "73-hour formal ceiling"
            )
    elif args.required_gpu_class != "A100":
        parser.error(
            "non-A100 hardware is permitted only for a registered short pilot"
        )
    elif args.scheduler_epochs != args.epochs:
        parser.error("non-directional stages require scheduler-epochs == epochs")
    if args.stage in {"one_seed", "formal"} and (
        not args.pilot_report or not args.performance_gate
    ):
        parser.error(
            "one-seed/formal stages require completed pilot and performance-gate receipts"
        )
    if args.stage in {"one_seed", "formal"} and (
        args.epochs != 80 or args.formal_epochs != 80
    ):
        parser.error("one-seed/formal stages require the frozen 80-epoch budget")
    if (
        args.stage in {"one_seed", "formal"}
        and args.max_formal_hours > MAXIMUM_FORMAL_HOURS
    ):
        parser.error(
            "one-seed/formal runtime policy cannot exceed "
            f"{MAXIMUM_FORMAL_HOURS:g} hours"
        )
    if args.stage == "one_seed" and args.seed != DEFAULT_SEED:
        parser.error("one-seed stage is frozen to seed 20260728")
    if args.stage == "formal" and (
        not args.one_seed_receipt
        or args.seed not in range(DEFAULT_SEED + 1, DEFAULT_SEED + 12)
    ):
        parser.error(
            "formal stage requires the sealed one-seed receipt and an exact seed in 20260729..20260739"
        )
    return args


def _validate_gpu_contract(
    *,
    gpu_name: str,
    total_memory_bytes: int,
    required_gpu_class: str,
    minimum_gpu_memory_gib: float,
) -> dict[str, Any]:
    """Bind a run to one declared GPU family and a fail-closed memory floor."""

    required = str(required_gpu_class).upper()
    observed = str(gpu_name)
    minimum_bytes = int(float(minimum_gpu_memory_gib) * (1024**3))
    if required not in observed.upper():
        raise RuntimeError(
            f"typed {required_gpu_class} required, observed {observed!r}"
        )
    if int(total_memory_bytes) < minimum_bytes:
        raise RuntimeError(
            "GPU memory is below the declared floor: "
            f"observed={int(total_memory_bytes)} required={minimum_bytes}"
        )
    return {
        "required_gpu_class": required_gpu_class,
        "observed_gpu_name": observed,
        "observed_total_memory_bytes": int(total_memory_bytes),
        "minimum_gpu_memory_gib": float(minimum_gpu_memory_gib),
        "family_match": True,
        "memory_floor_passed": True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _load_registry(
    directory: Path,
    *,
    expected_selection_rows: int = EXPECTED_SELECTION_ROWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    required = {
        name: directory / name
        for name in (
            "PLAN.json",
            "report.json",
            "DONE.json",
            "raw_index.jsonl",
            "reference.npz",
            "sample_identities.jsonl",
            "selected_indices.npy",
        )
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = _read_json(required["PLAN.json"])
    report = _read_json(required["report.json"])
    done = _read_json(required["DONE.json"])
    sealed = int(expected_selection_rows)
    if sealed < 1:
        raise ValueError("a cohort must seal at least one selection row")
    # Plan, report, and caller must all name the same seal.  Checking the
    # plan here, before the report check below, means a registry built for
    # another cohort is rejected on its own declaration rather than on a
    # count that happens to match.
    if (
        int(plan.get("selection_rows_excluded_before_variant_count_access", -1))
        != sealed
    ):
        raise ValueError(
            "registry plan seals a different number of selection rows "
            "than this run declares"
        )
    declared_rows = int(plan.get("selected_samples", -1))
    if declared_rows < 1:
        raise ValueError("registry plan does not declare a cohort size")
    if (
        report.get("status") != "complete"
        or done.get("status") != "complete"
        or done.get("report_sha256") != _sha256(required["report.json"])
        or done.get("raw_index_sha256") != _sha256(required["raw_index.jsonl"])
        or int(report.get("selected_samples", -1)) != declared_rows
        or int(report.get("selection_rows_excluded_before_variant_count_access", -1))
        != sealed
        or report.get("selection_excluded_before_raw_task_construction") is not True
        or report.get("index_only") is not True
        or report.get("all_raw_roundtrips_passed") is not True
    ):
        raise ValueError("carrier raw registry is incomplete or not fail-closed")
    # The raw index builder intentionally reuses ``identities_sha256`` and
    # ``architecture_split_sha256`` in the report for the selected output
    # manifests, whereas PLAN binds the full input manifests.  Compare only
    # immutable scientific/input fields whose meaning is unchanged.
    for frozen in (
        "schema",
        "schema_version",
        "observation_schema",
        "snapshot",
        "snapshot_sha256",
        "reference_cache",
        "reference_cache_sha256",
        "selected_samples",
        "selection_rows_excluded_before_variant_count_access",
        "selection_excluded_before_raw_task_construction",
        "physical_source_root_map",
        "verify_source_sha256",
        "runtime_source_sha256",
        "scientific_semantics",
    ):
        if report.get(frozen) != plan.get(frozen):
            raise ValueError(f"registry report changed PLAN field {frozen}")
    rows = _read_jsonl(required["raw_index.jsonl"])
    identities = _read_jsonl(required["sample_identities.jsonl"])
    if len(rows) != declared_rows or len(identities) != len(rows):
        raise ValueError("registry row count differs from its declared cohort")
    for index, (row, identity) in enumerate(zip(rows, identities, strict=True)):
        if (
            int(row.get("output_index", -1)) != index
            or not str(row.get("canonical_id", ""))
            or str(row.get("canonical_id")) != str(identity.get("canonical_id"))
            or int(row.get("variant_count", -1)) <= 0
        ):
            raise ValueError("registry identity/order/variant binding differs")
    with np.load(required["reference.npz"], allow_pickle=False) as loaded:
        names = {
            "context",
            "targets",
            "seeds",
            "split",
            "architecture_split",
            "selected_indices",
        }
        missing = names.difference(loaded.files)
        if missing:
            raise ValueError(f"registry reference lacks arrays: {sorted(missing)}")
        arrays = {name: np.asarray(loaded[name]).copy() for name in names}
    scale = _raw_registry_scale(plan, arrays["architecture_split"])
    n = scale.development_rows
    if (
        arrays["context"].shape != (n, 2)
        or arrays["targets"].shape != (n, EXPECTED_TARGETS)
        or arrays["selected_indices"].shape != (n,)
        or arrays["architecture_split"].shape != (n,)
        or int(np.count_nonzero(arrays["architecture_split"] == 0))
        != scale.train_rows
        or int(np.count_nonzero(arrays["architecture_split"] == 1))
        != scale.evaluation_rows
        or np.any(arrays["architecture_split"] > 1)
        or len(np.unique(arrays["selected_indices"])) != n
        or not np.array_equal(
            arrays["selected_indices"],
            np.asarray(np.load(required["selected_indices.npy"], allow_pickle=False)),
        )
    ):
        raise ValueError("registry reference violates frozen development split")
    return rows, identities, arrays, {
        "paths": {name: str(path) for name, path in required.items()},
        "sha256": {name: _sha256(path) for name, path in required.items()},
        "plan": plan,
        "report": report,
        "selection_rows_read": 0,
    }


_UNPHASED_REQUIRED_MEMBERS = frozenset(
    {
        "features",
        "coordinates",
        "context",
        "targets",
        "seeds",
        "split",
        "architecture_split",
        "selected_indices",
        "nw_proxy",
        "nw_log10",
        "folded_sfs_counts",
        "ld_features",
        "ld_pair_counts",
        "ld_effective_individual_sums",
        "ld_bin_edges_morgan",
        "ld_lags",
        "ibs_counts",
        "ibs_callable_pair_site_counts_leaf",
        "ibs_bin_edges_bp",
        "variant_counts",
        "missing_call_counts",
    }
)
# Written by the per-(chain, level) ladder builder and by nothing else.
LADDER_COLUMNS_MEMBER = "ladder_columns"

_RAW_RUNNER_CACHE_MEMBERS = (
    "context",
    "targets",
    "seeds",
    "split",
    "architecture_split",
    "selected_indices",
    "nw_log10",
    "variant_counts",
)


def _load_unphased_cache_slim(
    value: str | Path,
    *,
    expected_selection_rows: int = EXPECTED_SELECTION_ROWS,
) -> tuple[
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
]:
    """Validate the complete receipt but materialise only Raw-runner members."""

    paths = _unphased_cache_paths(value)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = _read_json(paths["plan"])
    report = _read_json(paths["report"])
    done = _read_json(paths["done"])
    for label, record in (("PLAN", plan), ("report", report), ("DONE", done)):
        if (
            record.get("schema") != UNPHASED_CACHE_SCHEMA
            or int(record.get("schema_version", -1))
            != UNPHASED_CACHE_SCHEMA_VERSION
        ):
            raise ValueError(f"unphased cache {label} schema differs")
    if report.get("status") != "complete" or done.get("status") != "complete":
        raise ValueError("unphased cache is incomplete")
    for key, frozen in plan.items():
        if key != "status" and report.get(key) != frozen:
            raise ValueError(f"cache report changed PLAN field {key}")

    identity_audit = report.get("identity_audit")
    cache_scale = _registry_scale(plan)
    planned_counts = plan.get("architecture_split_counts", {})
    planned_n = int(plan.get("selected_samples", -1))
    # This exclusion proof is checked before opening the row-bearing NPZ.
    if (
        not isinstance(identity_audit, dict)
        or identity_audit.get(
            "selection_raw_rows_not_indexed_and_features_not_constructed"
        )
        is not True
        or int(plan.get("sealed_selection_samples", -1))
        != int(expected_selection_rows)
        or int(plan.get("development_eligible_samples", -1))
        != cache_scale.development_rows
        or planned_n != cache_scale.development_rows
        or not isinstance(planned_counts, dict)
        or int(planned_counts.get("2", -1)) != 0
        or int(planned_counts.get("0", -1)) != cache_scale.train_rows
        or int(planned_counts.get("1", -1)) != cache_scale.evaluation_rows
        or plan.get("gate") is not None
    ):
        raise ValueError("cache does not prove pre-raw-access selection exclusion")

    hashes = {name: _sha256(path) for name, path in paths.items()}
    for receipt_key, path_key in (
        ("cache_sha256", "cache"),
        ("report_sha256", "report"),
        ("selected_indices_sha256", "selected_indices"),
        ("identities_sha256", "identities"),
        ("architecture_split_sha256", "architecture_split"),
        ("snapshot_sha256", "snapshot"),
    ):
        if str(done.get(receipt_key, "")) != hashes[path_key]:
            raise ValueError(f"{receipt_key} does not bind {paths[path_key]}")

    with np.load(paths["cache"], allow_pickle=False) as loaded:
        missing = _UNPHASED_REQUIRED_MEMBERS.difference(loaded.files)
        if missing:
            raise ValueError(f"unphased cache lacks arrays: {sorted(missing)}")
        arrays = {
            name: np.asarray(loaded[name]).copy()
            for name in _RAW_RUNNER_CACHE_MEMBERS
        }
        # Only the second generator lineage records how many sample pairs
        # coalesced in each target bin, so this member is optional and is
        # carried when present.  It is a label, never an input: nothing reads
        # it except the training-only auxiliary readout.
        if PAIR_COALESCENCE_LABEL_MEMBER in loaded.files:
            arrays[PAIR_COALESCENCE_LABEL_MEMBER] = np.asarray(
                loaded[PAIR_COALESCENCE_LABEL_MEMBER]
            ).copy()
        # Only a per-(chain, level) ladder cache records which diploids each of
        # its rows describes, so this is optional too.  It is the ladder's
        # source of truth: with it the raw path reads the subset the summaries
        # were computed from, instead of re-deriving it and having to agree
        # about the identity string to seed with.
        if LADDER_COLUMNS_MEMBER in loaded.files:
            columns = np.asarray(loaded[LADDER_COLUMNS_MEMBER]).copy()
            if (
                columns.ndim != 2
                or columns.shape[0] != planned_n
                or columns.dtype != np.int64
                or columns.shape[1] < 1
                or columns.shape[1] & (columns.shape[1] - 1)
            ):
                raise ValueError(
                    f"{LADDER_COLUMNS_MEMBER} must be int64 of shape "
                    f"({planned_n}, power-of-two); got {columns.shape} "
                    f"{columns.dtype}"
                )
            arrays[LADDER_COLUMNS_MEMBER] = columns
        available_members = tuple(sorted(loaded.files))
    n = planned_n
    shape_contract = {
        "context": (n, 2),
        "targets": (n, EXPECTED_TARGETS),
        "seeds": (n,),
        "split": (n,),
        "architecture_split": (n,),
        "selected_indices": (n,),
        "nw_log10": (n,),
        "variant_counts": (n,),
    }
    dtype_contract = {
        "context": np.float32,
        "targets": np.float32,
        "seeds": np.int64,
        "split": np.uint8,
        "architecture_split": np.uint8,
        "selected_indices": np.int64,
        "nw_log10": np.float32,
        "variant_counts": np.uint32,
    }
    for name in _RAW_RUNNER_CACHE_MEMBERS:
        if arrays[name].shape != shape_contract[name]:
            raise ValueError(f"{name} shape differs from slim Raw contract")
        if arrays[name].dtype != dtype_contract[name]:
            raise ValueError(f"{name} dtype differs from slim Raw contract")
    label = arrays.get(PAIR_COALESCENCE_LABEL_MEMBER)
    if label is not None:
        if label.shape != (n, EXPECTED_TARGETS) or label.dtype != np.float64:
            raise ValueError(
                f"{PAIR_COALESCENCE_LABEL_MEMBER} shape or dtype differs from "
                "the unphased cache contract"
            )
        if not np.all(np.isfinite(label)) or np.any(label < 0.0):
            raise ValueError(
                f"{PAIR_COALESCENCE_LABEL_MEMBER} is not a finite non-negative count"
            )
    if (
        not np.isfinite(arrays["context"]).all()
        or not np.isfinite(arrays["targets"]).all()
        or not np.isfinite(arrays["nw_log10"]).all()
        or np.any(arrays["variant_counts"] == 0)
    ):
        raise ValueError("slim Raw cache members contain invalid values")
    architecture = arrays["architecture_split"]
    if (
        np.any((architecture != 0) & (architecture != 1))
        or int(np.count_nonzero(architecture == 0)) != cache_scale.train_rows
        or int(np.count_nonzero(architecture == 1)) != cache_scale.evaluation_rows
        or len(np.unique(arrays["selected_indices"])) != n
    ):
        raise ValueError("development-only architecture split differs")
    selected_file = np.asarray(
        np.load(paths["selected_indices"], allow_pickle=False)
    )
    if not np.array_equal(selected_file, arrays["selected_indices"]):
        raise ValueError("selected_indices.npy differs from cache")
    identities = _load_unphased_jsonl(paths["identities"])
    split_records = _load_unphased_jsonl(paths["architecture_split"])
    if len(identities) != n or len(split_records) != n:
        raise ValueError("identity/split manifest row count differs")
    groups: list[str] = []
    for index, (identity, split_record) in enumerate(
        zip(identities, split_records, strict=True)
    ):
        expected_label = "train" if architecture[index] == 0 else "validation"
        if (
            not str(identity.get("canonical_id", ""))
            or str(identity.get("canonical_id"))
            != str(split_record.get("canonical_id", ""))
            or int(identity.get("seed", -1)) != int(arrays["seeds"][index])
            or int(split_record.get("seed", -1)) != int(arrays["seeds"][index])
            or str(split_record.get("split", "")) != expected_label
            or not str(split_record.get("architecture_group", ""))
        ):
            raise ValueError("identity/split/slim-cache row binding differs")
        groups.append(str(split_record["architecture_group"]))
    _release_page_cache(paths["cache"])
    materialized_bytes = _arrays_nbytes(arrays.values())
    provenance = {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "plan": plan,
        "report": report,
        "selection_rows_read": 0,
        "selection_access_count": 0,
        "materialization": {
            "strategy": "receipt_validated_slim_npz_members_v1",
            "available_member_names": list(available_members),
            "materialized_member_names": list(_RAW_RUNNER_CACHE_MEMBERS),
            "materialized_bytes": materialized_bytes,
            "unmaterialized_member_names": sorted(
                set(available_members).difference(_RAW_RUNNER_CACHE_MEMBERS)
            ),
            "selection_row_specific_members_materialized": 0,
        },
    }
    return arrays, identities, groups, provenance


def _release_page_cache(path: Path) -> None:
    """Ask the kernel to drop a just-read file's pages from the page cache.

    The corpus is a single-pass stream whose working set exceeds what this
    container can cache.  Pinned ladder members and anonymous training state are
    unreclaimable, so from the 12,500-row rung upward an epoch reads more than
    the file cache that is left and those pages are cycled out again before the
    next epoch can use them.  Retaining them costs more than it buys: the kernel
    must then reclaim synchronously on every later allocation, which measured as
    ``memory.pressure`` full avg10=93%, staging ranks at 100% kernel and 0% user
    time, and an effective read rate of 24 MB/s against 274 MB/s on the raw
    device.

    An I/O hint only: the bytes read are the same bytes, and every array is
    materialised before this runs, so no value downstream can change.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        with open(path, "rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass



def _bind_registry_to_cache(
    rows: Sequence[dict[str, Any]],
    identities: Sequence[dict[str, Any]],
    reference: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    cache_identities: Sequence[dict[str, Any]],
    *,
    expected_rows: int,
    sample_size_varies: bool = False,
) -> np.ndarray:
    """Map each registry row onto its cache row, checking the two agree.

    ``sample_size_varies`` relaxes exactly one comparison and only that one.
    The context's second channel is log2 of the haplotype count, so a cache
    built from thirty-two diploids records six where the registry, describing
    the row at its full width, records seven.  Both are right; that difference
    is what a ladder is.  Everything else -- reference index, selected index,
    target, seed, both splits -- must still match exactly, because those say
    which row this is, and a ladder changes how much of a row is read, never
    which one.
    """
    lookup = {
        str(identity.get("canonical_id", "")): index
        for index, identity in enumerate(cache_identities)
    }
    if len(lookup) != int(expected_rows) or "" in lookup:
        raise ValueError("unphased cache identities are missing or duplicated")
    mapped = np.empty(len(rows), dtype=np.int64)
    for index, (row, identity) in enumerate(zip(rows, identities, strict=True)):
        canonical = str(row["canonical_id"])
        local = lookup.get(canonical)
        if (
            local is None
            or canonical != str(identity.get("canonical_id", ""))
            or int(row.get("reference_index", -1))
            != int(reference["selected_indices"][index])
            or int(cache["selected_indices"][local])
            != int(reference["selected_indices"][index])
            or int(row.get("variant_count", -1)) != int(cache["variant_counts"][local])
            or not np.array_equal(
                reference["context"][index][: 1 if sample_size_varies else None],
                cache["context"][local][: 1 if sample_size_varies else None],
            )
            or not np.array_equal(reference["targets"][index], cache["targets"][local])
            or int(reference["seeds"][index]) != int(cache["seeds"][local])
            or int(reference["split"][index]) != int(cache["split"][local])
            or int(reference["architecture_split"][index])
            != int(cache["architecture_split"][local])
        ):
            raise ValueError("raw registry row differs from unphased cache")
        mapped[index] = local
    if len(np.unique(mapped)) != len(mapped):
        raise ValueError("raw registry maps duplicate cache rows")
    return mapped


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class _RegistryScale:
    """How many rows the bound registry declares, by split label.

    The sealed selection count is deliberately not a member.  It is
    declared by the run and cross-checked against the registry plan,
    the registry report, and the summary cache plan, so a cohort cannot
    shrink its own seal by describing itself differently.  Keeping it
    out of the scale keeps those two questions separate: how large the
    cohort is, and how much of it was sealed away.
    """

    train_rows: int
    evaluation_rows: int

    @property
    def development_rows(self) -> int:
        return self.train_rows + self.evaluation_rows


def _registry_scale(plan: dict[str, Any]) -> _RegistryScale:
    """Read a registry's declared cohort size out of its own plan.

    Every count the plan states must agree with every other one before any
    of them is used, so a plan that disagrees with itself fails here rather
    than somewhere downstream where the mismatch would be silent.
    """

    counts = plan.get("architecture_split_counts")
    if not isinstance(counts, dict):
        raise ValueError("registry plan lacks architecture split counts")
    train = int(counts.get("0", -1))
    evaluation = int(counts.get("1", -1))
    if train < 1 or evaluation < 1 or int(counts.get("2", -1)) != 0:
        raise ValueError("registry plan declares an unusable split")
    scale = _RegistryScale(train_rows=train, evaluation_rows=evaluation)
    declared = {
        int(plan.get("selected_samples", -1)),
        int(plan.get("development_eligible_samples", -1)),
    }
    if declared != {scale.development_rows}:
        raise ValueError("registry plan disagrees with itself on cohort size")
    return scale


def _raw_registry_scale(
    plan: dict[str, Any],
    architecture_split: np.ndarray,
) -> _RegistryScale:
    """Read the raw registry's cohort from its plan and its split array.

    The plan states the total only; the per-label counts are a property of
    the reference array, so the two are cross-checked against each other
    before either is trusted.
    """

    declared = int(plan.get("selected_samples", -1))
    if declared < 1:
        raise ValueError("registry plan does not declare a cohort size")
    labels = np.asarray(architecture_split)
    if labels.ndim != 1 or np.any((labels != 0) & (labels != 1)):
        raise ValueError("architecture split carries a label outside {0,1}")
    scale = _RegistryScale(
        train_rows=int(np.count_nonzero(labels == 0)),
        evaluation_rows=int(np.count_nonzero(labels == 1)),
    )
    if scale.train_rows < 1 or scale.evaluation_rows < 1:
        raise ValueError("architecture split leaves a side empty")
    if scale.development_rows != declared or len(labels) != declared:
        raise ValueError("architecture split disagrees with the declared cohort")
    return scale


def _measurement_diploids(args: argparse.Namespace) -> int:
    """The individual count every measurement in a run is taken at.

    Inner validation picks checkpoints and ranks arms, so it has to sit at
    one count for the whole run: a count that moved between passes would
    put the draw's variance straight into the selection metric.  Explicit
    when given, otherwise the fixed training count, otherwise the ladder's
    largest rung -- never the row's own width, which is what the frozen
    configuration cannot fit.
    """

    explicit = getattr(args, "measurement_diploids", 0)
    if explicit:
        return int(explicit)
    fixed = getattr(args, "diploid_fixed", 0)
    if fixed:
        return int(fixed)
    ladder = _parse_levels(getattr(args, "diploid_subsample", ""))
    if ladder:
        return int(max(ladder))
    # A nested-ladder run names its rungs by the caches it loads, not by
    # --diploid-subsample, so the largest loaded rung is the count. Without
    # this the whole family raised here -- after a full production epoch of
    # fitting, because the first measurement is the inner validation that
    # follows it.
    nested = _parse_levels(
        ",".join(
            item.partition(":")[0]
            for item in str(getattr(args, "unphased_cache_ladder", "") or "").split(",")
            if item.strip()
        )
    )
    if nested:
        return int(max(nested))
    raise ValueError(
        "measurements need a diploid count: pass --measurement-diploids, "
        "--diploid-fixed, --diploid-subsample, or --unphased-cache-ladder"
    )


def _parse_levels(raw: str | None) -> list[int]:
    """Return a JSON-round-trippable level list.

    ``model_config`` is written to ``report.json`` and later compared against a
    freshly parsed copy.  A tuple would come back as a list and fail that exact
    equality, so the list is the canonical form here.
    """

    if raw is None or not str(raw).strip():
        return []
    return [int(item) for item in str(raw).split(",") if item.strip()]


def _model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    result = {
        "context_dim": 2,
        "site_width": args.site_width,
        "individual_width": args.individual_width,
        "carrier_width": args.carrier_width,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "leaf_width_bp": args.leaf_width_bp,
        "macro_levels": args.macro_levels,
        "n_long_blocks": args.n_long_blocks,
        "n_regional_blocks": args.n_regional_blocks,
        "n_region_scales": args.n_region_scales,
        "chunk_sites": args.chunk_sites,
        "carrier_region_chunk": args.carrier_region_chunk,
        "use_checkpoint": args.activation_checkpoint,
        "segment_reduction_backend": args.segment_reduction_backend,
        "dropout": 0.0,
        # Scale-addressability options.  All default to the untapped,
        # unmodulated model, whose graph and initialization are unchanged.
        "tree_tap_levels": _parse_levels(args.tree_tap_levels),
        "smooth_time_query_modes": getattr(args, "smooth_time_query_modes", 0),
        "carrier_pool_seeds": getattr(args, "carrier_pool_seeds", 0),
        "ordered_leaf_scan_width": getattr(args, "ordered_leaf_scan_width", 0),
        # Training-only readout on the same latents the reported head reads.
        # Zero weight leaves the graph and its initialization untouched.
        "coalescence_auxiliary": bool(
            float(getattr(args, "coalescence_auxiliary_weight", 0.0)) > 0.0
        ),
        "region_sfs_bins": args.region_sfs_bins,
        "carrier_scale_levels": _parse_levels(args.carrier_scale_levels),
        "scale_film": args.scale_film,
        "use_site_frequency": args.use_site_frequency,
        "site_ld_lags": _parse_levels(args.site_ld_lags),
        "site_pair_identity_lags": _parse_levels(args.site_pair_identity_lags),
        "site_lag_span_features": args.site_lag_span_features,
        "site_genetic_span_features": args.site_genetic_span_features,
        "region_downsample_mode": args.region_downsample_mode,
        "bin_attention_temperature": args.bin_attention_temperature,
        "overlapping_regions": args.overlapping_regions,
        "site_state_width": args.site_state_width,
        "carrier_tree_levels": _parse_levels(args.carrier_tree_levels),
        "extreme_tap_levels": _parse_levels(args.extreme_tap_levels),
        "pair_concordance_scales_bp": _parse_levels(args.pair_concordance_scales_bp),
        "pair_concordance_width": args.pair_concordance_width,
        "pair_concordance_first_order": args.pair_concordance_first_order,
        "long_block_dilation_cap": args.long_block_dilation_cap,
        "tied_row_attention_heads": args.tied_row_attention_heads,
    }
    return result


def _loader_config(args: argparse.Namespace) -> dict[str, Any]:
    host_bytes = int(args.host_cache_gib * (1024**3))
    fractions = _resolved_host_budget_fractions(args)
    component_bytes = {
        name: int(host_bytes * fraction) for name, fraction in fractions.items()
    }
    allocated_bytes = sum(component_bytes.values())
    configured_allocated_fraction = sum(fractions.values())
    allocated_fraction = allocated_bytes / host_bytes
    unallocated_headroom_bytes = host_bytes - allocated_bytes
    unallocated_headroom_fraction = unallocated_headroom_bytes / host_bytes
    if allocated_fraction > _MAX_ALLOCATED_HOST_FRACTION:
        raise RuntimeError("configured persistent/working host budgets exceed 90%")
    return {
        **(
            {
                "host_budget_fraction_overrides": dict(
                    sorted(getattr(args, "host_budget_fractions", None).items())
                )
            }
            if getattr(args, "host_budget_fractions", None)
            else {}
        ),
        "host_cache_bytes": host_bytes,
        "host_cache_sites": int(args.host_cache_sites),
        "prefetch_rows": int(args.prefetch_rows),
        "stage_workers": int(getattr(args, "stage_workers", 1)),
        "artifact_cache_fraction": fractions["artifact"],
        "carrier_shift_cache_fraction": fractions["shift"],
        "segment_plan_cache_fraction": fractions["segment_plan"],
        "map_cache_fraction": fractions["map"],
        "map_exposure_cache_fraction": fractions["map_exposure"],
        "two_kind_map_cache_site_multiplier": 2,
        "pinned_prefetch_fraction": fractions["prefetch"],
        "transient_artifact_fraction": fractions["transient"],
        "allocated_fraction": allocated_fraction,
        "allocated_bytes": allocated_bytes,
        "allocated_component_bytes": component_bytes,
        "configured_allocated_fraction": configured_allocated_fraction,
        "maximum_allocated_fraction": _MAX_ALLOCATED_HOST_FRACTION,
        "unallocated_safety_fraction": unallocated_headroom_fraction,
        "unallocated_headroom_fraction": unallocated_headroom_fraction,
        "unallocated_headroom_bytes": unallocated_headroom_bytes,
        "h2d": "pinned_non_blocking",
        "positions_h2d_dtype": "uint32_then_device_int64",
        "raw_roundtrip": "registry_all_rows_proof_plus_runtime_raw_digest",
        "verify_source_sha256": bool(args.verify_source_sha256),
        "row_order": "unchanged_seed_plus_epoch_permutation",
        "selection_rows_cached": 0,
    }


def _root_key(row: dict[str, Any]) -> str:
    value = str(row.get("logical_dataset_root") or row.get("physical_dataset_root") or "")
    if not value:
        raise ValueError("registry row lacks dataset root")
    return value


_RAW_ARTIFACT_ARRAYS = (
    "variant_positions_bp",
    "genotype_packed",
    "missing_flat_idx",
    "obs_rec_pos",
    "obs_rec_rate",
    "obs_mut_pos",
    "obs_mut_rate",
)


def _artifact_key(row: dict[str, Any]) -> str:
    artifact = row.get("source_artifact")
    if not isinstance(artifact, dict) or not artifact.get("npz_path"):
        raise ValueError("raw index row lacks a source NPZ artifact")
    return str(Path(str(artifact["npz_path"])).resolve())


def _arrays_nbytes(values: Iterable[np.ndarray]) -> int:
    return sum(int(np.asarray(value).nbytes) for value in values)


def _strict_sha256(value: Any, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} is not a lowercase SHA256 digest")
    return digest


def _json_contract_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_packed_row_values(
    positions: np.ndarray,
    packed: np.ndarray,
    missing: np.ndarray,
    *,
    sequence_length_bp: int,
) -> None:
    """Strict CPU value validation for the trusted CUDA fast path.

    This is the sole boundary that authorizes ``values_prevalidated=True``.
    It is deliberately executed before a development slice enters any cache;
    subsequent uses compare immutable receipt/digest metadata in O(1).
    """

    positions = np.asarray(positions)
    packed = np.asarray(packed)
    missing = np.asarray(missing)
    if (
        positions.dtype != np.uint32
        or positions.ndim != 1
        or not positions.flags.c_contiguous
        or len(positions) == 0
    ):
        raise ValueError("positions must be nonempty contiguous uint32[V]")
    if (
        not isinstance(sequence_length_bp, (int, np.integer))
        or int(sequence_length_bp) <= 0
        or np.any(positions > np.uint64(int(sequence_length_bp)))
        or np.any(positions[1:] < positions[:-1])
    ):
        raise ValueError("positions are out of range or not nondecreasing")
    if (
        packed.dtype != np.uint8
        or packed.ndim != 2
        or packed.shape[0] != len(positions)
        or int(packed.shape[1]) < 1
        or not packed.flags.c_contiguous
    ):
        raise ValueError(
            "genotype_packed must be contiguous uint8[V,B] with B >= 1"
        )
    # Four diploids ride in every packed byte, two bits each.
    n_haplotype_calls = 8 * int(packed.shape[1])
    if (
        missing.dtype != np.uint32
        or missing.ndim != 1
        or not missing.flags.c_contiguous
    ):
        raise ValueError("missing_flat_idx must be contiguous uint32[M]")
    if len(missing) and (
        int(missing[-1]) >= len(positions) * n_haplotype_calls
        or np.any(missing[1:] <= missing[:-1])
    ):
        raise ValueError(
            "missing_flat_idx must be strictly increasing and inside "
            "[0,V*haplotypes)"
        )


@dataclass(frozen=True)
class _PackedRawRow:
    positions_bp: np.ndarray
    genotype_packed: np.ndarray
    missing_flat_idx: np.ndarray
    recombination_positions: np.ndarray
    recombination_rates: np.ndarray
    mutation_positions: np.ndarray
    mutation_rates: np.ndarray
    sequence_length_bp: int
    canonical_id: str
    registry_index: int = -1
    raw_observation_sha256: str = ""
    positions_sha256: str = ""
    receipt_sha256: str = ""
    values_prevalidated: bool = False


@dataclass(frozen=True)
class _DecodedRawRow:
    sample: _PackedRawRow
    local: int
    audit: dict[str, Any]
    nbytes: int
    sites: int


@dataclass(frozen=True)
class _CachedMaps:
    recombination_positions: np.ndarray
    recombination_rates: np.ndarray
    mutation_positions: np.ndarray
    mutation_rates: np.ndarray
    recombination_shift: int
    mutation_shift: int

    @property
    def nbytes(self) -> int:
        return _arrays_nbytes(
            (
                self.recombination_positions,
                self.recombination_rates,
                self.mutation_positions,
                self.mutation_rates,
            )
        )

    @property
    def interval_units(self) -> int:
        """Exact number of piecewise-constant map intervals cached."""

        return int(len(self.recombination_rates) + len(self.mutation_rates))


@dataclass(frozen=True)
class _CachedSegmentPlan:
    """Compact cache record decoupled from the model's plan representation."""

    plan: Any
    registry_index: int
    raw_positions_sha256: str
    plan_positions_sha256: str
    raw_observation_sha256: str
    source_receipt_sha256: str
    sequence_length_bp: int
    chunk_sites: int
    leaf_width_bp: int
    macro_levels: int
    nbytes: int
    sites: int


@dataclass(frozen=True)
class _StagedRawRow:
    positions: Any
    genotype_packed: Any
    missing_flat_idx: Any
    control_shifts: Any | None
    state_control_arm: str | None
    context: Any
    target: Any
    nw_log10: Any
    map_exposure: Any
    segment_plan: _CachedSegmentPlan
    positions_sha256: str
    raw_observation_sha256: str
    receipt_sha256: str
    values_prevalidated: bool
    sequence_length_bp: int
    long_residual_mode: str
    audit: dict[str, Any]
    nbytes: int
    sites: int
    recombination_positions: Any | None = None
    recombination_rates: Any | None = None
    recombination_prefix_mass: Any | None = None
    site_recombination_mode: str = "disabled"
    site_recombination_fusion_mode: str = "additive"
    map_feature_mode: str = "observed"
    raw_signal_mode: str = "real"
    physics_bias_mode: str = "real"
    diploid_columns: Any | None = None


class _SizedLRU:
    """Small exact LRU with independent byte and variant-site ceilings."""

    def __init__(self, *, max_bytes: int, max_sites: int, label: str) -> None:
        if max_bytes <= 0 or max_sites <= 0:
            raise ValueError("cache byte/site budgets must be positive")
        self.max_bytes = int(max_bytes)
        self.max_sites = int(max_sites)
        self.label = label
        self._values: OrderedDict[Any, tuple[Any, int, int]] = OrderedDict()
        self.bytes = 0
        self.sites = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejected = 0
        # get and put are each several steps -- look up, move to end, adjust
        # the byte and site totals, evict in a loop -- and the loader now
        # stages rows from more than one thread. The GIL makes each step
        # atomic and none of the sequences, so an unlocked put racing another
        # could evict twice for one shortfall or leave the totals disagreeing
        # with the entries, and the terminal audit checks exactly that.
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            record = self._values.get(key)
            if record is None:
                self.misses += 1
                return None
            self._values.move_to_end(key)
            self.hits += 1
            return record[0]

    def put(self, key: Any, value: Any, *, nbytes: int, sites: int) -> bool:
        nbytes = int(nbytes)
        sites = int(sites)
        if nbytes < 0 or sites < 0:
            raise ValueError("cache accounting cannot be negative")
        with self._lock:
            return self._put_locked(key, value, nbytes=nbytes, sites=sites)

    def _put_locked(self, key: Any, value: Any, *, nbytes: int, sites: int) -> bool:
        if nbytes > self.max_bytes or sites > self.max_sites:
            self.rejected += 1
            return False
        old = self._values.pop(key, None)
        if old is not None:
            self.bytes -= old[1]
            self.sites -= old[2]
        while self._values and (
            self.bytes + nbytes > self.max_bytes
            or self.sites + sites > self.max_sites
        ):
            _, (_, removed_bytes, removed_sites) = self._values.popitem(last=False)
            self.bytes -= removed_bytes
            self.sites -= removed_sites
            self.evictions += 1
        self._values[key] = (value, nbytes, sites)
        self.bytes += nbytes
        self.sites += sites
        if self.bytes > self.max_bytes or self.sites > self.max_sites:
            raise RuntimeError("cache exceeded its byte/site budget")
        return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return self._stats_locked()

    def _stats_locked(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "entries": len(self._values),
            "bytes": self.bytes,
            "sites": self.sites,
            "max_bytes": self.max_bytes,
            "max_sites": self.max_sites,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "rejected_oversize": self.rejected,
        }


class _PrefetchBudget:
    """Condition-variable budget held until a staged row has been consumed."""

    def __init__(self, *, max_bytes: int, max_sites: int) -> None:
        if max_bytes <= 0 or max_sites <= 0:
            raise ValueError("prefetch byte/site budgets must be positive")
        self.max_bytes = int(max_bytes)
        self.max_sites = int(max_sites)
        self.bytes = 0
        self.sites = 0
        self.peak_bytes = 0
        self.peak_sites = 0
        self._condition = threading.Condition()

    def acquire(self, nbytes: int, sites: int, *, cancel: Any | None = None) -> bool:
        """Charge a row; block until it fits, or return False if ``cancel`` is set.

        A producer that is winding down must not wait forever for budget
        that the consumer, already gone, will never release -- that is a
        join that never returns. With ``cancel`` the wait re-checks the event
        each tick and gives up; the caller then releases nothing, because
        nothing was charged.
        """

        nbytes, sites = int(nbytes), int(sites)
        if nbytes > self.max_bytes or sites > self.max_sites:
            raise RuntimeError(
                "one raw row exceeds the configured prefetch byte/site budget; "
                "increase --host-cache-gib/--host-cache-sites without thinning data"
            )
        fits = lambda: self.bytes + nbytes <= self.max_bytes and self.sites + sites <= self.max_sites  # noqa: E731
        with self._condition:
            if cancel is None:
                self._condition.wait_for(fits)
            else:
                while not fits():
                    if cancel.is_set():
                        return False
                    self._condition.wait(timeout=0.1)
            self.bytes += nbytes
            self.sites += sites
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.peak_sites = max(self.peak_sites, self.sites)
        return True

    def release(self, nbytes: int, sites: int) -> None:
        with self._condition:
            self.bytes -= int(nbytes)
            self.sites -= int(sites)
            if self.bytes < 0 or self.sites < 0:
                raise RuntimeError("prefetch accounting underflow")
            self._condition.notify_all()


def rung_site_mask(
    genotype_packed: np.ndarray,
    missing_flat_idx: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    """Sites that are polymorphic among the kept diploids' called haplotypes.

    A row's site list is every site segregating in the whole simulated panel.
    A rung of L diploids would never have seen most of them: a site where its
    haplotypes are all reference or all derived is not a variant of that
    sample, and its presence in the list -- its very position -- tells the
    model how many sites the panel of 64 carried, which is a Watterson
    estimate over diploids the rung must not know about.  The definition is
    the one the summary side uses for its folded spectrum and its Watterson
    anchor (``0 < derived < called``, counting a half-called diploid's one
    observed allele).  The pairwise streams read a strict subset of it --
    fully called diploids only, since a half-called diploid forms no pair --
    so a rung's LD and IBS describe 0.2-1.1% fewer sites than its spectrum,
    its anchor and this path.  See ``rung_polymorphic_sites`` in
    ``scripts/build_level_cache.py``.
    """

    packed = np.asarray(genotype_packed, dtype=np.uint8)
    if packed.ndim != 2:
        raise ValueError("genotype_packed must be (sites, bytes)")
    selected = np.asarray(columns, dtype=np.int64)
    width = 8 * int(packed.shape[1])
    if selected.ndim != 1 or selected.size == 0 or int(selected.max()) >= width // 2:
        raise ValueError("columns must name diploids inside the packing")
    haplotypes = np.empty(2 * selected.size, dtype=np.int64)
    haplotypes[0::2] = 2 * selected
    haplotypes[1::2] = 2 * selected + 1
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, haplotypes]
    called = np.ones(bits.shape, dtype=bool)
    flat = np.asarray(missing_flat_idx, dtype=np.int64)
    if flat.size:
        site, haplotype = np.divmod(flat, width)
        slot = np.full(width, -1, dtype=np.int64)
        slot[haplotypes] = np.arange(haplotypes.size, dtype=np.int64)
        kept = slot[haplotype]
        inside = kept >= 0
        called[site[inside], kept[inside]] = False
    derived = (bits.astype(np.int32) * called).sum(axis=1)
    n_called = called.sum(axis=1)
    return (derived > 0) & (derived < n_called)


def rung_site_subset(sample: _PackedRawRow, columns: np.ndarray) -> _PackedRawRow:
    """The row as a rung would have observed it: its own polymorphic sites only.

    Keeps the packing's full width (the raw path selects columns after the
    GPU decode) and drops rows of it; the flat missing index is re-expressed
    over the surviving sites, which preserves its order because the encoding
    is site-major.  The positions digest is recomputed so that every derived
    cache -- above all the segment plan, which is a function of the positions
    -- is keyed by the rung's site list and not the panel's.
    """

    keep = rung_site_mask(sample.genotype_packed, sample.missing_flat_idx, columns)
    if keep.shape != (len(sample.positions_bp),):
        raise ValueError("site mask does not cover the row")
    positions = np.ascontiguousarray(sample.positions_bp[keep])
    packed = np.ascontiguousarray(sample.genotype_packed[keep])
    flat = np.asarray(sample.missing_flat_idx, dtype=np.int64)
    if flat.size:
        width = 8 * int(sample.genotype_packed.shape[1])
        site, haplotype = np.divmod(flat, width)
        new_site = np.cumsum(keep, dtype=np.int64) - 1
        survives = keep[site]
        remapped = new_site[site[survives]] * width + haplotype[survives]
        missing = np.ascontiguousarray(
            remapped.astype(sample.missing_flat_idx.dtype, copy=False)
        )
    else:
        missing = np.ascontiguousarray(sample.missing_flat_idx)
    return dataclasses.replace(
        sample,
        positions_bp=positions,
        genotype_packed=packed,
        missing_flat_idx=missing,
        positions_sha256=_raw_array_digest(positions),
    )


class _ArtifactAwareRawLoader:
    """Development-only packed-artifact cache with bounded prefetch.

    Construction is intentionally deferred until the 7,133-row registry has
    been bound bijectively to the development cache.  Only those row records
    may be sliced from an artifact; no selection identity is accepted here.
    """

    def __init__(
        self,
        *,
        rows: Sequence[dict[str, Any]],
        registry_to_cache: np.ndarray,
        cache: dict[str, np.ndarray],
        verify_source_sha256: bool,
        max_bytes: int,
        max_sites: int,
        prefetch_rows: int,
        stage_workers: int = 1,
        budget_fractions: dict[str, float] | None = None,
        diploid_subsample: Sequence[int] = (),
        diploid_subsample_seed: int = 20260901,
        nested_ladder: bool = False,
        ladder_chains: Sequence[int] = CHAINS,
        diploid_fixed: int | None = None,
        expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
    ) -> None:
        if len(rows) != int(expected_rows):
            raise ValueError(
                "raw loader row count differs from the bound registry"
            )
        if registry_to_cache.shape != (len(rows),):
            raise ValueError("raw loader registry/cache mapping is not exact")
        self.rows = rows
        self.registry_to_cache = registry_to_cache
        self.cache = cache
        # The nested ladder is opt-in and mutually exclusive with the
        # independent draw: the two answer different questions and mixing them
        # would leave the summaries describing neither.
        self.nested_ladder = bool(nested_ladder)
        # Which chains this run may draw from.  A chain is a permutation of the
        # row's diploids, and the summary cache for (chain, level) describes
        # that chain's prefix and no other.  Drawing a chain whose cache is not
        # loaded is the one failure the ladder exists to prevent, and it is
        # silent: the raw path decodes chain 2's diploids, the evidence store
        # hands back chain 0's summaries of the same width, and the run trains
        # and converges on a mismatch.  The caller passes exactly the chains it
        # loaded, so the draw cannot leave them.
        # The bound cache states which diploids each of its rows describes.
        # It is the ladder's only source of truth for that; the draw below
        # chooses a rung, never a set of individuals.
        stored_columns = cache.get(LADDER_COLUMNS_MEMBER)
        self.ladder_columns = (
            np.ascontiguousarray(stored_columns, dtype=np.int64)
            if stored_columns is not None
            else None
        )
        if self.nested_ladder and self.ladder_columns is None:
            raise ValueError(
                "a nested-ladder run needs a bound cache that records "
                "ladder_columns"
            )
        if self.ladder_columns is not None and self.ladder_columns.shape[0] != len(rows):
            raise ValueError(
                "the bound cache's ladder_columns does not cover every row"
            )
        self.ladder_chains = tuple(int(chain) for chain in ladder_chains)
        if not self.ladder_chains or set(self.ladder_chains) - set(CHAINS):
            raise ValueError(
                f"ladder chains must be a non-empty subset of {CHAINS}; got "
                f"{tuple(ladder_chains)!r}"
            )
        self.diploid_subsample = tuple(int(item) for item in diploid_subsample)
        if any(item < 1 for item in self.diploid_subsample):
            raise ValueError("every diploid draw must be at least one")
        self.diploid_subsample_seed = int(diploid_subsample_seed)
        self.diploid_fixed = (
            None if diploid_fixed is None else int(diploid_fixed)
        )
        if self.diploid_fixed is not None and self.diploid_fixed < 1:
            raise ValueError("a fixed diploid count must be at least one")
        if self.diploid_fixed is not None and self.diploid_subsample:
            raise ValueError(
                "a fixed diploid count and a draw ladder are exclusive"
            )
        # Rebound once per epoch so a row is cut differently each pass.
        self.epoch = 0
        # Set only inside fixed_sample_size(); a measurement pins a count
        # rather than falling back to the row's own width.
        self.pinned_diploids: int | None = None
        self.verify_source_sha256 = bool(verify_source_sha256)
        self.prefetch_rows = int(prefetch_rows)
        # How many rows are staged concurrently. One reproduces the original
        # single-producer loader exactly. More than one lets zlib and the
        # numpy decode -- which release the GIL -- run in parallel while the
        # GPU consumes; staging costs about four times the GPU step on the
        # production cohort, so one thread leaves the GPU idle three quarters
        # of the time. Bounded by the queue depth so the reorder buffer, which
        # holds rows that finished ahead of their turn, can never exceed it.
        self.stage_workers = int(stage_workers)
        if self.stage_workers < 1:
            raise ValueError("stage_workers must be at least one")
        if self.stage_workers > self.prefetch_rows:
            raise ValueError(
                f"stage_workers ({self.stage_workers}) may not exceed "
                f"prefetch_rows ({self.prefetch_rows}); each worker holds one "
                "row and the budget is sized to the queue"
            )
        self._stats_lock = threading.Lock()
        self._inflight_guard = threading.Lock()
        self._inflight: dict[str, threading.Lock] = {}
        self._transient_bytes = 0
        self._transient_sites = 0
        # Built here, single-threaded, before any worker exists: a lazily
        # created Condition can be created twice by two workers' first charges.
        self._transient_cv = threading.Condition()
        # Rows in flight -- being staged, queued, or waiting in the reorder
        # buffer for an earlier row -- are capped at the queue depth. Without
        # the cap, workers run ahead of a slow row until the byte budget is
        # exhausted, and the worker staging that slow row then blocks on the
        # budget the others hold: a deadlock the consumer would only turn into
        # an error after thirty minutes. With it, the buffer is bounded by
        # construction and the budget need only hold prefetch_rows rows.
        self._inflight_slots = threading.BoundedSemaphore(self.prefetch_rows)
        self.host_max_bytes = int(max_bytes)
        self.host_max_sites = int(max_sites)
        fractions = dict(budget_fractions or _HOST_BUDGET_FRACTIONS)
        if set(fractions) != set(_HOST_BUDGET_FRACTIONS):
            raise ValueError("host budget split must name every component")
        self.budget_fractions = fractions
        # One percent of bytes is reserved for the largest observed transient
        # artifact materialisation plus its development-only copies.  The
        # persistent LRUs and pinned queue therefore cannot consume it.
        self.artifact_cache = _SizedLRU(
            max_bytes=max(1, int(max_bytes * fractions["artifact"])),
            max_sites=max_sites,
            label="packed_development_rows",
        )
        self.shift_cache = _SizedLRU(
            max_bytes=max(1, int(max_bytes * fractions["shift"])),
            max_sites=max_sites,
            label="carrier_control_shifts",
        )
        self.segment_plan_cache = _SizedLRU(
            max_bytes=max(
                1, int(max_bytes * fractions["segment_plan"])
            ),
            max_sites=max_sites,
            label="carrier_segment_plans",
        )
        self.map_cache = _SizedLRU(
            max_bytes=max(1, int(max_bytes * fractions["map"])),
            max_sites=2 * max_sites,
            label="observed_or_deranged_maps",
        )
        self.exposure_cache = _SizedLRU(
            max_bytes=max(
                1, int(max_bytes * fractions["map_exposure"])
            ),
            max_sites=2 * max_sites,
            label="map_exposure",
        )
        self.prefetch_budget = _PrefetchBudget(
            max_bytes=max(1, int(max_bytes * fractions["prefetch"])),
            max_sites=max(1, int(max_sites * 0.08)),
        )
        self.transient_max_bytes = max(
            1, int(max_bytes * fractions["transient"])
        )
        self.transient_max_sites = max(1, int(max_sites * 0.21))
        self.transient_peak_bytes = 0
        self.transient_peak_sites = 0
        self._verified_artifacts: set[str] = set()
        self._artifact_rows: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            self._artifact_rows.setdefault(_artifact_key(row), []).append(index)
        self.artifact_opens = 0
        self.rows_loaded = 0
        self.gpu_state_controls_staged = 0
        self.rows_staged = 0
        self.pinned_bytes_staged = 0
        self.sites_staged = 0
        # 过滤前的整行位点数。与 sites_staged (rung 过滤之后的数) 成对,
        # 两者之比就是这一档看不到的位点占比 —— 阶梯低档的这个比值必须
        # 明显小于 1, 否则说明 rung 过滤没生效, 低档偷看了整个面板的位点。
        self.row_sites_staged = 0
        self.packed_call_bytes_staged = 0
        self.cpu_dense_state_bytes_avoided = 0
        self.segment_plans_prepared = 0
        self.segment_plan_uses = 0
        self.segment_plan_checked_fallbacks = 0
        self.position_digest_checks = 0
        self.position_digest_mismatches = 0
        self.map_requests = 0
        self.map_exposure_requests = 0

    def _bump(self, name: str, delta: int = 1) -> None:
        """Advance a loader counter, atomically when rows are staged in parallel.

        ``self.x += 1`` is a load, an add and a store, and a thread switch
        between them loses an update. Lost updates would only miscount, but
        the counts go into the receipt's loader performance record and the
        terminal audit reconciles some of them against the caches, so they
        are kept exact.
        """

        lock = getattr(self, "_stats_lock", None)
        if lock is None or getattr(self, "stage_workers", 1) <= 1:
            setattr(self, name, getattr(self, name) + int(delta))
            return
        with lock:
            setattr(self, name, getattr(self, name) + int(delta))

    def _peak(self, name: str, value: int) -> None:
        lock = getattr(self, "_stats_lock", None)
        if lock is None or getattr(self, "stage_workers", 1) <= 1:
            setattr(self, name, max(getattr(self, name), int(value)))
            return
        with lock:
            setattr(self, name, max(getattr(self, name), int(value)))

    def _load_artifact(self, row: dict[str, Any]) -> dict[int, _PackedRawRow]:
        """Return only registered development slices from one source shard.

        NPZ members must be materialised to decompress them, but the complete
        shard is transient.  Mixed-artifact selection rows are never sliced,
        decoded, returned, or retained in either LRU.
        """

        key = _artifact_key(row)
        # Single flight per key, and exactly one counted lookup per request.
        # Two workers whose rows share a fragment would otherwise both miss,
        # both decompress, both put, and misses and opens would each count
        # twice for one artifact -- which the terminal audit reads as lost
        # accounting. The lookup happens inside the per-key lock so the second
        # arrival waits on the first and then records a hit: misses equal
        # opens equal entries, and hits plus misses equal requests. Locks are
        # per key, so unrelated fragments still decompress in parallel; an
        # uncontended lock costs nothing measurable against a decompression.
        guard = getattr(self, "_inflight_guard", None)
        if guard is None:
            cached = self.artifact_cache.get(key)
            if cached is not None:
                return cached
            return self._load_artifact_missed(row, key)
        with guard:
            inflight = self._inflight.setdefault(key, threading.Lock())
        with inflight:
            cached = self.artifact_cache.get(key)
            if cached is not None:
                return cached
            return self._load_artifact_missed(row, key)

    def _transient_condition(self) -> Any:
        """The one Condition every worker charges and releases through.

        Created in ``__init__``; this fallback exists only for loaders built
        without it (the tests' ``object.__new__`` stubs) and is guarded, because
        an unguarded check-then-create let two workers' first charges each
        build their own Condition: one then waited on a private object that no
        ``_transient_exit`` would ever notify, and the consumer's join waited
        on it for the rest of the allocation. Reviewers reproduced the hang.
        """

        condition = getattr(self, "_transient_cv", None)
        if condition is not None:
            return condition
        guard = getattr(self, "_stats_lock", None)
        if guard is None:
            guard = threading.Lock()
            self._stats_lock = guard
        with guard:
            condition = getattr(self, "_transient_cv", None)
            if condition is None:
                condition = threading.Condition()
                self._transient_cv = condition
        return condition

    def _transient_enter(self, nbytes: int, sites: int) -> None:
        """Charge a shard's transient bytes against the reserve, summed across workers.

        Waits for room rather than refusing it. The reserve was sized for one
        shard, and with N workers N shards materialise at once; a check that
        raised on the sum turned a reserve a little under N shards into an
        abort of the whole run, where the right thing is for the N-th worker
        to wait a few hundred milliseconds. Only a single charge that could
        never fit is refused, because no amount of waiting would admit it.
        Waiting here cannot deadlock: a worker waits holding only its own
        artifact's single-flight lock, and every other worker's charge is
        released after its put without touching any key lock.
        """

        nbytes, sites = int(nbytes), int(sites)
        if nbytes > self.transient_max_bytes or sites > self.transient_max_sites:
            raise RuntimeError(
                "one source artifact exceeds the reserved transient host byte/site "
                "budget; raise --host-budget-fraction transient="
            )
        condition = self._transient_condition()
        with condition:
            condition.wait_for(
                lambda: int(getattr(self, "_transient_bytes", 0)) + nbytes
                <= self.transient_max_bytes
                and int(getattr(self, "_transient_sites", 0)) + sites
                <= self.transient_max_sites
            )
            self._transient_bytes = int(getattr(self, "_transient_bytes", 0)) + nbytes
            self._transient_sites = int(getattr(self, "_transient_sites", 0)) + sites
            self.transient_peak_bytes = max(self.transient_peak_bytes, self._transient_bytes)
            self.transient_peak_sites = max(self.transient_peak_sites, self._transient_sites)

    def _transient_exit(self, nbytes: int, sites: int) -> None:
        condition = self._transient_condition()
        with condition:
            self._transient_bytes = int(getattr(self, "_transient_bytes", 0)) - int(nbytes)
            self._transient_sites = int(getattr(self, "_transient_sites", 0)) - int(sites)
            if self._transient_bytes < 0 or self._transient_sites < 0:
                raise RuntimeError("transient accounting underflow")
            condition.notify_all()

    def _load_artifact_missed(
        self, row: dict[str, Any], key: str
    ) -> dict[int, _PackedRawRow]:
        artifact = row["source_artifact"]
        path = Path(key)
        expected_size = int(artifact.get("npz_size_bytes", -1))
        if not path.is_file() or (expected_size >= 0 and path.stat().st_size != expected_size):
            raise ValueError("raw source artifact size differs from its receipt")
        if self.verify_source_sha256 and key not in self._verified_artifacts:
            if _sha256(path) != str(artifact.get("npz_sha256", "")):
                raise ValueError("raw source artifact SHA256 differs")
            self._verified_artifacts.add(key)
        with np.load(path, allow_pickle=False) as loaded:
            absent = sorted(set(_RAW_ARTIFACT_ARRAYS).difference(loaded.files))
            if absent:
                raise ValueError(f"raw source artifact lacks arrays: {absent}")
            arrays = {
                name: np.asarray(loaded[name])
                for name in _RAW_ARTIFACT_ARRAYS
            }
        _release_page_cache(path)
        if (
            arrays["variant_positions_bp"].dtype != np.uint32
            or arrays["variant_positions_bp"].ndim != 1
            or arrays["genotype_packed"].dtype != np.uint8
            or arrays["genotype_packed"].ndim != 2
            or arrays["genotype_packed"].shape[0]
            != len(arrays["variant_positions_bp"])
            or int(arrays["genotype_packed"].shape[1]) < 1
            or arrays["missing_flat_idx"].dtype != np.uint32
            or arrays["missing_flat_idx"].ndim != 1
            # Positions and rates do not share a dtype, and must not: a map
            # position is a base-pair coordinate, and float32 cannot separate
            # adjacent ones past about sixteen megabases -- two apart at fifty,
            # eight at a chromosome.  The generator therefore writes positions
            # float64 and rates float32, and the rates can afford it because
            # seven significant digits on a 1e-8 rate leave the cumulative
            # genetic distance four orders under the label's noise floor.
            # Older artifacts carry float32 positions on sequences short enough
            # for it, so both are accepted there; a float64 *rate* is not, as
            # it would mean the writer disagreed with the schema.
            or any(
                arrays[name].dtype not in (np.float32, np.float64)
                or arrays[name].ndim != 1
                for name in ("obs_rec_pos", "obs_mut_pos")
            )
            or any(
                arrays[name].dtype != np.float32 or arrays[name].ndim != 1
                for name in ("obs_rec_rate", "obs_mut_rate")
            )
        ):
            raise ValueError("raw source artifact dtypes/shapes differ from receipts")
        materialized_bytes = _arrays_nbytes(arrays.values())
        materialized_sites = len(arrays["variant_positions_bp"])
        self._bump("artifact_opens")
        development: dict[int, _PackedRawRow] = {}
        for registry_index in self._artifact_rows[key]:
            candidate = self.rows[registry_index]
            local = int(self.registry_to_cache[registry_index])
            positions = self._take(
                arrays, candidate, "variant_positions_bp", "variant"
            )
            packed = self._take(arrays, candidate, "genotype_packed", "variant")
            missing = self._take(arrays, candidate, "missing_flat_idx", "missing")
            sequence_length_bp = int(candidate["sequence_length_bp"])
            _validate_packed_row_values(
                positions,
                packed,
                missing,
                sequence_length_bp=sequence_length_bp,
            )
            raw_digest = _strict_sha256(
                candidate.get("raw_observation_sha256"),
                "registry raw observation digest",
            )
            positions_digest = _raw_array_digest(positions)
            receipt_digest = _strict_sha256(
                artifact.get("receipt_sha256"), "source receipt digest"
            )
            if (
                int(candidate.get("variant_count", -1)) != len(positions)
                or int(candidate.get("missing_call_count", -1)) != len(missing)
            ):
                raise ValueError("registry raw shape/count binding differs")
            target = np.asarray(self.cache["targets"][local], dtype=np.float32).copy()
            sample = _PackedRawRow(
                positions_bp=positions,
                genotype_packed=packed,
                missing_flat_idx=missing,
                recombination_positions=self._take(
                    arrays, candidate, "obs_rec_pos", "obs_rec_pos"
                ),
                recombination_rates=self._take(
                    arrays, candidate, "obs_rec_rate", "obs_rec_rate"
                ),
                mutation_positions=self._take(
                    arrays, candidate, "obs_mut_pos", "obs_mut_pos"
                ),
                mutation_rates=self._take(
                    arrays, candidate, "obs_mut_rate", "obs_mut_rate"
                ),
                sequence_length_bp=sequence_length_bp,
                canonical_id=str(candidate["canonical_id"]),
                registry_index=int(registry_index),
                raw_observation_sha256=raw_digest,
                positions_sha256=positions_digest,
                receipt_sha256=receipt_digest,
                values_prevalidated=True,
            )
            if _raw_digest(positions, packed, missing) != raw_digest:
                raise ValueError("raw observation digest differs from gate index")
            if _raw_array_digest(target) != candidate.get("target_sha256"):
                raise ValueError("bound development target digest differs from gate index")
            development[registry_index] = sample
        # Fail closed: the persistent value has exactly the registry indices
        # associated with this artifact and no source-local-row namespace.
        expected_indices = set(self._artifact_rows[key])
        if set(development) != expected_indices:
            raise RuntimeError("artifact cache contains a non-development row")
        nbytes = sum(
            _arrays_nbytes(
                (
                    sample.positions_bp,
                    sample.genotype_packed,
                    sample.missing_flat_idx,
                    sample.recombination_positions,
                    sample.recombination_rates,
                    sample.mutation_positions,
                    sample.mutation_rates,
                )
            )
            for sample in development.values()
        )
        artifact_sites = sum(len(sample.positions_bp) for sample in development.values())
        transient_bytes = materialized_bytes + nbytes
        transient_sites = materialized_sites + artifact_sites
        # Charged against the reserve as a running sum across workers, not per
        # call: with N workers, N shards are materialised at once, and a
        # per-call check sized for one let all N through while the recorded
        # peak was a fraction of the true one.
        self._transient_enter(transient_bytes, transient_sites)
        try:
            del arrays
            self._artifact_cache_put(key, development, nbytes, artifact_sites)
        finally:
            self._transient_exit(transient_bytes, transient_sites)
        return development

    def _artifact_cache_put(
        self, key: str, development: dict[int, _PackedRawRow], nbytes: int, sites: int
    ) -> None:
        self.artifact_cache.put(
            key, development, nbytes=nbytes, sites=sites
        )

    @staticmethod
    def _take(
        arrays: dict[str, np.ndarray],
        row: dict[str, Any],
        name: str,
        offset_name: str,
    ) -> np.ndarray:
        offsets = row.get("offsets")
        if not isinstance(offsets, dict) or offset_name not in offsets:
            raise ValueError("raw index row lacks bound offsets")
        start, stop = (int(value) for value in offsets[offset_name])
        values = arrays[name]
        if start < 0 or stop < start or stop > len(values):
            raise ValueError("raw index offsets lie outside artifact arrays")
        return np.asarray(values[start:stop]).copy()

    def _decode(self, row_index: int) -> _DecodedRawRow:
        row = self.rows[int(row_index)]
        local = int(self.registry_to_cache[int(row_index)])
        sample = self._load_artifact(row)[int(row_index)]
        positions = sample.positions_bp
        missing = sample.missing_flat_idx
        # The second entry is log2 of the row's haplotype count.  Four
        # diploids ride in every packed byte, so the row states its own
        # width; the development corpus's 64 was only ever one value of it.
        row_haplotypes = 8 * int(sample.genotype_packed.shape[1])
        expected_context = np.asarray(
            [np.log10(sample.sequence_length_bp), np.log2(row_haplotypes)],
            dtype=np.float32,
        )
        if (
            positions.dtype != np.uint32
            or not sample.values_prevalidated
            or sample.registry_index != int(row_index)
            or int(row.get("variant_count", -1)) != len(positions)
            or int(row.get("missing_call_count", -1)) != len(missing)
            or len(positions) != int(self.cache["variant_counts"][local])
            or not np.array_equal(
                expected_context[: 1 if self.nested_ladder else None],
                self.cache["context"][local][: 1 if self.nested_ladder else None],
            )
        ):
            raise ValueError("loaded raw sample differs from bound development cache")
        if self.nested_ladder:
            # A ladder's bound cache describes one rung, so its sample-size
            # channel is a property of the rung and cannot equal the row's --
            # comparing them would only ever be satisfied by a cache built at
            # the row's full width, which is the one width the ladder never
            # uses.  What must still hold is that the row is wide enough for
            # the ladder drawn from it, and that is checked directly.
            row_diploids = row_haplotypes // 2
            # ``ladder_levels`` is the precondition itself: it refuses a width
            # that is not a power of two of at least two, which is what makes
            # two chains disjoint halves of the row.
            ladder_levels(row_diploids)
        audit = {
            "canonical_id": sample.canonical_id,
            "reference_index": int(row["reference_index"]),
            "registry_index": int(row_index),
            "cache_index": local,
            # The view is part of the row's identity for this pass: the
            # summaries and the sidecar exist once per level, and reading the
            # wrong one trains on a description of diploids the raw path never
            # decoded.
            "ladder_chain": (
                self.ladder_chains[0] if self.nested_ladder else -1
            ),
            "ladder_level": (
                self._ladder_level(sample) if self.nested_ladder else -1
            ),
            "variant_count": len(positions),
            "missing_call_count": len(missing),
            "raw_observation_sha256": sample.raw_observation_sha256,
            "positions_sha256": sample.positions_sha256,
            "source_receipt_sha256": sample.receipt_sha256,
            "source_artifact_key": _artifact_key(row),
            "source_local_row": int(row.get("source_local_row", -1)),
        }
        nbytes = _arrays_nbytes(
            (
                sample.positions_bp,
                sample.genotype_packed,
                sample.missing_flat_idx,
                sample.recombination_positions,
                sample.recombination_rates,
                sample.mutation_positions,
                sample.mutation_rates,
            )
        )
        decoded = _DecodedRawRow(
            sample=sample,
            local=local,
            audit=audit,
            nbytes=nbytes,
            sites=len(positions),
        )
        self._bump("rows_loaded")
        return decoded

    @staticmethod
    def _derived_key(
        sample: _PackedRawRow,
        kind: str,
        *configuration: Any,
    ) -> tuple[Any, ...]:
        if not sample.values_prevalidated:
            raise ValueError("derived cache input lacks strict CPU prevalidation")
        return (
            str(kind),
            int(sample.registry_index),
            sample.raw_observation_sha256,
            sample.positions_sha256,
            sample.receipt_sha256,
            *configuration,
        )

    @staticmethod
    def _readonly(value: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(value)
        result.setflags(write=False)
        return result

    def _maps(self, sample: _PackedRawRow, map_kind: str) -> _CachedMaps:
        if map_kind not in ("observed", "deranged"):
            raise ValueError("map cache kind must be observed or deranged")
        self._bump("map_requests")
        key = self._derived_key(sample, f"map:{map_kind}")
        cached = self.map_cache.get(key)
        if cached is not None:
            return cached

        def copied_map(
            positions: np.ndarray, rates: np.ndarray, label: str
        ) -> tuple[np.ndarray, np.ndarray]:
            positions = np.asarray(positions, dtype=np.float64)
            rates = np.asarray(rates, dtype=np.float64)
            if (
                positions.ndim != 1
                or rates.ndim != 1
                or len(positions) != len(rates) + 1
                or len(rates) < 1
                or not np.isfinite(positions).all()
                or not np.isfinite(rates).all()
                or np.any(np.diff(positions) < 0)
                or np.any(rates < 0)
            ):
                raise ValueError(f"{label} map is invalid")
            return positions.copy(), rates.copy()

        rec_positions, rec_rates = copied_map(
            sample.recombination_positions,
            sample.recombination_rates,
            "recombination",
        )
        mut_positions, mut_rates = copied_map(
            sample.mutation_positions,
            sample.mutation_rates,
            "mutation",
        )
        rec_shift = mut_shift = 0
        if map_kind == "deranged":
            rec_positions, rec_rates, rec_shift = derange_map_rates(
                rec_positions,
                rec_rates,
                sample_identity=sample.canonical_id,
                label="recombination",
            )
            mut_positions, mut_rates, mut_shift = derange_map_rates(
                mut_positions,
                mut_rates,
                sample_identity=sample.canonical_id,
                label="mutation",
            )
        result = _CachedMaps(
            recombination_positions=self._readonly(rec_positions),
            recombination_rates=self._readonly(rec_rates),
            mutation_positions=self._readonly(mut_positions),
            mutation_rates=self._readonly(mut_rates),
            recombination_shift=int(rec_shift),
            mutation_shift=int(mut_shift),
        )
        self.map_cache.put(
            key,
            result,
            nbytes=result.nbytes,
            sites=result.interval_units,
        )
        return result

    def _map_exposure(
        self,
        torch: Any,
        model: Any,
        sample: _PackedRawRow,
        maps: _CachedMaps,
        map_kind: str,
    ) -> np.ndarray:
        self._bump("map_exposure_requests")
        key = self._derived_key(
            sample,
            f"exposure:{map_kind}",
            int(model.macro_width_bp),
            int(model.n_region_scales),
        )
        cached = self.exposure_cache.get(key)
        if cached is not None:
            return cached
        tensors = tuple(
            torch.from_numpy(np.array(value, dtype=np.float32, copy=True, order="C"))
            for value in (
                maps.recombination_positions,
                maps.recombination_rates,
                maps.mutation_positions,
                maps.mutation_rates,
            )
        )
        exposure = model.prepare_map_exposure(
            *tensors, sequence_length_bp=sample.sequence_length_bp
        ).detach().cpu().numpy().astype(np.float32, copy=False)
        exposure = self._readonly(exposure)
        self.exposure_cache.put(
            key,
            exposure,
            nbytes=exposure.nbytes,
            sites=int(exposure.shape[0]),
        )
        return exposure

    def set_epoch(self, epoch: int) -> None:
        """Rebind the pass the individual draw is keyed to."""

        self.epoch = int(epoch)

    @contextlib.contextmanager
    def fixed_sample_size(self, n_diploids: int) -> Any:
        """Hold the individual count at ``n_diploids`` for a measurement.

        Switching the draw off would fall back to the row's own width, which
        is what the frozen configuration cannot fit and is not the size the
        arm was trained at.  A measurement pins a count instead, and takes the
        nested prefix so repeated measurements at different counts sit on the
        same individuals.
        """

        if int(n_diploids) < 1:
            raise ValueError("a measurement needs at least one diploid")
        previous = self.pinned_diploids
        self.pinned_diploids = int(n_diploids)
        try:
            yield
        finally:
            self.pinned_diploids = previous

    def _row_permutation(self, sample: _PackedRawRow, width: int) -> np.ndarray:
        """One canonical ordering of a row's individuals.

        Prefixes of this ordering are nested by construction, which is what
        lets a sample-size curve be read on one fixed set of individuals: only
        the count changes between points, so individual identity cannot leak
        into the shape of the curve.  It depends on the row and the seed, never
        on the pass, so a resumed run reproduces it exactly.
        """

        digest = hashlib.sha256(
            f"{sample.canonical_id}|perm|"
            f"{self.diploid_subsample_seed}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(
            int.from_bytes(digest[:8], byteorder="little", signed=False)
        )
        return rng.permutation(int(width))

    def ladder_view(self, sample: _PackedRawRow, width: int) -> tuple[int, int]:
        """Return the (chain, level) this pass draws for one row.

        A row is seen once per epoch and has four chains of six levels, so the
        pair is drawn together from the row's identity, the epoch and the run
        seed.  The caller needs it as much as the loader does: the summary
        cache for that exact (chain, level) is the only one describing the
        diploids the raw path is about to read.
        """

        return self.ladder_view_for(sample.canonical_id, ladder_levels(width))

    def ladder_view_for(
        self, canonical_id: str, levels: Sequence[int]
    ) -> tuple[int, int]:
        """The same draw, for a caller that has an identity but no decoded row.

        The summary-only arm never opens an artifact, yet it must read the
        summary cache of the very rung the raw arms are reading this epoch, or
        the arms stop being comparable.  The draw therefore depends on nothing
        but the row's identity, the epoch, the run seed and the rung list --
        all of which that caller has.
        """

        levels = tuple(int(level) for level in levels)
        if not levels:
            raise ValueError("a ladder draw needs at least one level")
        digest = hashlib.sha256(
            f"{canonical_id}|{self.epoch}|ladder|"
            f"{self.diploid_subsample_seed}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(
            int.from_bytes(digest[:8], byteorder="little", signed=False)
        )
        chain = int(self.ladder_chains[int(rng.integers(len(self.ladder_chains)))])
        level = int(levels[int(rng.integers(len(levels)))])
        return chain, level

    def _ladder_level(self, sample: _PackedRawRow) -> int:
        """The rung this pass reads the row at -- pinned, or drawn."""

        width = 4 * int(sample.genotype_packed.shape[1])
        if self.pinned_diploids is not None:
            return int(self.pinned_diploids)
        return int(self.ladder_view(sample, width)[1])

    def _diploid_columns(self, sample: _PackedRawRow) -> np.ndarray | None:
        """Choose this pass's individuals for one row.

        Individuals are exchangeable draws from the population, so which ones
        are kept carries no information -- only how many.  Three modes, in
        precedence order: a pinned count for a measurement, a fixed count for
        the whole run, or a per-pass draw from the ladder.  The first two take
        a nested prefix of the row's canonical ordering; the third draws
        independently, since decorrelation is what a training view wants.
        """

        width = 4 * int(sample.genotype_packed.shape[1])
        nested = True
        if self.nested_ladder:
            # A pinned count under a ladder names a rung: measurements sit at
            # one rung for the whole run so the selection metric carries no
            # draw variance, and the evidence store is keyed by that rung. The
            # two used to disagree -- pinning suppressed the level, then the
            # store demanded one -- so every measurement raised.
            level = (
                int(self.pinned_diploids)
                if self.pinned_diploids is not None
                else self.ladder_view(sample, width)[1]
            )
            if level not in ladder_levels(width):
                raise ValueError(
                    f"{level} is not a rung of a {width}-diploid ladder; "
                    f"available rungs are {ladder_levels(width)}"
                )
            if level == width:
                return None
            # Read from the cache, not re-derived from an identity string.
            #
            # Both sides used to call ``level_columns`` and had to agree about
            # what to seed it with, and they did not: the builder passed the
            # simulator's ``sample_id`` while this passed ``canonical_id``.
            # Two different strings, two unrelated permutations, and each side
            # took a correct nested prefix -- of different diploids, with an
            # overlap of exactly chance (1.00 of 8 at level 8, measured). Every
            # fusion row was trained on a summary of individuals its raw path
            # never decoded, and nothing could detect it: both column sets are
            # valid, the shapes agree, and the run converges.
            #
            # The builder already writes ``ladder_columns`` -- the exact
            # diploids each cache row describes -- and nothing read it. Reading
            # it is what makes the question unaskable: there is one derivation,
            # it lives with the summaries it describes, and a level is a prefix
            # of it because the ladder is nested.
            columns = self.ladder_columns
            if columns is None:
                raise ValueError(
                    "the nested ladder needs the bound cache's ladder_columns; "
                    "this cache predates them, and re-deriving the subset from "
                    "an identity string is what this replaces"
                )
            local = int(self.registry_to_cache[int(sample.registry_index)])
            drawn = np.ascontiguousarray(columns[local][:level])
            if drawn.shape != (level,) or len(set(drawn.tolist())) != level:
                raise ValueError(
                    f"cache row {local} does not name {level} distinct diploids"
                )
            return drawn
        if self.pinned_diploids is not None:
            drawn = int(self.pinned_diploids)
        elif self.diploid_fixed is not None:
            drawn = int(self.diploid_fixed)
        elif self.diploid_subsample:
            allowed = tuple(
                item for item in self.diploid_subsample if item <= width
            )
            if not allowed:
                raise ValueError(
                    "every requested diploid draw exceeds the row's own width"
                )
            digest = hashlib.sha256(
                f"{sample.canonical_id}|{self.epoch}|"
                f"{self.diploid_subsample_seed}".encode("utf-8")
            ).digest()
            rng = np.random.default_rng(
                int.from_bytes(digest[:8], byteorder="little", signed=False)
            )
            drawn = int(allowed[int(rng.integers(len(allowed)))])
            nested = False
        else:
            return None
        if drawn < 1 or drawn > width:
            raise ValueError(
                "requested diploid count is outside the row's own width"
            )
        if drawn == width:
            return None
        if nested:
            return np.sort(
                self._row_permutation(sample, width)[:drawn]
            ).astype(np.int64, copy=False)
        digest = hashlib.sha256(
            f"{sample.canonical_id}|{self.epoch}|{drawn}|"
            f"{self.diploid_subsample_seed}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(
            int.from_bytes(digest[:8], byteorder="little", signed=False)
        )
        return np.sort(
            rng.choice(width, size=drawn, replace=False)
        ).astype(np.int64, copy=False)

    def _control_shifts(
        self,
        sample: _PackedRawRow,
        arm: str,
        *,
        n_individuals: int | None = None,
    ) -> np.ndarray | None:
        if arm not in (
            "sitewise_diploid_permuted",
            "longrange_block_permuted",
        ):
            return None
        width = (
            int(n_individuals)
            if n_individuals is not None
            else 4 * int(sample.genotype_packed.shape[1])
        )
        key = self._derived_key(sample, f"shift:{arm}", 12_800, width)
        cached = self.shift_cache.get(key)
        if cached is not None:
            return cached
        if arm == "sitewise_diploid_permuted":
            shifts = prepare_carrier_control_shifts(
                arm,
                sample_identity=sample.canonical_id,
                variant_count=len(sample.positions_bp),
                n_individuals=width,
            )
        else:
            shifts = prepare_carrier_control_shifts(
                arm,
                sample_identity=sample.canonical_id,
                variant_count=len(sample.positions_bp),
                positions=sample.positions_bp,
                block_width_bp=12_800,
            )
        shifts = self._readonly(shifts)
        if (
            shifts.dtype != np.uint8
            or shifts.shape != (len(sample.positions_bp),)
            or (len(shifts) and (int(shifts.min()) < 1 or int(shifts.max()) > 31))
        ):
            raise ValueError("prepared carrier-control shifts are invalid")
        self.shift_cache.put(
            key,
            shifts,
            nbytes=shifts.nbytes,
            sites=len(sample.positions_bp),
        )
        return shifts

    def _segment_plan(
        self, torch: Any, model: Any, sample: _PackedRawRow
    ) -> _CachedSegmentPlan:
        configuration = (
            int(sample.sequence_length_bp),
            int(model.chunk_sites),
            int(model.leaf_width_bp),
            int(model.macro_levels),
        )
        key = self._derived_key(sample, "segment_plan", *configuration)
        cached = self.segment_plan_cache.get(key)
        if cached is not None:
            return cached
        positions64 = torch.from_numpy(
            sample.positions_bp.astype(np.int64, copy=True)
        )
        plan = model.prepare_segment_plan(
            positions64,
            sequence_length_bp=sample.sequence_length_bp,
            receipt_sha256=sample.raw_observation_sha256,
        )
        # ``prepare_segment_plan`` hashes and validates the exact CPU int64
        # positions while constructing immutable descriptors.  Rebuilding it
        # through ``validate_segment_plan`` here would duplicate all O(V)
        # preparation work; hot-path uses compare only immutable metadata.
        if (
            plan.n_sites != len(sample.positions_bp)
            or plan.sequence_length_bp != sample.sequence_length_bp
            or plan.receipt_sha256 != sample.raw_observation_sha256
            or plan.chunk_sites != model.chunk_sites
            or plan.leaf_width_bp != model.leaf_width_bp
            or plan.macro_levels != model.macro_levels
            or not plan.cpu_validated
            or not str(plan.positions_sha256)
            or int(plan.nbytes) <= 0
            or int(plan.sites) != len(sample.positions_bp)
            or int(plan.transfer_calls_expected) != 0
        ):
            raise ValueError("prepared segment plan differs from row/config receipt")
        plan = plan.pin_memory()
        if not plan.is_pinned:
            raise RuntimeError("prepared segment plan is not pinned")
        cached_plan = _CachedSegmentPlan(
            plan=plan,
            registry_index=int(sample.registry_index),
            raw_positions_sha256=sample.positions_sha256,
            plan_positions_sha256=str(plan.positions_sha256),
            raw_observation_sha256=sample.raw_observation_sha256,
            source_receipt_sha256=sample.receipt_sha256,
            sequence_length_bp=int(sample.sequence_length_bp),
            chunk_sites=int(model.chunk_sites),
            leaf_width_bp=int(model.leaf_width_bp),
            macro_levels=int(model.macro_levels),
            nbytes=int(plan.nbytes),
            sites=int(plan.sites),
        )
        self._bump("segment_plans_prepared")
        self.segment_plan_cache.put(
            key,
            cached_plan,
            nbytes=cached_plan.nbytes,
            sites=cached_plan.sites,
        )
        return cached_plan

    def record_segment_plan_use(self, staged: _StagedRawRow, model: Any) -> None:
        """Authorize one fast-path use via O(1) receipt/config comparisons."""

        record = staged.segment_plan
        self.position_digest_checks += 1
        valid = (
            staged.values_prevalidated is True
            and record.registry_index == int(staged.audit["registry_index"])
            and record.raw_positions_sha256 == staged.positions_sha256
            and record.raw_observation_sha256 == staged.raw_observation_sha256
            and record.source_receipt_sha256 == staged.receipt_sha256
            and record.sequence_length_bp == staged.sequence_length_bp
            and record.chunk_sites == int(model.chunk_sites)
            and record.leaf_width_bp == int(model.leaf_width_bp)
            and record.macro_levels == int(model.macro_levels)
            and record.sites == staged.sites
            and record.plan_positions_sha256
            == str(record.plan.positions_sha256)
            and record.plan.receipt_sha256 == staged.raw_observation_sha256
            and record.plan.cpu_validated is True
        )
        if not valid:
            self.position_digest_mismatches += 1
            raise ValueError("staged segment plan differs from bound row/config metadata")
        self._bump("segment_plan_uses")

    def _prepare_controlled(
        self,
        arm: str,
        sample: _PackedRawRow,
    ) -> ControlledRawArm:
        screen_spec = screen_arm_spec(arm) if arm in SCREEN_ARMS else None
        map_kind = (
            screen_spec.map_kind
            if screen_spec is not None
            else ("deranged" if arm == "map_deranged" else "observed")
        )
        maps = self._maps(sample, map_kind)
        details: dict[str, Any] = {
            "arm": arm,
            "sample_identity_sha256": hashlib.sha256(
                sample.canonical_id.encode("utf-8")
            ).hexdigest(),
            "site_count": int(len(sample.positions_bp)),
            "block_width_bp": 12_800,
            "selection_fields_read": [],
            "same_graph_value_intervention": True,
        }
        if screen_spec is not None:
            changed_fields = [
                name
                for name, changed in (
                    ("raw_signal_gate", screen_spec.raw_signal_mode == "zero"),
                    (
                        "site_recombination_gate",
                        screen_spec.site_recombination_mode == "zero",
                    ),
                    (
                        "macro_map_features",
                        screen_spec.map_feature_mode != "recombination_only",
                    ),
                    ("recombination_map_alignment", map_kind == "deranged"),
                    ("physics_attention_bias_gate", True),
                )
                if changed
            ]
            details.update(
                {
                    "changed_fields": changed_fields,
                    "screening_only": True,
                    "map_kind": map_kind,
                    "raw_signal_mode": screen_spec.raw_signal_mode,
                    "site_recombination_mode": (
                        screen_spec.site_recombination_mode
                    ),
                    "site_recombination_fusion_mode": (
                        screen_spec.site_recombination_fusion_mode
                    ),
                    "map_feature_mode": screen_spec.map_feature_mode,
                    "long_residual_mode": "zero",
                    "physics_bias_mode": "zero",
                    "all_sites_still_executed": True,
                    "map_rate_shifts": {
                        "recombination": maps.recombination_shift,
                        "mutation": maps.mutation_shift,
                    },
                    "deranged_pair_multiset_span_and_mass_equal": (
                        map_kind == "deranged"
                    ),
                }
            )
            return ControlledRawArm(
                diploid_states=None,
                recombination_positions=maps.recombination_positions,
                recombination_rates=maps.recombination_rates,
                mutation_positions=maps.mutation_positions,
                mutation_rates=maps.mutation_rates,
                long_residual_mode="zero",
                audit=details,
                map_kind=map_kind,
                site_recombination_mode=screen_spec.site_recombination_mode,
                site_recombination_fusion_mode=(
                    screen_spec.site_recombination_fusion_mode
                ),
                map_feature_mode=screen_spec.map_feature_mode,
                raw_signal_mode=screen_spec.raw_signal_mode,
                physics_bias_mode="zero",
            )
        long_mode = "real"
        if arm == "sitewise_diploid_permuted":
            details.update(
                {
                    "changed_fields": ["diploid_states"],
                    "per_site_six_state_multiset_equal": True,
                    "representation": "sitewise nonidentity diploid permutation",
                }
            )
        elif arm == "longrange_block_permuted":
            details.update(
                {
                    "changed_fields": ["diploid_states"],
                    "per_site_six_state_multiset_equal": True,
                    "within_block_carrier_tracks_retained": True,
                    "representation": "physical-block diploid permutation",
                }
            )
        elif arm == "raw_ablated":
            details.update(
                {
                    "changed_fields": ["diploid_states"],
                    "exact_missingness_retained": True,
                    "representation": "alternate dosage ablated",
                }
            )
        elif arm == "map_deranged":
            changed_fields: list[str] = []
            zero_shift_labels: list[str] = []
            for label, field, shift in (
                (
                    "recombination",
                    "recombination_map",
                    maps.recombination_shift,
                ),
                ("mutation", "mutation_map", maps.mutation_shift),
            ):
                if shift:
                    changed_fields.append(field)
                else:
                    zero_shift_labels.append(label)
            details.update(
                {
                    "map_rate_shifts": {
                        "recombination": maps.recombination_shift,
                        "mutation": maps.mutation_shift,
                    },
                    "changed_fields": changed_fields,
                    "zero_shift_or_degenerate_constant_maps": zero_shift_labels,
                    "map_interval_width_rate_pair_multiset_equal": True,
                    "map_total_span_and_mass_equal": True,
                }
            )
        elif arm == "longrange_residual_zero":
            long_mode = "zero"
            details.update(
                {
                    "changed_fields": ["long_residual_gate"],
                    "long_branch_still_executed": True,
                }
            )
        elif arm == "carrier_correct":
            details["changed_fields"] = []
        else:
            raise ValueError(f"unknown raw arm: {arm}")
        return ControlledRawArm(
            diploid_states=None,  # all six arms decode/control on the GPU
            recombination_positions=maps.recombination_positions,
            recombination_rates=maps.recombination_rates,
            mutation_positions=maps.mutation_positions,
            mutation_rates=maps.mutation_rates,
            long_residual_mode=long_mode,
            audit=details,
            map_kind=map_kind,
        )

    @staticmethod
    def _pin(torch: Any, value: np.ndarray, *, dtype: Any | None = None) -> Any:
        # A private writable copy prevents PyTorch from inheriting read-only
        # NumPy storage from immutable LRUs; pinning then owns the async-H2D
        # staging allocation for exactly one queued row.
        tensor = torch.from_numpy(np.array(value, copy=True, order="C"))
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.pin_memory()

    def _stage(
        self,
        torch: Any,
        model: Any,
        row_index: int,
        arm: str,
    ) -> _StagedRawRow:
        decoded = self._decode(int(row_index))
        sample = decoded.sample
        controlled = self._prepare_controlled(arm, sample)
        map_kind = controlled.map_kind
        maps = self._maps(sample, map_kind)
        exposure = self._map_exposure(
            torch,
            model,
            sample,
            maps,
            map_kind,
        )
        diploid_columns = self._diploid_columns(sample)
        row_sites = int(decoded.sites)
        if diploid_columns is not None:
            # The rung sees only the sites polymorphic among its own diploids.
            # Before the segment plan, because the plan is a function of the
            # positions and is cached by their digest; a plan of the panel's
            # sites would hand the model the panel's site density at every
            # rung, which is exactly what a rung must not see.
            sample = rung_site_subset(sample, diploid_columns)
        segment_plan = self._segment_plan(torch, model, sample)
        context = np.asarray(self.cache["context"][decoded.local], dtype=np.float32)
        if diploid_columns is not None:
            # Tell the model the size it is about to be handed, not the
            # size the row was generated at.
            context = context.copy()
            context[1] = np.float32(np.log2(2 * len(diploid_columns)))
        target = np.asarray(self.cache["targets"][decoded.local], dtype=np.float32)
        positions = self._pin(torch, sample.positions_bp)
        packed = self._pin(torch, sample.genotype_packed)
        missing = self._pin(torch, sample.missing_flat_idx)
        shifts_array = self._control_shifts(
            sample,
            arm,
            n_individuals=(
                len(diploid_columns) if diploid_columns is not None else None
            ),
        )
        control_shifts = (
            self._pin(torch, shifts_array) if shifts_array is not None else None
        )
        state_control_arm = (
            arm
            if arm in (
                "sitewise_diploid_permuted",
                "longrange_block_permuted",
                "raw_ablated",
            )
            else None
        )
        if state_control_arm is not None:
            self._bump("gpu_state_controls_staged")
        context_tensor = self._pin(torch, context)
        target_tensor = self._pin(torch, target)
        nw_tensor = self._pin(
            torch,
            np.asarray(self.cache["nw_log10"][decoded.local], dtype=np.float32),
        )
        exposure_tensor = self._pin(torch, exposure)
        if controlled.site_recombination_mode == "disabled":
            rec_positions_tensor = None
            rec_rates_tensor = None
            rec_prefix_tensor = None
        else:
            rec_positions = np.asarray(
                controlled.recombination_positions, dtype=np.float64
            )
            rec_rates = np.asarray(controlled.recombination_rates, dtype=np.float64)
            rec_prefix = np.concatenate(
                (
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(np.diff(rec_positions) * rec_rates, dtype=np.float64),
                )
            )
            if (
                len(rec_prefix) != len(rec_positions)
                or not np.isfinite(rec_prefix).all()
                or np.any(np.diff(rec_prefix) < 0)
            ):
                raise ValueError("prepared site recombination prefix is invalid")
            rec_positions_tensor = self._pin(
                torch, rec_positions
            )
            rec_rates_tensor = self._pin(
                torch, rec_rates
            )
            rec_prefix_tensor = self._pin(
                torch, rec_prefix
            )
        staged_bytes = sum(
            int(value.numel() * value.element_size())
            for value in (
                positions,
                packed,
                missing,
                *(value for value in (control_shifts,) if value is not None),
                context_tensor,
                target_tensor,
                nw_tensor,
                exposure_tensor,
                *(
                    value
                    for value in (
                        rec_positions_tensor,
                        rec_rates_tensor,
                        rec_prefix_tensor,
                    )
                    if value is not None
                ),
            )
        ) + int(segment_plan.nbytes)
        self._bump("rows_staged")
        self._bump("pinned_bytes_staged", staged_bytes)
        self._bump("sites_staged", len(sample.positions_bp))
        self._bump("row_sites_staged", row_sites)
        self._bump(
            "packed_call_bytes_staged",
            int(
                packed.numel() * packed.element_size()
                + missing.numel() * missing.element_size()
            ),
        )
        self._bump("cpu_dense_state_bytes_avoided", int(decoded.sites * 32))
        return _StagedRawRow(
            positions=positions,
            genotype_packed=packed,
            missing_flat_idx=missing,
            control_shifts=control_shifts,
            state_control_arm=state_control_arm,
            context=context_tensor,
            target=target_tensor,
            nw_log10=nw_tensor,
            map_exposure=exposure_tensor,
            segment_plan=segment_plan,
            positions_sha256=sample.positions_sha256,
            raw_observation_sha256=sample.raw_observation_sha256,
            receipt_sha256=sample.receipt_sha256,
            values_prevalidated=sample.values_prevalidated,
            sequence_length_bp=sample.sequence_length_bp,
            long_residual_mode=controlled.long_residual_mode,
            audit={
                **decoded.audit,
                "control": controlled.audit,
                "row_sites": row_sites,
                "rung_sites": int(len(sample.positions_bp)),
            },
            nbytes=staged_bytes,
            sites=int(len(sample.positions_bp)),
            recombination_positions=rec_positions_tensor,
            recombination_rates=rec_rates_tensor,
            recombination_prefix_mass=rec_prefix_tensor,
            site_recombination_mode=controlled.site_recombination_mode,
            site_recombination_fusion_mode=(
                controlled.site_recombination_fusion_mode
            ),
            map_feature_mode=controlled.map_feature_mode,
            raw_signal_mode=controlled.raw_signal_mode,
            physics_bias_mode=controlled.physics_bias_mode,
            diploid_columns=diploid_columns,
        )

    def iter_staged(
        self,
        torch: Any,
        model: Any,
        indices: Iterable[int],
        arm: str,
    ) -> Iterable[_StagedRawRow]:
        """Yield exactly the caller's order from ``stage_workers`` producers.

        With one worker this is the original bounded single-producer loader.
        With more, each worker takes the next row from a shared cursor, stages
        it, and hands it back tagged with its sequence number; the consumer
        holds rows that arrive early in a reorder buffer and yields strictly
        in sequence. The order the caller sees, and therefore the SGD order
        and the row-order digest, is identical for any worker count -- only
        how far ahead staging runs changes.

        Bounds. The queue holds at most ``prefetch_rows`` rows; each worker
        holds at most one more; the reorder buffer can hold at most
        ``stage_workers - 1`` rows that finished ahead of the one the consumer
        is waiting for, since workers take rows in sequence. The prefetch
        budget is charged when a row is staged and released when the consumer
        is done with it, so the bytes in flight are bounded the way they were
        before, with the extra headroom the constructor requires.

        A hang is turned into an error. The one way this can deadlock is a
        worker blocking on the byte budget while the budget is held by rows
        the consumer cannot take because an earlier one is still being
        staged; that needs the budget to be smaller than a handful of rows, and
        rather than wait forever the consumer raises after a long silence.
        """

        workers = max(1, int(getattr(self, "stage_workers", 1)))
        work: queue.Queue[Any] = queue.Queue(maxsize=self.prefetch_rows)
        sentinel = object()
        stop = threading.Event()
        cursor = enumerate(indices)
        cursor_lock = threading.Lock()

        consumer_gone = threading.Event()

        def enqueue(value: Any) -> bool:
            """Hand a staged row over; give up if the run is winding down."""

            while not stop.is_set():
                try:
                    work.put(value, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def deliver(value: Any) -> None:
            """Hand over a terminal message -- an error or a sentinel.

            These must arrive even after ``stop`` is set, because ``stop`` is
            what a failing worker sets to halt its peers, and the consumer is
            waiting for exactly this message. Only a consumer that has already
            left releases the obligation; its exit drains the queue, so a put
            that was blocked on a full queue completes and the join below can
            return.
            """

            while not consumer_gone.is_set():
                try:
                    work.put(value, timeout=0.1)
                    return
                except queue.Full:
                    continue

        def next_task() -> tuple[int, int] | None:
            with cursor_lock:
                try:
                    sequence, row_index = next(cursor)
                except StopIteration:
                    return None
            return int(sequence), int(row_index)

        slots = getattr(self, "_inflight_slots", None)

        def take_slot() -> bool:
            """Wait for a row in flight to be yielded; give up on shutdown."""

            if slots is None:
                return True
            while not stop.is_set():
                if slots.acquire(timeout=0.1):
                    return True
            return False

        def produce() -> None:
            try:
                while not stop.is_set():
                    if not take_slot():
                        break
                    task = next_task()
                    if task is None:
                        if slots is not None:
                            slots.release()
                        break
                    sequence, row_index = task
                    try:
                        staged = self._stage(torch, model, row_index, arm)
                        charged = self.prefetch_budget.acquire(
                            staged.nbytes, staged.sites, cancel=consumer_gone
                        )
                    except BaseException:
                        if slots is not None:
                            slots.release()
                        raise
                    if not charged:
                        # The consumer left while this row waited for budget;
                        # nothing was charged, so nothing is released.
                        if slots is not None:
                            slots.release()
                        break
                    if not enqueue((sequence, staged)):
                        self.prefetch_budget.release(staged.nbytes, staged.sites)
                        if slots is not None:
                            slots.release()
                        break
            except BaseException as error:  # propagate producer failures exactly
                stop.set()
                deliver(error)
            finally:
                deliver(sentinel)

        producers = [
            threading.Thread(
                target=produce,
                name=f"carrier-raw-prefetch-{arm}-{worker}",
                daemon=True,
            )
            for worker in range(workers)
        ]
        for producer in producers:
            producer.start()

        pending: dict[int, _StagedRawRow] = {}
        expected = 0
        finished = 0
        try:
            while finished < workers:
                try:
                    value = work.get(timeout=1800.0)
                except queue.Empty:
                    if any(producer.is_alive() for producer in producers):
                        raise RuntimeError(
                            "staging made no progress for 30 minutes with a "
                            "producer still alive; the prefetch budget is too "
                            "small for the worker count, or a worker is stuck"
                        )
                    continue
                if value is sentinel:
                    finished += 1
                    continue
                if isinstance(value, BaseException):
                    raise value
                sequence, staged = value
                pending[sequence] = staged
                while expected in pending:
                    row = pending.pop(expected)
                    expected += 1
                    try:
                        yield row
                    finally:
                        self.prefetch_budget.release(row.nbytes, row.sites)
                        if slots is not None:
                            slots.release()
            if pending:
                raise RuntimeError(
                    f"staging finished with {len(pending)} rows never yielded; "
                    f"the sequence stalled at {expected}"
                )
        finally:
            stop.set()
            consumer_gone.set()

            def drain() -> None:
                while True:
                    try:
                        item = work.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(item, tuple) and isinstance(item[1], _StagedRawRow):
                        self.prefetch_budget.release(item[1].nbytes, item[1].sites)
                        if slots is not None:
                            slots.release()
                for row in pending.values():
                    self.prefetch_budget.release(row.nbytes, row.sites)
                    if slots is not None:
                        slots.release()
                pending.clear()

            # Drain once to unblock producers in Queue.put, then again after
            # join to cover the race where a put succeeds just after the first
            # get_nowait observed an empty queue.
            drain()
            for producer in producers:
                producer.join()
            drain()

    def stats(self) -> dict[str, Any]:
        return {
            "artifact_cache": self.artifact_cache.stats(),
            "shift_cache": self.shift_cache.stats(),
            "segment_plan_cache": self.segment_plan_cache.stats(),
            "map_cache": self.map_cache.stats(),
            "map_exposure_cache": self.exposure_cache.stats(),
            "prefetch": {
                "queue_rows": self.prefetch_rows,
                "stage_workers": int(getattr(self, "stage_workers", 1)),
                "max_bytes": self.prefetch_budget.max_bytes,
                "max_sites": self.prefetch_budget.max_sites,
                "peak_bytes": self.prefetch_budget.peak_bytes,
                "peak_sites": self.prefetch_budget.peak_sites,
            },
            "transient_artifact": {
                "max_bytes": self.transient_max_bytes,
                "max_sites": self.transient_max_sites,
                "peak_bytes": self.transient_peak_bytes,
                "peak_sites": self.transient_peak_sites,
            },
            "artifact_opens": self.artifact_opens,
            "rows_loaded": self.rows_loaded,
            "map_requests": self.map_requests,
            "map_exposure_requests": self.map_exposure_requests,
            "gpu_state_controls_staged": self.gpu_state_controls_staged,
            "cpu_dense_states_cached": 0,
            "rows_staged": self.rows_staged,
            "sites_staged": self.sites_staged,
            "row_sites_staged": self.row_sites_staged,
            "pinned_bytes_staged": self.pinned_bytes_staged,
            "packed_call_bytes_staged": self.packed_call_bytes_staged,
            "cpu_dense_state_h2d_bytes": 0,
            "cpu_dense_state_bytes_avoided": self.cpu_dense_state_bytes_avoided,
            "selection_rows_read": 0,
            "selection_rows_cached": 0,
            "cached_row_keyspace": "registry development indices only",
        }

    def segment_plan_usage(
        self, formal_cache_projection: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "planned_rows_prepared": int(self.segment_plans_prepared),
            "planned_uses": int(self.segment_plan_uses),
            "checked_fallbacks": int(self.segment_plan_checked_fallbacks),
            "position_digest_checks": int(self.position_digest_checks),
            "position_digest_mismatches": int(self.position_digest_mismatches),
            "cache_stats": self.segment_plan_cache.stats(),
            "formal_cache_projection": formal_cache_projection,
        }


def _balanced_quantiles(
    candidates: Iterable[int],
    rows: Sequence[dict[str, Any]],
    count: int,
) -> np.ndarray:
    values = [int(index) for index in candidates]
    if count <= 0 or len(values) < count:
        raise ValueError("not enough candidates for balanced selection")
    # Asking for the entire registered split is an exact census, not a
    # root-balanced subsample.  Preserve every row even when split hashing made
    # the logical-root counts unequal.
    if count == len(values):
        if len(set(values)) != count:
            raise ValueError("full split census contains duplicate candidates")
        return np.asarray(sorted(values), dtype=np.int64)
    groups: dict[str, list[int]] = {}
    for index in values:
        groups.setdefault(_root_key(rows[index]), []).append(index)
    roots = sorted(groups)
    allocation = {root: count // len(roots) for root in roots}
    for root in roots[: count % len(roots)]:
        allocation[root] += 1
    selected: list[int] = []
    for root in roots:
        ordered = sorted(
            groups[root],
            key=lambda index: (
                int(rows[index]["variant_count"]),
                str(rows[index]["canonical_id"]),
            ),
        )
        n_take = allocation[root]
        if len(ordered) < n_take:
            raise ValueError("one logical root cannot satisfy balanced allocation")
        positions = np.linspace(0, len(ordered) - 1, n_take, dtype=np.int64)
        selected.extend(ordered[int(position)] for position in positions)
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("balanced quantile selection is not exact/unique")
    return np.asarray(sorted(selected), dtype=np.int64)


def _select_stage_rows(
    stage: str,
    rows: Sequence[dict[str, Any]],
    registry_to_cache: np.ndarray,
    fit: np.ndarray,
    inner: np.ndarray,
    evaluation: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (
        registry_to_cache.shape != (len(rows),)
        or np.any(registry_to_cache < 0)
        or np.any(registry_to_cache >= len(rows))
    ):
        raise ValueError("registry/cache mapping indices are invalid")
    reverse = np.full(len(rows), -1, dtype=np.int64)
    reverse[registry_to_cache] = np.arange(len(rows), dtype=np.int64)
    if np.any(reverse < 0):
        raise ValueError("registry/cache mapping is not bijective")
    if stage in {"one_seed", "formal"}:
        return reverse[fit], reverse[inner], reverse[evaluation]
    if stage == "pilot":
        return (
            _balanced_quantiles(reverse[fit], rows, args.pilot_fit_rows),
            _balanced_quantiles(reverse[inner], rows, args.pilot_inner_rows),
            _balanced_quantiles(
                reverse[evaluation], rows, args.pilot_evaluation_rows
            ),
        )
    counts = np.asarray([int(row["variant_count"]) for row in rows], dtype=np.int64)
    low = int(np.argmin(counts))
    high = int(np.argmax(counts))
    if low == high:
        raise RuntimeError("minimum and maximum variant rows are identical")
    return np.asarray((low, high), dtype=np.int64), np.empty(0, np.int64), np.empty(0, np.int64)


def _epoch_row_order(indices: np.ndarray, *, seed: int, epoch: int) -> np.ndarray:
    """Preserve the frozen reference row permutation exactly.

    Artifact locality is handled only by caching/lookahead.  Source/root is
    never allowed to reorder SGD updates.
    """

    return np.random.default_rng(seed + epoch).permutation(indices)


class _ExecutionAudit:
    """Streaming hashes of rows actually trained and optimiser boundaries."""

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self.rows = rows
        self._row_digest = hashlib.sha256()
        self._source_digest = hashlib.sha256()
        self._boundary_digest = hashlib.sha256()
        self.training_yields = 0
        self.optimizer_steps = 0

    @staticmethod
    def _update(digest: Any, record: dict[str, Any]) -> None:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")

    def record_training_yield(
        self, *, arm: str, epoch: int, step: int, staged: _StagedRawRow
    ) -> None:
        row_index = int(staged.audit["registry_index"])
        row = self.rows[row_index]
        canonical_id = str(staged.audit["canonical_id"])
        if canonical_id != str(row.get("canonical_id", "")):
            raise ValueError("actual training yield differs from registry identity")
        artifact = row.get("source_artifact")
        if not isinstance(artifact, dict):
            raise ValueError("actual training yield lacks bound source artifact")
        self._update(
            self._row_digest,
            {
                "arm": arm,
                "epoch": int(epoch),
                "step": int(step),
                "registry_index": row_index,
                "canonical_id": canonical_id,
            },
        )
        self._update(
            self._source_digest,
            {
                "arm": arm,
                "epoch": int(epoch),
                "step": int(step),
                "canonical_id": canonical_id,
                "logical_dataset_root": str(row.get("logical_dataset_root", "")),
                "physical_dataset_root": str(row.get("physical_dataset_root", "")),
                "source_local_row": int(row.get("source_local_row", -1)),
                "source_npz_path": str(artifact.get("npz_path", "")),
                "source_npz_sha256": str(artifact.get("npz_sha256", "")),
                "raw_observation_sha256": str(
                    row.get("raw_observation_sha256", "")
                ),
            },
        )
        self.training_yields += 1

    def record_optimizer_boundary(
        self, *, arm: str, epoch: int, step: int, staged: _StagedRawRow
    ) -> None:
        self._update(
            self._boundary_digest,
            {
                "arm": arm,
                "epoch": int(epoch),
                "optimizer_step_after_row": int(step),
                "canonical_id": str(staged.audit["canonical_id"]),
            },
        )
        self.optimizer_steps += 1

    def replay_completed_epochs(
        self,
        *,
        arm: str,
        seed: int,
        fit_indices: np.ndarray,
        accumulate_rows: int,
        through_epoch: int,
    ) -> None:
        """Rebuild the streaming digests for epochs already on disk.

        Every field these hashes consume is a pure function of ``(arm, epoch,
        step, row)``, and the row order is a pure function of ``(seed, epoch)``
        -- ``_epoch_row_order`` takes no carried state.  So the record sequence
        of a finished epoch can be regenerated exactly without touching the GPU,
        which is what lets a resumed run produce byte-identical receipts instead
        of a second, incomparable hash scheme.  Saving hashlib state would have
        been the alternative; hashlib objects do not serialise, and changing to
        a per-epoch Merkle chain would have changed every published digest.
        """

        if through_epoch <= 0:
            return
        for epoch in range(1, through_epoch + 1):
            order = _epoch_row_order(fit_indices, seed=seed, epoch=epoch)
            for step, row_index in enumerate(order, start=1):
                row = self.rows[int(row_index)]
                canonical_id = str(row.get("canonical_id", ""))
                artifact = row.get("source_artifact")
                if not isinstance(artifact, dict):
                    raise ValueError("replayed row lacks bound source artifact")
                self._update(
                    self._row_digest,
                    {
                        "arm": arm,
                        "epoch": int(epoch),
                        "step": int(step),
                        "registry_index": int(row_index),
                        "canonical_id": canonical_id,
                    },
                )
                self._update(
                    self._source_digest,
                    {
                        "arm": arm,
                        "epoch": int(epoch),
                        "step": int(step),
                        "canonical_id": canonical_id,
                        "logical_dataset_root": str(row.get("logical_dataset_root", "")),
                        "physical_dataset_root": str(
                            row.get("physical_dataset_root", "")
                        ),
                        "source_local_row": int(row.get("source_local_row", -1)),
                        "source_npz_path": str(artifact.get("npz_path", "")),
                        "source_npz_sha256": str(artifact.get("npz_sha256", "")),
                        "raw_observation_sha256": str(
                            row.get("raw_observation_sha256", "")
                        ),
                    },
                )
                self.training_yields += 1
                if step % accumulate_rows == 0 or step == len(order):
                    self._update(
                        self._boundary_digest,
                        {
                            "arm": arm,
                            "epoch": int(epoch),
                            "optimizer_step_after_row": int(step),
                            "canonical_id": canonical_id,
                        },
                    )
                    self.optimizer_steps += 1

    def report(self) -> dict[str, Any]:
        return {
            "training_yields": int(self.training_yields),
            "optimizer_steps": int(self.optimizer_steps),
            "training_row_order_sha256": self._row_digest.hexdigest(),
            "training_source_order_sha256": self._source_digest.hexdigest(),
            "optimizer_step_boundaries_sha256": self._boundary_digest.hexdigest(),
            "artifact_locality_reordering_allowed": False,
        }


def _full_cohort_cache_projection(
    model: Any,
    rows: Sequence[dict[str, Any]],
    loader: _ArtifactAwareRawLoader,
    *,
    raw_index_sha256: str | None = None,
    expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
    accept_cache_refill: bool = False,
    packed_hap_bytes: int | None = None,
) -> dict[str, Any]:
    """Registry-only counts plus execution-specific persistent-cache bounds.

    By default a corpus that does not fit its persistent caches stops the run.
    That gate was written for a development corpus that fits in 36 GiB, where
    holding it costs nothing and re-reading it is pure waste.  A production
    cohort does not fit in any machine we can rent -- 31,500 rows of 30 Mb come
    to roughly 400 GB across the five caches -- so on that cohort the choice is
    not "fit or thrash" but "hold it or re-read it", and re-reading is the only
    option there is.

    ``accept_cache_refill`` is how a caller says so, and it is deliberately not
    the default: it makes the run re-read the corpus once per epoch, and the
    projection then records what that costs instead of asserting it away.  The
    cost is small -- each row is visited once per epoch either way, so the cache
    only ever saved the *second* epoch's reads -- but it is a real cost and it
    belongs in the receipt, not in a comment.
    """

    if len(rows) != int(expected_rows):
        raise ValueError(
            "cache projection requires every row of the bound registry"
        )
    total_sites = 0
    total_missing_calls = 0
    unique_source_artifacts = len({_artifact_key(row) for row in rows})
    artifact_bytes = 0
    map_one_kind_bytes = 0
    map_one_kind_interval_units = 0
    exposure_one_kind_bytes = 0
    exposure_one_kind_region_units = 0
    segment_bytes = 0
    for row in rows:
        sites = int(row.get("variant_count", -1))
        missing = int(row.get("missing_call_count", -1))
        sequence_length_bp = int(row.get("sequence_length_bp", -1))
        offsets = row.get("offsets")
        if sites <= 0 or missing < 0 or sequence_length_bp <= 0 or not isinstance(
            offsets, dict
        ):
            raise ValueError("registry row lacks cache-projection counts")

        def offset_count(name: str) -> int:
            value = offsets.get(name)
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"registry row lacks {name} offsets")
            start, stop = (int(item) for item in value)
            if start < 0 or stop < start:
                raise ValueError(f"registry row has invalid {name} offsets")
            return stop - start

        rec_pos = offset_count("obs_rec_pos")
        rec_rate = offset_count("obs_rec_rate")
        mut_pos = offset_count("obs_mut_pos")
        mut_rate = offset_count("obs_mut_rate")
        if rec_pos != rec_rate + 1 or mut_pos != mut_rate + 1:
            raise ValueError("registry map offsets violate position/rate shape")
        row_map_bytes = 8 * (rec_pos + rec_rate + mut_pos + mut_rate)
        total_sites += sites
        total_missing_calls += missing
        # Exact persistent artifact slices: uint32[V], uint8[V,B],
        # uint32[M], plus the four frozen float32 raw-map arrays.  Derived map
        # cache entries are explicitly copied to float64 and charged below.
        # B is the row's own packed width.  It used to fall back to the
        # development corpus's eight bytes whenever a registry did not record
        # it, which silently halved the dominant term for the sixteen-byte
        # production rows: the projection then certified a budget the run would
        # thrash straight through.  A projection that under-charges is worse
        # than none, and this one is registry-only by design -- it may not open
        # an artifact to find out -- so a registry that does not state the
        # width makes the caller state it, and guessing is not on the menu.
        width = row.get("packed_hap_bytes", packed_hap_bytes)
        if width is None:
            raise ValueError(
                "registry rows do not record packed_hap_bytes and none was "
                "supplied; pass the corpus's packed width (--packed-hap-bytes) "
                "rather than let the projection assume the development "
                "corpus's eight"
            )
        row_packed_hap_bytes = int(width)
        if row_packed_hap_bytes < 1:
            raise ValueError("registry row has an invalid packed width")
        artifact_bytes += (
            (4 + row_packed_hap_bytes) * sites + 4 * missing + row_map_bytes // 2
        )
        map_one_kind_bytes += row_map_bytes
        map_one_kind_interval_units += rec_rate + mut_rate
        row_regions = math.ceil(sequence_length_bp / int(model.macro_width_bp))
        exposure_one_kind_bytes += row_regions * 8 * 4
        exposure_one_kind_region_units += row_regions
        segment_bytes += int(
            model.segment_plan_upper_bound_nbytes(sites, sequence_length_bp)
        )

    execution_chunk_sites = int(getattr(model, "chunk_sites", 0))
    if execution_chunk_sites <= 0:
        raise ValueError("model chunk_sites must be a positive integer")
    projections = {
        "accounting_basis": (
            "LRU charged array/payload bytes; frozen float32 raw maps are "
            "enforced in the artifact cache and derived maps are float64 copies"
        ),
        "projected_development_rows": len(rows),
        "packed_hap_bytes": row_packed_hap_bytes,
        "projected_unique_source_artifacts": unique_source_artifacts,
        "projected_sites": total_sites,
        "projected_missing_calls": total_missing_calls,
        "artifact_exact_bytes": artifact_bytes,
        "control_shift_per_kind_exact_bytes": total_sites,
        "segment_plan_upper_bound_bytes": segment_bytes,
        "map_both_kinds_upper_bound_bytes": 2 * map_one_kind_bytes,
        "map_both_kinds_entries": 2 * len(rows),
        "map_both_kinds_interval_units": 2 * map_one_kind_interval_units,
        "map_exposure_both_kinds_upper_bound_bytes": 2
        * exposure_one_kind_bytes,
        "map_exposure_both_kinds_entries": 2 * len(rows),
        "map_exposure_both_kinds_region_units": 2
        * exposure_one_kind_region_units,
        "selection_rows_read": 0,
        "selection_access_count": 0,
    }
    cache_bindings = {
        "artifact": (
            artifact_bytes,
            total_sites,
            loader.artifact_cache.max_bytes,
            loader.artifact_cache.max_sites,
        ),
        "control_shift_per_kind": (
            total_sites,
            total_sites,
            loader.shift_cache.max_bytes,
            loader.shift_cache.max_sites,
        ),
        "segment_plan": (
            segment_bytes,
            total_sites,
            loader.segment_plan_cache.max_bytes,
            loader.segment_plan_cache.max_sites,
        ),
        "map_both_kinds": (
            2 * map_one_kind_bytes,
            2 * map_one_kind_interval_units,
            loader.map_cache.max_bytes,
            loader.map_cache.max_sites,
        ),
        "map_exposure_both_kinds": (
            2 * exposure_one_kind_bytes,
            2 * exposure_one_kind_region_units,
            loader.exposure_cache.max_bytes,
            loader.exposure_cache.max_sites,
        ),
    }
    fits: dict[str, bool] = {}
    for label, (
        projected_bytes,
        projected_sites,
        max_bytes,
        max_sites,
    ) in cache_bindings.items():
        fits[label] = projected_bytes <= max_bytes and projected_sites <= max_sites
    projections["cache_limits"] = {
        label: {
            "projected_bytes": int(values[0]),
            "projected_sites": int(values[1]),
            "cache_max_bytes": int(values[2]),
            "cache_max_sites": int(values[3]),
            "fits_without_cyclic_thrashing": fits[label],
        }
        for label, values in cache_bindings.items()
    }
    projections["all_persistent_projections_fit"] = all(fits.values())
    projections["accepted_cache_refill"] = bool(accept_cache_refill)
    refill_bytes = sum(
        max(0, int(values[0]) - int(values[2]))
        for label, values in cache_bindings.items()
        if not fits[label]
    )
    projections["projected_cache_refill_bytes_per_epoch"] = int(refill_bytes)
    if not projections["all_persistent_projections_fit"] and accept_cache_refill:
        print(
            json.dumps(
                {
                    "accepted_cache_refill": True,
                    "caches_that_refill": sorted(
                        label for label, value in fits.items() if not value
                    ),
                    "refill_bytes_per_epoch": int(refill_bytes),
                    "refill_gib_per_epoch": round(refill_bytes / (1024 ** 3), 1),
                    "note": (
                        "the corpus is re-read from disk each epoch; a row is "
                        "visited once per epoch either way, so this is the "
                        "second and later epochs' reads, not extra work within "
                        "an epoch"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not projections["all_persistent_projections_fit"] and not accept_cache_refill:
        # Name the shortfall, not just the label.  Every one of these is fixed
        # either by a larger --host-cache-gib/--host-cache-sites or by a budget
        # split that matches the corpus's demand profile, and neither choice is
        # possible from a message that says only which cache lost.
        shortfall = {
            label: {
                "projected_bytes": int(values[0]),
                "cache_max_bytes": int(values[2]),
                "bytes_short": max(0, int(values[0]) - int(values[2])),
                "projected_sites": int(values[1]),
                "cache_max_sites": int(values[3]),
                "sites_short": max(0, int(values[1]) - int(values[3])),
                "share_of_budget_needed": round(
                    int(values[0]) / max(1, int(loader.host_max_bytes)), 4
                ),
            }
            for label, values in sorted(cache_bindings.items())
            if not fits[label]
        }
        raise RuntimeError(
            "full-cohort cache projection does not fit: "
            + json.dumps(
                {
                    "host_max_bytes": int(loader.host_max_bytes),
                    "host_max_sites": int(getattr(loader, "host_max_sites", 0)),
                    "budget_fractions": dict(
                        getattr(loader, "budget_fractions", {})
                    ),
                    "shortfall": shortfall,
                },
                sort_keys=True,
            )
        )
    host_components = {
        "artifact_cache": int(loader.artifact_cache.max_bytes),
        "shift_cache": int(loader.shift_cache.max_bytes),
        "segment_plan_cache": int(loader.segment_plan_cache.max_bytes),
        "map_cache": int(loader.map_cache.max_bytes),
        "map_exposure_cache": int(loader.exposure_cache.max_bytes),
        "pinned_prefetch": int(loader.prefetch_budget.max_bytes),
        "transient_artifact": int(loader.transient_max_bytes),
    }
    allocated_host_bytes = sum(host_components.values())
    host_max_bytes = int(loader.host_max_bytes)
    if (
        allocated_host_bytes > host_max_bytes
        or allocated_host_bytes / host_max_bytes > _MAX_ALLOCATED_HOST_FRACTION
    ):
        raise RuntimeError("actual host cache/working budgets exceed the 90% envelope")
    projections["host_budget_contract"] = {
        "host_max_bytes": host_max_bytes,
        "component_max_bytes": host_components,
        "allocated_bytes": allocated_host_bytes,
        "allocated_fraction": allocated_host_bytes / host_max_bytes,
        "maximum_allocated_fraction": _MAX_ALLOCATED_HOST_FRACTION,
        "unallocated_headroom_bytes": host_max_bytes - allocated_host_bytes,
        "unallocated_headroom_fraction": (
            (host_max_bytes - allocated_host_bytes) / host_max_bytes
        ),
        "within_envelope": True,
    }
    projections["segment_plan_execution_projection"] = {
        "execution_dependent": True,
        "chunk_sites": execution_chunk_sites,
        "current_upper_bound_bytes": int(segment_bytes),
        "cache_max_bytes": int(loader.segment_plan_cache.max_bytes),
        "cache_max_sites": int(loader.segment_plan_cache.max_sites),
        "projected_sites": int(total_sites),
        "fits_current_cache": bool(fits["segment_plan"]),
    }
    # Every bound below is a property of one specific registry -- the frozen
    # 7,133-row development corpus -- recorded so that registry always projects
    # the same numbers.  It is checked when the run is bound to that registry
    # and is silent otherwise: a different cohort has different totals by
    # construction, and demanding it match this inventory would not be a gate
    # on that cohort but a claim that it is this one.
    inventory = dict(_FROZEN_DEVELOPMENT_INVENTORY)
    on_frozen_registry = raw_index_sha256 == inventory["raw_index_sha256"]
    projections["frozen_development_inventory_checked"] = bool(on_frozen_registry)
    if raw_index_sha256 is not None and not on_frozen_registry:
        projections["registry_raw_index_sha256"] = str(raw_index_sha256)
    if on_frozen_registry:
        reference_chunk_sites = int(
            inventory["reference_segment_plan_chunk_sites"]
        )
        reference_segment_bytes = int(
            inventory["reference_segment_plan_upper_bound_bytes"]
        )
        if reference_chunk_sites <= 0 or reference_segment_bytes <= 0:
            raise ValueError("frozen segment-plan reference must be positive")
        expected_projection = {
            "projected_development_rows": inventory["development_rows"],
            "projected_unique_source_artifacts": inventory[
                "unique_source_artifacts"
            ],
            "projected_sites": inventory["total_variant_sites"],
            "projected_missing_calls": inventory["total_missing_calls"],
            "artifact_exact_bytes": inventory["artifact_exact_bytes"],
            "map_both_kinds_upper_bound_bytes": inventory[
                "map_both_kinds_bytes"
            ],
            "map_both_kinds_interval_units": inventory[
                "map_both_kinds_interval_units"
            ],
        }
        if any(
            projections.get(key) != value
            for key, value in expected_projection.items()
        ):
            raise ValueError(
                "registry-only cache projection differs from frozen 7,133-row inventory"
            )
        reference_leaf_width_bp = int(
            inventory["reference_segment_plan_leaf_width_bp"]
        )
        reference_macro_levels = int(inventory["reference_segment_plan_macro_levels"])
        on_reference_plan_shape = (
            execution_chunk_sites == reference_chunk_sites
            and int(model.leaf_width_bp) == reference_leaf_width_bp
            and int(model.macro_levels) == reference_macro_levels
        )
        if on_reference_plan_shape and segment_bytes != reference_segment_bytes:
            raise ValueError(
                "reference-shape segment-plan projection differs from frozen bound"
            )
        reference_macro_width_bp = int(
            inventory["reference_map_exposure_macro_width_bp"]
        )
        execution_macro_width_bp = int(model.macro_width_bp)
        if execution_macro_width_bp == reference_macro_width_bp and (
            projections["map_exposure_both_kinds_region_units"]
            != inventory["reference_map_exposure_both_kinds_region_units"]
            or projections["map_exposure_both_kinds_upper_bound_bytes"]
            != inventory["reference_map_exposure_both_kinds_bytes"]
        ):
            raise ValueError(
                "reference-grid map-exposure projection differs from frozen bound"
            )
        projections["map_exposure_execution_projection"] = {
            "execution_dependent": True,
            "macro_width_bp": execution_macro_width_bp,
            "reference_macro_width_bp": reference_macro_width_bp,
            "reference_region_units": int(
                inventory["reference_map_exposure_both_kinds_region_units"]
            ),
            "reference_upper_bound_bytes": int(
                inventory["reference_map_exposure_both_kinds_bytes"]
            ),
            "reference_bound_checked": (
                execution_macro_width_bp == reference_macro_width_bp
            ),
        }
        projections["segment_plan_execution_projection"].update(
            {
                "reference_chunk_sites": reference_chunk_sites,
                "reference_leaf_width_bp": reference_leaf_width_bp,
                "reference_macro_levels": reference_macro_levels,
                "leaf_width_bp": int(model.leaf_width_bp),
                "macro_levels": int(model.macro_levels),
                "reference_upper_bound_bytes": reference_segment_bytes,
                "reference_bound_checked": (
                    on_reference_plan_shape
                ),
                "registry_invariants_held_constant": True,
            }
        )
        max_transient_bytes = int(inventory["max_transient_bytes"])
        max_transient_sites = int(inventory["max_transient_sites"])
        if (
            max_transient_bytes > loader.transient_max_bytes
            or max_transient_sites > loader.transient_max_sites
        ):
            raise RuntimeError("frozen maximum source artifact exceeds transient budget")
        projections["source_dtype_and_transient_inventory"] = {
            **inventory,
            "transient_cache_max_bytes": int(loader.transient_max_bytes),
            "transient_cache_headroom_bytes": int(
                loader.transient_max_bytes - max_transient_bytes
            ),
            "transient_cache_max_sites": int(loader.transient_max_sites),
            "transient_cache_headroom_sites": int(
                loader.transient_max_sites - max_transient_sites
            ),
            "max_transient_budget_fraction_used": (
                max_transient_bytes / loader.transient_max_bytes
            ),
            "inventory_bound_to_exact_raw_index": True,
        }
    return projections


def _audit_persistent_cache_terminal_contract(
    loader_stats: dict[str, Any],
    projection: dict[str, Any],
    *,
    stage: str,
    expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
) -> dict[str, Any]:
    """Validate measured persistent-cache accounting against its projection."""

    if stage not in {"smoke", "pilot", "one_seed", "formal"}:
        raise ValueError(f"unknown persistent-cache stage {stage!r}")
    contract_stage = "formal" if stage == "one_seed" else stage

    def counter(value: Any, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be an integer counter")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} must be an integer counter") from error
        if parsed < 0 or value != parsed:
            raise ValueError(f"{label} must be an exact nonnegative integer")
        return parsed

    def measured(name: str) -> dict[str, int]:
        raw = loader_stats.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"loader terminal report lacks {name}")
        result = {
            field: counter(raw.get(field), f"{name} {field}")
            for field in (
                "entries",
                "bytes",
                "sites",
                "max_bytes",
                "max_sites",
                "hits",
                "misses",
                "evictions",
                "rejected_oversize",
            )
        }
        if (
            result["bytes"] > result["max_bytes"]
            or result["sites"] > result["max_sites"]
        ):
            raise ValueError(f"{name} exceeded its persistent cache budget")
        return result

    artifact = measured("artifact_cache")
    shifts = measured("shift_cache")
    maps = measured("map_cache")
    exposures = measured("map_exposure_cache")
    rows_loaded = counter(loader_stats.get("rows_loaded"), "rows_loaded")
    artifact_opens = counter(loader_stats.get("artifact_opens"), "artifact_opens")
    map_requests = counter(loader_stats.get("map_requests"), "map_requests")
    exposure_requests = counter(
        loader_stats.get("map_exposure_requests"), "map_exposure_requests"
    )
    projected_rows = counter(
        projection.get("projected_development_rows"),
        "projected development rows",
    )
    unique_artifacts = counter(
        projection.get("projected_unique_source_artifacts"),
        "projected unique source artifacts",
    )
    artifact_bytes = counter(
        projection.get("artifact_exact_bytes"), "projected artifact bytes"
    )
    projected_sites = counter(
        projection.get("projected_sites"), "projected artifact sites"
    )
    limits = projection.get("cache_limits")
    if projected_rows != int(expected_rows) or not isinstance(limits, dict):
        raise ValueError(
            "persistent-cache projection does not cover the bound registry"
        )
    projected_limits: dict[str, dict[str, int]] = {}
    for projection_name, runtime in (
        ("artifact", artifact),
        ("map_both_kinds", maps),
        ("map_exposure_both_kinds", exposures),
    ):
        projected = limits.get(projection_name)
        # A cache that refills is acceptable only because the projection said
        # so out loud and the run was told to accept it; the audit re-reads
        # that declaration rather than trusting the per-cache flag alone.
        refill_accepted = projection.get("accepted_cache_refill") is True
        if (
            not isinstance(projected, dict)
            or (
                projected.get("fits_without_cyclic_thrashing") is not True
                and not refill_accepted
            )
            or counter(
                projected.get("cache_max_bytes"),
                f"{projection_name} projected max bytes",
            )
            != runtime["max_bytes"]
            or counter(
                projected.get("cache_max_sites"),
                f"{projection_name} projected max sites",
            )
            != runtime["max_sites"]
        ):
            raise ValueError(
                f"{projection_name} runtime budget differs from its projection"
            )
        projected_limits[projection_name] = {
            "bytes": counter(
                projected.get("projected_bytes"),
                f"{projection_name} projected bytes",
            ),
            "sites": counter(
                projected.get("projected_sites"),
                f"{projection_name} projected sites",
            ),
        }
        if (
            projected_limits[projection_name]["bytes"] > runtime["max_bytes"]
            or projected_limits[projection_name]["sites"] > runtime["max_sites"]
        ):
            raise ValueError(f"{projection_name} does not fit its projected budget")
    if (
        artifact["entries"] != artifact["misses"]
        or artifact["hits"] != rows_loaded - artifact["misses"]
        or artifact["evictions"] != 0
        or artifact["rejected_oversize"] != 0
    ):
        raise ValueError(f"{stage} packed-artifact cache accounting/churn failed")
    if (
        maps["entries"] != maps["misses"]
        or maps["entries"] > 2 * rows_loaded
        or maps["hits"] != map_requests - maps["misses"]
        or maps["evictions"] != 0
        or maps["rejected_oversize"] != 0
        or exposures["entries"] != exposures["misses"]
        or exposures["entries"] > 2 * rows_loaded
        or exposures["hits"] != exposure_requests - exposures["misses"]
        or exposures["evictions"] != 0
        or exposures["rejected_oversize"] != 0
    ):
        raise ValueError(f"{stage} map/exposure cache accounting/churn failed")
    if shifts["rejected_oversize"] != 0:
        raise ValueError(f"{stage} shift cache rejected an oversize control row")
    expected_map_entries = counter(
        projection.get("map_both_kinds_entries"),
        "projected map entries",
    )
    expected_exposure_entries = counter(
        projection.get("map_exposure_both_kinds_entries"),
        "projected map-exposure entries",
    )
    if (
        unique_artifacts <= 0
        or unique_artifacts > projected_rows
        or expected_map_entries != 2 * projected_rows
        or expected_exposure_entries != 2 * projected_rows
    ):
        raise ValueError("persistent-cache projected entry counts are invalid")
    if contract_stage == "formal" and (
        rows_loaded <= 0
        or artifact["entries"] != unique_artifacts
        or artifact["misses"] != unique_artifacts
        or artifact_opens != unique_artifacts
        or artifact["hits"] != rows_loaded - unique_artifacts
        or artifact["evictions"] != 0
        or artifact["rejected_oversize"] != 0
        or artifact["bytes"] != artifact_bytes
        or artifact["sites"] != projected_sites
        or maps["entries"] != expected_map_entries
        or maps["misses"] != expected_map_entries
        or maps["bytes"] != projected_limits["map_both_kinds"]["bytes"]
        or maps["sites"] != projected_limits["map_both_kinds"]["sites"]
        or exposures["entries"] != expected_exposure_entries
        or exposures["misses"] != expected_exposure_entries
        or exposures["bytes"]
        != projected_limits["map_exposure_both_kinds"]["bytes"]
        or exposures["sites"]
        != projected_limits["map_exposure_both_kinds"]["sites"]
    ):
        raise ValueError("formal persistent cache terminal contract failed")
    return {
        "stage": contract_stage,
        "packed_artifact": {
            "exact_full_cohort_residency_required": contract_stage == "formal",
            "unique_source_artifacts": unique_artifacts,
            "requests": rows_loaded,
            "entries": artifact["entries"],
            "misses": artifact["misses"],
            "hits": artifact["hits"],
            "zero_evictions": artifact["evictions"] == 0,
            "zero_rejected_oversize": artifact["rejected_oversize"] == 0,
            "bytes_and_sites_within_budget": True,
            "formal_exact_bytes_and_sites": contract_stage != "formal"
            or (
                artifact["bytes"] == artifact_bytes
                and artifact["sites"] == projected_sites
            ),
        },
        "map_cache": {
            "projection_fit_verified": True,
            "zero_rejected_oversize": True,
            "zero_evictions": True,
            "entries_equal_misses": True,
            "requests": map_requests,
            "hits_equal_requests_minus_misses": True,
            "formal_exact_two_kind_residency": contract_stage != "formal"
            or maps["entries"] == expected_map_entries,
        },
        "map_exposure_cache": {
            "projection_fit_verified": True,
            "zero_rejected_oversize": True,
            "zero_evictions": True,
            "entries_equal_misses": True,
            "requests": exposure_requests,
            "hits_equal_requests_minus_misses": True,
            "formal_exact_two_kind_residency": contract_stage != "formal"
            or exposures["entries"] == expected_exposure_entries,
        },
        "shift_cache": {
            "stats": shifts,
            "zero_rejected_oversize": True,
            "evictions_allowed_across_arm_lifetimes": True,
        },
    }


def _forward_row(
    torch: Any,
    model: Any,
    *,
    staged: _StagedRawRow,
    residual_anchor: Any,
    device: Any,
    loader: _ArtifactAwareRawLoader,
) -> tuple[Any, Any]:
    loader.record_segment_plan_use(staged, model)
    positions = staged.positions.to(device, non_blocking=True).to(dtype=torch.int64)
    states = transfer_packed_and_decode(
        staged.genotype_packed,
        staged.missing_flat_idx,
        device=device,
        chunk_sites=int(model.chunk_sites),
        non_blocking=True,
        pin_memory=True,
        require_deterministic_cuda=True,
        values_prevalidated=staged.values_prevalidated,
    )
    if staged.diploid_columns is not None:
        states = states.index_select(
            1,
            torch.as_tensor(
                staged.diploid_columns, dtype=torch.int64, device=states.device
            ),
        )
    if staged.state_control_arm is not None:
        shifts = (
            transfer_control_shifts(
                staged.control_shifts,
                device=device,
                non_blocking=True,
                pin_memory=True,
                values_prevalidated=staged.values_prevalidated,
            )
            if staged.control_shifts is not None
            else None
        )
        states = apply_diploid_control_torch(
            states,
            staged.state_control_arm,
            shifts=shifts,
            chunk_sites=int(model.chunk_sites),
            require_deterministic_cuda=True,
            values_prevalidated=staged.values_prevalidated,
        )
    context = staged.context.to(device, non_blocking=True)
    target = staged.target.to(device, non_blocking=True)
    nw = staged.nw_log10.to(device, non_blocking=True)
    map_exposure = staged.map_exposure.to(device, non_blocking=True)
    # Compact plans contain host control descriptors only.  CUDA derives the
    # unique/count/start tensors; there is deliberately no plan H2D payload.
    segment_plan = staged.segment_plan.plan
    value_control_kwargs: dict[str, Any] = {
        "map_feature_mode": staged.map_feature_mode,
        "raw_signal_mode": staged.raw_signal_mode,
        "physics_bias_mode": staged.physics_bias_mode,
    }
    if staged.site_recombination_mode != "disabled":
        if any(
            value is None
            for value in (
                staged.recombination_positions,
                staged.recombination_rates,
                staged.recombination_prefix_mass,
            )
        ):
            raise ValueError("map-rich staged row lacks recombination vectors")
        recombination_positions = staged.recombination_positions.to(
            device, non_blocking=True
        )
        recombination_rates = staged.recombination_rates.to(
            device, non_blocking=True
        )
        recombination_prefix_mass = staged.recombination_prefix_mass.to(
            device, non_blocking=True
        )
        value_control_kwargs.update({
            "site_recombination_mode": staged.site_recombination_mode,
            "site_recombination_fusion_mode": (
                staged.site_recombination_fusion_mode
            ),
            "recombination_prefix_mass": recombination_prefix_mass,
        })
    else:
        recombination_positions = None
        recombination_rates = None
    residual = model(
        positions,
        states,
        recombination_positions,
        recombination_rates,
        None,
        None,
        context,
        sequence_length_bp=staged.sequence_length_bp,
        long_residual_mode=staged.long_residual_mode,
        map_exposure=map_exposure,
        segment_plan=segment_plan,
        values_prevalidated=staged.values_prevalidated,
        **value_control_kwargs,
    )
    prediction = residual + residual_anchor + nw
    if (
        residual.shape != (EXPECTED_TARGETS,)
        or prediction.shape != (EXPECTED_TARGETS,)
        or int(model.last_sites_consumed) != staged.sites
    ):
        raise RuntimeError("direct64 prediction or all-site consumption is invalid")
    return prediction, target


def _clear_cuda_boundary(torch: Any) -> None:
    """Release allocator state only between arms/stages, never per raw row."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate(
    torch: Any,
    model: Any,
    arm: str,
    indices: np.ndarray,
    *,
    rows: Sequence[dict[str, Any]],
    registry_to_cache: np.ndarray,
    cache: dict[str, np.ndarray],
    residual_anchor: Any,
    args: argparse.Namespace,
    device: Any,
    loader: _ArtifactAwareRawLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int, float]:
    predictions: list[Any] = []
    targets: list[Any] = []
    original_rows: list[int] = []
    canonical_ids: list[str] = []
    sites = 0
    started = time.perf_counter()
    model.eval()
    with torch.no_grad(), loader.fixed_sample_size(
        _measurement_diploids(args)
    ):
        for staged in loader.iter_staged(torch, model, indices, arm):
            prediction, target = _forward_row(
                torch,
                model,
                staged=staged,
                residual_anchor=residual_anchor,
                device=device,
                loader=loader,
            )
            predictions.append(prediction.detach())
            targets.append(target.detach())
            original_rows.append(
                int(cache["selected_indices"][int(staged.audit["cache_index"])])
            )
            canonical_ids.append(str(staged.audit["canonical_id"]))
            sites += int(staged.audit["variant_count"])
            del prediction, target
    prediction_array = torch.stack(predictions).cpu().numpy().astype(np.float32)
    target_array = torch.stack(targets).cpu().numpy().astype(np.float32)
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise RuntimeError("evaluation produced a nonfinite direct64 value")
    return (
        prediction_array,
        target_array,
        np.asarray(original_rows, dtype=np.int64),
        canonical_ids,
        sites,
        time.perf_counter() - started,
    )


def _verify_formal_pilot(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    from .carrier_raw_performance_gate import _audit_segment_plan_cache_contract

    report_path = path / "report.json" if path.is_dir() else path
    done_path = path / "DONE.json" if path.is_dir() else path.with_name("DONE.json")
    report = _read_json(report_path)
    done = _read_json(done_path)
    segment_usage = report.get("segment_plan_usage")
    segment_contract = (
        segment_usage.get("terminal_cache_contract")
        if isinstance(segment_usage, dict)
        else None
    )
    persistent_contract = report.get("persistent_cache_terminal_contract")
    try:
        recomputed_segment_contract = _audit_segment_plan_cache_contract(
            segment_usage, stage="pilot"
        )
        recomputed_persistent_contract = _audit_persistent_cache_terminal_contract(
            report.get("loader_performance"),
            report.get("full_cohort_cache_projection"),
            stage="pilot",
            expected_rows=int(
                report.get("segment_cache_projection", {}).get(
                    "projected_development_rows", EXPECTED_DEVELOPMENT_ROWS
                )
            ),
        )
        recomputed_runtime_contract = _audit_formal_runtime_projection_report(report)
    except ValueError as error:
        raise ValueError(
            "pilot cache/runtime terminal contracts cannot be recomputed"
        ) from error
    runtime_contract = report.get("runtime_projection_contract")
    if (
        report.get("schema") != RUNNER_SCHEMA
        or report.get("stage") != "pilot"
        or report.get("status") != "complete"
        or report.get("directional_only") is not False
        or report.get("formal_authorization_eligible") is not True
        or report.get("formal_feasible") is not True
        or float(report.get("estimated_formal_hours", math.inf)) > args.max_formal_hours
        or report.get("selection_rows_read") != 0
        or report.get("selection_access_count") != 0
        or report.get("all_sites_consumed") is not True
        or report.get("strict_deterministic_cuda") is not True
        or report.get("arms") != list(RAW_ARMS)
        or report.get("model_config") != _model_kwargs(args)
        or report.get("loader_config") != _loader_config(args)
        or int(report.get("formal_epochs", -1)) != args.epochs
        or int(report.get("accumulate_rows", -1)) != args.accumulate_rows
        or float(report.get("learning_rate", math.nan)) != args.learning_rate
        or float(report.get("weight_decay", math.nan)) != args.weight_decay
        or float(report.get("gradient_clip", math.nan)) != args.gradient_clip
        or done.get("schema") != RUNNER_SCHEMA
        or done.get("status") != "complete"
        or done.get("directional_only") is not False
        or done.get("formal_authorization_eligible") is not True
        or done.get("experiment_signature") != report.get("experiment_signature")
        or done.get("selection_rows_read") != 0
        or done.get("selection_access_count") != 0
        or done.get("all_sites_consumed") is not True
        or not isinstance(segment_contract, dict)
        or not isinstance(persistent_contract, dict)
        or segment_contract != recomputed_segment_contract
        or persistent_contract != recomputed_persistent_contract
        or runtime_contract != recomputed_runtime_contract
        or done.get("segment_plan_cache_terminal_contract") != segment_contract
        or done.get("persistent_cache_terminal_contract") != persistent_contract
        or done.get("runtime_projection_contract") != runtime_contract
    ):
        raise ValueError("pilot did not certify the requested formal envelope")
    if done.get("report_sha256") != _sha256(report_path):
        raise ValueError("pilot DONE does not bind report")
    signature = _strict_sha256(
        report.get("experiment_signature"), "pilot experiment signature"
    )
    return {
        "path": str(report_path),
        "sha256": _sha256(report_path),
        "done_path": str(done_path),
        "done_sha256": _sha256(done_path),
        "experiment_signature": signature,
        "loader_config": report["loader_config"],
        "segment_plan_cache_terminal_contract_sha256": _json_contract_sha256(
            segment_contract
        ),
        "persistent_cache_terminal_contract_sha256": _json_contract_sha256(
            persistent_contract
        ),
        "runtime_projection_contract": runtime_contract,
        "runtime_projection_contract_sha256": _json_contract_sha256(
            runtime_contract
        ),
        "maximum_formal_hours": float(args.max_formal_hours),
    }


def _verify_performance_gate(
    path: Path,
    *,
    pilot_provenance: dict[str, Any],
    registry_sha256: dict[str, str],
    cache_sha256: dict[str, str],
) -> dict[str, Any]:
    from .carrier_raw_performance_gate import (
        PERFORMANCE_SCHEMA,
        _audit_gate_job_resource_summary,
        _audit_segment_plan_cache_contract,
        _audit_threshold_policy,
        _threshold_failures,
    )

    report_path = path / "report.json" if path.is_dir() else path
    done_path = path / "DONE.json" if path.is_dir() else path.with_name("DONE.json")
    report = _read_json(report_path)
    done = _read_json(done_path)
    pilot_audit = report.get("pilot_audit")
    reference_audit = report.get("reference_pilot_audit")
    equivalence = report.get("reference_optimized_equivalence")
    runtime_contract = pilot_audit.get("runtime_projection_contract") if isinstance(pilot_audit, dict) else None
    if (
        not isinstance(pilot_audit, dict)
        or not isinstance(reference_audit, dict)
        or not isinstance(equivalence, dict)
        or not isinstance(runtime_contract, dict)
    ):
        raise ValueError("performance gate lacks optimized-pilot receipt closure")
    telemetry_path_raw = report.get("pilot_telemetry_path")
    if not isinstance(telemetry_path_raw, str) or not telemetry_path_raw:
        raise ValueError("performance gate lacks a bound telemetry artifact")
    telemetry_path = Path(telemetry_path_raw)
    if not telemetry_path.is_absolute() or not telemetry_path.is_file():
        raise ValueError("performance gate telemetry artifact is unavailable")
    telemetry_sha256 = _strict_sha256(
        report.get("pilot_telemetry_sha256"),
        "performance gate telemetry SHA256",
    )
    if _sha256(telemetry_path) != telemetry_sha256:
        raise ValueError("performance gate telemetry artifact changed after audit")
    resource_summary_raw = report.get("pilot_resource_summary_path")
    if not isinstance(resource_summary_raw, str) or not resource_summary_raw:
        raise ValueError("performance gate lacks a bound resource summary")
    resource_summary_path = Path(resource_summary_raw)
    if resource_summary_path.is_symlink() or not resource_summary_path.is_file():
        raise ValueError("performance gate resource summary is unavailable")
    resource_summary_path = resource_summary_path.resolve()
    gate_job_resource_audit = _audit_gate_job_resource_summary(
        resource_summary_path,
        telemetry_path,
        expected_pilot_dir=Path(pilot_provenance["path"]).resolve().parent,
    )
    try:
        segment_contract = _audit_segment_plan_cache_contract(
            pilot_audit.get("segment_plan_usage"), stage="pilot"
        )
        persistent_contract = _audit_persistent_cache_terminal_contract(
            pilot_audit.get("loader_performance"),
            pilot_audit.get("full_cohort_cache_projection"),
            stage="pilot",
            expected_rows=int(
                pilot_audit.get("segment_cache_projection", {}).get(
                    "projected_development_rows", EXPECTED_DEVELOPMENT_ROWS
                )
            ),
        )
    except ValueError as error:
        raise ValueError(
            "performance gate lacks a zero-churn optimized-pilot cache audit"
        ) from error
    segment_hash = _json_contract_sha256(segment_contract)
    persistent_hash = _json_contract_sha256(persistent_contract)
    try:
        thresholds = _audit_threshold_policy(report.get("thresholds"))
        if thresholds["maximum_formal_hours"] > float(
            pilot_provenance["maximum_formal_hours"]
        ):
            raise ValueError("gate formal-hours threshold exceeds pilot envelope")
        recomputed_threshold_failures = _threshold_failures(
            reference_audit,
            pilot_audit,
            report.get("pilot_telemetry"),
            report.get("variant_strata"),
            thresholds,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("performance gate telemetry/probes cannot be re-audited") from error
    if (
        report.get("schema") != PERFORMANCE_SCHEMA
        or report.get("status") != "complete"
        or report.get("formal_authorized_by_performance_gate") is not True
        or report.get("scientific_equivalence") is not True
        or report.get("all_sites_consumed") is not True
        or report.get("strict_deterministic_cuda") is not True
        or report.get("selection_rows_read") != 0
        or report.get("selection_access_count") != 0
        or report.get("threshold_failures") != recomputed_threshold_failures
        or recomputed_threshold_failures != []
        or report.get("pilot_resource_summary_sha256")
        != _sha256(resource_summary_path)
        or report.get("performance_gate_job_resource_audit")
        != gate_job_resource_audit
        or report.get("registry_sha256") != registry_sha256
        or report.get("cache_sha256") != cache_sha256
        or pilot_audit.get("pilot_report_sha256") != pilot_provenance["sha256"]
        or pilot_audit.get("experiment_signature")
        != pilot_provenance["experiment_signature"]
        or pilot_audit.get("loader_config") != pilot_provenance["loader_config"]
        or pilot_audit.get("segment_plan_usage", {}).get(
            "terminal_cache_contract"
        )
        != segment_contract
        or pilot_audit.get("persistent_cache_terminal_contract")
        != persistent_contract
        or pilot_audit.get("segment_plan_cache_terminal_contract_sha256")
        != segment_hash
        or pilot_audit.get("persistent_cache_terminal_contract_sha256")
        != persistent_hash
        or segment_hash
        != pilot_provenance["segment_plan_cache_terminal_contract_sha256"]
        or persistent_hash
        != pilot_provenance["persistent_cache_terminal_contract_sha256"]
        or runtime_contract != pilot_provenance["runtime_projection_contract"]
        or report.get("pilot_formal_runtime_projection_sha256")
        != runtime_contract.get("projection_sha256")
        or float(
            report.get("pilot_guarded_projected_formal_seconds", math.nan)
        )
        != float(runtime_contract.get("projected_formal_total_seconds", math.nan))
        or equivalence.get("optimized_pilot_report_sha256")
        != pilot_provenance["sha256"]
        or equivalence.get("reference_pilot_report_sha256")
        != reference_audit.get("pilot_report_sha256")
        or equivalence.get("status") != "pass"
        or equivalence.get("scientific_model_config_equal") is not True
        or equivalence.get("artifact_locality_reordering_allowed") is not False
        or equivalence.get("paired_evaluation_identity_target_equal") is not True
        or equivalence.get("selection_rows_read") != 0
        or equivalence.get("selection_access_count") != 0
        or done.get("schema") != report.get("schema")
        or done.get("status") != "complete"
        or done.get("report_sha256") != _sha256(report_path)
        or done.get("pilot_report_sha256") != pilot_provenance["sha256"]
        or done.get("reference_pilot_report_sha256")
        != reference_audit.get("pilot_report_sha256")
        or done.get("pilot_telemetry_sha256") != telemetry_sha256
        or done.get("pilot_resource_summary_sha256")
        != _sha256(resource_summary_path)
        or done.get("selection_rows_read") != 0
        or done.get("selection_access_count") != 0
        or done.get("all_sites_consumed") is not True
        or done.get("strict_deterministic_cuda") is not True
        or done.get("formal_authorized_by_performance_gate") is not True
        or done.get("scientific_equivalence") is not True
        or done.get("pilot_segment_plan_cache_terminal_contract_sha256")
        != segment_hash
        or done.get("pilot_persistent_cache_terminal_contract_sha256")
        != persistent_hash
        or done.get("pilot_formal_runtime_projection_sha256")
        != runtime_contract.get("projection_sha256")
        or done.get("pilot_guarded_projected_formal_seconds")
        != runtime_contract.get("projected_formal_total_seconds")
    ):
        raise ValueError("performance gate did not authorize this exact optimized pilot")
    return {
        "path": str(report_path),
        "sha256": _sha256(report_path),
        "done_path": str(done_path),
        "done_sha256": _sha256(done_path),
        "pilot_report_sha256": pilot_provenance["sha256"],
        "pilot_experiment_signature": pilot_provenance["experiment_signature"],
        "pilot_segment_plan_cache_terminal_contract_sha256": segment_hash,
        "pilot_persistent_cache_terminal_contract_sha256": persistent_hash,
        "pilot_runtime_projection_contract_sha256": _json_contract_sha256(
            runtime_contract
        ),
        "pilot_resource_summary_sha256": _sha256(resource_summary_path),
        "performance_gate_job_resource_audit": gate_job_resource_audit,
        "formal_authorized_by_performance_gate": True,
    }


def _preflight_formal_authorization(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate pilot/gate/deployment before opening any row-level store.

    Registry and cache digests declared by the immutable gate are used only
    for this early receipt replay.  ``run`` later recomputes the actual data
    digests and invokes the complete performance verifier again before it
    creates an output directory.
    """

    def safe_receipt(value: str | Path, label: str) -> Path:
        supplied = Path(value)
        if supplied.is_symlink():
            raise ValueError(f"{label} root may not be a symbolic link")
        candidate = supplied.resolve()
        report = candidate / "report.json" if candidate.is_dir() else candidate
        done = report.with_name("DONE.json")
        if report.is_symlink() or done.is_symlink():
            raise ValueError(f"{label} report/DONE may not be symbolic links")
        if not report.is_file() or not done.is_file():
            raise FileNotFoundError(report if not report.is_file() else done)
        return candidate

    pilot_path = safe_receipt(args.pilot_report, "pilot receipt")
    gate_path = safe_receipt(args.performance_gate, "performance-gate receipt")
    pilot = _verify_formal_pilot(pilot_path, args)

    deployment_root = os.environ.get("DLCOAL_TRAIN_DEPLOY")
    if not deployment_root:
        raise ValueError("one-seed/formal stage requires DLCOAL_TRAIN_DEPLOY")
    from .carrier_raw_one_seed import (
        _current_resources,
        _performance_resource_policy,
        _validate_current_resources,
        deployment_binding,
    )

    deployment = deployment_binding(Path(deployment_root))
    expected_deployment = {
        "path": deployment["path"],
        "manifest": deployment["manifest_path"],
        "manifest_sha256": deployment["manifest_sha256"],
    }
    pilot_report = _read_json(
        pilot_path / "report.json" if pilot_path.is_dir() else pilot_path
    )
    if pilot_report.get("training_deployment") != expected_deployment:
        raise ValueError("pilot deployment differs from current immutable deployment")

    gate_report_path = gate_path / "report.json" if gate_path.is_dir() else gate_path
    gate_report = _read_json(gate_report_path)
    registry_sha256 = gate_report.get("registry_sha256")
    cache_sha256 = gate_report.get("cache_sha256")
    if (
        not isinstance(registry_sha256, dict)
        or not registry_sha256
        or not isinstance(cache_sha256, dict)
        or not cache_sha256
    ):
        raise ValueError("performance gate lacks declared data hashes")
    performance = _verify_performance_gate(
        gate_path,
        pilot_provenance=pilot,
        registry_sha256=registry_sha256,
        cache_sha256=cache_sha256,
    )
    performance_policy = _performance_resource_policy(gate_report)
    current_resources = _current_resources()
    _validate_current_resources(
        current_resources,
        expected_cpus=2,
        expected_memory_bytes=48 * 1024**3,
        expected_timelimit_seconds=74 * 3600,
        performance_policy=performance_policy,
    )
    result = {
        "pilot": pilot,
        "performance_gate": performance,
        "training_deployment": expected_deployment,
        "resource_preflight": {
            "current": current_resources,
            "performance_policy": performance_policy,
            "expected_cpus_per_task": 2,
            "expected_memory_bytes": 48 * 1024**3,
            "expected_timelimit_seconds": 74 * 3600,
            "expected_qos": "normal",
            "expected_gpu_count": 1,
        },
    }
    if args.stage == "formal":
        from .carrier_raw_one_seed import verify as verify_one_seed

        result["one_seed"] = verify_one_seed(
            receipt_dir=Path(args.one_seed_receipt),
            deployment_root=Path(deployment_root),
            registry_dir=Path(args.registry_dir),
            unphased_cache=Path(args.unphased_cache),
            pilot_dir=pilot_path,
            performance_gate_dir=gate_path,
            expected_backend=args.segment_reduction_backend,
            expected_signature=None,
            expected_registry_sha256=registry_sha256,
            expected_cache_sha256=cache_sha256,
            check_current_resources=True,
            expected_cpus=2,
            expected_memory_gib=48.0,
        )
    return result


def _formal_runtime_projection(
    arm_reports: Sequence[dict[str, Any]],
    *,
    pilot_epochs: int,
    formal_epochs: int,
    pilot_fit_sites: int,
    full_fit_sites: int,
    pilot_inner_sites: int,
    full_inner_sites: int,
    pilot_evaluation_sites: int,
    full_evaluation_sites: int,
    pilot_fit_rows: int,
    full_fit_rows: int,
    pilot_inner_rows: int,
    full_inner_rows: int,
    pilot_evaluation_rows: int,
    full_evaluation_rows: int,
    amortize_one_time_cold_start: bool = False,
    observed_pilot_elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Extrapolate recurrent work and a single cold-start surcharge separately.

    The optimized pilot keeps the packed artifacts and segment plans resident
    across all six arms and all epochs in one process.  Its first-arm cache
    population cost is therefore charged once, while every later-arm phase is
    conservatively upper-bounded by the slowest warm arm.  Non-phase pilot
    overhead is repeated by epoch (but not by site), then a five-percent
    gate-only guard is applied.  The full-budget seed remains the empirical
    runtime gate; this forecast never substitutes for that measurement.
    """

    if min(
        pilot_epochs,
        formal_epochs,
        pilot_fit_sites,
        full_fit_sites,
        pilot_inner_sites,
        full_inner_sites,
        pilot_evaluation_sites,
        full_evaluation_sites,
        pilot_fit_rows,
        full_fit_rows,
        pilot_inner_rows,
        full_inner_rows,
        pilot_evaluation_rows,
        full_evaluation_rows,
    ) <= 0:
        raise ValueError("runtime projection site/epoch denominators must be positive")
    phase_names = (
        "fit_seconds",
        "inner_validation_seconds",
        "evaluation_seconds",
    )
    phase_values = {
        phase: [float(row[phase]) for row in arm_reports]
        for phase in phase_names
    }
    observed = {phase: sum(values) for phase, values in phase_values.items()}
    if any(not math.isfinite(value) or value <= 0 for value in observed.values()):
        raise ValueError("runtime projection requires positive finite phase timings")
    if any(
        not math.isfinite(value) or value <= 0
        for values in phase_values.values()
        for value in values
    ):
        raise ValueError("runtime projection requires positive finite arm timings")
    epoch_ratio = float(formal_epochs) / float(pilot_epochs)
    ratios = {
        "fit_site_ratio": float(full_fit_sites) / float(pilot_fit_sites),
        "inner_site_ratio": float(full_inner_sites) / float(pilot_inner_sites),
        "evaluation_site_ratio": float(full_evaluation_sites)
        / float(pilot_evaluation_sites),
        "fit_row_ratio": float(full_fit_rows) / float(pilot_fit_rows),
        "inner_row_ratio": float(full_inner_rows) / float(pilot_inner_rows),
        "evaluation_row_ratio": float(full_evaluation_rows)
        / float(pilot_evaluation_rows),
        "epoch_ratio": epoch_ratio,
    }
    phase_scale = {
        "fit_seconds": (ratios["fit_site_ratio"], epoch_ratio),
        "inner_validation_seconds": (ratios["inner_site_ratio"], epoch_ratio),
        # Final mechanism evaluation is run once, after checkpoint selection.
        "evaluation_seconds": (ratios["evaluation_site_ratio"], 1.0),
    }
    cold_start: dict[str, Any] = {
        "enabled": bool(amortize_one_time_cold_start),
        "cold_surcharge_scaled_to_full_sites_once": bool(
            amortize_one_time_cold_start
        ),
        "cold_surcharge_multiplied_by_epoch_ratio": False,
    }
    if amortize_one_time_cold_start:
        if (
            pilot_epochs != 1
            or formal_epochs <= pilot_epochs
            or full_fit_sites < pilot_fit_sites
            or full_inner_sites < pilot_inner_sites
            or full_evaluation_sites < pilot_evaluation_sites
            or full_fit_rows < pilot_fit_rows
            or full_inner_rows < pilot_inner_rows
            or full_evaluation_rows < pilot_evaluation_rows
            or tuple(str(row.get("arm", "")) for row in arm_reports) != RAW_ARMS
        ):
            raise ValueError(
                "cold-start amortization requires the exact six-arm one-epoch pilot"
            )
        cold_start.update(
            {
                "first_arm": RAW_ARMS[0],
                "warm_reference_arms": list(RAW_ARMS[1:]),
                "warm_envelope_rule": (
                    "all_six_arms_at_maximum_later_arm_phase_seconds"
                ),
                "full_process_cache_reused_across_arms_and_epochs": True,
                "phases": {},
            }
        )
        projected: dict[str, float] = {}
        recurring_projected: dict[str, float] = {}
        one_time_projected: dict[str, float] = {}
        site_only_projected: dict[str, float] = {}
        for phase in phase_names:
            values = phase_values[phase]
            warm_max = max(values[1:])
            cold_excess = max(0.0, values[0] - warm_max)
            steady_upper = float(len(values)) * warm_max
            site_ratio, repeated_epochs = phase_scale[phase]
            recurring = steady_upper * site_ratio * repeated_epochs
            one_time = cold_excess * site_ratio
            site_only_projected[phase] = recurring + one_time
            recurring_projected[phase] = recurring
            one_time_projected[phase] = one_time
            cold_start["phases"][phase] = {
                "first_arm_seconds": values[0],
                "maximum_later_arm_seconds": warm_max,
                "one_time_cold_excess_seconds": cold_excess,
                "six_arm_steady_upper_seconds": steady_upper,
                "site_ratio": site_ratio,
                "repeated_epoch_ratio": repeated_epochs,
            }

        # ``inner`` and ``evaluation`` share exactly the same forward path.
        # Their warm-arm differences identify a per-site alpha and per-row
        # beta.  Reuse that beta for fit, solve a fit-specific alpha, and take
        # the slowest warm-arm full-cohort prediction.  The final phase bound
        # is the maximum of this paired row+site model and the site-only
        # envelope above, so extra full-cohort rows cannot disappear from the
        # forecast.
        site_delta = float(pilot_evaluation_sites - pilot_inner_sites)
        if site_delta <= 0:
            raise ValueError(
                "paired row/site projection requires evaluation sites above inner sites"
            )
        warm_models: list[dict[str, Any]] = []
        warm_full: dict[str, list[float]] = {phase: [] for phase in phase_names}
        for arm_index, arm_report in enumerate(arm_reports[1:], start=1):
            inner_seconds = phase_values["inner_validation_seconds"][arm_index]
            evaluation_seconds = phase_values["evaluation_seconds"][arm_index]
            alpha = (evaluation_seconds - inner_seconds) / site_delta
            beta = (
                inner_seconds - alpha * float(pilot_inner_sites)
            ) / float(pilot_inner_rows)
            fit_alpha = (
                phase_values["fit_seconds"][arm_index]
                - beta * float(pilot_fit_rows)
            ) / float(pilot_fit_sites)
            if any(
                not math.isfinite(value) or value <= 0
                for value in (alpha, beta, fit_alpha)
            ):
                raise ValueError(
                    "paired row/site projection produced a nonpositive coefficient"
                )
            full_values = {
                "fit_seconds": (
                    fit_alpha * float(full_fit_sites)
                    + beta * float(full_fit_rows)
                )
                * epoch_ratio,
                "inner_validation_seconds": (
                    alpha * float(full_inner_sites)
                    + beta * float(full_inner_rows)
                )
                * epoch_ratio,
                "evaluation_seconds": (
                    alpha * float(full_evaluation_sites)
                    + beta * float(full_evaluation_rows)
                ),
            }
            if any(not math.isfinite(value) or value <= 0 for value in full_values.values()):
                raise ValueError("paired row/site full-cohort projection is invalid")
            for phase, value in full_values.items():
                warm_full[phase].append(value)
            warm_models.append(
                {
                    "arm": str(arm_report["arm"]),
                    "shared_evaluation_path_seconds_per_site": alpha,
                    "shared_evaluation_path_seconds_per_row": beta,
                    "fit_seconds_per_site_after_row_component": fit_alpha,
                    "full_cohort_phase_seconds_per_arm": full_values,
                }
            )
        paired_projected: dict[str, float] = {}
        row_ratio_by_phase = {
            "fit_seconds": ratios["fit_row_ratio"],
            "inner_validation_seconds": ratios["inner_row_ratio"],
            "evaluation_seconds": ratios["evaluation_row_ratio"],
        }
        for phase in phase_names:
            warm_full_max = max(warm_full[phase])
            cold_excess = cold_start["phases"][phase][
                "one_time_cold_excess_seconds"
            ]
            site_ratio = cold_start["phases"][phase]["site_ratio"]
            cold_scale = max(site_ratio, row_ratio_by_phase[phase])
            paired_projected[phase] = (
                float(len(arm_reports)) * warm_full_max
                + cold_excess * cold_scale
            )
            projected[phase] = max(
                site_only_projected[phase], paired_projected[phase]
            )
            cold_start["phases"][phase].update(
                {
                    "row_ratio": row_ratio_by_phase[phase],
                    "paired_cold_scale_max_row_or_site": cold_scale,
                }
            )
        row_site_model: dict[str, Any] | None = {
            "method": "warm_inner_eval_paired_alpha_beta_with_fit_alpha",
            "warm_arm_coefficients": warm_models,
            "six_arm_plus_single_cold_phase_seconds": paired_projected,
            "site_only_six_arm_plus_single_cold_phase_seconds": (
                site_only_projected
            ),
            "phase_envelope_rule": "maximum_of_paired_row_site_and_site_only",
        }
        method = (
            "warm_arm_paired_row_site_envelope_plus_single_cold_surcharge_v3"
        )
    else:
        projected = {
            phase: observed[phase] * site_ratio * repeated_epochs
            for phase, (site_ratio, repeated_epochs) in phase_scale.items()
        }
        recurring_projected = dict(projected)
        one_time_projected = {phase: 0.0 for phase in phase_names}
        site_only_projected = dict(projected)
        row_site_model = None
        method = "separate_fit_inner_evaluation_site_extrapolation_v1"
    projected_phase_total = sum(projected.values())
    if amortize_one_time_cold_start:
        elapsed = float(observed_pilot_elapsed_seconds or math.nan)
        observed_phase_total = sum(observed.values())
        if (
            not math.isfinite(elapsed)
            or elapsed <= 0
            or elapsed + 1.0e-9 < observed_phase_total
        ):
            raise ValueError(
                "cold-start projection requires a bound whole-run pilot elapsed time"
            )
        observed_nonphase = max(0.0, elapsed - observed_phase_total)
        projected_nonphase = observed_nonphase * epoch_ratio
        unguarded_total = projected_phase_total + projected_nonphase
        guard_fraction = RUNTIME_PROJECTION_GUARD_FRACTION
        guard_seconds = unguarded_total * guard_fraction
        total = unguarded_total + guard_seconds
    else:
        elapsed = None
        observed_nonphase = 0.0
        projected_nonphase = 0.0
        unguarded_total = projected_phase_total
        guard_fraction = 0.0
        guard_seconds = 0.0
        total = projected_phase_total
    return {
        "method": method,
        "observed_pilot_seconds": observed,
        "projection_inputs": {
            "pilot_epochs": int(pilot_epochs),
            "formal_epochs": int(formal_epochs),
            "pilot_fit_sites": int(pilot_fit_sites),
            "full_fit_sites": int(full_fit_sites),
            "pilot_inner_sites": int(pilot_inner_sites),
            "full_inner_sites": int(full_inner_sites),
            "pilot_evaluation_sites": int(pilot_evaluation_sites),
            "full_evaluation_sites": int(full_evaluation_sites),
            "pilot_fit_rows": int(pilot_fit_rows),
            "full_fit_rows": int(full_fit_rows),
            "pilot_inner_rows": int(pilot_inner_rows),
            "full_inner_rows": int(full_inner_rows),
            "pilot_evaluation_rows": int(pilot_evaluation_rows),
            "full_evaluation_rows": int(full_evaluation_rows),
        },
        "scale_factors": ratios,
        "cold_start_amortization": cold_start,
        "projected_recurrent_seconds": recurring_projected,
        "projected_one_time_cold_start_seconds": one_time_projected,
        "site_only_projected_formal_seconds": site_only_projected,
        "paired_row_site_projection": row_site_model,
        "projected_formal_seconds": projected,
        "projected_phase_total_seconds": projected_phase_total,
        "nonphase_overhead_projection": {
            "observed_pilot_elapsed_seconds": elapsed,
            "observed_pilot_phase_seconds": sum(observed.values()),
            "observed_pilot_nonphase_seconds": observed_nonphase,
            "scaling_rule": "epoch_ratio_only_no_site_multiplier",
            "projected_formal_seconds": projected_nonphase,
        },
        "unguarded_projected_formal_total_seconds": unguarded_total,
        "gate_only_projection_guard": {
            "fraction": guard_fraction,
            "seconds": guard_seconds,
            "not_reapplied_to_full_budget_measurement": True,
        },
        "projected_formal_total_seconds": total,
        "projected_formal_total_hours": total / 3600.0,
    }


def _requires_formal_runtime_projection(
    *, stage: str, directional_only: bool
) -> bool:
    """Separate the formal one-epoch pilot gate from directional pilots.

    A directional pilot deliberately uses a short multi-epoch budget and can
    never authorize formal training.  It therefore must not enter the formal
    runtime-projection or projection-audit path, both of which are defined only
    for the exact one-epoch performance pilot.
    """

    return stage == "pilot" and not directional_only


def _audit_single_cold_start_cache_trace(report: dict[str, Any]) -> dict[str, Any]:
    """Require the observed cache trace that makes one cold surcharge valid."""

    arm_reports = report.get("arm_reports")
    if (
        not isinstance(arm_reports, list)
        or tuple(str(row.get("arm", "")) for row in arm_reports) != RAW_ARMS
    ):
        raise ValueError("pilot cold-start cache audit requires the exact arm order")
    snapshots: dict[str, list[dict[str, int]]] = {}
    for cache_name in ("artifact_cache", "segment_plan_cache"):
        normalized: list[dict[str, int]] = []
        for arm_report in arm_reports:
            loader = arm_report.get("loader_performance")
            cache = loader.get(cache_name) if isinstance(loader, dict) else None
            if not isinstance(cache, dict):
                raise ValueError(f"pilot arm lacks {cache_name} cache telemetry")
            values = {
                name: int(cache.get(name, -1))
                for name in (
                    "entries",
                    "hits",
                    "misses",
                    "evictions",
                    "rejected_oversize",
                )
            }
            if (
                min(values.values()) < 0
                or values["evictions"] != 0
                or values["rejected_oversize"] != 0
            ):
                raise ValueError(f"pilot {cache_name} cache churned")
            normalized.append(values)
        first = normalized[0]
        if (
            first["entries"] <= 0
            or first["entries"] != first["misses"]
            or any(row["misses"] != first["misses"] for row in normalized[1:])
            or any(row["entries"] != first["entries"] for row in normalized[1:])
        ):
            raise ValueError(
                f"pilot {cache_name} misses did not stop after the first arm"
            )
        snapshots[cache_name] = normalized
    return {
        "exact_arm_order": list(RAW_ARMS),
        "artifact_and_segment_population_completed_in_first_arm": True,
        "later_arm_artifact_miss_growth": 0,
        "later_arm_segment_plan_miss_growth": 0,
        "zero_evictions": True,
        "zero_rejected_oversize": True,
        "arm_cache_snapshots": snapshots,
    }


def _audit_formal_runtime_projection_report(
    report: dict[str, Any],
    *,
    expected_inputs: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Independently reconstruct the authorized one-cold-start pilot projection."""

    projection = report.get("formal_runtime_projection")
    arm_reports = report.get("arm_reports")
    if not isinstance(projection, dict) or not isinstance(arm_reports, list):
        raise ValueError("pilot lacks a formal runtime projection or arm timings")
    inputs = projection.get("projection_inputs")
    required = {
        "pilot_epochs",
        "formal_epochs",
        "pilot_fit_sites",
        "full_fit_sites",
        "pilot_inner_sites",
        "full_inner_sites",
        "pilot_evaluation_sites",
        "full_evaluation_sites",
        "pilot_fit_rows",
        "full_fit_rows",
        "pilot_inner_rows",
        "full_inner_rows",
        "pilot_evaluation_rows",
        "full_evaluation_rows",
    }
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise ValueError("pilot runtime projection inputs are incomplete")
    normalized_inputs: dict[str, int] = {}
    for name in sorted(required):
        value = inputs.get(name)
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"pilot runtime projection input {name} is invalid")
        normalized_inputs[name] = int(value)
    if expected_inputs is not None and normalized_inputs != expected_inputs:
        raise ValueError("pilot runtime projection inputs differ from bound split sites")
    recomputed = _formal_runtime_projection(
        arm_reports,
        **normalized_inputs,
        amortize_one_time_cold_start=True,
        observed_pilot_elapsed_seconds=float(
            report.get("elapsed_seconds", math.nan)
        ),
    )
    total_seconds = float(recomputed["projected_formal_total_seconds"])
    total_hours = float(recomputed["projected_formal_total_hours"])
    max_formal_hours = float(report.get("max_formal_hours", math.nan))
    if (
        projection != recomputed
        or projection.get("method")
        != "warm_arm_paired_row_site_envelope_plus_single_cold_surcharge_v3"
        or not math.isfinite(max_formal_hours)
        or max_formal_hours <= 0
        or float(report.get("estimated_formal_hours", math.nan)) != total_hours
        or report.get("formal_feasible") is not (total_hours <= max_formal_hours)
    ):
        raise ValueError("pilot formal runtime projection does not close exactly")
    cold_cache_trace = _audit_single_cold_start_cache_trace(report)
    full_cache_projection = report.get("full_cohort_cache_projection")
    segment_usage = report.get("segment_plan_usage")
    segment_terminal = (
        segment_usage.get("terminal_cache_contract")
        if isinstance(segment_usage, dict)
        else None
    )
    persistent_terminal = report.get("persistent_cache_terminal_contract")
    if (
        not isinstance(full_cache_projection, dict)
        or full_cache_projection.get("all_persistent_projections_fit") is not True
        or not isinstance(segment_terminal, dict)
        or not isinstance(persistent_terminal, dict)
    ):
        raise ValueError(
            "pilot runtime projection lacks full-cache and terminal contracts"
        )
    return {
        "method": recomputed["method"],
        "projection_sha256": _json_contract_sha256(recomputed),
        "projected_formal_total_seconds": total_seconds,
        "projected_formal_total_hours": total_hours,
        "maximum_formal_hours": max_formal_hours,
        "one_time_cold_start_occurrences": 1,
        "cold_surcharge_multiplied_by_epoch_ratio": False,
        "full_cohort_cache_fit_required": True,
        "terminal_zero_churn_required": True,
        "gate_only_guard_fraction": RUNTIME_PROJECTION_GUARD_FRACTION,
        "gate_only_guard_seconds": float(
            recomputed["gate_only_projection_guard"]["seconds"]
        ),
        "unguarded_projected_formal_total_seconds": float(
            recomputed["unguarded_projected_formal_total_seconds"]
        ),
        "cold_start_cache_trace": cold_cache_trace,
        "full_cohort_cache_fit_verified": True,
        "segment_terminal_contract_verified": True,
        "persistent_terminal_contract_verified": True,
    }


def _run_smoke(
    torch: Any,
    model_class: Any,
    *,
    rows: Sequence[dict[str, Any]],
    registry_to_cache: np.ndarray,
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    residual_anchor: np.ndarray,
    args: argparse.Namespace,
    device: Any,
    loader: _ArtifactAwareRawLoader,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    initial_hashes: set[str] = set()
    model_kwargs = _model_kwargs(args)
    residual_anchor_device = loader._pin(
        torch, np.asarray(residual_anchor, dtype=np.float32)
    ).to(device, non_blocking=True)
    for arm in args.arms:
        for row_index in indices:
            _seed_everything(torch, args.seed, device)
            model = model_class(**model_kwargs).to(device)
            initial_hash = _state_sha256(model)
            initial_hashes.add(initial_hash)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            with loader.fixed_sample_size(_measurement_diploids(args)):
                staged_rows = loader.iter_staged(
                    torch, model, (int(row_index),), arm
                )
                staged = next(iter(staged_rows))
            try:
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                prediction, target = _forward_row(
                    torch,
                    model,
                    staged=staged,
                    residual_anchor=residual_anchor_device,
                    device=device,
                    loader=loader,
                )
            finally:
                staged_rows.close()
            if not bool(torch.isfinite(prediction).all().item()):
                raise RuntimeError("smoke produced a nonfinite direct64 value")
            loss = torch.mean((prediction - target).square())
            loss.backward()
            site_gradient = model.site_projection[0].weight.grad
            long_gradient = model.long_blocks[0].depthwise.weight.grad
            physics_gradient = model.physics_attention_bias.grad
            if (
                site_gradient is None
                or not torch.isfinite(site_gradient).all()
                or float(torch.linalg.vector_norm(site_gradient)) <= 0
            ):
                raise RuntimeError("raw site branch gradient is invalid or zero")
            if arm == "carrier_correct" and (
                long_gradient is None
                or not torch.isfinite(long_gradient).all()
                or float(torch.linalg.vector_norm(long_gradient)) <= 0
                or physics_gradient is None
                or not torch.isfinite(physics_gradient).all()
                or float(torch.linalg.vector_norm(physics_gradient)) <= 0
            ):
                raise RuntimeError("correct-arm long/physics gradient is invalid or zero")
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            result = {
                "status": "complete",
                "arm": arm,
                **staged.audit,
                "loss": float(loss.detach().cpu()),
                "elapsed_seconds": time.perf_counter() - started,
                "all_sites_consumed": True,
                "initial_state_sha256": initial_hash,
                "post_step_state_sha256": _state_sha256(model),
                "raw_gradient_l2": float(torch.linalg.vector_norm(site_gradient).detach().cpu()),
                "long_gradient_l2": (
                    float(torch.linalg.vector_norm(long_gradient).detach().cpu())
                    if long_gradient is not None else 0.0
                ),
                "physics_gradient_l2": (
                    float(torch.linalg.vector_norm(physics_gradient).detach().cpu())
                    if physics_gradient is not None else 0.0
                ),
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            if result["peak_cuda_reserved_bytes"] > int(
                args.max_gpu_reserved_gib * (1024**3)
            ):
                raise RuntimeError("smoke exceeded the registered GPU memory envelope")
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            del model, optimizer, staged, prediction, target, loss
        _clear_cuda_boundary(torch)
    if len(results) != len(RAW_ARMS) * 2 or len(initial_hashes) != 1:
        raise RuntimeError("smoke did not complete exact 6x2 runs from one initial state")
    counts = [int(row["variant_count"]) for row in rows]
    if sorted(int(row["variant_count"]) for row in results[:0]) != []:
        raise AssertionError("unreachable")
    selected_counts = [int(rows[int(index)]["variant_count"]) for index in indices]
    if selected_counts != [min(counts), max(counts)]:
        raise RuntimeError("smoke pair is not exact global min/max V")
    return {
        "results": results,
        "same_initial_state": True,
        "all_sites_consumed": True,
        "runs_expected": len(RAW_ARMS) * 2,
        "runs_completed": len(results),
        "minimum_variant_count": selected_counts[0],
        "maximum_variant_count": selected_counts[1],
    }


RESUME_SCHEMA = "dlcoalsim-carrier-raw-resume-v1"


def _resume_binding(*, arm: str, seed: int, signature: str, initial_hash: str,
                    epochs: int, accumulate_rows: int) -> dict[str, Any]:
    """What a resume state must agree with before it may be adopted.

    A checkpoint that came from a different arm, seed, epoch budget or model
    initialization is not a continuation of this run, and silently adopting one
    would produce a result no receipt could describe.
    """

    return {
        "schema": RESUME_SCHEMA,
        "arm": str(arm),
        "seed": int(seed),
        "experiment_signature": str(signature),
        "initial_state_sha256": str(initial_hash),
        "epochs": int(epochs),
        "accumulate_rows": int(accumulate_rows),
    }


def _write_resume_state(torch: Any, path: Path, payload: dict[str, Any]) -> None:
    """Replace the resume state atomically.

    A partial file here is worse than none: the next start would load a
    truncated optimizer and continue from a state that never existed.  Write
    beside it, flush to disk, then rename -- rename is atomic within a
    directory, so a reader sees either the previous state or the new one.
    """

    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _train_arm(
    torch: Any,
    functional: Any,
    model_class: Any,
    *,
    arm: str,
    seed: int,
    output: Path,
    fit_indices: np.ndarray,
    inner_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    rows: Sequence[dict[str, Any]],
    registry_to_cache: np.ndarray,
    cache: dict[str, np.ndarray],
    residual_anchor: np.ndarray,
    signature: str,
    args: argparse.Namespace,
    device: Any,
    loader: _ArtifactAwareRawLoader,
    execution_audit: _ExecutionAudit,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    attempt = output.with_name(f".{output.name}.attempt-{os.getpid()}-{time.time_ns()}")
    attempt.mkdir(parents=True, exist_ok=False)
    # The attempt directory carries the process id, so it is new on every start
    # and cannot hold state across one.  Resume state lives beside the final
    # output instead, where a restarted job can find it.
    resume_path = output.with_name(output.name + ".resume.pt")
    _seed_everything(torch, seed, device)
    model = model_class(**_model_kwargs(args)).to(device)
    residual_anchor_device = loader._pin(
        torch, np.asarray(residual_anchor, dtype=np.float32)
    ).to(device, non_blocking=True)
    initial_hash = _state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.scheduler_epochs, eta_min=args.learning_rate * 0.05
    )
    best_rmse = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    curve: list[dict[str, Any]] = []
    training_sites = 0
    fit_seconds_total = 0.0
    inner_seconds_total = 0.0
    first_epoch = 1
    binding = _resume_binding(
        arm=arm, seed=seed, signature=signature, initial_hash=initial_hash,
        epochs=int(args.epochs), accumulate_rows=int(args.accumulate_rows),
    )
    if getattr(args, "resume", False) and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("binding") != binding:
            raise RuntimeError(
                "resume state does not describe this run: "
                f"{state.get('binding')!r} against {binding!r}"
            )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_rmse = float(state["best_rmse"])
        best_epoch = int(state["best_epoch"])
        best_state = state["best_state"]
        curve = list(state["curve"])
        training_sites = int(state["training_sites"])
        fit_seconds_total = float(state["fit_seconds_total"])
        inner_seconds_total = float(state["inner_seconds_total"])
        first_epoch = int(state["epoch"]) + 1
        # Rebuild the streaming receipts for the epochs already done.  Nothing
        # about them is stochastic, so this reproduces the digests an
        # uninterrupted run would have had rather than approximating them.
        execution_audit.replay_completed_epochs(
            arm=arm, seed=seed, fit_indices=fit_indices,
            accumulate_rows=int(args.accumulate_rows),
            through_epoch=int(state["epoch"]),
        )
        print(
            json.dumps({"resumed_from_epoch": int(state["epoch"]),
                        "resume_path": str(resume_path)}, sort_keys=True),
            flush=True,
        )
    torch.cuda.reset_peak_memory_stats(device)
    # One rank sees every row and steps every accumulate_rows, which is what
    # the single-device loop has always done; a wider world deals each block of
    # rows by predicted cost and cuts every rank's share at the same global row
    # counts, so the optimizer sees the same batch either way.
    world = distributed_env()
    row_costs = predict_row_cost(cache["variant_counts"])
    for epoch in range(first_epoch, args.epochs + 1):
        epoch_plan = plan_epoch(
            _epoch_row_order(fit_indices, seed=seed, epoch=epoch),
            row_costs,
            world_size=world.world_size,
            rank=world.rank,
            accumulate_rows=int(args.accumulate_rows),
        )
        order = list(epoch_plan.rows)
        loader.set_epoch(epoch)
        fit_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_squared = torch.zeros((), dtype=torch.float64, device=device)
        accumulation_finite = torch.ones((), dtype=torch.bool, device=device)
        elements = 0
        epoch_sites = 0
        for step, staged in enumerate(
            loader.iter_staged(torch, model, order, arm), start=1
        ):
            prediction, target = _forward_row(
                torch,
                model,
                staged=staged,
                residual_anchor=residual_anchor_device,
                device=device,
                loader=loader,
            )
            loss = functional.mse_loss(prediction, target)
            (loss / float(args.accumulate_rows)).backward()
            execution_audit.record_training_yield(
                arm=arm, epoch=epoch, step=step, staged=staged
            )
            total_squared.add_(loss.detach().to(dtype=torch.float64) * target.numel())
            accumulation_finite.logical_and_(torch.isfinite(loss.detach()))
            elements += target.numel()
            epoch_sites += int(staged.audit["variant_count"])
            if step in epoch_plan.step_after:
                # Reduce before the finiteness check and before clipping: a
                # rank that clipped its own gradient would scale a different
                # tensor from the one the optimizer sees, and a nonfinite loss
                # anywhere must stop every rank, not just the one that saw it.
                all_reduce_gradients(model.parameters(), env=world)
                if world.world_size > 1:
                    torch.distributed.all_reduce(
                        accumulation_finite,
                        op=torch.distributed.ReduceOp.MIN,
                    )
                if not bool(accumulation_finite.item()):
                    raise RuntimeError("training produced a nonfinite loss")
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                execution_audit.record_optimizer_boundary(
                    arm=arm, epoch=epoch, step=step, staged=staged
                )
                optimizer.zero_grad(set_to_none=True)
                accumulation_finite.fill_(True)
            del staged, prediction, target, loss
        training_sites += epoch_sites
        # The scalar D2H makes the fit timer complete without a device-wide
        # synchronize and keeps pending fit kernels out of the inner timer.
        fit_squared = float(total_squared.detach().cpu())
        fit_seconds = time.perf_counter() - fit_started
        fit_seconds_total += fit_seconds
        scheduler.step()
        inner_prediction, inner_target, _, _, inner_sites, inner_seconds = _evaluate(
            torch, model, arm, inner_indices, rows=rows,
            registry_to_cache=registry_to_cache, cache=cache,
            residual_anchor=residual_anchor_device, args=args, device=device,
            loader=loader,
        )
        inner_seconds_total += inner_seconds
        inner_metrics = _metrics(inner_prediction, inner_target)
        record = {
            "epoch": epoch,
            "fit_rmse": math.sqrt(fit_squared / elements),
            "inner_validation_rmse": float(inner_metrics["rmse"]),
            "inner_validation_mae": float(inner_metrics["mae"]),
            "fit_sites_consumed": epoch_sites,
            "fit_seconds": fit_seconds,
            "inner_sites_consumed": inner_sites,
            "inner_seconds": inner_seconds,
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        curve.append(record)
        print(json.dumps({"arm": arm, "seed": seed, **record}, sort_keys=True), flush=True)
        if record["inner_validation_rmse"] < best_rmse:
            best_rmse = record["inner_validation_rmse"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if getattr(args, "resume", False):
            # Written after the epoch's inner validation, so the state on disk
            # is always a whole epoch: a restart repeats no partial epoch and
            # therefore no partial optimizer step.  Everything the loop carries
            # is here; the audit digests are not, because they are replayed
            # from (seed, epoch) instead.
            _write_resume_state(torch, resume_path, {
                "binding": binding,
                "epoch": int(epoch),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_rmse": float(best_rmse),
                "best_epoch": int(best_epoch),
                "best_state": best_state,
                "curve": curve,
                "training_sites": int(training_sites),
                "fit_seconds_total": float(fit_seconds_total),
                "inner_seconds_total": float(inner_seconds_total),
            })
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    training_seconds = fit_seconds_total
    model.load_state_dict(best_state)
    prediction, target, original_rows, canonical_ids, eval_sites, eval_seconds = _evaluate(
        torch, model, arm, evaluation_indices, rows=rows,
        registry_to_cache=registry_to_cache, cache=cache,
        residual_anchor=residual_anchor_device, args=args, device=device,
        loader=loader,
    )
    metrics = _metrics(prediction, target)
    checkpoint_path = attempt / "checkpoint.pt"
    torch.save(
        {
            "state_dict": best_state,
            "arm": arm,
            "seed": seed,
            "experiment_signature": signature,
            "direct64": True,
            "nw_anchor": "fit-only mean residual plus per-row log10(N_W)",
        },
        checkpoint_path,
    )
    # The run finished, so the resume state is now a stale description of a
    # partial run.  Leaving it would let a rerun of the same output silently
    # skip training entirely.
    resume_path.unlink(missing_ok=True)
    prediction_path = attempt / "mechanism_evaluation.npz"
    np.savez(
        prediction_path,
        row_index=original_rows,
        canonical_id=np.asarray(canonical_ids, dtype="U160"),
        prediction=prediction,
        target=target,
    )
    report = {
        "schema": RUNNER_SCHEMA,
        "status": "complete",
        "stage": args.stage,
        "scientific_status": "provisional mechanism development",
        "publication_benchmark": False,
        "arm": arm,
        "seed": seed,
        "experiment_signature": signature,
        "initial_state_sha256": initial_hash,
        "model_name": _active_model_name(args),
        "model_parameters": int(sum(value.numel() for value in model.parameters())),
        "fixed_epoch_budget": args.epochs,
        "scheduler_epoch_horizon": args.scheduler_epochs,
        "epochs_ran": args.epochs,
        "best_epoch": best_epoch,
        "best_inner_validation_rmse": best_rmse,
        "mechanism_evaluation_metrics": metrics,
        "fit_rows": len(fit_indices),
        "inner_rows": len(inner_indices),
        "mechanism_evaluation_rows": len(evaluation_indices),
        "training_sites_consumed": training_sites,
        "evaluation_sites_consumed": eval_sites,
        "training_seconds": training_seconds,
        "fit_seconds": fit_seconds_total,
        "inner_validation_seconds": inner_seconds_total,
        "evaluation_seconds": eval_seconds,
        "device": str(device),
        "gpu_name": str(torch.cuda.get_device_name(device)),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint_artifact": {
            "path": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
        },
        "mechanism_evaluation_artifact": {
            "path": prediction_path.name,
            "sha256": _sha256(prediction_path),
        },
        "curve": curve,
        "all_sites_consumed": True,
        "selection_rows_read": 0,
        "selection_access_count": 0,
        "direct_64_bin_output": True,
        "hazard_tail_survival_dual_or_class_head": False,
        "directional_only": bool(args.directional_only),
        "screening_only": bool(args.screening_only),
        "screening_scope": SCREEN_SCOPE if args.screening_only else None,
        "screen_arm_contract": (
            vars(screen_arm_spec(arm)) if args.screening_only else None
        ),
        "formal_authorization_eligible": (
            False if _short_budget_only(args) else None
        ),
        "loader_performance": loader.stats(),
    }
    if report["peak_cuda_reserved_bytes"] > int(
        args.max_gpu_reserved_gib * (1024**3)
    ):
        raise RuntimeError("training exceeded the registered GPU memory envelope")
    report_path = attempt / "report.json"
    _json_atomic(report_path, report)
    _json_atomic(
        attempt / "DONE.json",
        {
            "schema": RUNNER_SCHEMA,
            "status": "complete",
            "arm": arm,
            "seed": seed,
            "experiment_signature": signature,
            "report_sha256": _sha256(report_path),
            "checkpoint_artifact": report["checkpoint_artifact"],
            "mechanism_evaluation_artifact": report["mechanism_evaluation_artifact"],
            "selection_rows_read": 0,
            "selection_access_count": 0,
            "all_sites_consumed": True,
            "direct_64_bin_output": True,
            "directional_only": bool(args.directional_only),
            "screening_only": bool(args.screening_only),
            "screen_arm_contract": (
                vars(screen_arm_spec(arm)) if args.screening_only else None
            ),
            "formal_authorization_eligible": (
                False if _short_budget_only(args) else None
            ),
        },
    )
    attempt.replace(output)
    del model, optimizer, scheduler, best_state
    _clear_cuda_boundary(torch)
    return report


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    registry_dir = Path(args.registry_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    receipt_preflight = None
    if args.stage in {"one_seed", "formal"}:
        receipt_preflight = _preflight_formal_authorization(args)
    rows, identities, reference, registry_provenance = _load_registry(
        registry_dir, expected_selection_rows=args.expected_selection_rows
    )
    # The bound registry states its own size; the sealed selection count
    # stays the module constant and is checked inside _load_registry.
    scale = _raw_registry_scale(
        registry_provenance["plan"], reference["architecture_split"]
    )
    cache, cache_identities, architecture_groups, cache_provenance = (
        _load_unphased_cache_slim(
            args.unphased_cache,
            expected_selection_rows=args.expected_selection_rows,
        )
    )
    if int(cache_provenance.get("selection_rows_read", -1)) != 0:
        raise ValueError("unphased cache does not prove selection exclusion")
    registry_to_cache = _bind_registry_to_cache(
        rows,
        identities,
        reference,
        cache,
        cache_identities,
        expected_rows=scale.development_rows,
    )
    fit, inner, evaluation, inner_split_sha, inner_manifest = _inner_split(
        cache_identities,
        cache["architecture_split"],
        architecture_groups,
        basis_points=args.inner_validation_basis_points,
    )
    if (
        len(fit) < 1
        or len(inner) < 1
        or len(fit) + len(inner) != scale.train_rows
        or len(evaluation) != scale.evaluation_rows
        or len(rows) != scale.development_rows
    ):
        raise ValueError(
            "fit/inner/evaluation split does not partition the bound registry"
        )
    fit_rows, inner_rows, evaluation_rows = _select_stage_rows(
        args.stage, rows, registry_to_cache, fit, inner, evaluation, args
    )
    anchor_cache_rows = (
        registry_to_cache[fit_rows]
        if (args.screening_only or args.het_gap_screening_only)
        else fit
    )
    residual_anchor = (
        cache["targets"][anchor_cache_rows]
        - cache["nw_log10"][anchor_cache_rows, None]
    ).mean(axis=0, dtype=np.float64).astype(np.float32)

    # CUBLAS determinism must be configured by the batch environment before
    # importing torch.  Refuse a late/in-process approximation.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be set before torch import")
    try:
        import torch
        from torch.nn import functional
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("carrier raw runner requires PyTorch") from error
    from .carrier_aware_raw_site_model import CarrierAwareRawGenotypeDirect64

    if not torch.cuda.is_available():
        raise RuntimeError("carrier raw stages require strict CUDA")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    gpu_name = str(torch.cuda.get_device_name(device))
    gpu_properties = torch.cuda.get_device_properties(device)
    gpu_contract = _validate_gpu_contract(
        gpu_name=gpu_name,
        total_memory_bytes=int(gpu_properties.total_memory),
        required_gpu_class=args.required_gpu_class,
        minimum_gpu_memory_gib=args.minimum_gpu_memory_gib,
    )
    if out_dir.exists():
        raise FileExistsError(out_dir)
    loader_config = _loader_config(args)
    loader = _ArtifactAwareRawLoader(
        rows=rows,
        registry_to_cache=registry_to_cache,
        cache=cache,
        verify_source_sha256=args.verify_source_sha256,
        max_bytes=int(loader_config["host_cache_bytes"]),
        max_sites=int(loader_config["host_cache_sites"]),
        prefetch_rows=int(loader_config["prefetch_rows"]),
        diploid_subsample=_parse_levels(args.diploid_subsample),
        diploid_subsample_seed=args.diploid_subsample_seed,
        diploid_fixed=args.diploid_fixed or None,
        expected_rows=scale.development_rows,
    )
    projection_model = CarrierAwareRawGenotypeDirect64(**_model_kwargs(args))
    full_cohort_cache_projection = _full_cohort_cache_projection(
        projection_model,
        rows,
        loader,
        raw_index_sha256=registry_provenance["sha256"]["raw_index.jsonl"],
        expected_rows=scale.development_rows,
    )
    del projection_model
    segment_cache_projection = {
        "projected_development_rows": scale.development_rows,
        "projected_upper_bound_bytes": int(
            full_cohort_cache_projection["segment_plan_upper_bound_bytes"]
        ),
        "projected_upper_bound_sites": int(
            full_cohort_cache_projection["projected_sites"]
        ),
        "cache_max_bytes": int(loader.segment_plan_cache.max_bytes),
        "cache_max_sites": int(loader.segment_plan_cache.max_sites),
        "fits_without_cyclic_thrashing": True,
    }
    pilot_provenance = None
    performance_gate_provenance = None
    if args.stage in {"one_seed", "formal"}:
        pilot_provenance = receipt_preflight["pilot"]
        performance_gate_provenance = _verify_performance_gate(
            Path(args.performance_gate).resolve(),
            pilot_provenance=pilot_provenance,
            registry_sha256=registry_provenance["sha256"],
            cache_sha256=cache_provenance["sha256"],
        )
    signature_payload = {
        "schema": RUNNER_SCHEMA,
        "model_name": _active_model_name(args),
        "model_config": _model_kwargs(args),
        "loader_config": loader_config,
        "arms": list(args.arms),
        "epochs": args.epochs,
        "scheduler_epochs": args.scheduler_epochs,
        "formal_epochs": args.formal_epochs,
        "accumulate_rows": args.accumulate_rows,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "max_formal_hours": args.max_formal_hours,
        "max_gpu_reserved_gib": args.max_gpu_reserved_gib,
        "required_gpu_class": args.required_gpu_class,
        "minimum_gpu_memory_gib": args.minimum_gpu_memory_gib,
        "directional_only": bool(args.directional_only),
        "screening_only": bool(args.screening_only),
        "screening_scope": SCREEN_SCOPE if args.screening_only else None,
        "nw_anchor_fit_rows": int(len(anchor_cache_rows)),
        "nw_anchor_row_index_sha256": _array_sha256(
            np.asarray(anchor_cache_rows, dtype=np.int64)
        ),
        "inner_split_sha256": inner_split_sha,
        "registry_sha256": registry_provenance["sha256"],
        "cache_sha256": cache_provenance["sha256"],
        "input_bindings": {
            "registry_dir": str(registry_dir),
            "unphased_cache": str(Path(args.unphased_cache).resolve()),
            "het_gap_cache": (
                str(Path(args.het_gap_cache).resolve())
                if args.het_gap_screening_only
                else None
            ),
        },
        "fit_rows": [int(value) for value in fit_rows],
        "inner_rows": [int(value) for value in inner_rows],
        "evaluation_rows": [int(value) for value in evaluation_rows],
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    one_seed_provenance = None
    if args.stage == "formal":
        deployment_root = os.environ.get("DLCOAL_TRAIN_DEPLOY")
        if not deployment_root:
            raise ValueError("formal stage requires DLCOAL_TRAIN_DEPLOY")
        from .carrier_raw_one_seed import verify as _verify_one_seed

        one_seed_provenance = _verify_one_seed(
            receipt_dir=Path(args.one_seed_receipt),
            deployment_root=Path(deployment_root),
            registry_dir=registry_dir,
            unphased_cache=Path(args.unphased_cache),
            pilot_dir=Path(args.pilot_report),
            performance_gate_dir=Path(args.performance_gate),
            expected_backend=args.segment_reduction_backend,
            expected_signature=signature,
            expected_registry_sha256=registry_provenance["sha256"],
            expected_cache_sha256=cache_provenance["sha256"],
            check_current_resources=True,
            expected_cpus=2,
            expected_memory_gib=48.0,
        )
    # Do not create a formal output directory until every immutable receipt,
    # including the full-budget seed-0 seal, has passed preflight.
    out_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        **signature_payload,
        "status": "running",
        "stage": args.stage,
        "seed": args.seed,
        "experiment_signature": signature,
        "scientific_status": "provisional mechanism development",
        "publication_benchmark": False,
        "directional_only": bool(args.directional_only),
        "screening_only": bool(args.screening_only),
        "directional_scope": (
            "three_seed_short_budget_full_840_development_evaluation"
            if args.directional_only
            else None
        ),
        "screening_scope": SCREEN_SCOPE if args.screening_only else None,
        "formal_authorization_eligible": (
            False if _short_budget_only(args) else None
        ),
        "selection_rows_read": 0,
        "selection_access_count": 0,
        "selection_excluded_before_raw_access": True,
        "split_counts": {
            "architecture_fit": len(fit_rows),
            "inner_validation": len(inner_rows),
            "repeated_development_evaluation": len(evaluation_rows),
            "sealed_selection": int(args.expected_selection_rows),
            "selection_rows_read": 0,
            "selection_access_count": 0,
        },
        "nw_residual_anchor_sha256": _array_sha256(residual_anchor),
        "inner_split_manifest": inner_manifest,
        "unphased_cache_materialization": cache_provenance["materialization"],
        "pilot_provenance": pilot_provenance,
        "performance_gate_provenance": performance_gate_provenance,
        "one_seed_provenance": one_seed_provenance,
        "formal_resource_preflight": (
            receipt_preflight["resource_preflight"]
            if receipt_preflight is not None
            else None
        ),
        "full_cohort_cache_projection": full_cohort_cache_projection,
        "strict_deterministic_cuda": True,
        "environment": _environment(torch, device),
        "gpu_contract": gpu_contract,
        "training_deployment": _training_deployment(),
        "all_sites_no_thinning_topk_fourier_crop": True,
        "direct_64_bin_output": True,
        "hazard_tail_survival_dual_or_class_head": False,
    }
    _json_atomic(out_dir / "PLAN.json", plan)
    started = time.perf_counter()
    if args.stage == "smoke":
        stage_result = _run_smoke(
            torch,
            CarrierAwareRawGenotypeDirect64,
            rows=rows,
            registry_to_cache=registry_to_cache,
            cache=cache,
            indices=fit_rows,
            residual_anchor=residual_anchor,
            args=args,
            device=device,
            loader=loader,
        )
        report = {
            **plan,
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - started,
            **stage_result,
            "selection_rows_read": 0,
            "selection_access_count": 0,
            "selection_rows_cached": 0,
            "strict_deterministic_cuda": True,
            "loader_performance": loader.stats(),
            "segment_plan_usage": loader.segment_plan_usage(
                segment_cache_projection
            ),
        }
    else:
        arm_reports: list[dict[str, Any]] = []
        initial_hashes: set[str] = set()
        execution_audit = _ExecutionAudit(rows)
        for arm in args.arms:
            arm_report = _train_arm(
                torch,
                functional,
                CarrierAwareRawGenotypeDirect64,
                arm=arm,
                seed=args.seed,
                output=out_dir / arm,
                fit_indices=fit_rows,
                inner_indices=inner_rows,
                evaluation_indices=evaluation_rows,
                rows=rows,
                registry_to_cache=registry_to_cache,
                cache=cache,
                residual_anchor=residual_anchor,
                signature=signature,
                args=args,
                device=device,
                loader=loader,
                execution_audit=execution_audit,
            )
            arm_reports.append(arm_report)
            initial_hashes.add(str(arm_report["initial_state_sha256"]))
        expected_arms = _expected_arms(args)
        if len(arm_reports) != len(expected_arms) or len(initial_hashes) != 1:
            raise RuntimeError(
                "registered arms did not share one seeded initial state"
            )
        observed_training_seconds = sum(
            float(record["training_seconds"]) for record in arm_reports
        )
        fit_sites_per_arm = sum(
            int(rows[int(index)]["variant_count"]) for index in fit_rows
        )
        inner_sites_per_arm = sum(
            int(rows[int(index)]["variant_count"]) for index in inner_rows
        )
        evaluation_sites_per_arm = sum(
            int(rows[int(index)]["variant_count"]) for index in evaluation_rows
        )
        requires_formal_runtime_projection = _requires_formal_runtime_projection(
            stage=args.stage, directional_only=_short_budget_only(args)
        )
        if requires_formal_runtime_projection:
            runner_elapsed_seconds = time.perf_counter() - started
            runtime_projection = _formal_runtime_projection(
                arm_reports,
                pilot_epochs=args.epochs,
                formal_epochs=args.formal_epochs,
                pilot_fit_sites=fit_sites_per_arm,
                full_fit_sites=sum(int(cache["variant_counts"][int(i)]) for i in fit),
                pilot_inner_sites=inner_sites_per_arm,
                full_inner_sites=sum(
                    int(cache["variant_counts"][int(i)]) for i in inner
                ),
                pilot_evaluation_sites=evaluation_sites_per_arm,
                full_evaluation_sites=sum(
                    int(cache["variant_counts"][int(i)]) for i in evaluation
                ),
                pilot_fit_rows=len(fit_rows),
                full_fit_rows=len(fit),
                pilot_inner_rows=len(inner_rows),
                full_inner_rows=len(inner),
                pilot_evaluation_rows=len(evaluation_rows),
                full_evaluation_rows=len(evaluation),
                amortize_one_time_cold_start=True,
                observed_pilot_elapsed_seconds=runner_elapsed_seconds,
            )
            estimated_formal_hours = float(
                runtime_projection["projected_formal_total_hours"]
            )
            formal_feasible = estimated_formal_hours <= args.max_formal_hours
        elif args.stage == "pilot":
            runner_elapsed_seconds = time.perf_counter() - started
            runtime_projection = None
            estimated_formal_hours = None
            formal_feasible = False
        else:
            runner_elapsed_seconds = time.perf_counter() - started
            runtime_projection = _formal_runtime_projection(
                arm_reports,
                pilot_epochs=args.epochs,
                formal_epochs=args.epochs,
                pilot_fit_sites=fit_sites_per_arm,
                full_fit_sites=fit_sites_per_arm,
                pilot_inner_sites=inner_sites_per_arm,
                full_inner_sites=inner_sites_per_arm,
                pilot_evaluation_sites=evaluation_sites_per_arm,
                full_evaluation_sites=evaluation_sites_per_arm,
                pilot_fit_rows=len(fit_rows),
                full_fit_rows=len(fit_rows),
                pilot_inner_rows=len(inner_rows),
                full_inner_rows=len(inner_rows),
                pilot_evaluation_rows=len(evaluation_rows),
                full_evaluation_rows=len(evaluation_rows),
            )
            estimated_formal_hours = float(
                runtime_projection["projected_formal_total_hours"]
            )
            formal_feasible = estimated_formal_hours <= args.max_formal_hours
        report = {
            **plan,
            "status": "complete",
            "elapsed_seconds": runner_elapsed_seconds,
            "arm_reports": arm_reports,
            "same_initial_state": True,
            "all_sites_consumed": True,
            "observed_execution": execution_audit.report(),
            "observed_training_seconds_all_arms": observed_training_seconds,
            "formal_runtime_projection": runtime_projection,
            "formal_runtime_projection_status": (
                "computed"
                if runtime_projection is not None
                else "not_applicable_registered_short_budget"
            ),
            "estimated_formal_hours": estimated_formal_hours,
            "max_formal_hours": args.max_formal_hours,
            "formal_feasible": formal_feasible,
            "formal_authorization_eligible": (
                args.stage == "pilot"
                and not _short_budget_only(args)
                and formal_feasible
            ),
            "selection_rows_read": 0,
            "selection_access_count": 0,
            "selection_rows_cached": 0,
            "loader_performance": loader.stats(),
            "segment_plan_usage": loader.segment_plan_usage(
                segment_cache_projection
            ),
        }
        if (
            args.stage in {"pilot", "one_seed", "formal"}
            and not formal_feasible
            and not (args.stage == "pilot" and _short_budget_only(args))
        ):
            report["status"] = "failed_runtime_gate"
    from .carrier_raw_performance_gate import _audit_segment_plan_cache_contract

    cache_audit_stage = "formal" if args.stage == "one_seed" else args.stage
    segment_cache_terminal_contract = _audit_segment_plan_cache_contract(
        report["segment_plan_usage"], stage=cache_audit_stage
    )
    report["segment_plan_usage"][
        "terminal_cache_contract"
    ] = segment_cache_terminal_contract
    persistent_cache_terminal_contract = _audit_persistent_cache_terminal_contract(
        report["loader_performance"],
        full_cohort_cache_projection,
        stage=args.stage,
        expected_rows=scale.development_rows,
    )
    report[
        "persistent_cache_terminal_contract"
    ] = persistent_cache_terminal_contract
    runtime_projection_contract = None
    if requires_formal_runtime_projection:
        runtime_projection_contract = _audit_formal_runtime_projection_report(report)
        report["runtime_projection_contract"] = runtime_projection_contract
    report_path = out_dir / "report.json"
    _json_atomic(report_path, report)
    if report.get("status") != "complete":
        raise RuntimeError(
            f"{args.stage} runtime gate failed; downstream formal work remains dependency-blocked"
        )
    _json_atomic(
        out_dir / "DONE.json",
        {
            "schema": RUNNER_SCHEMA,
            "status": "complete",
            "stage": args.stage,
            "seed": args.seed,
            "experiment_signature": signature,
            "report_sha256": _sha256(report_path),
            "selection_rows_read": 0,
            "selection_access_count": 0,
            "all_sites_consumed": True,
            "direct_64_bin_output": True,
            "directional_only": bool(args.directional_only),
            "screening_only": bool(args.screening_only),
            "screening_scope": SCREEN_SCOPE if args.screening_only else None,
            "formal_authorization_eligible": bool(
                report.get("formal_authorization_eligible", False)
            ),
            "segment_plan_cache_terminal_contract": (
                segment_cache_terminal_contract
            ),
            "persistent_cache_terminal_contract": (
                persistent_cache_terminal_contract
            ),
            **(
                {"runtime_projection_contract": runtime_projection_contract}
                if runtime_projection_contract is not None
                else {}
            ),
        },
    )
    return report


def main() -> None:
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
