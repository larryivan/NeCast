from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from .config import Config, apply_preset, validate_config
from .dataset_files import write_dataset_files
from .dataset_state import DatasetConfigMismatchError
from .empirical_maps import load_empirical_maps
from .shards import (
    DatasetGenerationIncompleteError,
    DatasetGenerationLockedError,
    DatasetSourceUnavailableError,
    UnsafeOutputDirectoryError,
    dataset_output_lock,
    finalize_dataset,
    generate_dataset,
    generate_dataset_partition,
    prepare_dataset,
    reuse_completed_dataset_if_compatible,
)
from .slurm import SlurmEnvironmentError, distributed_context


def parse_args() -> Config:
    base = apply_preset(Config())
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    bool_fields = {"force", "compression", "include_truth", "enable_stdpopsim", "mut_map_scale_to_baseline"}
    choice_fields = {"recomb_map_unit", "mut_map_unit"}
    for f in dataclasses.fields(Config):
        if f.name in bool_fields or f.name in choice_fields:
            continue
        default = getattr(base, f.name)
        parser.add_argument("--" + f.name.replace("_", "-"), dest=f.name, type=type(default), default=None)
    parser.add_argument("--n", dest="n_samples", type=int, default=None, help="alias for --n-samples")
    parser.add_argument("--recomb-map-unit", choices=["per_bp", "cM_per_Mb"], default=None)
    parser.add_argument("--mut-map-unit", choices=["per_bp", "relative", "roulette_raw"], default=None)

    parser.add_argument("--force", action="store_true", help="overwrite an existing simulator output directory")
    parser.add_argument("--include-truth", action="store_true", help="store true maps for analysis; not model inputs")
    parser.add_argument("--enable-stdpopsim", action="store_true", help="include optional stdpopsim anchors")
    parser.add_argument(
        "--mut-map-scale-to-baseline",
        dest="mut_map_scale_to_baseline",
        action="store_true",
        help="normalize relative or raw mutation maps to baseline_mu after unit conversion",
    )
    parser.add_argument(
        "--no-mut-map-scale-to-baseline",
        dest="mut_map_scale_to_baseline",
        action="store_false",
        help="preserve the absolute scale implied by --mut-map-unit",
    )
    parser.add_argument("--no-compression", dest="compression", action="store_false", help="write uncompressed npz shards")
    parser.set_defaults(compression=None)
    parser.set_defaults(mut_map_scale_to_baseline=None)
    generation_mode = parser.add_mutually_exclusive_group()
    generation_mode.add_argument(
        "--prepare-only",
        dest="_generation_mode",
        action="store_const",
        const="prepare",
        help="create or validate the resumable generation plan, then exit",
    )
    generation_mode.add_argument(
        "--generate-only",
        dest="_generation_mode",
        action="store_const",
        const="generate",
        help="generate only this rank's deterministic partition, then exit",
    )
    generation_mode.add_argument(
        "--finalize-only",
        dest="_generation_mode",
        action="store_const",
        const="finalize",
        help="merge all committed fragments and publish the completed dataset",
    )
    parser.add_argument(
        "--rank",
        dest="_distributed_rank",
        type=int,
        metavar="RANK",
        default=None,
        help="explicit zero-based generation rank (normally inferred from SLURM_PROCID)",
    )
    parser.add_argument(
        "--world-size",
        dest="_distributed_world_size",
        type=int,
        metavar="N",
        default=None,
        help="explicit number of generation ranks (normally inferred from SLURM_NTASKS)",
    )
    parser.set_defaults(_generation_mode="auto")

    ns = parser.parse_args()
    preset = ns.preset if ns.preset is not None else "mini"
    try:
        cfg = apply_preset(Config(preset=preset))
    except ValueError as exc:
        parser.error(str(exc))
    for f in dataclasses.fields(Config):
        val = getattr(ns, f.name, None)
        if val is not None:
            setattr(cfg, f.name, val)
    if cfg.enable_stdpopsim and cfg.p_stdpopsim_anchor == 0.0:
        cfg.p_stdpopsim_anchor = 0.10
        cfg.p_custom_hetero = max(0.0, cfg.p_custom_hetero - 0.10)
    try:
        validate_config(cfg)
    except ValueError as exc:
        parser.error(str(exc))
    cfg._generation_mode = ns._generation_mode
    cfg._distributed_rank = ns._distributed_rank
    cfg._distributed_world_size = ns._distributed_world_size
    return cfg


def _load_maps(cfg: Config):
    try:
        return load_empirical_maps(cfg)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[error] failed to load empirical maps: {exc}") from None


def _write_completed_dataset(cfg: Config, sample_manifest: dict) -> None:
    write_dataset_files(cfg, sample_manifest)
    print("[done] dataset written to", cfg.out_dir)
    print("       manifest:", Path(cfg.out_dir) / "manifest.json")
    print("       loader:  ", Path(cfg.out_dir) / "scripts" / "loader.py")


def main() -> None:
    cfg = parse_args()
    mode = getattr(cfg, "_generation_mode", "auto")
    explicit_rank = getattr(cfg, "_distributed_rank", None)
    explicit_world_size = getattr(cfg, "_distributed_world_size", None)
    print("=" * 80)
    print(f"DLCoalSim generator: {cfg.dataset_name}")
    print(f"out_dir={cfg.out_dir}")
    print(
        f"preset={cfg.preset} seq_len={cfg.seq_len:,} H={cfg.n_haplotypes} "
        f"target_T={cfg.time_bins} history_T={cfg.demography_generation_bins}"
    )
    print("=" * 80)
    sys.stdout.flush()

    try:
        if mode in {"prepare", "finalize"}:
            if explicit_rank is not None or explicit_world_size is not None:
                raise ValueError(
                    "--rank/--world-size apply only to generation, not "
                    "--prepare-only or --finalize-only"
                )
            if mode == "prepare":
                prepare_dataset(cfg, _load_maps(cfg))
                print("[prepared] resumable generation plan:", Path(cfg.out_dir) / "samples" / "PLAN.json")
                return

            with dataset_output_lock(cfg.out_dir, blocking=True):
                sample_manifest = finalize_dataset(
                    cfg,
                    _lock_already_held=True,
                )
                _write_completed_dataset(cfg, sample_manifest)
            return

        context = distributed_context(
            rank=explicit_rank,
            world_size=explicit_world_size,
        )
        if mode == "generate" or context.distributed:
            rank_result = generate_dataset_partition(
                cfg,
                _load_maps(cfg),
                rank=context.rank,
                world_size=context.world_size,
            )
            print(
                f"[rank done] rank={context.rank}/{context.world_size} "
                f"generated_chunks={rank_result.get('generated_chunks', 0)} "
                f"reused_chunks={rank_result.get('reused_chunks', 0)}"
            )
            return

        with dataset_output_lock(cfg.out_dir):
            sample_manifest = reuse_completed_dataset_if_compatible(
                cfg,
                _lock_already_held=True,
            )
            if sample_manifest is None:
                sample_manifest = generate_dataset(
                    cfg,
                    _load_maps(cfg),
                    _lock_already_held=True,
                )
            _write_completed_dataset(cfg, sample_manifest)
    except (
        DatasetConfigMismatchError,
        DatasetGenerationIncompleteError,
        DatasetGenerationLockedError,
        DatasetSourceUnavailableError,
        SlurmEnvironmentError,
        UnsafeOutputDirectoryError,
        ValueError,
    ) as exc:
        raise SystemExit(f"[error] {exc}") from None
    except OSError as exc:
        raise SystemExit(f"[error] filesystem operation failed: {exc}") from None
