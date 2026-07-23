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
    DatasetGenerationLockedError,
    DatasetSourceUnavailableError,
    UnsafeOutputDirectoryError,
    dataset_output_lock,
    generate_dataset,
    reuse_completed_dataset_if_compatible,
)


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
    return cfg


def main() -> None:
    cfg = parse_args()
    print("=" * 80)
    print(f"DLCoalSim generator: {cfg.dataset_name}")
    print(f"out_dir={cfg.out_dir}")
    print(f"preset={cfg.preset} seq_len={cfg.seq_len:,} H={cfg.n_haplotypes} T={cfg.time_bins}")
    print("=" * 80)
    sys.stdout.flush()

    try:
        with dataset_output_lock(cfg.out_dir):
            sample_manifest = reuse_completed_dataset_if_compatible(
                cfg,
                _lock_already_held=True,
            )
            if sample_manifest is None:
                try:
                    empirical = load_empirical_maps(cfg)
                except (OSError, ValueError) as exc:
                    raise SystemExit(f"[error] failed to load empirical maps: {exc}") from None
                sample_manifest = generate_dataset(cfg, empirical, _lock_already_held=True)
            write_dataset_files(cfg, sample_manifest)
    except (
        DatasetConfigMismatchError,
        DatasetGenerationLockedError,
        DatasetSourceUnavailableError,
        UnsafeOutputDirectoryError,
    ) as exc:
        raise SystemExit(f"[error] {exc}") from None
    except OSError as exc:
        raise SystemExit(f"[error] filesystem operation failed: {exc}") from None
    print("[done] dataset written to", cfg.out_dir)
    print("       manifest:", Path(cfg.out_dir) / "manifest.json")
    print("       loader:  ", Path(cfg.out_dir) / "scripts" / "loader.py")
