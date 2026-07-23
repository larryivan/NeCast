from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from .config import Config, DEMO_IDS, SOURCE_IDS, validate_config
from .utils import time_grid


README_TEMPLATE = """# {dataset_name}

DLCoalSim is a minimally processed coalescent simulation dataset for demographic
history inference from haplotype data.

## Core Philosophy

The dataset stores simulated observations, not model-specific feature engineering:

- bit-packed haplotype genotypes
- sparse missing-genotype coordinates
- diploid individuals represented by adjacent haplotype columns
- variant positions
- observed recombination/mutation maps
- target log10 Ne(t), averaged within each logarithmic time bin
- structured, reproducible sample metadata

Handcrafted features such as SFS, LD, pi, and haplotype diversity are intentionally
not precomputed in the core dataset.

## Dataset Layout

- `samples/shard_*.npz`: sample shards with ragged arrays encoded by offsets
- `samples/shard_*.jsonl.gz`: full per-sample metadata aligned to each shard
- `metadata/samples.csv`: flat metadata summary for filtering and benchmark recipes
- `metadata/samples.jsonl.gz`: full metadata for all samples
- `metadata/samples.parquet`: optional summary parquet when the local environment supports it
- `config.json` and `manifest.json`: generation config and dataset manifest

Shard files are written as lossless NPZ archives using the configured DEFLATE
level. Missing genotypes are stored as sparse per-sample row-major indices
(`missing_flat_idx`, `missing_offsets`) rather than as a dense missing matrix.

Targets are log-time bin averages, not midpoint samples. This keeps short
events such as recent bottlenecks visible in labels when they overlap a time bin.
Main simulator samples are generated as diploid individuals (`ploidy=2`), so
`n_haplotypes` must be even. Phase switch noise swaps the two haplotypes within
each diploid individual after sampled switch points; this is recorded as
`phase_switch_model="diploid_pair_switch"` and `phase_pairing`.
`scenario_key` combines demography, map mode, and noise profile for convenient
filtering without making fixed train/validation/test splits.

Demography sampling uses a registry of named samplers rather than one large
conditional block. It includes constant and near-constant controls, recent
bottlenecks, recent founder recovery, continuous exponential growth/decline
approximated by multiple epochs, serial founder events, ancient+recent compound
events, mild/strong oscillating histories, dense bin-aligned histories, classic
sawtooth histories, and optional empirical human Ne templates inferred by tools
such as SMC++, PHLaSH, and MSMC2. Metadata includes summary fields
such as `n_epochs`, `has_recent_event`, `has_ancient_event`, `min_Ne`, `max_Ne`,
`Ne_ratio_max_min`, `recent_min_Ne`, `ancient_mean_Ne`, `event_severity`, and
`event_duration` for filtering and benchmark recipes.
Sample quality metadata includes `variant_density_per_mb`, explicit noise rates,
missing genotype counts, phase switch counts, and observed map rate summaries
(`mean/std/min/max`) for quick corpus filtering without reading the shard arrays.

stdpopsim samples are optional anchors/stress samples. They are disabled by
default (`p_stdpopsim_anchor=0`) and are included only when requested with
`--enable-stdpopsim`. Their target metadata keeps `target_quality` because many
stdpopsim models include population structure, migration, or admixture, so these
anchors should not be treated as the same strict single-population supervision as
the custom simulator samples.
Two-population migration/admixture stress cases should follow the same anchor
pattern and be labeled with proxy target quality rather than mixed into the strict
single-population supervised core.

## Official Model Inputs

Allowed:

- genotype matrix / haplotypes
- variant positions
- observed recombination map
- observed mutation map
- sequence length and sample size

Not allowed as model input:

- true maps, if included
- demography metadata
- source type
- noise parameters
- tree sequence or TMRCA truth
- target Ne(t)

## Tutorials

Start with:

1. `notebooks/00_quickstart.ipynb` - load a shard and inspect one sample.
2. `notebooks/01_visualize_single_sample.ipynb` - visualize genotypes, maps, and target Ne(t).
3. `notebooks/02_dataset_overview.ipynb` - inspect metadata and dataset composition.
4. `notebooks/03_filter_by_metadata.ipynb` - build custom subsets from sample annotations.
5. `notebooks/04_compute_basic_features.ipynb` - compute simple baseline window features from raw data.
6. `notebooks/05_quality_check_preview.ipynb` - run quick metadata and shard sanity checks.
"""


FEATURES_PY = r'''
"""Baseline feature recipes for DLCoalSim."""
import numpy as np


def unpack_genotypes(genotype_packed, n_haplotypes):
    bits = np.unpackbits(genotype_packed, axis=1, bitorder="little")
    return bits[:, :n_haplotypes].astype(np.uint8)


def window_indices(positions_bp, seq_len, n_windows):
    w = np.floor(positions_bp.astype(float) / (seq_len / n_windows)).astype(int)
    return np.clip(w, 0, n_windows - 1)


def basic_window_stats(positions_bp, G_vh, seq_len, n_windows=512):
    out = np.zeros((n_windows, 4), dtype=np.float32)
    if len(positions_bp) == 0:
        return out
    H = G_vh.shape[1]
    wi = window_indices(positions_bp, seq_len, n_windows)
    for w in range(n_windows):
        idx = np.where(wi == w)[0]
        if len(idx) == 0:
            continue
        X = G_vh[idx]
        ac = X.sum(axis=1)
        p = ac / max(1, H)
        out[w, 0] = len(idx) / max(1.0, seq_len / n_windows)
        out[w, 1] = np.mean(2 * p * (1 - p))
        informative = (ac > 0) & (ac < H)
        out[w, 2] = np.mean(ac[informative] == 1) if informative.any() else 0.0
        out[w, 3] = np.mean(p) if len(p) else 0.0
    return out
'''


LOADER_PY = r'''
"""Tiny loader example for DLCoalSim shards."""
import gzip
import json
from pathlib import Path

import numpy as np


class DLCoalSimShard:
    def __init__(self, shard_path):
        self.shard_path = Path(shard_path)
        self.data = np.load(self.shard_path, allow_pickle=False)
        self.n = len(self.data["sample_id"])
        self.n_haplotypes = int(self.data["n_haplotypes"][0])
        self.packed_hap_bytes = int(self.data["packed_hap_bytes"][0])
        meta_path = self.shard_path.with_suffix(".jsonl.gz")
        self.metadata = []
        if meta_path.exists():
            with gzip.open(meta_path, "rt", encoding="utf-8") as f:
                self.metadata = [json.loads(line) for line in f]

    def __len__(self):
        return self.n

    def _slice_offsets(self, key, offsets_key, i):
        off = self.data[offsets_key]
        return self.data[key][off[i]:off[i + 1]]

    def get(self, i, unpack=True):
        v0, v1 = self.data["variant_offsets"][i], self.data["variant_offsets"][i + 1]
        pos = self.data["variant_positions_bp"][v0:v1]
        packed = self.data["genotype_packed"][v0:v1]
        m0, m1 = self.data["missing_offsets"][i], self.data["missing_offsets"][i + 1]
        missing_flat_idx = self.data["missing_flat_idx"][m0:m1].astype(np.int64, copy=False)
        if unpack:
            G = np.unpackbits(packed, axis=1, bitorder="little")[:, :self.n_haplotypes].astype(np.uint8)
            M = np.zeros((len(pos), self.n_haplotypes), dtype=np.uint8)
            if len(missing_flat_idx):
                M[missing_flat_idx // self.n_haplotypes, missing_flat_idx % self.n_haplotypes] = 1
        else:
            G = packed
            M = {
                "storage": "sparse_flat_index_uint32",
                "flat_idx": missing_flat_idx,
            }
        seq = self.data["sequence_length"]
        seq_len = int(seq[i] if len(seq) > 1 else seq[0])
        sample = {
            "sample_id": str(self.data["sample_id"][i]),
            "sequence_length": seq_len,
            "n_haplotypes": self.n_haplotypes,
            "positions_bp": pos,
            "genotypes": G,
            "missing_mask": M,
            "observed_recombination": {
                "position_bp": self._slice_offsets("obs_rec_pos", "obs_rec_pos_offsets", i),
                "rate": self._slice_offsets("obs_rec_rate", "obs_rec_rate_offsets", i),
            },
            "observed_mutation": {
                "position_bp": self._slice_offsets("obs_mut_pos", "obs_mut_pos_offsets", i),
                "rate": self._slice_offsets("obs_mut_rate", "obs_mut_rate_offsets", i),
            },
            "target_log10_ne": self.data["target_log10_ne"][i],
        }
        if self.metadata:
            sample["metadata"] = self.metadata[i]
        return sample
'''


VIZ_PY = r'''
"""Reusable visualization helpers for DLCoalSim tutorial notebooks."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:  # pragma: no cover - only used outside notebooks
    display = print

try:
    from .loader import DLCoalSimShard
except ImportError:
    from loader import DLCoalSimShard


def plot_target_curve(sample, time_mids=None):
    y = np.asarray(sample["target_log10_ne"], dtype=float)
    if time_mids is None:
        x = np.arange(len(y))
        x_label = "Time bin"
        log_x = False
    else:
        x = np.asarray(time_mids, dtype=float)[: len(y)]
        x_label = "Generations ago"
        log_x = True

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", linewidth=1)
    if log_x:
        plt.xscale("log")
    plt.xlabel(x_label)
    plt.ylabel("log10 Ne")
    plt.title(f"Target Ne(t): {sample.get('sample_id', '')}")
    plt.tight_layout()
    plt.show()


def plot_genotype_heatmap(sample, max_variants=300, max_haplotypes=64):
    G = np.asarray(sample["genotypes"])
    if G.size == 0:
        print("No variants in this sample.")
        return

    G_show = G[:max_variants, :max_haplotypes]
    plt.figure(figsize=(8, 5))
    plt.imshow(G_show, aspect="auto", interpolation="nearest")
    plt.xlabel("Haplotypes")
    plt.ylabel("Variants")
    plt.title("Haplotype genotype matrix")
    plt.tight_layout()
    plt.show()


def plot_variant_density(sample, n_windows=200):
    pos = np.asarray(sample["positions_bp"])
    seq_len = int(sample["sequence_length"])
    if len(pos) == 0:
        print("No variants in this sample.")
        return

    bins = np.linspace(0, seq_len, n_windows + 1)
    counts, _ = np.histogram(pos, bins=bins)
    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(8, 3))
    plt.plot(centers, counts)
    plt.xlabel("Position (bp)")
    plt.ylabel("Variant count")
    plt.title("Variant density")
    plt.tight_layout()
    plt.show()


def _plot_piecewise_map(map_dict, title, ylabel):
    pos = np.asarray(map_dict["position_bp"], dtype=float)
    rate = np.asarray(map_dict["rate"], dtype=float)
    if len(pos) < 2 or len(rate) == 0:
        print(f"No map data for {title}.")
        return

    x = pos[:-1]
    plt.figure(figsize=(8, 3))
    plt.step(x, rate, where="post")
    plt.xlabel("Position (bp)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_observed_maps(sample):
    _plot_piecewise_map(
        sample["observed_recombination"],
        "Observed recombination map",
        "Rate per bp per generation",
    )
    _plot_piecewise_map(
        sample["observed_mutation"],
        "Observed mutation map",
        "Rate per bp per generation",
    )


def show_metadata_card(sample):
    meta = sample.get("metadata", {})
    keys = [
        "sample_id",
        "source_type",
        "scenario_key",
        "demography_type",
        "map_mode",
        "noise_profile",
        "n_variants",
        "variant_density_per_mb",
        "num_trees",
        "n_epochs",
        "has_recent_event",
        "has_ancient_event",
        "min_Ne",
        "max_Ne",
        "Ne_ratio_max_min",
        "event_severity",
        "event_duration",
        "genotype_error",
        "missing_rate",
        "n_missing_genotypes",
        "phase_switch_pair_count",
    ]
    rows = [(k, meta.get(k, "")) for k in keys if k in meta]
    display(pd.DataFrame(rows, columns=["field", "value"]))


def plot_column_distribution(df, column):
    counts = df[column].value_counts(dropna=False)
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()


def plot_numeric_distribution(df, column, bins=50):
    plt.figure(figsize=(6, 4))
    df[column].dropna().hist(bins=bins)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()


def _as_shards(dataset_or_loader):
    if hasattr(dataset_or_loader, "get"):
        return [dataset_or_loader]
    if isinstance(dataset_or_loader, (str, Path)):
        root = Path(dataset_or_loader)
        if root.is_file():
            return [DLCoalSimShard(root)]
        return [DLCoalSimShard(p) for p in sorted((root / "samples").glob("shard_*.npz"))]
    return list(dataset_or_loader)


def _sample_by_id(shards, sample_id):
    target = str(sample_id)
    for shard in shards:
        ids = [str(x) for x in shard.data["sample_id"]]
        if target in ids:
            return shard.get(ids.index(target))
    raise KeyError(f"sample_id not found in provided shards: {sample_id}")


def plot_random_targets(dataset_or_loader, metadata, n=20, time_mids=None, random_state=0):
    df = metadata.sample(min(n, len(metadata)), random_state=random_state)
    shards = _as_shards(dataset_or_loader)

    plt.figure(figsize=(7, 4))
    for _, row in df.iterrows():
        sample = _sample_by_id(shards, row["sample_id"])
        y = np.asarray(sample["target_log10_ne"], dtype=float)
        if time_mids is None:
            x = np.arange(len(y))
        else:
            x = np.asarray(time_mids, dtype=float)[: len(y)]
        plt.plot(x, y, alpha=0.45, linewidth=1)
    if time_mids is not None:
        plt.xscale("log")
        plt.xlabel("Generations ago")
    else:
        plt.xlabel("Time bin")
    plt.ylabel("log10 Ne")
    plt.title(f"Random target curves (n={len(df)})")
    plt.tight_layout()
    plt.show()
'''


VALIDATE_DATASET_PY = r'''
"""Quick DLCoalSim dataset validation checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .loader import DLCoalSimShard
except ImportError:
    from loader import DLCoalSimShard


def _check_monotone(name, arr):
    if len(arr) > 1 and np.any(np.diff(arr) < 0):
        raise AssertionError(f"{name} positions are not monotone")


def validate_dataset(data_dir):
    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata" / "samples.csv"
    manifest_path = data_dir / "manifest.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    df = pd.read_csv(meta_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert df["sample_id"].is_unique, "sample_id values must be unique"
    assert df["n_variants"].ge(0).all(), "n_variants must be non-negative"
    assert df["sequence_length"].gt(0).all(), "sequence_length must be positive"
    for column in ["mean_obs_rec_rate", "mean_obs_mut_rate", "variant_density_per_mb"]:
        if column in df:
            assert np.isfinite(df[column]).all(), f"{column} contains non-finite values"

    shards = sorted((data_dir / "samples").glob("shard_*.npz"))
    assert shards, "no shard_*.npz files found"
    seen = 0
    for shard_path in shards:
        shard = DLCoalSimShard(shard_path)
        for i in range(len(shard)):
            sample = shard.get(i, unpack=False)
            assert np.isfinite(sample["target_log10_ne"]).all(), "target contains non-finite values"
            assert sample["sequence_length"] > 0, "sample sequence_length must be positive"
            _check_monotone("variant", sample["positions_bp"])
            _check_monotone("observed recombination", sample["observed_recombination"]["position_bp"])
            _check_monotone("observed mutation", sample["observed_mutation"]["position_bp"])
            missing = sample["missing_mask"]
            mf = np.asarray(missing["flat_idx"], dtype=np.int64)
            if len(mf):
                assert int(mf.min()) >= 0, "missing flat index out of range"
                assert int(mf.max()) < len(sample["positions_bp"]) * sample["n_haplotypes"], "missing flat index out of range"
                assert np.all(np.diff(mf) >= 0), "missing flat indices must be sorted"
            seen += 1

    expected = int(manifest.get("samples", {}).get("n_samples", len(df)))
    assert seen == expected == len(df), f"sample count mismatch: shards={seen}, metadata={len(df)}, manifest={expected}"
    print(f"OK: {seen} samples validated in {data_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default=".", help="DLCoalSim dataset directory")
    args = parser.parse_args(argv)
    validate_dataset(args.data_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
'''


def _source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").splitlines()]


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(text)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source_lines(text)}


def _notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = r'''
from pathlib import Path
import json
import sys

DATA_DIR = Path("..").resolve()
if not (DATA_DIR / "manifest.json").exists():
    DATA_DIR = Path(".").resolve()
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

SAMPLES_DIR = DATA_DIR / "samples"
META_PATH = DATA_DIR / "metadata" / "samples.csv"
SHARD_PATH = sorted(SAMPLES_DIR.glob("shard_*.npz"))[0]
CONFIG = json.loads((DATA_DIR / "config.json").read_text())
TIME_MIDS = CONFIG.get("time_mids")
'''


def tutorial_notebooks() -> dict[str, dict]:
    intro = """This notebook is part of the DLCoalSim tutorial suite.

DLCoalSim stores minimally processed simulated observations rather than model-specific precomputed statistics. You are encouraged to compute your own features from haplotypes, variant positions, and observed maps."""

    return {
        "00_quickstart.ipynb": _notebook(
            [
                _md(f"# DLCoalSim Quickstart\n\n{intro}\n\nThis notebook shows how to load one shard, inspect one sample, and access haplotypes, variant positions, observed maps, target log10 Ne(t), and metadata."),
                _code(COMMON_SETUP),
                _code("import pandas as pd\n\nfrom scripts.loader import DLCoalSimShard"),
                _code("df = pd.read_csv(META_PATH)\ndf.head()"),
                _code("shard = DLCoalSimShard(SHARD_PATH)\nsample = shard.get(0)\n\nsample.keys()"),
                _code(
                    'print("sample_id:", sample["sample_id"])\n'
                    'print("sequence_length:", sample["sequence_length"])\n'
                    'print("positions:", sample["positions_bp"].shape)\n'
                    'print("genotypes:", sample["genotypes"].shape)\n'
                    'print("target:", sample["target_log10_ne"].shape)\n'
                    'print("metadata keys:", list(sample["metadata"].keys())[:20])'
                ),
            ]
        ),
        "01_visualize_single_sample.ipynb": _notebook(
            [
                _md(f"# Visualize One Sample\n\n{intro}\n\nThis notebook visualizes the raw observations and target for one sample."),
                _code(COMMON_SETUP),
                _code(
                    "import pandas as pd\n\n"
                    "from scripts.loader import DLCoalSimShard\n"
                    "from scripts.viz import (\n"
                    "    plot_target_curve,\n"
                    "    plot_genotype_heatmap,\n"
                    "    plot_variant_density,\n"
                    "    plot_observed_maps,\n"
                    "    show_metadata_card,\n"
                    ")"
                ),
                _code("df = pd.read_csv(META_PATH)\nshard = DLCoalSimShard(SHARD_PATH)\nsample = shard.get(0)\nsample['metadata']"),
                _md("The target is a log-time bin average of log10 Ne, not a single midpoint sample."),
                _code("plot_target_curve(sample, time_mids=TIME_MIDS)"),
                _md("Rows are variant sites and columns are haplotypes. Adjacent haplotype columns are the two haplotypes of one diploid individual."),
                _code("plot_genotype_heatmap(sample, max_variants=300, max_haplotypes=64)"),
                _code("plot_variant_density(sample, n_windows=200)"),
                _code("plot_observed_maps(sample)"),
                _code("show_metadata_card(sample)"),
            ]
        ),
        "02_dataset_overview.ipynb": _notebook(
            [
                _md(f"# Dataset Overview\n\n{intro}\n\nThis notebook reads metadata only for most plots, then samples a few target curves from shards."),
                _code(COMMON_SETUP),
                _code("import pandas as pd\n\nfrom scripts.viz import plot_column_distribution, plot_numeric_distribution, plot_random_targets"),
                _code("df = pd.read_csv(META_PATH)\ndf.head()"),
                _code("plot_column_distribution(df, 'source_type')"),
                _code("plot_column_distribution(df, 'demography_type')"),
                _code("plot_column_distribution(df, 'map_mode')"),
                _code("plot_column_distribution(df, 'noise_profile')"),
                _code("plot_numeric_distribution(df, 'n_variants', bins=50)"),
                _code("plot_numeric_distribution(df, 'Ne_ratio_max_min', bins=50)"),
                _code("plot_numeric_distribution(df, 'variant_density_per_mb', bins=50)"),
                _code("plot_random_targets(DATA_DIR, df, n=20, time_mids=TIME_MIDS, random_state=0)"),
            ]
        ),
        "03_filter_by_metadata.ipynb": _notebook(
            [
                _md(f"# Filter By Metadata\n\n{intro}\n\nThe simulator creates a fully annotated sample pool. Use metadata to create your own train/validation/test or stress subsets."),
                _code(COMMON_SETUP),
                _code("import pandas as pd\n\ndf = pd.read_csv(META_PATH)\ndf.head()"),
                _code("recent = df[df['has_recent_event'] == True]\nrecent.head()"),
                _code("clean = df[df['source_type'] == 'clean_constant_map']\nclean.head()"),
                _code("strong = df[df['Ne_ratio_max_min'] > 20]\nstrong.head()"),
                _code("empirical = df[df['map_mode'] == 'empirical_slice']\nempirical.head()"),
                _code("high_quality = df[(df['variant_density_per_mb'] > 100) & (df['missing_rate'] < 0.01)]\nhigh_quality.head()"),
                _code(
                    "OUT = DATA_DIR / 'benchmarks' / 'tutorial_examples'\n"
                    "OUT.mkdir(parents=True, exist_ok=True)\n"
                    "recent['sample_id'].to_csv(OUT / 'recent_samples.txt', index=False, header=False)\n"
                    "strong['sample_id'].to_csv(OUT / 'strong_demography_samples.txt', index=False, header=False)\n"
                    "OUT"
                ),
            ]
        ),
        "04_compute_basic_features.ipynb": _notebook(
            [
                _md(f"# Compute Basic Features\n\n{intro}\n\nDLCoalSim-Core does not precompute SFS, LD, pi, or haplotype features. This notebook shows a baseline window feature recipe that you can replace."),
                _code(COMMON_SETUP),
                _code("import matplotlib.pyplot as plt\n\nfrom scripts.loader import DLCoalSimShard\nfrom scripts.features import basic_window_stats"),
                _code("shard = DLCoalSimShard(SHARD_PATH)\nsample = shard.get(0)\n\nG = sample['genotypes']\npos = sample['positions_bp']\nseq_len = sample['sequence_length']\n\nX = basic_window_stats(pos, G, seq_len, n_windows=512)\nX.shape"),
                _code("plt.figure(figsize=(8, 3))\nplt.plot(X[:, 0])\nplt.xlabel('Window')\nplt.ylabel('SNP density')\nplt.tight_layout()\nplt.show()"),
                _code("plt.figure(figsize=(8, 3))\nplt.plot(X[:, 1])\nplt.xlabel('Window')\nplt.ylabel('Pi proxy')\nplt.tight_layout()\nplt.show()"),
                _code("plt.figure(figsize=(8, 3))\nplt.plot(X[:, 2])\nplt.xlabel('Window')\nplt.ylabel('Singleton fraction')\nplt.tight_layout()\nplt.show()"),
                _md("The output `X` is only a baseline recipe. For model development, build your own feature extractor from haplotypes, variant positions, and observed maps."),
            ]
        ),
        "05_quality_check_preview.ipynb": _notebook(
            [
                _md(f"# Quality Check Preview\n\n{intro}\n\nUse this notebook for quick sanity checks before sharing or training on a generated dataset."),
                _code(COMMON_SETUP),
                _code("import numpy as np\nimport pandas as pd\n\nfrom scripts.validate_dataset import validate_dataset"),
                _code("df = pd.read_csv(META_PATH)\ndf.describe(include='all')"),
                _code("assert df['n_variants'].ge(0).all()\nassert df['sequence_length'].gt(0).all()\nassert np.isfinite(df['variant_density_per_mb']).all()\nassert np.isfinite(df['mean_obs_rec_rate']).all()\nassert np.isfinite(df['mean_obs_mut_rate']).all()\nprint('metadata checks passed')"),
                _code("df[['source_type', 'demography_type', 'map_mode', 'noise_profile']].apply(lambda s: s.value_counts()).fillna(0)"),
                _code("df[['n_variants', 'variant_density_per_mb', 'missing_rate', 'n_missing_genotypes', 'mean_obs_rec_rate', 'mean_obs_mut_rate', 'Ne_ratio_max_min']].hist(figsize=(12, 8), bins=40);"),
                _code("validate_dataset(DATA_DIR)"),
            ]
        ),
    }


def write_dataset_files(cfg: Config, sample_manifest: dict) -> None:
    validate_config(cfg)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    manifest_path.with_name(f"{manifest_path.name}.tmp").unlink(missing_ok=True)

    def write_atomic(path: Path, text: str) -> None:
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)

    edges, mids = time_grid(cfg)
    config_dict = dataclasses.asdict(cfg)
    config_dict["time_edges"] = edges.tolist()
    config_dict["time_mids"] = mids.tolist()
    config_dict["source_ids"] = SOURCE_IDS
    config_dict["demography_ids"] = DEMO_IDS
    write_atomic(
        out / "config.json",
        json.dumps(config_dict, indent=2, ensure_ascii=False),
    )

    manifest = {
        "dataset_name": cfg.dataset_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples": sample_manifest,
        "time_edges": edges.tolist(),
        "time_mids": mids.tolist(),
        "source_ids": SOURCE_IDS,
        "demography_ids": DEMO_IDS,
    }
    (out / "README.md").write_text(README_TEMPLATE.format(dataset_name=cfg.dataset_name), encoding="utf-8")

    scripts = out / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "__init__.py").write_text("", encoding="utf-8")
    (scripts / "features.py").write_text(FEATURES_PY, encoding="utf-8")
    (scripts / "loader.py").write_text(LOADER_PY, encoding="utf-8")
    (scripts / "viz.py").write_text(VIZ_PY, encoding="utf-8")
    (scripts / "validate_dataset.py").write_text(VALIDATE_DATASET_PY, encoding="utf-8")

    notebooks = out / "notebooks"
    notebooks.mkdir(exist_ok=True)
    for name, notebook in tutorial_notebooks().items():
        (notebooks / name).write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")

    # The root manifest is the final publication marker for all dataset support
    # files; publish it atomically only after every other artifact succeeds.
    write_atomic(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False),
    )
