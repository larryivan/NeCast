# DLCoalSim

DLCoalSim generates minimally processed coalescent-simulation datasets for
demographic-history inference. Each dataset contains bit-packed haplotypes,
variant positions, observed mutation and recombination maps, binned
`log10(Ne)` targets, and aligned metadata.

## Install

Python 3.10 or newer is required. An editable install from the repository keeps
the version-controlled empirical maps and human `Ne` templates available at
their default paths:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

Install optional features as needed:

```bash
# Tests
python3 -m pip install -e ".[test]"

# Parquet output, stdpopsim anchors, notebooks, and map-processing tools
python3 -m pip install -e ".[parquet,stdpopsim,tutorial,map-processing,test]"
```

`pigz` is an optional system dependency for mutation-map preprocessing; install
it with the operating-system package manager rather than pip.

## Generate a dataset

Inspect all configuration flags:

```bash
dlcoalsim --help
# Equivalent from a source checkout:
python3 -m simulator --help
```

A small smoke run can avoid loading the large empirical maps:

```bash
dlcoalsim \
  --preset mini \
  --n 4 \
  --shard-size 2 \
  --seq-len 100000 \
  --workers 1 \
  --recomb-map "" \
  --mut-map "" \
  --out-dir ./DLCoalSim-Smoke
```

The built-in presets are:

| Preset | Samples | Sequence length | Haplotypes | Target bins | History-generation bins | Workers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mini` | 512 | 1 Mb | 24 | 64 | 32 | 1 |
| `core_v01` | 20,000 | 10 Mb | 64 | 64 | 48 | all CPUs minus one |
| `full_v01` | 100,000 | 20 Mb | 64 | 64 | 48 | all CPUs minus one |

All built-in presets use a 64-bin supervised target. `time_bins` controls only
that target grid. The separate `demography_generation_bins` setting preserves
the population-history prior that each preset used before the 64-bin upgrade
(32 for `mini`, 48 for `core_v01` and `full_v01`). Consequently, changing
target resolution does not change the same-seed latent history, simulated
genotypes, maps, or observation noise. `config.json` and `manifest.json` record
the edges and midpoints of both grids.

For a custom schema-v3 command, the old `time_bins` value controlled both the
target and the population-history sampler. To reproduce that old latent prior
under schema v4, pass the old value explicitly as
`--demography-generation-bins`; choose the new target resolution separately
with `--time-bins` (64 for the current training format).

Use a new output directory for every distinct configuration. A completed
directory records a SHA-256 fingerprint of every sample-generation setting and
the sample-generator schema, then verifies the recorded shard layout before
reuse. Changed parameters, generator semantics, missing shards, extra stale
shards, and incompatible shard structure are rejected instead of silently
reusing data. Interrupted runs have no valid `DONE.json`;
their generated fragments are removed automatically on the next run. An
output-directory lock prevents two generators from writing the same dataset at
once. Outputs created before configuration fingerprints were introduced require
explicit regeneration with `--force`; a legacy mutable `config.json` is not
trusted to certify old shards.

Use `--force` only when intentional regeneration is required; it invalidates
the completion marker before removing simulator-generated shard and metadata
files, so verify the target path first. Filesystem roots, the repository root,
the home directory, and non-empty directories without a DLCoalSim marker are
rejected as output targets.

For large presets, first run a small pilot with an explicit `--workers` value to
measure memory, runtime, and disk use. With multiple workers, parsed empirical
map arrays are placed in read-only shared memory rather than loaded once per
worker.

The first empirical-map load builds a content-addressed binary cache. Later
runs verify the source-file, parser, dependency, and cached-array hashes before
memory-mapping the cached arrays. Set `DLCOALSIM_CACHE_DIR` to place this cache
somewhere other than the platform user-cache directory.

Simulation work is scheduled in bounded chunks independently of the final shard
layout. `--compute-chunk-size` controls that scheduling granularity without
changing sample seeds or final arrays; single-worker runs automatically use one
compute task per final shard. Shards use lossless DEFLATE level 1 by default.
Use `--compression-level 0..9` to trade write time for file size without
changing any decoded array values.

Unavailable empirical map or human-`Ne` template components are reported and
their weights are disabled only when another configured source remains. If an
unavailable component is the sole source, generation stops. Explicitly enabled
stdpopsim anchors also fail fast when the optional dependency or requested
model setup is unavailable.

Single-population continuous-growth scenarios use msprime's native
linear-time exponential convention, `N(t) = N(0) exp(-g t)`, rather than a
stepwise approximation. Full epoch growth rates are stored in sample metadata,
and targets are analytic log-time-bin averages of that same trajectory.

Demographic metadata has two explicit levels. `demography_type` is the compact
public scientific label (`constant`, `bottleneck`, `exponential_growth`,
`zigzag`, and so on), while `generator_recipe` preserves the exact
fine-grained sampler name for reproducibility. Shards store
`demography_type_id` for the public type and `generator_recipe_id` for the
recipe; the legacy `demography_id` remains a recipe-ID alias so its old numeric
namespace never changes. All mappings are embedded in `config.json` and
`manifest.json`. Shape and difficulty are orthogonal
metadata (`history_family`, `trajectory_form`, `direction`, `time_regime`,
`log10_ne_range`, `total_variation_log10`, and `turn_count`) rather than extra
population-history classes. `event_severity` is consistently defined as
`min_Ne / max_Ne`; the former recipe-specific value remains available as
`recipe_event_severity`.

This is generator schema v4. It upgrades every built-in target to 64 bins and
separates target resolution from latent-history generation. Existing v3
outputs keep their original 32- or 48-bin targets and must be regenerated in a
new directory or with an intentional `--force` to use the v4 format. Every
shard stores its schema version; numeric IDs should always be decoded with that
dataset's own manifest.

`canonical_zigzag_1s14` and `canonical_smcpp_sawtooth` preserve the two
slightly different published validation models as fixed anchors. Their default
mixture weights are zero so a benchmark can enable them explicitly without
leaking the exact evaluation curves into the default training distribution.
These flags fix the demographic trajectory only; reproducing an entire paper
benchmark still requires that benchmark's sequence layout, sample size, rates,
error model, and replicate protocol.
`classic_sawtooth` is a scaled, literature-shaped training family, while
`oscillating_mild` and `zigzag_strong` generate randomized continuous
exponential expansions and contractions. The legacy `zigzag` configuration key
is an exact alias of `zigzag_strong`; it remains active at its historical low
weight so existing seeds continue to select the same recipes.
The canonical models' first rate change is at 33.33 generations; set
`--min-time` to 33 or lower when a benchmark must score that most-recent epoch
(the default target grid starts at 50 generations, although the simulation
itself still contains the event).

## Validate output

Every generated dataset includes a standalone validator:

```bash
python3 ./DLCoalSim-Smoke/scripts/validate_dataset.py ./DLCoalSim-Smoke
```

The repository admin tool performs stricter shard and metadata checks:

```bash
dlcoalsim-dataset-admin check ./DLCoalSim-Smoke
# Equivalent from a source checkout:
python3 scripts/dataset_admin.py check ./DLCoalSim-Smoke
```

If corrupt shards are found, the admin command interactively offers to update
metadata and move them into quarantine; it is not strictly read-only in that
case.

## Test

```bash
python3 -m pytest
```

For a reproducible production run, record the Git commit, Python version,
installed package versions, and checksums of any external empirical maps
alongside the generated dataset.
