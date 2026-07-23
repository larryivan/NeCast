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

| Preset | Samples | Sequence length | Haplotypes | Workers |
| --- | ---: | ---: | ---: | ---: |
| `mini` | 512 | 1 Mb | 24 | 1 |
| `core_v01` | 20,000 | 10 Mb | 64 | all CPUs minus one |
| `full_v01` | 100,000 | 20 Mb | 64 | all CPUs minus one |

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

Unavailable empirical map or human-`Ne` template components are reported and
their weights are disabled only when another configured source remains. If an
unavailable component is the sole source, generation stops. Explicitly enabled
stdpopsim anchors also fail fast when the optional dependency or requested
model setup is unavailable.

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
