# DLCoalSim

Coalescent simulation of demographic histories, for training and evaluating
methods that infer effective population size through time, N<sub>e</sub>(t),
from sequence data.

A row is one simulated sample: bit-packed haplotypes, variant positions, the
recombination and mutation maps *as a method would receive them* rather than as
they truly were, and a 64-bin log-time target. Nothing is precomputed into
summary statistics — the point is that a method computes its own.

```bash
pip install -e .
dlcoalsim --preset mini --out-dir runs/first
```

That writes 512 rows in a few minutes. `--preset core_v01` is 20,000 rows at
10 Mb; `--preset full_v01` is 100,000 rows at 20 Mb.

## Reading a dataset

```
runs/first/
  samples/shard_00000.npz        numerical arrays, 128 rows a shard
  samples/shard_00000.jsonl.gz   one metadata object per row
  metadata/samples.csv           the same metadata, whole dataset
  config.json  manifest.json     what produced it
  scripts/loader.py              a minimal reader, written beside the data
```

```python
import numpy as np

shard = np.load("runs/first/samples/shard_00000.npz")
row = 0
lo, hi = shard["variant_offsets"][row], shard["variant_offsets"][row + 1]

target    = shard["target_log10_ne"][row]              # (64,) log10 Ne per time bin
positions = shard["variant_positions_bp"][lo:hi]       # uint32
packed    = shard["genotype_packed"][lo:hi]            # (variants, 8) uint8
n_haplotypes = int(shard["n_haplotypes"][0])
haplotypes = np.unpackbits(packed, axis=1, bitorder="little")[:, :n_haplotypes]
```

Time bins are log-spaced from 50 to 500,000 generations; `time_edges` in
`manifest.json` gives the boundaries. Genotypes are packed one bit per
haplotype, `n_haplotypes` to a row. Missing calls are a sparse flat index (`missing_flat_idx`,
`missing_offsets`) rather than a mask.

A map stores one offsets array, not two. Its `*_pos` array holds the
breakpoints and `*_rate` the rates *between* them, so a row holds one fewer
rate than positions — none of either where a row has no map:

```python
offsets = shard["obs_rec_offsets"]
lengths = np.diff(offsets.astype(np.int64))
rate_at = np.concatenate(([0], np.cumsum(np.maximum(lengths - 1, 0))))

breakpoints = shard["obs_rec_pos"][offsets[row]:offsets[row + 1]]
rates       = shard["obs_rec_rate"][rate_at[row]:rate_at[row + 1]]
```

The maps come in two forms. `obs_rec_*` and `obs_mut_*` are what a method is
allowed to see: a published map's resolution, its error, and its overall scale
offset. `true_rec_*` and `true_mut_*` are what the sequence was actually
simulated under, written for `--truth-fraction` of rows — 2% by default — so
the gap between the two can be measured on a subsample. **Training on the true
maps is cheating.**

## What a row can be

Two mixtures are drawn independently, so any history can appear under any map.

### Demographic history

| recipe | weight | what it is |
| --- | --- | --- |
| `spectral_walk` | 0.200 | Band-limited Fourier series on continuous log time. Spectral slope sets roughness, an explicit log<sub>10</sub> range sets amplitude, an envelope tilt sets the recent–ancient asymmetry. |
| `empirical_human_ne_template` | 0.190 | 191 curves inferred from real human data by SMC++, phlash, MSMC2 and MSMC-IM, with time warp, level shift, contrast and smoothed noise. |
| `size_event` | 0.100 | One size change: time, duration, signed depth, and the level it settles at. A duration running past the window is a permanent step. |
| `bottleneck` | 0.100 | A drop of 0.4 to 3 decades that ends and recovers. |
| `event_on_background` | 0.086 | A quieter spectral walk with one or two sharp events on it. |
| `multi_event` | 0.080 | Two to four events with independent signs. |
| `ooa_chain` | 0.080 | Ancestral size, ancient expansion, a bottleneck held open below the ancestral level, then recent exponential growth — ordered. |
| `constant` | 0.040 | A single effective size. The control. |
| `exponential` | 0.040 | A signed rate through msprime's own `growth_rate`, so the trajectory is exponential in linear time. |
| `sawtooth` | 0.030 | Periodic growth and decline, with strength, placement, extent, regularity and amplitude drift all drawn. |
| `literature_sawtooth` | 0.030 | The Zigzag_1S14 character: a drawn number of half cycles, spaced about fourfold, each about a decade deep. |
| `stdpopsim_single_pop` | 0.024 | All 18 single-population stdpopsim models across 8 species, verbatim. Rows carry `held_out_evaluation`. |

Sizes span 10<sup>2</sup> to 10<sup>6</sup>, and the level prior is near-uniform
across that range — the bottom decade is real (a vaquita's effective size is
under three thousand) and a method that has never seen it cannot be trusted
there.

### Maps

| source | weight | what it is |
| --- | --- | --- |
| `custom_heterogeneous` | 0.55 | Synthetic: hotspots at 60 per Mb, median width 498 bp, carrying 2.0% of sequence above ten times the mean, on an Ornstein–Uhlenbeck background. Calibrated against deCODE 2019 measured on chr20. |
| `empirical_map_slice` | 0.20 | A random interval of the real deCODE and Roulette maps, taken at the same chromosomal coordinates so the two stay in register. |
| `random_global_mu_r` | 0.15 | Flat maps, recombination drawn 0.3–3× baseline and mutation 0.5–2×. |
| `clean_constant_map` | 0.10 | Flat at baseline, no observation noise. The control. |

Observation error is a *fraction of each map's own spread*, split between a
regional component correlated over 80 kb–1.5 Mb and a per-cell one. The
regional part is what makes the error survive averaging, which is what a real
map's disagreement does: two published maps of one region correlate 0.687 at a
kilobase and 0.890 at a hundred.

## Configuration

Every field of `simulator.config.Config` is a command-line flag with the
underscores turned into dashes. `--preset` sets a coherent group of them:

| preset | rows | length | haplotypes |
| --- | --- | --- | --- |
| `mini` | 512 | 1 Mb | 24 |
| `core_v01` | 20,000 | 10 Mb | 64 |
| `full_v01` | 100,000 | 20 Mb | 64 |

Useful ones beyond the preset:

- `--truth-fraction` — what share of rows carry the true maps
- `--arg-detail` — `none`, `summary` (default: pair coalescences and branch
  length per time bin, 512 bytes a row) or `full` (the tree sequence itself,
  which runs 0.8–28 MB a row)
- `--workers`, `--shard-size`, `--compute-chunk-size` — parallelism and layout

Generation is resumable and content-addressed: an interrupted run picks up from
its receipts rather than starting over, and a completed dataset is verified
against them before it is trusted.

## Data that ships with the package

Everything the simulator needs is inside `simulator/assets`, resolved against
the package rather than the working directory, so an installed copy works from
anywhere.

| | source |
| --- | --- |
| `maps/decode2019_avg_hg38.bed` | deCODE 2019, Halldórsson et al., *Science* 363:eaau1043. 4,531,535 localised crossovers, 682 bp average resolution, native GRCh38. A pedigree map, so it carries no demographic assumption — the LD-based alternatives are inferred *under* one, and this project infers demography from linkage. |
| `maps/roulette_2023_...bed` | Roulette, Seplyarskiy et al., *Nature Genetics* 2023. 1 kb bins. |
| `templates/human_ne/` | 191 inferred human curves: SMC++ on 1000 Genomes, phlash, MSMC2 and MSMC-IM. |
| `templates/stdpopsim_single_pop/` | The 18 single-population models in the stdpopsim catalogue, across 8 species. |

`simulator/assets/SOURCES.md` carries the full provenance. The build scripts
that produced each frozen file sit beside it; rebuilding needs
`pip install -e ".[build-assets]"`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite is calibration as much as correctness. Where a number in the
generator was measured from real data, a test pins it to that measurement and
says which one — the synthetic map's hotspot width against deCODE chr20, the
demography mixture's turning points against the 191 inferred curves, the
observed map's correlation with the truth against two published maps of one
region.

## License

MIT. The maps and templates under `simulator/assets` are third-party data
redistributed under their own terms; see `SOURCES.md`.
