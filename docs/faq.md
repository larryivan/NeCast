# Questions and problems

## Error messages

**`... has no index (.tbi or .csi)`**
NeCast reads the VCF window by window, which needs an index. Make one with
`bcftools index data.vcf.gz`. An uncompressed VCF must first be compressed
with `bgzip data.vcf`.

**`found no window`**
No chromosome has 30 Mb of accessible sequence without a masked stretch
longer than 1 Mb. Check that the mask covers the chromosomes you expect
(`awk '{n[$1] += $3 - $2} END {for (c in n) print c, n[c]}' mask.bed` prints
the accessible length of each), and that it is not cut into pieces by many
long gaps. Small chromosomes and very fragmented assemblies may give no
window at all.

**`NeCast reads 32 diploid individuals at a time, and there are only ...`**
The network was trained on 32 individuals and cannot read fewer.

**`... of the samples are not in the VCF`**
The names in the `--samples` file must match the VCF header exactly. Print
the header's names with `bcftools query -l data.vcf.gz`.

**`the VCF and the mask have no autosome in common`**
The chromosome names differ between the files. `chr1` and `1` are treated as
the same; other differences (such as `NC_000001.11`) have to be fixed in one of
the files.

**`--human expects GRCh38 coordinates`**
The VCF is on another assembly, most often GRCh37. The masks and maps of
`--human` are on GRCh38: use a GRCh38 call set, or lift the VCF over.

**`found haploid genotypes`**
NeCast reads diploid autosomes. Leave out chrX, chrY and the mitochondria
(`--chromosomes` selects what is read).

**`the recombination map ... has no rates for chromosome ...`**
Every chromosome that is read needs rates in the map. Add them, or read fewer
chromosomes with `--chromosomes`.

## Questions

**Which individuals should I use?**
32 or more unrelated individuals from one population. Relatives, or a mix of
populations, change the history NeCast sees. With more than 32, NeCast reads
them in groups of 32 and averages the groups.

**Does the data need to be phased?**
No. NeCast reads unphased genotypes and ignores phase if it is there.

**What about missing genotypes?**
They are allowed, both missing (`./.`) and half-missing (`0/.`). The training
simulations included missing calls.

**Do I need ancestral alleles?**
No. NeCast takes the reference allele as ancestral, as the paper did.

**How do I get times in years?**
Add `--generation-time`, the years per generation (29 is common for humans).
The table then has columns in years as well, and the figure uses years.

**Why does the estimate stop at 50 and 500,000 generations?**
These are the limits of the 64 intervals the network was trained to estimate.

**How many windows do I need?**
One window gives an estimate, but a noisy one. More windows give a more
stable average; the paper's human analysis used 51 windows per population.

**Does NeCast need a GPU?**
No. It runs on the CPU, and a window takes a few seconds with 16 threads.

**Can I use NeCast from Python?**
Yes:

```python
from necast.cli import run
log10_ne = run("data.vcf.gz", "mask.bed", "results", mu=1.25e-8, rec_rate=1e-8)
```

`run` takes the same options as `necast run`, with underscores for dashes,
and returns log10 Ne in the 64 intervals.
