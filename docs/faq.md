[NeCast](../README.md) · [Guide](guide.md) · **Questions** · [Simulating](simulate.md) · [Training](train.md)

# Questions and problems

<table>
<thead>
<tr><th align="left">Error messages</th><th align="left">Questions</th></tr>
</thead>
<tbody>
<tr>
<td valign="top">

[`has no index`](#file-has-no-index-tbi-or-csi)<br>
[`found no window`](#found-no-window)<br>
[`there are only N samples`](#necast-reads-32-diploid-individuals-at-a-time-and-there-are-only-n-samples)<br>
[`samples are not in the VCF`](#n-of-the-samples-are-not-in-the-vcf)<br>
[`no autosome in common`](#the-vcf-and-the-mask-have-no-autosome-in-common)<br>
[`--human expects GRCh38`](#--human-expects-grch38-coordinates)<br>
[`found haploid genotypes`](#found-haploid-genotypes)<br>
[`cannot read the recombination map`](#cannot-read-the-recombination-map-file)<br>
[`has no rates for chromosome`](#the-recombination-map-file-has-no-rates-for-chromosome-name)

</td>
<td valign="top">

[Which individuals should I use?](#which-individuals-should-i-use)<br>
[What if the population was structured?](#what-does-the-curve-show-if-the-population-was-structured)<br>
[Should I filter the variants?](#should-i-filter-the-variants)<br>
[Does sequencing depth matter?](#does-sequencing-depth-matter)<br>
[Does the data need to be phased?](#does-the-data-need-to-be-phased)<br>
[What about missing genotypes?](#what-about-missing-genotypes)<br>
[Do I need ancestral alleles?](#do-i-need-ancestral-alleles)<br>
[How do I get times in years?](#how-do-i-get-times-in-years)<br>
[Why 50 to 500,000 generations?](#why-does-the-estimate-stop-at-50-and-500000-generations)<br>
[Can I rescale for a new mutation rate?](#i-changed-the-mutation-rate-can-i-rescale-my-result)<br>
[How many windows do I need?](#how-many-windows-do-i-need)<br>
[Does NeCast need a GPU?](#does-necast-need-a-gpu)<br>
[Can I use NeCast from Python?](#can-i-use-necast-from-python)

</td>
</tr>
</tbody>
</table>

## Error messages

When NeCast cannot read the input it stops with one of these messages. `FILE`,
`N` and `NAME` stand for the file, number or name in yours.

### `FILE has no index (.tbi or .csi)`

NeCast reads the VCF window by window, which needs an index. Make one with
`bcftools index data.vcf.gz`. An uncompressed VCF must first be compressed
with `bgzip data.vcf`.

### `found no window`

No chromosome has 30 Mb of accessible sequence without a masked stretch
longer than 1 Mb, or the VCF has no sites there, or the individuals read vary
at none of them, as when all their calls there are missing (the log names
each group left out). Check that

- the mask covers the chromosomes you expect; this prints the accessible
  length of each:

  ```bash
  awk '{n[$1] += $3 - $2} END {for (c in n) print c, n[c]}' accessible.bed
  ```

- the mask is not cut into pieces by many long gaps;
- the VCF holds those chromosomes, with calls for the individuals read.

Small chromosomes and very fragmented assemblies may give no window at all.

### `NeCast reads 32 diploid individuals at a time, and there are only N samples`

The network was trained on 32 individuals and cannot read fewer.

### `N of the samples are not in the VCF`

The names in the `--samples` file must match the VCF header exactly. Print
the header's names with `bcftools query -l data.vcf.gz`.

### `the VCF and the mask have no autosome in common`

The chromosome names differ between the files. `chr1` and `1` are treated as
the same; other differences (such as `NC_000001.11`) have to be fixed in one of
the files.

### `--human expects GRCh38 coordinates`

The VCF is on another assembly, most often GRCh37. The masks and maps of
`--human` are on GRCh38: use a GRCh38 call set, or lift the VCF over.

### `found haploid genotypes`

NeCast reads diploid autosomes. Leave out chrX, chrY and the mitochondria
(`--chromosomes` selects what is read).

### `cannot read the recombination map FILE`

Maps are BED files: chromosome, start, end, rate. The
[guide](guide.md#recombination) shows how to convert a map in HapMap format.

### `the recombination map FILE has no rates for chromosome NAME`

Every chromosome that is read needs rates in the map. Add them, or read fewer
chromosomes with `--chromosomes`.

## Questions

### Which individuals should I use?

32 or more unrelated individuals from one population. Relatives, or a mix of
populations, change the history NeCast sees. With more than 32, NeCast reads
them in groups of 32 and averages the groups.

### What does the curve show if the population was structured?

Like PSMC and its successors, NeCast follows the rate at which pairs of
genomes find a common ancestor, and reports its inverse as a size. That is
the population size when mating is random; structure and admixture change it.

### Should I filter the variants?

By call quality only. Filtering by allele frequency, removing singletons or
pruning for linkage disequilibrium removes the signal NeCast reads.

### Does sequencing depth matter?

The training simulations included genotype errors and missing calls. For the
same 1000 Genomes individuals, estimates from the low-coverage data and from
the 30x data differed by 0.08 on the log10 scale on average, 50 to 500
generations ago.

### Does the data need to be phased?

No. NeCast reads unphased genotypes and ignores phase if it is there.

### What about missing genotypes?

They are allowed, both missing (`./.` or `.`) and half-missing (`0/.`). The
training simulations included missing calls.

### Do I need ancestral alleles?

No. NeCast takes the reference allele as ancestral, as the paper did.

### How do I get times in years?

Add `--generation-time`, the years per generation (29 is common for humans).
The table then has columns in years as well, and the figure uses years.

### Why does the estimate stop at 50 and 500,000 generations?

These are the limits of the 64 intervals the network was trained to estimate.

### I changed the mutation rate. Can I rescale my result?

No: run NeCast again. The mutation rate changes the shape of the estimate as
well as its level.

### How many windows do I need?

One window gives an estimate, but a noisy one. More windows give a more
stable average; the paper's human analysis used 51 windows per population.

### Does NeCast need a GPU?

No. It runs on the CPU, and a window takes a few seconds with 16 threads.

### Can I use NeCast from Python?

Yes:

```python
from necast.cli import run
log10_ne = run("data.vcf.gz", "accessible.bed", "results", mu=1.25e-8, rec_rate=1e-8)
```

`run` takes the same options as `necast run`, with underscores for dashes,
and returns log10 Ne in the 64 intervals.
