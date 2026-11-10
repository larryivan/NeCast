# Example data

One simulated window of 30 Mb with 32 diploid individuals, whose true size
history is known. `necast example` runs NeCast on it and compares the result
with the truth and with `expected.tsv`.

| File | Contents |
|:---|:---|
| `example.vcf.gz`, `.tbi` | genotypes of 32 individuals, unphased, with some missing calls |
| `mask.bed` | the accessible sequence: the whole window |
| `recombination.bed.gz` | recombination map, in cM/Mb |
| `mutation.bed.gz` | mutation map, per bp per generation (mean 8.34e-9) |
| `truth.tsv` | the true size history, on NeCast's 64 time intervals |
| `expected.tsv` | the result of `necast example` when this release was made |

The window is the first of history EMP_CEU (a published size history of the
CEU population) in setting b (slices of the deCODE and Roulette maps, with
sequencing errors) of the
[NeCast benchmark](https://huggingface.co/datasets/Larrivhan/NeCast-benchmark);
`scripts/make_example.py` builds these files from it.
