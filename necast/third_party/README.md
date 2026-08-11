# Third-party code and data

These files come from other projects and keep their own terms, given below
for each. The MIT licence of NeCast covers its own code, not these files.

## psmcplus/

`transition_matrix.py` is from PSMC+ (Trevor Cousins, MIT licence, see
`psmcplus/LICENSE.txt`; https://github.com/trevorcousins/PSMCplus). It builds
the transition matrix of the coalescent hidden Markov model. The only change
is `@njit` → `@njit(cache=True)`, so that compiled code is kept between runs.
SHA-256 of the unchanged file:
`3dd56ffca3c4170a323eb337d7e4efe7235d542de31abbab4aa61d5d06fcf8a2`.

Cousins T, Tabin D, Patterson N, Reich D, Durvasula A (2024). Accurate
inference of population history in the presence of background selection.
bioRxiv, doi:10.1101/2024.01.18.576291.

## Recombination map: `maps/decode2019_avg_hg38.bed`

Halldórsson et al., *Science* 363:eaau1043 (2019). 4,531,535 crossovers
localised in Icelandic pedigrees, average map resolution 682 bp, native GRCh38.
Obtained as `recombAvg.bw` from UCSC:

    https://hgdownload.soe.ucsc.edu/gbdb/hg38/recombRate/recombAvg.bw

Rebuilt by `maps/build_decode2019.py`, which keeps the source units (cM/Mb)
so that the file can be checked against UCSC. It caps rates at 1,000 cM/Mb:
34 intervals exceed 10,000 cM/Mb, most of them one base pair wide, where a
crossover was placed at a single point. It floors rates at 0.1 cM/Mb: a
pedigree map reads zero wherever no crossover was seen (18.4% of 1 kb cells
on chr1 between 20 and 40 Mb), while HapMap II and pyrho are never zero there.

A pedigree map is used because LD-based maps such as HapMap II and pyrho are
inferred under a demographic model, which is what NeCast estimates.

Terms: distributed by UCSC for research use, with credit to the source.

## Mutation map: `maps/roulette_2023_autosomes_hg38_1000bp.bed`

Seplyarskiy et al., *Nature Genetics* (2023). Roulette models the mutation
rate of each site from its sequence context, with a pseudo-R² of 0.86 against
observed variation. The per-site rates were downloaded from
http://genetics.bwh.harvard.edu/downloads/Vova/Roulette/ and averaged in 1 kb
bins by `generator/roulette_map/download_and_process.py` in
https://huggingface.co/datasets/Larrivhan/NeCast-simulations.

Terms: the source states no licence.

The simulator draws its map slices from these two files, and `necast run
--human` reads them for human data, using only the shape of the mutation map:
its mean in each window is set to the mutation rate.

## Inferred human histories: `histories/human_ne/`

191 size histories inferred from human data. The simulator draws some of its
training histories from them (`empirical_human_ne_template` in
docs/simulate.md), and its other kinds of history were tuned against them.

| set | method | source |
| --- | --- | --- |
| 26 populations | SMC++ | Spence & Song, *Science Advances* 5:eaaw9206 (2019), from the pyrho repository; method: Terhorst, Kamm & Song, *Nature Genetics* 49:303 (2017) |
| 154 populations | phlash | Terhorst, *Nature Genetics* (2025), figure 7a |
| 7 superpopulations | phlash | the same, figure 7b |
| 2 | MSMC2 | Yoruba–French, Wang et al., *PLOS Genetics* 16:e1008552 (2020) |
| 2 | MSMC-IM | the same |

Times are in generations, except the SMC++ curves, which are in years and are
divided by 29 when read. `histories/human_ne/README.md` gives the files and
how each was converted.

Terms: the SMC++ curves are under the MIT licence of pyrho; the phlash_paper
and MSMC-IM repositories state no licence, and the MSMC2 and MSMC-IM curves
are included with Ke Wang's permission.

## Catalogue models: `histories/stdpopsim_single_pop/`

The 18 single-population demographic models in the stdpopsim catalogue
(Adrion et al., *eLife* 9:e54967, 2020), across 8 species: *Anopheles gambiae*,
*Arabidopsis thaliana*, *Bos taurus*, *Drosophila melanogaster*, *Homo
sapiens*, *Mus musculus*, *Papio anubis* and *Phocoena sinus*. Written by
`histories/build_stdpopsim_single_pop.py` from stdpopsim 0.3.0.

They are kept in a file so that every run draws the same 18 histories,
whatever version of stdpopsim is installed. Models with more than one
population are left out, because one population read out of a structured
model would turn migration into changes of size.

Three histories have sizes outside 100 to 1,000,000. The simulator multiplies
each by one constant to fit, which keeps its shape: *A. gambiae* by 0.246, the
Indian *M. m. castaneus* by 0.197 and one Holstein model by 1.111. The file
holds the published sizes; each window records its factor as
`stdpopsim_ne_rescale`.

Terms: GPL-3.0, the licence of stdpopsim.

## GRCh38 gaps: `reference/grch38_gaps.bed`

For chromosomes 1-22, the assembly gaps of GRCh38 (UCSC table `gap`) and one
span per centromere, from the first to the last row of UCSC table
`centromeres` (https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/).
`necast run --human` leaves these out of the accessible sequence and reads
the chromosome lengths from them.

Terms: UCSC tables, free to use.
