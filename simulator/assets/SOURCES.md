# Where the shipped data came from

Everything here is third-party, redistributed under its own terms. The MIT
licence on this repository covers the code, not these files.

## Recombination map — `maps/decode2019_avg_hg38.bed`

Halldórsson et al., *Science* 363:eaau1043 (2019). 4,531,535 crossovers
localised in Icelandic pedigrees, average map resolution 682 bp, native GRCh38.
Obtained as `recombAvg.bw` from UCSC:

    https://hgdownload.soe.ucsc.edu/gbdb/hg38/recombRate/recombAvg.bw

Rebuilt by `maps/build_decode2019.py`, which writes cM/Mb — the source units,
so the file stays checkable against UCSC — and applies two bounds. Rates are
capped at 1,000 cM/Mb, because 34 intervals genome-wide exceed 10,000 with a
median width of one base pair: those are the artefact of localising a crossover
to a point, not hotspots, and both this ingest and msprime's own `RateMap`
reject rates above 1e-4 per bp. Rates are floored at 0.1 cM/Mb, because a
pedigree map reports zero where no crossover was observed, and that is sparsity
rather than an absence of recombination — 18.4% of 1 kb cells on chr1 between
20 and 40 Mb read exactly zero, while in those same cells HapMap II reports a
median 6.6% of its mean and pyrho CEU 17.0%, neither ever zero.

A pedigree map is used rather than an LD-based one on purpose. HapMap II and
pyrho are inferred *under* a demographic model, and this project infers
demography from linkage; using them would be circular.

## Mutation map — `maps/roulette_2023_autosomes_hg38_1000bp.bed`

Seplyarskiy et al., *Nature Genetics* (2023). Roulette models the mutation rate
from local sequence context and reports it in 1 kb bins, with a pseudo-R² of
0.86 against observed variation. Build script in `maps/build/`.

## Inferred human histories — `templates/human_ne/`

191 curves, used as the one part of the demography mixture that nobody
designed, and as the calibration target for every synthetic recipe.

| set | method | source |
| --- | --- | --- |
| 26 populations | SMC++ | Terhorst, Kamm & Song, *Nature Genetics* 49:303 (2017), applied to 1000 Genomes |
| 154 populations | phlash | Terhorst, *Nature Genetics* (2025), figure 7a |
| 7 superpopulations | phlash | the same, figure 7b |
| 2 | MSMC2 | Yoruba–French, published estimates |
| 2 | MSMC-IM | Yoruba–French, published estimates |

Times are given in generations; where a source published years, the conversion
used a generation time of 29 years and the file records it.

## Catalogue models — `templates/stdpopsim_single_pop/`

The 18 single-population demographic models in the stdpopsim catalogue
(Adrion et al., *eLife* 9:e54967, 2020), across 8 species: *Anopheles gambiae*,
*Arabidopsis thaliana*, *Bos taurus*, *Drosophila melanogaster*, *Homo
sapiens*, *Mus musculus*, *Papio anubis* and *Phocoena sinus*. Frozen by
`build_stdpopsim_single_pop.py` from stdpopsim 0.3.0.

They are frozen rather than imported at runtime because they are the held-out
evaluation set: what matters is that the same 18 trajectories are available to
every run on every machine, and a runtime import makes that depend on whichever
version happens to be installed. Multi-population models are skipped —
reading one population out of a structured model would silently turn migration
into size change.

Three trajectories carry an effective size outside this project's declared
band and are multiplied by one constant to fit it: *A. gambiae* by 0.246, the
Indian *M. m. castaneus* by 0.197, and one Holstein model by 1.111. A uniform
factor leaves the shape — which is what generalisation is measured on —
bit-for-bit intact, and each factor is recorded in the file.
