# Third-party code and data

These files come from other projects and keep their own licences. The MIT
licence of NeCast covers its own code, not these files.

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
