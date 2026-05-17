"""NeCast: population size history from the unphased genotypes of 32 individuals.

A trained network reads one window of 30 Mb of a genome, together with the
recombination and mutation maps of that window, and returns the effective
population size on 64 time intervals between 50 and 500,000 generations ago.
"""

__version__ = "0.1.0"
