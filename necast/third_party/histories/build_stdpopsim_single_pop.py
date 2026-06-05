"""Freeze the stdpopsim single-population trajectories into a data file.

Why a data file rather than a runtime import.  These eighteen models are the
held-out evaluation set: the point of them is that the model has never seen
their shape during training, so what matters is that the exact same eighteen
trajectories are available to every run, on every machine, for the life of the
project.  A runtime ``import stdpopsim`` makes that depend on which version
happens to be installed -- the catalogue is versioned and its parameters do get
corrected -- and makes the generator fail wherever the package is missing.  The
human Ne templates under ``necast/third_party/histories/human_ne`` are frozen for the same
reason.

What is frozen is the size trajectory of the single population, including its
native exponential growth rates, in generations.  Multi-population models are
skipped: reading one population out of a structured model would silently turn
migration into size change, which is the one thing this project must not do.

Usage (needs ``pip install stdpopsim``, a build-time dependency only):
    python -m necast.third_party.histories.build_stdpopsim_single_pop
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from necast.simulator.stdpopsim_support import _extract_single_population_trajectory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "stdpopsim_single_pop" / "trajectories.csv"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import stdpopsim

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "species",
                "model_id",
                "population",
                "generation_time_years",
                "epoch_index",
                "start_generations_ago",
                "size",
                "growth_rate_per_generation",
            ]
        )
        for species in stdpopsim.all_species():
            for model in species.demographic_models:
                demography = model.model
                if len(demography.populations) != 1:
                    continue
                breaks, values, rates = _extract_single_population_trajectory(demography)
                starts = [0.0, *breaks]
                population = demography.populations[0].name
                for index, (start, size, rate) in enumerate(zip(starts, values, rates)):
                    writer.writerow(
                        [
                            species.id,
                            model.id,
                            population,
                            # Round-trip precision, not display precision.  A
                            # breakpoint written as %.6f loses its last four
                            # digits -- 33.333334976 becomes 33.333335 -- and
                            # the frozen file then no longer is the published
                            # model, which is the one thing it has to be.
                            repr(float(model.generation_time)),
                            index,
                            repr(float(start)),
                            repr(float(size)),
                            repr(float(rate)),
                        ]
                    )
                written += 1
    print("wrote %d single-population models to %s" % (written, out_path))
    print("stdpopsim version: %s" % stdpopsim.__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
