"""Loader for the frozen stdpopsim single-population trajectories.

These eighteen models, across eight species, are the held-out evaluation set.
They exist so that generalisation can be measured against histories that were
published by other people for other reasons, rather than against our own prior
re-drawn -- which measures nothing.  phlash validates the same way, on twelve
stdpopsim models across eight species.

The trajectories are frozen into ``necast/third_party/histories/stdpopsim_single_pop``
by ``necast/third_party/histories/build_stdpopsim_single_pop.py`` rather than
imported at runtime; the reasons are in that script's docstring.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .assets import asset_path
from .config import NE_LOG10_BOUNDS, Config


DEFAULT_STDPOPSIM_TEMPLATE_PATH = "histories/stdpopsim_single_pop/trajectories.csv"

# Models held out of the draw because another recipe produces them.  Empty at
# present: Zigzag_1S14 had a recipe of its own for a while and was excluded here
# so that exactly one code path produced it; that recipe is gone and the model
# is back in the catalogue.  The hand-written published event table stays in
# ``demography.py`` as the independent source this freeze is checked against.
CATALOGUE_EXCLUSIONS: frozenset[str] = frozenset()


@dataclass(frozen=True)
class StdpopsimTemplate:
    species: str
    model_id: str
    population: str
    generation_time_years: float
    breaks: tuple[float, ...]
    values: tuple[float, ...]
    growth_rates: tuple[float, ...]
    ne_rescale: float

    @property
    def key(self) -> str:
        return "%s/%s" % (self.species, self.model_id)


def _fit_to_bounds(values: tuple[float, ...]) -> float:
    """Return the single factor that slides a trajectory into the Ne band.

    Two of the eighteen models -- *Anopheles gambiae* and the Indian
    *M. m. castaneus* -- carry effective sizes above a million, and one Holstein
    model dips below a hundred.  Clipping them would flatten exactly the
    features they are held out to test, so the whole trajectory is multiplied by
    one constant instead: Ne is only identifiable jointly with the mutation
    rate, and a uniform factor leaves the shape -- which is what generalisation
    is being measured on -- bit-for-bit intact.  The factor is recorded so the
    published model can always be recovered.
    """

    low, high = 10.0 ** NE_LOG10_BOUNDS[0], 10.0 ** NE_LOG10_BOUNDS[1]
    smallest, largest = min(values), max(values)
    if largest / smallest > high / low:
        raise ValueError("trajectory spans more than the declared Ne band")
    if largest > high:
        return high / largest
    if smallest < low:
        return low / smallest
    return 1.0


@lru_cache(maxsize=4)
def _load(path: str) -> tuple[StdpopsimTemplate, ...]:
    source = asset_path(path)
    if not source.exists():
        return ()
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault((row["species"], row["model_id"]), []).append(row)

    out: list[StdpopsimTemplate] = []
    for (species, model_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: int(item["epoch_index"]))
        starts = [float(item["start_generations_ago"]) for item in rows]
        sizes = tuple(float(item["size"]) for item in rows)
        rates = tuple(float(item["growth_rate_per_generation"]) for item in rows)
        rescale = _fit_to_bounds(sizes)
        out.append(
            StdpopsimTemplate(
                species=species,
                model_id=model_id,
                population=rows[0]["population"],
                generation_time_years=float(rows[0]["generation_time_years"]),
                breaks=tuple(starts[1:]),
                values=tuple(size * rescale for size in sizes),
                growth_rates=rates,
                ne_rescale=float(rescale),
            )
        )
    return tuple(out)


def load_stdpopsim_templates(cfg: Config) -> tuple[StdpopsimTemplate, ...]:
    return _load(str(getattr(cfg, "stdpopsim_template_path", DEFAULT_STDPOPSIM_TEMPLATE_PATH)))


def stdpopsim_templates_available(cfg: Config) -> bool:
    return len(drawable_stdpopsim_templates(cfg)) > 0


def drawable_stdpopsim_templates(cfg: Config) -> tuple[StdpopsimTemplate, ...]:
    """The catalogue minus the models that have a recipe of their own."""

    return tuple(
        template
        for template in load_stdpopsim_templates(cfg)
        if template.model_id not in CATALOGUE_EXCLUSIONS
    )


def choose_stdpopsim_template(
    rng: np.random.Generator, cfg: Config
) -> StdpopsimTemplate | None:
    templates = drawable_stdpopsim_templates(cfg)
    if not templates:
        return None
    return templates[int(rng.integers(0, len(templates)))]
