from __future__ import annotations

from typing import Optional

import numpy as np

from .config import (
    DEMOGRAPHY_TYPE_IDS,
    GENERATOR_RECIPE_IDS,
    Config,
)
from .demography import summarize_demography
from .utils import bin_average_log10_ne, time_grid


class StdpopsimSimulationError(RuntimeError):
    """Raised when a requested stdpopsim anchor cannot be generated."""


def _extract_single_population_trajectory(dem) -> tuple[list[float], list[float], list[float]]:
    """Extract first-population size and growth-rate epochs from msprime.

    This remains a proxy for structured models, but it exactly preserves native
    exponential growth for genuinely single-population stdpopsim models.
    """
    first_pop = dem.populations[0]
    pop_name = first_pop.name
    pop_id = getattr(first_pop, "id", 0)
    current_size = float(first_pop.initial_size)
    current_rate = float(first_pop.growth_rate)
    previous_time = 0.0
    breaks: list[float] = []
    values = [current_size]
    growth_rates = [current_rate]

    events = sorted(getattr(dem, "events", []), key=lambda event: getattr(event, "time", 0))
    for event in events:
        if event.__class__.__name__ != "PopulationParametersChange":
            continue
        population = getattr(event, "population", None)
        population_id = getattr(event, "population_id", None)
        if population is not None:
            applies = population in {-1, 0, pop_id, pop_name}
        elif population_id is not None:
            applies = population_id in {-1, 0, pop_id}
        else:
            applies = True
        if not applies:
            continue

        event_time = float(event.time)
        current_size *= float(np.exp(-current_rate * (event_time - previous_time)))
        if getattr(event, "initial_size", None) is not None:
            current_size = float(event.initial_size)
        if getattr(event, "growth_rate", None) is not None:
            current_rate = float(event.growth_rate)

        if event_time <= 0:
            values[0] = current_size
            growth_rates[0] = current_rate
        elif breaks and np.isclose(event_time, breaks[-1], rtol=0.0, atol=1e-12):
            values[-1] = current_size
            growth_rates[-1] = current_rate
        else:
            breaks.append(event_time)
            values.append(current_size)
            growth_rates.append(current_rate)
        previous_time = event_time
    return breaks, values, growth_rates


def validate_stdpopsim_setup(cfg: Config) -> None:
    """Fail before generation if an explicitly enabled anchor setup is unusable."""

    if not cfg.enable_stdpopsim or cfg.p_stdpopsim_anchor <= 0.0:
        return
    try:
        import stdpopsim  # type: ignore

        species = stdpopsim.get_species(cfg.stdpopsim_species)
        model_ids = [model.strip() for model in cfg.stdpopsim_models.split(",") if model.strip()]
        if model_ids:
            for model_id in model_ids:
                species.get_demographic_model(model_id)
        elif not getattr(species, "demographic_models", None):
            raise ValueError(f"species {cfg.stdpopsim_species!r} has no demographic models")
        stdpopsim.get_engine("msprime")
    except Exception as exc:
        raise StdpopsimSimulationError(
            "stdpopsim anchors are enabled but the dependency, species, models, "
            f"or engine are unavailable: {exc}"
        ) from exc


def try_stdpopsim_sample(
    rng: np.random.Generator,
    cfg: Config,
) -> Optional[tuple[object, np.ndarray, dict, object]]:
    if not cfg.enable_stdpopsim:
        return None
    try:
        import stdpopsim  # type: ignore
        species = stdpopsim.get_species(cfg.stdpopsim_species)
        model_ids = [m.strip() for m in cfg.stdpopsim_models.split(",") if m.strip()]
        if not model_ids:
            model_ids = [m.id for m in species.demographic_models[: min(5, len(species.demographic_models))]]
        model_id = str(rng.choice(model_ids))
        model = species.get_demographic_model(model_id)
        contig_length = cfg.stdpopsim_contig_length if cfg.stdpopsim_contig_length > 0 else cfg.seq_len
        contig = species.get_contig(
            length=contig_length,
            mutation_rate=cfg.baseline_mu,
            recombination_rate=cfg.baseline_rec,
        )
        pop_id = model.populations[0].id if getattr(model, "populations", None) else "pop_0"
        n_dip = max(1, cfg.n_haplotypes // 2)
        try:
            samples = model.get_samples(n_dip, *([0] * max(0, len(model.populations) - 1)))
        except Exception:
            samples = {pop_id: n_dip}
        engine = stdpopsim.get_engine("msprime")
        ts = engine.simulate(model, contig, samples, seed=int(rng.integers(1, 2**31 - 2)))
        try:
            ts = ts.trim()
        except Exception:
            pass

        edges, _ = time_grid(cfg)
        breaks: list[float] = []
        values: list[float] = []
        growth_rates: list[float] = []
        dem = getattr(model, "model", None)
        if dem is None:
            raise ValueError(
                f"stdpopsim model {model_id!r} exposes no msprime demography"
            )
        breaks, values, growth_rates = _extract_single_population_trajectory(dem)
        if not values:
            raise ValueError(
                f"stdpopsim model {model_id!r} has no extractable population trajectory"
            )
        if not growth_rates:
            raise ValueError(
                f"stdpopsim model {model_id!r} has no extractable growth-rate trajectory"
            )

        y = bin_average_log10_ne(edges, breaks, values, growth_rates)
        trajectory_meta = {
            "source_type": "stdpopsim_anchor",
            "target_quality": "heuristic_single_population",
            "native_continuous_growth": any(
                float(rate) != 0.0 for rate in growth_rates
            ),
        }
        meta = {
            **trajectory_meta,
            **summarize_demography(
                edges,
                breaks,
                values,
                growth_rates,
                trajectory_meta,
                generator_recipe="stdpopsim",
                demography_type="stdpopsim",
            ),
            "scenario": "stdpopsim",
            "generator_recipe": "stdpopsim",
            "generator_recipe_id": GENERATOR_RECIPE_IDS["stdpopsim"],
            "demography_type": "stdpopsim",
            "demography_type_id": DEMOGRAPHY_TYPE_IDS["stdpopsim"],
            "demography_mixture_version": cfg.demography_mixture_version,
            "stdpopsim_species": cfg.stdpopsim_species,
            "stdpopsim_model_id": model_id,
            "stdpopsim_population": pop_id,
            "stdpopsim_contig_length": int(contig_length),
            "target_scale": "log10_Ne",
            "target_aggregation": "log_time_bin_average",
            "target_time_bins": int(cfg.time_bins),
            "demography_resolution_log10": float(cfg.demography_resolution_log10),
            "demography_breaks": [float(x) for x in breaks],
            "demography_values": [float(x) for x in values],
            "demography_growth_rates": [float(x) for x in growth_rates],
            "demography_function": (
                "piecewise_exponential_linear_time"
                if any(float(rate) != 0.0 for rate in growth_rates)
                else "piecewise_constant"
            ),
        }
        return ts, y, meta, contig
    except Exception as exc:
        raise StdpopsimSimulationError(f"failed to generate a stdpopsim anchor: {exc}") from exc
