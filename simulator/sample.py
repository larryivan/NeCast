from __future__ import annotations

import atexit
from pathlib import Path

import msprime
import numpy as np

from .config import (
    DEMOGRAPHY_TYPE_IDS,
    GENERATOR_RECIPE_IDS,
    SOURCE_IDS,
    Config,
    validate_config,
)
from .demography import sample_custom_demography
from .dataset_state import SAMPLE_GENERATOR_SCHEMA_VERSION
from .empirical_maps import EmpiricalMapStore
from .maps import choose_source, constant_map, make_rate_map, sample_maps_for_source
from .noise import add_false_variants, apply_genotype_noise, pack_haplotypes
from .stdpopsim_support import StdpopsimSimulationError, try_stdpopsim_sample
from .utils import rng_from_seed, time_grid


_WORKER_CFG: Config | None = None
_WORKER_EMPIRICAL: EmpiricalMapStore | None = None
_WORKER_SHARED_MAP_HANDLES: list | None = None


def _close_worker_shared_maps() -> None:
    global _WORKER_EMPIRICAL, _WORKER_SHARED_MAP_HANDLES
    _WORKER_EMPIRICAL = None
    if _WORKER_SHARED_MAP_HANDLES:
        from .shared_maps import release_shared_memory

        release_shared_memory(_WORKER_SHARED_MAP_HANDLES)
    _WORKER_SHARED_MAP_HANDLES = None


atexit.register(_close_worker_shared_maps)


def init_worker(
    cfg_dict: dict,
    empirical: EmpiricalMapStore | None = None,
    empirical_shared_spec=None,
) -> None:
    global _WORKER_CFG, _WORKER_EMPIRICAL, _WORKER_SHARED_MAP_HANDLES
    if empirical is not None and empirical_shared_spec is not None:
        raise ValueError("provide either empirical or empirical_shared_spec, not both")
    _close_worker_shared_maps()
    _WORKER_CFG = Config(**cfg_dict)
    validate_config(_WORKER_CFG)
    if empirical_shared_spec is not None:
        from .shared_maps import attach_shared_empirical_map_store

        _WORKER_EMPIRICAL, _WORKER_SHARED_MAP_HANDLES = attach_shared_empirical_map_store(empirical_shared_spec)
    else:
        _WORKER_EMPIRICAL = empirical


def _get_worker_cfg(cfg_dict: dict | None = None) -> Config:
    global _WORKER_CFG
    if cfg_dict is not None:
        cfg = Config(**cfg_dict)
        validate_config(cfg)
        return cfg
    if _WORKER_CFG is None:
        raise RuntimeError("simulator worker was not initialized")
    return _WORKER_CFG


def _get_worker_empirical(cfg: Config, empirical_dict: dict | None = None) -> EmpiricalMapStore:
    global _WORKER_EMPIRICAL
    if empirical_dict is not None:
        return EmpiricalMapStore(
            rec_maps=empirical_dict.get("rec_maps", {}),
            mut_maps=empirical_dict.get("mut_maps", {}),
        )
    if _WORKER_EMPIRICAL is None:
        from .empirical_maps import load_empirical_maps

        _WORKER_EMPIRICAL = load_empirical_maps(cfg)
    return _WORKER_EMPIRICAL


def _parse_task(task: tuple) -> tuple[int, Config, EmpiricalMapStore, int]:
    if len(task) == 2:
        index, seed = task
        cfg = _get_worker_cfg()
        empirical = _get_worker_empirical(cfg)
        return int(index), cfg, empirical, int(seed)
    if len(task) == 4:
        index, cfg_dict, empirical_dict, seed = task
        cfg = _get_worker_cfg(cfg_dict)
        empirical = _get_worker_empirical(cfg, empirical_dict)
        return int(index), cfg, empirical, int(seed)
    raise ValueError(f"Unsupported simulator task shape: {len(task)}")


def _ensure_haplotype_count(G: np.ndarray, n_haplotypes: int) -> np.ndarray:
    if G.shape[1] == n_haplotypes:
        return G
    if G.shape[1] > n_haplotypes:
        return G[:, :n_haplotypes]
    pad = np.zeros((G.shape[0], n_haplotypes - G.shape[1]), dtype=G.dtype)
    return np.concatenate([G, pad], axis=1)


def _rate_summary(prefix: str, rates: np.ndarray) -> dict:
    arr = np.asarray(rates, dtype=np.float64)
    if arr.size == 0:
        return {
            f"mean_{prefix}_rate": 0.0,
            f"std_{prefix}_rate": 0.0,
            f"{prefix}_rate_min": 0.0,
            f"{prefix}_rate_max": 0.0,
        }
    return {
        f"mean_{prefix}_rate": float(np.mean(arr)),
        f"std_{prefix}_rate": float(np.std(arr)),
        f"{prefix}_rate_min": float(np.min(arr)),
        f"{prefix}_rate_max": float(np.max(arr)),
    }


def _reproducible_subset(index: int, fraction: float) -> bool:
    """Select a row by its index rather than by drawing.

    Drawing would consume the noise stream and make the selection depend on
    how many rows came before, so two runs that differ only in row count would
    disagree about which rows carry the extra artefacts.
    """

    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    return (int(index) * 2_654_435_761) % 1_000_000 < fraction * 1_000_000


def genealogy_summary(
    ts, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pair coalescences and branch length per target time bin.

    Pair coalescences, not node counts.  A coalescence rate is a statement
    about *pairs* -- two lineages meet at rate 1/(2Ne) per generation -- so
    counting internal nodes and dividing by branch length is dimensionally
    wrong, and an oracle built on it comes out at an RMSE of 3.7 against a
    label that spans four decades.  ``pair_coalescence_counts`` weights each
    node by how many sample pairs actually coalesce there and by the sequence
    it holds, which is the quantity PSMC and its descendants estimate.  The
    counts sum to C(n, 2) -- 2,016 for sixty-four haplotypes -- which is what
    makes them self-checking.

    From the counts alone the whole trajectory follows: the pairs still
    uncoalesced entering a bin are the total minus what has already met, so
    the hazard is ``count / remaining / width`` and Ne is one over twice that.
    Nothing else needs storing for the rate.

    Branch length is kept beside it because it answers the other question --
    not how fast lineages met, but how much sequence there was to see it in.
    A bin can be empty because the population was large or because no lineage
    was left, and the two look identical in the counts alone.  It comes off
    the edge table rather than tree by tree: an edge already carries its span
    and both endpoint times, so the whole calculation is one clipped overlap
    per bin -- 0.19 seconds on a row whose ancestry simulation took 233.
    """

    bins = np.asarray(edges, dtype=np.float64)
    windows = np.concatenate(([0.0], bins, [np.inf]))
    pair_counts = np.asarray(
        ts.pair_coalescence_counts(time_windows=windows, span_normalise=True),
        dtype=np.float64,
    )[1:-1]

    node_times = ts.tables.nodes.time
    edge_table = ts.tables.edges
    child = node_times[edge_table.child]
    parent = node_times[edge_table.parent]
    span = np.asarray(edge_table.right, dtype=np.float64) - np.asarray(
        edge_table.left, dtype=np.float64
    )
    exposure = np.empty(bins.size - 1, dtype=np.float64)
    for index in range(bins.size - 1):
        overlap = np.clip(
            np.minimum(parent, bins[index + 1]) - np.maximum(child, bins[index]),
            0.0,
            None,
        )
        exposure[index] = float((overlap * span).sum())
    return pair_counts, exposure


def simulate_one(task: tuple) -> dict:
    index, cfg, empirical, seed = _parse_task(task)
    rng = rng_from_seed(seed)
    empirical_available = empirical.is_available(cfg.seq_len)
    sample_id = f"sim_{index:08d}"

    source = choose_source(rng, cfg, empirical_available)

    ts = None
    std_meta: dict = {}
    if source == "stdpopsim_anchor":
        std_res = try_stdpopsim_sample(rng, cfg)
        if std_res is not None:
            ts, y, std_meta, _ = std_res
            source = "stdpopsim_anchor"
        else:
            source = "custom_heterogeneous"

    true_rec_pos = true_rec_rate = obs_rec_pos = obs_rec_rate = None
    true_mut_pos = true_mut_rate = obs_mut_pos = obs_mut_rate = None
    dem_meta: dict = {}
    map_meta: dict = {}

    if ts is None:
        dem, y, dem_meta = sample_custom_demography(rng, cfg)
        maps = sample_maps_for_source(rng, cfg, source, empirical)
        true_rec_pos, true_rec_rate, obs_rec_pos, obs_rec_rate, true_mut_pos, true_mut_rate, obs_mut_pos, obs_mut_rate, map_meta = maps
        rec_map = make_rate_map(true_rec_pos, true_rec_rate)
        mut_map = make_rate_map(true_mut_pos, true_mut_rate)
        n_diploids = cfg.n_haplotypes // 2
        ts = msprime.sim_ancestry(
            samples={"pop": n_diploids},
            demography=dem,
            sequence_length=cfg.seq_len,
            recombination_rate=rec_map,
            ploidy=2,
            random_seed=int(rng.integers(1, 2**31 - 2)),
        )
        try:
            mut_model = msprime.BinaryMutationModel()
            ts = msprime.sim_mutations(ts, rate=mut_map, model=mut_model, random_seed=int(rng.integers(1, 2**31 - 2)))
        except Exception:
            ts = msprime.sim_mutations(ts, rate=mut_map, random_seed=int(rng.integers(1, 2**31 - 2)))
    else:
        dem_meta = std_meta
        source = "stdpopsim_anchor"
        true_rec_pos, true_rec_rate = constant_map(int(ts.sequence_length), cfg.baseline_rec)
        true_mut_pos, true_mut_rate = constant_map(int(ts.sequence_length), cfg.baseline_mu)
        obs_rec_pos, obs_rec_rate = true_rec_pos.copy(), true_rec_rate.copy()
        obs_mut_pos, obs_mut_rate = true_mut_pos.copy(), true_mut_rate.copy()
        map_meta = {
            "map_type": "stdpopsim_or_constant_fallback",
            "map_mode": "constant_fallback",
            "recomb_mode": "constant_fallback",
            "mu_mode": "constant_fallback",
        }

    G = ts.genotype_matrix()
    if G.shape[0] > 0:
        G = _ensure_haplotype_count(G, cfg.n_haplotypes)
        positions = np.asarray(ts.sites_position, dtype=np.float64)
    else:
        G = np.zeros((0, cfg.n_haplotypes), dtype=np.uint8)
        positions = np.zeros(0, dtype=np.float64)

    from .noise import noise_params_for_sample

    noise = noise_params_for_sample(rng, cfg, source)
    # Invented sites go in before the other channels so they are subject to the
    # same missingness and the same phasing as any other site: a caller does
    # not know which of its sites are real.
    G, positions, false_stats = add_false_variants(
        G,
        positions,
        rng,
        int(ts.sequence_length),
        float(noise.get("false_variant_rate_per_bp", 0.0)),
    )
    G_noisy, missing_flat_idx, noise_stats = apply_genotype_noise(G, positions, rng, noise)
    noise.update(false_stats)
    noise.update(noise_stats)

    packed = pack_haplotypes(G_noisy, cfg.n_haplotypes)
    positions_u32 = np.clip(np.rint(positions), 0, np.iinfo(np.uint32).max).astype(np.uint32)

    meta = {
        "sample_id": sample_id,
        "sample_index": int(index),
        "source_type": source,
        "source_id": SOURCE_IDS.get(source, -1),
        "scenario": dem_meta.get("scenario", dem_meta.get("demography_type", "stdpopsim")),
        "noise_profile": noise.get("profile", "unknown"),
        "sequence_length": int(round(ts.sequence_length)),
        "n_haplotypes": cfg.n_haplotypes,
        "n_diploid_individuals": cfg.n_haplotypes // 2,
        "sample_ploidy": 2,
        "phase_pairing": "diploid_individual_haplotypes",
        "n_variants": int(G_noisy.shape[0]),
        "variant_density_per_mb": float(G_noisy.shape[0] / max(1e-12, int(round(ts.sequence_length)) / 1_000_000.0)),
        "genotype_error": float(noise.get("genotype_error", 0.0)),
        "missing_rate": float(noise.get("missing_rate", 0.0)),
        "n_missing_genotypes": int(len(missing_flat_idx)),
        "missing_storage": "sparse_flat_index_uint32",
        "phase_switch_pair_count": int(noise.get("phase_switch_pair_count", 0)),
        "phaseable_pair_count": int(noise.get("phaseable_pair_count", 0)),
        "num_trees": int(ts.num_trees),
        "num_sites": int(ts.num_sites),
        "seed": int(seed),
        "noise": noise,
        **dem_meta,
        **map_meta,
        **_rate_summary("obs_rec", obs_rec_rate),
        **_rate_summary("obs_mut", obs_mut_rate),
    }
    demography_type = str(meta.get("demography_type", "stdpopsim"))
    generator_recipe = str(meta.get("generator_recipe", meta.get("scenario", "stdpopsim")))
    try:
        demography_type_id = int(DEMOGRAPHY_TYPE_IDS[demography_type])
    except KeyError as exc:
        raise ValueError(f"Unknown public demography type: {demography_type!r}") from exc
    try:
        generator_recipe_id = int(GENERATOR_RECIPE_IDS[generator_recipe])
    except KeyError as exc:
        raise ValueError(f"Unknown demography generator recipe: {generator_recipe!r}") from exc
    # Keep the schema-v1/v2 numeric namespace stable for existing consumers.
    demography_id = generator_recipe_id
    meta["demography_type_id"] = demography_type_id
    meta["generator_recipe"] = generator_recipe
    meta["generator_recipe_id"] = generator_recipe_id
    meta["generator_schema_version"] = SAMPLE_GENERATOR_SCHEMA_VERSION
    meta["scenario_key"] = "|".join(
        [
            demography_type,
            str(meta.get("map_mode", meta.get("map_type", "unknown"))),
            str(meta.get("noise_profile", "unknown")),
        ]
    )
    meta["generator_recipe_key"] = "|".join(
        [
            generator_recipe,
            str(meta.get("map_mode", meta.get("map_type", "unknown"))),
            str(meta.get("noise_profile", "unknown")),
        ]
    )

    out = {
        "sample_id": sample_id,
        "source_id": SOURCE_IDS.get(source, -1),
        "demography_type_id": demography_type_id,
        "generator_recipe_id": generator_recipe_id,
        "positions": positions_u32,
        "geno_packed": packed,
        "missing_flat_idx": missing_flat_idx,
        # Positions are float64, rates stay float32.  A float32 cannot hold an
        # integer above 2**24 = 16,777,216, so on a twenty-megabase row every
        # map breakpoint past 16.8 Mb rounds to an even base pair -- harmless
        # against cells that are one to twenty kilobases wide, but a hard wall
        # the moment the sequence grows: two base pairs at fifty megabases,
        # eight at a chromosome.  Rates are around 1e-8 and float32 carries
        # seven significant digits, so the cumulative genetic distance across
        # twenty thousand cells comes out with a relative error near 1e-7,
        # nineteen thousand times under the 0.0019 noise floor on the label.
        "obs_rec_pos": np.asarray(obs_rec_pos, dtype=np.float64),
        "obs_rec_rate": np.asarray(obs_rec_rate, dtype=np.float32),
        "obs_mut_pos": np.asarray(obs_mut_pos, dtype=np.float64),
        "obs_mut_rate": np.asarray(obs_mut_rate, dtype=np.float32),
        "target_log10_ne": np.asarray(y, dtype=np.float32),
        "metadata": meta,
    }
    # Written for a fraction of rows rather than all of them: the true maps
    # roughly double a sample on disk, and an oracle ablation -- hand the model
    # the map the data was simulated under and see how much of the gap is map
    # error rather than map use -- needs statistical power, not every row.
    # Keyed off the sample index so the subset is reproducible and does not
    # consume the noise stream.
    arg_detail = str(getattr(cfg, "arg_detail", "none"))
    write_genealogy = arg_detail != "none" and _reproducible_subset(
        index, float(getattr(cfg, "arg_fraction", 1.0))
    )
    meta["arg_detail"] = arg_detail if write_genealogy else "none"
    if arg_detail != "none":
        # Always present, like the truth maps, so the shard layout does not
        # depend on which rows were selected.  A row that was not selected
        # carries zeros and says so in ``arg_detail``.
        target_edges, _ = time_grid(cfg)
        if write_genealogy and ts is not None:
            pair_coalescences, exposure = genealogy_summary(ts, target_edges)
        else:
            pair_coalescences = np.zeros(int(cfg.time_bins), dtype=np.float64)
            exposure = np.zeros(int(cfg.time_bins), dtype=np.float64)
        out["pair_coalescences_per_bin"] = pair_coalescences.astype(np.float32)
        out["branch_length_per_bin"] = exposure.astype(np.float32)
        meta["pair_coalescences_total"] = float(pair_coalescences.sum())
        # A bin holding fewer than ten pair coalescences cannot support a rate:
        # this is the same threshold the SMC literature uses for its own time
        # discretisation.
        meta["resolvable_bins"] = int((pair_coalescences >= 10).sum())
        if arg_detail == "full" and write_genealogy and ts is not None:
            arg_dir = Path(cfg.out_dir) / "samples" / "arg"
            arg_dir.mkdir(parents=True, exist_ok=True)
            arg_path = arg_dir / f"{sample_id}.trees"
            ts.dump(str(arg_path))
            meta["arg_path"] = str(arg_path.relative_to(Path(cfg.out_dir)))
        else:
            meta["arg_path"] = ""

    truth_fraction = float(getattr(cfg, "truth_fraction", 1.0))
    write_truth = bool(cfg.include_truth) and _reproducible_subset(
        index, truth_fraction
    )
    meta["truth_maps_written"] = write_truth
    if cfg.include_truth:
        # The fields are always present so the shard layout does not depend on
        # which rows were selected; a row without truth carries zero-length
        # arrays, which the ragged offsets already express, and says so in
        # ``truth_maps_written``.
        empty = np.zeros(0, dtype=np.float32)
        empty64 = np.zeros(0, dtype=np.float64)
        out.update(
            {
                "true_rec_pos": np.asarray(true_rec_pos, dtype=np.float64) if write_truth else empty64,
                "true_rec_rate": np.asarray(true_rec_rate, dtype=np.float32) if write_truth else empty,
                "true_mut_pos": np.asarray(true_mut_pos, dtype=np.float64) if write_truth else empty64,
                "true_mut_rate": np.asarray(true_mut_rate, dtype=np.float32) if write_truth else empty,
            }
        )
    return out


def _task_with_seed(task: tuple, seed: int) -> tuple:
    if len(task) == 2:
        return (task[0], seed)
    if len(task) == 4:
        return (task[0], task[1], task[2], seed)
    raise ValueError(f"Unsupported simulator task shape: {len(task)}")


def simulate_one_with_retries(task: tuple) -> dict:
    cfg = _get_worker_cfg(task[1] if len(task) == 4 else None)
    current = task
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            return simulate_one(current)
        except StdpopsimSimulationError:
            raise
        except Exception as exc:
            last_error = exc
            seed = int(current[1] if len(current) == 2 else current[3])
            current = _task_with_seed(current, seed + 17 * (attempt + 1))
    raise RuntimeError(f"sample[{task[0]}] failed after {cfg.max_retries} attempts: {last_error}") from last_error
