from __future__ import annotations

import atexit

import msprime
import numpy as np

from .config import Config, DEMO_IDS, SOURCE_IDS, validate_config
from .demography import sample_custom_demography
from .empirical_maps import EmpiricalMapStore
from .maps import choose_source, constant_map, make_rate_map, sample_maps_for_source
from .noise import apply_genotype_noise, pack_haplotypes
from .stdpopsim_support import StdpopsimSimulationError, try_stdpopsim_sample
from .utils import rng_from_seed


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
    G_noisy, missing_flat_idx, noise_stats = apply_genotype_noise(G, positions, rng, noise)
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
    meta["scenario_key"] = "|".join(
        [
            str(meta.get("demography_type", "unknown")),
            str(meta.get("map_mode", meta.get("map_type", "unknown"))),
            str(meta.get("noise_profile", "unknown")),
        ]
    )

    out = {
        "sample_id": sample_id,
        "source_id": SOURCE_IDS.get(source, -1),
        "demo_id": DEMO_IDS.get(meta.get("demography_type", "stdpopsim"), 9),
        "positions": positions_u32,
        "geno_packed": packed,
        "missing_flat_idx": missing_flat_idx,
        "obs_rec_pos": np.asarray(obs_rec_pos, dtype=np.float32),
        "obs_rec_rate": np.asarray(obs_rec_rate, dtype=np.float32),
        "obs_mut_pos": np.asarray(obs_mut_pos, dtype=np.float32),
        "obs_mut_rate": np.asarray(obs_mut_rate, dtype=np.float32),
        "target_log10_ne": np.asarray(y, dtype=np.float32),
        "metadata": meta,
    }
    if cfg.include_truth:
        out.update(
            {
                "true_rec_pos": np.asarray(true_rec_pos, dtype=np.float32),
                "true_rec_rate": np.asarray(true_rec_rate, dtype=np.float32),
                "true_mut_pos": np.asarray(true_mut_pos, dtype=np.float32),
                "true_mut_rate": np.asarray(true_mut_rate, dtype=np.float32),
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
