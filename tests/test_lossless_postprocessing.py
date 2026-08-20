from __future__ import annotations

import dataclasses
import inspect
import math
import textwrap
from typing import Any

import numpy as np
import pytest

import simulator.noise as noise_module
import simulator.sample as sample_module
from simulator.config import (
    DEMOGRAPHY_PROB_FIELDS,
    SOURCE_PROB_FIELDS,
    Config,
    validate_config,
)
from simulator.noise import apply_genotype_noise, pack_haplotypes


def _reference_apply_phase_switch_errors(
    G: np.ndarray,
    positions_bp: np.ndarray,
    rng: np.random.Generator,
    switch_rate_per_het: float,
) -> tuple[np.ndarray, dict]:
    """A deliberately plain statement of the switch model.

    Switches are drawn at heterozygous sites, because that is where a phasing
    algorithm makes a decision; the parity is then carried forward to every
    later site, homozygous ones included, since the two haplotypes really are
    exchanged there even though the swap is unobservable.
    """

    X = np.empty_like(G, dtype=np.uint8, order="K", subok=True)
    np.greater(G, 0, out=X)
    stats = {
        "phase_switch_count": 0,
        "phase_switch_pair_count": 0,
        "phaseable_pair_count": int(X.shape[1] // 2),
        "phase_switch_heterozygous_sites": 0,
    }
    if X.shape[0] < 2 or X.shape[1] < 2 or switch_rate_per_het <= 0:
        return X, stats
    probability = float(np.clip(switch_rate_per_het, 0.0, 1.0))
    for left in range(0, X.shape[1] - 1, 2):
        heterozygous = X[:, left] != X[:, left + 1]
        het_index = np.flatnonzero(heterozygous)
        stats["phase_switch_heterozygous_sites"] += len(het_index)
        if len(het_index) < 2:
            continue
        switches = rng.random(len(het_index)) < probability
        switches[0] = False
        if not switches.any():
            continue
        parity = False
        flipped = np.zeros(X.shape[0], dtype=bool)
        cursor = 0
        for site in range(X.shape[0]):
            if cursor < len(het_index) and het_index[cursor] == site:
                if switches[cursor]:
                    parity = not parity
                cursor += 1
            flipped[site] = parity
        stats["phase_switch_count"] += int(switches.sum())
        stats["phase_switch_pair_count"] += 1
        a = X[flipped, left].copy()
        X[flipped, left] = X[flipped, left + 1]
        X[flipped, left + 1] = a
    return X, stats


def _reference_apply_genotype_noise(
    G: np.ndarray,
    positions_bp: np.ndarray,
    rng: np.random.Generator,
    noise: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if G.size == 0:
        stats = {"phase_switch_count": 0, "phase_switch_pair_count": 0, "phaseable_pair_count": 0}
        return G.astype(np.uint8), np.zeros(0, dtype=np.uint32), stats

    ge = float(noise.get("genotype_error", 0.0))
    mr = float(noise.get("missing_rate", 0.0))
    phase_rate = float(noise.get("phase_switch_rate_per_het", 0.0))
    X, stats = _reference_apply_phase_switch_errors(G, positions_bp, rng, phase_rate)
    flat = X.reshape(-1)
    if ge > 0:
        flip_idx = noise_module._sample_flat_indices(rng, flat.size, ge)
        flat[flip_idx] = 1 - flat[flip_idx]
    missing_flat_idx = noise_module._sample_flat_indices(rng, flat.size, mr)
    if len(missing_flat_idx):
        flat[missing_flat_idx] = 0
    return X, missing_flat_idx, stats


def _reference_pack_haplotypes(G: np.ndarray, n_haplotypes: int) -> np.ndarray:
    if G.shape[1] != n_haplotypes:
        raise ValueError(f"Expected H={n_haplotypes}, got {G.shape[1]}")
    return np.packbits(G.astype(np.uint8), axis=1, bitorder="little")


def _assert_exact(left: Any, right: Any, path: str = "value") -> None:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and isinstance(right, np.ndarray), path
        assert left.shape == right.shape, path
        assert left.dtype == right.dtype, path
        assert np.array_equal(left, right, equal_nan=True), path
        assert left.tobytes(order="C") == right.tobytes(order="C"), path
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert list(left) == list(right), path
        for key in left:
            _assert_exact(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert type(left) is type(right), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_exact(left_item, right_item, f"{path}[{index}]")
        return
    assert type(left) is type(right), path
    if isinstance(left, float) and math.isnan(left):
        assert math.isnan(right), path
    else:
        assert left == right, path


@pytest.mark.parametrize(
    "noise",
    [
        {
            "genotype_error": 0.0,
            "missing_rate": 0.0,
            "phase_switch_rate_per_het": 0.0,
        },
        {
            "genotype_error": 0.25,
            "missing_rate": 0.20,
            "phase_switch_rate_per_het": 0.05,
        },
    ],
)
@pytest.mark.parametrize("seed", [7, 991, 20_241])
def test_preallocated_uint8_noise_is_byte_exact_and_preserves_rng_order(
    noise: dict,
    seed: int,
) -> None:
    G = np.array(
        [
            [-1, 0, 1, 2, 3, 9],
            [3, 2, 1, 0, -1, -5],
            [0, -1, 0, 1, 9, 2],
            [2, 0, -5, 0, 1, 3],
        ],
        dtype=np.int32,
    )[:, ::-1]
    assert not G.flags.c_contiguous
    positions = np.array([0.25, 17.5, 31.25, 100.75], dtype=np.float64)
    reference_rng = np.random.default_rng(seed)
    candidate_rng = np.random.default_rng(seed)

    reference = _reference_apply_genotype_noise(G, positions, reference_rng, noise)
    candidate = apply_genotype_noise(G, positions, candidate_rng, noise)

    _assert_exact(reference, candidate)
    np.testing.assert_array_equal(
        reference_rng.integers(0, 2**31, size=32),
        candidate_rng.integers(0, 2**31, size=32),
    )


def test_preallocated_uint8_noise_preserves_empty_matrix_contract() -> None:
    G = np.empty((0, 6), dtype=np.int32)
    positions = np.empty(0, dtype=np.float64)
    noise = {
        "genotype_error": 0.5,
        "missing_rate": 0.5,
        "phase_switch_rate_per_het": 1.0,
    }

    reference = _reference_apply_genotype_noise(
        G, positions, np.random.default_rng(3), noise
    )
    candidate = apply_genotype_noise(G, positions, np.random.default_rng(3), noise)

    _assert_exact(reference, candidate)


def test_preallocated_uint8_noise_preserves_fortran_order_semantics() -> None:
    G = np.asfortranarray(
        [
            [0, 1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 0],
        ],
        dtype=np.int32,
    )
    positions = np.array([0.0, 10.0, 25.0, 80.0], dtype=np.float64)
    noise = {
        "genotype_error": 0.25,
        "missing_rate": 0.20,
        "phase_switch_rate_per_het": 0.05,
    }
    reference_rng = np.random.default_rng(9127)
    candidate_rng = np.random.default_rng(9127)

    reference = _reference_apply_genotype_noise(
        G, positions, reference_rng, noise
    )
    candidate = apply_genotype_noise(G, positions, candidate_rng, noise)

    _assert_exact(reference, candidate)
    assert reference[0].flags.f_contiguous
    assert candidate[0].flags.f_contiguous
    np.testing.assert_array_equal(
        reference_rng.integers(0, 2**31, size=32),
        candidate_rng.integers(0, 2**31, size=32),
    )


class _AstypeForbidden(np.ndarray):
    def astype(self, *args, **kwargs):
        raise AssertionError("uint8 pack path must not call astype")


def test_pack_haplotypes_skips_astype_for_uint8_and_keeps_legacy_fallback() -> None:
    uint8_genotypes = np.array(
        [[0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1]],
        dtype=np.uint8,
    ).view(_AstypeForbidden)
    expected_uint8 = np.packbits(
        np.asarray(uint8_genotypes), axis=1, bitorder="little"
    )

    packed_uint8 = pack_haplotypes(uint8_genotypes, 6)

    _assert_exact(expected_uint8, packed_uint8)
    assert packed_uint8.dtype == np.uint8

    legacy_input = np.array(
        [[-1, 0, 1, 2, 255, 256], [3, 4, 0, 0, 1, -2]],
        dtype=np.int32,
    )
    expected_legacy = _reference_pack_haplotypes(legacy_input, 6)
    packed_legacy = pack_haplotypes(legacy_input, 6)
    _assert_exact(expected_legacy, packed_legacy)


def _reference_simulate_one():
    source = textwrap.dedent(inspect.getsource(sample_module.simulate_one))
    replacements = [
        (
            "positions = np.asarray(ts.sites_position, dtype=np.float64)",
            "positions = np.array([site.position for site in ts.sites()], dtype=np.float64)",
        ),
        (
            '"geno_packed": packed,',
            '"geno_packed": packed.astype(np.uint8),',
        ),
    ]
    for current, reference in replacements:
        assert source.count(current) == 1
        source = source.replace(current, reference)
    namespace = dict(vars(sample_module))
    namespace["apply_genotype_noise"] = _reference_apply_genotype_noise
    namespace["pack_haplotypes"] = _reference_pack_haplotypes
    exec(compile(source, "<reference simulate_one>", "exec"), namespace)
    return namespace["simulate_one"]


def _fixed_config(source: str, *, include_truth: bool) -> Config:
    cfg = Config(
        dataset_name="lossless-postprocessing-test",
        out_dir="/tmp/dlcoal-lossless-postprocessing-test-output",
        n_samples=1,
        shard_size=1,
        seq_len=25_000,
        n_haplotypes=8,
        time_bins=8,
        min_time=50.0,
        max_time=20_000.0,
        baseline_mu=1.0e-7,
        baseline_rec=1.0e-8,
        map_segments_min=4,
        map_segments_max=8,
        genotype_error_min=0.05,
        genotype_error_max=0.05,
        missing_rate_min=0.10,
        missing_rate_max=0.10,
        phase_switch_rate_per_het_min=1.0,
        phase_switch_rate_per_het_max=1.0,
        include_truth=include_truth,
        enable_stdpopsim=False,
        workers=1,
    )
    for field in SOURCE_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    source_field = {
        "clean_constant_map": "p_clean_constant",
        "custom_heterogeneous": "p_custom_hetero",
    }[source]
    setattr(cfg, source_field, 1.0)
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    setattr(cfg, "p_constant" if source == "clean_constant_map" else "p_multi_event", 1.0)
    validate_config(cfg)
    return cfg


@pytest.mark.parametrize(
    ("source", "include_truth", "seed"),
    [
        ("clean_constant_map", False, 73_531),
        ("custom_heterogeneous", True, 88_127),
    ],
)
def test_fixed_seed_complete_sample_is_byte_exact_to_reference(
    source: str,
    include_truth: bool,
    seed: int,
) -> None:
    cfg = _fixed_config(source, include_truth=include_truth)
    task = (0, dataclasses.asdict(cfg), {"rec_maps": {}, "mut_maps": {}}, seed)
    reference = _reference_simulate_one()(task)
    candidate = sample_module.simulate_one(task)

    _assert_exact(reference, candidate, "sample")
    assert candidate["positions"].dtype == np.uint32
    assert candidate["geno_packed"].dtype == np.uint8
    assert candidate["missing_flat_idx"].dtype == np.uint32
    assert len(candidate["positions"]) > 0
    if source == "custom_heterogeneous":
        assert len(candidate["missing_flat_idx"]) > 0
