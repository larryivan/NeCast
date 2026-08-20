from __future__ import annotations

import numpy as np
import pytest

import simulator.empirical_maps as empirical_maps_module
from simulator.config import Config
from simulator.empirical_maps import (
    EmpiricalMapStore,
    PiecewiseRateMap,
    load_empirical_rate_table,
    slice_piecewise_map,
)
from simulator.maps import sample_maps_for_source


def _rate_map(chrom: str, positions: list[float], rates: list[float]) -> PiecewiseRateMap:
    return PiecewiseRateMap(
        chrom=chrom,
        position=np.asarray(positions, dtype=np.float64),
        rate=np.asarray(rates, dtype=np.float64),
    )


def test_common_chroms_require_a_shared_coordinate_span() -> None:
    store = EmpiricalMapStore(
        rec_maps={"chr1": _rate_map("chr1", [100, 250, 400], [1e-8, 2e-8])},
        mut_maps={"chr1": _rate_map("chr1", [120, 300, 500], [3e-8, 4e-8])},
    )

    assert store.common_slice_start_bounds("chr1", 280) == (120.0, 120.0)
    assert store.common_chroms(280) == ["chr1"]
    assert store.common_chroms(281) == []


def test_slice_piecewise_map_accepts_an_explicit_start_and_exact_end_boundary() -> None:
    rate_map = _rate_map("chr1", [100, 150, 250, 400], [1e-8, 2e-8, 3e-8])

    pos, true_rate, obs_pos, _, meta = slice_piecewise_map(
        np.random.default_rng(1),
        rate_map,
        length=100,
        obs_noise_sigma=0.0,
        baseline_rate=1e-8,
        start=150,
    )

    np.testing.assert_array_equal(pos, [0.0, 100.0])
    np.testing.assert_array_equal(obs_pos, pos)
    np.testing.assert_array_equal(true_rate, [2e-8])
    assert meta["slice_start_bp"] == 150.0
    assert meta["slice_end_bp"] == 250.0


def test_empirical_source_uses_one_colocated_interval_for_both_maps() -> None:
    cfg = Config(seq_len=100, baseline_rec=1e-8, baseline_mu=1e-8, observed_map_noise=0.0)
    store = EmpiricalMapStore(
        rec_maps={"chr1": _rate_map("chr1", [100, 200, 300, 400], [1e-8, 2e-8, 3e-8])},
        mut_maps={"chr1": _rate_map("chr1", [120, 220, 320, 420], [4e-8, 5e-8, 6e-8])},
    )

    *_, meta = sample_maps_for_source(
        np.random.default_rng(7),
        cfg,
        "empirical_map_slice",
        store,
    )

    rec_meta = meta["rec_slice"]
    mut_meta = meta["mut_slice"]
    assert meta["slice_alignment"] == "shared_chromosomal_interval"
    assert meta["slice_chrom"] == rec_meta["chrom"] == mut_meta["chrom"] == "chr1"
    assert meta["slice_start_bp"] == rec_meta["slice_start_bp"] == mut_meta["slice_start_bp"]
    assert meta["slice_end_bp"] == rec_meta["slice_end_bp"] == mut_meta["slice_end_bp"]
    assert meta["slice_end_bp"] - meta["slice_start_bp"] == pytest.approx(cfg.seq_len)
    assert 120.0 <= meta["slice_start_bp"] <= 300.0


def test_explicit_start_must_fit_inside_the_source_map() -> None:
    rate_map = _rate_map("chr1", [100, 200, 300], [1e-8, 2e-8])

    with pytest.raises(ValueError, match="outside the valid range"):
        slice_piecewise_map(
            np.random.default_rng(1),
            rate_map,
            length=100,
            obs_noise_sigma=0.0,
            baseline_rate=1e-8,
            start=201,
        )


def test_non_finite_empirical_coordinates_are_rejected(tmp_path) -> None:
    path = tmp_path / "invalid-map.bed"
    path.write_text(
        "chr1 0 100 1e-8\n"
        "chr1 100 inf 2e-8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite"):
        load_empirical_rate_table(
            str(path),
            baseline_rate=1e-8,
            map_kind="recombination",
        )


def test_non_finite_manual_rate_map_is_rejected_before_slicing() -> None:
    rate_map = _rate_map("chr1", [0.0, 100.0], [float("inf")])

    with pytest.raises(ValueError, match="non-finite"):
        slice_piecewise_map(
            np.random.default_rng(1),
            rate_map,
            length=50,
            obs_noise_sigma=0.0,
            baseline_rate=1e-8,
        )


def test_common_span_validation_is_cached_off_the_sample_hot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EmpiricalMapStore(
        rec_maps={"chr1": _rate_map("chr1", [0, 100, 200], [1e-8, 2e-8])},
        mut_maps={"chr1": _rate_map("chr1", [0, 120, 220], [3e-8, 4e-8])},
    )
    original = empirical_maps_module._validate_piecewise_rate_map
    calls = 0

    def counted(rate_map: PiecewiseRateMap) -> None:
        nonlocal calls
        calls += 1
        original(rate_map)

    monkeypatch.setattr(empirical_maps_module, "_validate_piecewise_rate_map", counted)

    assert store.is_available(100)
    assert store.is_available(100)
    assert store.common_chroms(100) == ["chr1"]
    assert store.common_slice_start_bounds("chr1", 100) == (0.0, 100.0)
    assert calls == 2

    cfg = Config(seq_len=100, observed_map_noise=0.0)
    sample_maps_for_source(np.random.default_rng(3), cfg, "empirical_map_slice", store)
    sample_maps_for_source(np.random.default_rng(4), cfg, "empirical_map_slice", store)
    assert calls == 2
