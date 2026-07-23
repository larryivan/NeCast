from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import msprime
import numpy as np
import pytest

from simulator.config import (
    DEMOGRAPHY_PROB_FIELDS,
    SOURCE_PROB_FIELDS,
    Config,
    apply_preset,
    validate_config,
)
from simulator.dataset_files import write_dataset_files
from simulator.demography import (
    DEMO_PROB_FIELDS,
    DEMO_SAMPLERS,
    _clean_demography,
    sample_custom_demography,
)
from simulator.sample import simulate_one
from simulator.stdpopsim_support import try_stdpopsim_sample
from simulator.utils import bin_average_log10_ne, time_grid


def _only_demography(
    name: str,
    *,
    target_bins: int,
    generation_bins: int = 48,
) -> Config:
    cfg = dataclasses.replace(
        Config(),
        time_bins=target_bins,
        demography_generation_bins=generation_bins,
    )
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    setattr(cfg, dict(DEMO_PROB_FIELDS)[name], 1.0)
    return cfg


def _without_target_resolution(meta: dict) -> dict:
    result = dict(meta)
    result.pop("target_time_bins")
    return result


@pytest.mark.parametrize("generator_recipe", sorted(DEMO_SAMPLERS))
def test_target_resolution_does_not_change_any_latent_history_or_rng(
    generator_recipe: str,
) -> None:
    cfg48 = _only_demography(generator_recipe, target_bins=48)
    cfg64 = dataclasses.replace(cfg48, time_bins=64)
    seed = 3 if generator_recipe == "ultra_stress" else 830_001
    rng48 = np.random.default_rng(seed)
    rng64 = np.random.default_rng(seed)

    demography48, target48, meta48 = sample_custom_demography(rng48, cfg48)
    demography64, target64, meta64 = sample_custom_demography(rng64, cfg64)

    assert demography48.asdict() == demography64.asdict()
    assert _without_target_resolution(meta48) == _without_target_resolution(meta64)
    assert rng48.bit_generator.state == rng64.bit_generator.state
    assert target48.shape == (48,)
    assert target64.shape == (64,)
    if generator_recipe == "ultra_stress":
        assert meta48["stress_type"] == "ultra_random_walk"

    edges64, _ = time_grid(cfg64)
    expected64 = bin_average_log10_ne(
        edges64,
        meta64["demography_breaks"],
        meta64["demography_values"],
        meta64["demography_growth_rates"],
    )
    np.testing.assert_array_equal(target64, expected64)


def test_64_bin_target_nests_to_the_same_32_bin_analytic_target() -> None:
    cfg32 = _only_demography("dense_random_walk", target_bins=32)
    cfg64 = dataclasses.replace(cfg32, time_bins=64)

    _, target32, meta32 = sample_custom_demography(np.random.default_rng(101), cfg32)
    _, target64, meta64 = sample_custom_demography(np.random.default_rng(101), cfg64)

    assert _without_target_resolution(meta32) == _without_target_resolution(meta64)
    edges32, _ = time_grid(cfg32)
    edges64, _ = time_grid(cfg64)
    np.testing.assert_array_equal(edges64[::2], edges32)
    coarsened64 = target64.astype(np.float64).reshape(32, 2).mean(axis=1)
    np.testing.assert_allclose(target32, coarsened64, rtol=0.0, atol=5e-7)


def _full_sample_config(*, target_bins: int) -> Config:
    cfg = dataclasses.replace(
        Config(),
        dataset_name="target-grid-regression",
        seq_len=10_000,
        n_haplotypes=8,
        time_bins=target_bins,
        demography_generation_bins=48,
        baseline_mu=1.0e-7,
        map_segments_min=8,
        map_segments_max=8,
        include_truth=True,
    )
    for field in SOURCE_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_custom_hetero = 1.0
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    cfg.p_dense_random_walk = 1.0
    validate_config(cfg)
    return cfg


def test_target_resolution_does_not_change_full_same_seed_sample() -> None:
    cfg48 = _full_sample_config(target_bins=48)
    cfg64 = dataclasses.replace(cfg48, time_bins=64)
    empirical = {"rec_maps": {}, "mut_maps": {}}

    sample48 = simulate_one((0, dataclasses.asdict(cfg48), empirical, 991_337))
    sample64 = simulate_one((0, dataclasses.asdict(cfg64), empirical, 991_337))

    for key in sample48:
        if key in {"target_log10_ne", "metadata"}:
            continue
        if isinstance(sample48[key], np.ndarray):
            assert sample48[key].dtype == sample64[key].dtype, key
            np.testing.assert_array_equal(sample48[key], sample64[key], err_msg=key)
            assert sample48[key].tobytes(order="C") == sample64[key].tobytes(
                order="C"
            ), key
        else:
            assert sample48[key] == sample64[key], key

    assert sample48["target_log10_ne"].shape == (48,)
    assert sample64["target_log10_ne"].shape == (64,)
    assert sample48["target_log10_ne"].dtype == np.float32
    assert sample64["target_log10_ne"].dtype == np.float32
    assert _without_target_resolution(sample48["metadata"]) == _without_target_resolution(
        sample64["metadata"]
    )


def test_core_manifest_records_distinct_target_and_history_grids(
    tmp_path: Path,
) -> None:
    cfg = apply_preset(Config(preset="core_v01"))
    cfg.out_dir = str(tmp_path / "dataset")

    write_dataset_files(cfg, {"n_samples": 0})

    config = json.loads(
        (Path(cfg.out_dir) / "config.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (Path(cfg.out_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    for payload in (config, manifest):
        assert payload["time_bins"] == 64
        assert payload["demography_generation_bins"] == 48
        assert len(payload["time_edges"]) == 65
        assert len(payload["time_mids"]) == 64
        assert len(payload["demography_generation_time_edges"]) == 49
        assert len(payload["demography_generation_time_mids"]) == 48
        assert payload["time_edges"][0] == payload["demography_generation_time_edges"][0]
        assert payload["time_edges"][-1] == payload["demography_generation_time_edges"][-1]


def test_stdpopsim_target_resolution_does_not_change_model_or_simulation_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_demography = msprime.Demography()
    source_demography.add_population(
        name="target",
        initial_size=80_000.0,
        growth_rate=0.001,
    )
    source_demography.add_population_parameters_change(
        time=777.0,
        initial_size=12_000.0,
        growth_rate=0.0,
        population="target",
    )
    source_demography.sort_events()

    class FakeTreeSequence:
        sequence_length = 10_000

        def trim(self):
            return self

    simulation_seeds: list[int] = []

    class FakeEngine:
        def simulate(self, model, contig, samples, seed):
            simulation_seeds.append(int(seed))
            return FakeTreeSequence()

    model = SimpleNamespace(
        id="FakeModel",
        populations=[SimpleNamespace(id="target")],
        model=source_demography,
        get_samples=lambda *args: {"target": int(args[0])},
    )
    species = SimpleNamespace(
        demographic_models=[model],
        get_demographic_model=lambda model_id: model,
        get_contig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    fake_stdpopsim = SimpleNamespace(
        get_species=lambda species_id: species,
        get_engine=lambda engine_id: FakeEngine(),
    )
    monkeypatch.setitem(sys.modules, "stdpopsim", fake_stdpopsim)

    cfg48 = dataclasses.replace(
        Config(),
        enable_stdpopsim=True,
        stdpopsim_models="FakeModel",
        time_bins=48,
        demography_generation_bins=48,
    )
    cfg64 = dataclasses.replace(cfg48, time_bins=64)
    result48 = try_stdpopsim_sample(np.random.default_rng(2026), cfg48)
    result64 = try_stdpopsim_sample(np.random.default_rng(2026), cfg64)

    assert result48 is not None
    assert result64 is not None
    _, target48, meta48, _ = result48
    _, target64, meta64, _ = result64
    assert simulation_seeds[0] == simulation_seeds[1]
    assert _without_target_resolution(meta48) == _without_target_resolution(meta64)
    assert target48.shape == (48,)
    assert target64.shape == (64,)


def _trajectory_hash(
    breaks: list[float],
    values: list[float],
    growth_rates: list[float],
) -> str:
    digest = hashlib.sha256()
    for items in (breaks, values, growth_rates):
        array = np.asarray(items, dtype="<f8")
        digest.update(np.asarray([array.size], dtype="<u8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("recipe", "seed", "expected_hash", "expected_tail"),
    [
        (
            "near_constant",
            830_001,
            "6532168ce5d042a5536431a1969fb4ebb720b7a6bc83e1d3404d5f30d5803b5d",
            12_069_388_081_325_197_133,
        ),
        (
            "smooth_random_walk",
            830_001,
            "f72e067f27687275783898ac8b5b4a652c315327268c3f1033b73929c33e7d45",
            11_108_812_135_067_975_242,
        ),
        (
            "smooth_random_walk_stress",
            830_001,
            "63673dd4a70a471c01ef48c148cefb61ca98001bedca166669020052507bf736",
            13_076_481_703_583_119_576,
        ),
        (
            "dense_random_walk",
            830_001,
            "2fcd667f7081442c24a97c87a7733918f7c6d69b2f0c325a74450b865a3968c4",
            11_108_812_135_067_975_242,
        ),
        (
            "dense_multiscale_random_walk",
            830_001,
            "c474d0d1d0b349a469be6e6ed4c2480fde829f78d95c1381c75434d6e379d968",
            2_521_459_529_592_817_122,
        ),
        (
            "dense_recent_wiggle",
            830_001,
            "12e70a940e2ce1a17cac770951772dfc48b855bc5fbb2072fc494f6822537af5",
            11_108_812_135_067_975_242,
        ),
        (
            "dense_random_walk_stress",
            830_001,
            "f9a579476fc85fd5f2a20cc3a14a961b7f30fadb8cebf4afef136e46a4a8ca9b",
            18_158_613_176_814_642_464,
        ),
        (
            "empirical_human_ne_template",
            830_001,
            "119b43b228985443b06372a112a0516d2f528713248f1bb3591b6b1a7c54da77",
            4_014_981_613_137_965_452,
        ),
        (
            "ultra_stress",
            3,
            "4630cfc6899230e5ee7a1f278d159d324fcdb6cb10933220f19aea1e9f65a6bf",
            361_563_904_759_131_822,
        ),
    ],
)
def test_generation_grid_preserves_pre_upgrade_latent_golden(
    recipe: str,
    seed: int,
    expected_hash: str,
    expected_tail: int,
) -> None:
    cfg = Config(time_bins=48, demography_generation_bins=48)
    edges, mids = time_grid(cfg)
    rng = np.random.default_rng(seed)

    breaks, values, meta = DEMO_SAMPLERS[recipe](rng, cfg, edges, mids)
    breaks, values, growth_rates = _clean_demography(
        breaks,
        values,
        meta.get("demography_growth_rates"),
    )

    assert _trajectory_hash(breaks, values, growth_rates) == expected_hash
    if recipe == "ultra_stress":
        assert meta["stress_type"] == "ultra_random_walk"
    assert int(rng.bit_generator.random_raw()) == expected_tail
