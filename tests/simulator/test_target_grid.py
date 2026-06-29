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

from necast.simulator.config import (
    DEMOGRAPHY_PROB_FIELDS,
    SOURCE_PROB_FIELDS,
    Config,
    apply_preset,
    validate_config,
)
from necast.simulator.dataset_files import write_dataset_files
from necast.simulator.demography import (
    DEMO_PROB_FIELDS,
    DEMO_SAMPLERS,
    _clean_demography,
    epochs_from_curve,
    sample_custom_demography,
    spectral_walk_curve,
)
from necast.simulator.sample import simulate_one
from necast.simulator.stdpopsim_support import try_stdpopsim_sample
from necast.simulator.utils import bin_average_log10_ne, time_grid


def _only_demography(
    name: str,
    *,
    target_bins: int,
) -> Config:
    cfg = dataclasses.replace(
        Config(),
        time_bins=target_bins,
    )
    for field in DEMOGRAPHY_PROB_FIELDS:
        setattr(cfg, field, 0.0)
    setattr(cfg, dict(DEMO_PROB_FIELDS)[name], 1.0)
    return cfg


def _without_target_resolution(meta: dict) -> dict:
    """Drop what the target grid decides, keeping what the history decides.

    The genealogy summary is bucketed on the target grid, so its total is a
    sum over a different number of terms at 48 bins than at 64 and differs in
    the last bits; the count of resolvable bins is resolution-dependent by
    definition.
    """

    result = dict(meta)
    for key in (
        "target_time_bins",
        "pair_coalescences_total",
        "resolvable_bins",
    ):
        result.pop(key, None)
    return result


@pytest.mark.parametrize("generator_recipe", sorted(DEMO_SAMPLERS))
def test_target_resolution_does_not_change_any_latent_history_or_rng(
    generator_recipe: str,
) -> None:
    cfg48 = _only_demography(generator_recipe, target_bins=48)
    cfg64 = dataclasses.replace(cfg48, time_bins=64)
    seed = 830_001
    rng48 = np.random.default_rng(seed)
    rng64 = np.random.default_rng(seed)

    demography48, target48, meta48 = sample_custom_demography(rng48, cfg48)
    demography64, target64, meta64 = sample_custom_demography(rng64, cfg64)

    assert demography48.asdict() == demography64.asdict()
    assert _without_target_resolution(meta48) == _without_target_resolution(meta64)
    assert rng48.bit_generator.state == rng64.bit_generator.state
    assert target48.shape == (48,)
    assert target64.shape == (64,)

    edges64, _ = time_grid(cfg64)
    expected64 = bin_average_log10_ne(
        edges64,
        meta64["demography_breaks"],
        meta64["demography_values"],
        meta64["demography_growth_rates"],
    )
    np.testing.assert_array_equal(target64, expected64)


def test_64_bin_target_nests_to_the_same_32_bin_analytic_target() -> None:
    cfg32 = _only_demography("spectral_walk", target_bins=32)
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
    cfg.p_spectral_walk = 1.0
    validate_config(cfg)
    return cfg


def test_target_resolution_does_not_change_full_same_seed_sample() -> None:
    cfg48 = _full_sample_config(target_bins=48)
    cfg64 = dataclasses.replace(cfg48, time_bins=64)
    empirical = {"rec_maps": {}, "mut_maps": {}}

    sample48 = simulate_one((0, dataclasses.asdict(cfg48), empirical, 991_337))
    sample64 = simulate_one((0, dataclasses.asdict(cfg64), empirical, 991_337))

    for key in sample48:
        # The genealogy summary is binned on the target grid, like the label
        # itself, so it is a function of the resolution by design: it exists to
        # say how many coalescences fell in each *target* bin.  Everything else
        # is the latent history and the sequence, which the resolution must not
        # touch.
        if key in {
            "target_log10_ne",
            "metadata",
            "pair_coalescences_per_bin",
            "branch_length_per_bin",
        }:
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
    assert sample48["pair_coalescences_per_bin"].shape == (48,)
    assert sample64["pair_coalescences_per_bin"].shape == (64,)
    # Resolution changes how the genealogy is bucketed, never how much of it
    # there was.
    assert sample48["pair_coalescences_per_bin"].sum() == pytest.approx(
        sample64["pair_coalescences_per_bin"].sum()
    )
    assert sample48["branch_length_per_bin"].sum() == pytest.approx(
        sample64["branch_length_per_bin"].sum(), rel=1e-5
    )
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
        assert payload["demography_resolution_log10"] > 0.0
        assert len(payload["time_edges"]) == 65
        assert len(payload["time_mids"]) == 64


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
            "bottleneck",
            830_001,
            "70b639b73c9625ec5ef906a8e0055e957a6892cfe6250e70a82c0556f6c85be5",
            17_656_288_953_729_531_408,
        ),
        (
            "constant",
            830_001,
            "4e5cf9c3bcc7ef9cc8bb48c7ab4802b3da9155508bd9e41f77ad497ffece37fa",
            12_215_142_104_914_073_923,
        ),
        (
            "empirical_human_ne_template",
            830_001,
            "3887f17b0e7f640d4b899e57157f313bb91a4c62eff9849b40209f1dbdb55947",
            6_711_622_359_113_815_556,
        ),
        (
            "event_on_background",
            830_001,
            "ac2ca4238a86c4d7997f04bba9edd40503304ba31b4b5e30c547328395c134d0",
            1_791_161_688_086_831_483,
        ),
        (
            "exponential",
            830_001,
            "ded50d0cf62a2014618688b97f45e6f7c23c773bca484c51bb69c35006e1d248",
            11_741_280_483_675_874_754,
        ),
        (
            "literature_sawtooth",
            830_001,
            "a36bf2b88c4f3820f5b65fb36e68fb9de5f0d3f3db210543cca4f97413fef756",
            6_088_537_202_534_173_187,
        ),
        (
            "multi_event",
            830_001,
            "d7e8a8310ea3d3a05c38772af19e9595385063b05f1e1928a143cbc339a0c233",
            16_161_667_349_681_506_290,
        ),
        (
            "ooa_chain",
            830_001,
            "42b256f2f267d84e5e2512be52a0fc1f6b5e1aa57f0656a358825d6a5368326d",
            14_361_798_432_631_189_947,
        ),
        (
            "sawtooth",
            830_001,
            "ac5b75ae0f99ae1011f3d044a5294594b6bff86693db72936174c4441ecca331",
            9_737_454_704_092_519_409,
        ),
        (
            "size_event",
            830_001,
            "c435c005291ef1914244545f9e34254f3f2676b40be18a0c59a81af191554f2d",
            10_801_209_010_986_097_841,
        ),
        (
            "spectral_walk",
            830_001,
            "7f1579fa7c95f1ef917b2947e059b9052effb7fef5fd15ee3af94679ae9d0b75",
            12_005_372_267_971_068_725,
        ),
        (
            "stdpopsim_single_pop",
            830_001,
            "35dbe4e6be41c2b07689b21ae28086d84478a05c16c1737791afc943d6de362c",
            12_215_142_104_914_073_923,
        ),
    ],
)
def test_latent_histories_are_pinned(
    # Nine of these twelve are byte-identical to the goldens taken before
    # histories became functions of continuous time, random-number tails
    # included, which is the evidence that the change reached only what it was
    # aimed at.  The three that moved are the three that were being evaluated on
    # a grid: ``spectral_walk`` and ``event_on_background`` keep their tails and
    # changed only where their epochs fall, and ``empirical_human_ne_template``
    # draws its perturbation per template knot instead of per label bin, which
    # is a different number of draws and so a different tail.
    recipe: str,
    seed: int,
    expected_hash: str,
    expected_tail: int,
) -> None:
    cfg = Config()
    edges, mids = time_grid(cfg)
    rng = np.random.default_rng(seed)

    breaks, values, meta = DEMO_SAMPLERS[recipe](rng, cfg, edges, mids)
    breaks, values, growth_rates = _clean_demography(
        breaks,
        values,
        meta.get("demography_growth_rates"),
        # Mirror the production path: the band a sampler is clipped to is part
        # of what the golden pins, so reading it here would silently test the
        # regular band for a stress recipe.
        stress="stress" in recipe,
    )

    assert _trajectory_hash(breaks, values, growth_rates) == expected_hash
    assert int(rng.bit_generator.random_raw()) == expected_tail


# --------------------------------------------------------------------------
# The label grid is a view of the history, never an input to it
# --------------------------------------------------------------------------


def test_the_latent_history_does_not_depend_on_the_label_resolution() -> None:
    """Asking for a finer label must not change the prior the data comes from.

    The samplers used to step ``cfg.time_bins`` times and place their knots at a
    fixed fraction of that count, so moving the target from 48 bins to 64 moved
    the histories with it.  That is why a separate, frozen generation grid had
    to exist at all.  Histories are now functions of continuous time, so the
    same seed gives the same history at any label resolution and the grid is
    free to move.
    """

    for seed in range(120):
        histories = []
        for bins in (64, 96, 128):
            cfg = dataclasses.replace(Config(), time_bins=bins)
            _, target, meta = sample_custom_demography(np.random.default_rng(seed), cfg)
            assert np.asarray(target).shape == (bins,)
            histories.append(
                (
                    meta["generator_recipe"],
                    tuple(meta["demography_breaks"]),
                    tuple(meta["demography_values"]),
                    tuple(meta["demography_growth_rates"]),
                )
            )
        assert histories[0] == histories[1] == histories[2], (
            f"seed {seed} ({histories[0][0]}) drew a different history per label grid"
        )


def test_every_label_resolution_is_a_view_of_the_same_curve() -> None:
    """And the label itself is that history averaged onto whichever grid."""

    for seed in range(60):
        for bins in (64, 96, 128):
            cfg = dataclasses.replace(Config(), time_bins=bins)
            _, target, meta = sample_custom_demography(np.random.default_rng(seed), cfg)
            np.testing.assert_allclose(
                bin_average_log10_ne(
                    time_grid(cfg)[0],
                    list(meta["demography_breaks"]),
                    list(meta["demography_values"]),
                    list(meta["demography_growth_rates"]),
                ),
                target,
                rtol=0.0,
                atol=1e-9,
            )


def test_epochs_follow_the_curve_to_the_tolerance_they_promise() -> None:
    """``epochs_from_curve`` is defined by an error bound, so hold it to it.

    The bound is what replaces the bin count: nothing downstream may assume a
    number of epochs, only that the piecewise-constant history is never further
    than ``tol`` from the curve it stands for.
    """

    rng = np.random.default_rng(3)
    for _ in range(30):
        seed = int(rng.integers(1 << 40))
        slope = float(rng.uniform(0.6, 2.0))
        modes = int(rng.integers(6, 49))
        tilt = float(rng.uniform(-1.2, 1.2))
        span = float(rng.uniform(0.3, 3.0))

        def curve(t: np.ndarray) -> np.ndarray:
            return spectral_walk_curve(t, seed, slope, modes, tilt, span, 4.0)

        for tol in (0.1, 0.02):
            breaks, values = epochs_from_curve(curve, 50.0, 500_000.0, tol)
            edges = np.concatenate(([50.0], breaks, [500_000.0]))
            for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                probe = curve(np.geomspace(left, right, 33))
                held = np.log10(values[index])
                assert float(np.max(np.abs(probe - held))) <= tol, (
                    f"epoch {index} drifts past its {tol} bound"
                )


# --------------------------------------------------------------------------
# The genealogy summary has to be sufficient for what it claims
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_haplotypes", [64, 128, 256])
def test_the_stored_genealogy_summary_reconstructs_the_coalescence_rate(
    n_haplotypes: int,
) -> None:
    """The counts are stored so a rate can be rebuilt from them; hold them to it.

    Rebuilding needs the pairs still uncoalesced entering each bin, which is the
    total minus everything that has already met -- including whatever met more
    recently than the first bin edge.  That is one to two percent of pairs, and
    leaving it out barely moves the recent bins but compounds into the deep
    ones, where only a handful of pairs are left: measured at 256 haplotypes it
    reached 3.3 in log10 Ne before ``pair_coalescences_outside`` existed.
    """

    import msprime

    from necast.simulator.sample import genealogy_summary

    edges, _ = time_grid(Config())
    windows = np.concatenate(([0.0], edges, [np.inf]))
    ts = msprime.sim_ancestry(
        samples=n_haplotypes // 2,
        sequence_length=5_000_000,
        recombination_rate=1.25e-8,
        population_size=12_000,
        ploidy=2,
        random_seed=13,
    )
    counts, _, outside = genealogy_summary(ts, edges)

    total = n_haplotypes * (n_haplotypes - 1) // 2
    assert float(counts.sum() + outside.sum()) == pytest.approx(total, rel=1e-9)
    assert outside[0] > 0.0, "some pairs always coalesce more recently than 50 generations"

    # Rebuild the rate the way a reader would, and check it against tskit's own.
    remaining = total - outside[0] - np.concatenate(([0.0], np.cumsum(counts)[:-1]))
    with np.errstate(all="ignore"):
        fraction = np.clip(counts / np.maximum(remaining, 1e-12), 0.0, 1.0 - 1e-12)
        rebuilt = -np.log1p(-fraction) / np.diff(edges)
    reference = np.asarray(ts.pair_coalescence_rates(time_windows=windows))[1:-1]

    # A bin in which every pair still at risk coalesces has no rate to report:
    # the survival form divides by what is left over and nothing is.  tskit
    # answers such a bin with its own convention; exclude it rather than pin
    # someone else's tail handling.
    exhausted = counts >= remaining - 1e-9
    usable = np.isfinite(reference) & (remaining >= 1.0) & ~exhausted
    assert usable.sum() >= 40, "too few bins to be a real check"
    np.testing.assert_allclose(rebuilt[usable], reference[usable], rtol=1e-9)


def test_the_genealogy_summary_is_per_unit_sequence() -> None:
    """Both halves of it, so their ratio means one thing at every length.

    ``pair_coalescence_counts`` normalises by span on its own.  If the branch
    length beside it were left as a total it would scale with the sequence while
    the counts did not, and anyone forming the natural ratio would carry a
    factor of the length that belongs to neither -- across a matrix spanning
    20 to 80 Mb that is a fourfold artefact.
    """

    import msprime

    from necast.simulator.sample import genealogy_summary

    edges, _ = time_grid(Config())
    summaries = {}
    for sequence_length in (5_000_000, 20_000_000, 80_000_000):
        ts = msprime.sim_ancestry(
            samples=32,
            sequence_length=sequence_length,
            recombination_rate=1.25e-8,
            population_size=12_000,
            ploidy=2,
            random_seed=13,
        )
        summaries[sequence_length] = genealogy_summary(ts, edges)

    reference_counts = summaries[5_000_000][0].sum()
    reference_exposure = summaries[5_000_000][1].sum()
    for sequence_length, (counts, exposure, _) in summaries.items():
        assert counts.sum() == pytest.approx(reference_counts, rel=0.01), sequence_length
        assert exposure.sum() == pytest.approx(reference_exposure, rel=0.05), sequence_length
