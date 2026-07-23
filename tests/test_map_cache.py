from __future__ import annotations

import bz2
from dataclasses import fields
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from simulator.config import Config
import simulator.empirical_maps as empirical_maps_module
import simulator.map_cache as map_cache_module
from simulator.empirical_maps import (
    PiecewiseRateMap,
    load_empirical_maps,
    load_empirical_rate_table,
)
from simulator.map_cache import (
    ARRAY_FIELDS,
    CACHE_ENV_VAR,
    CACHE_SCHEMA,
    build_cache_fingerprint,
    load_or_build_rate_maps,
    map_cache_directory,
)


def _write_interval_map(path: Path) -> None:
    path.write_text(
        "chr2 0 100 1e-8\n"
        "chr1 0 50 2e-8\n"
        "chr1 50 120 3e-8\n"
        "chr2 100 200 4e-8\n",
        encoding="utf-8",
    )


def _raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()


def _assert_maps_exact(
    expected: dict[str, PiecewiseRateMap],
    actual: dict[str, PiecewiseRateMap],
) -> None:
    assert list(actual) == list(expected)
    scalar_fields = [
        field.name
        for field in fields(PiecewiseRateMap)
        if field.name not in ARRAY_FIELDS
    ]
    for chrom in expected:
        expected_map = expected[chrom]
        actual_map = actual[chrom]
        for field in scalar_fields:
            expected_value = getattr(expected_map, field)
            actual_value = getattr(actual_map, field)
            assert type(actual_value) is type(expected_value)
            assert actual_value == expected_value
        for field in ARRAY_FIELDS:
            expected_array = np.asarray(getattr(expected_map, field))
            actual_array = np.asarray(getattr(actual_map, field))
            assert actual_array.dtype == expected_array.dtype
            assert actual_array.shape == expected_array.shape
            assert actual_array.nbytes == expected_array.nbytes
            assert _raw_sha256(actual_array) == _raw_sha256(expected_array)
            assert actual_array.tobytes(order="C") == expected_array.tobytes(order="C")


def _cache_entries(cache_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in cache_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _concurrent_cache_worker(
    source_text: str,
    cache_dir_text: str,
    counter_text: str,
    gate,
    result_queue,
) -> None:
    os.environ[CACHE_ENV_VAR] = cache_dir_text
    source = Path(source_text)
    counter = Path(counter_text)

    def builder(snapshot: Path) -> dict[str, PiecewiseRateMap]:
        descriptor = os.open(counter, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        finally:
            os.close(descriptor)
        time.sleep(0.25)
        return load_empirical_rate_table(
            str(snapshot),
            baseline_rate=1e-8,
            map_kind="recombination",
            rate_unit="per_bp",
            scale_to_baseline=False,
        )

    gate.wait(timeout=10)
    maps = load_or_build_rate_maps(
        source_path=source,
        parse_parameters={
            "parser": "concurrency-test",
            "baseline_rate": 1e-8,
            "map_kind": "recombination",
            "rate_unit": "per_bp",
            "scale_to_baseline": False,
        },
        related_code_paths={"test_parser": Path(__file__)},
        builder=builder,
        map_class=PiecewiseRateMap,
    )
    first = next(iter(maps.values()))
    result_queue.put(
        (
            type(first.position).__name__,
            first.position.dtype.str,
            first.position.shape,
            _raw_sha256(first.position),
        )
    )


def test_fingerprint_covers_source_parameters_schema_code_and_versions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "map.bed"
    parser_code = tmp_path / "parser.py"
    _write_interval_map(source)
    parser_code.write_text("PARSER_VERSION = 1\n", encoding="utf-8")
    parameters = {
        "baseline_rate": 1e-8,
        "map_kind": "recombination",
        "rate_unit": "per_bp",
        "scale_to_baseline": False,
    }

    fingerprint, payload = build_cache_fingerprint(
        source,
        parameters,
        {"parser": parser_code},
    )

    assert len(fingerprint) == 64
    assert payload["schema"] == CACHE_SCHEMA
    assert payload["source"] == {
        "name": source.name,
        "size": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    assert payload["parse_parameters"] == parameters
    implementation = payload["implementation"]
    assert implementation["numpy_version"] == np.__version__
    assert implementation["pandas_version"] == pd.__version__
    assert implementation["python_version"]
    assert implementation["platform"]
    assert implementation["machine"]
    assert implementation["byteorder"] in {"little", "big"}
    assert implementation["code_sha256"]["parser"] == hashlib.sha256(
        parser_code.read_bytes()
    ).hexdigest()
    assert "cache_schema_implementation" in implementation["code_sha256"]

    changed_parameters = {**parameters, "scale_to_baseline": True}
    changed_parameter_fingerprint, _ = build_cache_fingerprint(
        source,
        changed_parameters,
        {"parser": parser_code},
    )
    assert changed_parameter_fingerprint != fingerprint

    parser_code.write_text("PARSER_VERSION = 2\n", encoding="utf-8")
    changed_code_fingerprint, _ = build_cache_fingerprint(
        source,
        parameters,
        {"parser": parser_code},
    )
    assert changed_code_fingerprint != fingerprint

    source.write_text(source.read_text(encoding="utf-8") + "chr3 0 10 5e-8\n", encoding="utf-8")
    changed_source_fingerprint, _ = build_cache_fingerprint(
        source,
        parameters,
        {"parser": parser_code},
    )
    assert changed_source_fingerprint not in {fingerprint, changed_code_fingerprint}


def test_cached_load_is_mmap_and_strictly_matches_uncached_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed"
    cache_dir = tmp_path / "cache"
    _write_interval_map(source)
    monkeypatch.setenv(CACHE_ENV_VAR, str(cache_dir))
    cfg = Config(
        recomb_map=str(source),
        mut_map="",
        baseline_rec=1e-8,
        recomb_map_unit="per_bp",
    )
    direct = load_empirical_rate_table(
        str(source),
        baseline_rate=cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )

    first = load_empirical_maps(cfg)
    _assert_maps_exact(direct, first.rec_maps)
    assert all(
        isinstance(getattr(rate_map, field), np.memmap)
        and not getattr(rate_map, field).flags.writeable
        for rate_map in first.rec_maps.values()
        for field in ARRAY_FIELDS
    )
    assert first.metadata["recombination"] == empirical_maps_module._summarize_maps(
        str(source),
        direct,
        cfg.recomb_map_unit,
    )
    assert map_cache_directory() == cache_dir.resolve()

    entries = _cache_entries(cache_dir)
    assert len(entries) == 1
    manifest = json.loads((entries[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == CACHE_SCHEMA
    assert manifest["fingerprint"] == entries[0].name
    assert len(manifest["maps"]) == len(direct)
    fingerprint_payload = manifest["fingerprint_payload"]
    assert fingerprint_payload["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert fingerprint_payload["parse_parameters"] == {
        "parser": "load_empirical_rate_table",
        "compression": "none",
        "baseline_rate": cfg.baseline_rec,
        "map_kind": "recombination",
        "rate_unit": cfg.recomb_map_unit,
        "scale_to_baseline": False,
    }
    assert set(fingerprint_payload["implementation"]["code_sha256"]) == {
        "cache_schema_implementation",
        "empirical_maps_parser",
        "rate_validation_config",
    }
    assert not list(cache_dir.glob(".*.tmp-*"))
    assert not list(cache_dir.glob(".*.source-*"))

    def cache_miss_is_a_failure(*_args, **_kwargs):
        raise AssertionError("a valid cache hit must not invoke the text parser")

    monkeypatch.setattr(
        empirical_maps_module,
        "load_empirical_rate_table",
        cache_miss_is_a_failure,
    )
    second = load_empirical_maps(cfg)
    _assert_maps_exact(direct, second.rec_maps)


def test_corrupt_array_or_scalar_metadata_is_safely_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed"
    cache_dir = tmp_path / "cache"
    _write_interval_map(source)
    monkeypatch.setenv(CACHE_ENV_VAR, str(cache_dir))
    cfg = Config(recomb_map=str(source), mut_map="", recomb_map_unit="per_bp")
    direct = load_empirical_rate_table(
        str(source),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )
    load_empirical_maps(cfg)
    entry = _cache_entries(cache_dir)[0]
    manifest_path = entry / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    array_path = entry / manifest["maps"][0]["arrays"]["position"]["file"]
    array_path.write_bytes(b"not a valid npy")

    original_parser = empirical_maps_module.load_empirical_rate_table
    parser_calls = 0

    def counted_parser(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(
        empirical_maps_module,
        "load_empirical_rate_table",
        counted_parser,
    )
    rebuilt_from_array_damage = load_empirical_maps(cfg)
    assert parser_calls == 1
    _assert_maps_exact(direct, rebuilt_from_array_damage.rec_maps)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["maps"][0]["scalars"]["source_format"] = "corrupt-but-valid-json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rebuilt_from_metadata_damage = load_empirical_maps(cfg)
    assert parser_calls == 2
    _assert_maps_exact(direct, rebuilt_from_metadata_damage.rec_maps)
    repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert repaired_manifest["maps"][0]["scalars"]["source_format"] == "bed_intervals"

    repaired_manifest["maps"][0]["scalars"]["mean_rate_before_scaling"] = float("nan")
    manifest_path.write_text(json.dumps(repaired_manifest), encoding="utf-8")

    rebuilt_from_nonfinite_json = load_empirical_maps(cfg)
    assert parser_calls == 3
    _assert_maps_exact(direct, rebuilt_from_nonfinite_json.rec_maps)
    repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert np.isfinite(
        repaired_manifest["maps"][0]["scalars"]["mean_rate_before_scaling"]
    )
    assert not list(cache_dir.glob(".*.previous-*"))
    assert not list(cache_dir.glob(".*.tmp-*"))


def test_cache_hit_rechecks_source_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed"
    cache_dir = tmp_path / "cache"
    _write_interval_map(source)
    monkeypatch.setenv(CACHE_ENV_VAR, str(cache_dir))
    cfg = Config(recomb_map=str(source), mut_map="")
    initial = load_empirical_maps(cfg)
    initial_rate_sha256 = _raw_sha256(initial.rec_maps["chr1"].rate)

    original_load_entry = map_cache_module._load_entry
    source_changed = False

    def load_entry_after_source_change(*args, **kwargs):
        nonlocal source_changed
        if not source_changed:
            source_changed = True
            source.write_text(
                source.read_text(encoding="utf-8").replace(
                    "chr1 50 120 3e-8",
                    "chr1 50 120 9e-8",
                ),
                encoding="utf-8",
            )
        return original_load_entry(*args, **kwargs)

    original_parser = empirical_maps_module.load_empirical_rate_table
    parser_calls = 0

    def counted_parser(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(map_cache_module, "_load_entry", load_entry_after_source_change)
    monkeypatch.setattr(
        empirical_maps_module,
        "load_empirical_rate_table",
        counted_parser,
    )

    actual = load_empirical_maps(cfg)
    expected = original_parser(
        str(source),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )

    assert source_changed
    assert parser_calls == 1
    _assert_maps_exact(expected, actual.rec_maps)
    assert _raw_sha256(actual.rec_maps["chr1"].rate) != initial_rate_sha256
    assert len(_cache_entries(cache_dir)) == 2


def test_cache_build_rechecks_source_after_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed"
    cache_dir = tmp_path / "cache"
    _write_interval_map(source)
    monkeypatch.setenv(CACHE_ENV_VAR, str(cache_dir))
    cfg = Config(recomb_map=str(source), mut_map="")

    original_publish = map_cache_module._build_and_publish
    source_changed = False

    def publish_then_change_source(*args, **kwargs):
        nonlocal source_changed
        original_publish(*args, **kwargs)
        if not source_changed:
            source_changed = True
            source.write_text(
                source.read_text(encoding="utf-8").replace(
                    "chr1 50 120 3e-8",
                    "chr1 50 120 8e-8",
                ),
                encoding="utf-8",
            )

    original_parser = empirical_maps_module.load_empirical_rate_table
    parser_calls = 0

    def counted_parser(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(
        map_cache_module,
        "_build_and_publish",
        publish_then_change_source,
    )
    monkeypatch.setattr(
        empirical_maps_module,
        "load_empirical_rate_table",
        counted_parser,
    )

    actual = load_empirical_maps(cfg)
    expected = original_parser(
        str(source),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )

    assert source_changed
    assert parser_calls == 2
    _assert_maps_exact(expected, actual.rec_maps)
    assert len(_cache_entries(cache_dir)) == 2


def test_unavailable_cache_falls_back_without_swallowing_parser_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed"
    blocked_cache_path = tmp_path / "cache-is-a-file"
    _write_interval_map(source)
    blocked_cache_path.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setenv(CACHE_ENV_VAR, str(blocked_cache_path))
    cfg = Config(recomb_map=str(source), mut_map="")
    direct = load_empirical_rate_table(
        str(source),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )

    with pytest.warns(RuntimeWarning, match="falling back to direct text parsing"):
        fallback = load_empirical_maps(cfg)

    _assert_maps_exact(direct, fallback.rec_maps)
    assert all(
        not isinstance(getattr(rate_map, field), np.memmap)
        for rate_map in fallback.rec_maps.values()
        for field in ARRAY_FIELDS
    )
    assert blocked_cache_path.is_file()

    def cache_root_resolution_failure() -> Path:
        raise OSError("cache root resolution failure")

    monkeypatch.setattr(
        map_cache_module,
        "map_cache_directory",
        cache_root_resolution_failure,
    )
    with pytest.warns(RuntimeWarning, match="falling back to direct text parsing"):
        resolution_fallback = load_empirical_maps(cfg)
    _assert_maps_exact(direct, resolution_fallback.rec_maps)

    def scientific_parser_failure(*_args, **_kwargs):
        raise ValueError("scientific parser failure")

    monkeypatch.setattr(
        empirical_maps_module,
        "load_empirical_rate_table",
        scientific_parser_failure,
    )
    with pytest.warns(RuntimeWarning, match="falling back to direct text parsing"):
        with pytest.raises(ValueError, match="scientific parser failure"):
            load_empirical_maps(cfg)


def test_crash_residue_is_cleaned_and_missing_entry_is_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed"
    cache_dir = tmp_path / "cache"
    _write_interval_map(source)
    monkeypatch.setenv(CACHE_ENV_VAR, str(cache_dir))
    cfg = Config(recomb_map=str(source), mut_map="")
    direct = load_empirical_rate_table(
        str(source),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )
    load_empirical_maps(cfg)
    entry = _cache_entries(cache_dir)[0]
    fingerprint = entry.name
    previous = cache_dir / f".{fingerprint}.previous-crashed"
    staging = cache_dir / f".{fingerprint}.tmp-crashed"
    snapshot = cache_dir / f".{fingerprint}.source-crashed"
    entry.replace(previous)
    staging.mkdir()
    (staging / "partial.npy").write_bytes(b"partial")
    snapshot.write_bytes(b"partial source snapshot")

    original_parser = empirical_maps_module.load_empirical_rate_table
    parser_calls = 0

    def counted_parser(*args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(
        empirical_maps_module,
        "load_empirical_rate_table",
        counted_parser,
    )
    rebuilt = load_empirical_maps(cfg)

    assert parser_calls == 1
    _assert_maps_exact(direct, rebuilt.rec_maps)
    assert (cache_dir / fingerprint).is_dir()
    assert not previous.exists()
    assert not staging.exists()
    assert not snapshot.exists()


def test_source_content_change_uses_a_new_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed"
    cache_dir = tmp_path / "cache"
    _write_interval_map(source)
    monkeypatch.setenv(CACHE_ENV_VAR, str(cache_dir))
    cfg = Config(recomb_map=str(source), mut_map="")

    before = load_empirical_maps(cfg)
    before_sha = _raw_sha256(before.rec_maps["chr1"].rate)
    source.write_text(
        source.read_text(encoding="utf-8").replace("chr1 50 120 3e-8", "chr1 50 120 7e-8"),
        encoding="utf-8",
    )
    direct_after = load_empirical_rate_table(
        str(source),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )
    after = load_empirical_maps(cfg)

    _assert_maps_exact(direct_after, after.rec_maps)
    assert _raw_sha256(after.rec_maps["chr1"].rate) != before_sha
    assert len(_cache_entries(cache_dir)) == 2


def test_cache_preserves_symlink_filename_compression_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressed_target = tmp_path / "compressed-map-data.bin"
    with gzip.open(compressed_target, "wt", encoding="utf-8") as handle:
        handle.write("chr1 0 100 1e-8\nchr1 100 200 2e-8\n")
    source_alias = tmp_path / "map.bed.gz"
    try:
        source_alias.symlink_to(compressed_target)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "cache"))
    cfg = Config(recomb_map=str(source_alias), mut_map="")
    direct = load_empirical_rate_table(
        str(source_alias),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )

    cached = load_empirical_maps(cfg)

    _assert_maps_exact(direct, cached.rec_maps)
    entry = _cache_entries(tmp_path / "cache")[0]
    manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint_payload"]["parse_parameters"]["compression"] == "gzip"


def test_cache_preserves_plain_alias_to_misleading_gzip_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain_target = tmp_path / "plain-text-map.gz"
    plain_target.write_text(
        "chr1 0 100 1e-8\nchr1 100 200 2e-8\n",
        encoding="utf-8",
    )
    source_alias = tmp_path / "map.bed"
    try:
        source_alias.symlink_to(plain_target)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "cache"))
    cfg = Config(recomb_map=str(source_alias), mut_map="")
    direct = load_empirical_rate_table(
        str(source_alias),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )

    cached = load_empirical_maps(cfg)

    _assert_maps_exact(direct, cached.rec_maps)
    entry = _cache_entries(tmp_path / "cache")[0]
    manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint_payload"]["parse_parameters"]["compression"] == "none"


def test_cache_preserves_legacy_non_gzip_codec_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed.bz2"
    with bz2.open(source, "wt", encoding="utf-8") as handle:
        handle.write("chr1 0 100 1e-8\nchr1 100 200 2e-8\n")

    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "cache"))
    cfg = Config(recomb_map=str(source), mut_map="")
    direct = load_empirical_rate_table(
        str(source),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )

    cached = load_empirical_maps(cfg)

    _assert_maps_exact(direct, cached.rec_maps)
    entry = _cache_entries(tmp_path / "cache")[0]
    manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint_payload"]["parse_parameters"]["compression"] == "bz2"


def test_symlink_retarget_during_build_cannot_poison_resolved_source_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_a = tmp_path / "target-a.bed"
    target_b = tmp_path / "target-b.bed"
    source_alias = tmp_path / "map-alias.bed"
    target_a.write_text("chr1 0 100 1e-8\n", encoding="utf-8")
    target_b.write_text("chr1 0 100 9e-8\n", encoding="utf-8")
    try:
        source_alias.symlink_to(target_a)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "cache"))
    direct_a = load_empirical_rate_table(
        str(target_a),
        baseline_rate=1e-8,
        map_kind="recombination",
        rate_unit="per_bp",
        scale_to_baseline=False,
    )
    direct_b = load_empirical_rate_table(
        str(target_b),
        baseline_rate=1e-8,
        map_kind="recombination",
        rate_unit="per_bp",
        scale_to_baseline=False,
    )
    original_load_entry = map_cache_module._load_entry
    retargeted = False

    def load_entry_after_retarget(*args, **kwargs):
        nonlocal retargeted
        if not retargeted:
            retargeted = True
            source_alias.unlink()
            source_alias.symlink_to(target_b)
        return original_load_entry(*args, **kwargs)

    monkeypatch.setattr(
        map_cache_module,
        "_load_entry",
        load_entry_after_retarget,
    )
    through_alias = load_empirical_maps(
        Config(recomb_map=str(source_alias), mut_map="")
    )
    later_direct_target_hit = load_empirical_maps(
        Config(recomb_map=str(target_a), mut_map="")
    )

    assert retargeted
    _assert_maps_exact(direct_a, through_alias.rec_maps)
    _assert_maps_exact(direct_a, later_direct_target_hit.rec_maps)
    assert _raw_sha256(
        later_direct_target_hit.rec_maps["chr1"].rate
    ) != _raw_sha256(direct_b["chr1"].rate)


def test_builder_parses_verified_snapshot_during_aba_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "map.bed"
    source_a = "chr1 0 100 1e-8\n"
    source_b = "chr1 0 100 9e-8\n"
    source.write_text(source_a, encoding="utf-8")
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path / "cache"))
    cfg = Config(recomb_map=str(source), mut_map="")
    direct_a = load_empirical_rate_table(
        str(source),
        cfg.baseline_rec,
        map_kind="recombination",
        rate_unit=cfg.recomb_map_unit,
        scale_to_baseline=False,
    )
    original_parser = empirical_maps_module.load_empirical_rate_table
    parser_calls = 0

    def parse_while_original_source_changes(path, *args, **kwargs):
        nonlocal parser_calls
        parser_calls += 1
        source.write_text(source_b, encoding="utf-8")
        try:
            return original_parser(path, *args, **kwargs)
        finally:
            source.write_text(source_a, encoding="utf-8")

    monkeypatch.setattr(
        empirical_maps_module,
        "load_empirical_rate_table",
        parse_while_original_source_changes,
    )
    first = load_empirical_maps(cfg)
    second = load_empirical_maps(cfg)

    assert parser_calls == 1
    _assert_maps_exact(direct_a, first.rec_maps)
    _assert_maps_exact(direct_a, second.rec_maps)


def test_concurrent_processes_build_one_atomic_entry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "map.bed"
    cache_dir = tmp_path / "cache"
    counter = tmp_path / "builder-calls.txt"
    _write_interval_map(source)
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_cache_worker,
            args=(
                str(source),
                str(cache_dir),
                str(counter),
                gate,
                result_queue,
            ),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    gate.set()
    for process in processes:
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert [process.exitcode for process in processes] == [0, 0]
    results = [result_queue.get(timeout=2) for _ in processes]
    assert results[0] == results[1]
    assert results[0][0] == "memmap"
    assert len(counter.read_text(encoding="utf-8").splitlines()) == 1
    assert len(_cache_entries(cache_dir)) == 1
    assert not list(cache_dir.glob(".*.tmp-*"))
    assert not list(cache_dir.glob(".*.previous-*"))
    assert not list(cache_dir.glob(".*.source-*"))
