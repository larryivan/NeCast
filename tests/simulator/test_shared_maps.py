from __future__ import annotations

import dataclasses

import numpy as np

from necast.simulator.config import Config
from necast.simulator.empirical_maps import EmpiricalMapStore, PiecewiseRateMap
from necast.simulator.sample import _close_worker_shared_maps, _get_worker_empirical, init_worker
from necast.simulator.shared_maps import (
    attach_shared_empirical_map_store,
    create_shared_empirical_map_store,
    release_shared_memory,
)


def _tiny_store() -> EmpiricalMapStore:
    rate_map = PiecewiseRateMap(
        chrom="chr1",
        position=np.array([0.0, 10.0, 25.0], dtype=np.float64),
        rate=np.array([1.0e-8, 2.0e-8], dtype=np.float64),
        scaling_method="identity",
        source_format="bed_intervals",
    )
    return EmpiricalMapStore(
        rec_maps={"chr1": rate_map},
        mut_maps={"chr1": rate_map},
        metadata={"fixture": {"name": "tiny"}},
    )


def test_shared_store_attaches_read_only_views() -> None:
    source = _tiny_store()
    assert source.is_available(10)
    spec, owner_handles = create_shared_empirical_map_store(source)
    assert len(owner_handles) == 1
    assert spec.memory_name == owner_handles[0].name
    attached = None
    attached_handles = []
    try:
        attached, attached_handles = attach_shared_empirical_map_store(spec)
        np.testing.assert_array_equal(attached.rec_maps["chr1"].position, [0.0, 10.0, 25.0])
        np.testing.assert_array_equal(attached.mut_maps["chr1"].rate, [1.0e-8, 2.0e-8])
        assert not attached.rec_maps["chr1"].position.flags.writeable
        assert not attached.mut_maps["chr1"].rate.flags.writeable
        assert attached.metadata == {"fixture": {"name": "tiny"}}
        assert attached.common_chroms(10) == ["chr1"]
        assert attached._common_span_cache == source._common_span_cache
        assert attached._usable_chrom_cache == source._usable_chrom_cache
    finally:
        attached = None
        release_shared_memory(attached_handles)
        release_shared_memory(owner_handles, unlink=True)


def test_worker_uses_shared_store_without_reloading_files() -> None:
    cfg = Config(
        recomb_map="this-file-must-not-be-read.bed",
        mut_map="this-file-must-not-be-read.bed",
    )
    spec, owner_handles = create_shared_empirical_map_store(_tiny_store())
    try:
        init_worker(dataclasses.asdict(cfg), empirical_shared_spec=spec)
        attached = _get_worker_empirical(cfg)
        assert attached.is_available(10)
        assert attached.metadata["fixture"]["name"] == "tiny"
    finally:
        _close_worker_shared_maps()
        release_shared_memory(owner_handles, unlink=True)
