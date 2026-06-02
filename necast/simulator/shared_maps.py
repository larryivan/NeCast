from __future__ import annotations

from dataclasses import dataclass, fields
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from .empirical_maps import EmpiricalMapStore, PiecewiseRateMap


@dataclass(frozen=True)
class SharedArraySpec:
    offset: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class SharedPiecewiseRateMapSpec:
    attributes: dict[str, Any]
    position: SharedArraySpec
    rate: SharedArraySpec


@dataclass(frozen=True)
class SharedEmpiricalMapStoreSpec:
    memory_name: str | None
    memory_size: int
    rec_maps: dict[str, SharedPiecewiseRateMapSpec]
    mut_maps: dict[str, SharedPiecewiseRateMapSpec]
    metadata: dict[str, dict]
    common_span_cache: dict[str, tuple[float, float]]
    usable_chrom_cache: dict[float, tuple[str, ...]]


def _piecewise_attributes(rate_map: PiecewiseRateMap) -> dict[str, Any]:
    return {
        field.name: getattr(rate_map, field.name)
        for field in fields(PiecewiseRateMap)
        if field.name not in {"position", "rate"}
    }


def _layout_array(array: np.ndarray, offset: int) -> tuple[SharedArraySpec, int]:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("empirical map arrays must not use object dtype")
    alignment = max(1, int(value.dtype.alignment))
    aligned_offset = ((offset + alignment - 1) // alignment) * alignment
    spec = SharedArraySpec(
        offset=aligned_offset,
        shape=tuple(int(size) for size in value.shape),
        dtype=value.dtype.str,
    )
    return spec, aligned_offset + int(value.nbytes)


def _layout_maps(
    maps: dict[str, PiecewiseRateMap],
    offset: int,
) -> tuple[dict[str, SharedPiecewiseRateMapSpec], int]:
    out: dict[str, SharedPiecewiseRateMapSpec] = {}
    for chrom, rate_map in maps.items():
        position_spec, offset = _layout_array(rate_map.position, offset)
        rate_spec, offset = _layout_array(rate_map.rate, offset)
        out[chrom] = SharedPiecewiseRateMapSpec(
            attributes=_piecewise_attributes(rate_map),
            position=position_spec,
            rate=rate_spec,
        )
    return out, offset


def _copy_array(
    array: np.ndarray,
    spec: SharedArraySpec,
    shm: shared_memory.SharedMemory,
) -> None:
    source = np.ascontiguousarray(array)
    view = np.ndarray(
        spec.shape,
        dtype=np.dtype(spec.dtype),
        buffer=shm.buf,
        offset=spec.offset,
    )
    view[...] = source


def _copy_maps(
    maps: dict[str, PiecewiseRateMap],
    specs: dict[str, SharedPiecewiseRateMapSpec],
    shm: shared_memory.SharedMemory,
) -> None:
    for chrom, rate_map in maps.items():
        spec = specs[chrom]
        _copy_array(rate_map.position, spec.position, shm)
        _copy_array(rate_map.rate, spec.rate, shm)


def create_shared_empirical_map_store(
    store: EmpiricalMapStore,
) -> tuple[SharedEmpiricalMapStoreSpec, list[shared_memory.SharedMemory]]:
    """Copy all empirical arrays into one shared-memory segment."""

    rec_specs, offset = _layout_maps(store.rec_maps, 0)
    mut_specs, memory_size = _layout_maps(store.mut_maps, offset)
    if memory_size == 0:
        return (
            SharedEmpiricalMapStoreSpec(
                memory_name=None,
                memory_size=0,
                rec_maps=rec_specs,
                mut_maps=mut_specs,
                metadata=store.metadata,
                common_span_cache=dict(store._common_span_cache),
                usable_chrom_cache=dict(store._usable_chrom_cache),
            ),
            [],
        )

    shm = shared_memory.SharedMemory(create=True, size=memory_size)
    try:
        _copy_maps(store.rec_maps, rec_specs, shm)
        _copy_maps(store.mut_maps, mut_specs, shm)
        spec = SharedEmpiricalMapStoreSpec(
            memory_name=shm.name,
            memory_size=memory_size,
            rec_maps=rec_specs,
            mut_maps=mut_specs,
            metadata=store.metadata,
            common_span_cache=dict(store._common_span_cache),
            usable_chrom_cache=dict(store._usable_chrom_cache),
        )
    except Exception:
        shm.close()
        shm.unlink()
        raise
    return spec, [shm]


def _attach_array(
    spec: SharedArraySpec,
    shm: shared_memory.SharedMemory,
) -> np.ndarray:
    array = np.ndarray(
        spec.shape,
        dtype=np.dtype(spec.dtype),
        buffer=shm.buf,
        offset=spec.offset,
    )
    array.flags.writeable = False
    return array


def _attach_piecewise_map(
    spec: SharedPiecewiseRateMapSpec,
    shm: shared_memory.SharedMemory,
) -> PiecewiseRateMap:
    return PiecewiseRateMap(
        position=_attach_array(spec.position, shm),
        rate=_attach_array(spec.rate, shm),
        **spec.attributes,
    )


def attach_shared_empirical_map_store(
    spec: SharedEmpiricalMapStoreSpec,
) -> tuple[EmpiricalMapStore, list[shared_memory.SharedMemory]]:
    """Attach read-only NumPy views to a shared empirical map store."""

    if spec.memory_name is None:
        if spec.memory_size != 0 or spec.rec_maps or spec.mut_maps:
            raise ValueError("shared empirical map specification is missing its memory segment")
        return (
            EmpiricalMapStore(
                metadata=spec.metadata,
                _common_span_cache=dict(spec.common_span_cache),
                _usable_chrom_cache=dict(spec.usable_chrom_cache),
            ),
            [],
        )

    shm = shared_memory.SharedMemory(name=spec.memory_name)
    try:
        store = EmpiricalMapStore(
            rec_maps={
                chrom: _attach_piecewise_map(map_spec, shm)
                for chrom, map_spec in spec.rec_maps.items()
            },
            mut_maps={
                chrom: _attach_piecewise_map(map_spec, shm)
                for chrom, map_spec in spec.mut_maps.items()
            },
            metadata=spec.metadata,
            _common_span_cache=dict(spec.common_span_cache),
            _usable_chrom_cache=dict(spec.usable_chrom_cache),
        )
    except Exception:
        shm.close()
        raise
    return store, [shm]


def release_shared_memory(
    handles: list[shared_memory.SharedMemory],
    *,
    unlink: bool = False,
) -> None:
    for shm in handles:
        try:
            shm.close()
        except FileNotFoundError:
            pass
        if unlink:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
