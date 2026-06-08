from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, TypeVar
import uuid
import warnings

import numpy as np
import pandas as pd


CACHE_SCHEMA = "necast.empirical-map-cache.v1"
CACHE_ENV_VAR = "NECAST_CACHE_DIR"
MANIFEST_NAME = "manifest.json"
# ``measured_runs`` is a (n, 2) array of the spans a source file actually
# measured.  It rides with the rate arrays rather than the scalars because it is
# per-chromosome and variable length, and a cache that dropped it would hand back
# a map that claims its filled gaps were observed.
ARRAY_FIELDS = ("position", "rate", "measured_runs")
SCALAR_FIELDS = (
    "chrom",
    "rate_unit",
    "rate_unit_before",
    "rate_unit_after",
    "mean_rate_before_scaling",
    "mean_rate_after_scaling",
    "scaling_method",
    "source_format",
)
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_INPUT_CHANGE_RETRIES = 3


RateMapT = TypeVar("RateMapT")


class MapCacheError(RuntimeError):
    """Base class for empirical-map cache failures."""


class MapCacheValidationError(MapCacheError):
    """Raised internally when an existing cache entry is incomplete or corrupt."""


class MapCacheStorageError(MapCacheError):
    """Raised internally when the optional cache storage cannot be used."""


class _CacheInputsChanged(MapCacheError):
    """Raised when a source or parser input changes while an entry is built."""


def map_cache_directory() -> Path:
    """Return the empirical-map cache root, honoring ``NECAST_CACHE_DIR``."""
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve(strict=False)

    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data).expanduser() if local_app_data else Path.home() / "AppData" / "Local"
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache"
    return (base / "necast" / "empirical-maps").resolve(strict=False)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_compatible(value: object) -> Any:
    """Cache-key values, sorted so that equal keys encode identically.

    ``dataset_state`` has a function of the same name and a different
    contract; see the note there.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("cache fingerprint values must be finite")
        return value
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(f"Unsupported cache fingerprint value: {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _source_descriptor(path: Path) -> dict[str, object]:
    before = path.stat()
    sha256 = _sha256_file(path)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise _CacheInputsChanged(f"Empirical map changed while hashing: {path}")
    return {
        "name": path.name,
        "size": after.st_size,
        "sha256": sha256,
    }


def build_cache_fingerprint(
    source_path: str | Path,
    parse_parameters: Mapping[str, object],
    related_code_paths: Mapping[str, str | Path],
) -> tuple[str, dict[str, object]]:
    """Build a content-addressed fingerprint for one parsed empirical map."""
    source = Path(source_path).expanduser().resolve(strict=True)
    code_paths = {
        **{
            str(name): Path(path).expanduser().resolve(strict=True)
            for name, path in related_code_paths.items()
        },
        "cache_schema_implementation": Path(__file__).resolve(),
    }
    payload: dict[str, object] = {
        "schema": CACHE_SCHEMA,
        "source": _source_descriptor(source),
        "parse_parameters": _json_compatible(parse_parameters),
        "implementation": {
            "code_sha256": {
                name: _sha256_file(path)
                for name, path in sorted(code_paths.items())
            },
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
    }
    fingerprint = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return fingerprint, payload


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.asarray(array)
    if not contiguous.flags.c_contiguous:
        contiguous = np.ascontiguousarray(contiguous)
    view = memoryview(contiguous).cast("B")
    digest = hashlib.sha256()
    for offset in range(0, len(view), _HASH_CHUNK_BYTES):
        digest.update(view[offset : offset + _HASH_CHUNK_BYTES])
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = (_canonical_json(value) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


@contextmanager
def _exclusive_cache_lock(path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
    except OSError as exc:
        raise MapCacheStorageError(f"Cannot open empirical-map cache lock: {path}") from exc
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EDEADLK}:
                        raise
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError as exc:
        try:
            handle.close()
        except OSError:
            pass
        raise MapCacheStorageError(f"Cannot lock empirical-map cache: {path}") from exc

    try:
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass


def _manifest_content_sha256(manifest: Mapping[str, object]) -> str:
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    try:
        encoded = _canonical_json(content).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise MapCacheValidationError("Cache manifest contains invalid JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reject_nonstandard_json_constant(value: str):
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed")


def _serialize_maps(
    staging: Path,
    maps: Mapping[str, RateMapT],
    fingerprint: str,
    fingerprint_payload: Mapping[str, object],
) -> None:
    records: list[dict[str, object]] = []
    for index, (chrom, rate_map) in enumerate(maps.items()):
        scalars = {field: _json_compatible(getattr(rate_map, field)) for field in SCALAR_FIELDS}
        if scalars["chrom"] != chrom:
            raise ValueError(
                f"Empirical map dictionary key {chrom!r} does not match map chromosome "
                f"{scalars['chrom']!r}"
            )

        arrays: dict[str, dict[str, object]] = {}
        for field in ARRAY_FIELDS:
            array = np.asarray(getattr(rate_map, field))
            if array.dtype.hasobject:
                raise TypeError("empirical map cache does not support object arrays")
            array = np.ascontiguousarray(array)
            filename = f"map-{index:05d}-{field}.npy"
            _atomic_write_npy(staging / filename, array)
            arrays[field] = {
                "file": filename,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": array.nbytes,
                "sha256": _array_sha256(array),
            }
        records.append(
            {
                "key": chrom,
                "scalars": scalars,
                "arrays": arrays,
            }
        )

    manifest: dict[str, object] = {
        "schema": CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "maps": records,
    }
    manifest["content_sha256"] = _manifest_content_sha256(manifest)
    _atomic_write_json(staging / MANIFEST_NAME, manifest)
    _fsync_directory(staging)


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise MapCacheValidationError(f"Cache manifest is unexpectedly large: {path}")
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except MapCacheValidationError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise MapCacheValidationError(f"Cannot read cache manifest: {path}") from exc
    if not isinstance(value, dict):
        raise MapCacheValidationError(f"Cache manifest is not an object: {path}")
    return value


def _validated_array(
    entry: Path,
    descriptor: object,
    expected_filename: str,
) -> np.memmap:
    if not isinstance(descriptor, dict):
        raise MapCacheValidationError("Cache array descriptor is not an object")
    expected_keys = {"file", "dtype", "shape", "nbytes", "sha256"}
    if set(descriptor) != expected_keys:
        raise MapCacheValidationError("Cache array descriptor has an unexpected schema")

    filename = descriptor["file"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename != expected_filename
    ):
        raise MapCacheValidationError("Cache array filename is unsafe")
    array_path = entry / filename
    if array_path.is_symlink():
        raise MapCacheValidationError(f"Cache array must not be a symlink: {array_path}")
    try:
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise MapCacheValidationError(f"Cannot load cache array: {array_path}") from exc
    if not isinstance(array, np.memmap):
        raise MapCacheValidationError(f"Cache array did not load as a memory map: {array_path}")

    shape = descriptor["shape"]
    if (
        not isinstance(descriptor["dtype"], str)
        or not isinstance(shape, list)
        or not all(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0
            for size in shape
        )
        or not isinstance(descriptor["nbytes"], int)
        or isinstance(descriptor["nbytes"], bool)
        or not isinstance(descriptor["sha256"], str)
        or len(descriptor["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in descriptor["sha256"])
    ):
        raise MapCacheValidationError(f"Cache array metadata is invalid: {array_path}")
    if array.dtype.str != descriptor["dtype"]:
        raise MapCacheValidationError(f"Cache array dtype mismatch: {array_path}")
    if list(array.shape) != shape:
        raise MapCacheValidationError(f"Cache array shape mismatch: {array_path}")
    if array.nbytes != descriptor["nbytes"]:
        raise MapCacheValidationError(f"Cache array byte count mismatch: {array_path}")
    if _array_sha256(array) != descriptor["sha256"]:
        raise MapCacheValidationError(f"Cache array hash mismatch: {array_path}")
    if array.flags.writeable:
        raise MapCacheValidationError(f"Cache array unexpectedly loaded writeable: {array_path}")
    return array


def _validated_scalars(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(SCALAR_FIELDS):
        raise MapCacheValidationError("Cache scalar metadata has an unexpected schema")
    for field in (
        "chrom",
        "rate_unit",
        "rate_unit_before",
        "rate_unit_after",
        "scaling_method",
        "source_format",
    ):
        if not isinstance(value[field], str):
            raise MapCacheValidationError(f"Cache scalar metadata field {field!r} is invalid")
    for field in ("mean_rate_before_scaling", "mean_rate_after_scaling"):
        scalar = value[field]
        if not isinstance(scalar, float) or not np.isfinite(scalar):
            raise MapCacheValidationError(f"Cache scalar metadata field {field!r} is invalid")
    return dict(value)


def _load_entry(
    entry: Path,
    expected_fingerprint: str,
    expected_payload: Mapping[str, object],
    map_class: type[RateMapT],
) -> dict[str, RateMapT]:
    if entry.is_symlink() or not entry.is_dir():
        raise MapCacheValidationError(f"Cache entry is missing or unsafe: {entry}")
    manifest = _load_manifest(entry / MANIFEST_NAME)
    expected_manifest_keys = {
        "schema",
        "fingerprint",
        "fingerprint_payload",
        "maps",
        "content_sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise MapCacheValidationError("Cache manifest has an unexpected schema")
    if manifest["schema"] != CACHE_SCHEMA:
        raise MapCacheValidationError("Cache schema does not match")
    if manifest["fingerprint"] != expected_fingerprint:
        raise MapCacheValidationError("Cache fingerprint does not match")
    if manifest["fingerprint_payload"] != expected_payload:
        raise MapCacheValidationError("Cache fingerprint payload does not match")
    if manifest["content_sha256"] != _manifest_content_sha256(manifest):
        raise MapCacheValidationError("Cache manifest content hash does not match")

    records = manifest["maps"]
    if not isinstance(records, list) or not records:
        raise MapCacheValidationError("Cache manifest has no empirical maps")
    maps: dict[str, RateMapT] = {}
    expected_array_files: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"key", "scalars", "arrays"}:
            raise MapCacheValidationError("Cached map record has an unexpected schema")
        key = record["key"]
        if not isinstance(key, str) or key in maps:
            raise MapCacheValidationError("Cached map key is invalid or duplicated")
        scalars = _validated_scalars(record["scalars"])
        if scalars["chrom"] != key:
            raise MapCacheValidationError("Cached map key and chromosome do not match")
        arrays = record["arrays"]
        if not isinstance(arrays, dict) or set(arrays) != set(ARRAY_FIELDS):
            raise MapCacheValidationError("Cached map arrays have an unexpected schema")
        loaded_arrays: dict[str, np.memmap] = {}
        for field in ARRAY_FIELDS:
            expected_filename = f"map-{index:05d}-{field}.npy"
            loaded_arrays[field] = _validated_array(
                entry,
                arrays[field],
                expected_filename,
            )
            expected_array_files.add(expected_filename)
        try:
            maps[key] = map_class(**scalars, **loaded_arrays)
        except (TypeError, ValueError) as exc:
            raise MapCacheValidationError(f"Cannot restore cached empirical map {key!r}") from exc
    actual_array_files = {path.name for path in entry.glob("*.npy") if path.is_file()}
    if actual_array_files != expected_array_files:
        raise MapCacheValidationError("Cache entry contains missing or unexpected array files")
    return maps


def _publish_entry(staging: Path, entry: Path) -> None:
    previous: Path | None = None
    if os.path.lexists(entry):
        previous = entry.with_name(f".{entry.name}.previous-{uuid.uuid4().hex}")
        os.replace(entry, previous)
    try:
        os.replace(staging, entry)
        _fsync_directory(entry.parent)
    except BaseException:
        if previous is not None and not os.path.lexists(entry):
            os.replace(previous, entry)
        raise
    else:
        if previous is not None:
            try:
                _remove_path(previous)
            except OSError:
                pass


def _cleanup_stale_artifacts(cache_root: Path, fingerprint: str) -> None:
    try:
        stale_paths = [
            *cache_root.glob(f".{fingerprint}.tmp-*"),
            *cache_root.glob(f".{fingerprint}.previous-*"),
            *cache_root.glob(f".{fingerprint}.source-*"),
        ]
        for path in stale_paths:
            _remove_path(path)
    except OSError as exc:
        raise MapCacheStorageError(
            f"Cannot clean stale empirical-map cache artifacts in {cache_root}"
        ) from exc


@contextmanager
def _verified_source_snapshot(
    source: Path,
    cache_root: Path,
    fingerprint: str,
    expected_source: Mapping[str, object],
):
    """Yield a private source snapshot whose bytes match the fingerprint."""

    try:
        descriptor, snapshot_text = tempfile.mkstemp(
            prefix=f".{fingerprint}.source-",
            dir=cache_root,
        )
    except OSError as exc:
        raise MapCacheStorageError(
            f"Cannot create empirical-map source snapshot in {cache_root}"
        ) from exc

    snapshot = Path(snapshot_text)
    digest = hashlib.sha256()
    size = 0
    try:
        try:
            with source.open("rb") as source_handle, os.fdopen(
                descriptor,
                "wb",
            ) as snapshot_handle:
                descriptor = -1
                while chunk := source_handle.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
                    size += len(chunk)
                    snapshot_handle.write(chunk)
                snapshot_handle.flush()
                os.fsync(snapshot_handle.fileno())
        except OSError as exc:
            raise MapCacheStorageError(
                f"Cannot create verified empirical-map snapshot for {source}"
            ) from exc

        if (
            expected_source.get("size") != size
            or expected_source.get("sha256") != digest.hexdigest()
        ):
            raise _CacheInputsChanged(
                f"Empirical map changed before its verified snapshot was parsed: {source}"
            )
        yield snapshot
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            snapshot.unlink(missing_ok=True)
        except OSError:
            pass


def _validated_builder_output(
    builder: Callable[[Path], Mapping[str, RateMapT]],
    source: Path,
) -> dict[str, RateMapT]:
    maps = builder(source)
    if not isinstance(maps, Mapping) or not maps:
        raise ValueError("Empirical map parser returned no maps")
    return dict(maps)


def _build_and_publish(
    *,
    entry: Path,
    fingerprint: str,
    fingerprint_payload: Mapping[str, object],
    maps: Mapping[str, RateMapT],
) -> None:
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{fingerprint}.tmp-",
            dir=entry.parent,
        )
    )
    try:
        _serialize_maps(staging, maps, fingerprint, fingerprint_payload)
        _publish_entry(staging, entry)
    finally:
        if os.path.lexists(staging):
            _remove_path(staging)


def _warn_cache_fallback(source: Path, exc: BaseException) -> None:
    try:
        warnings.warn(
            f"Empirical-map cache unavailable for {source}; falling back to direct text "
            f"parsing ({type(exc).__name__}: {exc})",
            RuntimeWarning,
            stacklevel=3,
        )
    except RuntimeWarning:
        # A cache remains an optimization even when the caller treats warnings as errors.
        pass


def load_or_build_rate_maps(
    *,
    source_path: str | Path,
    parse_parameters: Mapping[str, object],
    related_code_paths: Mapping[str, str | Path],
    builder: Callable[[Path], Mapping[str, RateMapT]],
    map_class: type[RateMapT],
) -> dict[str, RateMapT]:
    """Load verified read-only map arrays, rebuilding an invalid cache atomically."""
    source = Path(source_path).expanduser().resolve(strict=True)
    try:
        cache_root = map_cache_directory()
    except (OSError, RuntimeError, ValueError) as exc:
        _warn_cache_fallback(source, exc)
        return _validated_builder_output(builder, source)

    for attempt in range(_MAX_INPUT_CHANGE_RETRIES):
        try:
            fingerprint, payload = build_cache_fingerprint(
                source,
                parse_parameters,
                related_code_paths,
            )
        except _CacheInputsChanged:
            if attempt + 1 == _MAX_INPUT_CHANGE_RETRIES:
                raise RuntimeError(
                    f"Empirical map inputs kept changing while building cache: {source}"
                ) from None
            continue

        try:
            cache_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _warn_cache_fallback(source, exc)
            return _validated_builder_output(builder, source)

        entry = cache_root / fingerprint
        lock = cache_root / f"{fingerprint}.lock"
        try:
            with _exclusive_cache_lock(lock):
                _cleanup_stale_artifacts(cache_root, fingerprint)
                try:
                    cached_maps = _load_entry(entry, fingerprint, payload, map_class)
                except MapCacheValidationError:
                    with _verified_source_snapshot(
                        source,
                        cache_root,
                        fingerprint,
                        payload["source"],
                    ) as snapshot:
                        parsed_maps = _validated_builder_output(builder, snapshot)
                    current_fingerprint, current_payload = build_cache_fingerprint(
                        source,
                        parse_parameters,
                        related_code_paths,
                    )
                    if (
                        current_fingerprint != fingerprint
                        or current_payload != payload
                    ):
                        raise _CacheInputsChanged(
                            f"Empirical map inputs changed while building cache entry "
                            f"for {source}"
                        )
                    try:
                        _build_and_publish(
                            entry=entry,
                            fingerprint=fingerprint,
                            fingerprint_payload=payload,
                            maps=parsed_maps,
                        )
                        published_maps = _load_entry(
                            entry,
                            fingerprint,
                            payload,
                            map_class,
                        )
                    except (OSError, MapCacheValidationError) as exc:
                        _warn_cache_fallback(source, exc)
                        return parsed_maps
                    final_fingerprint, final_payload = build_cache_fingerprint(
                        source,
                        parse_parameters,
                        related_code_paths,
                    )
                    if final_fingerprint != fingerprint or final_payload != payload:
                        raise _CacheInputsChanged(
                            f"Empirical map inputs changed while publishing cache "
                            f"entry for {source}"
                        )
                    return published_maps

                current_fingerprint, current_payload = build_cache_fingerprint(
                    source,
                    parse_parameters,
                    related_code_paths,
                )
                if current_fingerprint != fingerprint or current_payload != payload:
                    raise _CacheInputsChanged(
                        f"Empirical map inputs changed while loading cache entry for {source}"
                    )
                return cached_maps
        except MapCacheStorageError as exc:
            _warn_cache_fallback(source, exc)
            return _validated_builder_output(builder, source)
        except _CacheInputsChanged:
            if attempt + 1 == _MAX_INPUT_CHANGE_RETRIES:
                raise RuntimeError(
                    f"Empirical map inputs kept changing while building cache: {source}"
                ) from None
            continue

    raise AssertionError("unreachable")
