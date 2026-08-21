"""Durable file operations, in one place.

Every module here that writes a receipt, a plan or a completion marker needs
the same three things: a write that either lands whole or not at all, a hash
that identifies a file's contents, and a canonical encoding for the values that
get hashed.  Three modules grew their own, and the copies drifted -- which is
the failure mode that makes duplication worth removing here rather than merely
untidy.

``replacement.py``'s atomic write fsynced the file and then renamed it without
fsyncing the directory, so a crash between the rename and the next flush could
lose the rename itself and leave the old file in place.  That is precisely the
loss the module exists to prevent.  ``shards.py``'s directory fsync opened the
directory with no ``O_DIRECTORY`` and no ``except OSError``, where
``map_cache.py``'s tolerated both -- two fixes applied to two copies of the
same function, neither aware of the other.

One implementation now, with the strongest guarantee any caller needed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Read in whole-megabyte chunks: large enough that the syscall overhead
# disappears against a multi-megabyte shard, small enough not to hold the file
# in memory.
_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def file_sha256(path: Path) -> str:
    """Hash a file's contents without reading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    """Encode a value so equal values encode identically.

    Sorted keys and no incidental whitespace, because the result is hashed:
    two dictionaries that differ only in insertion order have to produce one
    digest, or a receipt stops identifying what it was meant to identify.
    ``allow_nan`` is off because NaN has no JSON spelling and the encoders that
    invent one do not agree with each other.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """The digest of a value's canonical encoding."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def fsync_directory(path: Path) -> None:
    """Flush a directory entry, so a rename into it survives a crash.

    Renaming is atomic but not durable: the new name can still be in the
    kernel's cache when the machine stops.  Opening a directory is not portable
    -- Windows refuses it outright -- so a failure here is a platform that
    cannot offer the guarantee rather than an error to propagate.
    """

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


def write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
    *,
    canonical: bool = False,
) -> None:
    """Write JSON so a reader sees the old file or the new one, never a partial.

    Write to a temporary name in the same directory, flush it to disk, rename
    it over the target, then flush the directory.  Every step is needed: a
    rename across filesystems is not atomic, an unflushed file can be renamed
    into place empty, and an unflushed directory can lose the rename.

    ``canonical`` selects the encoding used for values that get hashed --
    sorted and compact -- rather than the indented form meant to be read.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The pid and clock keep two processes writing the same target from
    # colliding on the temporary name.
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
    try:
        payload = (
            canonical_bytes(value)
            if canonical
            else json.dumps(
                value, indent=2, ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
        )
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path, description: str, error: type[Exception]) -> dict[str, Any]:
    """Read a JSON object, raising ``error`` with the path in the message.

    The caller supplies the exception type because the two replacement modules
    have their own, and which one is raised is part of their contract.
    """

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error(f"{description} at {path} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise error(f"{description} at {path} is not a JSON object")
    return value
