from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FINGERPRINT_VERSION = 1
# Bump whenever identical Config values would produce meaningfully different
# samples or shard semantics.
SAMPLE_GENERATOR_SCHEMA_VERSION = 1
DONE_CONFIG_STATE_KEY = "sample_generation_config"
RUNTIME_CONFIG_FIELDS = frozenset(
    {
        "dataset_name",
        "compute_chunk_size",
        "force",
        "out_dir",
        "workers",
    }
)


class DatasetConfigMismatchError(RuntimeError):
    """Raised when an existing dataset cannot safely be reused."""


def _as_mapping(config: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dataclass instance or mapping")


def _json_compatible(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"config value of type {type(value).__name__} is not JSON-compatible")


def sample_generation_config(config: Any) -> dict[str, Any]:
    """Return the config fields that determine generated sample data or its format."""
    raw = _as_mapping(config)
    return {
        key: _json_compatible(value)
        for key, value in raw.items()
        if key not in RUNTIME_CONFIG_FIELDS
    }


def sample_generation_fingerprint(config: Any) -> str:
    relevant = sample_generation_config(config)
    canonical = json.dumps(
        relevant,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generation_config_state(config: Any) -> dict[str, Any]:
    relevant = sample_generation_config(config)
    return {
        "version": FINGERPRINT_VERSION,
        "generator_schema_version": SAMPLE_GENERATOR_SCHEMA_VERSION,
        "algorithm": "sha256",
        "sha256": sample_generation_fingerprint(relevant),
        "ignored_fields": sorted(RUNTIME_CONFIG_FIELDS),
        "config": relevant,
    }


def stamp_done_manifest(manifest: Mapping[str, Any], config: Any) -> dict[str, Any]:
    stamped = dict(manifest)
    stamped[DONE_CONFIG_STATE_KEY] = generation_config_state(config)
    return stamped


def _mismatch_details(stored: Any, current: Any) -> str:
    if not isinstance(stored, Mapping) or not isinstance(current, Mapping):
        return ""
    keys = sorted(set(stored) | set(current))
    changed = [key for key in keys if stored.get(key) != current.get(key)]
    if not changed:
        return ""
    preview = ", ".join(changed[:8])
    if len(changed) > 8:
        preview += f", ... (+{len(changed) - 8} more)"
    return f" Differing sample-generation fields: {preview}."


def _reuse_error(message: str) -> DatasetConfigMismatchError:
    return DatasetConfigMismatchError(
        f"{message} Refusing to reuse the existing shards; rerun with --force to regenerate them."
    )


def validate_done_for_reuse(
    done_manifest: Mapping[str, Any],
    current_config: Any,
) -> dict[str, Any]:
    """
    Validate that DONE belongs to the requested sample-generation configuration.

    Returns the canonical state for a fingerprinted DONE. Legacy completion
    files are deliberately rejected because a mutable config.json cannot prove
    which configuration generated their shards.
    """
    current_state = generation_config_state(current_config)
    stored_state = done_manifest.get(DONE_CONFIG_STATE_KEY)

    if stored_state is None:
        raise _reuse_error(
            "DONE.json predates trustworthy configuration fingerprints; legacy "
            "config.json cannot prove which settings generated the existing shards."
        )

    if not isinstance(stored_state, Mapping):
        raise _reuse_error(f"{DONE_CONFIG_STATE_KEY!r} in DONE.json is malformed.")
    if stored_state.get("version") != FINGERPRINT_VERSION:
        raise _reuse_error(
            "DONE.json uses an unsupported sample-generation fingerprint version "
            f"({stored_state.get('version')!r})."
        )
    if stored_state.get("generator_schema_version") != SAMPLE_GENERATOR_SCHEMA_VERSION:
        raise _reuse_error(
            "DONE.json was produced by an unsupported sample-generator schema "
            f"({stored_state.get('generator_schema_version')!r})."
        )
    if stored_state.get("algorithm") != "sha256":
        raise _reuse_error(
            "DONE.json uses an unsupported sample-generation fingerprint algorithm "
            f"({stored_state.get('algorithm')!r})."
        )

    stored_sha256 = stored_state.get("sha256")
    if not isinstance(stored_sha256, str) or len(stored_sha256) != 64:
        raise _reuse_error("DONE.json contains a malformed sample-generation fingerprint.")

    stored_config = stored_state.get("config")
    if stored_config is not None:
        if not isinstance(stored_config, Mapping):
            raise _reuse_error("DONE.json contains a malformed sample-generation config snapshot.")
        if sample_generation_fingerprint(stored_config) != stored_sha256:
            raise _reuse_error(
                "The sample-generation config snapshot in DONE.json does not match its fingerprint."
            )

    if stored_sha256 != current_state["sha256"]:
        details = _mismatch_details(stored_config, current_state["config"])
        raise _reuse_error(
            "The requested configuration does not match the configuration that generated DONE.json."
            + details
        )
    return current_state
