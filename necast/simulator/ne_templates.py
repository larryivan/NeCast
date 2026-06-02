from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .assets import asset_path
from .config import Config


@dataclass(frozen=True)
class NeTemplate:
    source_key: str
    method: str
    population: str
    generations_ago: np.ndarray
    ne: np.ndarray
    path: str
    time_unit_note: str


@dataclass
class EmpiricalNeTemplateStore:
    templates: list[NeTemplate]
    source_counts: dict[str, int]

    def choose(self, rng: np.random.Generator, cfg: Config) -> NeTemplate:
        weights = _parse_source_mix(cfg.empirical_ne_template_source_mix)
        available = [key for key in weights if self.source_counts.get(key, 0) > 0 and weights[key] > 0]
        if not available:
            available = [key for key, count in self.source_counts.items() if count > 0]
        if not available:
            raise RuntimeError("no empirical Ne templates are available")

        p = np.asarray([weights.get(key, 1.0) for key in available], dtype=np.float64)
        source_key = str(rng.choice(available, p=p / p.sum()))
        choices = [t for t in self.templates if t.source_key == source_key]
        return choices[int(rng.integers(0, len(choices)))]


_STORE_CACHE: dict[str, EmpiricalNeTemplateStore] = {}


def empirical_ne_templates_available(cfg: Config) -> bool:
    return len(load_empirical_ne_templates(cfg).templates) > 0


def empirical_ne_template_provenance(
    cfg: Config,
    *,
    include_source_files: bool = True,
) -> dict:
    """Fingerprint the exact parsed template curves and their source files."""

    store = load_empirical_ne_templates(cfg)
    digest = hashlib.sha256()
    ordered = sorted(
        store.templates,
        key=lambda template: (
            template.source_key,
            template.method,
            template.population,
            template.path,
        ),
    )

    def update_text(value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    for template in ordered:
        for value in (
            template.source_key,
            template.method,
            template.population,
            template.time_unit_note,
        ):
            update_text(value)
        for array in (template.generations_ago, template.ne):
            values = np.asarray(array, dtype="<f8")
            digest.update(values.size.to_bytes(8, "little"))
            digest.update(values.tobytes(order="C"))

    files: list[dict[str, int | str]] = []
    if include_source_files:
        for raw_path in sorted({template.path for template in ordered}):
            path = Path(raw_path).expanduser().resolve(strict=True)
            before = path.stat()
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            after = path.stat()
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if before_identity != after_identity:
                raise ValueError(
                    "Empirical Ne template changed while recording provenance: "
                    f"{path}"
                )
            files.append(
                {
                    "path": str(path),
                    "source_size_bytes": int(after.st_size),
                    "source_sha256": file_digest.hexdigest(),
                }
            )
    return {
        "n_templates": len(ordered),
        "source_counts": {
            str(key): int(value)
            for key, value in sorted(store.source_counts.items())
        },
        "parsed_templates_sha256": digest.hexdigest(),
        "source_files": files,
    }


def load_empirical_ne_templates(cfg: Config) -> EmpiricalNeTemplateStore:
    root = _resolve_template_dir(cfg.empirical_ne_template_dir)
    key = str(root.resolve()) if root.exists() else str(root)
    cached = _STORE_CACHE.get(key)
    if cached is not None:
        return cached

    templates: list[NeTemplate] = []
    templates.extend(_load_smcpp_1kg(root / "smcpp_popsizes_1kg.csv"))
    templates.extend(
        _load_normalized(
            root / "phlash_unified_populations_normalized.csv",
            source_key="phlash_unified",
            method="PHLaSH",
            time_unit_note="generations; normalized from phlash_paper fig7a",
        )
    )
    if not any(t.source_key == "phlash_unified" for t in templates):
        templates.extend(
            _load_phlash_raw(
                root / "phlash_fig7a_unified_populations.csv",
                source_key="phlash_unified",
                method="PHLaSH",
            )
        )
    templates.extend(
        _load_normalized(
            root / "phlash_superpopulations_normalized.csv",
            source_key="phlash_super",
            method="PHLaSH",
            time_unit_note="generations; normalized from phlash_paper fig7b",
        )
    )
    if not any(t.source_key == "phlash_super" for t in templates):
        templates.extend(
            _load_phlash_raw(
                root / "phlash_fig7b_superpopulations.csv",
                source_key="phlash_super",
                method="PHLaSH",
            )
        )
    templates.extend(
        _load_normalized(
            root / "msmc2_yoruba_french_within_ne_normalized.csv",
            source_key="msmc2_example",
            method="MSMC2",
            time_unit_note="generations; converted from MSMC2 scaled time using mu=1.25e-8",
        )
    )
    templates.extend(
        _load_normalized(
            root / "msmc_im_yoruba_french_ne_normalized.csv",
            source_key="msmc_im_example",
            method="MSMC-IM",
            time_unit_note="generations; MSMC-IM example output",
        )
    )

    counts: dict[str, int] = {}
    for template in templates:
        counts[template.source_key] = counts.get(template.source_key, 0) + 1
    store = EmpiricalNeTemplateStore(templates=templates, source_counts=counts)
    _STORE_CACHE[key] = store
    return store


def _resolve_template_dir(path: str) -> Path:
    return asset_path(path)


def _parse_source_mix(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            key, value = token.split(":", 1)
            out[key.strip()] = max(0.0, float(value))
        else:
            out[token] = 1.0
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_smcpp_1kg(path: Path) -> list[NeTemplate]:
    rows = _read_csv(path)
    if not rows:
        return []
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("plot_type", "path") != "path":
            continue
        pop = row["label"]
        generations = float(row["x"]) / 29.0
        grouped.setdefault(pop, []).append((generations, float(row["y"])))
    return _templates_from_groups(
        grouped,
        source_key="smcpp_1kg",
        method="SMC++",
        path=path,
        time_unit_note="years converted to generations using generation_time=29",
    )


def _load_normalized(path: Path, source_key: str, method: str, time_unit_note: str) -> list[NeTemplate]:
    rows = _read_csv(path)
    if not rows:
        return []
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        pop = row["population"]
        grouped.setdefault(pop, []).append((float(row["generations_ago"]), float(row["Ne"])))
    return _templates_from_groups(grouped, source_key, method, path, time_unit_note)


def _load_phlash_raw(path: Path, source_key: str, method: str) -> list[NeTemplate]:
    rows = _read_csv(path)
    if not rows:
        return []
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        pop = _clean_phlash_population(row["pop"])
        grouped.setdefault(pop, []).append((float(row["years"]), float(row["Ne"])))
    return _templates_from_groups(
        grouped,
        source_key=source_key,
        method=method,
        path=path,
        time_unit_note="generations; phlash_paper CSV column is named years but generated from T in generations",
    )


def _clean_phlash_population(value: str) -> str:
    match = re.search(r"unified/(.*?)/phlash/estimates\.pkl", value)
    return match.group(1) if match else value


def _templates_from_groups(
    grouped: dict[str, list[tuple[float, float]]],
    source_key: str,
    method: str,
    path: Path,
    time_unit_note: str,
) -> list[NeTemplate]:
    out: list[NeTemplate] = []
    for pop, pairs in grouped.items():
        arr = np.asarray(sorted(pairs), dtype=np.float64)
        arr = arr[np.isfinite(arr).all(axis=1)]
        arr = arr[(arr[:, 0] >= 0) & (arr[:, 1] > 0)]
        if arr.shape[0] < 2:
            continue
        _, unique_idx = np.unique(arr[:, 0], return_index=True)
        arr = arr[np.sort(unique_idx)]
        out.append(
            NeTemplate(
                source_key=source_key,
                method=method,
                population=pop,
                generations_ago=arr[:, 0].astype(np.float64),
                ne=arr[:, 1].astype(np.float64),
                path=str(path),
                time_unit_note=time_unit_note,
            )
        )
    return out
