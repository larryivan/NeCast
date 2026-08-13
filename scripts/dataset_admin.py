#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import sys
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


REQUIRED_NPZ_KEYS = {
    "sample_id",
    "target_log10_ne",
    "variant_positions_bp",
    "variant_offsets",
    "genotype_packed",
    "missing_flat_idx",
    "missing_offsets",
    "packed_hap_bytes",
    "n_haplotypes",
    "sequence_length",
    "obs_rec_pos",
    "obs_rec_offsets",
    "obs_rec_rate",
    "obs_mut_pos",
    "obs_mut_offsets",
    "obs_mut_rate",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def shard_paths(data_dir: Path) -> list[Path]:
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        rels = read_json(manifest).get("samples", {}).get("shards", [])
        if rels:
            paths = [data_dir / rel for rel in rels]
            if all(path.exists() for path in paths):
                return paths
    return sorted(path for path in (data_dir / "samples").glob("*.npz") if ".tmp" not in path.name)


def check_offsets(name: str, offsets: np.ndarray, data_len: int, n_samples: int) -> None:
    if offsets.shape != (n_samples + 1,):
        raise ValueError(f"{name}: shape={offsets.shape}, expected {(n_samples + 1,)}")
    if len(offsets) and (offsets[0] != 0 or offsets[-1] != data_len):
        raise ValueError(f"{name}: invalid endpoints {offsets[0]}..{offsets[-1]}, data_len={data_len}")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"{name}: offsets are not monotone")


def check_monotone_by_offsets(name: str, values: np.ndarray, offsets: np.ndarray) -> None:
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        if b - a > 1 and np.any(np.diff(values[a:b]) < 0):
            raise ValueError(f"{name}: positions are not monotone in local sample {i}")


def check_shard(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        missing_keys = REQUIRED_NPZ_KEYS.difference(data.files)
        if missing_keys:
            raise ValueError(f"{path.name}: missing keys {sorted(missing_keys)}")

        ids = [str(x) for x in data["sample_id"]]
        n = len(ids)
        h = int(data["n_haplotypes"][0])
        packed_bytes = int(data["packed_hap_bytes"][0])
        if packed_bytes != int(np.ceil(h / 8)):
            raise ValueError(f"{path.name}: invalid packed_hap_bytes")

        variant_offsets = data["variant_offsets"]
        positions = data["variant_positions_bp"]
        check_offsets("variant_offsets", variant_offsets, len(positions), n)
        if data["genotype_packed"].shape != (len(positions), packed_bytes):
            raise ValueError(f"{path.name}: genotype_packed shape does not match variants")
        check_monotone_by_offsets("variant_positions_bp", positions, variant_offsets)

        missing_offsets = data["missing_offsets"]
        missing_flat = data["missing_flat_idx"]
        check_offsets("missing_offsets", missing_offsets, len(missing_flat), n)
        for i, (a, b) in enumerate(zip(missing_offsets[:-1], missing_offsets[1:])):
            flat = missing_flat[a:b].astype(np.int64, copy=False)
            n_variants = int(variant_offsets[i + 1] - variant_offsets[i])
            if len(flat) and (flat[0] < 0 or flat[-1] >= n_variants * h or np.any(np.diff(flat) < 0)):
                raise ValueError(f"{path.name}: invalid missing_flat_idx in local sample {i}")

        for prefix in ("obs_rec", "obs_mut"):
            pos = data[f"{prefix}_pos"]
            rate = data[f"{prefix}_rate"]
            pos_offsets = data[f"{prefix}_offsets"]
            check_offsets(f"{prefix}_offsets", pos_offsets, len(pos), n)
            check_monotone_by_offsets(f"{prefix}_pos", pos, pos_offsets)
            # One offsets array serves both: a rate lives between two
            # breakpoints, so a row holds one fewer rate than positions.  The
            # derived offsets have to land exactly on the rate array's end.
            lengths = np.diff(np.asarray(pos_offsets, dtype=np.int64))
            if int(np.maximum(lengths - 1, 0).sum()) != len(rate):
                raise ValueError(f"{path.name}: {prefix} rate count disagrees with its offsets")

        if data["target_log10_ne"].shape[0] != n or not np.isfinite(data["target_log10_ne"]).all():
            raise ValueError(f"{path.name}: invalid target_log10_ne")
        if np.any(data["sequence_length"] <= 0):
            raise ValueError(f"{path.name}: non-positive sequence_length")

    shard_meta = path.with_suffix(".jsonl.gz")
    if not shard_meta.exists():
        raise ValueError(f"missing {shard_meta}")
    meta_ids = [str(row.get("sample_id", "")) for row in read_jsonl_gz(shard_meta)]
    if meta_ids != ids:
        raise ValueError(f"{shard_meta.name}: sample_id order differs from npz")
    return ids


def shard_ids_from_any(path: Path) -> list[str]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return [str(x) for x in data["sample_id"]]
    except Exception:
        meta = path.with_suffix(".jsonl.gz")
        if meta.exists():
            try:
                return [str(row.get("sample_id", "")) for row in read_jsonl_gz(meta)]
            except Exception:
                pass
    return []


def scan_shards(shards: list[Path]) -> tuple[list[Path], list[str], list[tuple[Path, list[str], str]]]:
    good_shards: list[Path] = []
    good_ids: list[str] = []
    bad: list[tuple[Path, list[str], str]] = []
    for shard in tqdm(shards, desc="check shards", unit="shard"):
        try:
            ids = check_shard(shard)
        except Exception as exc:
            bad.append((shard, shard_ids_from_any(shard), str(exc)))
        else:
            good_shards.append(shard)
            good_ids.extend(ids)
    return good_shards, good_ids, bad


def rewrite_metadata_after_drop(data_dir: Path, keep_ids: set[str]) -> pd.DataFrame | None:
    meta_dir = data_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    csv_path = meta_dir / "samples.csv"
    df = None
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["sample_id"].astype(str).isin(keep_ids)].copy()
        tmp_csv = csv_path.with_suffix(".tmp.csv")
        df.to_csv(tmp_csv, index=False)
        tmp_csv.replace(csv_path)
    else:
        print(f"[warn] missing {csv_path}; skip global csv metadata update", file=sys.stderr)

    parquet_path = meta_dir / "samples.parquet"
    if parquet_path.exists() and df is not None:
        tmp_parquet = parquet_path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp_parquet, index=False)
        tmp_parquet.replace(parquet_path)

    jsonl_path = meta_dir / "samples.jsonl.gz"
    if jsonl_path.exists():
        rows = [row for row in read_jsonl_gz(jsonl_path) if str(row.get("sample_id", "")) in keep_ids]
        write_jsonl_gz(jsonl_path, rows)
    return df


def update_manifest_after_drop(
    path: Path,
    data_dir: Path,
    good_shards: list[Path],
    n_samples: int,
    df: pd.DataFrame | None,
) -> None:
    if not path.exists():
        return
    obj = read_json(path)
    target = obj.get("samples") if isinstance(obj.get("samples"), dict) else obj
    rel_shards = [str(p.relative_to(data_dir)) for p in good_shards]
    target["n_samples"] = int(n_samples)
    target["n_shards"] = int(len(good_shards))
    target["shards"] = rel_shards

    counts = {
        "source_counts": "source_type",
        "demography_counts": "demography_type",
        "map_counts": "map_mode",
    }
    if df is not None:
        for out_key, col in counts.items():
            if col in df.columns and out_key in target:
                target[out_key] = {str(k): int(v) for k, v in df[col].value_counts().items()}

    sample_bytes = 0
    for shard in good_shards:
        sample_bytes += shard.stat().st_size
        meta = shard.with_suffix(".jsonl.gz")
        if meta.exists():
            sample_bytes += meta.stat().st_size
    if "samples_bytes" in target:
        target["samples_bytes"] = int(sample_bytes)
        target["samples_gb"] = float(sample_bytes / 1e9)
        target["avg_bytes_per_sample"] = float(sample_bytes / max(1, n_samples))
        target["avg_mb_per_sample"] = float(sample_bytes / max(1, n_samples) / 1e6)
        target["estimated_total_samples_gb"] = float(sample_bytes / 1e9)
    write_json(path, obj)


def drop_bad_shards(data_dir: Path, good_shards: list[Path], good_ids: list[str], bad: list[tuple[Path, list[str], str]]) -> None:
    unknown = [shard for shard, ids, _ in bad if not ids]
    if unknown:
        names = ", ".join(p.name for p in unknown[:5])
        raise ValueError(f"cannot drop bad shards with unknown sample ids: {names}")

    keep_ids = set(good_ids)
    df = rewrite_metadata_after_drop(data_dir, keep_ids)
    update_manifest_after_drop(data_dir / "manifest.json", data_dir, good_shards, len(good_ids), df)
    update_manifest_after_drop(data_dir / "samples" / "DONE.json", data_dir, good_shards, len(good_ids), df)

    quarantine = data_dir / "samples" / f"_bad_{datetime.now():%Y%m%d_%H%M%S}"
    quarantine.mkdir(parents=True, exist_ok=True)
    for shard, _, _ in tqdm(bad, desc="move bad shards", unit="shard"):
        for path in [shard, shard.with_suffix(".jsonl.gz")]:
            if path.exists():
                shutil.move(str(path), str(quarantine / path.name))
    print(f"dropped {sum(len(ids) for _, ids, _ in bad)} samples from {len(bad)} bad shards")
    print(f"bad shard files moved to {quarantine}")


def check_dataset(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    shards = shard_paths(data_dir)
    if not shards:
        raise SystemExit(f"no shard_*.npz found in {data_dir / 'samples'}")

    good_shards, shard_ids, bad = scan_shards(shards)
    bad_ids = [sample_id for _, ids, _ in bad for sample_id in ids]

    print(f"checked {len(shards)} shards")
    print(f"good: {len(good_shards)} shards, {len(shard_ids)} samples")
    print(f"bad:  {len(bad)} shards, {len(bad_ids)} samples")
    for shard, ids, reason in bad[:20]:
        print(f"  BAD {shard.name}: {len(ids)} samples; {reason}")
    if len(bad) > 20:
        print(f"  ... {len(bad) - 20} more bad shards")

    if bad:
        try:
            answer = input("Drop bad shards from dataset metadata and move files to quarantine? [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer in {"y", "yes"}:
            drop_bad_shards(data_dir, good_shards, shard_ids, bad)
            return 0
        print("bad shards kept; dataset not modified")
        return 1

    if len(shard_ids) != len(set(shard_ids)):
        raise ValueError("duplicate sample_id in shards")

    csv_path = data_dir / "metadata" / "samples.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        csv_ids = df["sample_id"].astype(str).tolist()
        if len(csv_ids) != len(shard_ids) or set(csv_ids) != set(shard_ids):
            raise ValueError("metadata/samples.csv sample_id set differs from shards")
    else:
        print(f"[warn] missing {csv_path}; checked shard-level metadata only", file=sys.stderr)

    jsonl_path = data_dir / "metadata" / "samples.jsonl.gz"
    if jsonl_path.exists():
        jsonl_ids = [str(row.get("sample_id", "")) for row in read_jsonl_gz(jsonl_path)]
        if len(jsonl_ids) != len(shard_ids) or set(jsonl_ids) != set(shard_ids):
            raise ValueError("metadata/samples.jsonl.gz sample_id set differs from shards")

    manifest = data_dir / "manifest.json"
    if manifest.exists():
        expected = int(read_json(manifest).get("samples", {}).get("n_samples", len(shard_ids)))
        if expected != len(shard_ids):
            raise ValueError(f"manifest n_samples={expected}, actual={len(shard_ids)}")

    print(f"OK: {len(shard_ids)} samples, {len(shards)} shards")
    return 0


def collect_sample_ids(data_dir: Path, order: str, reverse: bool) -> list[tuple[Path, int, str]]:
    records: list[tuple[tuple, Path, int, str]] = []
    for shard in tqdm(shard_paths(data_dir), desc="scan shards", unit="shard"):
        with np.load(shard, allow_pickle=False) as data:
            ids = [str(x) for x in data["sample_id"]]
        if order == "sample_index":
            meta = read_jsonl_gz(shard.with_suffix(".jsonl.gz"))
            keys = [int(row.get("sample_index", i)) for i, row in enumerate(meta)]
        else:
            keys = [i for i in range(len(ids))]
        for i, old_id in enumerate(ids):
            key = (keys[i], shard.name, i) if order == "sample_index" else (shard.stat().st_mtime, shard.name, i)
            records.append((key, shard, i, old_id))
    records.sort(reverse=reverse)
    return [(shard, i, old_id) for _, shard, i, old_id in records]


def new_sample_id(prefix: str, value: int, width: int) -> str:
    return f"{prefix}{value:0{width}d}" if width > 0 else f"{prefix}{value}"


def append_npz_sample_ids(shard: Path, sample_ids: list[str]) -> None:
    buf = io.BytesIO()
    np.save(buf, np.array(sample_ids, dtype="U64"), allow_pickle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(shard, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("sample_id.npy", buf.getvalue())


def shard_name_from_ids(sample_ids: list[str]) -> str:
    first = sample_ids[0]
    last = sample_ids[-1]
    stem = first if first == last else f"{first}-{last}"
    return stem.replace("/", "_")


def move_if_needed(src: Path, dst: Path) -> Path:
    if src == dst:
        return src
    if dst.exists():
        raise FileExistsError(f"target file already exists: {dst}")
    src.replace(dst)
    return dst


def rename_shard_files(shard: Path, sample_ids: list[str]) -> Path:
    new_npz = shard.with_name(f"{shard_name_from_ids(sample_ids)}.npz")
    old_meta = shard.with_suffix(".jsonl.gz")
    new_meta = new_npz.with_suffix(".jsonl.gz")
    moved_npz = move_if_needed(shard, new_npz)
    if old_meta.exists():
        move_if_needed(old_meta, new_meta)
    return moved_npz


def rewrite_shard_ids(shard: Path, mapping: dict[str, str]) -> Path:
    with np.load(shard, allow_pickle=False) as data:
        old_ids = [str(x) for x in data["sample_id"]]
    ids = [mapping[old_id] for old_id in old_ids]
    if ids != old_ids:
        append_npz_sample_ids(shard, ids)

    meta_path = shard.with_suffix(".jsonl.gz")
    rows = read_jsonl_gz(meta_path)
    for row in rows:
        row["sample_id"] = mapping[str(row["sample_id"])]
    write_jsonl_gz(meta_path, rows)
    return rename_shard_files(shard, ids)


def rewrite_metadata_ids(data_dir: Path, mapping: dict[str, str]) -> None:
    meta_dir = data_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    csv_path = meta_dir / "samples.csv"
    df = None
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["sample_id"] = df["sample_id"].astype(str).map(mapping)
        if df["sample_id"].isna().any():
            raise ValueError("metadata/samples.csv contains sample_id not present in shards")
        tmp_csv = csv_path.with_suffix(".tmp.csv")
        df.to_csv(tmp_csv, index=False)
        tmp_csv.replace(csv_path)
    else:
        print(f"[warn] missing {csv_path}; skip global csv metadata rename", file=sys.stderr)

    parquet_path = meta_dir / "samples.parquet"
    if parquet_path.exists() and df is not None:
        tmp_parquet = parquet_path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp_parquet, index=False)
        tmp_parquet.replace(parquet_path)

    jsonl_path = meta_dir / "samples.jsonl.gz"
    if jsonl_path.exists():
        rows = read_jsonl_gz(jsonl_path)
        for row in rows:
            row["sample_id"] = mapping[str(row["sample_id"])]
        write_jsonl_gz(jsonl_path, rows)


def read_rename_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["old_sample_id"].astype(str), df["new_sample_id"].astype(str), strict=True))


def current_id_mapping(records: list[tuple[Path, int, str]], old_to_new: dict[str, str]) -> dict[str, str]:
    final_ids = set(old_to_new.values())
    mapping: dict[str, str] = {}
    for _, _, sample_id in records:
        if sample_id in old_to_new:
            mapping[sample_id] = old_to_new[sample_id]
        elif sample_id in final_ids:
            mapping[sample_id] = sample_id
        else:
            raise ValueError(f"sample_id not found in rename map: {sample_id}")
    return mapping


def update_manifest_after_rename(path: Path, data_dir: Path, shards: list[Path], n_samples: int) -> None:
    if not path.exists():
        return
    obj = read_json(path)
    target = obj.get("samples") if isinstance(obj.get("samples"), dict) else obj
    target["n_samples"] = int(n_samples)
    target["n_shards"] = int(len(shards))
    target["shards"] = [str(shard.relative_to(data_dir)) for shard in shards]
    write_json(path, obj)


def rename_samples(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir).resolve()
    records = collect_sample_ids(data_dir, args.order, args.reverse)
    if not records:
        raise ValueError(f"no shards found in {data_dir / 'samples'}")
    if args.map_file:
        mapping = current_id_mapping(records, read_rename_map(Path(args.map_file)))
    else:
        if args.prefix is None or args.start_id is None:
            raise ValueError("rename requires --prefix and --start-id unless --map-file is used")
        mapping = {
            old_id: new_sample_id(args.prefix, args.start_id + i, args.width)
            for i, (_, _, old_id) in enumerate(records)
        }
    if len(mapping) != len(records) or len(set(mapping.values())) != len(mapping):
        raise ValueError("sample ids are not unique")
    if max(map(len, mapping.values())) > 64:
        raise ValueError("new sample_id exceeds shard dtype U64")

    print(f"rename {len(mapping)} samples")
    for old_id, new_id in list(mapping.items())[:5]:
        print(f"  {old_id} -> {new_id}")
    if len(mapping) > 5:
        old_id = next(reversed(mapping))
        print("  ...")
        print(f"  {old_id} -> {mapping[old_id]}")
    if not args.apply:
        print("dry run; add --apply to rewrite files")
        return 0

    (data_dir / "metadata").mkdir(parents=True, exist_ok=True)
    if args.map_file:
        map_path = Path(args.map_file)
    else:
        map_path = data_dir / "metadata" / f"sample_id_rename_map_{datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame({"old_sample_id": mapping.keys(), "new_sample_id": mapping.values()}).to_csv(map_path, index=False)
    rewrite_metadata_ids(data_dir, mapping)
    shards = sorted({shard for shard, _, _ in records})
    workers = max(1, min(args.workers, len(shards)))
    if workers == 1:
        new_shards = []
        for shard in tqdm(shards, desc="rewrite shards", unit="shard"):
            new_shards.append(rewrite_shard_ids(shard, mapping))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(rewrite_shard_ids, shard, mapping) for shard in shards]
            new_shards = []
            for fut in tqdm(as_completed(futures), total=len(futures), desc="rewrite shards", unit="shard"):
                new_shards.append(fut.result())
    new_shards = sorted(new_shards)
    update_manifest_after_rename(data_dir / "manifest.json", data_dir, new_shards, len(records))
    update_manifest_after_rename(data_dir / "samples" / "DONE.json", data_dir, new_shards, len(records))

    print(f"written rename map: {map_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="NeCast simulation dataset admin")
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser("check")
    check.add_argument("data_dir")
    check.set_defaults(func=check_dataset)

    rename = sub.add_parser("rename")
    rename.add_argument("data_dir")
    rename.add_argument("--prefix")
    rename.add_argument("--start-id", type=int)
    rename.add_argument("--width", type=int, default=8)
    rename.add_argument("--order", choices=["mtime", "sample_index"], default="mtime")
    rename.add_argument("--reverse", action="store_true")
    rename.add_argument("--apply", action="store_true")
    rename.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    rename.add_argument("--map-file", help="resume/apply an existing sample_id_rename_map csv")
    rename.set_defaults(func=rename_samples)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
