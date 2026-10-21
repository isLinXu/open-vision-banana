#!/usr/bin/env python3
"""Build a single-pass indexed_gzip (.gzidx) + tar_index.json for split SNVC-50M volumes,
and optionally extract only the exact files referenced by a JSONL manifest in O(1) time.

Usage:
    # 1. Build both tar_index.json and tar_index.gzidx in ONE pass over the volumes:
    PYTHONPATH=src python scripts/build_snvc_tar_index.py \
        --tar-root /path/to/SenseNova-Vision-Corpus-50M \
        --output /path/to/SenseNova-Vision-Corpus-50M/tar_index.json

    # 2. (Optional) Materialize only the files needed by a pairs.jsonl using O(1) seek:
    PYTHONPATH=src python scripts/build_snvc_tar_index.py \
        --tar-root /path/to/SenseNova-Vision-Corpus-50M \
        --output /path/to/SenseNova-Vision-Corpus-50M/tar_index.json \
        --extract-manifest data/qwen_baseline/pairs_clean.jsonl \
        --extract-to /path/to/extracted_subset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.tar_dataset import (  # noqa: E402
    IndexedTarReader,
    SequentialTarReader,
    build_snvc_tar_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_snvc_tar_index")


def discover_volumes(tar_root: Path, prefix: str = "SenseNova-Vision-Corpus-50M.tar.gz") -> list[Path]:
    vols = []
    for p in sorted(tar_root.glob(f"{prefix}[0-9]*")):
        if not p.is_file():
            continue
        if p.suffix in {".aria2", ".tmp", ".part", ".gzidx", ".json"}:
            continue
        if (p.parent / f"{p.name}.aria2").exists():
            logger.warning("Skipping incomplete volume (has .aria2): %s", p)
            continue
        vols.append(p)
    return vols


def extract_manifest_files(
    index_payload: dict,
    manifest_path: Path,
    extract_to: Path,
    inner_prefix: str = "SenseNova-Vision-Corpus-50M",
) -> int:
    """Extract only the files referenced in ``manifest_path`` using O(1) seek sorted by offset."""
    files_map: dict[str, dict[str, int]] = index_payload["files"]
    volumes = [Path(v["path"]) for v in index_payload["volumes"]]
    gzidx = Path(index_payload["gzip_index"]) if index_payload.get("gzip_index") else None

    def resolve_key(rel: str) -> str | None:
        clean = rel.lstrip("./")
        if clean in files_map:
            return clean
        if inner_prefix:
            prefixed = f"{inner_prefix.rstrip('/')}/{clean}"
            if prefixed in files_map:
                return prefixed
        return None

    needed: dict[str, str] = {}  # tar_key -> relative output path
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("input", "target"):
                rel = row.get(field, "")
                key = resolve_key(rel)
                if key is not None:
                    needed[key] = rel.lstrip("./")

    if not needed:
        logger.info("No manifest files matched entries in tar_index.json")
        return 0

    # Sort by offset so even SequentialTarReader or OS page cache reads monotonically
    ordered = sorted(needed.items(), key=lambda item: files_map[item[0]]["offset"])
    reader = (
        IndexedTarReader(volumes, gzidx)
        if gzidx and gzidx.is_file()
        else SequentialTarReader(volumes)
    )
    extracted = 0
    try:
        for tar_key, rel_out in ordered:
            out_path = extract_to / rel_out
            if out_path.is_file():
                extracted += 1
                continue
            info = files_map[tar_key]
            data = reader.read_at(info["offset"], info["size"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            extracted += 1
    finally:
        reader.close()

    logger.info("Extracted %d manifest files to %s", extracted, extract_to)
    return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar-root", type=Path, required=True)
    parser.add_argument("--prefix", default="SenseNova-Vision-Corpus-50M.tar.gz")
    parser.add_argument("--output", type=Path, required=True, help="Output tar_index.json path")
    parser.add_argument("--gzip-index", type=Path, default=None, help="Output .gzidx path")
    parser.add_argument("--spacing-mb", type=int, default=64)
    parser.add_argument("--extract-manifest", type=Path, default=None)
    parser.add_argument("--extract-to", type=Path, default=None)
    parser.add_argument("--inner-prefix", default="SenseNova-Vision-Corpus-50M")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.is_file() and args.extract_manifest is not None:
        index_payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        volumes = discover_volumes(args.tar_root, args.prefix)
        if not volumes:
            raise SystemExit(f"No {args.prefix}[0-9]* volumes found in {args.tar_root}")
        index_payload = build_snvc_tar_index(
            volumes,
            args.output,
            gzip_index_path=args.gzip_index,
            spacing_mb=args.spacing_mb,
        )
        logger.info(
            "Built index %s (%d files, gzip_index=%s)",
            args.output,
            len(index_payload["files"]),
            index_payload["gzip_index"],
        )

    if args.extract_manifest is not None:
        if args.extract_to is None:
            raise SystemExit("--extract-to is required when --extract-manifest is set")
        extract_manifest_files(
            index_payload,
            args.extract_manifest,
            args.extract_to,
            inner_prefix=args.inner_prefix,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
