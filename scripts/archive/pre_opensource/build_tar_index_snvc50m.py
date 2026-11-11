#!/usr/bin/env python3
"""Build a file index for SN-VC-50M split tar.gz volumes.

Scans every tar.gz volume and records which files reside in which volume.
The resulting JSON index is used by ``TarArchiveDataset`` to read training
images directly from the compressed archives without full extraction.

Index format (JSON)::

    {
      "volumes": [
        {"path": "/abs/path/to/...tar.gz000", "num_entries": 12345},
        ...
      ],
      "files": {
        "SenseNova-Vision-Corpus-50M/coco2017/train2017/000009.jpg": {
          "vol": 0, "offset": 1024, "size": 204800
        },
        ...
      }
    }

Usage::

    python scripts/build_tar_index_snvc50m.py \\
        --tar-root data/raw/SenseNova-Vision-Corpus-50M \\
        --output   output/snvc50m_manifest/tar_index.json \\
        --manifest output/snvc50m_manifest/pairs_all.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import struct
import sys
import tarfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tar_index")

# ── Tar scanning ──────────────────────────────────────────────────────────

_TAR_BLOCK = 512


def _round_block(n: int) -> int:
    """Round up to next tar block boundary."""
    return (n + _TAR_BLOCK - 1) // _TAR_BLOCK * _TAR_BLOCK


def scan_volume(vol_path: Path, needed_prefixes: set[str] | None = None) -> dict:
    """Scan a single tar.gz volume and return file entries.

    Parameters
    ----------
    vol_path:
        Path to a ``.tar.gzNNN`` split volume.
    needed_prefixes:
        If given, only record files whose path starts with one of these
        prefixes (e.g. ``SenseNova-Vision-Corpus-50M/coco2017/``).

    Returns
    -------
    dict with keys ``num_entries``, ``files`` (mapping relpath → {offset, size}).
    """
    entries: dict[str, dict] = {}
    num_entries = 0

    try:
        with gzip.open(vol_path, "rb") as gz:
            offset = 0
            while True:
                # Read tar header block
                header_block = gz.read(_TAR_BLOCK)
                if not header_block or len(header_block) < _TAR_BLOCK:
                    break
                offset += _TAR_BLOCK

                # Check for end-of-archive (two zero blocks)
                if header_block == b"\x00" * _TAR_BLOCK:
                    break

                # Parse header
                name = header_block[:100].rstrip(b"\x00").decode("utf-8", errors="replace")
                # Size is at offset 124, 12 bytes, octal
                try:
                    size_str = header_block[124:136].rstrip(b"\x00").rstrip(b" ")
                    size = int(size_str, 8) if size_str else 0
                except (ValueError, IndexError):
                    size = 0

                typeflag = chr(header_block[156])

                # Only record regular files
                if typeflag in ("0", "\x00", "") and size > 0:
                    num_entries += 1
                    if needed_prefixes is None or any(
                        name.startswith(p) for p in needed_prefixes
                    ):
                        entries[name] = {"offset": offset, "size": size}

                # Skip file data (rounded up to block boundary)
                skip = _round_block(size)
                # Read and discard — gzip needs sequential consumption
                # We read in chunks to avoid loading huge files into memory
                remaining = skip
                while remaining > 0:
                    chunk = min(remaining, 4 * 1024 * 1024)  # 4MB chunks
                    data = gz.read(chunk)
                    if not data:
                        break
                    offset += len(data)
                    remaining -= len(data)

    except (gzip.BadGzipFile, OSError, EOFError) as exc:
        logger.warning("Error scanning %s: %s", vol_path.name, exc)

    return {"num_entries": num_entries, "files": entries}


def discover_volumes(tar_root: Path) -> list[Path]:
    """Find all split tar.gz volumes in order."""
    volumes = []
    for f in sorted(tar_root.glob("SenseNova-Vision-Corpus-50M.tar.gz[0-9]*")):
        if f.name.endswith(".aria2"):
            continue
        volumes.append(f)
    return volumes


def needed_prefixes(manifest_path: Path | None) -> set[str] | None:
    """Derive the set of top-level directory prefixes we need from the manifest.

    This avoids indexing files we'll never use (e.g. structure_view_understanding).
    """
    if manifest_path is None or not manifest_path.is_file():
        return None  # Index everything

    prefixes: set[str] = set()
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for key in ("input", "target"):
                p = rec.get(key, "")
                if p:
                    # Take first two path components: <root>/<dataset>/...
                    parts = p.split("/")
                    if len(parts) >= 2:
                        prefixes.add(f"{parts[0]}/{parts[1]}/")
    logger.info("Indexing %d directory prefixes: %s", len(prefixes), sorted(prefixes)[:10])
    return prefixes if prefixes else None


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tar.gz volume index for SN-VC-50M")
    parser.add_argument(
        "--tar-root",
        type=Path,
        default=Path("data/raw/SenseNova-Vision-Corpus-50M"),
        help="Directory containing the split tar.gz volumes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/snvc50m_manifest/tar_index.json"),
        help="Output index JSON path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/snvc50m_manifest/pairs_all.jsonl"),
        help="Training manifest (to determine which prefixes to index)",
    )
    parser.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Only scan the first N volumes (for testing)",
    )
    args = parser.parse_args()

    volumes = discover_volumes(args.tar_root)
    logger.info("Found %d tar.gz volumes in %s", len(volumes), args.tar_root)

    if args.max_volumes:
        volumes = volumes[: args.max_volumes]
        logger.info("Limiting to first %d volumes", len(volumes))

    prefixes = needed_prefixes(args.manifest)

    all_files: dict[str, dict] = {}
    volume_info: list[dict] = []
    t0 = time.time()

    for vol_idx, vol_path in enumerate(volumes):
        t1 = time.time()
        result = scan_volume(vol_path, prefixes)
        elapsed = time.time() - t1

        volume_info.append({
            "path": str(vol_path),
            "num_entries": result["num_entries"],
        })

        # Merge file entries (prefix volume index)
        for name, info in result["files"].items():
            all_files[name] = {"vol": vol_idx, "offset": info["offset"], "size": info["size"]}

        logger.info(
            "  [%d/%d] %s: %d entries (%.1fs)",
            vol_idx + 1, len(volumes), vol_path.name, result["num_entries"], elapsed,
        )

    total_time = time.time() - t0
    logger.info("=" * 60)
    logger.info("Index built in %.1f minutes", total_time / 60)
    logger.info("  Volumes scanned: %d", len(volumes))
    logger.info("  Files indexed: %s", f"{len(all_files):,}")

    # Write index
    args.output.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "volumes": volume_info,
        "files": all_files,
        "tar_root": str(args.tar_root),
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    logger.info("Index written → %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
