#!/usr/bin/env python3
"""Scan all tar.gz volumes and build a file→volume mapping.

This script scans each tar.gz volume with Python's tarfile module to discover
which files are in which volume.  The resulting mapping is saved as a compact
JSON file that TarTargetDataset uses for direct volume access at training time.

This is a ONE-TIME operation.  For 8 TB of compressed data it may take 2-4 hours.
Run it once in the background; subsequent training runs reuse the mapping.

Usage:
    python scripts/build_volume_map.py \
        --tar-root data/snvc50m \
        --output output/snvc50m_manifest/volume_map.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def scan_volume(vol_path: Path) -> list[str]:
    """Return list of member names in a tar.gz volume."""
    members = []
    try:
        with tarfile.open(str(vol_path), "r:gz") as tf:
            for member in tf:
                if member.isfile():
                    members.append(member.name)
    except Exception as e:
        logger.warning("Error scanning %s: %s", vol_path, e)
    return members


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tar-root",
        type=Path,
        default=Path("data/snvc50m"),
        help="Directory containing split tar.gz volumes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/snvc50m_manifest/volume_map.json"),
        help="Output JSON file for the volume mapping",
    )
    args = parser.parse_args()

    volumes = sorted(args.tar_root.glob("SenseNova-Vision-Corpus-50M.tar.gz*"))
    logger.info("Found %d volumes in %s", len(volumes), args.tar_root)

    # file_path → volume_index
    file_to_vol: dict[str, int] = {}
    vol_stats: list[dict] = []

    t_start = time.time()
    for vol_idx, vol_path in enumerate(volumes):
        t_vol = time.time()
        members = scan_volume(vol_path)
        elapsed = time.time() - t_vol

        for name in members:
            file_to_vol[name] = vol_idx

        vol_stats.append({
            "path": str(vol_path),
            "size": vol_path.stat().st_size,
            "members": len(members),
        })

        total_elapsed = time.time() - t_start
        rate = (vol_idx + 1) / total_elapsed if total_elapsed > 0 else 0
        eta = (len(volumes) - vol_idx - 1) / rate if rate > 0 else 0
        logger.info(
            "  [%d/%d] %s: %d members (%.1fs), ETA: %.0f min",
            vol_idx + 1, len(volumes), vol_path.name, len(members),
            elapsed, eta / 60,
        )

    # Save mapping
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "volumes": vol_stats,
        "file_to_vol": file_to_vol,
        "total_files": len(file_to_vol),
        "total_volumes": len(volumes),
    }
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))

    total_time = time.time() - t_start
    logger.info(
        "Done! %d files mapped across %d volumes in %.1f min",
        len(file_to_vol), len(volumes), total_time / 60,
    )
    logger.info("Output: %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
