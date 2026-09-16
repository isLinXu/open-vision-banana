#!/usr/bin/env python3
"""Build file→volume mapping using ISIZE estimation + tar index offsets.

Instead of scanning all 8 TB of compressed data, we:
1. Read ISIZE from each volume's gzip footer (last 4 bytes)
2. Estimate actual decompressed sizes by distributing the known total
   proportionally to compressed sizes
3. Compute cumulative offsets
4. Map each file (from tar index) to its volume via binary search
5. Save the mapping for use by TarTargetDataset

This runs in SECONDS instead of hours.

Usage:
    python scripts/build_volume_map_fast.py
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import struct
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
    )
    parser.add_argument(
        "--tar-index",
        type=Path,
        default=Path("output/snvc50m_manifest/tar_index_v3.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/snvc50m_manifest/volume_map.json"),
    )
    args = parser.parse_args()

    # Step 1: Read ISIZE from each volume
    volumes = sorted(args.tar_root.glob("SenseNova-Vision-Corpus-50M.tar.gz*"))
    logger.info("Found %d volumes", len(volumes))

    vol_isizes = []
    vol_csizes = []
    for vol in volumes:
        with open(vol, "rb") as f:
            f.seek(-4, 2)
            isize = struct.unpack("<I", f.read(4))[0]
            vol_isizes.append(isize)
            vol_csizes.append(vol.stat().st_size)

    total_csize = sum(vol_csizes)
    sum_isize = sum(vol_isizes)
    logger.info("Total compressed: %.1f GB", total_csize / 1e9)
    logger.info("Sum ISIZE (mod 2^32): %.1f GB", sum_isize / 1e9)

    # Step 2: Load tar index to get total decompressed size
    logger.info("Loading tar index to get total decompressed size...")
    with args.tar_index.open() as f:
        idx = json.load(f)

    files = idx["files"]
    max_end = 0
    for path, info in files.items():
        end = info["offset"] + info["size"]
        if end > max_end:
            max_end = end

    total_dsize = max_end  # Total decompressed size from tar index
    logger.info("Total decompressed (from tar index): %.1f GB", total_dsize / 1e9)

    # Step 3: Estimate actual decompressed sizes
    # Each volume's actual dsize = isize_i + k_i * 2^32
    # We distribute excess blocks proportionally to compressed size
    excess = total_dsize - sum_isize
    total_excess_blocks = excess / (2**32)  # ~1432 blocks
    logger.info("Excess: %.1f GB = %.1f blocks of 4GB", excess / 1e9, total_excess_blocks)

    vol_dsizes = []
    for i in range(len(volumes)):
        frac = vol_csizes[i] / total_csize
        k_i = total_excess_blocks * frac
        actual_dsize = vol_isizes[i] + k_i * (2**32)
        vol_dsizes.append(actual_dsize)

    # Normalize to match total exactly
    current_total = sum(vol_dsizes)
    scale = total_dsize / current_total
    vol_dsizes = [d * scale for d in vol_dsizes]

    # Step 4: Compute cumulative offsets
    cum_offsets = [0.0]
    for ds in vol_dsizes:
        cum_offsets.append(cum_offsets[-1] + ds)

    logger.info("Cumulative offsets computed. Last: %.1f GB", cum_offsets[-1] / 1e9)

    # Step 5: Map each file to its volume
    logger.info("Mapping %d files to volumes...", len(files))
    file_to_vol: dict[str, int] = {}
    for path, info in files.items():
        offset = info["offset"]
        vol_idx = bisect.bisect_right(cum_offsets, offset) - 1
        vol_idx = max(0, min(vol_idx, len(volumes) - 1))
        file_to_vol[path] = vol_idx

    # Step 6: Save mapping
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vol_info = []
    for i, vol in enumerate(volumes):
        vol_info.append({
            "path": str(vol),
            "compressed_size": vol_csizes[i],
            "decompressed_size_est": vol_dsizes[i],
            "cum_offset_start": cum_offsets[i],
            "cum_offset_end": cum_offsets[i + 1],
        })

    result = {
        "volumes": vol_info,
        "file_to_vol": file_to_vol,
        "total_files": len(file_to_vol),
        "total_volumes": len(volumes),
        "total_decompressed": total_dsize,
        "method": "isize_estimation",
    }
    with args.output.open("w") as f:
        json.dump(result, f, separators=(",", ":"))

    logger.info("Saved mapping to %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)

    # Verify: check distribution
    from collections import Counter
    vol_counts = Counter(file_to_vol.values())
    logger.info("Files per volume: min=%d, max=%d, avg=%.0f",
                min(vol_counts.values()), max(vol_counts.values()),
                sum(vol_counts.values()) / len(vol_counts))


if __name__ == "__main__":
    main()
