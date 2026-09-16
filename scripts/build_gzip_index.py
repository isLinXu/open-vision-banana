#!/usr/bin/env python3
"""Pre-build gzip seek index for the concatenated SNVC-50M tar stream.

This enables O(1) random access into the 8 TB compressed archive using
indexed_gzip.  The index is built once by sequentially scanning the stream,
then saved to disk.  Subsequent training runs load the index instantly.

Expected time: 1-3 hours for 8 TB (one-time cost).

Usage:
    python scripts/build_gzip_index.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    tar_root = Path("data/snvc50m")
    tar_index_path = Path("output/snvc50m_manifest/tar_index_v3.json")
    output_index = Path("output/snvc50m_manifest/gzip_seek_index.gzidx")

    # Load volume paths
    logger.info("Loading tar index...")
    with tar_index_path.open() as f:
        idx = json.load(f)
    volumes = [Path(v["path"]) for v in idx["volumes"]]
    logger.info("  %d volumes", len(volumes))

    # Import here to avoid import-time dependency
    from indexed_gzip import IndexedGzipFile
    from open_vision_banana.data.tar_dataset import MultiFileReader

    # Open concatenated stream
    logger.info("Opening concatenated stream...")
    raw = MultiFileReader(volumes)

    # Create IndexedGzipFile with 4 MB spacing
    logger.info("Creating IndexedGzipFile (spacing=4MB)...")
    igz = IndexedGzipFile(fileobj=raw, spacing=4 * 1024 * 1024)

    # Build full index by scanning the stream
    logger.info("Building full gzip seek index (this will take 1-3 hours)...")
    t0 = time.time()
    igz.build_full_index()
    elapsed = time.time() - t0
    logger.info("  Index built in %.1f min", elapsed / 60)

    # Save index
    output_index.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving index to %s...", output_index)
    igz.export_index(str(output_index))
    size_mb = output_index.stat().st_size / 1e6
    logger.info("  Saved: %.1f MB", size_mb)

    # Update tar index to reference the gzip index
    idx["gzip_index"] = str(output_index)
    with tar_index_path.open("w") as f:
        json.dump(idx, f, separators=(",", ":"))
    logger.info("Updated tar index with gzip_index path")

    igz.close()
    raw.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
