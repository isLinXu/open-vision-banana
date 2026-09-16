#!/usr/bin/env python3
"""Build per-task tar indexes from the full SNVC-50M tar index.

The full tar index (tar_index_v3.json, ~2.4 GB) contains all 20M files.
This script splits it into small per-task indexes for efficient training.

Task classification is based on directory patterns in the tar paths:
  - depth:  */depth/*
  - normal: */normal/*
  - seg:    */genseg/*

Usage:
    python scripts/build_task_tar_index.py \
        --tar-index output/snvc50m_manifest/tar_index_v3.json \
        --output-dir output/snvc50m_manifest/task_indexes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory patterns → task name
TASK_PATTERNS = {
    "depth": ["/depth/"],
    "normal": ["/normal/"],
    "seg": ["/genseg/"],
}


def classify_task(file_path: str) -> str | None:
    """Classify a tar file path into a task based on directory patterns."""
    for task, patterns in TASK_PATTERNS.items():
        for pat in patterns:
            if pat in file_path:
                return task
    return None


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tar-index",
        type=Path,
        default=Path("output/snvc50m_manifest/tar_index_v3.json"),
        help="Full tar index JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/snvc50m_manifest/task_indexes"),
        help="Output directory for per-task indexes",
    )
    args = parser.parse_args()

    # Load full tar index
    logger.info("Loading full tar index from %s ...", args.tar_index)
    with args.tar_index.open(encoding="utf-8") as f:
        full_idx = json.load(f)

    files = full_idx["files"]
    volumes = full_idx["volumes"]
    logger.info("  %d files, %d volumes", len(files), len(volumes))

    # Classify files by task
    task_files: dict[str, dict] = {task: {} for task in TASK_PATTERNS}
    skipped = 0
    for path, info in files.items():
        task = classify_task(path)
        if task is not None:
            task_files[task][path] = info
        else:
            skipped += 1

    logger.info("Task classification:")
    for task in TASK_PATTERNS:
        count = len(task_files[task])
        logger.info("  %s: %d files", task, count)
    logger.info("  skipped (recon/gcg/other): %d files", skipped)

    # Save per-task indexes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task, task_entries in task_files.items():
        if not task_entries:
            logger.warning("No files for task '%s', skipping", task)
            continue
        index = {
            "task": task,
            "volumes": volumes,
            "files": task_entries,
            "count": len(task_entries),
        }
        out_path = args.output_dir / f"tar_index_{task}.json"
        logger.info("Writing %s (%d entries, %.1f MB) ...",
                     out_path, len(task_entries),
                     out_path.stat().st_size / 1e6 if out_path.exists() else 0)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, separators=(",", ":"))
        size_mb = out_path.stat().st_size / 1e6
        logger.info("  → %s: %.1f MB", out_path, size_mb)

    logger.info("Done!")


if __name__ == "__main__":
    main()
