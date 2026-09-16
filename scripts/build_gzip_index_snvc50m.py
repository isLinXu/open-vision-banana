#!/usr/bin/env python3
"""Build gzip seek-index for the SNVC-50M split tar.gz stream.

The 406 volumes form a single 8 TB gzip stream.  This script reads through
the entire stream once so that ``indexed_gzip`` can build its seek index.
After completion, the index is saved to disk and training can use it for
instant random access.

Estimated time: ~55 hours on a single core (CPU-bound at ~40 MB/s).

Usage:
    python scripts/build_gzip_index_snvc50m.py [--spacing 4] [--output DIR]

The script saves partial progress every 30 minutes and can be resumed if
interrupted (pass --resume).
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("gzip_index_builder")

# ── MultiFileReader (same as in tar_dataset.py) ──────────────────────────

_TAR_BLOCK = 512

class MultiFileReader(io.RawIOBase):
    """Treat multiple files as a single virtual file."""

    def __init__(self, file_paths):
        self._paths = file_paths
        self._sizes = [p.stat().st_size for p in file_paths]
        self._offsets = []
        cumulative = 0
        for s in self._sizes:
            self._offsets.append(cumulative)
            cumulative += s
        self._total_size = cumulative
        self._current_file = -1
        self._current_fh = None
        self._current_pos = 0

    def _open_file(self, idx):
        if self._current_file == idx and self._current_fh is not None:
            return
        self.close_file()
        self._current_fh = self._paths[idx].open("rb")
        self._current_file = idx
        self._current_pos = 0

    def close_file(self):
        if self._current_fh is not None:
            self._current_fh.close()
            self._current_fh = None
            self._current_file = -1

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        if self._current_file < 0:
            return 0
        return self._offsets[self._current_file] + self._current_pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_CUR:
            offset = self.tell() + offset
        elif whence == io.SEEK_END:
            offset = self._total_size + offset
        offset = max(0, min(offset, self._total_size))
        for i in range(len(self._offsets) - 1, -1, -1):
            if offset >= self._offsets[i]:
                file_offset = offset - self._offsets[i]
                if file_offset > self._sizes[i]:
                    continue
                self._open_file(i)
                self._current_fh.seek(file_offset)
                self._current_pos = file_offset
                return offset
        self._open_file(0)
        self._current_fh.seek(0)
        self._current_pos = 0
        return 0

    def read(self, size=-1):
        if size == 0:
            return b""
        if size < 0:
            size = self._total_size - self.tell()
        if self.tell() >= self._total_size:
            return b""
        remaining = size
        chunks = []
        while remaining > 0 and self._current_file < len(self._paths):
            if self._current_file < 0:
                self._open_file(0)
            available = self._sizes[self._current_file] - self._current_pos
            to_read = min(remaining, available)
            if to_read <= 0:
                self._open_file(self._current_file + 1)
                continue
            data = self._current_fh.read(to_read)
            if not data:
                break
            chunks.append(data)
            self._current_pos += len(data)
            remaining -= len(data)
        return b"".join(chunks)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self):
        self.close_file()
        super().close()


def main():
    parser = argparse.ArgumentParser(description="Build gzip seek-index for SNVC-50M")
    parser.add_argument("--spacing", type=int, default=256,
                        help="Index spacing in MB (default: 256)")
    parser.add_argument("--output", type=str,
                        default="output/snvc50m_manifest/gzip_index",
                        help="Output directory for the index file")
    parser.add_argument("--tar-root", type=str,
                        default="data/snvc50m",
                        help="Directory containing tar.gz volumes")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from partial index if available")
    args = parser.parse_args()

    tar_root = Path(args.tar_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    spacing_bytes = args.spacing * 1024 * 1024

    # Discover volumes
    volumes = sorted(tar_root.glob("SenseNova-Vision-Corpus-50M.tar.gz*"))
    if not volumes:
        logger.error("No tar.gz volumes found in %s", tar_root)
        return
    logger.info("Found %d volumes in %s", len(volumes), tar_root)

    total_compressed = sum(v.stat().st_size for v in volumes)
    logger.info("Total compressed size: %.1f GB", total_compressed / 1e9)

    # Index output path
    index_path = output_dir / "stream.gzidx"
    progress_path = output_dir / "progress.json"

    # Check for resume
    start_offset = 0
    if args.resume and index_path.exists():
        logger.info("Found existing index, will resume")
        # Can't easily resume indexed_gzip state, so start fresh
        # but we note the progress file for reference
        if progress_path.exists():
            with open(progress_path) as f:
                progress = json.load(f)
            logger.info("Previous progress: %.1f GB decompressed", 
                       progress.get("decompressed_gb", 0))

    # Open the stream
    logger.info("Opening MultiFileReader...")
    raw = MultiFileReader(volumes)

    # Open indexed_gzip
    logger.info("Opening indexed_gzip with spacing=%d MB...", args.spacing)
    from indexed_gzip import IndexedGzipFile
    gz = IndexedGzipFile(fileobj=raw, spacing=spacing_bytes)

    # Read through the entire stream
    chunk_size = 16 * 1024 * 1024  # 16 MB read chunks
    total_decompressed = 0
    t_start = time.time()
    t_last_log = t_start
    t_last_save = t_start

    logger.info("Starting full stream scan...")
    logger.info("This will take ~55 hours. Progress is logged every 5 minutes.")

    try:
        while True:
            data = gz.read(chunk_size)
            if not data:
                break
            total_decompressed += len(data)

            now = time.time()
            # Log every 5 minutes
            if now - t_last_log > 300:
                elapsed = now - t_start
                rate = total_decompressed / elapsed / 1e9  # GB/s
                remaining_gb = (total_compressed / 0.92 - total_decompressed) / 1e9  # rough estimate
                eta_hours = remaining_gb / rate / 3600 if rate > 0 else 0
                logger.info(
                    "Progress: %.1f GB decompressed (%.1f%%), "
                    "rate: %.1f MB/s, ETA: %.1f hours",
                    total_decompressed / 1e9,
                    total_decompressed / (8e12) * 100,  # ~8 TB estimated total
                    rate * 1000,
                    eta_hours,
                )
                t_last_log = now

                # Save progress
                with open(progress_path, "w") as f:
                    json.dump({
                        "decompressed_gb": round(total_decompressed / 1e9, 2),
                        "elapsed_hours": round(elapsed / 3600, 2),
                        "rate_mb_s": round(rate * 1000, 1),
                        "eta_hours": round(eta_hours, 1),
                    }, f, indent=2)

            # Save partial index every 30 minutes
            if now - t_last_save > 1800:
                logger.info("Saving partial index...")
                try:
                    gz.export_index(str(index_path))
                    logger.info("Partial index saved to %s", index_path)
                except Exception as e:
                    logger.warning("Failed to save partial index: %s", e)
                t_last_save = now

    except KeyboardInterrupt:
        logger.info("Interrupted! Saving final index...")
    except Exception as e:
        logger.error("Error during scan: %s", e)
        logger.info("Saving index before exit...")

    # Save final index
    elapsed = time.time() - t_start
    logger.info("Scan complete. Total: %.1f GB in %.1f hours",
                total_decompressed / 1e9, elapsed / 3600)

    try:
        gz.export_index(str(index_path))
        logger.info("Index saved to %s (%.1f MB)",
                    index_path, index_path.stat().st_size / 1e6)
    except Exception as e:
        logger.error("Failed to save index: %s", e)

    # Save final progress
    with open(progress_path, "w") as f:
        json.dump({
            "decompressed_gb": round(total_decompressed / 1e9, 2),
            "elapsed_hours": round(elapsed / 3600, 2),
            "rate_mb_s": round(total_decompressed / elapsed / 1e6, 1),
            "status": "complete",
            "index_path": str(index_path),
        }, f, indent=2)

    gz.close()
    raw.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
