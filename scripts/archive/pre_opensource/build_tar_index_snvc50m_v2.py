#!/usr/bin/env python3
"""Build a file index for SN-VC-50M split tar.gz volumes (v2).

The split tar.gz volumes form a **single gzip stream** when concatenated.
This script builds a JSON index by scanning the concatenated decompressed
stream, recording the byte offset of every tar entry.

Uses ``indexed_gzip`` for O(1) random access into the gzip stream after
the initial index build.

Index format (JSON)::

    {
      "volumes": [
        {"path": "/abs/path/to/...tar.gz000", "size": 20000000000},
        ...
      ],
      "files": {
        "SenseNova-Vision-Corpus-50M/coco2017/train2017/000009.jpg": {
          "offset": 1024, "size": 204800
        },
        ...
      },
      "gzip_index": "<path to .gzidx file>"
    }

Usage::

    python scripts/build_tar_index_snvc50m_v2.py \\
        --tar-root /media/linxu/thinkHub/datasets/SenseNova-Vision-Corpus-50M \\
        --output   output/snvc50m_manifest/tar_index.json
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
logger = logging.getLogger("tar_index_v2")

_TAR_BLOCK = 512


def _round_block(n: int) -> int:
    return (n + _TAR_BLOCK - 1) // _TAR_BLOCK * _TAR_BLOCK


class MultiFileReader(io.RawIOBase):
    """Treat multiple files as a single virtual file for sequential reading.

    This is used to concatenate all split tar.gz volumes into a single
    byte stream without copying data.
    """

    def __init__(self, file_paths: list[Path]) -> None:
        self._paths = file_paths
        self._sizes = [p.stat().st_size for p in file_paths]
        self._offsets = []
        cumulative = 0
        for s in self._sizes:
            self._offsets.append(cumulative)
            cumulative += s
        self._total_size = cumulative
        self._current_file: int = -1
        self._current_fh: io.BufferedReader | None = None
        self._current_pos: int = 0  # Position within current file

    def _open_file(self, idx: int) -> None:
        if self._current_file == idx and self._current_fh is not None:
            return
        self.close_file()
        self._current_fh = self._paths[idx].open("rb")
        self._current_file = idx
        self._current_pos = 0

    def close_file(self) -> None:
        if self._current_fh is not None:
            self._current_fh.close()
            self._current_fh = None
            self._current_file = -1

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        if self._current_file < 0:
            return 0
        return self._offsets[self._current_file] + self._current_pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            offset = self.tell() + offset
        elif whence == io.SEEK_END:
            offset = self._total_size + offset

        # Clamp to valid range
        offset = max(0, min(offset, self._total_size))

        # Find which file this offset is in
        for i in range(len(self._offsets) - 1, -1, -1):
            if offset >= self._offsets[i]:
                file_offset = offset - self._offsets[i]
                if file_offset > self._sizes[i]:
                    continue
                self._open_file(i)
                assert self._current_fh is not None
                self._current_fh.seek(file_offset)
                self._current_pos = file_offset
                return offset
        # If we get here, offset is 0
        self._open_file(0)
        assert self._current_fh is not None
        self._current_fh.seek(0)
        self._current_pos = 0
        return 0

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            size = self._total_size - self.tell()

        # If already at end, return empty
        if self.tell() >= self._total_size:
            return b""

        remaining = size
        chunks = []
        while remaining > 0 and self._current_file < len(self._paths):
            if self._current_file < 0:
                self._open_file(0)
            assert self._current_fh is not None

            available = self._sizes[self._current_file] - self._current_pos
            to_read = min(remaining, available)
            if to_read <= 0:
                # Move to next file
                self._open_file(self._current_file + 1)
                continue

            data = self._current_fh.read(to_read)
            if not data:
                break
            chunks.append(data)
            self._current_pos += len(data)
            remaining -= len(data)

        return b"".join(chunks)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self.close_file()
        super().close()


def scan_concatenated_tar(
    tar_root: Path,
    output_path: Path,
    gzip_index_path: Path | None = None,
) -> None:
    """Scan all split tar.gz volumes as a single stream and build file index.

    Uses plain ``gzip`` for fast sequential scanning.  The resulting index
    can later be loaded by ``TarArchiveDataset`` which uses ``indexed_gzip``
    for random access during training.
    """

    # Find all volumes in order
    volumes = sorted(
        f for f in tar_root.glob("SenseNova-Vision-Corpus-50M.tar.gz[0-9]*")
        if not f.name.endswith(".aria2")
    )
    logger.info("Found %d split volumes", len(volumes))
    total_size = sum(f.stat().st_size for f in volumes)
    logger.info("Total compressed size: %.1f GB", total_size / 1e9)

    # Create multi-file reader
    raw = MultiFileReader(volumes)

    # Use plain gzip for fast sequential scanning
    gz = gzip.GzipFile(fileobj=raw)

    # Scan tar entries
    entries: dict[str, dict] = {}
    offset = 0
    num_entries = 0
    t0 = time.time()
    report_interval = 50_000  # Report every 50K entries

    while True:
        try:
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
            try:
                size_str = header_block[124:136].rstrip(b"\x00").rstrip(b" ")
                size = int(size_str, 8) if size_str else 0
            except (ValueError, IndexError):
                size = 0

            typeflag = chr(header_block[156])

            # Only record regular files with non-zero size
            if typeflag in ("0", "\x00", "") and size > 0:
                num_entries += 1
                entries[name] = {"offset": offset, "size": size}

                if num_entries % report_interval == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        "  Scanned %s entries (offset %.1f GB, %.1f min)",
                        f"{num_entries:,}", offset / 1e9, elapsed / 60,
                    )

            # Skip file data by reading it (plain gzip is sequential)
            skip = _round_block(size)
            if skip > 0:
                # Read in chunks to avoid huge memory allocation
                remaining = skip
                while remaining > 0:
                    chunk = min(remaining, 8 * 1024 * 1024)
                    data = gz.read(chunk)
                    if not data:
                        break
                    offset += len(data)
                    remaining -= len(data)

        except (EOFError, OSError) as exc:
            logger.warning("Scan stopped at offset %.1f GB: %s", offset / 1e9, exc)
            break

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Index built in %.1f minutes", elapsed / 60)
    logger.info("  Files indexed: %s", f"{len(entries):,}")

    # Write file index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "volumes": [{"path": str(p), "size": p.stat().st_size} for p in volumes],
        "files": entries,
        "tar_root": str(tar_root),
        "gzip_index": None,  # Will be built lazily during training
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    logger.info("Index written → %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tar.gz volume index for SN-VC-50M (v2)")
    parser.add_argument(
        "--tar-root",
        type=Path,
        default=Path("/media/linxu/thinkHub/datasets/SenseNova-Vision-Corpus-50M"),
        help="Directory containing the split tar.gz volumes",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/snvc50m_manifest/tar_index.json"),
        help="Output index JSON path",
    )
    args = parser.parse_args()

    scan_concatenated_tar(args.tar_root, args.output)


if __name__ == "__main__":
    main()
