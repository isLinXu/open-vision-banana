#!/usr/bin/env python3
"""Build a file index for SN-VC-50M split tar.gz volumes (v3).

Uses ``indexed_gzip`` + ``MultiFileReader`` for fast sequential scanning
of the concatenated gzip stream across all split volumes.

Improvements over v2:
- Uses ``indexed_gzip`` (C extension) — much faster and more robust
- Incremental save every N entries
- Graceful error recovery: skips corrupted regions
- Volume filtering: skip incomplete downloads

Requires: ``indexed_gzip`` (``pip install indexed_gzip``)

Usage::

    # Full scan (all complete volumes)
    python scripts/build_tar_index_snvc50m_v3.py

    # Scan only first N volumes (for testing)
    python scripts/build_tar_index_snvc50m_v3.py --max-volumes 10

    # Custom output
    python scripts/build_tar_index_snvc50m_v3.py --output output/snvc50m_manifest/tar_index_v3.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tar_index_v3")

_TAR_BLOCK = 512


def _round_block(n: int) -> int:
    return (n + _TAR_BLOCK - 1) // _TAR_BLOCK * _TAR_BLOCK


# ── Multi-file reader (self-contained, no dependency on tar_dataset.py) ──


class MultiFileReader(io.RawIOBase):
    """Treat multiple split volumes as a single virtual byte stream."""

    def __init__(self, file_paths: list[Path]) -> None:
        self._paths = file_paths
        self._sizes = [p.stat().st_size for p in file_paths]
        self._offsets: list[int] = []
        cumulative = 0
        for s in self._sizes:
            self._offsets.append(cumulative)
            cumulative += s
        self._total_size = cumulative
        self._current_file: int = -1
        self._current_fh: io.BufferedReader | None = None
        self._current_pos: int = 0

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
        offset = max(0, min(offset, self._total_size))
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
        if self.tell() >= self._total_size:
            return b""
        remaining = size
        chunks: list[bytes] = []
        while remaining > 0 and self._current_file < len(self._paths):
            if self._current_file < 0:
                self._open_file(0)
            assert self._current_fh is not None
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

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self.close_file()
        super().close()


# ── Volume discovery ─────────────────────────────────────────────────────


def discover_volumes(
    tar_root: Path,
    max_volumes: int | None = None,
    skip_incomplete: bool = True,
) -> list[Path]:
    """Find split tar.gz volumes, optionally skipping incomplete downloads."""
    all_vols = sorted(
        f for f in tar_root.glob("SenseNova-Vision-Corpus-50M.tar.gz[0-9]*")
        if not f.name.endswith(".aria2")
    )
    if not skip_incomplete:
        return all_vols[:max_volumes] if max_volumes else all_vols

    complete = []
    skipped = 0
    expected_min = 19 * 1024**3  # 19 GB

    for vol in all_vols:
        aria2 = vol.parent / (vol.name + ".aria2")
        if aria2.exists() and vol.stat().st_size < expected_min:
            skipped += 1
            continue
        complete.append(vol)

    logger.info("Volumes: %d complete, %d skipped", len(complete), skipped)
    return complete[:max_volumes] if max_volumes else complete


# ── Scanner ──────────────────────────────────────────────────────────────


def scan_volumes(
    volumes: list[Path],
    output_path: Path,
    save_every: int = 50_000,
) -> None:
    """Scan all volumes via ``cat | gzip -dc`` pipe (pure sequential I/O).

    This avoids the seek overhead of ``indexed_gzip`` which degrades as the
    internal zran index grows.  Sequential pipe I/O keeps the disk reading
    at full throughput.
    """
    import subprocess

    vol_args = " ".join(f"'{v}'" for v in volumes)
    # Try pigz first (parallel, faster), fall back to gzip -dc
    decompressor = "pigz -dc" if _cmd_exists("pigz") else "gzip -dc"
    cmd = f"cat {vol_args} | {decompressor}"
    logger.info("Running: %s ... | %s", cmd[:120], decompressor)

    proc = subprocess.Popen(
        cmd, shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=16 * 1024 * 1024,  # 16 MB pipe buffer
    )
    fh = proc.stdout
    assert fh is not None
    reader = BufferedPipeReader(fh)

    entries: dict[str, dict] = {}
    offset = 0
    num_entries = 0
    t0 = time.time()
    last_save = 0

    try:
        while True:
            header = reader.read(_TAR_BLOCK)
            if not header or len(header) < _TAR_BLOCK:
                logger.info("End of stream at offset %.2f GB", offset / 1e9)
                break
            offset += _TAR_BLOCK

            if header == b"\x00" * _TAR_BLOCK:
                logger.info("End-of-archive at offset %.2f GB", offset / 1e9)
                break

            # Parse header
            name = header[:100].rstrip(b"\x00").decode("utf-8", errors="replace")
            if not name:
                # Still need to skip any data if size > 0
                try:
                    size_str = header[124:136].rstrip(b"\x00").rstrip(b" ")
                    size = int(size_str, 8) if size_str else 0
                except (ValueError, IndexError):
                    size = 0
                skip = _round_block(size)
                if skip > 0:
                    _skip_bytes(reader, skip)
                    offset += skip
                continue

            try:
                size_str = header[124:136].rstrip(b"\x00").rstrip(b" ")
                size = int(size_str, 8) if size_str else 0
            except (ValueError, IndexError):
                size = 0
            typeflag = chr(header[156]) if len(header) > 156 else ""

            if typeflag in ("0", "\x00", "") and size > 0:
                num_entries += 1
                entries[name] = {"offset": offset, "size": size}

                if num_entries % 10_000 == 0:
                    elapsed = time.time() - t0
                    rate = offset / elapsed / 1e9 if elapsed > 0 else 0
                    logger.info(
                        "  %s files (decompressed %.1f GB, %.1f min, %.0f MB/s)",
                        f"{num_entries:,}", offset / 1e9, elapsed / 60, rate * 1000,
                    )

                # Incremental save
                if num_entries - last_save >= save_every:
                    _save(entries, volumes, output_path, offset, num_entries)
                    last_save = num_entries

            # Skip file data (sequential read & discard)
            skip = _round_block(size)
            if skip > 0:
                _skip_bytes(reader, skip)
                offset += skip

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    except Exception as exc:
        logger.warning("Error at offset %.2f GB: %s", offset / 1e9, exc)
    finally:
        proc.terminate()
        proc.wait()
        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info("Scan done: %s files in %.1f min", f"{len(entries):,}", elapsed / 60)
        logger.info("  Final decompressed offset: %.1f GB", offset / 1e9)

    # Final save
    _save(entries, volumes, output_path, offset, num_entries, final=True)


def _cmd_exists(cmd: str) -> bool:
    """Check if a command exists on the system."""
    import shutil
    return shutil.which(cmd.split()[0]) is not None


def _skip_bytes(fh, n: int) -> None:
    """Skip n bytes from a file handle by reading in large chunks."""
    remaining = n
    while remaining > 0:
        chunk = min(remaining, 8 * 1024 * 1024)  # 8 MB
        data = fh.read(chunk)
        if not data:
            break
        remaining -= len(data)


class BufferedPipeReader:
    """Buffered reader for pipe streams.

    Reads from the underlying file in large chunks (default 16 MB) and
    serves smaller reads from the internal buffer.  This dramatically
    reduces the number of system calls when reading tar headers (512 B)
    from a pipe.
    """

    def __init__(self, fh, buf_size: int = 16 * 1024 * 1024) -> None:
        self._fh = fh
        self._buf = b""
        self._pos = 0  # Current read position within _buf
        self._total_read = 0  # Total bytes consumed

    def read(self, n: int) -> bytes:
        """Read exactly n bytes (or less at EOF)."""
        if self._pos >= len(self._buf):
            # Buffer exhausted — refill
            self._buf = self._fh.read(max(n, 16 * 1024 * 1024))
            self._pos = 0
            if not self._buf:
                return b""

        available = len(self._buf) - self._pos
        if n <= available:
            data = self._buf[self._pos : self._pos + n]
            self._pos += n
        else:
            # Need more data than what's in the buffer
            chunks = [self._buf[self._pos:]]
            need = n - available
            self._pos = len(self._buf)
            while need > 0:
                chunk = self._fh.read(max(need, 16 * 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                need -= len(chunk)
            data = b"".join(chunks)
            # Put any excess back into buffer
            if len(data) > n:
                self._buf = data[n:]
                self._pos = 0
                data = data[:n]
            else:
                self._buf = b""
                self._pos = 0

        self._total_read += len(data)
        return data

    @property
    def total_read(self) -> int:
        return self._total_read


def _save(
    entries: dict,
    volumes: list[Path],
    output_path: Path,
    decompressed_offset: int,
    num_entries: int,
    final: bool = False,
) -> None:
    """Atomically save index to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    index = {
        "volumes": [{"path": str(p), "size": p.stat().st_size} for p in volumes],
        "files": entries,
        "tar_root": str(volumes[0].parent) if volumes else "",
        "gzip_index": None,
        "_meta": {
            "decompressed_offset": decompressed_offset,
            "num_entries": num_entries,
            "complete": final,
        },
    }
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    tmp.rename(output_path)
    mb = output_path.stat().st_size / 1e6
    tag = "Final" if final else "Checkpoint"
    logger.info("  [%s] %s (%.1f MB, %s files)", tag, output_path, mb, f"{len(entries):,}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tar index for SN-VC-50M (v3)")
    parser.add_argument(
        "--tar-root", type=Path,
        default=Path("/media/linxu/thinkHub/datasets/SenseNova-Vision-Corpus-50M"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/snvc50m_manifest/tar_index_v3.json"),
    )
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument(
        "--keep-incomplete", action="store_true",
        help="Include incomplete volumes",
    )
    args = parser.parse_args()

    volumes = discover_volumes(
        args.tar_root,
        max_volumes=args.max_volumes,
        skip_incomplete=not args.keep_incomplete,
    )
    total_gb = sum(f.stat().st_size for f in volumes) / 1e9
    logger.info("Total compressed size: %.1f GB across %d volumes", total_gb, len(volumes))

    scan_volumes(volumes, args.output)


if __name__ == "__main__":
    main()
