"""TarArchiveDataset – read training data directly from split tar.gz volumes.

Instead of extracting 7 TB of compressed archives, this module builds a
lightweight JSON index and reads images on-the-fly from the compressed
tarballs during training.

Key design decisions
--------------------
* **Multi-file virtual stream** – split tar.gz volumes are concatenated
  into a single logical byte stream using ``MultiFileReader``.
* **indexed_gzip** – provides O(1) random access into the gzip stream
  after an initial index build.
* **Sequential batch processing** – samples are sorted by tar offset so
  each batch reads sequentially from the stream.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import struct
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger("tar_dataset")

_TAR_BLOCK = 512


def _round_block(n: int) -> int:
    return (n + _TAR_BLOCK - 1) // _TAR_BLOCK * _TAR_BLOCK


# ── Multi-file reader ─────────────────────────────────────────────────────


class MultiFileReader(io.RawIOBase):
    """Treat multiple files as a single virtual file for sequential reading."""

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

        # Clamp to valid range
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


# ── Indexed tar reader ─────────────────────────────────────────────────────


class IndexedTarReader:
    """Reads files from a split tar.gz archive using indexed_gzip for random access."""

    def __init__(
        self,
        volumes: list[Path],
        gzip_index_path: Path | None = None,
    ) -> None:
        self.volumes = volumes
        self._raw = MultiFileReader(volumes)
        from indexed_gzip import IndexedGzipFile

        if gzip_index_path and gzip_index_path.is_file():
            logger.info("Loading gzip seek index from %s", gzip_index_path)
            self._gz = IndexedGzipFile(
                fileobj=self._raw,
                spacing=4 * 1024 * 1024,
            )
            self._gz.import_index(str(gzip_index_path))
        else:
            logger.info("No gzip index found, will build during first access")
            self._gz = IndexedGzipFile(
                fileobj=self._raw,
                spacing=4 * 1024 * 1024,
            )

    def read_at(self, offset: int, size: int) -> bytes:
        """Read *size* bytes starting at decompressed *offset*."""
        self._gz.seek(offset)
        return self._gz.read(size)

    def close(self) -> None:
        self._gz.close()
        self._raw.close()


# ── Dataset ───────────────────────────────────────────────────────────────


class TarArchiveDataset(Dataset):
    """PyTorch Dataset that reads images directly from tar.gz archives.

    Parameters
    ----------
    manifest_path:
        JSONL file with ``{task, input, target, split}`` records.
    tar_index_path:
        JSON index built by ``build_tar_index_snvc50m_v2.py``.
    tar_root:
        Directory containing the split tar.gz volumes.
    resolution:
        Target image resolution.
    split:
        Which split to use (``"train"`` or ``"val"``).
    tasks:
        Set of task names to include.
    inner_prefix:
        Prefix inside the tar archive (e.g. ``"SenseNova-Vision-Corpus-50M"``).
    disk_cache_dir:
        Optional disk cache for extracted images.
    max_samples:
        Limit total number of samples.
    """

    # Default prompts (used when manifest does not carry its own prompt)
    DEFAULT_PROMPTS: dict[str, tuple[str, str]] = {
        "depth": (
            "depth",
            "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map.",
        ),
        "normal": (
            "normal",
            "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
        ),
        "semantic": (
            "semantic",
            "Generate a semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
        ),
    }

    # Alias mapping: manifest task name → internal mapped name
    TASK_ALIASES: dict[str, str] = {
        "depth": "depth",
        "normal": "normal",
        "seg": "semantic",
        "segmentation": "semantic",
        "semantic": "semantic",
    }

    # Tasks whose targets are discrete masks → use NEAREST resampling
    MASK_TASKS = {"semantic"}

    def __init__(
        self,
        manifest_path: Path,
        tar_index_path: Path,
        tar_root: Path,
        resolution: int,
        split: str,
        tasks: set[str],
        inner_prefix: str = "SenseNova-Vision-Corpus-50M",
        disk_cache_dir: Path | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.resolution = resolution
        self.split = split
        self.tasks = tasks
        self.inner_prefix = inner_prefix
        self.disk_cache_dir = disk_cache_dir
        self.tar_root = tar_root

        # Load tar index
        logger.info("Loading tar index from %s", tar_index_path)
        with tar_index_path.open(encoding="utf-8") as f:
            idx = json.load(f)
        self._file_index: dict[str, dict] = idx["files"]
        self._volume_paths = [Path(v["path"]) for v in idx["volumes"]]
        self._gzip_index_path = Path(idx["gzip_index"]) if idx.get("gzip_index") else None
        logger.info(
            "  %d files indexed across %d volumes",
            len(self._file_index),
            len(self._volume_paths),
        )

        # Create indexed tar reader (lazy - will be initialized on first access)
        self._reader: IndexedTarReader | None = None

        # Load manifest
        self.samples: list[dict] = []
        self._load_manifest(manifest_path)

        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

        # Sort by offset for sequential access
        self._sort_for_sequential_access()

        logger.info(
            "TarArchiveDataset: %d samples (%s), sorted for sequential access",
            len(self.samples),
            split,
        )

    def _get_reader(self) -> IndexedTarReader:
        if self._reader is None:
            self._reader = IndexedTarReader(self._volume_paths, self._gzip_index_path)
        return self._reader

    def _load_manifest(self, manifest_path: Path) -> None:
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("split") != self.split:
                    continue
                raw_task = rec.get("task", "")
                # Resolve task alias → mapped name
                mapped = self.TASK_ALIASES.get(raw_task, raw_task)
                # Skip tasks not requested
                if mapped not in self.tasks:
                    continue
                rec["mapped_task"] = mapped
                # Prefer prompt from manifest; fall back to default
                default_info = self.DEFAULT_PROMPTS.get(raw_task)
                if not rec.get("prompt") and default_info:
                    rec["prompt"] = default_info[1]
                if not rec.get("prompt"):
                    rec["prompt"] = f"Generate a {mapped} image."
                self.samples.append(rec)

    def _sort_for_sequential_access(self) -> None:
        def sort_key(sample: dict) -> int:
            inp = self._resolve_tar_path(sample.get("input", ""))
            info = self._file_index.get(inp)
            if info:
                return info["offset"]
            tgt = self._resolve_tar_path(sample.get("target", ""))
            info = self._file_index.get(tgt)
            if info:
                return info["offset"]
            return 2**63  # Unknown files go last

        self.samples.sort(key=sort_key)

    def _resolve_tar_path(self, rel_path: str) -> str:
        if self.inner_prefix:
            return f"{self.inner_prefix}/{rel_path}"
        return rel_path

    def _read_from_tar(self, tar_path: str) -> bytes | None:
        info = self._file_index.get(tar_path)
        if info is None:
            return None
        reader = self._get_reader()
        try:
            return reader.read_at(info["offset"], info["size"])
        except Exception as e:
            logger.warning("Failed to read %s: %s", tar_path, e)
            return None

    def _load_image_bytes(self, tar_path: str) -> bytes | None:
        if self.disk_cache_dir:
            cache_path = self.disk_cache_dir / tar_path.replace("/", os.sep)
            if cache_path.is_file():
                return cache_path.read_bytes()

        data = self._read_from_tar(tar_path)

        if data and self.disk_cache_dir:
            cache_path = self.disk_cache_dir / tar_path.replace("/", os.sep)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(data)

        return data

    def _bytes_to_pil(self, data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, idx: int) -> dict[str, Any]:
        rec = self.samples[idx]
        task = rec["mapped_task"]
        prompt = rec["prompt"]

        inp_tar = self._resolve_tar_path(rec.get("input", ""))
        inp_data = self._load_image_bytes(inp_tar)
        if inp_data is None:
            logger.warning("Cannot load input: %s (idx=%d)", inp_tar, idx)
            inp_img = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        else:
            inp_img = self._bytes_to_pil(inp_data)

        tgt_tar = self._resolve_tar_path(rec.get("target", ""))
        tgt_data = self._load_image_bytes(tgt_tar)
        if tgt_data is None:
            logger.warning("Cannot load target: %s (idx=%d)", tgt_tar, idx)
            tgt_img = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        else:
            tgt_img = self._bytes_to_pil(tgt_data)

        inp_img = inp_img.convert("RGB").resize(
            (self.resolution, self.resolution), Image.Resampling.LANCZOS
        )
        input_tensor = torch.from_numpy(
            np.asarray(inp_img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)

        target_resampling = (
            Image.Resampling.NEAREST if task in self.MASK_TASKS else Image.Resampling.LANCZOS
        )
        tgt_img = tgt_img.convert("RGB").resize(
            (self.resolution, self.resolution), target_resampling
        )
        target_tensor = torch.from_numpy(
            np.asarray(tgt_img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)

        return {
            "prompt": prompt,
            "image": input_tensor * 2.0 - 1.0,
            "image_range": "-1_1",
            "target_rgb": target_tensor,
            "target_rgb_range": "0_1",
            "task": task,
            "flow_source": "input",
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_err = None
        for attempt in range(3):
            try:
                return self._load_sample(idx)
            except Exception as e:
                last_err = e
                idx = (idx + 1) % len(self.samples)
        logger.error("Failed to load sample after 3 attempts: %s", last_err)
        return {
            "prompt": "Generate an image.",
            "image": torch.zeros(3, self.resolution, self.resolution),
            "image_range": "-1_1",
            "target_rgb": torch.zeros(3, self.resolution, self.resolution),
            "target_rgb_range": "0_1",
            "task": "depth",
            "flow_source": "input",
        }

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __del__(self) -> None:
        self.close()
