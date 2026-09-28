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


# ── Indexed tar reader (uses pre-built gzip index) ───────────────────────


class IndexedTarReader:
    """Read files from a split tar.gz archive using indexed_gzip with a pre-built index.

    Requires a gzip seek index (``.gzidx`` file) produced by
    ``scripts/build_gzip_index_snvc50m.py``.  With the index, random access
    is O(1) – no sequential decompression needed.
    """

    def __init__(self, volumes: list[Path], gzip_index_path: Path) -> None:
        self.volumes = volumes
        self._raw = MultiFileReader(volumes)
        from indexed_gzip import IndexedGzipFile

        abs_index = gzip_index_path.resolve()
        logger.info("Loading gzip seek index from %s (resolved: %s, size: %d MB, %d volumes)",
                    gzip_index_path, abs_index,
                    abs_index.stat().st_size // (1024 * 1024) if abs_index.exists() else -1,
                    len(volumes))
        # Use default spacing (let indexed_gzip auto-detect from the imported index)
        self._gz = IndexedGzipFile(fileobj=self._raw)
        logger.info("IndexedGzipFile created, calling import_index...")
        self._gz.import_index(str(abs_index))
        logger.info("Gzip index loaded successfully")

    def read_at(self, offset: int, size: int) -> bytes:
        """Read *size* bytes at decompressed *offset* (random access)."""
        self._gz.seek(offset)
        return self._gz.read(size)

    def close(self) -> None:
        self._gz.close()
        self._raw.close()


# ── Sequential gzip reader (no index needed) ─────────────────────────────


class SequentialTarReader:
    """Read files from a split tar.gz archive using sequential gzip reading.

    Instead of indexed_gzip (which requires a full index build), this reader
    decompresses sequentially through the gzip stream.  Files MUST be accessed
    in ascending offset order for efficiency – the reader simply skips forward
    through any intervening bytes.

    Thread-safe: uses a lock so multiple DataLoader workers can share one
    reader (though sequential access is most efficient with a single worker).
    """

    def __init__(self, volumes: list[Path]) -> None:
        self._raw = MultiFileReader(volumes)
        self._gz = gzip.GzipFile(fileobj=self._raw, mode="rb")
        self._pos: int = 0  # current decompressed stream position
        self._lock = __import__("threading").Lock()

    def read_at(self, offset: int, size: int) -> bytes:
        """Read *size* bytes at decompressed *offset* (sequential skip)."""
        with self._lock:
            if offset < self._pos:
                # Cannot seek backward – would need to restart the stream.
                # This should not happen if files are sorted by offset.
                raise RuntimeError(
                    f"SequentialTarReader: cannot go backward from "
                    f"pos={self._pos} to offset={offset}. "
                    f"Ensure files are sorted by offset."
                )
            skip = offset - self._pos
            if skip > 0:
                # Skip forward by reading and discarding
                remaining = skip
                while remaining > 0:
                    chunk = min(remaining, 4 * 1024 * 1024)  # 4 MB
                    data = self._gz.read(chunk)
                    if not data:
                        raise EOFError(
                            f"Unexpected end of gzip stream at pos={self._pos}, "
                            f"trying to skip to offset={offset}"
                        )
                    remaining -= len(data)
                    self._pos += len(data)
            # Now at the right position – read the file
            buf = b""
            remaining = size
            while remaining > 0:
                data = self._gz.read(min(remaining, 4 * 1024 * 1024))
                if not data:
                    raise EOFError(
                        f"Unexpected end of gzip stream at pos={self._pos}, "
                        f"reading {size} bytes"
                    )
                buf += data
                self._pos += len(data)
                remaining -= len(data)
            return buf

    def reset(self) -> None:
        """Reset stream to the beginning (for new epoch)."""
        with self._lock:
            self._gz.close()
            self._raw.seek(0)
            self._gz = gzip.GzipFile(fileobj=self._raw, mode="rb")
            self._pos = 0

    def close(self) -> None:
        with self._lock:
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

        # Create sequential tar reader (lazy - will be initialized on first access)
        self._reader: SequentialTarReader | None = None

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

    def _get_reader(self) -> SequentialTarReader:
        if self._reader is None:
            self._reader = SequentialTarReader(self._volume_paths)
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


# ── Target-only dataset (no input RGB images needed) ─────────────────────


class TarTargetDataset(Dataset):
    """PyTorch Dataset that reads ONLY target images from tar.gz archives.

    Unlike ``TarArchiveDataset`` which requires both input and target images
    (for img2img flow matching), this dataset reads only target images and
    trains with noise→target flow matching.  This is needed because the
    SNVC-50M tar volumes contain only supervision targets (depth maps,
    normal maps, segmentation masks) without the original RGB inputs.

    The dataset is built from a *task-specific tar index* produced by
    ``scripts/build_task_tar_index.py``, which already filters files by
    task directory pattern.

    Parameters
    ----------
    tar_index_path:
        Path to a task-specific tar index JSON (contains ``task``, ``files``,
        ``volumes`` keys).
    tar_root:
        Directory containing the split tar.gz volumes.
    task:
        Task name (``"depth"``, ``"normal"``, or ``"seg"``).
    resolution:
        Target image resolution.
    inner_prefix:
        Prefix inside the tar archive (e.g. ``"SenseNova-Vision-Corpus-50M"``).
    disk_cache_dir:
        Optional disk cache for extracted images.
    max_samples:
        Limit total number of samples.
    seed:
        Random seed for epoch-based shuffling.
    """

    DEFAULT_PROMPTS: dict[str, str] = {
        "depth": (
            "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map."
        ),
        "normal": (
            "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry."
        ),
        "seg": (
            "Generate a semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries."
        ),
    }

    MASK_TASKS = {"seg"}

    def __init__(
        self,
        tar_index_path: Path,
        tar_root: Path,
        task: str,
        resolution: int = 512,
        inner_prefix: str = "SenseNova-Vision-Corpus-50M",
        disk_cache_dir: Path | None = None,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.task = task
        self.dir_name = f"tar_{task}"  # For logging compatibility
        self.resolution = resolution
        self.inner_prefix = inner_prefix
        self.disk_cache_dir = disk_cache_dir
        self.tar_root = tar_root
        self.seed = seed
        self.epoch = 0

        # Load task-specific tar index
        logger.info("Loading task tar index from %s", tar_index_path)
        with tar_index_path.open(encoding="utf-8") as f:
            idx = json.load(f)
        self._file_index: dict[str, dict] = idx["files"]
        self._volume_paths = [Path(v["path"]) for v in idx["volumes"]]
        self._gzip_index_path = Path(idx["gzip_index"]) if idx.get("gzip_index") else None

        # Build ordered file list (sorted by tar offset for sequential access)
        self._file_list: list[tuple[str, int, int]] = sorted(
            [(path, info["offset"], info["size"]) for path, info in self._file_index.items()],
            key=lambda x: x[1],
        )
        if max_samples and len(self._file_list) > max_samples:
            self._file_list = self._file_list[:max_samples]

        # Lazy reader (will be initialized on first access)
        self._reader: IndexedTarReader | SequentialTarReader | None = None

        # Check for pre-built gzip index
        self._gzip_index_path: Path | None = None
        default_gzip_index = Path("output/snvc50m_manifest/gzip_index/stream.gzidx")
        if default_gzip_index.is_file():
            self._gzip_index_path = default_gzip_index
            logger.info("Found pre-built gzip index: %s", default_gzip_index)

        logger.info(
            "TarTargetDataset [%s]: %d files across %d volumes (%s mode)",
            task, len(self._file_list), len(self._volume_paths),
            "indexed" if self._gzip_index_path else "sequential",
        )

    def _get_reader(self) -> IndexedTarReader | SequentialTarReader:
        if self._reader is None:
            if self._gzip_index_path and self._gzip_index_path.is_file():
                # The pre-built gzip index was created from ALL volumes in tar_root.
                # Use all volumes (not just self._volume_paths which may be a subset)
                # so that the gzip stream matches the index.
                all_volumes = self._get_all_volumes()
                self._reader = IndexedTarReader(all_volumes, self._gzip_index_path)
            else:
                self._reader = SequentialTarReader(self._volume_paths)
        return self._reader

    def _get_all_volumes(self) -> list[Path]:
        """Get all tar.gz volumes from tar_root (for gzip index compatibility)."""
        if not self.tar_root:
            logger.warning("_get_all_volumes: tar_root is None, using %d task volumes", len(self._volume_paths))
            return self._volume_paths
        all_vols = sorted(self.tar_root.glob("*.tar.gz*"))
        logger.info("_get_all_volumes: tar_root=%s, found %d volumes", self.tar_root, len(all_vols))
        if all_vols:
            return all_vols
        return self._volume_paths

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
        return len(self._file_list)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _load_sample(self, idx: int) -> dict[str, Any]:
        tar_path, _offset, _size = self._file_list[idx]
        task = self.task
        prompt = self.DEFAULT_PROMPTS.get(task, f"Generate a {task} image.")

        tgt_data = self._load_image_bytes(tar_path)
        if tgt_data is None:
            logger.warning("Cannot load target: %s (idx=%d)", tar_path, idx)
            tgt_img = Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        else:
            tgt_img = self._bytes_to_pil(tgt_data)

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
            "target_rgb": target_tensor,
            "target_rgb_range": "0_1",
            "task": task,
            "flow_source": "noise",
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_err = None
        for attempt in range(3):
            try:
                return self._load_sample(idx)
            except Exception as e:
                last_err = e
                idx = (idx + 1) % len(self._file_list)
        logger.error("Failed to load sample after 3 attempts: %s", last_err)
        return {
            "prompt": "Generate an image.",
            "target_rgb": torch.zeros(3, self.resolution, self.resolution),
            "target_rgb_range": "0_1",
            "task": self.task,
            "flow_source": "noise",
        }

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __del__(self) -> None:
        self.close()


__all__ = ["TarArchiveDataset", "TarTargetDataset", "SequentialMultiTaskDataset",
           "IndexedTarReader", "SequentialTarReader", "MultiFileReader"]


# ── Sequential multi-task dataset (no random access) ─────────────────────


class SequentialMultiTaskDataset(Dataset):
    """Dataset that wraps multiple ``TarTargetDataset`` instances and ensures
    each task's gzip stream is accessed strictly sequentially.

    Unlike ``MixedVisionDataset`` (which randomly samples sub-datasets), this
    dataset cycles through tasks round-robin and advances each task's cursor
    by one file per access.  This guarantees forward-only stream access for
    the underlying ``SequentialTarReader``.

    Shuffling is approximated by pre-shuffling each task's file list at init
    time (subject to the constraint that files must still be accessed in
    ascending offset order – so shuffling is limited to within-stream
    reordering which is NOT possible with sequential access).

    For true sequential access the file list MUST remain sorted by offset.
    Shuffling is therefore NOT applied; the natural diversity of the tar
    stream (which interleaves multiple source datasets) provides sufficient
    variety for training.

    Parameters
    ----------
    task_datasets:
        Dict mapping task name → ``TarTargetDataset``.
    task_order:
        Order in which tasks are cycled (e.g. ``["depth", "normal", "seg"]``).
    """

    def __init__(
        self,
        task_datasets: dict[str, TarTargetDataset],
        task_order: list[str] | None = None,
    ) -> None:
        self.task_datasets = task_datasets
        self.task_order = task_order or list(task_datasets.keys())
        self._cursors: dict[str, int] = {t: 0 for t in self.task_order}
        self._total = sum(len(ds) for ds in task_datasets.values())

        # Compute total steps (one full pass through all tasks)
        self._step_count = 0

        # For logging compatibility
        self.dir_name = "sequential_multi_task"

        logger.info(
            "SequentialMultiTaskDataset: %d tasks, %d total samples, order=%s",
            len(task_datasets), self._total, self.task_order,
        )

    def __len__(self) -> int:
        return self._total

    @property
    def task(self) -> str:
        """Return current task name (for logging compatibility)."""
        return "multi"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Find next task with remaining samples (round-robin)
        for _ in range(len(self.task_order)):
            task = self.task_order[self._step_count % len(self.task_order)]
            self._step_count += 1
            ds = self.task_datasets[task]
            cursor = self._cursors[task]
            if cursor < len(ds):
                self._cursors[task] = cursor + 1
                return ds[cursor]
        # All tasks exhausted – should not happen within __len__
        raise IndexError(f"All tasks exhausted (idx={idx})")

    def set_epoch(self, epoch: int) -> None:
        """Reset cursors for a new epoch and update sub-dataset epochs."""
        for t in self.task_order:
            self._cursors[t] = 0
            self.task_datasets[t].set_epoch(epoch)
        self._step_count = 0
        logger.info("SequentialMultiTaskDataset: reset for epoch %d", epoch)

    def close(self) -> None:
        for ds in self.task_datasets.values():
            ds.close()


class StreamOrderedMultiTaskDataset(Dataset):
    """Dataset that merges multiple TarTargetDatasets and accesses samples
    in global stream-offset order for optimal HDD performance.

    All tasks share the same underlying gzip stream (the concatenated tar.gz
    volumes).  By sorting all samples across all tasks by their stream offset,
    consecutive accesses are close together and the stream head only moves
    forward.  This avoids the massive seek penalty of random access on HDD.

    Parameters
    ----------
    task_datasets:
        Dict mapping task name → TarTargetDataset.
    shared_reader:
        A shared IndexedTarReader or SequentialTarReader instance.
        All tasks use this reader instead of creating their own.
    """

    def __init__(
        self,
        task_datasets: dict[str, TarTargetDataset],
        shared_reader: IndexedTarReader | SequentialTarReader,
    ) -> None:
        self.task_datasets = task_datasets
        self._shared_reader = shared_reader

        # Collect all (task, tar_path, offset, size) tuples
        self._samples: list[tuple[str, str, int, int]] = []
        for task, ds in task_datasets.items():
            for tar_path, info in ds._file_index.items():
                self._samples.append((task, tar_path, info["offset"], info["size"]))

        # Sort by stream offset for sequential access
        self._samples.sort(key=lambda x: x[2])

        # For logging compatibility
        self.dir_name = "stream_ordered_multi_task"

        logger.info(
            "StreamOrderedMultiTaskDataset: %d tasks, %d total samples, "
            "offset range [%d, %d]",
            len(task_datasets), len(self._samples),
            self._samples[0][2] if self._samples else 0,
            self._samples[-1][2] if self._samples else 0,
        )

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def task(self) -> str:
        return "multi"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task, tar_path, offset, size = self._samples[idx]
        ds = self.task_datasets[task]

        # Read target data using shared reader
        try:
            tgt_data = self._shared_reader.read_at(offset, size)
        except Exception as e:
            logger.warning("Failed to read %s at offset %d: %s", tar_path, offset, e)
            tgt_data = None

        if tgt_data is None:
            tgt_img = Image.new("RGB", (ds.resolution, ds.resolution), (0, 0, 0))
        else:
            tgt_img = ds._bytes_to_pil(tgt_data)

        prompt = ds.DEFAULT_PROMPTS.get(task, ("Generate an image.",))[0]
        target_resampling = (
            Image.Resampling.NEAREST if task in ds.MASK_TASKS else Image.Resampling.LANCZOS
        )
        tgt_img = tgt_img.convert("RGB").resize(
            (ds.resolution, ds.resolution), target_resampling
        )
        target_tensor = torch.from_numpy(
            np.asarray(tgt_img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)

        return {
            "prompt": prompt,
            "target_rgb": target_tensor,
            "target_rgb_range": "0_1",
            "task": task,
            "flow_source": "noise",
        }

    def set_epoch(self, epoch: int) -> None:
        """Reset for a new epoch (no-op for stream-ordered access)."""
        logger.info("StreamOrderedMultiTaskDataset: epoch %d", epoch)

    def close(self) -> None:
        if self._shared_reader is not None:
            self._shared_reader.close()
            self._shared_reader = None
        for ds in self.task_datasets.values():
            ds.close()
