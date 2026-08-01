"""Multi-task dataset adapter for TinyVisionBanana training.

Wraps the existing v3 dataset directory layout and produces batches
suitable for the standalone tiny model.  Supports task balancing,
random sampling, and split-clean train/val separation.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset as _Dataset

logger = logging.getLogger(__name__)

# Directory name in v3 -> canonical task name
DIR_TO_TASK: dict[str, str] = {
    "depth": "depth",
    "normal": "normal",
    "segmentation": "semantic",
    "structure": "edge",
    "generation": "generation",
}

# Canonical task ordering
TINY_TASK_ORDER: tuple[str, ...] = ("depth", "normal", "edge", "semantic", "generation")


@dataclass(frozen=True)
class TinySample:
    """A single training sample."""

    image_path: str
    target_path: str
    task: str
    prompt: str
    split: str  # "train" or "val"


def _load_image(path: str, resolution: int, range_: tuple[float, float]) -> torch.Tensor:
    """Load, resize, and normalize an image to a tensor."""
    img = Image.open(path).convert("RGB")
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
    arr = torch.from_numpy(
        __import__("numpy").asarray(img, dtype="float32") / 255.0
    ).permute(2, 0, 1)
    lo, hi = range_
    return arr * (hi - lo) + lo


class TinyMultiTaskDataset(_Dataset):
    """Multi-task dataset that yields (image, target_rgb, task) tuples.

    Parameters
    ----------
    data_root : str
        Path to the v3 dataset root (contains ``depth/``, ``normal/``, etc.).
    split : str
        ``"train"`` or ``"val"``.
    tasks : sequence of str
        Which tasks to include.  Defaults to all 5.
    resolution : int
        Target resolution (images will be resized).
    balance : bool
        If True, oversample minority tasks to match the largest task.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        *,
        tasks: Sequence[str] = TINY_TASK_ORDER,
        resolution: int = 256,
        balance: bool = True,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        root = Path(data_root)
        if not root.is_dir():
            raise FileNotFoundError(f"Data root not found: {root}")

        # Load split manifest
        split_path = root / "train_val_split.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"train_val_split.json not found at {split_path}")
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        split_names = split_data.get(split, {})
        if not split_names:
            raise ValueError(f"No samples found for split={split!r}")

        self.resolution = resolution
        self.tasks = tuple(tasks)

        # Build sample list
        all_samples: list[TinySample] = []
        per_task_samples: dict[str, list[TinySample]] = {}

        for task in self.tasks:
            # Find directory name for this task
            dir_name = next(
                (d for d, t in DIR_TO_TASK.items() if t == task),
                task,
            )
            task_split_names = split_names.get(dir_name, [])
            if not task_split_names:
                logger.warning(f"No {split} samples for task {task!r} in split manifest")
                continue

            img_dir = root / dir_name / "images"
            tgt_dir = root / dir_name / "targets"
            prompt_dir = root / dir_name / "prompts"

            # Build prompt index
            task_samples: list[TinySample] = []
            img_exts = {".jpg", ".jpeg", ".png", ".webp"}

            # Index targets for O(1) lookup
            target_index: dict[str, Path] = {}
            if tgt_dir.is_dir():
                with os.scandir(str(tgt_dir)) as entries:
                    for entry in entries:
                        if entry.is_file():
                            p = Path(entry.path)
                            if p.suffix.lower() in img_exts:
                                target_index[p.stem] = p

            # Scan images
            if img_dir.is_dir():
                with os.scandir(str(img_dir)) as entries:
                    for entry in sorted(entries, key=lambda e: e.name):
                        if not entry.is_file():
                            continue
                        p = Path(entry.path)
                        if p.suffix.lower() not in img_exts:
                            continue
                        stem = p.stem
                        filename = p.name  # e.g. "depth_hyp_real_00000.jpg"
                        # Split manifest may store either stem or full filename
                        if stem not in task_split_names and filename not in task_split_names:
                            continue
                        tgt_path = target_index.get(stem)
                        if tgt_path is None:
                            if task == "generation":
                                # For generation, the target is the image itself
                                # (targets/ dir contains .txt prompt files, not images)
                                tgt_path = p
                            else:
                                continue

                        # Load optional prompt
                        prompt = ""
                        prompt_path = prompt_dir / f"{stem}.txt"
                        if prompt_path.is_file():
                            prompt = prompt_path.read_text(encoding="utf-8").strip()
                        # For generation, targets/ contains .txt prompt files
                        if not prompt and task == "generation":
                            gen_prompt_path = tgt_dir / f"{stem}.txt"
                            if gen_prompt_path.is_file():
                                prompt = gen_prompt_path.read_text(encoding="utf-8").strip()

                        sample = TinySample(
                            image_path=str(p),
                            target_path=str(tgt_path),
                            task=task,
                            prompt=prompt,
                            split=split,
                        )
                        task_samples.append(sample)

            if not task_samples:
                logger.warning(f"No valid samples found for task {task!r}")
                continue

            per_task_samples[task] = task_samples
            all_samples.extend(task_samples)
            logger.info(f"  {task}: {len(task_samples)} samples")

        # Balance tasks by oversampling minority
        if balance and len(per_task_samples) > 1:
            max_count = max(len(s) for s in per_task_samples.values())
            balanced: list[TinySample] = []
            for task, samples in per_task_samples.items():
                if len(samples) == 0:
                    continue
                # Repeat to match max_count
                repeats = max_count // len(samples)
                remainder = max_count % len(samples)
                balanced.extend(samples * repeats)
                if remainder:
                    balanced.extend(samples[:remainder])
            self.samples = balanced
            logger.info(f"  Total after balancing: {len(self.samples)} (max/task={max_count})")
        else:
            self.samples = all_samples
            logger.info(f"  Total samples: {len(self.samples)}")

        self._per_task = per_task_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        image = _load_image(sample.image_path, self.resolution, (-1.0, 1.0))
        target = _load_image(sample.target_path, self.resolution, (0.0, 1.0))
        return {
            "image": image,
            "target_rgb": target,
            "task": sample.task,
            "prompt": sample.prompt,
        }

    def task_counts(self) -> dict[str, int]:
        """Return per-task sample counts."""
        return {task: len(samples) for task, samples in self._per_task.items()}


def collate_tiny_batch(
    batch: list[dict[str, torch.Tensor | str]],
) -> dict[str, torch.Tensor | list[str]]:
    """Collate function for TinyMultiTaskDataset.

    Stacks image/target tensors and keeps task names as a list.
    """
    images = torch.stack([b["image"] for b in batch])
    targets = torch.stack([b["target_rgb"] for b in batch])
    tasks = [b["task"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    return {
        "image": images,
        "target_rgb": targets,
        "task": tasks,
        "prompt": prompts,
    }


__all__ = [
    "DIR_TO_TASK",
    "TINY_TASK_ORDER",
    "TinyMultiTaskDataset",
    "TinySample",
    "collate_tiny_batch",
]
