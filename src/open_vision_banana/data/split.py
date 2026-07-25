"""Dataset splitting utilities for train/validation.

Supports both sample-level and source-level splitting. Source-level splitting
groups samples by their source image (identified by stem prefix + MD5) to prevent
data leakage when the same source image appears in multiple tasks (e.g., Hypersim
images used for both depth and normal estimation).
"""

from __future__ import annotations

import bisect
import hashlib
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset, random_split

logger = logging.getLogger(__name__)


def split_dataset(
    dataset: Dataset,
    val_ratio: float,
    seed: int = 42,
) -> tuple[Dataset, Dataset | None]:
    """Split a dataset into train and validation subsets.

    Args:
        dataset: Source dataset.
        val_ratio: Fraction of samples for validation (0 disables split).
        seed: Random seed for reproducible splits.

    Returns:
        (train_dataset, val_dataset) where val_dataset is None when
        val_ratio <= 0 or the dataset is too small to split.
    """
    if val_ratio <= 0.0:
        return dataset, None

    n = len(dataset)
    if n < 2:
        return dataset, None

    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    if n_train < 1:
        return dataset, None

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds


@lru_cache(maxsize=16384)
def _compute_source_key(img_path: Path) -> str:
    """Compute a source key from complete image file content (SHA-256).

    This groups images that are pixel-identical (same source), even if
    they appear in different task directories.
    """
    h = hashlib.sha256()
    try:
        with open(img_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        # Fallback to stem-based grouping
        return img_path.stem
    return h.hexdigest()


def _extract_source_key_from_sample(sample: dict[str, Any]) -> str | None:
    """Try to extract source key from a VisionTaskDataset sample dict."""
    # VisionTaskDataset samples have an 'image' key with the file path
    img_path = sample.get("image")
    if not isinstance(img_path, (str, Path)):
        img_path = sample.get("image_path")
    if isinstance(img_path, (str, Path)) and Path(img_path).exists():
        return _compute_source_key(Path(img_path))
    # Fallback: use stem prefix (e.g., 'hypersim_000001')
    img_name = sample.get("image_name", "")
    if img_name:
        # Group by first part of stem (e.g., 'hypersim_000001' -> 'hypersim_000001')
        return img_name.split("_")[0] if "_" in img_name else img_name
    return None


def _extract_source_key(dataset: Dataset, index: int) -> str | None:
    """Read a stable source key without decoding an image when metadata is available."""
    from torch.utils.data import ConcatDataset, Subset

    if isinstance(dataset, Subset):
        return _extract_source_key(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, ConcatDataset):
        dataset_index = bisect.bisect_right(dataset.cumulative_sizes, index)
        previous_size = 0 if dataset_index == 0 else dataset.cumulative_sizes[dataset_index - 1]
        return _extract_source_key(dataset.datasets[dataset_index], index - previous_size)

    samples = getattr(dataset, "samples", None)
    if isinstance(samples, list) and index < len(samples) and isinstance(samples[index], dict):
        key = _extract_source_key_from_sample(samples[index])
        if key is not None:
            return key
    sample = dataset[index]
    return _extract_source_key_from_sample(sample) if isinstance(sample, dict) else None


def split_vision_datasets(
    datasets: list[Dataset],
    val_ratio: float,
    seed: int = 42,
    source_level: bool = True,
) -> tuple[list[Dataset], Dataset | None]:
    """Split concatenated vision datasets for validation monitoring.

    Args:
        datasets: List of VisionTaskDataset instances.
        val_ratio: Fraction of samples for validation.
        seed: Random seed.
        source_level: If True, group samples by source image to prevent
            data leakage across train/val (same source image only in one split).
            Falls back to random split if source grouping is not possible.

    Returns:
        (train_datasets, val_dataset) tuple.
    """
    from torch.utils.data import ConcatDataset

    if not datasets or val_ratio <= 0.0:
        return datasets, None

    if source_level:
        try:
            return _split_source_level(datasets, val_ratio, seed)
        except Exception as e:
            logger.warning(f"Source-level split failed ({e}), falling back to random split")

    combined: ConcatDataset = ConcatDataset(datasets)
    train_part, val_part = split_dataset(combined, val_ratio, seed=seed)
    return [train_part], val_part


def _split_source_level(
    datasets: list[Dataset],
    val_ratio: float,
    seed: int,
) -> tuple[list[Dataset], Dataset | None]:
    """Source-level split: group by source image, then split groups."""
    from torch.utils.data import ConcatDataset

    combined: ConcatDataset = ConcatDataset(datasets)
    n = len(combined)

    # Build source_key -> indices mapping
    source_groups: dict[str, list[int]] = defaultdict(list)
    ungrouped: list[int] = []

    for i in range(n):
        key = _extract_source_key(combined, i)
        if key:
            source_groups[key].append(i)
        else:
            ungrouped.append(i)

    logger.info(f"Source grouping: {len(source_groups)} unique sources, "
                f"{len(ungrouped)} ungrouped, {n} total")

    # If too few groups, fall back to random split
    if len(source_groups) < 10:
        raise ValueError(f"Only {len(source_groups)} source groups, too few for source-level split")

    # Add ungrouped to their own singleton groups
    for i in ungrouped:
        source_groups[f"__ungrouped_{i}"] = [i]

    # Shuffle group keys
    group_keys = list(source_groups.keys())
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(group_keys), generator=generator)
    group_keys = [group_keys[i] for i in perm.tolist()]

    # Split groups
    n_val_target = max(1, int(n * val_ratio))
    val_indices = []
    train_indices = []
    n_val_so_far = 0

    for key in group_keys:
        indices = source_groups[key]
        if n_val_so_far < n_val_target:
            val_indices.extend(indices)
            n_val_so_far += len(indices)
        else:
            train_indices.extend(indices)

    if not train_indices or not val_indices:
        raise ValueError("Source-level split produced empty train or val set")

    logger.info(f"Source-level split: train={len(train_indices)}, val={len(val_indices)} "
                f"({len(val_indices)/n*100:.1f}%)")

    train_by_dataset: list[list[int]] = [[] for _ in datasets]
    for global_index in train_indices:
        dataset_index = bisect.bisect_right(combined.cumulative_sizes, global_index)
        previous_size = 0 if dataset_index == 0 else combined.cumulative_sizes[dataset_index - 1]
        train_by_dataset[dataset_index].append(global_index - previous_size)
    train_ds = [
        Subset(dataset, indices)
        for dataset, indices in zip(datasets, train_by_dataset, strict=True)
        if indices
    ]
    val_ds = Subset(combined, val_indices)
    return train_ds, val_ds


__all__ = ["split_dataset", "split_vision_datasets"]
