"""Dataset splitting utilities for train/validation."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Subset, random_split


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
        ``(train_dataset, val_dataset)`` where val_dataset is None when
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


def split_vision_datasets(
    datasets: list[Dataset],
    val_ratio: float,
    seed: int = 42,
) -> tuple[list[Dataset], Dataset | None]:
    """Split concatenated vision datasets for validation monitoring."""
    from torch.utils.data import ConcatDataset

    if not datasets or val_ratio <= 0.0:
        return datasets, None

    combined = ConcatDataset(datasets)
    train_part, val_part = split_dataset(combined, val_ratio, seed=seed)
    return [train_part], val_part


__all__ = ["split_dataset", "split_vision_datasets"]
