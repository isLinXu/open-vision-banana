"""Mixed dataset implementing the 9:1 visual-to-language data ratio.

Combines vision task datasets with a generation dataset at the
ratio specified in the paper (9:1 generation-to-vision).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MixedVisionDataset(Dataset):
    """Mixed dataset implementing the 9:1 visual-to-language data ratio.

    Combines vision task datasets with a generation dataset at the
    ratio specified in the paper (9:1 generation-to-vision).
    """

    def __init__(
        self,
        vision_datasets: list[Dataset],
        generation_dataset: Dataset | None = None,
        vision_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.vision_datasets = vision_datasets
        self.generation_dataset = generation_dataset
        self.vision_ratio = vision_ratio
        self.rng = torch.Generator().manual_seed(seed)

        self.total_vision = sum(len(ds) for ds in vision_datasets)
        self.vision_weight = vision_ratio
        self.gen_weight = 1.0 - vision_ratio

    def __len__(self) -> int:
        return self.total_vision

    def __getitem__(self, idx: int) -> dict:
        if torch.rand(1, generator=self.rng).item() < self.vision_weight:
            ds_idx = idx % len(self.vision_datasets)
            ds = self.vision_datasets[ds_idx]
            sample_idx = idx % len(ds)
            return ds[sample_idx]
        elif self.generation_dataset is not None:
            sample_idx = idx % len(self.generation_dataset)
            return self.generation_dataset[sample_idx]
        else:
            ds_idx = idx % len(self.vision_datasets)
            ds = self.vision_datasets[ds_idx]
            sample_idx = idx % len(ds)
            return ds[sample_idx]


__all__ = ["MixedVisionDataset"]
