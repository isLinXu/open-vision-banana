"""Mixed dataset implementing the 9:1 visual-to-language data ratio.

Combines vision task datasets with a generation dataset at the
ratio specified in the paper (9:1 generation-to-vision).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MixedVisionDataset(Dataset):
    """Mixed dataset implementing a configurable vision-to-generation ratio.

    At each call to ``__getitem__``, one of the underlying datasets is sampled
    according to the configured ``vision_ratio``:

    - With probability ``vision_ratio``: a sample is drawn from one of the
      vision task datasets (chosen uniformly at random, then a random sample
      within that dataset).
    - With probability ``1 - vision_ratio``: a sample is drawn from the
      generation dataset (if provided), otherwise falls back to vision data.

    This preserves the paper's 9:1 generation-to-vision mixture without
    introducing index-based sampling bias (the old ``idx % len`` approach
    was highly correlated with the DataLoader's sequential index ordering).
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
        self._vision_lengths = [len(ds) for ds in vision_datasets]

        # Backward-compatible aliases
        self.vision_weight = vision_ratio
        self.gen_weight = 1.0 - vision_ratio

    def __len__(self) -> int:
        return self.total_vision

    def __getitem__(self, idx: int) -> dict:
        use_vision = (
            self.generation_dataset is None
            or torch.rand(1, generator=self.rng).item() < self.vision_ratio
        )

        if use_vision:
            # Pick a vision dataset uniformly at random, then a random sample.
            ds_idx = int(torch.randint(len(self.vision_datasets), (1,), generator=self.rng).item())
            ds = self.vision_datasets[ds_idx]
            sample_idx = int(torch.randint(len(ds), (1,), generator=self.rng).item())
            return ds[sample_idx]
        else:
            assert self.generation_dataset is not None
            sample_idx = int(torch.randint(len(self.generation_dataset), (1,), generator=self.rng).item())
            return self.generation_dataset[sample_idx]


__all__ = ["MixedVisionDataset"]
