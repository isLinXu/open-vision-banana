"""Mixed dataset implementing the 9:1 visual-to-language data ratio.

Combines vision task datasets with a generation dataset at the
ratio specified in the paper (9:1 generation-to-vision).
"""

from __future__ import annotations

from typing import Any

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

    ``__len__`` (P1-6 fix):
        The original implementation returned ``total_vision`` — the raw sum of
        all vision dataset sizes.  This was misleading because:

        1. Sampling is probabilistic and ignores the index passed to
           ``__getitem__``, so the *effective* epoch length is arbitrary.
        2. When a generation dataset is present (which is 9× larger in the
           paper), returning only the vision size grossly under-counts the
           data seen per epoch.

        The corrected ``__len__`` returns an *effective epoch length* that
        accounts for the mixing ratio:

            effective_len = total_vision / vision_ratio

        This estimates how many calls are needed to see all vision samples
        once on average (matching the DataLoader's progress-bar expectations).
        When no generation dataset is provided the original total_vision is
        returned unchanged (100 % vision sampling).
    """

    def __init__(
        self,
        vision_datasets: list[Dataset],
        generation_dataset: Dataset | None = None,
        vision_ratio: float = 0.1,
        seed: int = 42,
    ):
        if not 0.0 < vision_ratio <= 1.0:
            raise ValueError(f"vision_ratio must be in (0, 1]; got {vision_ratio}")

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
        """Return an effective epoch length that reflects the mixing ratio.

        When a generation dataset is present, we want the DataLoader to issue
        roughly ``total_vision / vision_ratio`` calls so that vision samples
        appear ~total_vision times on average per epoch.

        When no generation dataset is provided sampling is 100% vision, so
        we return total_vision directly (unchanged behaviour).
        """
        if self.generation_dataset is None or self.vision_ratio >= 1.0:
            return self.total_vision
        # Avoid division-by-zero (already guarded by __init__ assertion)
        return max(1, round(self.total_vision / self.vision_ratio))

    def __getitem__(self, idx: int) -> dict[str, Any]:
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

    def __repr__(self) -> str:
        return (
            f"MixedVisionDataset("
            f"n_vision_datasets={len(self.vision_datasets)}, "
            f"total_vision={self.total_vision}, "
            f"has_generation={self.generation_dataset is not None}, "
            f"vision_ratio={self.vision_ratio}, "
            f"effective_len={len(self)})"
        )


__all__ = ["MixedVisionDataset"]
