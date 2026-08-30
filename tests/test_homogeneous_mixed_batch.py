"""Homogeneous vision/generation micro-batch sampling."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from open_vision_banana.data.mixed import (
    GEN_INDEX_BASE,
    HomogeneousMixedBatchSampler,
    MixedVisionDataset,
)


class _Tiny(Dataset):
    def __init__(self, task: str, n: int = 4) -> None:
        self.task = task
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            "prompt": self.task,
            "image": torch.zeros(3, 2, 2) if self.task != "generation" else None,
            "target_rgb": torch.ones(3, 2, 2),
            "task": self.task,
            "flow_source": "noise" if self.task == "generation" else "input",
        }


def test_homogeneous_batches_are_single_branch():
    sampler = HomogeneousMixedBatchSampler(
        num_batches=40,
        batch_size=4,
        vision_ratio=0.25,
        seed=0,
    )
    vision_batches = 0
    for batch in sampler:
        assert len(batch) == 4
        is_vision = all(i < 0 for i in batch)
        is_gen = all(i >= GEN_INDEX_BASE for i in batch)
        assert is_vision ^ is_gen
        vision_batches += int(is_vision)
    assert 5 <= vision_batches <= 20


def test_forced_indices_skip_per_sample_coin_flip():
    mixed = MixedVisionDataset(
        vision_datasets=[_Tiny("depth")],
        generation_dataset=_Tiny("generation", n=8),
        vision_ratio=0.01,  # legacy path almost never samples vision
        seed=0,
    )
    vision = mixed[-(1)]
    generation = mixed[GEN_INDEX_BASE + 3]
    assert vision["task"] == "depth"
    assert generation["task"] == "generation"
