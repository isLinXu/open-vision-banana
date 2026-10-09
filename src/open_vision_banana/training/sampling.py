"""Deterministic task-balanced sampling with equal distributed rank lengths."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Sampler


class TaskWeightedSampler(Sampler[int]):
    """Draw tasks then samples with replacement; shard one global draw stream."""

    def __init__(
        self, task_indices: dict[str, list[int]], strategy: str = "proportional",
        weights: dict[str, float] | None = None, seed: int = 42,
        rank: int = 0, world_size: int = 1, num_samples: int | None = None,
    ):
        if strategy not in {"proportional", "uniform", "sqrt"}:
            raise ValueError(f"Unknown task sampling strategy: {strategy}")
        if not task_indices or any(not indices for indices in task_indices.values()):
            raise ValueError("Every task must have samples")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank/world_size")
        weights = weights or {}
        if set(weights) - set(task_indices):
            raise ValueError("Sampling weights contain unknown tasks")
        if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError("Sampling weights must be positive and finite")
        self.groups = task_indices
        self.tasks = sorted(task_indices)
        masses = {
            t: (len(task_indices[t]) if strategy == "proportional" else
                math.sqrt(len(task_indices[t])) if strategy == "sqrt" else 1)
            * weights.get(t, 1) for t in self.tasks
        }
        total = sum(masses.values())
        self.task_probabilities = {t: masses[t] / total for t in self.tasks}
        self.seed, self.epoch = seed, 0
        self.rank, self.world_size = rank, world_size
        requested = num_samples if num_samples is not None else sum(map(len, task_indices.values()))
        self.global_samples = requested // world_size * world_size
        if self.global_samples < world_size:
            raise ValueError("Not enough samples for distributed ranks")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.global_samples // self.world_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        probabilities = torch.tensor([self.task_probabilities[t] for t in self.tasks])
        tasks = torch.multinomial(probabilities, self.global_samples, True, generator=generator)
        draws = [
            self.groups[self.tasks[t]][torch.randint(
                len(self.groups[self.tasks[t]]), (1,), generator=generator
            ).item()] for t in tasks.tolist()
        ]
        return iter(draws[self.rank::self.world_size])
