"""Configurable vision-task and generation-preservation sampling."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

# Homogeneous batch encoding: indices >= GEN_INDEX_BASE force the generation
# branch; smaller indices force vision. Keeps the paper 9:1 ratio at batch
# granularity so micro-batches >1 do not dilute vision gradients.
GEN_INDEX_BASE = 1 << 30


class MixedVisionDataset(Dataset):
    """Mixed dataset implementing a configurable vision-to-generation ratio.

    At each call to ``__getitem__``, one of the underlying datasets is sampled
    according to the configured ``vision_ratio``:

    - With probability ``vision_ratio``: a sample is drawn from one of the
      vision task datasets (chosen uniformly at random, then a random sample
      within that dataset).
    - With probability ``1 - vision_ratio``: a sample is drawn from the
      generation dataset (if provided), otherwise falls back to vision data.

    The ratio is a project configuration and must be recorded in experiment
    manifests; it is not treated as a paper-disclosed constant.

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
        colormaps: list[str] | None = None,
        task_sampling: str = "uniform",
        task_weights: dict[str, float] | None = None,
        depth_lam: float = -3.0,
        depth_c: float = 10.0 / 3.0,
        depth_curve: str = "metric_power",
        depth_log_alpha: float = 0.4,
    ):
        if not 0.0 < vision_ratio <= 1.0:
            raise ValueError(f"vision_ratio must be in (0, 1]; got {vision_ratio}")
        if task_sampling not in {"uniform", "dataset_uniform", "proportional"}:
            raise ValueError(f"Unsupported task_sampling: {task_sampling}")

        self.vision_datasets = [dataset for dataset in vision_datasets if len(dataset) > 0]
        self.generation_dataset = generation_dataset
        self.vision_ratio = vision_ratio
        self.seed = seed
        self.epoch = 0
        self.task_sampling = task_sampling
        self.task_weights = task_weights or {}
        self.depth_lam = depth_lam
        self.depth_c = depth_c
        self.depth_curve = depth_curve
        self.depth_log_alpha = depth_log_alpha

        self.total_vision: int = sum(len(ds) for ds in self.vision_datasets)  # type: ignore[misc]
        self._vision_lengths = [len(ds) for ds in self.vision_datasets]
        self._task_to_datasets: dict[str, list[Dataset]] = {}
        for dataset in self.vision_datasets:
            task = _dataset_task(dataset)
            self._task_to_datasets.setdefault(task, []).append(dataset)

        # Multi-colormap training augmentation (paper Section 3.2)
        self.colormaps = colormaps or ["cube_edge"]

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
        # The ratio defines how many calls are needed to expose the complete
        # vision set once on average. Generation samples are intentionally
        # drawn with replacement, matching the paper's replay formulation.
        return max(1, round(self.total_vision / self.vision_ratio))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Homogeneous batch samplers encode the branch in the index so every
        # slot in a micro-batch shares vision-only or generation-only flow:
        #   idx < 0              → forced vision (seed = -(idx + 1))
        #   idx >= GEN_INDEX_BASE → forced generation
        #   otherwise            → legacy per-sample 9:1 mixture
        if idx >= GEN_INDEX_BASE:
            if self.generation_dataset is None:
                raise IndexError("Generation branch requested but no generation dataset")
            rng = self._generator_for_index(idx - GEN_INDEX_BASE)
            return self._draw_generation_sample(rng)
        if idx < 0:
            rng = self._generator_for_index(-(idx + 1))
            return self._draw_vision_sample(rng)

        rng = self._generator_for_index(idx)
        use_vision = (
            self.generation_dataset is None
            or torch.rand(1, generator=rng).item() < self.vision_ratio
        )
        if use_vision:
            return self._draw_vision_sample(rng)
        return self._draw_generation_sample(rng)

    def _draw_vision_sample(self, rng: torch.Generator) -> dict[str, Any]:
        if not self.vision_datasets:
            raise IndexError("MixedVisionDataset has no non-empty vision datasets")
        ds = self._sample_vision_dataset(rng)
        sample_idx = int(torch.randint(len(ds), (1,), generator=rng).item())
        sample = ds[sample_idx]
        if isinstance(sample, dict):
            sample = dict(sample)
            sample.setdefault("task", _dataset_task(ds, sample_idx))
        if isinstance(sample, dict) and sample.get("task") == "depth":
            cmap_idx = int(torch.randint(len(self.colormaps), (1,), generator=rng).item())
            colormap = self.colormaps[cmap_idx]
            sample["colormap"] = colormap
            need_reencode = colormap != "cube_edge" or self.depth_curve != "metric_power"
            if need_reencode and "target_rgb" in sample:
                from ..core.depth_mapper import DepthRGBMapper

                paper = DepthRGBMapper(lam=self.depth_lam, c=self.depth_c)
                mapper = DepthRGBMapper(
                    lam=self.depth_lam,
                    c=self.depth_c,
                    curve=self.depth_curve,
                    log_alpha=self.depth_log_alpha,
                )
                depth = sample.get("gt_depth")
                if not isinstance(depth, torch.Tensor):
                    target_rgb = sample["target_rgb"]
                    if isinstance(target_rgb, torch.Tensor):
                        depth = paper.rgb_to_depth(target_rgb.unsqueeze(0)).squeeze(0)
                if isinstance(depth, torch.Tensor):
                    if depth.dim() == 2:
                        depth = depth.unsqueeze(0)
                    mapper.set_colormap(colormap)
                    sample["target_rgb"] = mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        return sample

    def _draw_generation_sample(self, rng: torch.Generator) -> dict[str, Any]:
        assert self.generation_dataset is not None
        sample_idx = int(torch.randint(len(self.generation_dataset), (1,), generator=rng).item())
        return self.generation_dataset[sample_idx]

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic sampling streams between training epochs."""
        self.epoch = int(epoch)

    def _generator_for_index(self, index: int) -> torch.Generator:
        # DistributedSampler assigns disjoint indices to ranks. Deriving all
        # randomness from the index prevents cloned worker/rank RNG streams.
        sample_seed = self.seed + self.epoch * 1_000_003 + int(index)
        return torch.Generator().manual_seed(sample_seed)

    def _sample_vision_dataset(self, rng: torch.Generator) -> Dataset:
        if self.task_sampling == "dataset_uniform":
            index = int(
                torch.randint(len(self.vision_datasets), (1,), generator=rng).item()
            )
            return self.vision_datasets[index]

        if self.task_sampling == "proportional":
            weights = torch.tensor(self._vision_lengths, dtype=torch.float64)
            index = int(torch.multinomial(weights, 1, generator=rng).item())
            return self.vision_datasets[index]

        tasks = sorted(self._task_to_datasets)
        task_weights = torch.tensor(
            [self.task_weights.get(task, 1.0) for task in tasks], dtype=torch.float64
        )
        task_index = int(torch.multinomial(task_weights, 1, generator=rng).item())
        datasets = self._task_to_datasets[tasks[task_index]]
        dataset_index = int(torch.randint(len(datasets), (1,), generator=rng).item())
        return datasets[dataset_index]

    def __repr__(self) -> str:
        return (
            f"MixedVisionDataset("
            f"n_vision_datasets={len(self.vision_datasets)}, "
            f"total_vision={self.total_vision}, "
            f"has_generation={self.generation_dataset is not None}, "
            f"vision_ratio={self.vision_ratio}, "
            f"task_sampling={self.task_sampling}, "
            f"effective_len={len(self)})"
        )


class HomogeneousMixedBatchSampler(Sampler[list[int]]):
    """Yield whole micro-batches from one branch so mean-loss stays undiluted.

    With probability ``vision_ratio`` the batch is all vision samples; otherwise
    all generation. Expected vision fraction stays at ``vision_ratio``, matching
    the paper mix, without averaging a lone depth sample against three t2i
    samples inside ``batch_size>1``.
    """

    def __init__(
        self,
        *,
        num_batches: int,
        batch_size: int,
        vision_ratio: float,
        seed: int = 42,
    ) -> None:
        if num_batches < 1:
            raise ValueError("num_batches must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 < vision_ratio <= 1.0:
            raise ValueError(f"vision_ratio must be in (0, 1]; got {vision_ratio}")
        self.num_batches = int(num_batches)
        self.batch_size = int(batch_size)
        self.vision_ratio = float(vision_ratio)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        vision_batches = round(self.num_batches * self.vision_ratio)
        branch_schedule = [True] * vision_batches + [False] * (self.num_batches - vision_batches)
        order = torch.randperm(self.num_batches, generator=rng).tolist()
        use_vision_by_batch = [False] * self.num_batches
        for position, batch_id in enumerate(order):
            use_vision_by_batch[batch_id] = branch_schedule[position]
        for batch_id in range(self.num_batches):
            use_vision = use_vision_by_batch[batch_id]
            indices: list[int] = []
            for slot in range(self.batch_size):
                local = int(
                    torch.randint(0, GEN_INDEX_BASE, (1,), generator=rng).item()
                )
                # Fold batch/slot into the seed space so slots differ.
                local = (local + batch_id * 10_007 + slot * 97) % GEN_INDEX_BASE
                if use_vision:
                    indices.append(-(local + 1))
                else:
                    indices.append(GEN_INDEX_BASE + local)
            yield indices


def _dataset_task(dataset: Dataset, index: int | None = None) -> str:
    """Resolve a task label through Subset/ConcatDataset wrappers."""
    from torch.utils.data import ConcatDataset, Subset

    if isinstance(dataset, Subset):
        inner_index = None if index is None else int(dataset.indices[index])
        return _dataset_task(dataset.dataset, inner_index)
    if isinstance(dataset, ConcatDataset):
        if index is not None:
            import bisect

            dataset_index = bisect.bisect_right(dataset.cumulative_sizes, index)
            previous_size = 0 if dataset_index == 0 else dataset.cumulative_sizes[dataset_index - 1]
            sample_index = index - previous_size
            return _dataset_task(dataset.datasets[dataset_index], sample_index)
        tasks = {_dataset_task(child) for child in dataset.datasets}
        return next(iter(tasks)) if len(tasks) == 1 else "mixed"
    task = getattr(dataset, "TASK_TYPE", None) or getattr(dataset, "task", None)
    return str(task or "unknown")


__all__ = [
    "MixedVisionDataset",
    "HomogeneousMixedBatchSampler",
    "GEN_INDEX_BASE",
]
