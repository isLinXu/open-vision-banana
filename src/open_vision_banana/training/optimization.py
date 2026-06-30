"""Training optimization utilities for MPS and CUDA environments.

Provides memory management, optimized DataLoader creation, learning rate
scheduler factory, and advanced data augmentation tailored for diffusion
model fine-tuning on Apple Silicon (MPS) and NVIDIA GPUs.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Memory Management
# ──────────────────────────────────────────────────────────────

class MemoryManager:
    """Adaptive memory management for MPS and CUDA training.

    Monitors memory usage and triggers cache clearing when thresholds
    are exceeded.  Essential for MPS where memory fragmentation can
    cause OOM even with sufficient total memory.
    """

    def __init__(self, device: torch.device, threshold_gb: float = 0.85):
        self.device = device
        self.threshold = threshold_gb
        self._mps_available = device.type == "mps" and torch.backends.mps.is_available()
        self._cuda_available = device.type == "cuda" and torch.cuda.is_available()

    def current_usage_gb(self) -> float:
        """Return current memory usage in GB."""
        if self._mps_available:
            # MPS memory API is limited; use allocated bytes if available
            try:
                return torch.mps.current_allocated_memory() / 1e9
            except Exception:
                return 0.0
        elif self._cuda_available:
            return torch.cuda.memory_allocated(self.device) / 1e9
        return 0.0

    def total_memory_gb(self) -> float:
        """Return total device memory in GB."""
        if self._cuda_available:
            return torch.cuda.get_device_properties(self.device).total_memory / 1e9
        # MPS: heuristic based on system memory (not precise)
        return 32.0  # Assume 32GB M1 Pro

    def should_clear(self) -> bool:
        """Check if memory usage exceeds threshold."""
        if not (self._mps_available or self._cuda_available):
            return False
        total = self.total_memory_gb()
        if total == 0:
            return False
        return self.current_usage_gb() / total > self.threshold

    def clear_cache(self) -> None:
        """Clear device memory cache."""
        if self._mps_available:
            try:
                torch.mps.empty_cache()
                logger.debug("MPS cache cleared")
            except Exception as e:
                logger.debug(f"MPS cache clear failed: {e}")
        elif self._cuda_available:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

    def maybe_clear(self) -> bool:
        """Clear cache if usage exceeds threshold. Returns True if cleared."""
        if self.should_clear():
            self.clear_cache()
            return True
        return False


# ──────────────────────────────────────────────────────────────
# Optimized DataLoader Factory
# ──────────────────────────────────────────────────────────────

def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    collate_fn: Any = None,
    device: torch.device | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool = False,
    drop_last: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create an optimized DataLoader with environment-aware defaults.

    Args:
        dataset: The dataset to load.
        batch_size: Batch size per iteration.
        shuffle: Whether to shuffle data.
        collate_fn: Custom collate function.
        device: Training device (used to determine pin_memory and num_workers).
        num_workers: Number of worker processes.  Auto-detected if None.
        pin_memory: Whether to pin memory.  Auto-detected if None.
        persistent_workers: Whether to keep workers alive between epochs.
        drop_last: Whether to drop incomplete batches.
        prefetch_factor: Number of batches to prefetch per worker.

    Returns:
        Optimized DataLoader instance.
    """
    if device is None:
        device = torch.device("cpu")

    # Auto-detect optimal settings based on device
    if num_workers is None:
        if device.type == "mps":
            # MPS: multiprocessing is problematic; use 0 workers
            num_workers = 0
        elif device.type == "cuda":
            # CUDA: use 4 workers (tuned for most GPU setups)
            num_workers = min(4, (torch.multiprocessing.cpu_count() or 4))
        else:
            num_workers = 0

    if pin_memory is None:
        # Pin memory helps CPU→GPU transfer but not MPS
        pin_memory = device.type == "cuda"

    # Persistent workers only when num_workers > 0
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None  # prefetch_factor requires num_workers > 0

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )
    if prefetch_factor is not None and num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn

    logger.info(
        f"DataLoader: batch={batch_size}, workers={num_workers}, "
        f"pin_memory={pin_memory}, persistent={persistent_workers}, "
        f"device={device.type}"
    )
    return DataLoader(**kwargs)


# ──────────────────────────────────────────────────────────────
# Learning Rate Scheduler Factory
# ──────────────────────────────────────────────────────────────

class SchedulerFactory:
    """Factory for creating learning rate schedulers with various strategies.

    Supported schedulers:
    - ``cosine``: cosine annealing with warmup (default, paper-aligned)
    - ``cosine_restarts``: cosine with warm restarts (good for long training)
    - ``polynomial``: polynomial decay (good for convergence)
    - ``constant``: constant after warmup (good for small datasets)
    - ``one_cycle``: one-cycle policy (fast convergence)
    """

    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int | None = None,
        warmup_ratio: float = 0.1,
        scheduler_type: str = "cosine",
        min_lr_ratio: float = 0.01,
        restart_period: int | None = None,
        power: float = 2.0,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Create a learning rate scheduler.

        Args:
            optimizer: The optimizer to schedule.
            total_steps: Total training steps.
            warmup_steps: Explicit warmup steps.  If None, computed from ratio.
            warmup_ratio: Fraction of total_steps for warmup.
            scheduler_type: One of "cosine", "cosine_restarts", "polynomial",
                "constant", "one_cycle".
            min_lr_ratio: Minimum LR as fraction of initial LR.
            restart_period: Period for cosine restarts (steps).  Defaults to
                total_steps // 3.
            power: Power for polynomial decay.

        Returns:
            Configured LR scheduler.
        """
        if warmup_steps is None:
            warmup_steps = max(1, int(total_steps * warmup_ratio))

        logger.info(
            f"LR scheduler: {scheduler_type}, total={total_steps}, "
            f"warmup={warmup_steps}, min_ratio={min_lr_ratio}"
        )

        if scheduler_type == "cosine":
            return SchedulerFactory._cosine_with_warmup(
                optimizer, total_steps, warmup_steps, min_lr_ratio
            )
        elif scheduler_type == "cosine_restarts":
            period = restart_period or max(1, total_steps // 3)
            return SchedulerFactory._cosine_restarts_with_warmup(
                optimizer, total_steps, warmup_steps, period, min_lr_ratio
            )
        elif scheduler_type == "polynomial":
            return SchedulerFactory._polynomial_with_warmup(
                optimizer, total_steps, warmup_steps, power, min_lr_ratio
            )
        elif scheduler_type == "constant":
            return SchedulerFactory._constant_with_warmup(
                optimizer, warmup_steps
            )
        elif scheduler_type == "one_cycle":
            return SchedulerFactory._one_cycle(
                optimizer, total_steps, warmup_steps
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    @staticmethod
    def _cosine_with_warmup(
        optimizer, total_steps, warmup_steps, min_lr_ratio
    ):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _cosine_restarts_with_warmup(
        optimizer, total_steps, warmup_steps, period, min_lr_ratio
    ):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            step_after_warmup = step - warmup_steps
            # Cosine with restarts
            cycle_progress = (step_after_warmup % period) / period
            decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
            # Global decay: each restart starts at lower LR
            global_progress = step_after_warmup / total_steps
            global_decay = 1.0 - global_progress * (1.0 - min_lr_ratio)
            return max(min_lr_ratio, decay * global_decay)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _polynomial_with_warmup(
        optimizer, total_steps, warmup_steps, power, min_lr_ratio
    ):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            decay = (1.0 - progress) ** power
            return min_lr_ratio + (1.0 - min_lr_ratio) * decay

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _constant_with_warmup(optimizer, warmup_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _one_cycle(optimizer, total_steps, warmup_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            # One cycle: up then down
            if progress < 0.4:
                return 1.0 + progress * 2.5  # ramp up to 2x
            else:
                return 2.0 - (progress - 0.4) * 2.0 / 0.6  # ramp down to 0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────
# Advanced Data Augmentation
# ──────────────────────────────────────────────────────────────

class AdvancedAugmentation:
    """Advanced data augmentation pipeline for vision task training.

    Extends the basic flip/scale/brightness with:
    - Color jitter (hue, saturation, contrast)
    - Gaussian noise (simulates diffusion process)
    - Random erasing (cutout, improves robustness)
    - Mixup / CutMix (inter-sample augmentation)
    - Multi-colormap randomization (depth-specific)
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        scale_range: tuple[float, float] = (0.8, 1.0),
        scale_prob: float = 0.3,
        brightness_range: tuple[float, float] = (-0.1, 0.1),
        brightness_prob: float = 0.5,
        color_jitter_prob: float = 0.3,
        color_jitter_strength: float = 0.1,
        gaussian_noise_prob: float = 0.2,
        gaussian_noise_std: float = 0.02,
        cutout_prob: float = 0.1,
        cutout_size: float = 0.2,
    ):
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.scale_prob = scale_prob
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.color_jitter_prob = color_jitter_prob
        self.color_jitter_strength = color_jitter_strength
        self.gaussian_noise_prob = gaussian_noise_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size

    def __call__(self, image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentation to image and target pair.

        Args:
            image: Input image [3, H, W] in [-1, 1].
            target: Target visualization [3, H, W] in [0, 1].

        Returns:
            Augmented (image, target).
        """
        # Horizontal flip (apply to both)
        if np.random.random() < self.flip_prob:
            image = torch.flip(image, dims=[2])
            target = torch.flip(target, dims=[2])

        # Random scale + crop (apply to both)
        if np.random.random() < self.scale_prob:
            scale = np.random.uniform(*self.scale_range)
            _, H, W = image.shape
            crop_h, crop_w = int(H * scale), int(W * scale)
            if crop_h < H and crop_w < W:
                top = np.random.randint(0, H - crop_h + 1)
                left = np.random.randint(0, W - crop_w + 1)
                image = image[:, top:top+crop_h, left:left+crop_w]
                target = target[:, top:top+crop_h, left:left+crop_w]
                image = F.interpolate(
                    image.unsqueeze(0), size=(H, W),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                target = F.interpolate(
                    target.unsqueeze(0), size=(H, W),
                    mode='nearest'
                ).squeeze(0)

        # Brightness (input only, not target)
        if np.random.random() < self.brightness_prob:
            brightness = 1.0 + np.random.uniform(*self.brightness_range)
            image = (image * brightness).clamp(-1.0, 1.0)

        # Color jitter (input only: hue, saturation, contrast)
        if np.random.random() < self.color_jitter_prob:
            image = self._color_jitter(image)

        # Gaussian noise (input only: simulates diffusion process)
        if np.random.random() < self.gaussian_noise_prob:
            noise = torch.randn_like(image) * self.gaussian_noise_std
            image = (image + noise).clamp(-1.0, 1.0)

        # Cutout (random erasing on input only)
        if np.random.random() < self.cutout_prob:
            image = self._cutout(image)

        return image, target

    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply hue, saturation, and contrast jitter."""
        # Convert to [0, 1] for color manipulation
        img = (image + 1.0) / 2.0  # [-1, 1] → [0, 1]
        s = self.color_jitter_strength

        # Hue shift (rotate RGB channels)
        if np.random.random() < 0.5:
            shift = np.random.uniform(-s, s)
            img = img.roll(int(shift * 3), dims=0)

        # Saturation
        if np.random.random() < 0.5:
            gray = img.mean(dim=0, keepdim=True)
            sat = 1.0 + np.random.uniform(-s, s)
            img = gray + (img - gray) * sat

        # Contrast
        if np.random.random() < 0.5:
            mean = img.mean()
            contrast = 1.0 + np.random.uniform(-s, s)
            img = (img - mean) * contrast + mean

        img = img.clamp(0, 1)
        return img * 2.0 - 1.0  # [0, 1] → [-1, 1]

    def _cutout(self, image: torch.Tensor) -> torch.Tensor:
        """Apply random erasing (cutout) to input image."""
        _, H, W = image.shape
        size = int(min(H, W) * self.cutout_size)
        if size <= 0:
            return image
        top = np.random.randint(0, H - size + 1) if H > size else 0
        left = np.random.randint(0, W - size + 1) if W > size else 0
        image = image.clone()
        image[:, top:top+size, left:left+size] = 0.0
        return image


__all__ = [
    "MemoryManager",
    "create_optimized_dataloader",
    "SchedulerFactory",
    "AdvancedAugmentation",
]