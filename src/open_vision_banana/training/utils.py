"""Training utilities for Vision Banana.

Provides production-quality training components:
- EMAModel: Exponential moving average of model parameters
- GradientCheckpointing: Memory-efficient activation recomputation
- DDPWrapper: Distributed data parallel training coordinator
- VisionAugmentation: Task-aware geometric/color augmentations
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of model weights that evolves as an
    exponential moving average. At inference time, the EMA weights
    typically produce better results than the online weights.

    Standard practice for diffusion model training (DDPM, DiT, FLUX).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        update_after_step: int = 0,
    ) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        self.step = 0

        # Capture (name, param) pairs once to avoid repeated iteration
        # over model.named_parameters() during every update() call.
        self._param_refs: list[tuple[str, torch.Tensor]] = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        self.shadow_params = {
            name: p.data.clone() for name, p in self._param_refs
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters from online model."""
        self.step += 1
        if self.step < self.update_after_step:
            return

        effective_decay = min(
            self.decay,
            (1.0 + self.step) / (10.0 + self.step),
        )

        for name, param in self._param_refs:
            if name not in self.shadow_params:
                continue
            self.shadow_params[name].mul_(effective_decay).add_(
                param.data, alpha=1.0 - effective_decay
            )

    def apply_shadow(self, model: nn.Module) -> dict:
        """Apply EMA shadow weights to model, returning backup."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])
        return backup

    def restore_shadow(self, model: nn.Module, backup: dict) -> None:
        """Restore online weights from backup (undo apply_shadow)."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    @contextlib.contextmanager
    def applied(self, model: nn.Module) -> Iterator[None]:
        """Run the block with the EMA weights in ``model``; the online weights return after it."""
        backup = self.apply_shadow(model)
        try:
            yield
        finally:
            self.restore_shadow(model, backup)


def enable_gradient_checkpointing(
    model: nn.Module,
    layers_to_checkpoint: list[str] | None = None,
) -> None:
    """Enable gradient checkpointing on specified layers.

    Reduces VRAM from O(n) activations to O(sqrt(n)) at the cost
    of ~30% slower training. Essential for 12B-param FLUX on 80GB GPUs.
    """
    if layers_to_checkpoint is None:
        layers_to_checkpoint = [
            "block", "layer", "single_transformer", "double_transformer"
        ]

    count = 0
    for name, module in model.named_modules():
        should_checkpoint = any(key in name for key in layers_to_checkpoint)
        if should_checkpoint and hasattr(module, 'forward'):
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
                count += 1
            elif hasattr(module, 'use_checkpoint'):
                module.use_checkpoint = True
                count += 1

    if count == 0 and hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        count = 1

    logger.info(f"Gradient checkpointing enabled on {count} modules")


class DDPWrapper:
    """Wraps model and training loop for multi-GPU distributed training."""

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        local_rank: int = 0,
        backend: str = "nccl",
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self._initialized = False

    def initialize(self) -> None:
        """Initialize distributed process group."""
        if self.world_size <= 1:
            return
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
            )
            self._initialized = True
            logger.info(f"DDP initialized: rank={self.rank}/{self.world_size}")

    def wrap(self, model: nn.Module, device: str | None = None) -> nn.Module:
        """Wrap model with DistributedDataParallel."""
        self.initialize()
        if self.world_size <= 1:
            return model
        device = device or f"cuda:{self.local_rank}"
        model = model.to(device)
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.local_rank] if device.startswith("cuda") else None,
            output_device=self.local_rank if device.startswith("cuda") else None,
        )

    def cleanup(self) -> None:
        """Cleanup distributed process group."""
        if self._initialized:
            torch.distributed.destroy_process_group()

    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.world_size > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def is_main(self) -> bool:
        """Whether this is the main (rank 0) process."""
        return self.rank == 0


class VisionAugmentation:
    """Task-aware geometric and color augmentations for vision training.

    Applies consistent geometric transforms to both input image and
    target RGB visualization, plus color augmentations only to the input.

    For normal estimation tasks, horizontal flip must also negate the x-component
    of the normal (the R channel in the (1-x)/2 encoding), ensuring the
    flipped image and its ground-truth normal map remain consistent.
    """

    def __init__(
        self,
        resolution: int = 512,
        hflip_prob: float = 0.5,
        crop_scale: tuple[float, float] = (0.8, 1.0),
        color_jitter: dict | None = None,
        task: str = "depth",
    ) -> None:
        self.resolution = resolution
        self.hflip_prob = hflip_prob
        self.crop_scale = crop_scale
        self.color_jitter = color_jitter or {}
        self.task = task

    def __call__(
        self,
        image: torch.Tensor,
        target_rgb: torch.Tensor,
        task: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations consistently to image-target pair.

        Args:
            image: Input image [C, H, W] in [-1, 1].
            target_rgb: Target visualization [C, H, W] in [0, 1].
            task: Per-sample task override for mixed-task batches.

        Returns:
            Augmented (image, target_rgb) tuple.
        """
        do_flip = torch.rand(1).item() < self.hflip_prob

        if do_flip:
            image = torch.flip(image, [-1])
            target_rgb = torch.flip(target_rgb, [-1])
            # Fix normal map consistency: horizontal flip negates the x-component
            # In NormalRGBMapper encoding: R = (1-x)/2, so flipped-x means R → 1-R
            if (task or self.task) == "normal":
                target_rgb = target_rgb.clone()
                target_rgb[0] = 1.0 - target_rgb[0]

        # Native-depth admission verified the loader's output geometry. This
        # legacy augmentation receives RGB only, so cannot verify another crop.
        # Keep depth geometry intact; flips and input-only jitter remain safe.
        if self.crop_scale[0] < 1.0 and (task or self.task) not in {
            "depth", "metric_depth", "depth_estimation", "depth_edge"
        }:
            _, H, W = image.shape
            scale = torch.FloatTensor(1).uniform_(*self.crop_scale).item()
            crop_h, crop_w = int(H * scale), int(W * scale)
            top = torch.randint(0, H - crop_h + 1, (1,)).item()
            left = torch.randint(0, W - crop_w + 1, (1,)).item()

            image = image[:, top:top+crop_h, left:left+crop_w]
            target_rgb = target_rgb[:, top:top+crop_h, left:left+crop_w]

            image = F.interpolate(
                image.unsqueeze(0), size=(self.resolution, self.resolution),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            # Use nearest interpolation for target_rgb to preserve
            # sharp boundaries in depth maps and segmentation masks.
            # Bilinear would blur depth discontinuities and class boundaries.
            target_rgb = F.interpolate(
                target_rgb.unsqueeze(0), size=(self.resolution, self.resolution),
                mode='nearest',
            ).squeeze(0)

        if self.color_jitter:
            brightness = self.color_jitter.get('brightness', 0.0)
            contrast = self.color_jitter.get('contrast', 0.0)

            if brightness > 0:
                factor = 1.0 + torch.FloatTensor(1).uniform_(-brightness, brightness).item()
                image = image * factor

            if contrast > 0:
                mean = image.mean(dim=(-2, -1), keepdim=True)
                factor = 1.0 + torch.FloatTensor(1).uniform_(-contrast, contrast).item()
                image = (image - mean) * factor + mean

            # Clamp to valid range after color augmentation to prevent NaN/Inf in loss.
            image = image.clamp(-1.0, 1.0)

        return image, target_rgb


__all__ = [
    "EMAModel",
    "enable_gradient_checkpointing",
    "DDPWrapper",
    "VisionAugmentation",
]
