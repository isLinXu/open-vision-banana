"""VAE latent cache wrapper for datasets.

Caches VAE-encoded latents for target_rgb and input_image to avoid
repeatedly encoding the same images on every epoch. This is especially
beneficial on MPS where VAE encoding is a significant bottleneck.

Usage:
    dataset = HypersimDepthDataset(...)
    cached_dataset = LatentCacheDataset(dataset, vae, model_cfg, device, dtype)
    # After the first epoch, VAE encoding is skipped entirely.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LatentCacheDataset(Dataset):
    """Wraps a vision dataset and caches VAE latents.

    On first access, encodes target_rgb and image to latent space using
    the provided VAE, then caches the result. Subsequent accesses return
    the cached latents directly, skipping VAE encoding entirely.

    This trades memory for compute: each cached sample uses ~64KB of
    latent storage (at 256px, 4×32×32 = 4096 float16 values = 8KB per
    channel × 4 channels = 32KB for target + 32KB for input = 64KB).
    For 2K samples, total cache ≈ 128MB — well within MPS limits.

    The cache can be disabled at any time by calling ``clear_cache()``.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        vae: Any,
        model_cfg: Any,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
        cache_input: bool = True,
        cache_target: bool = True,
    ) -> None:
        self.base = base_dataset
        self.vae = vae
        self.model_cfg = model_cfg
        self.device = torch.device(device)
        self.dtype = dtype
        self.cache_input = cache_input
        self.cache_target = cache_target

        # Lazy caches: index → latent tensor
        self._input_cache: dict[int, torch.Tensor] = {}
        self._target_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.base)

    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode pixels [3, H, W] in [-1, 1] to latent [C, H', W']."""
        from open_vision_banana.model_registry import vae_encode

        # Ensure pixels are in the right format
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)  # [1, 3, H, W]
        pixels = pixels.to(self.dtype)
        with torch.no_grad():
            latent = vae_encode(self.vae, pixels, self.model_cfg)
        return latent.squeeze(0).cpu()  # [C, H', W']

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]

        # Cache target latent
        if self.cache_target and "target_rgb" in sample and idx not in self._target_cache:
            target = sample["target_rgb"]
            if target.dim() == 3:
                target = target.unsqueeze(0)
            target = (target * 2.0 - 1.0).clamp(-1.0, 1.0)
            self._target_cache[idx] = self._encode(target)
            sample["target_latent"] = self._target_cache[idx]
            del sample["target_rgb"]  # free memory
        elif self.cache_target and idx in self._target_cache:
            sample["target_latent"] = self._target_cache[idx]
            sample.pop("target_rgb", None)

        # Cache input latent
        if self.cache_input and "image" in sample and idx not in self._input_cache:
            img = sample["image"]
            self._input_cache[idx] = self._encode(img)
            sample["input_latent"] = self._input_cache[idx]
            del sample["image"]  # free memory
        elif self.cache_input and idx in self._input_cache:
            sample["input_latent"] = self._input_cache[idx]
            sample.pop("image", None)

        return sample

    def clear_cache(self) -> None:
        """Clear all cached latents."""
        self._input_cache.clear()
        self._target_cache.clear()
        logger.info("Latent cache cleared")

    @property
    def cache_size(self) -> int:
        """Number of cached entries."""
        return len(self._input_cache) + len(self._target_cache)

    def warm_cache(self, max_items: int | None = None) -> None:
        """Pre-encode all samples to populate the cache.

        Args:
            max_items: Maximum number of items to cache. None for all.
        """
        n = min(max_items or len(self), len(self))
        for i in range(n):
            _ = self[i]
        logger.info(f"Cache warmed: {n} items ({self.cache_size} latents cached)")


__all__ = ["LatentCacheDataset"]
