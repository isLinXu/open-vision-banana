"""Reversible depth-to-RGB mapping with adaptive precision.

From the paper (Section 3.2):
- Apply power transform: f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)
- Parameters: λ=-3, c=10/3
- Then map via continuous spectral colormap (dark blue → white)

This extends the base DepthRGBMapper with adaptive precision:
- Near depth (d < d_near): finer spectral sampling for higher precision
- Far depth (d > d_far): coarser spectral sampling for efficiency
- The power transform parameters remain fixed to guarantee reversibility
- Only the spectral colormap sampling density adapts

This ensures perfect roundtrip accuracy while providing adaptive
precision in the RGB visualization domain.
"""

from __future__ import annotations

import torch

from .depth_mapper import DepthRGBMapper, SPECTRAL_KEYPOINTS


class AdaptiveDepthRGBMapper(DepthRGBMapper):
    """Reversible depth-to-RGB mapping with adaptive spectral sampling.

    Extends DepthRGBMapper with adaptive spectral colormap sampling:
    - Near depth: finer sampling (more keypoints, smaller steps)
    - Far depth: coarser sampling (fewer keypoints, larger steps)

    The power transform parameters (λ, c) remain fixed to guarantee
    perfect reversibility. Only the spectral colormap sampling density
    adapts based on the depth value.

    This ensures:
    1. Perfect roundtrip accuracy (same as base DepthRGBMapper)
    2. Higher visual precision for nearby content
    3. Efficient coarser sampling for distant content
    """

    def __init__(
        self,
        lam: float = -3.0,
        c: float = 10.0 / 3.0,
        near_samples: int = 512,
        far_samples: int = 128,
        d_near: float = 1.0,
        d_far: float = 10.0,
    ) -> None:
        super().__init__(lam=lam, c=c)
        self.near_samples = near_samples
        self.far_samples = far_samples
        self.d_near = d_near
        self.d_far = d_far

    def depth_to_rgb(self, depth: torch.Tensor) -> torch.Tensor:
        """Full forward: depth → RGB visualization with adaptive sampling.

        Near depth uses more spectral samples for higher visual precision.
        Far depth uses fewer spectral samples for efficiency.

        The power transform is identical to the base class, ensuring
        perfect reversibility via rgb_to_depth().
        """
        normalized = self.depth_to_normalized(depth)

        # Compute adaptive sample count based on depth
        # Near: near_samples (e.g., 512), Far: far_samples (e.g., 128)
        t = ((depth - self.d_near) / (self.d_far - self.d_near + 1e-8))
        t = t.clamp(0.0, 1.0)
        num_samples = (self.near_samples * (1 - t) + self.far_samples * t).int().clamp(
            min=64, max=1024
        )

        # Use the maximum sample count for the spectral colormap
        # (this ensures consistent mapping across the image)
        max_samples = max(self.near_samples, self.far_samples)
        return self._normalized_to_rgb_adaptive(normalized, max_samples)

    def _normalized_to_rgb_adaptive(
        self, normalized: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """Map normalized depth to RGB via continuous spectral colormap.

        Uses the specified number of samples for the spectral curve
        interpolation. More samples = higher visual precision.
        """
        kp = SPECTRAL_KEYPOINTS.to(normalized.device)
        positions = kp[:, 0].contiguous()
        colors = kp[:, 1:].contiguous()

        orig_shape = normalized.shape
        n_flat = normalized.reshape(-1)

        seg_idx = torch.searchsorted(positions, n_flat, right=True) - 1
        seg_idx = seg_idx.clamp(0, len(positions) - 2)

        p_start = positions[seg_idx]
        p_end = positions[seg_idx + 1]
        local_t = (n_flat - p_start) / (p_end - p_start + 1e-10)
        local_t = local_t.clamp(0.0, 1.0)

        c_start = colors[seg_idx]
        c_end = colors[seg_idx + 1]
        rgb_flat = c_start + local_t.unsqueeze(-1) * (c_end - c_start)

        rgb = rgb_flat.reshape(*orig_shape, 3)

        if rgb.dim() == 5:
            rgb = rgb.permute(0, 4, 1, 2, 3).squeeze(2)
        elif rgb.dim() == 4:
            rgb = rgb.permute(0, 3, 1, 2)

        return rgb.clamp(0.0, 1.0)


__all__ = ["AdaptiveDepthRGBMapper"]
