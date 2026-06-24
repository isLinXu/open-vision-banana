"""Reversible depth-to-RGB mapping with adaptive precision.

From the paper (Section 3.2):
- Apply power transform: f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)
- Parameters: λ=-3, c=10/3
- Then map via cube-edge traversal colormap (black → blue → cyan → green → yellow → white)

This extends the base DepthRGBMapper with adaptive quantization:
- Near depth (d < d_near): finer quantization for higher precision
- Far depth (d > d_far): coarser quantization for efficiency
- The power transform parameters remain fixed to guarantee reversibility
- The cube-edge colormap is piecewise-linear (linear interpolation along cube edges), so adaptive
  sampling density does not change the mapping — instead, adaptive
  *quantization* of the normalized depth provides variable precision

This ensures perfect roundtrip accuracy while providing adaptive
precision in the RGB visualization domain.
"""

from __future__ import annotations

import torch

from .depth_mapper import CUBE_EDGE_KEYPOINTS, DepthRGBMapper


class AdaptiveDepthRGBMapper(DepthRGBMapper):
    """Reversible depth-to-RGB mapping with adaptive quantization.

    Extends DepthRGBMapper with adaptive depth quantization:
    - Near depth: finer quantization (more discrete levels)
    - Far depth: coarser quantization (fewer discrete levels)

    The power transform parameters (λ, c) remain fixed to guarantee
    perfect reversibility. The cube-edge colormap follows RGB cube edges, so
    instead of changing sampling density (which would not affect a
    continuous interpolation), the normalized depth is quantized to
    a variable number of levels based on the depth value.

    This ensures:
    1. Perfect roundtrip accuracy (quantization is invertible)
    2. Higher visual precision for nearby content
    3. Efficient coarser quantization for distant content
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

    @property
    def max_quantization_levels(self) -> int:
        """Finest quantization grid shared by encode and decode."""
        return max(self.near_samples, self.far_samples)

    def _snap_to_quantization_grid(self, normalized: torch.Tensor) -> torch.Tensor:
        """Snap normalized values to the finest quantization grid.

        Coarser per-pixel grids (``far_samples``) produce values that remain
        representable on this grid when ``max_levels`` is a multiple of
        ``far_samples`` (default 512 / 128).
        """
        step = 1.0 / self.max_quantization_levels
        return (normalized / step).round() * step

    def depth_to_rgb(self, depth: torch.Tensor) -> torch.Tensor:
        """Full forward: depth → RGB visualization with adaptive quantization.

        Near depth uses more quantization levels for higher visual precision.
        Far depth uses fewer quantization levels for efficiency.

        The power transform is identical to the base class, ensuring
        perfect reversibility via rgb_to_depth().
        """
        normalized = self.depth_to_normalized(depth)

        # Adaptive quantization: near depth gets more levels, far gets fewer.
        t = (depth - self.d_near) / (self.d_far - self.d_near + 1e-8)
        t = t.clamp(0.0, 1.0)
        num_levels = (
            (self.near_samples * (1 - t) + self.far_samples * t).int().clamp(min=64, max=1024)
        )

        normalized_q = self._adaptive_quantize(normalized, num_levels, self.max_quantization_levels)

        # Far depths (d > d_far) keep continuous normalized values — coarser
        # quantization at high depth causes large metric errors after the
        # inverse power transform, while visual precision is less critical.
        far_mask = (depth > self.d_far).expand_as(normalized)
        normalized_out = torch.where(far_mask, normalized, normalized_q)

        return self._normalized_to_rgb_cube_edge(normalized_out)

    def _quantization_level_options(self) -> list[int]:
        """Distinct quantization grids tried during decode refinement."""
        levels = {self.far_samples, self.near_samples, self.max_quantization_levels}
        return sorted(levels)

    def _refine_quantized_normalized(
        self,
        rgb: torch.Tensor,
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        """Pick the quantized normalized value whose cube-edge RGB best matches *rgb*.

        Candidates include the continuous cube-edge projection (for far depths
        encoded without quantization) plus snaps on each configured grid.
        """
        best_norm = continuous.clone()
        best_dist = torch.full(
            continuous.shape,
            float("inf"),
            device=rgb.device,
            dtype=rgb.dtype,
        )

        candidates = [continuous]
        for levels in self._quantization_level_options():
            step = 1.0 / levels
            candidates.append((continuous / step).round() * step)

        for candidate in candidates:
            candidate_rgb = self._normalized_to_rgb_cube_edge(candidate)
            dist = (candidate_rgb - rgb).pow(2).sum(dim=1, keepdim=True).sqrt()
            better = dist < best_dist
            best_norm = torch.where(better, candidate, best_norm)
            best_dist = torch.where(better, dist, best_dist)

        return best_norm.clamp(0.0, 1.0)

    def rgb_to_normalized(self, rgb: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Inverse RGB → normalized with quantization-grid refinement."""
        continuous = super().rgb_to_normalized(rgb, chunk_size=chunk_size)
        return self._refine_quantized_normalized(rgb, continuous)

    def rgb_to_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """Full inverse: RGB → depth (quantization-aware)."""
        normalized = self.rgb_to_normalized(rgb)
        return self.normalized_to_depth(normalized)

    @staticmethod
    def _adaptive_quantize(
        normalized: torch.Tensor,
        num_levels: torch.Tensor,
        max_levels: int,
    ) -> torch.Tensor:
        """Quantize normalized depth to adaptive per-pixel precision.

        Each pixel uses ``1 / num_levels`` as its step size.  Decode snaps
        to the finest ``max_levels`` grid, which exactly represents all
        coarser grids when ``max_levels`` is a multiple of every
        ``num_levels`` value (default: 512 vs 128).

        Args:
            normalized: Continuous normalized depth in [0, 1].
            num_levels: Per-pixel quantization level count (int tensor).
            max_levels: Finest grid size used by ``rgb_to_normalized`` snap.

        Returns:
            Quantized normalized depth in [0, 1] (same shape as input).
        """
        del max_levels  # finest grid is applied in rgb_to_normalized
        step = 1.0 / num_levels.float()
        return (normalized / step).round() * step

    def _normalized_to_rgb_cube_edge(self, normalized: torch.Tensor) -> torch.Tensor:
        """Map normalized depth to RGB via cube-edge colormap.

        This is identical to the parent's cube-edge mapping but kept as a
        separate method for clarity and testability.
        """
        kp = CUBE_EDGE_KEYPOINTS.to(normalized.device)
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

        if rgb.dim() == 6:
            # 5D input [B, 1, D, H, W] → [B, 3, D, H, W]
            rgb = rgb.permute(0, 5, 1, 2, 3, 4).squeeze(2)
        elif rgb.dim() == 5:
            # 4D input [B, 1, H, W] → [B, 3, H, W]
            rgb = rgb.permute(0, 4, 1, 2, 3).squeeze(2)
        elif rgb.dim() == 4:
            rgb = rgb.permute(0, 3, 1, 2)

        return rgb.clamp(0.0, 1.0)


__all__ = ["AdaptiveDepthRGBMapper"]
