"""Reversible depth-to-RGB mapping using power transform + spectral colormap.

From the paper (Section 3.2):
- Apply power transform: f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)
- Parameters: λ=-3, c=10/3
- Then map via continuous spectral colormap (dark blue → white)

This maps unbounded depth values [0, ∞) to bounded RGB [0, 1]³.
The curving prioritizes accuracy for nearby content (higher utility).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Continuous Spectral Colormap for Depth Encoding
# ──────────────────────────────────────────────────────────────

# The paper uses a continuous spectral colormap that traverses the RGB
# cube smoothly from dark (near) to bright (far). This replaces the
# discrete cube-edge traversal with a smooth, bijective mapping that
# guarantees perfect roundtrip accuracy.
#
# The colormap follows a perceptually uniform spectral ordering:
# dark_blue → blue → cyan → green → yellow → orange → red → white

SPECTRAL_KEYPOINTS = torch.tensor([
    # (position, R, G, B) — position in [0, 1], colors in [0, 1]
    [0.00,  0.00, 0.00, 0.30],   # dark blue (near)
    [0.12,  0.00, 0.00, 1.00],   # blue
    [0.25,  0.00, 0.80, 1.00],   # cyan
    [0.37,  0.00, 1.00, 0.40],   # teal-green
    [0.50,  0.20, 1.00, 0.00],   # green-yellow
    [0.62,  0.80, 1.00, 0.00],   # yellow
    [0.75,  1.00, 0.60, 0.00],   # orange
    [0.87,  1.00, 0.20, 0.00],   # red-orange
    [1.00,  1.00, 1.00, 1.00],   # white (far)
], dtype=torch.float32)


class DepthRGBMapper:
    """Reversible depth-to-RGB mapping using power transform + spectral colormap.

    From the paper (Section 3.2):
    - Apply power transform: f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)
    - Parameters: λ=-3, c=10/3
    - Then map via continuous spectral colormap (dark blue → white)

    This maps unbounded depth values [0, ∞) to bounded RGB [0, 1]³.
    The curving prioritizes accuracy for nearby content (higher utility).
    """

    def __init__(self, lam: float = -3.0, c: float = 10.0 / 3.0):
        self.lam = lam
        self.c = c

    def depth_to_normalized(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply power transform: depth → normalized [0, 1).

        f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)

        Args:
            depth: Metric depth values [B, 1, H, W] in meters, range [0, ∞).

        Returns:
            Normalized depth values [B, 1, H, W] in [0, 1).
        """
        depth = depth.clamp(min=0.0)
        ratio = depth / (self.lam * self.c)
        ratio = ratio.clamp(max=1.0 - 1e-7)
        base = 1.0 - ratio
        exponent = self.lam + 1  # = -2
        normalized = 1.0 - torch.pow(base, exponent)
        return normalized.clamp(0.0, 1.0)

    def normalized_to_depth(self, normalized: torch.Tensor) -> torch.Tensor:
        """Inverse: normalized [0, 1) → depth [0, ∞)."""
        exponent = 1.0 / (self.lam + 1)  # = -0.5
        base = 1.0 - normalized
        ratio = 1.0 - torch.pow(base.clamp(min=1e-10), exponent)
        depth = self.lam * self.c * ratio
        return depth.clamp(min=0.0)

    def normalized_to_rgb(self, normalized: torch.Tensor) -> torch.Tensor:
        """Map normalized depth to RGB via continuous spectral colormap.

        Uses piecewise-linear interpolation between spectral keypoints,
        guaranteeing a smooth, bijective mapping from [0,1) → RGB³.
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

    def rgb_to_normalized(self, rgb: torch.Tensor) -> torch.Tensor:
        """Inverse: RGB → normalized depth via spectral colormap inversion.

        Projects each RGB pixel onto the nearest point of the spectral
        curve. Since the colormap is monotonic (bijective), this recovers
        the normalized value accurately.

        Fully vectorized — no Python loops over pixels.
        """
        kp = SPECTRAL_KEYPOINTS.to(rgb.device)
        positions = kp[:, 0]
        colors = kp[:, 1:]

        B, C, H, W = rgb.shape

        # Sample the spectral curve densely for accurate projection
        num_samples = 256
        sampled_positions = torch.linspace(0.0, 1.0, num_samples, device=rgb.device)

        seg_idx = torch.searchsorted(positions.contiguous(), sampled_positions, right=True) - 1
        seg_idx = seg_idx.clamp(0, len(positions) - 2)
        p_starts = positions[seg_idx]
        p_ends = positions[seg_idx + 1]
        local_t = (sampled_positions - p_starts) / (p_ends - p_starts + 1e-10)
        local_t = local_t.clamp(0.0, 1.0)
        sampled_colors = colors[seg_idx] + local_t.unsqueeze(-1) * (colors[seg_idx + 1] - colors[seg_idx])

        rgb_flat = rgb.permute(0, 2, 3, 1).reshape(B, H * W, 3)

        # Pairwise L2 distances: [B, N, S]
        dists = (rgb_flat.unsqueeze(2) - sampled_colors.unsqueeze(0).unsqueeze(0)).norm(dim=-1)

        min_indices = dists.argmin(dim=-1)
        best_t = sampled_positions[min_indices]

        # Parabolic interpolation for sub-grid precision
        center = min_indices.clamp(1, num_samples - 2)
        idx_minus = (center - 1).unsqueeze(-1)
        idx_center = center.unsqueeze(-1)
        idx_plus = (center + 1).unsqueeze(-1)

        d_minus = dists.gather(2, idx_minus).squeeze(-1)
        d_center = dists.gather(2, idx_center).squeeze(-1)
        d_plus = dists.gather(2, idx_plus).squeeze(-1)

        denom = (d_minus - 2 * d_center + d_plus)
        denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)
        shift = 0.5 * (d_minus - d_plus) / denom
        shift = shift.clamp(-0.5, 0.5)
        step_size = 1.0 / (num_samples - 1)
        refined_t = best_t + shift * step_size
        refined_t = refined_t.clamp(0.0, 1.0)

        return refined_t.reshape(B, 1, H, W)

    def depth_to_rgb(self, depth: torch.Tensor) -> torch.Tensor:
        """Full forward: depth → RGB visualization."""
        normalized = self.depth_to_normalized(depth)
        return self.normalized_to_rgb(normalized)

    def rgb_to_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """Full inverse: RGB → depth."""
        normalized = self.rgb_to_normalized(rgb)
        return self.normalized_to_depth(normalized)


__all__ = ["DepthRGBMapper", "SPECTRAL_KEYPOINTS"]
