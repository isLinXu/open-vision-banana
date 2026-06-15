"""Reversible normal-to-RGB mapping.

From the paper (Section 3.2):
Camera-space normals (right-handed: +x right, +y up, +z toward camera):
- R = trunc((1-x)/2) * 255
- G = trunc((1+y)/2) * 255
- B = trunc((1+z)/2) * 255

This maps unit normal vectors (x,y,z) ∈ [-1,1]³ to RGB ∈ [0,255]³.
"""

from __future__ import annotations

import torch


class NormalRGBMapper:
    """Reversible normal-to-RGB mapping.

    Camera-space normals (right-handed: +x right, +y up, +z toward camera):
    - R = (1-x)/2, G = (1+y)/2, B = (1+z)/2

    This maps unit normal vectors (x,y,z) ∈ [-1,1]³ to RGB ∈ [0,1]³.
    """

    def normal_to_rgb(self, normal: torch.Tensor) -> torch.Tensor:
        """Map surface normals to RGB.

        Args:
            normal: Normal vectors [B, 3, H, W] with values in [-1, 1].

        Returns:
            RGB tensor [B, 3, H, W] with values in [0, 1].
        """
        x, y, z = normal[:, 0:1], normal[:, 1:2], normal[:, 2:3]
        r = (1.0 - x) / 2.0
        g = (1.0 + y) / 2.0
        b = (1.0 + z) / 2.0
        rgb = torch.cat([r, g, b], dim=1)
        return rgb.clamp(0.0, 1.0)

    def rgb_to_normal(self, rgb: torch.Tensor) -> torch.Tensor:
        """Inverse: RGB → surface normals.

        Args:
            rgb: RGB tensor [B, 3, H, W] with values in [0, 1].

        Returns:
            Normal vectors [B, 3, H, W] with values in [-1, 1].
        """
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        x = 1.0 - 2.0 * r
        y = 2.0 * g - 1.0
        z = 2.0 * b - 1.0
        normal = torch.cat([x, y, z], dim=1)

        # Renormalize to unit vectors
        norm = normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normal = normal / norm
        return normal


__all__ = ["NormalRGBMapper"]
