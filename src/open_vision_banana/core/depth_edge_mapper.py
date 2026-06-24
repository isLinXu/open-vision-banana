"""Depth edge detection RGB encoder/decoder.

Depth edge: edges detected from depth discontinuities (not image edges).
The depth gradient exceeds a threshold -> white edge, otherwise black background.

Encoding: depth map -> compute depth gradient -> threshold -> white edges on black background.
Decoding: RGB -> luminance threshold -> binary depth edge map.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DepthEdgeRGBMapper:
    """Depth edge detection RGB encoder/decoder.

    Encoding: depth gradient -> white edges on black background.
    Decoding: luminance threshold -> binary depth edge map.
    """

    def __init__(self, grad_threshold: float = 1.0, line_width: int = 1) -> None:
        self.grad_threshold = grad_threshold
        self.line_width = line_width

    def depth_to_edge_rgb(self, depth: torch.Tensor) -> torch.Tensor:
        """Encode depth map to depth edge RGB visualization.

        Args:
            depth: [B, 1, H, W] metric depth in meters.

        Returns:
            RGB tensor [B, 3, H, W] in [0, 1]. White edges, black background.
        """
        B, _, H, W = depth.shape

        # Compute depth gradient using Sobel-like kernels
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-10)

        edge_mask = grad_magnitude > self.grad_threshold
        white = edge_mask.float().repeat(1, 3, 1, 1)
        return white

    def rgb_to_depth_edge(self, rgb: torch.Tensor) -> torch.Tensor:
        """Decode RGB visualization back to depth edge mask.

        Args:
            rgb: [B, 3, H, W] in [0, 1].

        Returns:
            Binary depth edge mask [B, 1, H, W] (bool).
        """
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance > 0.5


__all__ = ["DepthEdgeRGBMapper"]
