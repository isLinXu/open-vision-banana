"""Edge detection RGB encoder/decoder.

From the paper approach:
- Encoding: edge mask (bool) -> white (1,1,1) edges, black (0,0,0) background
- Decoding: RGB -> luminance threshold -> binary edge map
"""

from __future__ import annotations

import torch


class EdgeRGBMapper:
    """Edge detection RGB encoder/decoder.

    Encoding: edge mask (bool) -> white (1,1,1) edges, black (0,0,0) background.
    Decoding: RGB -> luminance threshold -> binary edge map.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def edge_to_rgb(self, edge_mask: torch.Tensor) -> torch.Tensor:
        """Encode edge mask to RGB visualization.

        Args:
            edge_mask: Binary edge mask [B, 1, H, W] (bool or float).
                True / 1.0 = edge, False / 0.0 = background.

        Returns:
            RGB tensor [B, 3, H, W] in [0, 1].
            White (1,1,1) for edges, black (0,0,0) for background.
        """
        white = edge_mask.float().clamp(0.0, 1.0)
        return white.repeat(1, 3, 1, 1)

    def rgb_to_edge(self, rgb: torch.Tensor) -> torch.Tensor:
        """Decode RGB visualization back to edge mask.

        Uses luminance = 0.299*R + 0.587*G + 0.114*B, then threshold.

        Args:
            rgb: RGB tensor [B, 3, H, W] in [0, 1].

        Returns:
            Binary edge mask [B, 1, H, W] (bool).
        """
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return luminance > self.threshold


__all__ = ["EdgeRGBMapper"]
