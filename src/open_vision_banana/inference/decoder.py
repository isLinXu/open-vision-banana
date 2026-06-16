"""Task-specific visualization decoder.

Decodes RGB outputs into segmentation / depth / normal outputs
for visualization and evaluation.
"""

from __future__ import annotations

import math

import torch

from ..core.rgb_mapper import RGBMapper
from .prompt import PromptBuilder


class VisualizationDecoder:
    """Decodes mapped RGB outputs into task-specific visualizations.

    Provides high-level API for:
    1. Generating task-specific prompts
    2. Decoding RGB outputs back to vision task outputs
    3. Creating visualization images for display
    """

    def __init__(self, rgb_mapper: RGBMapper | None = None) -> None:
        self.rgb_mapper = rgb_mapper or RGBMapper()
        self.prompt_builder = PromptBuilder()

    def decode_depth(
        self,
        rgb_image: torch.Tensor,
        return_visualization: bool = True,
    ) -> dict:
        """Decode depth estimation from RGB visualization."""
        depth = self.rgb_mapper.decode(rgb_image, task="depth")
        result = {"depth": depth}

        if return_visualization:
            result["visualization"] = self._depth_colormap(depth)

        return result

    def decode_normal(
        self,
        rgb_image: torch.Tensor,
        return_visualization: bool = True,
    ) -> dict:
        """Decode surface normal estimation from RGB visualization."""
        normal = self.rgb_mapper.decode(rgb_image, task="normal")
        result = {"normal": normal}

        if return_visualization:
            result["visualization"] = rgb_image

        return result

    def decode_semantic(
        self,
        rgb_image: torch.Tensor,
        class_color_map: dict[str, tuple[int, int, int]],
    ) -> dict:
        """Decode semantic segmentation from RGB visualization."""
        mask = self.rgb_mapper.decode(
            rgb_image, task="semantic", class_color_map=class_color_map
        )
        return {
            "mask": mask,
            "class_names": list(class_color_map.keys()),
        }

    def decode_instance(
        self,
        rgb_image: torch.Tensor,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> dict:
        """Decode instance segmentation from RGB visualization."""
        instances = self.rgb_mapper.decode(
            rgb_image, task="instance", background_color=background_color
        )
        return {"instances": instances}

    def decode_referring(
        self,
        rgb_image: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict:
        """Decode referring expression segmentation from RGB visualization.

        The model renders the referred object as white pixels on a black
        background.  This method binarises the result.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            threshold: Mean channel threshold for foreground classification.

        Returns:
            Dict with keys:
            - "mask": bool tensor [B, 1, H, W], True = referred region.
            - "visualization": original rgb_image for display.
        """
        mask = rgb_image.mean(dim=1, keepdim=True) > threshold
        return {
            "mask": mask,
            "visualization": rgb_image,
        }

    @staticmethod
    def _depth_colormap(depth: torch.Tensor) -> torch.Tensor:
        """Apply viridis-like colormap to depth for visualization."""
        d_min = depth.amin(dim=(-2, -1), keepdim=True)
        d_max = depth.amax(dim=(-2, -1), keepdim=True)
        d_norm = (depth - d_min) / (d_max - d_min + 1e-8)

        r = torch.clamp(2.0 * d_norm - 0.5, 0, 1)
        g = torch.clamp(torch.sin(d_norm * math.pi), 0, 1)
        b = torch.clamp(1.5 - 2.0 * d_norm, 0, 1)

        return torch.cat([r, g, b], dim=1)


__all__ = ["VisualizationDecoder"]
