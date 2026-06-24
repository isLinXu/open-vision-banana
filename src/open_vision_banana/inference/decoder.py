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
        target_color: tuple[int, int, int] = (255, 255, 255),
        threshold: float | None = None,
    ) -> dict:
        """Decode referring expression segmentation from RGB visualization.

        From the paper (Page 3):
        "By prompting the model to 'Segment the skateboard category in pure
        yellow (<255, 255, 0>)', we can easily parse the mask for skateboard by
        clustering pixels whose values are close to <255, 255, 0>."

        Computes the L2 distance in RGB space from each pixel to the
        target color, then thresholds to produce a binary mask.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            target_color: Target RGB color for the referred object,
                as (R, G, B) in [0, 255]. Default is white (255, 255, 255).
            threshold: Distance threshold in [0, 1] normalized RGB space.
                If None, uses an adaptive threshold based on the distance
                distribution (mean - 0.5 * std heuristic).

        Returns:
            Dict with keys:
            - "mask": bool tensor [B, 1, H, W], True = referred region.
            - "visualization": original rgb_image for display.
        """
        B, _, H, W = rgb_image.shape
        device = rgb_image.device

        # Normalize target color to [0, 1]
        target = (
            torch.tensor(target_color, dtype=torch.float32, device=device)
            / 255.0
        ).view(1, 3, 1, 1)

        # L2 distance in RGB space [B, 1, H, W]
        dist = torch.norm(rgb_image - target, dim=1, keepdim=True)

        if threshold is None:
            # Adaptive threshold: mean distance minus half std
            # This works well when the referred object is a distinct color
            # against a different background.
            mean_dist = dist.mean(dim=(-2, -1), keepdim=True)
            std_dist = dist.std(dim=(-2, -1), keepdim=True)
            threshold = (mean_dist - 0.5 * std_dist).clamp(min=0.05, max=0.5)

        mask = dist < threshold
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
