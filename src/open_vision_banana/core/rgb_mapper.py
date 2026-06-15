"""Unified RGB mapping dispatcher for all vision tasks.

Routes to the appropriate mapper based on task type:
- "depth": DepthRGBMapper (power transform + spectral colormap)
- "normal": NormalRGBMapper (direct linear mapping)
- "semantic": SegmentationDecoder (nearest-color assignment)
- "instance": SegmentationDecoder (multi-stage clustering)
"""

from __future__ import annotations

import torch

from .depth_mapper import DepthRGBMapper
from .normal_mapper import NormalRGBMapper
from .segmentation import SegmentationDecoder


class RGBMapper:
    """Unified RGB mapping dispatcher for all vision tasks."""

    def __init__(self) -> None:
        self.depth_mapper = DepthRGBMapper()
        self.normal_mapper = NormalRGBMapper()
        self.seg_decoder = SegmentationDecoder()

    def encode(self, data: torch.Tensor, task: str) -> torch.Tensor:
        """Encode vision task output to RGB visualization.

        Args:
            data: Task-specific output tensor.
            task: One of "depth", "normal".

        Returns:
            RGB visualization tensor [B, 3, H, W] in [0, 1].
        """
        if task == "depth":
            return self.depth_mapper.depth_to_rgb(data)
        elif task == "normal":
            return self.normal_mapper.normal_to_rgb(data)
        else:
            raise ValueError(f"Encoding not supported for task: {task}")

    def decode(
        self,
        rgb: torch.Tensor,
        task: str,
        class_color_map: dict | None = None,
        background_color: tuple = (0, 0, 0),
    ) -> torch.Tensor:
        """Decode RGB visualization back to vision task output.

        Args:
            rgb: Generated RGB image [B, 3, H, W] in [0, 1].
            task: One of "depth", "normal", "semantic", "instance".
            class_color_map: Required for semantic segmentation.
            background_color: Background color for instance segmentation.

        Returns:
            Decoded vision task output.
        """
        if task == "depth":
            return self.depth_mapper.rgb_to_depth(rgb)
        elif task == "normal":
            return self.normal_mapper.rgb_to_normal(rgb)
        elif task == "semantic":
            if class_color_map is None:
                raise ValueError("class_color_map required for semantic segmentation")
            return self.seg_decoder.decode_semantic(rgb, class_color_map)
        elif task == "instance":
            return self.seg_decoder.decode_instance(rgb, background_color)
        else:
            raise ValueError(f"Unknown task: {task}")


__all__ = ["RGBMapper"]
