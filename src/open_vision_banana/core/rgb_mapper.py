"""Unified RGB mapping dispatcher for all vision tasks.

Routes to the appropriate mapper based on task type:
- "depth": DepthRGBMapper (power transform + cube-edge colormap)
- "normal": NormalRGBMapper (direct linear mapping)
- "semantic": SegmentationDecoder (nearest-color assignment)
- "instance": SegmentationDecoder (multi-stage clustering)
"""

from __future__ import annotations

from typing import Any

import torch

from .depth_edge_mapper import DepthEdgeRGBMapper
from .depth_mapper import DepthRGBMapper
from .edge_mapper import EdgeRGBMapper
from .flow_mapper import FlowRGBMapper
from .keypoint_mapper import KeypointRGBMapper
from .normal_mapper import NormalRGBMapper
from .object_detection_mapper import ObjectDetectionRGBMapper
from .segmentation import (
    PAPER_INSTANCE_DECODING_CONFIG,
    InstanceDecodingConfig,
    SegmentationDecoder,
)


class RGBMapper:
    """Unified RGB mapping dispatcher for all vision tasks."""

    @classmethod
    def paper_reproduction(cls, depth_colormap: str = "cube_edge") -> RGBMapper:
        """Build a mapper with the paper's fixed-tolerance instance protocol."""
        return cls(
            depth_colormap=depth_colormap,
            instance_decoding_config=PAPER_INSTANCE_DECODING_CONFIG,
        )

    def __init__(
        self,
        depth_colormap: str = "cube_edge",
        *,
        depth_curve: str = "metric_power",
        depth_log_alpha: float = 0.4,
        instance_grouping_mode: str = "seed",
        instance_adaptive_color_tolerance: bool = False,
        instance_decoding_config: InstanceDecodingConfig | None = None,
    ) -> None:
        self.depth_mapper = DepthRGBMapper(
            colormap=depth_colormap,
            curve=depth_curve,
            log_alpha=depth_log_alpha,
        )
        self.normal_mapper = NormalRGBMapper()
        decoder_config = instance_decoding_config or InstanceDecodingConfig(
            grouping_mode=instance_grouping_mode,
            adaptive_color_tolerance=instance_adaptive_color_tolerance,
        )
        self.seg_decoder = SegmentationDecoder(
            **decoder_config.to_dict(),
        )
        self.edge_mapper = EdgeRGBMapper()
        self.flow_mapper = FlowRGBMapper()
        self.keypoint_mapper = KeypointRGBMapper()
        self.object_detection_mapper = ObjectDetectionRGBMapper()
        self.depth_edge_mapper = DepthEdgeRGBMapper()

    def encode(self, data: torch.Tensor, task: str, **kwargs) -> torch.Tensor:
        """Encode vision task output to RGB visualization.

        Args:
            data: Task-specific output tensor.
            task: One of "depth", "normal", "edge", "flow", "keypoint", "object_detection", "depth_edge".
            **kwargs: Task-specific extra arguments.

        Returns:
            RGB visualization tensor [B, 3, H, W] in [0, 1].
        """
        if task == "depth":
            return self.depth_mapper.depth_to_rgb(data)
        elif task == "normal":
            return self.normal_mapper.normal_to_rgb(data)
        elif task == "edge":
            return self.edge_mapper.edge_to_rgb(data)
        elif task == "flow":
            return self.flow_mapper.flow_to_rgb(data)
        elif task == "keypoint":
            visibility = kwargs.get("visibility")
            return self.keypoint_mapper.keypoints_to_rgb(data, visibility)
        elif task == "object_detection":
            class_ids = kwargs["class_ids"]
            scores = kwargs.get("scores")
            return self.object_detection_mapper.bboxes_to_rgb(data, class_ids, scores)
        elif task == "depth_edge":
            return self.depth_edge_mapper.depth_to_edge_rgb(data)
        else:
            raise ValueError(f"Encoding not supported for task: {task}")

    def decode(
        self,
        rgb: torch.Tensor,
        task: str,
        class_color_map: dict | None = None,
        background_color: tuple = (0, 0, 0),
        grouping_mode: str | None = None,
        adaptive_color_tolerance: bool | None = None,
        **kwargs,
    ) -> Any:
        """Decode RGB visualization back to vision task output.

        Args:
            rgb: Generated RGB image [B, 3, H, W] in [0, 1].
            task: One of supported vision tasks.
            class_color_map: Required for semantic segmentation.
            background_color: Background color for instance segmentation.
            grouping_mode: "adjacent" or "seed" for instance segmentation.
                ``None`` uses the mapper's recorded decoder configuration.
            adaptive_color_tolerance: Optional instance-decoder override.
            **kwargs: Task-specific extra arguments.

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
            return self.seg_decoder.decode_instance(
                rgb,
                background_color,
                grouping_mode=grouping_mode,
                adaptive_color_tolerance=adaptive_color_tolerance,
            )
        elif task == "edge":
            return self.edge_mapper.rgb_to_edge(rgb)
        elif task == "flow":
            return self.flow_mapper.rgb_to_flow(rgb)
        elif task == "keypoint":
            return self.keypoint_mapper.rgb_to_keypoints(rgb)
        elif task == "object_detection":
            return self.object_detection_mapper.rgb_to_bboxes(rgb)
        elif task == "depth_edge":
            return self.depth_edge_mapper.rgb_to_depth_edge(rgb)
        else:
            raise ValueError(f"Unknown task: {task}")


__all__ = ["RGBMapper"]
