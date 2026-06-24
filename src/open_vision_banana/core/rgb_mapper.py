"""Unified RGB mapping dispatcher for all vision tasks.

Routes to the appropriate mapper based on task type:
- "depth": DepthRGBMapper (power transform + cube-edge colormap)
- "normal": NormalRGBMapper (direct linear mapping)
- "semantic": SegmentationDecoder (nearest-color assignment)
- "instance": SegmentationDecoder (multi-stage clustering)
"""

from __future__ import annotations

import torch

from .depth_edge_mapper import DepthEdgeRGBMapper
from .depth_mapper import DepthRGBMapper
from .edge_mapper import EdgeRGBMapper
from .flow_mapper import FlowRGBMapper
from .keypoint_mapper import KeypointRGBMapper
from .normal_mapper import NormalRGBMapper
from .object_detection_mapper import ObjectDetectionRGBMapper
from .segmentation import SegmentationDecoder


class RGBMapper:
    """Unified RGB mapping dispatcher for all vision tasks."""

    def __init__(self) -> None:
        self.depth_mapper = DepthRGBMapper()
        self.normal_mapper = NormalRGBMapper()
        self.seg_decoder = SegmentationDecoder()
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
        grouping_mode: str = "adjacent",
        **kwargs,
    ) -> torch.Tensor:
        """Decode RGB visualization back to vision task output.

        Args:
            rgb: Generated RGB image [B, 3, H, W] in [0, 1].
            task: One of supported vision tasks.
            class_color_map: Required for semantic segmentation.
            background_color: Background color for instance segmentation.
            grouping_mode: "adjacent" or "seed" for instance segmentation.
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
            return self.seg_decoder.decode_instance(rgb, background_color, grouping_mode=grouping_mode)
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
