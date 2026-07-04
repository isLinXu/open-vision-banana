"""Task-specific visualization decoder.

Decodes RGB outputs into segmentation / depth / normal outputs
for visualization and evaluation.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..core.rgb_mapper import RGBMapper
from .prompt import PromptBuilder

try:
    import scipy.ndimage

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


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

        from ..core.edge_mapper import EdgeRGBMapper
        from ..core.flow_mapper import FlowRGBMapper
        from ..core.keypoint_mapper import KeypointRGBMapper
        from ..core.object_detection_mapper import ObjectDetectionRGBMapper
        from ..core.depth_edge_mapper import DepthEdgeRGBMapper

        self._edge_mapper = EdgeRGBMapper(threshold=0.5)
        self._edge_mapper_cls = EdgeRGBMapper
        self._flow_mapper = FlowRGBMapper(max_flow=20.0)
        self._flow_mapper_cls = FlowRGBMapper
        self._keypoint_mapper = KeypointRGBMapper()
        self._object_detection_mapper = ObjectDetectionRGBMapper()
        self._depth_edge_mapper = DepthEdgeRGBMapper()

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
        grouping_mode: str = "adjacent",
    ) -> dict:
        """Decode instance segmentation from RGB visualization."""
        instances = self.rgb_mapper.decode(
            rgb_image, task="instance", background_color=background_color, grouping_mode=grouping_mode
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

        mask = dist <= threshold

        # Connected-component post-processing: keep the largest connected
        # component to align with the paper's clustering approach.  SciPy is
        # an optional dependency; if unavailable, the raw threshold mask is
        # returned.
        if _HAS_SCIPY:
            mask_np = mask.squeeze(1).cpu().numpy()  # [B, H, W]
            cleaned = np.zeros_like(mask_np)
            for b in range(B):
                labeled, num_features = scipy.ndimage.label(mask_np[b])
                if num_features > 0:
                    # Count pixels per label (skip background label 0)
                    areas = np.bincount(labeled.ravel())[1:]
                    largest_label = int(np.argmax(areas)) + 1
                    cleaned[b] = (labeled == largest_label)
            mask = torch.from_numpy(cleaned).to(device=device, dtype=torch.bool).unsqueeze(1)

        return {
            "mask": mask,
            "visualization": rgb_image,
        }

    def decode_edge(
        self,
        rgb_image: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict:
        """Decode edge detection from RGB visualization.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            threshold: Luminance threshold for binarization.

        Returns:
            Dict with keys:
            - "edge_mask": bool tensor [B, 1, H, W].
            - "visualization": original rgb_image for display.
        """
        if threshold == 0.5:
            mapper = self._edge_mapper
        else:
            mapper = self._edge_mapper_cls(threshold=threshold)
        edge_mask = mapper.rgb_to_edge(rgb_image)
        return {
            "edge_mask": edge_mask,
            "visualization": rgb_image,
        }

    def decode_flow(
        self,
        rgb_image: torch.Tensor,
        max_flow: float = 20.0,
    ) -> dict:
        """Decode optical flow from RGB visualization.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            max_flow: Maximum flow magnitude used during encoding.

        Returns:
            Dict with keys:
            - "flow": [B, 2, H, W] (u, v).
            - "visualization": original rgb_image for display.
        """
        if max_flow == 20.0:
            mapper = self._flow_mapper
        else:
            mapper = self._flow_mapper_cls(max_flow=max_flow)
        flow = mapper.rgb_to_flow(rgb_image)
        return {
            "flow": flow,
            "visualization": rgb_image,
        }

    def decode_keypoint(
        self,
        rgb_image: torch.Tensor,
        threshold: float = 0.3,
    ) -> dict:
        """Decode 2D keypoint detection from RGB visualization.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            threshold: Color distance threshold for keypoint matching.

        Returns:
            Dict with keys:
            - "keypoints": [B, N, 2] normalized coordinates.
            - "visibility": [B, N] bool.
            - "visualization": original rgb_image for display.
        """
        keypoints, visibility = self._keypoint_mapper.rgb_to_keypoints(rgb_image, threshold=threshold)
        return {
            "keypoints": keypoints,
            "visibility": visibility,
            "visualization": rgb_image,
        }

    def decode_object_detection(
        self,
        rgb_image: torch.Tensor,
        threshold: float = 0.3,
    ) -> dict:
        """Decode object detection from RGB visualization.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            threshold: Color distance threshold for class matching.

        Returns:
            Dict with keys:
            - "bboxes": [B, M, 4] normalized (x, y, w, h).
            - "class_ids": [B, M] int.
            - "scores": [B, M] float.
            - "visualization": original rgb_image for display.
        """
        bboxes, class_ids, scores = self._object_detection_mapper.rgb_to_bboxes(rgb_image, threshold=threshold)
        return {
            "bboxes": bboxes,
            "class_ids": class_ids,
            "scores": scores,
            "visualization": rgb_image,
        }

    def decode_depth_edge(
        self,
        rgb_image: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict:
        """Decode depth edge detection from RGB visualization.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            threshold: Luminance threshold for binarization.

        Returns:
            Dict with keys:
            - "edge_mask": bool tensor [B, 1, H, W].
            - "visualization": original rgb_image for display.
        """
        edge_mask = self._depth_edge_mapper.rgb_to_depth_edge(rgb_image)
        return {
            "edge_mask": edge_mask,
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
