"""Task-specific visualization decoder.

Decodes RGB outputs into segmentation / depth / normal outputs
for visualization and evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..core.rgb_mapper import RGBMapper
from .prompt import PromptBuilder

try:
    import scipy.ndimage

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ──────────────────────────────────────────────────────────────
# ADE20K 151-class segmentation palette (0=background, 1-150=foreground)
# Matches vb_snvc_train_v2/segmentation/palette_info.json
# ──────────────────────────────────────────────────────────────
ADE20K_151_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0), 1: (0, 255, 235), 2: (4, 250, 7), 3: (0, 255, 133),
    4: (120, 120, 120), 5: (255, 0, 235), 6: (235, 255, 7), 7: (255, 41, 0),
    8: (10, 255, 71), 9: (255, 0, 255), 10: (255, 8, 41), 11: (255, 0, 214),
    12: (7, 255, 255), 13: (255, 224, 0), 14: (6, 51, 255), 15: (0, 255, 184),
    16: (4, 200, 3), 17: (255, 0, 152), 18: (8, 255, 51), 19: (245, 0, 255),
    20: (140, 140, 140), 21: (255, 0, 5), 22: (143, 255, 140), 23: (255, 51, 0),
    24: (255, 194, 7), 25: (255, 245, 0), 26: (140, 140, 140), 27: (255, 0, 31),
    28: (255, 122, 8), 29: (255, 214, 0), 30: (20, 255, 0), 31: (255, 163, 0),
    32: (0, 0, 0), 33: (133, 255, 0), 34: (224, 5, 255), 35: (255, 0, 0),
    36: (180, 120, 120), 37: (245, 0, 255), 38: (150, 5, 61), 39: (255, 163, 0),
    40: (255, 41, 10), 41: (255, 0, 163), 42: (255, 5, 153), 43: (255, 0, 41),
    44: (224, 255, 8), 45: (0, 0, 0), 46: (235, 12, 255), 47: (255, 0, 245),
    48: (120, 120, 80), 49: (0, 0, 0), 50: (255, 6, 82), 51: (255, 0, 112),
    52: (0, 0, 0), 53: (255, 245, 0), 54: (204, 255, 4), 55: (0, 255, 163),
    56: (0, 0, 0), 57: (0, 0, 0), 58: (250, 10, 15), 59: (0, 0, 0),
    60: (0, 255, 20), 61: (0, 255, 122), 62: (31, 255, 0), 63: (255, 0, 133),
    64: (204, 70, 3), 65: (255, 235, 0), 66: (220, 220, 220), 67: (0, 255, 214),
    68: (61, 230, 250), 69: (92, 0, 255), 70: (112, 9, 255), 71: (255, 0, 184),
    72: (255, 31, 0), 73: (255, 235, 0), 74: (255, 71, 0), 75: (255, 0, 82),
    76: (255, 224, 0), 77: (0, 0, 0), 78: (0, 235, 255), 79: (0, 0, 0),
    80: (11, 102, 255), 81: (255, 0, 92), 82: (8, 255, 214), 83: (255, 0, 71),
    84: (255, 9, 224), 85: (255, 255, 0), 86: (255, 184, 6), 87: (0, 255, 245),
    88: (153, 255, 0), 89: (255, 0, 173), 90: (0, 173, 255), 91: (255, 0, 204),
    92: (0, 0, 0), 93: (255, 0, 61), 94: (11, 200, 200), 95: (0, 0, 0),
    96: (0, 102, 200), 97: (0, 0, 0), 98: (255, 9, 92), 99: (255, 102, 0),
    100: (255, 6, 51), 101: (255, 0, 235), 102: (0, 0, 0), 103: (255, 214, 0),
    104: (0, 0, 0), 105: (0, 0, 0), 106: (0, 0, 0), 107: (0, 0, 0),
    108: (0, 0, 0), 109: (255, 0, 122), 110: (0, 0, 0), 111: (0, 255, 102),
    112: (255, 7, 71), 113: (255, 0, 20), 114: (7, 255, 224), 115: (0, 0, 0),
    116: (9, 7, 230), 117: (51, 0, 255), 118: (0, 0, 0), 119: (255, 184, 0),
    120: (0, 0, 255), 121: (255, 0, 10), 122: (31, 0, 255), 123: (0, 0, 0),
    124: (0, 0, 0), 125: (0, 255, 10), 126: (255, 82, 0), 127: (0, 0, 0),
    128: (6, 230, 230), 129: (0, 0, 0), 130: (120, 120, 70), 131: (0, 0, 0),
    132: (80, 50, 50), 133: (255, 224, 0), 134: (0, 0, 0), 135: (255, 204, 0),
    136: (102, 8, 255), 137: (255, 0, 102), 138: (160, 150, 20), 139: (255, 0, 51),
    140: (255, 61, 6), 141: (0, 0, 0), 142: (0, 163, 255), 143: (255, 82, 0),
    144: (204, 5, 255), 145: (255, 184, 0), 146: (0, 0, 0), 147: (0, 0, 0),
    148: (230, 230, 230), 149: (0, 255, 61), 150: (255, 51, 7),
}


def get_ade20k_151_color_map() -> dict[str, tuple[int, int, int]]:
    """Return ADE20K 151-class palette as string-keyed dict for decode_semantic."""
    return {f"class_{k}": v for k, v in ADE20K_151_PALETTE.items()}


def load_segmentation_color_map(
    path: str | Path,
) -> dict[str, tuple[int, int, int]]:
    """Load a dataset palette without assuming the legacy v2 ADE20K colors."""
    palette_path = Path(path).expanduser().resolve()
    if palette_path.is_dir():
        candidates = (
            palette_path / "segmentation" / "_unified_palette.json",
            palette_path / "_unified_palette.json",
            palette_path / "segmentation" / "palette_info.json",
            palette_path / "palette_info.json",
        )
        palette_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not palette_path.is_file():
        raise FileNotFoundError(f"Segmentation palette not found: {palette_path}")
    payload = json.loads(palette_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Segmentation palette JSON must contain an object")

    raw_palette = payload.get("palette")
    if isinstance(raw_palette, str):
        unified_path = palette_path.with_name("_unified_palette.json")
        if unified_path.is_file():
            payload = json.loads(unified_path.read_text(encoding="utf-8"))
            raw_palette = payload.get("palette") if isinstance(payload, dict) else None
        else:
            raw_palette = payload.get("palette_mapping")
        if raw_palette is None:
            raise ValueError(
                f"Palette metadata {palette_path} names a palette but contains no colors"
            )

    if isinstance(raw_palette, list):
        entries = [(f"palette_{index}", color) for index, color in enumerate(raw_palette)]
    elif isinstance(raw_palette, dict):
        entries = [
            (f"class_{name}" if str(name).isdigit() else str(name), color)
            for name, color in raw_palette.items()
        ]
    else:
        entries = list(payload.items())
    if not entries:
        raise ValueError("Segmentation palette must contain at least one color")

    result: dict[str, tuple[int, int, int]] = {}
    for name, raw_color in entries:
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(raw_color, (list, tuple))
            or len(raw_color) != 3
            or any(type(channel) is not int or not 0 <= channel <= 255 for channel in raw_color)
        ):
            raise ValueError(f"Invalid segmentation palette entry: {name!r}={raw_color!r}")
        result[name] = tuple(raw_color)
    return result


class VisualizationDecoder:
    """Decodes mapped RGB outputs into task-specific visualizations.

    Provides high-level API for:
    1. Generating task-specific prompts
    2. Decoding RGB outputs back to vision task outputs
    3. Creating visualization images for display
    """

    def __init__(
        self,
        rgb_mapper: RGBMapper | None = None,
        *,
        semantic_color_map: dict[str, tuple[int, int, int]] | None = None,
        semantic_palette_path: str | Path | None = None,
    ) -> None:
        if semantic_color_map is not None and semantic_palette_path is not None:
            raise ValueError(
                "Provide semantic_color_map or semantic_palette_path, not both"
            )
        self.rgb_mapper = rgb_mapper or RGBMapper()
        self.prompt_builder = PromptBuilder()
        self.semantic_color_map = (
            load_segmentation_color_map(semantic_palette_path)
            if semantic_palette_path is not None
            else semantic_color_map
        )

        from ..core.depth_edge_mapper import DepthEdgeRGBMapper
        from ..core.edge_mapper import EdgeRGBMapper
        from ..core.flow_mapper import FlowRGBMapper
        from ..core.keypoint_mapper import KeypointRGBMapper
        from ..core.object_detection_mapper import ObjectDetectionRGBMapper

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
        class_color_map: dict[str, tuple[int, int, int]] | None = None,
    ) -> dict:
        """Decode semantic segmentation from RGB visualization.

        Args:
            rgb_image: RGB visualization tensor [3, H, W] in [0, 1].
            class_color_map: Class name -> (R, G, B) mapping. If None, uses
                the decoder's configured dataset palette and otherwise falls
                back to the legacy v2 ADE20K palette.
        """
        if class_color_map is None:
            class_color_map = self.semantic_color_map or get_ade20k_151_color_map()
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
        mapper = self._flow_mapper if max_flow == 20.0 else self._flow_mapper_cls(max_flow=max_flow)
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


__all__ = [
    "VisualizationDecoder",
    "get_ade20k_151_color_map",
    "load_segmentation_color_map",
]
