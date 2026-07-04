"""Multi-stage clustering algorithm for instance segmentation decoding.

From the paper (Appendix A):
1. Background Initialization: τ=14 color tolerance
2. Color Similarity Grouping: Union-Find with 8-connectivity
3. Noise Pruning: θ_size = 2×10⁻⁴ (0.02% of image area)
4. Boundary Artifact Elimination: 3×3 erosion, θ_erosion=0.1
5. Spatially Constrained Merging: γ=5.0 bounding box expansion
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


from .grouping import GroupingMixin
from .postprocess import PostprocessMixin


class SegmentationDecoder(GroupingMixin, PostprocessMixin):
    """Multi-stage clustering algorithm for parsing instance masks.

    Implements the full 5-stage algorithm from Appendix A with
    vectorized Union-Find for O(E α(N)) connected-component labeling.
    """

    def __init__(
        self,
        color_tolerance: float = 14.0,
        min_area_fraction: float = 2e-4,
        erosion_threshold: float = 0.1,
        bbox_expansion_factor: float = 5.0,
    ) -> None:
        self.color_tolerance = color_tolerance
        self.min_area_fraction = min_area_fraction
        self.erosion_threshold = erosion_threshold
        self.bbox_expansion_factor = bbox_expansion_factor

    def decode_semantic(
        self,
        rgb_image: torch.Tensor,
        class_color_map: dict[str, tuple[int, int, int]],
    ) -> torch.Tensor:
        """Decode semantic segmentation from RGB visualization.

        Assigns each pixel to the class whose target color is closest
        in RGB space using vectorized pairwise distance computation.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            class_color_map: Mapping from class name to (R, G, B) tuple
                with values in [0, 255].

        Returns:
            Segmentation mask [B, 1, H, W] with class indices.
        """
        B, C, H, W = rgb_image.shape
        device = rgb_image.device

        class_names = list(class_color_map.keys())
        class_colors = (
            torch.tensor(
                [class_color_map[name] for name in class_names],
                dtype=torch.float32,
                device=device,
            )
            / 255.0
        )

        pixels = rgb_image.permute(0, 2, 3, 1).reshape(B, -1, 3)
        colors = class_colors.unsqueeze(0).expand(B, -1, -1)

        # Nearest-color assignment via vectorized cdist.
        # When adaptive_color_tolerance is True, we still use nearest-color
        # assignment (argmin) for correctness; the adaptive tau only affects
        # instance segmentation, not semantic (where every pixel must be
        # assigned to exactly one class regardless of distance).
        distances = torch.cdist(pixels, colors)
        labels = distances.argmin(dim=-1)
        mask = labels.reshape(B, 1, H, W)

        return mask

    def decode_instance(
        self,
        rgb_image: torch.Tensor,
        background_color: tuple[int, int, int] = (0, 0, 0),
        adaptive_color_tolerance: bool = True,
        grouping_mode: str = "adjacent",  # 新增
    ) -> list[list[dict]]:
        """Decode instance segmentation using multi-stage clustering.

        Implements the full 5-stage algorithm from Appendix A.
        Processes all images in the batch with shared tensor operations
        for Stage 1 (background detection), then per-image for clustering.

        When adaptive_color_tolerance is True, the color tolerance τ
        is dynamically adjusted based on local image statistics:
        - High-contrast regions: smaller τ for sharper boundaries
        - Low-contrast regions: larger τ for smoother transitions
        - Near depth discontinuities: smaller τ for precision
        - Far from edges: larger τ for efficiency

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            background_color: Background color (R, G, B) in [0, 255].
            adaptive_color_tolerance: Whether to adaptively adjust τ.
            grouping_mode: "adjacent" (default) or "seed".  Seed-based
                uses the component's seed color for comparison, which
                produces more compact instances per paper Appendix A.

        Returns:
            List of lists of instance masks, one per image in batch.
        """
        B, C, H, W = rgb_image.shape
        total_pixels = H * W
        min_area = self.min_area_fraction * total_pixels

        img_255 = (rgb_image * 255.0).clamp(0, 255)
        bg_color = torch.tensor(background_color, dtype=torch.float32)

        # ── P2: Compute adaptive color tolerance map ────────────────
        tau_map = self._compute_adaptive_tau(rgb_image) if adaptive_color_tolerance else None

        all_instances = []
        for b in range(B):
            img = img_255[b].permute(1, 2, 0)  # [H, W, 3]

            # Stage 1: Background Initialization
            if tau_map is not None:
                bg_mask = (img - bg_color.to(img.device)).norm(dim=-1) < tau_map[b]
            else:
                bg_mask = (img - bg_color.to(img.device)).norm(dim=-1) < self.color_tolerance

            # Stage 2: Color Similarity Grouping (vectorized Union-Find)
            if grouping_mode == "seed":
                labels = self._seed_based_union_find(img, bg_mask)
            elif grouping_mode == "adjacent":
                if tau_map is not None:
                    labels = self._union_find_grouping_adaptive(img, bg_mask, tau_map[b])
                else:
                    labels = self._union_find_grouping(img, bg_mask)
            else:
                raise ValueError(f"Unknown grouping_mode: {grouping_mode}")

            # Stage 3: Noise Pruning (vectorized)
            labels = self._noise_pruning(labels, min_area)

            # Stage 4: Boundary Artifact Elimination (vectorized)
            labels = self._boundary_artifact_elimination(labels)

            # Stage 5: Spatially Constrained Merging (vectorized)
            labels = self._spatial_merging(labels, img)

            # Extract instance masks
            instances = self._extract_instances(labels, img)
            all_instances.append(instances)

        return all_instances



__all__ = ["SegmentationDecoder"]
