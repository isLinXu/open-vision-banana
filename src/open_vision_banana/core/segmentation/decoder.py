"""Multi-stage clustering algorithm for instance segmentation decoding.

From the paper (Appendix A):
1. Background Initialization: τ=14 color tolerance
2. Color Similarity Grouping: seed-based floodfill with 16-connectivity (Eq. 3)
3. Noise Pruning: θ_size = 2×10⁻⁴ (0.02% of image area)
4. Boundary Artifact Elimination: 3×3 erosion, θ_erosion=0.1
5. Spatially Constrained Merging: γ=5.0 bounding box expansion (Eq. 6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .grouping import GroupingMixin
from .postprocess import PostprocessMixin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstanceDecodingConfig:
    """Complete, serializable contract for the five-stage instance decoder."""

    color_tolerance: float = 14.0
    min_area_fraction: float = 2e-4
    erosion_threshold: float = 0.1
    bbox_expansion_factor: float = 5.0
    # Paper protocol (Appendix A) is the default: seed-based grouping with a
    # fixed tolerance τ=14. The adjacent/adaptive variants remain available
    # as explicit opt-ins for ablation studies.
    grouping_mode: str = "seed"
    adaptive_color_tolerance: bool = False
    # Paper Eq.(3) compares each pixel against the component seed only
    # (``pixel_only``). ``pixel_and_seed`` additionally requires the two
    # component seeds to be within tau, an opt-in stricter variant.
    seed_merge_rule: str = "pixel_only"
    # Paper Eq.(6) uses one area symbol A(.) for the union bounding box and for
    # the two components, so ``bbox`` reads the individual terms as bounding-box
    # areas. ``pixel`` uses pixel counts (|S_a| + |S_b|), which is stricter.
    spatial_merge_area_basis: str = "bbox"

    def __post_init__(self) -> None:
        if self.color_tolerance <= 0:
            raise ValueError("color_tolerance must be positive")
        if self.min_area_fraction < 0:
            raise ValueError("min_area_fraction must be non-negative")
        if not 0.0 <= self.erosion_threshold <= 1.0:
            raise ValueError("erosion_threshold must be in [0, 1]")
        if self.bbox_expansion_factor <= 0:
            raise ValueError("bbox_expansion_factor must be positive")
        if self.grouping_mode not in {"adjacent", "seed"}:
            raise ValueError("grouping_mode must be 'adjacent' or 'seed'")
        if not isinstance(self.adaptive_color_tolerance, bool):
            raise ValueError("adaptive_color_tolerance must be boolean")
        if self.seed_merge_rule not in {"pixel_only", "pixel_and_seed"}:
            raise ValueError("seed_merge_rule must be 'pixel_only' or 'pixel_and_seed'")
        if self.spatial_merge_area_basis not in {"bbox", "pixel"}:
            raise ValueError("spatial_merge_area_basis must be 'bbox' or 'pixel'")

    def to_dict(self) -> dict[str, float | str | bool]:
        """Return a JSON-safe decoding contract for run metadata."""
        return {
            "color_tolerance": self.color_tolerance,
            "min_area_fraction": self.min_area_fraction,
            "erosion_threshold": self.erosion_threshold,
            "bbox_expansion_factor": self.bbox_expansion_factor,
            "grouping_mode": self.grouping_mode,
            "adaptive_color_tolerance": self.adaptive_color_tolerance,
            "seed_merge_rule": self.seed_merge_rule,
            "spatial_merge_area_basis": self.spatial_merge_area_basis,
        }


PAPER_INSTANCE_DECODING_CONFIG = InstanceDecodingConfig(
    grouping_mode="seed",
    adaptive_color_tolerance=False,
    seed_merge_rule="pixel_only",
    spatial_merge_area_basis="bbox",
)


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
        grouping_mode: str = "seed",
        adaptive_color_tolerance: bool = False,
        seed_merge_rule: str = "pixel_only",
        spatial_merge_area_basis: str = "bbox",
    ) -> None:
        self.instance_config = InstanceDecodingConfig(
            color_tolerance=color_tolerance,
            min_area_fraction=min_area_fraction,
            erosion_threshold=erosion_threshold,
            bbox_expansion_factor=bbox_expansion_factor,
            grouping_mode=grouping_mode,
            adaptive_color_tolerance=adaptive_color_tolerance,
            seed_merge_rule=seed_merge_rule,
            spatial_merge_area_basis=spatial_merge_area_basis,
        )
        self.color_tolerance = self.instance_config.color_tolerance
        self.min_area_fraction = self.instance_config.min_area_fraction
        self.erosion_threshold = self.instance_config.erosion_threshold
        self.bbox_expansion_factor = self.instance_config.bbox_expansion_factor
        self.seed_merge_rule = self.instance_config.seed_merge_rule
        self.spatial_merge_area_basis = self.instance_config.spatial_merge_area_basis

    @classmethod
    def paper_reproduction(cls) -> SegmentationDecoder:
        """Build the fixed-tolerance, seed-grouping decoder from Appendix A."""
        return cls(**PAPER_INSTANCE_DECODING_CONFIG.to_dict())

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
        adaptive_color_tolerance: bool | None = None,
        grouping_mode: str | None = None,
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
            adaptive_color_tolerance: Whether to adaptively adjust τ. If None,
                uses this decoder's recorded instance configuration.
            grouping_mode: "adjacent" or "seed". If None, uses this
                decoder's recorded instance configuration. Use
                :meth:`paper_reproduction` for the paper protocol.

        Returns:
            List of lists of instance masks, one per image in batch.
        """
        if adaptive_color_tolerance is None:
            adaptive_color_tolerance = self.instance_config.adaptive_color_tolerance
        if grouping_mode is None:
            grouping_mode = self.instance_config.grouping_mode
        if grouping_mode not in {"adjacent", "seed"}:
            raise ValueError(f"Unknown grouping_mode: {grouping_mode}")

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


__all__ = [
    "InstanceDecodingConfig",
    "PAPER_INSTANCE_DECODING_CONFIG",
    "SegmentationDecoder",
]
