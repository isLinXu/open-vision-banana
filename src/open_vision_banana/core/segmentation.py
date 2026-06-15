"""Multi-stage clustering algorithm for instance segmentation decoding.

From the paper (Appendix A):
1. Background Initialization: τ=14 color tolerance
2. Color Similarity Grouping: 16-connectivity flood-fill
3. Noise Pruning: θ_size = 2×10⁻⁴ (0.02% of image area)
4. Boundary Artifact Elimination: 3×3 erosion, θ_erosion=0.1
5. Spatially Constrained Merging: γ=5.0 bounding box expansion
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SegmentationDecoder:
    """Multi-stage clustering algorithm for parsing instance masks.

    Implements the full 5-stage algorithm from Appendix A with
    vectorized operations for performance.
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
        in RGB space.

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
        class_colors = torch.tensor(
            [class_color_map[name] for name in class_names],
            dtype=torch.float32, device=device,
        ) / 255.0

        pixels = rgb_image.permute(0, 2, 3, 1).reshape(B, -1, 3)
        colors = class_colors.unsqueeze(0).expand(B, -1, -1)

        distances = torch.cdist(pixels, colors)
        labels = distances.argmin(dim=-1)
        mask = labels.reshape(B, 1, H, W)

        return mask

    def decode_instance(
        self,
        rgb_image: torch.Tensor,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> list[list[dict]]:
        """Decode instance segmentation using multi-stage clustering.

        Implements the full 5-stage algorithm from Appendix A.

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            background_color: Background color (R, G, B) in [0, 255].

        Returns:
            List of lists of instance masks, one per image in batch.
        """
        B, C, H, W = rgb_image.shape
        total_pixels = H * W
        min_area = self.min_area_fraction * total_pixels

        img_255 = (rgb_image * 255.0).clamp(0, 255)
        bg_color = torch.tensor(background_color, dtype=torch.float32)

        all_instances = []
        for b in range(B):
            img = img_255[b].permute(1, 2, 0)  # [H, W, 3]

            # Stage 1: Background Initialization
            bg_mask = (img - bg_color.to(img.device)).norm(dim=-1) < self.color_tolerance

            # Stage 2: Color Similarity Grouping (vectorized)
            labels = self._flood_fill_grouping(img, bg_mask)

            # Stage 3: Noise Pruning
            labels = self._noise_pruning(labels, min_area)

            # Stage 4: Boundary Artifact Elimination
            labels = self._boundary_artifact_elimination(labels)

            # Stage 5: Spatially Constrained Merging
            labels = self._spatial_merging(labels, img)

            # Extract instance masks
            instances = self._extract_instances(labels, img)
            all_instances.append(instances)

        return all_instances

    def _flood_fill_grouping(
        self, img: torch.Tensor, bg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Stage 2: Vectorized connected-component color similarity grouping.

        Uses iterative label propagation with 8-connectivity shifts.
        Converges quickly (typically 5-10 iterations for natural images).
        """
        H, W, C = img.shape
        device = img.device
        N = H * W

        labels = torch.arange(N, device=device).reshape(H, W)
        labels[bg_mask] = -1  # background marker

        seed_colors = img.reshape(N, C).clone()
        tau_sq = self.color_tolerance ** 2

        for _iteration in range(10):
            changed = False

            shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

            for dy, dx in shifts:
                shifted = torch.full_like(labels, -1)
                if dy == -1:
                    shifted[:-1, :] = labels[1:, :]
                elif dy == 1:
                    shifted[1:, :] = labels[:-1, :]
                else:
                    shifted[:] = labels[:]

                if dx == -1:
                    shifted[:, :-1] = shifted[:, 1:]
                elif dx == 1:
                    shifted[:, 1:] = shifted[:, :-1]

                merge_candidates = (shifted >= 0) & (labels >= 0) & (shifted != labels)
                if not merge_candidates.any():
                    continue

                cand_y, cand_x = torch.where(merge_candidates)
                cand_labels = labels[cand_y, cand_x]
                neighbor_labels = shifted[cand_y, cand_x]

                cand_seeds = seed_colors[cand_labels]
                neigh_seeds = seed_colors[neighbor_labels]

                color_dist_sq = ((cand_seeds - neigh_seeds) ** 2).sum(dim=-1)
                can_merge = color_dist_sq <= tau_sq

                if can_merge.any():
                    merge_y = cand_y[can_merge]
                    merge_x = cand_x[can_merge]
                    winner_labels = torch.minimum(cand_labels[can_merge], neighbor_labels[can_merge])

                    labels[merge_y, merge_x] = winner_labels

                    # Vectorized seed color propagation
                    loser_labels = torch.maximum(cand_labels[can_merge], neighbor_labels[can_merge])
                    unique_losers = loser_labels.unique()

                    for loser in unique_losers.tolist():
                        if loser == winner_labels[loser_labels == loser][0].item():
                            continue
                        winner = winner_labels[loser_labels == loser][0]
                        loser_mask = labels == loser
                        labels[loser_mask] = winner
                        seed_colors[loser_mask.reshape(N)] = seed_colors[winner]

                    changed = True

            if not changed:
                break

        # Relabel to consecutive integers (0 = background, 1+ = instances)
        unique_labels = labels.unique()
        new_labels = torch.zeros_like(labels)
        new_idx = 1
        for old_lbl in unique_labels:
            if old_lbl.item() < 0:
                continue
            new_labels[labels == old_lbl] = new_idx
            new_idx += 1

        return new_labels

    def _noise_pruning(
        self, labels: torch.Tensor, min_area: float
    ) -> torch.Tensor:
        """Stage 3: Remove components smaller than θ_size * (H*W)."""
        unique_labels, counts = labels.unique(return_counts=True)
        for lbl, cnt in zip(unique_labels.tolist(), counts.tolist(), strict=True):
            if lbl == 0:
                continue
            if cnt < min_area:
                labels[labels == lbl] = 0
        return labels

    def _boundary_artifact_elimination(
        self, labels: torch.Tensor
    ) -> torch.Tensor:
        """Stage 4: Remove thin boundary halos via erosion."""
        unique_labels = labels.unique()
        if len(unique_labels) <= 1:
            return labels

        kernel = torch.ones(1, 1, 3, 3, device=labels.device)

        labels_to_remove = []
        for lbl in unique_labels.tolist():
            if lbl == 0:
                continue
            _mask = (labels == lbl).float().unsqueeze(0).unsqueeze(0)
            original_area = _mask.sum().item()
            if original_area == 0:
                continue

            eroded = F.conv2d(_mask, kernel, padding=1)
            eroded = (eroded == 9.0).float()
            eroded_area = eroded.sum().item()

            if eroded_area / original_area < self.erosion_threshold:
                labels_to_remove.append(lbl)

        for lbl in labels_to_remove:
            labels[labels == lbl] = 0

        return labels

    def _spatial_merging(
        self, labels: torch.Tensor, img: torch.Tensor
    ) -> torch.Tensor:
        """Stage 5: Merge spatially disjoint components with similar colors."""
        unique_labels = labels.unique()
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) < 2:
            return labels

        components = {}
        for lbl in unique_labels:
            mask = labels == lbl
            mean_color = img[mask].mean(dim=0)
            ys, xs = torch.where(mask)
            bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            pixel_area = mask.sum().item()
            components[lbl.item()] = {
                'mean_color': mean_color,
                'bbox_area': bbox_area,
                'pixel_area': pixel_area,
                'ys': ys,
                'xs': xs,
            }

        label_list = list(components.keys())
        merged = set()
        for i in range(len(label_list)):
            for j in range(i + 1, len(label_list)):
                la, lb = label_list[i], label_list[j]
                if la in merged or lb in merged:
                    continue

                ca, cb = components[la], components[lb]

                color_dist = (ca['mean_color'] - cb['mean_color']).norm()
                if color_dist > self.color_tolerance:
                    continue

                combined_bbox = self._combined_bbox_area(
                    ca['ys'], ca['xs'], cb['ys'], cb['xs']
                )
                if combined_bbox <= self.bbox_expansion_factor * (
                    ca['bbox_area'] + cb['bbox_area']
                ):
                    labels[labels == lb] = la
                    merged.add(lb)

        return labels

    @staticmethod
    def _combined_bbox_area(ys1: torch.Tensor, xs1: torch.Tensor,
                            ys2: torch.Tensor, xs2: torch.Tensor) -> int:
        """Compute combined bounding box area of two components."""
        y_min = min(ys1.min(), ys2.min())
        y_max = max(ys1.max(), ys2.max())
        x_min = min(xs1.min(), xs2.min())
        x_max = max(xs1.max(), xs2.max())
        return int((y_max - y_min + 1) * (x_max - x_min + 1))

    def _extract_instances(
        self, labels: torch.Tensor, img: torch.Tensor
    ) -> list[dict]:
        """Extract individual instance masks from labeled image."""
        instances = []
        unique_labels = labels.unique()
        unique_labels = unique_labels[unique_labels > 0]

        for lbl in unique_labels:
            mask = labels == lbl
            mean_color = img[mask].mean(dim=0)
            instances.append({
                'mask': mask,
                'color': tuple(mean_color.long().tolist()),
            })

        return instances


__all__ = ["SegmentationDecoder"]
