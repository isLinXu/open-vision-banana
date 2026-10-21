"""Segmentation PostprocessMixin stage (P2-1 split)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PostprocessMixin:
    # Class-level type declarations for attributes provided by the
    # composing SegmentationDecoder (mixin pattern).
    erosion_threshold: float
    color_tolerance: float
    bbox_expansion_factor: float
    spatial_merge_area_basis: str

    def _noise_pruning(self, labels: torch.Tensor, min_area: float) -> torch.Tensor:
        """Stage 3: Remove components smaller than θ_size * (H*W).

        Vectorized: uses bincount instead of per-label Python loop.
        """
        unique_labels, counts = labels.unique(return_counts=True)
        # Build a mask of labels to keep (non-background, above min_area)
        keep = (unique_labels != 0) & (counts >= min_area)
        # Also keep background (label 0)
        keep = keep | (unique_labels == 0)
        # Build a lookup: labels not in keep → 0
        max_label = int(unique_labels.max().item())
        keep_lut = torch.zeros(max_label + 1, dtype=torch.bool, device=labels.device)
        keep_lut[unique_labels[keep]] = True
        # Apply: any pixel whose label is not in keep → 0
        valid = (labels <= max_label) & keep_lut[labels.clamp(0, max_label)]
        labels = torch.where(valid, labels, torch.zeros_like(labels))
        return labels

    def _boundary_artifact_elimination(self, labels: torch.Tensor) -> torch.Tensor:
        """Stage 4: Remove thin boundary halos via erosion.

        Vectorized: processes all labels in a single conv2d pass by
        encoding labels as a multi-hot [K, 1, H, W] tensor.
        """
        unique_labels = labels.unique()
        if len(unique_labels) <= 1:
            return labels

        # Filter to non-background labels
        non_bg = unique_labels[unique_labels > 0]
        if len(non_bg) == 0:
            return labels

        # Pre-filter: drop components that cannot survive 3x3 erosion.
        # A 3x3 erosion keeps a pixel only if its full 3x3 neighbourhood
        # shares the label, so a component with fewer than 9 pixels can never
        # contain a fully-filled 3x3 block; its eroded area is exactly 0 and
        # it would be removed anyway. Dropping such tiny components up front
        # is exact and keeps the multi-hot conv cost from scaling with the
        # (often huge) count of singleton components produced by noisy or
        # low-resolution inputs.
        max_label = int(unique_labels.max().item())  # GPU sync point
        counts = torch.bincount(labels.reshape(-1).clamp(min=0), minlength=max_label + 1)
        survivable = non_bg[counts[non_bg] >= 9]
        if len(survivable) < len(non_bg):
            tiny = non_bg[counts[non_bg] < 9]
            if len(tiny) > 0:
                drop_lut = torch.zeros(max_label + 1, dtype=torch.bool, device=labels.device)
                drop_lut[tiny] = True
                to_drop = drop_lut[labels.clamp(0, max_label)]
                labels = torch.where(to_drop, torch.zeros_like(labels), labels)
            non_bg = survivable
        if len(non_bg) == 0:
            return labels

        K = len(non_bg)
        H, W = labels.shape
        device = labels.device

        # Build multi-hot encoding: [K, 1, H, W]
        # label_map[i] = 1 where labels == non_bg[i]
        label_map = (labels.unsqueeze(0) == non_bg.view(-1, 1, 1)).float().unsqueeze(1)

        # Erode all labels at once: conv2d with 3×3 ones kernel
        kernel = torch.ones(1, 1, 3, 3, device=device)
        # Process in groups to limit memory
        eroded_areas = []
        original_areas = []
        chunk = min(K, 64)
        for start in range(0, K, chunk):
            end = min(start + chunk, K)
            chunk_size = end - start
            chunk_map = label_map[start:end]  # [chunk_size, 1, H, W]
            # Reshape to [1, chunk_size, H, W] for grouped conv2d
            chunk_flat = chunk_map.reshape(1, chunk_size, H, W)
            group_kernel = kernel.expand(chunk_size, 1, 3, 3)
            chunk_eroded = F.conv2d(chunk_flat, group_kernel, padding=1, groups=chunk_size)
            chunk_eroded = (chunk_eroded.reshape(chunk_size, 1, H, W) == 9.0).float()
            eroded_areas.append(chunk_eroded.sum(dim=(2, 3)))
            original_areas.append(chunk_map.sum(dim=(2, 3)))

        eroded_areas_t = torch.cat(eroded_areas).squeeze(1)  # [K]
        original_areas_t = torch.cat(original_areas).squeeze(1)  # [K]

        # Determine which labels to remove
        ratio = eroded_areas_t / (original_areas_t + 1e-8)
        remove_mask = ratio < self.erosion_threshold

        # Build removal lookup
        labels_to_remove = non_bg[remove_mask]
        if len(labels_to_remove) == 0:
            return labels

        # Vectorized removal: set all pixels matching any label in remove set to 0
        remove_lut = torch.zeros(int(non_bg.max().item()) + 1, dtype=torch.bool, device=device)
        remove_lut[labels_to_remove] = True
        to_remove = remove_lut[labels.clamp(0, int(non_bg.max().item()))]
        labels = torch.where(to_remove, torch.zeros_like(labels), labels)

        return labels

    def _spatial_merging(self, labels: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Stage 5: Merge spatially disjoint components with similar colors.

        Vectorized: computes pairwise color distances and bbox areas
        using tensor operations instead of Python double loops.
        """
        unique_labels = labels.unique()
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) < 2:
            return labels

        K = len(unique_labels)
        device = labels.device

        # Compute mean colors and bounding boxes for all components at once
        # Use scatter-based aggregation for efficiency
        max_label = int(unique_labels.max().item())

        # Mean color per component: scatter_add + count
        flat_labels = labels.reshape(-1)
        flat_img = img.reshape(-1, 3)

        # Filter to non-background pixels
        valid = flat_labels > 0
        valid_labels = flat_labels[valid]
        valid_img = flat_img[valid]

        color_sum = torch.zeros(max_label + 1, 3, device=device)
        color_sum.scatter_add_(0, valid_labels.unsqueeze(1).expand(-1, 3), valid_img)
        count = torch.zeros(max_label + 1, 1, device=device)
        count.scatter_add_(
            0, valid_labels.unsqueeze(1), torch.ones(valid_labels.shape[0], 1, device=device)
        )
        count = count.squeeze(1)

        # Avoid division by zero
        safe_count = count.clamp(min=1)
        mean_colors = color_sum / safe_count.unsqueeze(1)  # [max_label+1, 3]

        # Bounding boxes: scatter min/max for y and x
        ys, xs = torch.where(labels > 0)
        label_at_pos = labels[ys, xs]

        y_min = torch.full((max_label + 1,), labels.shape[0], dtype=torch.long, device=device)
        y_max = torch.zeros(max_label + 1, dtype=torch.long, device=device)
        x_min = torch.full((max_label + 1,), labels.shape[1], dtype=torch.long, device=device)
        x_max = torch.zeros(max_label + 1, dtype=torch.long, device=device)

        y_min.scatter_reduce_(0, label_at_pos, ys, reduce="amin")
        y_max.scatter_reduce_(0, label_at_pos, ys, reduce="amax")
        x_min.scatter_reduce_(0, label_at_pos, xs, reduce="amin")
        x_max.scatter_reduce_(0, label_at_pos, xs, reduce="amax")

        # Eq. (6) writes A(S_a ∪ S_b) ≤ γ (A(S_a) + A(S_b)) with one symbol A on both
        # sides, so the individual terms are read as bounding-box areas by default;
        # ``spatial_merge_area_basis="pixel"`` keeps the stricter pixel-count variant.
        if self.spatial_merge_area_basis == "bbox":
            area = ((y_max - y_min + 1) * (x_max - x_min + 1)).float()
        else:
            area = count.float()

        # Extract data for unique_labels only
        ul = unique_labels.long()
        mc = mean_colors[ul]  # [K, 3]
        pa = area[ul]  # [K]
        ymn = y_min[ul]  # [K]
        ymx = y_max[ul]  # [K]
        xmn = x_min[ul]  # [K]
        xmx = x_max[ul]  # [K]

        # Pairwise color distance matrix: [K, K]
        color_dist = torch.cdist(mc, mc)

        # Pairwise combined bbox area: [K, K]
        combined_ymin = torch.minimum(ymn.unsqueeze(1), ymn.unsqueeze(0))
        combined_ymax = torch.maximum(ymx.unsqueeze(1), ymx.unsqueeze(0))
        combined_xmin = torch.minimum(xmn.unsqueeze(1), xmn.unsqueeze(0))
        combined_xmax = torch.maximum(xmx.unsqueeze(1), xmx.unsqueeze(0))
        combined_bbox = (
            (combined_ymax - combined_ymin + 1) * (combined_xmax - combined_xmin + 1)
        ).float()

        # Merge condition: color_dist ≤ τ AND combined_bbox ≤ γ * (area_i + area_j)
        sum_area = pa.unsqueeze(1) + pa.unsqueeze(0)
        should_merge = (color_dist <= self.color_tolerance) & (
            combined_bbox <= self.bbox_expansion_factor * sum_area
        )

        # Only consider upper triangle (i < j) to avoid double-merging
        should_merge = should_merge & torch.triu(
            torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1
        )

        # Vectorized connected-component merging via tensor Union-Find.
        # The previous implementation iterated merge pairs in a Python loop
        # with per-pair .item() calls -- an O(P) scalar bottleneck (P up to
        # K^2) that dominated S5 latency at high resolution. We instead build
        # a parent LUT over component *indices* [0, K) and propagate the
        # minimum reachable index along every merge edge with scatter_reduce,
        # then path-compress. This yields the same final partition (connected
        # components of the should_merge graph) with zero Python-level loops
        # over pairs.
        mi, mj = torch.where(should_merge)
        if mi.numel() > 0:
            # parent[k] starts as k; repeatedly relax parent to the min of
            # itself and the parents of its merge neighbours (both directions).
            parent = torch.arange(K, dtype=torch.long, device=device)
            # A bounded number of relaxation rounds suffices: each round plus
            # path compression at least halves any merge-chain height.
            max_rounds = max(1, K.bit_length() + 1)
            for _ in range(max_rounds):
                src = torch.cat([mi, mj])
                dst = torch.cat([mj, mi])
                cand = torch.minimum(parent[src], parent[dst])
                new_parent = parent.clone()
                new_parent.scatter_reduce_(0, src, cand, reduce="amin")
                # Path compression
                new_parent = new_parent[new_parent]
                if torch.equal(new_parent, parent):
                    break
                parent = new_parent

            # Map component-index roots back to original label values.
            root_labels = ul[parent]  # [K], canonical label per component
            remap = torch.arange(max_label + 1, dtype=torch.long, device=device)
            remap[ul] = root_labels
            # Chase remap to final roots (handles transitive label hops).
            for _ in range(4):
                remap = remap[remap]
            labels = remap[labels.clamp(0, max_label)]

        return labels

    def _extract_instances(self, labels: torch.Tensor, img: torch.Tensor) -> list[dict]:
        """Extract individual instance masks from labeled image (vectorized)."""
        unique_labels = labels.unique()
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            return []

        K = len(unique_labels)
        device = labels.device
        max_label = unique_labels.max().item()

        # Build a direct LUT: original label → index into [0, K)
        index_lut = torch.zeros(max_label + 1, dtype=torch.long, device=device)
        index_lut[unique_labels] = torch.arange(K, device=device)

        # Index map: each pixel gets component index or 0 for background
        flat_labels = labels.reshape(-1)
        flat_img = img.reshape(-1, 3)
        valid = flat_labels > 0
        comp_idx = index_lut[flat_labels.clamp(0, max_label)]

        # Scatter-based mean color per component
        color_sum = torch.zeros(K, 3, device=device)
        color_sum.scatter_add_(0, comp_idx[valid].unsqueeze(1).expand(-1, 3), flat_img[valid])
        counts = torch.bincount(comp_idx[valid], minlength=K).clamp(min=1)
        mean_colors = (color_sum / counts.unsqueeze(1)).long()

        instances = []
        for i in range(K):
            mask = labels == unique_labels[i]
            instances.append(
                {
                    "mask": mask,
                    "color": tuple(mean_colors[i].tolist()),
                }
            )

        return instances


__all__ = ["PostprocessMixin"]
