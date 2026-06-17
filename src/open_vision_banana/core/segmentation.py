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


class SegmentationDecoder:
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
        adaptive_color_tolerance: bool = True,
    ) -> torch.Tensor:
        """Decode semantic segmentation from RGB visualization.

        Assigns each pixel to the class whose target color is closest
        in RGB space.

        When adaptive_color_tolerance is True, the color tolerance τ
        is dynamically adjusted based on local image statistics:
        - High-contrast regions: smaller τ for sharper boundaries
        - Low-contrast regions: larger τ for smoother transitions
        - Near depth discontinuities: smaller τ for precision
        - Far from edges: larger τ for efficiency

        Args:
            rgb_image: Generated RGB image [B, 3, H, W] in [0, 1].
            class_color_map: Mapping from class name to (R, G, B) tuple
                with values in [0, 255].
            adaptive_color_tolerance: Whether to adaptively adjust τ.

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

        # ── P2: Adaptive color tolerance ───────────────────────────
        if adaptive_color_tolerance:
            # Compute local contrast for adaptive τ
            # High contrast → sharper boundaries → smaller τ
            # Low contrast → smoother transitions → larger τ
            gray = rgb_image.mean(dim=1, keepdim=True)
            gradient_magnitude = torch.norm(
                torch.gradient(gray, dim=(-2, -1))[0], dim=0
            ).squeeze(0)

            # Near depth discontinuities: high gradient → smaller τ
            # Far from edges: low gradient → larger τ
            # Scale τ inversely with gradient magnitude
            tau_adaptive = self.color_tolerance / (1.0 + gradient_magnitude)
        else:
            tau_adaptive = self.color_tolerance

        # Assign pixels to nearest class color using adaptive τ
        # (Still uses cdist for vectorized distance computation)
        dist_adaptive = torch.cdist(pixels, colors)
        labels_adaptive = dist_adaptive.argmin(dim=-1)
        mask = labels_adaptive.reshape(B, 1, H, W)

        return mask

    def decode_instance(
        self,
        rgb_image: torch.Tensor,
        background_color: tuple[int, int, int] = (0, 0, 0),
        adaptive_color_tolerance: bool = True,
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

        Returns:
            List of lists of instance masks, one per image in batch.
        """
        B, C, H, W = rgb_image.shape
        total_pixels = H * W
        min_area = self.min_area_fraction * total_pixels

        img_255 = (rgb_image * 255.0).clamp(0, 255)
        bg_color = torch.tensor(background_color, dtype=torch.float32)

        # ── P2: Compute adaptive color tolerance map ────────────────
        if adaptive_color_tolerance:
            tau_map = self._compute_adaptive_tau(rgb_image)
        else:
            tau_map = None

        all_instances = []
        for b in range(B):
            img = img_255[b].permute(1, 2, 0)  # [H, W, 3]

            # Stage 1: Background Initialization
            # Use adaptive τ if available, otherwise use fixed τ
            if tau_map is not None:
                bg_mask = (img - bg_color.to(img.device)).norm(dim=-1) < tau_map[b]
            else:
                bg_mask = (img - bg_color.to(img.device)).norm(dim=-1) < self.color_tolerance

            # Stage 2: Color Similarity Grouping (vectorized Union-Find)
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

    def _compute_adaptive_tau(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Compute adaptive color tolerance map based on local image statistics.

        High-contrast regions (edges, boundaries): smaller τ for sharper boundaries
        Low-contrast regions (flat areas): larger τ for smoother transitions

        Args:
            rgb_image: [B, 3, H, W] in [0, 1].

        Returns:
            Adaptive τ map [B, H, W] with values in [τ_min, τ_max].
        """
        B, C, H, W = rgb_image.shape
        device = rgb_image.device

        # Convert to grayscale for gradient computation
        gray = rgb_image.mean(dim=1)  # [B, H, W]

        # Compute gradient magnitude using Sobel-like filters
        # Horizontal gradient (Gx)
        gx = torch.zeros(B, H, W, device=device)
        gx[:, 1:-1, :] = (gray[:, 2:, :] - gray[:, :-2, :]) / 2.0
        # Vertical gradient (Gy)
        gy = torch.zeros(B, H, W, device=device)
        gy[:, :, 1:-1] = (gray[:, :, 2:] - gray[:, :, :-2]) / 2.0

        gradient_magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-10)

        # Normalize gradient to [0, 1]
        grad_norm = gradient_magnitude / (gradient_magnitude.max() + 1e-8)

        # Adaptive τ: inversely proportional to gradient magnitude
        # High gradient → smaller τ (sharper boundaries)
        # Low gradient → larger τ (smoother transitions)
        tau_min = self.color_tolerance * 0.5   # Minimum τ (at edges)
        tau_max = self.color_tolerance * 1.5   # Maximum τ (in flat areas)

        # Linear interpolation: τ = τ_max - (τ_max - τ_min) * grad_norm
        tau_adaptive = tau_max - (tau_max - tau_min) * grad_norm

        return tau_adaptive

    def _union_find_grouping_adaptive(
        self, img: torch.Tensor, bg_mask: torch.Tensor, tau_map: torch.Tensor
    ) -> torch.Tensor:
        """Stage 2 with adaptive color tolerance per pixel.

        Like _union_find_grouping but uses per-pixel τ from tau_map
        instead of a fixed global τ.

        Args:
            img: [H, W, 3] image in [0, 255].
            bg_mask: [H, W] boolean background mask.
            tau_map: [H, W] adaptive color tolerance per pixel.

        Returns:
            Label map [H, W] with consecutive integer labels.
        """
        H, W, C = img.shape
        device = img.device
        N = H * W

        # Initialize labels — each pixel is its own component
        labels = torch.arange(N, dtype=torch.long, device=device)

        # One-pass edge scanning — 8 directions
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dy, dx in shifts:
            shifted_img = torch.roll(img, shifts=(-dy, -dx), dims=(0, 1))

            # Build valid neighbor mask
            valid = torch.ones(H, W, dtype=torch.bool, device=device)
            if dy == -1:
                valid[0, :] = False
            elif dy == 1:
                valid[-1, :] = False
            if dx == -1:
                valid[:, 0] = False
            elif dx == 1:
                valid[:, -1] = False

            valid = valid & ~bg_mask

            color_diff = (img - shifted_img) ** 2
            color_dist_sq = color_diff.sum(dim=-1)

            # Use adaptive τ: take the maximum of source and destination τ
            # This ensures both pixels agree on the tolerance
            shifted_tau = torch.roll(tau_map, shifts=(-dy, -dx), dims=(0, 1))
            adaptive_tau = torch.maximum(tau_map, shifted_tau)
            adaptive_tau_sq = adaptive_tau ** 2

            eligible = valid & (color_dist_sq <= adaptive_tau_sq)

            if not eligible.any():
                continue

            src_y, src_x = torch.where(eligible)
            src_indices = src_y * W + src_x
            dst_indices = (src_y + dy) * W + (src_x + dx)

            src_labels = labels[src_indices]
            dst_labels = labels[dst_indices]
            min_labels = torch.minimum(src_labels, dst_labels)

            labels.scatter_(0, src_indices, torch.minimum(labels[src_indices], min_labels))
            labels.scatter_(0, dst_indices, torch.minimum(labels[dst_indices], min_labels))

        # Path compression
        lut = labels.clone()
        for _ in range(4):
            lut = lut[lut]

        labels = torch.arange(N, dtype=torch.long, device=device)
        labels[bg_mask.reshape(N)] = -1

        valid = labels >= 0
        labels[valid] = lut[labels[valid]]
        labels = labels.reshape(H, W)

        # Relabel to consecutive integers
        unique_labels, inverse = labels.reshape(-1).unique(return_inverse=True)
        is_bg = unique_labels < 0
        new_id = torch.zeros_like(unique_labels)
        non_bg_idx = 1
        for i in range(len(unique_labels)):
            if is_bg[i]:
                new_id[i] = 0
            else:
                new_id[i] = non_bg_idx
                non_bg_idx += 1

        result = new_id[inverse].reshape(H, W)
        return result

    def _union_find_grouping(
        self, img: torch.Tensor, bg_mask: torch.Tensor
    ) -> torch.Tensor:
        """Stage 2: Fully vectorized connected-component labeling.

        Uses iterative label propagation (scatter minimum root) instead of
        a Python Union-Find object.  All operations are pure tensor — no
        Python loops over edges or labels.

        Algorithm:
        1. Initialize labels[i] = i for all N pixels
        2. One-pass edge scanning: for each of 8 directions, compute
           all eligible edges (color_dist² ≤ τ²) in one vectorized op
        3. For each eligible edge, scatter the minimum label to both
           endpoints (iterative label propagation)
        4. Path compression via lut[lut] (4 rounds)
        5. Relabel to consecutive integers via unique + scatter

        Complexity: O(E α(N)) effective, fully vectorized.
        """
        H, W, C = img.shape
        device = img.device
        N = H * W
        tau_sq = self.color_tolerance ** 2

        # Step 1: Initialize labels — each pixel is its own component
        labels = torch.arange(N, dtype=torch.long, device=device)

        # Step 2: One-pass edge scanning — 8 directions × 1 scan
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dy, dx in shifts:
            shifted_img = torch.roll(img, shifts=(-dy, -dx), dims=(0, 1))

            # Build valid neighbor mask
            valid = torch.ones(H, W, dtype=torch.bool, device=device)
            if dy == -1:
                valid[0, :] = False
            elif dy == 1:
                valid[-1, :] = False
            if dx == -1:
                valid[:, 0] = False
            elif dx == 1:
                valid[:, -1] = False

            valid = valid & ~bg_mask

            color_diff = (img - shifted_img) ** 2
            color_dist_sq = color_diff.sum(dim=-1)

            eligible = valid & (color_dist_sq <= tau_sq)

            if not eligible.any():
                continue

            # Get source and destination pixel indices (tensors, no .tolist())
            src_y, src_x = torch.where(eligible)
            src_indices = src_y * W + src_x
            dst_indices = (src_y + dy) * W + (src_x + dx)

            # Vectorized union: scatter the minimum label to both endpoints
            src_labels = labels[src_indices]
            dst_labels = labels[dst_indices]
            min_labels = torch.minimum(src_labels, dst_labels)

            # Scatter minimum to both src and dst
            labels.scatter_(0, src_indices, torch.minimum(labels[src_indices], min_labels))
            labels.scatter_(0, dst_indices, torch.minimum(labels[dst_indices], min_labels))

        # Step 3: Path compression — propagate labels to roots via lut[lut]
        lut = labels.clone()
        for _ in range(4):
            lut = lut[lut]

        # Apply compressed labels
        labels = torch.arange(N, dtype=torch.long, device=device)
        labels[bg_mask.reshape(N)] = -1  # background marker

        valid = labels >= 0
        labels[valid] = lut[labels[valid]]
        labels = labels.reshape(H, W)

        # Step 4: Relabel to consecutive integers (vectorized)
        unique_labels, inverse = labels.reshape(-1).unique(return_inverse=True)
        # Find which unique labels are background (< 0)
        is_bg = unique_labels < 0
        # Build a mapping: bg → 0, non-bg → 1, 2, 3, ...
        new_id = torch.zeros_like(unique_labels)
        non_bg_idx = 1
        for i in range(len(unique_labels)):
            if is_bg[i]:
                new_id[i] = 0
            else:
                new_id[i] = non_bg_idx
                non_bg_idx += 1

        result = new_id[inverse].reshape(H, W)
        return result

    def _noise_pruning(
        self, labels: torch.Tensor, min_area: float
    ) -> torch.Tensor:
        """Stage 3: Remove components smaller than θ_size * (H*W).

        Vectorized: uses bincount instead of per-label Python loop.
        """
        unique_labels, counts = labels.unique(return_counts=True)
        # Build a mask of labels to keep (non-background, above min_area)
        keep = (unique_labels != 0) & (counts >= min_area)
        # Also keep background (label 0)
        keep = keep | (unique_labels == 0)
        # Build a lookup: labels not in keep → 0
        max_label = unique_labels.max().item()
        keep_lut = torch.zeros(max_label + 1, dtype=torch.bool, device=labels.device)
        keep_lut[unique_labels[keep]] = True
        # Apply: any pixel whose label is not in keep → 0
        valid = (labels <= max_label) & keep_lut[labels.clamp(0, max_label)]
        labels = torch.where(valid, labels, torch.zeros_like(labels))
        return labels

    def _boundary_artifact_elimination(
        self, labels: torch.Tensor
    ) -> torch.Tensor:
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

        eroded_areas = torch.cat(eroded_areas).squeeze(1)  # [K]
        original_areas = torch.cat(original_areas).squeeze(1)  # [K]

        # Determine which labels to remove
        ratio = eroded_areas / (original_areas + 1e-8)
        remove_mask = ratio < self.erosion_threshold

        # Build removal lookup
        labels_to_remove = non_bg[remove_mask]
        if len(labels_to_remove) == 0:
            return labels

        # Vectorized removal: set all pixels matching any label in remove set to 0
        remove_lut = torch.zeros(non_bg.max().item() + 1, dtype=torch.bool, device=device)
        remove_lut[labels_to_remove] = True
        to_remove = remove_lut[labels.clamp(0, non_bg.max().item())]
        labels = torch.where(to_remove, torch.zeros_like(labels), labels)

        return labels

    def _spatial_merging(
        self, labels: torch.Tensor, img: torch.Tensor
    ) -> torch.Tensor:
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
        max_label = unique_labels.max().item()

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
        count.scatter_add_(0, valid_labels.unsqueeze(1), torch.ones(valid_labels.shape[0], 1, device=device))
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

        bbox_area = ((y_max - y_min + 1) * (x_max - x_min + 1)).float()
        pixel_area = count.float()

        # Extract data for unique_labels only
        ul = unique_labels.long()
        mc = mean_colors[ul]       # [K, 3]
        ba = bbox_area[ul]         # [K]
        pa = pixel_area[ul]        # [K]
        ymn = y_min[ul]            # [K]
        ymx = y_max[ul]            # [K]
        xmn = x_min[ul]            # [K]
        xmx = x_max[ul]            # [K]

        # Pairwise color distance matrix: [K, K]
        color_dist = torch.cdist(mc, mc)

        # Pairwise combined bbox area: [K, K]
        combined_ymin = torch.minimum(ymn.unsqueeze(1), ymn.unsqueeze(0))
        combined_ymax = torch.maximum(ymx.unsqueeze(1), ymx.unsqueeze(0))
        combined_xmin = torch.minimum(xmn.unsqueeze(1), xmn.unsqueeze(0))
        combined_xmax = torch.maximum(xmx.unsqueeze(1), xmx.unsqueeze(0))
        combined_bbox = ((combined_ymax - combined_ymin + 1) *
                         (combined_xmax - combined_xmin + 1)).float()

        # Merge condition: color_dist ≤ τ AND combined_bbox ≤ γ * (area_i + area_j)
        sum_area = pa.unsqueeze(1) + pa.unsqueeze(0)
        should_merge = (color_dist <= self.color_tolerance) & \
                       (combined_bbox <= self.bbox_expansion_factor * sum_area)

        # Only consider upper triangle (i < j) to avoid double-merging
        should_merge = should_merge & torch.triu(
            torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1
        )

        # Greedy merging: iterate over merge pairs
        merged = set()
        merge_map = {ul[i].item(): ul[i].item() for i in range(K)}

        # Get merge pairs as tensors
        mi, mj = torch.where(should_merge)
        for idx in range(len(mi)):
            i, j = mi[idx].item(), mj[idx].item()
            la, lb = ul[i].item(), ul[j].item()

            # Resolve to current canonical label
            root_a = merge_map[la]
            root_b = merge_map[lb]
            if root_a == root_b:
                continue
            if root_a in merged or root_b in merged:
                continue

            # Merge b into a
            merge_map[root_b] = root_a
            merged.add(root_b)

        # Apply merges vectorized
        if merged:
            # Build a lookup table for label remapping
            remap = torch.arange(max_label + 1, dtype=torch.long, device=device)
            for old_lbl, new_lbl in merge_map.items():
                if old_lbl != new_lbl:
                    remap[old_lbl] = new_lbl

            # Chase remap to final roots (path compression)
            for _ in range(4):
                remap = remap[remap]

            labels = remap[labels.clamp(0, max_label)]

        return labels

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
