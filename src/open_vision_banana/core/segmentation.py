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

    def _compute_adaptive_tau(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Compute adaptive color tolerance map based on local image statistics.

        Uses conv2d-based Sobel filters for efficient gradient computation.

        Args:
            rgb_image: [B, 3, H, W] in [0, 1].

        Returns:
            Adaptive τ map [B, H, W] with values in [τ_min, τ_max].
        """
        B, C, H, W = rgb_image.shape
        device = rgb_image.device

        # Grayscale for gradient computation
        gray = rgb_image.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # Sobel filters via conv2d (more efficient than manual differencing)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=gray.dtype,
            device=device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=gray.dtype,
            device=device,
        ).view(1, 1, 3, 3)

        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(gx**2 + gy**2 + 1e-10)
        gradient_magnitude = gradient_magnitude.squeeze(1)  # [B, H, W]

        # Normalize gradient to [0, 1]
        grad_norm = gradient_magnitude / (gradient_magnitude.max() + 1e-8)

        # Adaptive τ: inversely proportional to gradient magnitude
        tau_min = self.color_tolerance * 0.5
        tau_max = self.color_tolerance * 1.5
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

        # One-pass edge scanning — 16 directions (8-connectivity + knight's move)
        shifts = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # orthogonal
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonal
            (-2, -1), (-2, 1), (2, -1), (2, 1),  # knight's move (vertical 2)
            (-1, -2), (-1, 2), (1, -2), (1, 2),  # knight's move (horizontal 2)
        ]

        for dy, dx in shifts:
            shifted_img = torch.roll(img, shifts=(-dy, -dx), dims=(0, 1))

            # Build valid neighbor mask — handle 1px and 2px offsets
            valid = torch.ones(H, W, dtype=torch.bool, device=device)
            if dy == -1:
                valid[0, :] = False
            elif dy == -2:
                valid[0:2, :] = False
            elif dy == 1:
                valid[-1, :] = False
            elif dy == 2:
                valid[-2:, :] = False
            if dx == -1:
                valid[:, 0] = False
            elif dx == -2:
                valid[:, 0:2] = False
            elif dx == 1:
                valid[:, -1] = False
            elif dx == 2:
                valid[:, -2:] = False

            valid = valid & ~bg_mask

            color_diff = (img - shifted_img) ** 2
            color_dist_sq = color_diff.sum(dim=-1)

            # Use adaptive τ: take the maximum of source and destination τ
            # This ensures both pixels agree on the tolerance
            shifted_tau = torch.roll(tau_map, shifts=(-dy, -dx), dims=(0, 1))
            adaptive_tau = torch.maximum(tau_map, shifted_tau)
            adaptive_tau_sq = adaptive_tau**2

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

        # Relabel to consecutive integers (fully vectorized)
        unique_labels, inverse = labels.reshape(-1).unique(return_inverse=True)
        is_bg = unique_labels < 0
        # Vectorized: bg → 0, non-bg → 1, 2, 3, ...
        # cumsum of (~is_bg) gives consecutive IDs starting from 0 for bg,
        # then 1, 2, 3, ... for each non-bg label in order.
        non_bg_cumsum = (~is_bg).cumsum(dim=0)
        new_id = non_bg_cumsum.long()
        # Ensure background labels map to 0
        new_id = torch.where(is_bg, torch.zeros_like(new_id), new_id)

        result = new_id[inverse].reshape(H, W)
        return result

    def _union_find_grouping(self, img: torch.Tensor, bg_mask: torch.Tensor) -> torch.Tensor:
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
        5. Relabel to consecutive integers via unique + cumsum

        Complexity: O(E α(N)) effective, fully vectorized.
        """
        H, W, C = img.shape
        device = img.device
        N = H * W
        tau_sq = self.color_tolerance**2

        # Step 1: Initialize labels — each pixel is its own component
        labels = torch.arange(N, dtype=torch.long, device=device)

        # Step 2: One-pass edge scanning — 16 directions (8-connectivity + knight's move)
        # 8-connectivity: orthogonal + diagonal
        # Knight's move: (±2, ±1), (±1, ±2) — 16-connectivity per paper Appendix A
        shifts = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # orthogonal
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonal
            (-2, -1), (-2, 1), (2, -1), (2, 1),  # knight's move (vertical 2)
            (-1, -2), (-1, 2), (1, -2), (1, 2),  # knight's move (horizontal 2)
        ]

        for dy, dx in shifts:
            shifted_img = torch.roll(img, shifts=(-dy, -dx), dims=(0, 1))

            # Build valid neighbor mask — handle 1px and 2px offsets
            valid = torch.ones(H, W, dtype=torch.bool, device=device)
            if dy == -1:
                valid[0, :] = False
            elif dy == -2:
                valid[0:2, :] = False
            elif dy == 1:
                valid[-1, :] = False
            elif dy == 2:
                valid[-2:, :] = False
            if dx == -1:
                valid[:, 0] = False
            elif dx == -2:
                valid[:, 0:2] = False
            elif dx == 1:
                valid[:, -1] = False
            elif dx == 2:
                valid[:, -2:] = False

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

        # Step 4: Relabel to consecutive integers (fully vectorized)
        unique_labels, inverse = labels.reshape(-1).unique(return_inverse=True)
        is_bg = unique_labels < 0
        # cumsum gives consecutive IDs: bg=0, then 1, 2, 3, ...
        non_bg_cumsum = (~is_bg).cumsum(dim=0)
        new_id = non_bg_cumsum.long()
        new_id = torch.where(is_bg, torch.zeros_like(new_id), new_id)

        result = new_id[inverse].reshape(H, W)
        return result

    def _seed_based_union_find(self, img: torch.Tensor, bg_mask: torch.Tensor) -> torch.Tensor:
        """Stage 2: Seed-based Union-Find grouping.

        From the paper (Appendix A):
        Each pixel is compared against the *seed* color of its neighbor's
        component, not the neighbor's own color.  This produces more
        compact instances because the comparison criterion is anchored
        to the component's original seed rather than drifting as the
        component grows.

        Algorithm:
        1. Initialize labels[i] = i, seed_colors[i] = img[i] for all N pixels
        2. For 5 iterations:
           a. Scan 16 directions
           b. For each eligible edge, compare pixel color vs neighbor component seed
           c. If |img[pixel] - seed[neighbor_root]|² ≤ τ², merge
           d. Update seed: seed[new_root] = seed[min_root]
        3. Path compression + relabel to consecutive integers

        Args:
            img: [H, W, 3] image in [0, 255].
            bg_mask: [H, W] boolean background mask.

        Returns:
            Label map [H, W] with consecutive integer labels.
        """
        H, W, C = img.shape
        device = img.device
        N = H * W
        tau_sq = self.color_tolerance**2

        # Step 1: Initialize
        labels = torch.arange(N, dtype=torch.long, device=device)
        # Seed colors: each pixel's own RGB value, [N, 3]
        seed_colors = img.reshape(N, 3).clone()

        # 16-connectivity shifts
        shifts = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # orthogonal
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonal
            (-2, -1), (-2, 1), (2, -1), (2, 1),  # knight's move
            (-1, -2), (-1, 2), (1, -2), (1, 2),  # knight's move
        ]

        for _ in range(5):  # 5 iterations until convergence
            changed = False
            for dy, dx in shifts:
                shifted_img = torch.roll(img, shifts=(-dy, -dx), dims=(0, 1))

                valid = torch.ones(H, W, dtype=torch.bool, device=device)
                if dy == -1:
                    valid[0, :] = False
                elif dy == -2:
                    valid[0:2, :] = False
                elif dy == 1:
                    valid[-1, :] = False
                elif dy == 2:
                    valid[-2:, :] = False
                if dx == -1:
                    valid[:, 0] = False
                elif dx == -2:
                    valid[:, 0:2] = False
                elif dx == 1:
                    valid[:, -1] = False
                elif dx == 2:
                    valid[:, -2:] = False

                valid = valid & ~bg_mask

                if not valid.any():
                    continue

                src_y, src_x = torch.where(valid)
                src_indices = src_y * W + src_x
                dst_indices = (src_y + dy) * W + (src_x + dx)

                # Get current roots for src and dst
                src_roots = labels[src_indices]
                dst_roots = labels[dst_indices]

                # Compare src pixel color against dst component's seed color
                src_pixels = img.reshape(N, 3)[src_indices]
                dst_seeds = seed_colors[dst_roots]
                color_dist_sq = ((src_pixels - dst_seeds) ** 2).sum(dim=-1)

                eligible = color_dist_sq <= tau_sq
                if not eligible.any():
                    continue

                changed = True

                eligible_src = src_indices[eligible]
                eligible_dst = dst_indices[eligible]
                min_roots = torch.minimum(eligible_src, eligible_dst)

                # Union: scatter minimum root
                labels.scatter_(0, eligible_src, torch.minimum(labels[eligible_src], min_roots))
                labels.scatter_(0, eligible_dst, torch.minimum(labels[eligible_dst], min_roots))

                # Update seeds: for merged components, use min root's seed
                # We need to update seeds for all roots that were merged
                merged_src = eligible_src
                merged_dst = eligible_dst
                for _ in range(2):
                    # Propagate seeds to roots
                    root_labels = labels.clone()
                    for _ in range(4):
                        root_labels = root_labels[root_labels]
                    # Update seed for any component whose root changed
                    seed_colors[merged_src] = seed_colors[root_labels[merged_src]]
                    seed_colors[merged_dst] = seed_colors[root_labels[merged_dst]]

            if not changed:
                break

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
        non_bg_cumsum = (~is_bg).cumsum(dim=0)
        new_id = non_bg_cumsum.long()
        new_id = torch.where(is_bg, torch.zeros_like(new_id), new_id)

        result = new_id[inverse].reshape(H, W)
        return result

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

        # bbox_area = ((y_max - y_min + 1) * (x_max - x_min + 1)).float()  # unused
        pixel_area = count.float()

        # Extract data for unique_labels only
        ul = unique_labels.long()
        mc = mean_colors[ul]  # [K, 3]
        # ba = bbox_area[ul]       # [K]  # unused, kept for reference
        pa = pixel_area[ul]  # [K]
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


__all__ = ["SegmentationDecoder"]
