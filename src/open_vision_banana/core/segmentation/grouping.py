"""Segmentation GroupingMixin stage (P2-1 split)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GroupingMixin:
    # Class-level type declarations for attributes provided by the
    # composing SegmentationDecoder (mixin pattern).
    color_tolerance: float

    @staticmethod
    def _compress_labels(labels: torch.Tensor, max_rounds: int = 24) -> torch.Tensor:
        """Pointer-jump ``labels = labels[labels]`` until convergence (exact root finding)."""
        lut = labels.clone()
        for _ in range(max_rounds):
            next_lut = lut[lut]
            if torch.equal(next_lut, lut):
                break
            lut = next_lut
        return lut

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

            labels.scatter_reduce_(0, src_labels, min_labels, reduce="amin")
            labels.scatter_reduce_(0, dst_labels, min_labels, reduce="amin")
            labels.scatter_reduce_(0, src_indices, min_labels, reduce="amin")
            labels.scatter_reduce_(0, dst_indices, min_labels, reduce="amin")

        # Exact path compression to convergence
        lut = self._compress_labels(labels)

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

        for _ in range(8):
            prev_labels = labels.clone()
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

                dst_bg = torch.roll(bg_mask, shifts=(-dy, -dx), dims=(0, 1))
                valid = valid & ~bg_mask & ~dst_bg

                color_diff = (img - shifted_img) ** 2
                color_dist_sq = color_diff.sum(dim=-1)

                eligible = valid & (color_dist_sq <= tau_sq)

                if not eligible.any():
                    continue

                # Get source and destination pixel indices (tensors, no .tolist())
                src_y, src_x = torch.where(eligible)
                src_indices = src_y * W + src_x
                dst_indices = (src_y + dy) * W + (src_x + dx)

                # Vectorized union: hook component roots as well as endpoints
                src_labels = labels[src_indices]
                dst_labels = labels[dst_indices]
                min_labels = torch.minimum(src_labels, dst_labels)

                labels.scatter_reduce_(0, src_labels, min_labels, reduce="amin")
                labels.scatter_reduce_(0, dst_labels, min_labels, reduce="amin")
                labels.scatter_reduce_(0, src_indices, min_labels, reduce="amin")
                labels.scatter_reduce_(0, dst_indices, min_labels, reduce="amin")

            labels = self._compress_labels(labels)
            if torch.equal(labels, prev_labels):
                break

        lut = labels

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

        for _ in range(12):  # Iterate until seed-anchored components converge
            prev_labels = labels.clone()
            for dy, dx in shifts:
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

                dst_bg = torch.roll(bg_mask, shifts=(-dy, -dx), dims=(0, 1))
                valid = valid & ~bg_mask & ~dst_bg

                if not valid.any():
                    continue

                src_y, src_x = torch.where(valid)
                src_indices = src_y * W + src_x
                dst_indices = (src_y + dy) * W + (src_x + dx)

                # Get current roots for src and dst
                src_roots = labels[src_indices]
                dst_roots = labels[dst_indices]

                # Compare src pixel color AND src root seed against dst component's seed color
                src_pixels = img.reshape(N, 3)[src_indices]
                src_seeds = seed_colors[src_roots]
                dst_seeds = seed_colors[dst_roots]
                pixel_dist_sq = ((src_pixels - dst_seeds) ** 2).sum(dim=-1)
                seed_dist_sq = ((src_seeds - dst_seeds) ** 2).sum(dim=-1)

                eligible = (pixel_dist_sq <= tau_sq) & (seed_dist_sq <= tau_sq)
                if not eligible.any():
                    continue

                eligible_src = src_indices[eligible]
                eligible_dst = dst_indices[eligible]
                eligible_src_roots = src_roots[eligible]
                eligible_dst_roots = dst_roots[eligible]
                min_roots = torch.minimum(eligible_src_roots, eligible_dst_roots)

                # Union: hook both component roots and pixel indices to min_roots
                labels.scatter_reduce_(0, eligible_src_roots, min_roots, reduce="amin")
                labels.scatter_reduce_(0, eligible_dst_roots, min_roots, reduce="amin")
                labels.scatter_reduce_(0, eligible_src, min_roots, reduce="amin")
                labels.scatter_reduce_(0, eligible_dst, min_roots, reduce="amin")

            labels = self._compress_labels(labels)
            seed_colors = seed_colors[labels]
            if torch.equal(labels, prev_labels):
                break

        lut = labels

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


__all__ = ["GroupingMixin"]
