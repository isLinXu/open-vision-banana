"""2D keypoint detection RGB encoder/decoder.

Encoding: keypoint coordinates (x, y) + keypoint type -> colored dots on black background.
Each keypoint type is assigned a specific color, rendered as a small filled circle at (x, y).

Decoding: cluster pixels of each keypoint color -> compute centroid -> recover coordinates.

Supports COCO-style (17 keypoints) and custom keypoint sets.
"""

from __future__ import annotations

import torch

# COCO 17 keypoint color palette (predefined, distinguishable)
COCO_KEYPOINT_COLORS = torch.tensor(
    [
        [1.00, 0.00, 0.00],  # 0  nose         — red
        [0.00, 1.00, 0.00],  # 1  left_eye     — green
        [0.00, 0.00, 1.00],  # 2  right_eye    — blue
        [1.00, 1.00, 0.00],  # 3  left_ear     — yellow
        [0.00, 1.00, 1.00],  # 4  right_ear    — cyan
        [1.00, 0.00, 1.00],  # 5  left_shoulder— magenta
        [0.50, 0.00, 0.00],  # 6  right_shoulder— dark red
        [0.00, 0.50, 0.00],  # 7  left_elbow   — dark green
        [0.00, 0.00, 0.50],  # 8  right_elbow  — dark blue
        [0.50, 0.50, 0.00],  # 9  left_wrist   — olive
        [0.00, 0.50, 0.50],  # 10 right_wrist  — teal
        [0.50, 0.00, 0.50],  # 11 left_hip     — purple
        [1.00, 0.50, 0.00],  # 12 right_hip    — orange
        [0.50, 1.00, 0.00],  # 13 left_knee    — lime
        [0.00, 0.50, 1.00],  # 14 right_knee   — sky blue
        [1.00, 0.00, 0.50],  # 15 left_ankle   — pink
        [0.50, 0.00, 1.00],  # 16 right_ankle  — violet
    ],
    dtype=torch.float32,
)

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class KeypointRGBMapper:
    """2D keypoint detection RGB encoder/decoder.

    Encoding: keypoint array -> colored dots on black background.
    Decoding: colored clusters -> centroid coordinates.
    """

    def __init__(
        self,
        num_keypoints: int = 17,
        dot_radius: int = 3,
        colors: torch.Tensor | None = None,
        output_height: int = 512,
        output_width: int = 512,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.dot_radius = dot_radius
        self.colors = colors if colors is not None else COCO_KEYPOINT_COLORS[:num_keypoints]
        self.output_height = output_height
        self.output_width = output_width

    def keypoints_to_rgb(
        self,
        keypoints: torch.Tensor,
        visibility: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Encode keypoints to RGB visualization.

        Args:
            keypoints: [B, N, 2] normalized coordinates in [0, 1] (x, y).
            visibility: [B, N] bool, True = visible. If None, all visible.
            output_size: Optional (H, W) override for the output resolution.
                Defaults to ``self.output_height`` / ``self.output_width``.

        Returns:
            RGB tensor [B, 3, H, W] in [0, 1].
        """
        B, N, _ = keypoints.shape
        H, W = output_size if output_size is not None else (self.output_height, self.output_width)
        device = keypoints.device
        rgb = torch.zeros(B, 3, H, W, device=device)

        if visibility is None:
            visibility = torch.ones(B, N, dtype=torch.bool, device=device)

        colors = self.colors.to(device)
        px = (keypoints[..., 0] * W).long().clamp(0, W - 1)
        py = (keypoints[..., 1] * H).long().clamp(0, H - 1)

        r = self.dot_radius
        dy = torch.arange(-r, r + 1, device=device)
        dx = torch.arange(-r, r + 1, device=device)
        dy_grid, dx_grid = torch.meshgrid(dy, dx, indexing="ij")
        circle_mask = dx_grid ** 2 + dy_grid ** 2 <= r * r
        offsets = torch.stack([dy_grid[circle_mask], dx_grid[circle_mask]], dim=-1)
        num_offsets = offsets.shape[0]

        # P1-3: vectorized encoding — eliminate Python loop over batch
        all_y = (py.unsqueeze(2) + offsets[:, 0]).clamp(0, H - 1).long()  # [B, N, num_offsets]
        all_x = (px.unsqueeze(2) + offsets[:, 1]).clamp(0, W - 1).long()  # [B, N, num_offsets]

        vis_expanded = visibility.unsqueeze(2).expand(-1, -1, num_offsets)  # [B, N, num_offsets]
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, num_offsets)

        valid_mask = vis_expanded
        b_flat = b_idx[valid_mask]
        y_flat = all_y[valid_mask]
        x_flat = all_x[valid_mask]

        colors_expanded = colors.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]
        colors_per_offset = colors_expanded.unsqueeze(2).expand(-1, -1, num_offsets, -1)  # [B, N, num_offsets, 3]
        colors_flat = colors_per_offset[valid_mask]  # [K, 3]

        rgb_perm = rgb.permute(0, 2, 3, 1)  # [B, H, W, 3]
        rgb_perm[b_flat, y_flat, x_flat] = colors_flat

        return rgb

    def rgb_to_keypoints(
        self,
        rgb: torch.Tensor,
        threshold: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode RGB visualization back to keypoint coordinates.

        Args:
            rgb: [B, 3, H, W] in [0, 1].
            threshold: Color distance threshold for matching.

        Returns:
            keypoints: [B, N, 2] normalized coordinates in [0, 1].
            visibility: [B, N] bool.
        """
        B, _, H, W = rgb.shape
        N = self.num_keypoints
        device = rgb.device

        colors = self.colors.to(device)
        rgb_perm = rgb.permute(0, 2, 3, 1)

        colors_expanded = colors.view(1, 1, 1, N, 3)
        rgb_expanded = rgb_perm.unsqueeze(3)

        dist = ((rgb_expanded - colors_expanded) ** 2).sum(dim=-1)
        mask = dist < threshold

        ys = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1, 1)
        xs = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W, 1)

        ys_masked = ys * mask
        xs_masked = xs * mask

        count = mask.sum(dim=(1, 2))
        keypoints_y = ys_masked.sum(dim=(1, 2)) / (count + 1e-8)
        keypoints_x = xs_masked.sum(dim=(1, 2)) / (count + 1e-8)

        keypoints = torch.stack([keypoints_x / W, keypoints_y / H], dim=-1)
        visibility = count > 0

        return keypoints, visibility


__all__ = ["KeypointRGBMapper", "COCO_KEYPOINT_COLORS", "COCO_KEYPOINT_NAMES"]
