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
    ) -> None:
        self.num_keypoints = num_keypoints
        self.dot_radius = dot_radius
        self.colors = colors if colors is not None else COCO_KEYPOINT_COLORS[:num_keypoints]

    def keypoints_to_rgb(
        self,
        keypoints: torch.Tensor,
        visibility: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode keypoints to RGB visualization.

        Args:
            keypoints: [B, N, 2] normalized coordinates in [0, 1] (x, y).
            visibility: [B, N] bool, True = visible. If None, all visible.

        Returns:
            RGB tensor [B, 3, H, W] in [0, 1].
        """
        B, N, _ = keypoints.shape
        H, W = 512, 512  # fixed resolution for visualization
        rgb = torch.zeros(B, 3, H, W)

        if visibility is None:
            visibility = torch.ones(B, N, dtype=torch.bool)

        for b in range(B):
            for n in range(N):
                if not visibility[b, n]:
                    continue
                x = int(keypoints[b, n, 0].item() * W)
                y = int(keypoints[b, n, 1].item() * H)
                color = self.colors[n]

                # Draw filled circle
                for dy in range(-self.dot_radius, self.dot_radius + 1):
                    for dx in range(-self.dot_radius, self.dot_radius + 1):
                        if dx * dx + dy * dy <= self.dot_radius * self.dot_radius:
                            py, px = y + dy, x + dx
                            if 0 <= py < H and 0 <= px < W:
                                rgb[b, :, py, px] = color

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
        keypoints = torch.zeros(B, N, 2)
        visibility = torch.zeros(B, N, dtype=torch.bool)

        for b in range(B):
            img = rgb[b].permute(1, 2, 0)  # [H, W, 3]
            for n in range(N):
                target_color = self.colors[n].to(rgb.device)
                # Find pixels close to target color
                dist = ((img - target_color) ** 2).sum(dim=-1)
                mask = dist < threshold
                if mask.any():
                    ys, xs = torch.where(mask)
                    keypoints[b, n, 0] = xs.float().mean() / W
                    keypoints[b, n, 1] = ys.float().mean() / H
                    visibility[b, n] = True

        return keypoints, visibility


__all__ = ["KeypointRGBMapper", "COCO_KEYPOINT_COLORS", "COCO_KEYPOINT_NAMES"]
