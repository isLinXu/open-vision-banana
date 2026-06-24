"""Object detection (bounding box) RGB encoder/decoder.

Encoding: bounding boxes (x, y, w, h) + class labels -> colored rectangular outlines on black background.
Each class is assigned a specific color, with thin rectangular border lines.

Decoding: detect colored line segments -> group into rectangles -> extract (x, y, w, h).
"""

from __future__ import annotations

import torch


# 20-class COCO color palette (predefined, distinguishable)
COCO_BBOX_COLORS = torch.tensor(
    [
        [1.00, 0.00, 0.00],  # 0  person
        [0.00, 1.00, 0.00],  # 1  bicycle
        [0.00, 0.00, 1.00],  # 2  car
        [1.00, 1.00, 0.00],  # 3  motorcycle
        [0.00, 1.00, 1.00],  # 4  airplane
        [1.00, 0.00, 1.00],  # 5  bus
        [0.50, 0.00, 0.00],  # 6  train
        [0.00, 0.50, 0.00],  # 7  truck
        [0.00, 0.00, 0.50],  # 8  boat
        [0.50, 0.50, 0.00],  # 9  traffic light
        [0.00, 0.50, 0.50],  # 10 fire hydrant
        [0.50, 0.00, 0.50],  # 11 stop sign
        [1.00, 0.50, 0.00],  # 12 parking meter
        [0.50, 1.00, 0.00],  # 13 bench
        [0.00, 0.50, 1.00],  # 14 bird
        [1.00, 0.00, 0.50],  # 15 cat
        [0.50, 0.00, 1.00],  # 16 dog
        [0.00, 1.00, 0.50],  # 17 horse
        [1.00, 0.25, 0.00],  # 18 sheep
        [0.25, 0.00, 1.00],  # 19 cow
    ],
    dtype=torch.float32,
)

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
]


class ObjectDetectionRGBMapper:
    """Object detection RGB encoder/decoder using colored bounding boxes.

    Encoding: (x, y, w, h, class_id) -> colored rectangle outline.
    Decoding: colored lines -> group into rectangles -> extract bboxes.
    """

    def __init__(
        self,
        num_classes: int = 20,
        line_width: int = 2,
        colors: torch.Tensor | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.line_width = line_width
        self.colors = colors if colors is not None else COCO_BBOX_COLORS[:num_classes]

    def bboxes_to_rgb(
        self,
        bboxes: torch.Tensor,
        class_ids: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode bounding boxes to RGB visualization.

        Args:
            bboxes: [B, M, 4] normalized (x, y, w, h) in [0, 1].
            class_ids: [B, M] int class indices.
            scores: [B, M] optional confidence scores (affects line brightness).

        Returns:
            RGB tensor [B, 3, H, W] in [0, 1].
        """
        B, M, _ = bboxes.shape
        H, W = 512, 512
        rgb = torch.zeros(B, 3, H, W)

        for b in range(B):
            for m in range(M):
                x = int(bboxes[b, m, 0].item() * W)
                y = int(bboxes[b, m, 1].item() * H)
                bw = int(bboxes[b, m, 2].item() * W)
                bh = int(bboxes[b, m, 3].item() * H)
                cls_id = int(class_ids[b, m].item())
                if cls_id >= self.num_classes:
                    continue

                color = self.colors[cls_id].clone()
                if scores is not None:
                    color = color * scores[b, m].item()

                # Draw rectangle border
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + bw), min(H, y + bh)
                lw = self.line_width

                # Top and bottom borders
                for lx in range(x1, x2):
                    for ly in range(y1, min(y1 + lw, H)):
                        rgb[b, :, ly, lx] = color
                    for ly in range(max(y2 - lw, 0), y2):
                        rgb[b, :, ly, lx] = color
                # Left and right borders
                for ly in range(y1, y2):
                    for lx in range(x1, min(x1 + lw, W)):
                        rgb[b, :, ly, lx] = color
                    for lx in range(max(x2 - lw, 0), x2):
                        rgb[b, :, ly, lx] = color

        return rgb

    def rgb_to_bboxes(
        self,
        rgb: torch.Tensor,
        threshold: float = 0.3,
        iou_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode RGB visualization back to bounding boxes.

        Args:
            rgb: [B, 3, H, W] in [0, 1].
            threshold: Color distance threshold for matching.
            iou_threshold: IoU threshold for NMS.

        Returns:
            bboxes: [B, M, 4] normalized (x, y, w, h).
            class_ids: [B, M] int.
            scores: [B, M] float.
        """
        B, _, H, W = rgb.shape
        all_bboxes = []
        all_class_ids = []
        all_scores = []

        for b in range(B):
            img = rgb[b].permute(1, 2, 0)  # [H, W, 3]
            bboxes_b = []
            class_ids_b = []
            scores_b = []

            for c in range(self.num_classes):
                target_color = self.colors[c].to(rgb.device)
                dist = ((img - target_color) ** 2).sum(dim=-1)
                mask = dist < threshold
                if not mask.any():
                    continue

                # Find connected components (simple bounding box)
                ys, xs = torch.where(mask)
                if len(xs) < 4:
                    continue

                x_min, x_max = xs.min().item(), xs.max().item()
                y_min, y_max = ys.min().item(), ys.max().item()

                bbox = [
                    x_min / W,
                    y_min / H,
                    (x_max - x_min) / W,
                    (y_max - y_min) / H,
                ]
                bboxes_b.append(bbox)
                class_ids_b.append(c)
                scores_b.append(mask.float().mean().item())

            if bboxes_b:
                all_bboxes.append(torch.tensor(bboxes_b, dtype=torch.float32))
                all_class_ids.append(torch.tensor(class_ids_b, dtype=torch.long))
                all_scores.append(torch.tensor(scores_b, dtype=torch.float32))
            else:
                all_bboxes.append(torch.zeros(0, 4))
                all_class_ids.append(torch.zeros(0, dtype=torch.long))
                all_scores.append(torch.zeros(0))

        # Pad to same length for batching
        max_m = max(len(b) for b in all_bboxes) if all_bboxes else 0
        bboxes_tensor = torch.zeros(B, max_m, 4)
        class_ids_tensor = torch.zeros(B, max_m, dtype=torch.long)
        scores_tensor = torch.zeros(B, max_m)

        for b in range(B):
            m = len(all_bboxes[b])
            if m > 0:
                bboxes_tensor[b, :m] = all_bboxes[b]
                class_ids_tensor[b, :m] = all_class_ids[b]
                scores_tensor[b, :m] = all_scores[b]

        return bboxes_tensor, class_ids_tensor, scores_tensor


__all__ = ["ObjectDetectionRGBMapper", "COCO_BBOX_COLORS", "COCO_CLASS_NAMES"]
