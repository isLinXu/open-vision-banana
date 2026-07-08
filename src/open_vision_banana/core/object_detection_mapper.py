"""Object detection (bounding box) RGB encoder/decoder.

Encoding: bounding boxes (x, y, w, h) + class labels -> colored rectangular outlines on black background.
Each class is assigned a specific color, with thin rectangular border lines.

Decoding: detect colored line segments -> group into rectangles -> extract (x, y, w, h).
NMS (Non-Maximum Suppression) is applied to remove duplicate detections.
"""

from __future__ import annotations

import torch


def _nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Non-Maximum Suppression for bounding boxes.

    Args:
        boxes: [N, 4] tensor in (x1, y1, x2, y2) format, normalized to [0, 1].
        scores: [N] confidence scores.
        iou_threshold: IoU threshold above which duplicates are removed.

    Returns:
        indices: [K] long tensor of kept box indices.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Sort by score descending
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        # Compute IoU between the selected box and the rest
        xx1 = torch.max(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.max(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.min(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.min(boxes[i, 3], boxes[rest, 3])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter
        iou = inter / union.clamp(min=1e-8)

        # Keep boxes with IoU below threshold
        order = rest[iou < iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


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


def _connected_components_pytorch(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """4-connected component labeling in pure PyTorch.

    Args:
        mask: [H, W] bool tensor on any device.

    Returns:
        labels: [H, W] long tensor on same device as mask (0=background).
        num_labels: int, number of components.
    """
    H, W = mask.shape
    device = mask.device

    ys, xs = torch.where(mask)
    n = ys.shape[0]
    if n == 0:
        return torch.zeros(H, W, dtype=torch.long, device=device), 0

    flat = ys * W + xs
    coord_to_idx = torch.full((H * W,), -1, dtype=torch.long, device=device)
    coord_to_idx[flat] = torch.arange(n, device=device)

    parent = torch.arange(n, device=device)

    def _find(x: torch.Tensor) -> torch.Tensor:
        while True:
            px = parent[x]
            changed = px != x
            if not changed.any():
                break
            x = torch.where(changed, px, x)
        return x

    for dy, dx in [(0, 1), (1, 0)]:
        ny = ys + dy
        nx = xs + dx
        valid = (ny < H) & (nx < W)
        c_idx = torch.arange(n, device=device)[valid]
        ny, nx = ny[valid], nx[valid]
        n_flat = ny * W + nx
        n_idx = coord_to_idx[n_flat]
        conn = n_idx >= 0

        if conn.any():
            c_idx = c_idx[conn]
            n_idx = n_idx[conn]
            rc = _find(c_idx)
            rn = _find(n_idx)
            merge = rc != rn
            if merge.any():
                parent[rc[merge]] = rn[merge]

    parent = _find(torch.arange(n, device=device))

    unique = torch.unique(parent)
    label_map = torch.searchsorted(unique, parent)

    labels = torch.zeros(H, W, dtype=torch.long, device=device)
    labels[ys, xs] = label_map + 1

    return labels, len(unique)


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
        output_height: int = 512,
        output_width: int = 512,
    ) -> None:
        self.num_classes = num_classes
        self.line_width = line_width
        self.colors = colors if colors is not None else COCO_BBOX_COLORS[:num_classes]
        self.output_height = output_height
        self.output_width = output_width

    def bboxes_to_rgb(
        self,
        bboxes: torch.Tensor,
        class_ids: torch.Tensor,
        scores: torch.Tensor | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Encode bounding boxes to RGB visualization.

        Args:
            bboxes: [B, M, 4] normalized (x, y, w, h) in [0, 1].
            class_ids: [B, M] int class indices.
            scores: [B, M] optional confidence scores (affects line brightness).
            output_size: Optional (H, W) override for the output resolution.
                Defaults to ``self.output_height`` / ``self.output_width``.

        Returns:
            RGB tensor [B, 3, H, W] in [0, 1].
        """
        B, M, _ = bboxes.shape
        H, W = output_size if output_size is not None else (self.output_height, self.output_width)
        device = bboxes.device
        rgb = torch.zeros(B, 3, H, W, device=device)
        colors = self.colors.to(device)
        lw = self.line_width

        x = (bboxes[..., 0] * W).long().clamp(0, W - 1)
        y = (bboxes[..., 1] * H).long().clamp(0, H - 1)
        bw = (bboxes[..., 2] * W).long().clamp(1, W)
        bh = (bboxes[..., 3] * H).long().clamp(1, H)
        valid = class_ids < self.num_classes

        # P1-4: fully vectorized bbox drawing using index_put_
        valid_b, valid_m = valid.nonzero(as_tuple=True)
        K = valid_b.shape[0]
        if K == 0:
            return rgb

        x1 = x[valid_b, valid_m]
        y1 = y[valid_b, valid_m]
        x2 = (x1 + bw[valid_b, valid_m]).clamp(max=W)
        y2 = (y1 + bh[valid_b, valid_m]).clamp(max=H)

        cls_ids = class_ids[valid_b, valid_m].long()
        colors_per_box = colors[cls_ids]
        if scores is not None:
            colors_per_box = colors_per_box * scores[valid_b, valid_m].unsqueeze(1)

        max_w = int((x2 - x1).max().item())
        max_h = int((y2 - y1).max().item())

        h_range = torch.arange(0, max_h, device=device).view(1, -1, 1)
        w_range = torch.arange(0, max_w, device=device).view(1, -1, 1)
        lw_range = torch.arange(0, lw, device=device).view(1, 1, -1)

        bbox_w = (x2 - x1).view(K, 1, 1)
        bbox_h = (y2 - y1).view(K, 1, 1)

        # --- Top border ---
        top_h = torch.min(y1 + lw, torch.full_like(y1, H)) - y1
        top_mask = (w_range < bbox_w) & (lw_range < top_h.view(K, 1, 1))
        top_y = (y1.view(K, 1, 1) + lw_range).expand(K, max_w, lw)[top_mask]
        top_x = (x1.view(K, 1, 1) + w_range).expand(K, max_w, lw)[top_mask]
        top_b = valid_b.view(K, 1, 1).expand(K, max_w, lw)[top_mask]

        # --- Bottom border ---
        bot_y_start = torch.max(y2 - lw, torch.zeros_like(y2))
        bot_h = y2 - bot_y_start
        bot_mask = (w_range < bbox_w) & (lw_range < bot_h.view(K, 1, 1))
        bot_y = (bot_y_start.view(K, 1, 1) + lw_range).expand(K, max_w, lw)[bot_mask]
        bot_x = (x1.view(K, 1, 1) + w_range).expand(K, max_w, lw)[bot_mask]
        bot_b = valid_b.view(K, 1, 1).expand(K, max_w, lw)[bot_mask]

        # --- Left border ---
        left_w = torch.min(x1 + lw, torch.full_like(x1, W)) - x1
        left_mask = (lw_range < left_w.view(K, 1, 1)) & (h_range < bbox_h)
        left_y = (y1.view(K, 1, 1) + h_range).expand(K, max_h, lw)[left_mask]
        left_x = (x1.view(K, 1, 1) + lw_range).expand(K, max_h, lw)[left_mask]
        left_b = valid_b.view(K, 1, 1).expand(K, max_h, lw)[left_mask]

        # --- Right border ---
        right_x_start = torch.max(x2 - lw, torch.zeros_like(x2))
        right_w = x2 - right_x_start
        right_mask = (lw_range < right_w.view(K, 1, 1)) & (h_range < bbox_h)
        right_y = (y1.view(K, 1, 1) + h_range).expand(K, max_h, lw)[right_mask]
        right_x = (right_x_start.view(K, 1, 1) + lw_range).expand(K, max_h, lw)[right_mask]
        right_b = valid_b.view(K, 1, 1).expand(K, max_h, lw)[right_mask]

        # Merge coordinates and colors
        all_b = torch.cat([top_b, bot_b, left_b, right_b])
        all_y = torch.cat([top_y, bot_y, left_y, right_y])
        all_x = torch.cat([top_x, bot_x, left_x, right_x])

        top_pixels = top_mask.sum(dim=(1, 2))
        bot_pixels = bot_mask.sum(dim=(1, 2))
        left_pixels = left_mask.sum(dim=(1, 2))
        right_pixels = right_mask.sum(dim=(1, 2))

        top_colors = torch.repeat_interleave(colors_per_box, top_pixels, dim=0)
        bot_colors = torch.repeat_interleave(colors_per_box, bot_pixels, dim=0)
        left_colors = torch.repeat_interleave(colors_per_box, left_pixels, dim=0)
        right_colors = torch.repeat_interleave(colors_per_box, right_pixels, dim=0)

        all_colors = torch.cat([top_colors, bot_colors, left_colors, right_colors])

        rgb_perm = rgb.permute(0, 2, 3, 1)
        rgb_perm[all_b, all_y, all_x] = all_colors

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
        device = rgb.device
        colors = self.colors.to(device)
        all_bboxes = []
        all_class_ids = []
        all_scores = []

        for b in range(B):
            img = rgb[b].permute(1, 2, 0)  # [H, W, 3]
            bboxes_b = []
            class_ids_b = []
            scores_b = []

            for c in range(self.num_classes):
                target_color = colors[c]
                dist = ((img - target_color) ** 2).sum(dim=-1)
                mask = dist < threshold
                if not mask.any():
                    continue

                # P1-5: connected-component labeling to separate same-color instances
                labels, num_labels = _connected_components_pytorch(mask)
                for label_id in range(1, num_labels + 1):
                    component_mask = labels == label_id
                    ys, xs = torch.where(component_mask)
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
                    scores_b.append(component_mask.float().mean().item())

            if bboxes_b:
                bboxes_t = torch.tensor(bboxes_b, dtype=torch.float32)
                class_ids_t = torch.tensor(class_ids_b, dtype=torch.long)
                scores_t = torch.tensor(scores_b, dtype=torch.float32)

                # P0 fix: Apply NMS per class to remove duplicate detections.
                # Without NMS, overlapping connected components from the same
                # object produce multiple near-identical detections.
                keep_mask = torch.ones(len(bboxes_t), dtype=torch.bool)
                for c in class_ids_t.unique():
                    c_mask = class_ids_t == c
                    c_indices = c_mask.nonzero(as_tuple=True)[0]
                    if len(c_indices) <= 1:
                        continue
                    # Convert (x, y, w, h) → (x1, y1, x2, y2) for NMS
                    c_boxes = bboxes_t[c_indices]
                    c_boxes_xyxy = torch.stack([
                        c_boxes[:, 0],
                        c_boxes[:, 1],
                        c_boxes[:, 0] + c_boxes[:, 2],
                        c_boxes[:, 1] + c_boxes[:, 3],
                    ], dim=1)
                    c_scores = scores_t[c_indices]
                    keep_local = _nms(c_boxes_xyxy, c_scores, iou_threshold)
                    # Build keep mask: keep only the selected indices for this class
                    c_keep_bool = torch.zeros(len(c_indices), dtype=torch.bool)
                    c_keep_bool[keep_local] = True
                    keep_mask[c_indices] = c_keep_bool

                bboxes_t = bboxes_t[keep_mask]
                class_ids_t = class_ids_t[keep_mask]
                scores_t = scores_t[keep_mask]

                all_bboxes.append(bboxes_t)
                all_class_ids.append(class_ids_t)
                all_scores.append(scores_t)
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


__all__ = ["ObjectDetectionRGBMapper", "COCO_BBOX_COLORS", "COCO_CLASS_NAMES", "_nms"]
