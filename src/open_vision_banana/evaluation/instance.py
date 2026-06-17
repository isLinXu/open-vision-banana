"""Instance segmentation evaluation metrics.

Standard COCO-style metrics:
- AP@0.5, AP@0.75
- mAP@[0.5:0.95]
"""

from __future__ import annotations

import torch


def compute_iou_box(box1: tuple, box2: tuple) -> float:
    """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_iou_mask(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    """Compute IoU between two boolean masks."""
    intersection = (mask1 & mask2).sum().float().item()
    union = (mask1 | mask2).sum().float().item()
    return intersection / union if union > 0 else 0.0


def compute_iou_matrix_batched(
    pred_instances: list[dict],
    gt_instances: list[dict],
) -> torch.Tensor:
    """Compute IoU matrix between all pred and GT instances.

    Vectorized: stacks all masks into tensors and computes intersection
    and union via matrix multiplication.

    Returns:
        IoU matrix [n_pred, n_gt].
    """
    n_pred = len(pred_instances)
    n_gt = len(gt_instances)
    if n_pred == 0 or n_gt == 0:
        return torch.zeros(n_pred, n_gt)

    # Flatten all masks and stack
    pred_masks = torch.stack([
        p['mask'].reshape(-1).float() for p in pred_instances
    ])  # [n_pred, H*W]
    gt_masks = torch.stack([
        g['mask'].reshape(-1).float() for g in gt_instances
    ])  # [n_gt, H*W]

    # Intersection: pred_masks @ gt_masks.T
    intersection = pred_masks @ gt_masks.T  # [n_pred, n_gt]
    # Union: |A| + |B| - |A∩B|
    pred_areas = pred_masks.sum(dim=1).unsqueeze(1)  # [n_pred, 1]
    gt_areas = gt_masks.sum(dim=1).unsqueeze(0)      # [1, n_gt]
    union = pred_areas + gt_areas - intersection

    iou_matrix = intersection / (union + 1e-8)
    return iou_matrix


def evaluate_instance(
    pred_instances: list[dict],
    gt_instances: list[dict],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Compute instance segmentation AP metrics.

    Uses vectorized IoU matrix computation for efficiency.

    Args:
        pred_instances: List of predicted instance dicts with 'mask' key.
        gt_instances: List of ground truth instance dicts with 'mask' key.
        iou_thresholds: IoU thresholds for AP computation.  Defaults to the
            standard COCO set [0.50, 0.55, …, 0.95].

    Returns:
        Dictionary with AP at various thresholds.
    """
    if iou_thresholds is None:
        # COCO: 0.50, 0.55, …, 0.95 (10 thresholds)
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    if len(gt_instances) == 0:
        return {"AP@0.5": 0.0, "AP@0.75": 0.0, "mAP@[0.5:0.95]": 0.0}

    n_pred = len(pred_instances)
    n_gt = len(gt_instances)

    # Vectorized IoU matrix computation
    iou_matrix = compute_iou_matrix_batched(pred_instances, gt_instances)

    aps = {}
    for thresh in iou_thresholds:
        matched_gt = set()
        tp = 0
        fp = 0

        for i in range(n_pred):
            best_j = -1
            best_iou = 0.0
            for j in range(n_gt):
                if j in matched_gt:
                    continue
                if iou_matrix[i, j] > best_iou:
                    best_iou = iou_matrix[i, j]
                    best_j = j

            if best_iou >= thresh and best_j >= 0:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1

        fn = n_gt - len(matched_gt)
        ap = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        aps[f"AP@{thresh:.2f}"] = ap

    result = {
        "AP@0.5": aps.get("AP@0.50", 0.0),
        "AP@0.75": aps.get("AP@0.75", 0.0),
    }
    coco_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    coco_aps = [aps.get(f"AP@{t:.2f}", 0.0) for t in coco_thresholds]
    result["mAP@[0.5:0.95]"] = sum(coco_aps) / len(coco_aps)

    return result


__all__ = ["evaluate_instance", "compute_iou_mask", "compute_iou_matrix_batched", "compute_iou_box"]
