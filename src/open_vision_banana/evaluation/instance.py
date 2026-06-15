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


def evaluate_instance(
    pred_instances: list[dict],
    gt_instances: list[dict],
    iou_thresholds: list[float] = None,
) -> dict[str, float]:
    """Compute instance segmentation AP metrics.

    Args:
        pred_instances: List of predicted instance dicts with 'mask' key.
        gt_instances: List of ground truth instance dicts with 'mask' key.
        iou_thresholds: IoU thresholds for AP computation.

    Returns:
        Dictionary with AP at various thresholds.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75] + [0.5 + 0.05 * i for i in range(10)]

    if len(gt_instances) == 0:
        return {"AP@0.5": 0.0, "AP@0.75": 0.0, "mAP@[0.5:0.95]": 0.0}

    n_pred = len(pred_instances)
    n_gt = len(gt_instances)
    iou_matrix = torch.zeros(n_pred, n_gt)

    for i, pred in enumerate(pred_instances):
        for j, gt in enumerate(gt_instances):
            iou_matrix[i, j] = compute_iou_mask(pred['mask'], gt['mask'])

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
    coco_thresholds = [0.5 + 0.05 * i for i in range(10)]
    coco_aps = [aps.get(f"AP@{t:.2f}", 0.0) for t in coco_thresholds]
    result["mAP@[0.5:0.95]"] = sum(coco_aps) / len(coco_aps)

    return result


__all__ = ["evaluate_instance", "compute_iou_mask", "compute_iou_box"]
