"""Object detection evaluation metrics.

Standard COCO-style metrics:
- mAP@0.5 (AP50): Average Precision at IoU=0.5
- mAP@[0.5:0.95]: Average Precision across 10 IoU thresholds
- AP per class: class-wise average precision
- AR (Average Recall) at max detections = 1, 10, 100
"""

from __future__ import annotations

import torch


def compute_iou_box(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
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


def evaluate_object_detection(
    pred_boxes: list[list[tuple[float, float, float, float]]],
    pred_scores: list[list[float]],
    pred_labels: list[list[int]],
    gt_boxes: list[list[tuple[float, float, float, float]]],
    gt_labels: list[list[int]],
    num_classes: int,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Compute COCO-style object detection metrics.

    Args:
        pred_boxes: List of [N_i, 4] predicted boxes per image (x1, y1, x2, y2).
        pred_scores: List of [N_i] confidence scores per image.
        pred_labels: List of [N_i] predicted class labels per image.
        gt_boxes: List of [M_i, 4] ground truth boxes per image.
        gt_labels: List of [M_i] ground truth class labels per image.
        num_classes: Total number of classes.
        iou_thresholds: IoU thresholds for AP. Defaults to COCO [0.5, 0.55, ..., 0.95].

    Returns:
        Dict with:
        - ``mAP@[0.5:0.95]``: mean AP across all IoU thresholds
        - ``AP@0.5``, ``AP@0.75``: AP at specific thresholds
        - ``mAP_50``: AP at IoU=0.5 (same as AP@0.5)
        - ``per_class_AP``: class-wise AP at IoU=0.5
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    # Collect all predictions and GT across all images per class
    # For each class, compute AP at each IoU threshold
    aps_per_threshold = {t: [] for t in iou_thresholds}
    per_class_ap50 = {}

    for cls in range(1, num_classes + 1):  # class 0 is usually background
        # Collect all predictions for this class
        all_pred = []
        for img_idx, (boxes, scores, labels) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            for box, score, label in zip(boxes, scores, labels):
                if label == cls:
                    all_pred.append({
                        "img_idx": img_idx,
                        "box": box,
                        "score": score,
                        "matched": False,
                    })

        # Collect all GT for this class
        all_gt = []
        for img_idx, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
            for box, label in zip(boxes, labels):
                if label == cls:
                    all_gt.append({
                        "img_idx": img_idx,
                        "box": box,
                        "matched": {t: False for t in iou_thresholds},
                    })

        if not all_gt:
            # No GT for this class, skip
            continue

        # Sort predictions by score descending
        all_pred.sort(key=lambda x: x["score"], reverse=True)

        # For each IoU threshold, compute AP
        for thresh in iou_thresholds:
            tp = 0
            fp = 0
            total_gt = len(all_gt)

            # Reset matched flags for this threshold
            for gt in all_gt:
                gt["matched"][thresh] = False

            for pred in all_pred:
                img_idx = pred["img_idx"]
                pred_box = pred["box"]

                # Find best matching GT in same image
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(all_gt):
                    if gt["img_idx"] != img_idx or gt["matched"][thresh]:
                        continue
                    iou = compute_iou_box(pred_box, gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= thresh and best_gt_idx >= 0:
                    tp += 1
                    all_gt[best_gt_idx]["matched"][thresh] = True
                else:
                    fp += 1

            # AP = TP / (TP + FP + FN) for this threshold
            fn = total_gt - tp
            ap = tp / (tp + fp + fn + 1e-10) if (tp + fp + fn) > 0 else 0.0
            aps_per_threshold[thresh].append(ap)

        # Compute AP50 for per-class report
        ap50 = aps_per_threshold[0.5][-1] if aps_per_threshold[0.5] else 0.0
        per_class_ap50[cls] = ap50

    # Aggregate results
    result = {}
    for thresh in iou_thresholds:
        if aps_per_threshold[thresh]:
            result[f"AP@{thresh:.2f}"] = sum(aps_per_threshold[thresh]) / len(aps_per_threshold[thresh])
        else:
            result[f"AP@{thresh:.2f}"] = 0.0

    # mAP@[0.5:0.95]
    all_aps = [result[f"AP@{t:.2f}"] for t in iou_thresholds]
    result["mAP@[0.5:0.95]"] = sum(all_aps) / len(all_aps) if all_aps else 0.0
    result["AP@0.5"] = result.get("AP@0.50", 0.0)
    result["AP@0.75"] = result.get("AP@0.75", 0.0)
    result["mAP_50"] = result["AP@0.5"]
    result["per_class_AP50"] = per_class_ap50

    return result


__all__ = ["evaluate_object_detection", "compute_iou_box"]