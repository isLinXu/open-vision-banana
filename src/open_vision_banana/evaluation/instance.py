"""Instance segmentation evaluation metrics.

Metrics:
- AP@0.5, AP@0.75
- mAP@[0.5:0.95]
- SA-Co/Gold pmF1, IL_MCC, and cgF1
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

SACO_IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


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


def _match_instance_counts(
    iou_matrix: torch.Tensor,
    threshold: float,
) -> tuple[int, int, int]:
    """Return one-to-one TP/FP/FN counts at an IoU threshold.

    Candidate pairs are matched in descending IoU order so the result does not
    depend on the input instance ordering.
    """
    n_pred, n_gt = iou_matrix.shape
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt

    candidates = torch.nonzero(iou_matrix >= threshold, as_tuple=False)
    if candidates.numel() == 0:
        return 0, n_pred, n_gt

    scores = iou_matrix[candidates[:, 0], candidates[:, 1]]
    order = torch.argsort(scores, descending=True)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for candidate_idx in order.tolist():
        pred_idx, gt_idx = candidates[candidate_idx].tolist()
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)

    tp = len(matched_pred)
    return tp, n_pred - tp, n_gt - tp


def compute_image_level_mcc(
    pred_present: Sequence[bool],
    gt_present: Sequence[bool],
) -> float:
    """Compute the SA-Co image-level Matthews correlation coefficient."""
    if len(pred_present) != len(gt_present):
        raise ValueError("pred_present and gt_present must have the same length")
    if not gt_present:
        return 0.0

    tp = tn = fp = fn = 0
    for pred, gt in zip(pred_present, gt_present, strict=True):
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and gt:
            fn += 1
        else:
            tn += 1

    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0


class SACoAccumulator:
    """Dataset-level accumulator for the official SA-Co/Gold metrics.

    ``pmF1`` is micro-averaged over positive queries at each of the ten IoU
    thresholds, then averaged across thresholds. ``cgF1`` follows equation 7
    of the paper: ``100 * pmF1 * IL_MCC``.
    """

    def __init__(self, iou_thresholds: Sequence[float] | None = None) -> None:
        thresholds = SACO_IOU_THRESHOLDS if iou_thresholds is None else iou_thresholds
        if not thresholds:
            raise ValueError("iou_thresholds must not be empty")
        self.iou_thresholds = tuple(float(value) for value in thresholds)
        self.true_positives = [0] * len(self.iou_thresholds)
        self.false_positives = [0] * len(self.iou_thresholds)
        self.false_negatives = [0] * len(self.iou_thresholds)
        self.pred_presence: list[bool] = []
        self.gt_presence: list[bool] = []

    def update(
        self,
        pred_instances: list[dict],
        gt_instances: list[dict],
        *,
        pred_present: bool | None = None,
        gt_present: bool | None = None,
    ) -> None:
        gt_is_present = bool(gt_instances) if gt_present is None else bool(gt_present)
        pred_is_present = bool(pred_instances) if pred_present is None else bool(pred_present)
        self.gt_presence.append(gt_is_present)
        self.pred_presence.append(pred_is_present)

        if not gt_is_present:
            return

        gated_predictions = pred_instances if pred_is_present else []
        iou_matrix = compute_iou_matrix_batched(gated_predictions, gt_instances)
        for index, threshold in enumerate(self.iou_thresholds):
            tp, fp, fn = _match_instance_counts(iou_matrix, threshold)
            self.true_positives[index] += tp
            self.false_positives[index] += fp
            self.false_negatives[index] += fn

    def compute(self) -> dict[str, float | int]:
        threshold_f1 = []
        for tp, fp, fn in zip(
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            strict=True,
        ):
            denominator = 2 * tp + fp + fn
            threshold_f1.append((2 * tp / denominator) if denominator > 0 else 0.0)

        pm_f1 = sum(threshold_f1) / len(threshold_f1)
        il_mcc = compute_image_level_mcc(self.pred_presence, self.gt_presence)
        return {
            "pmF1": pm_f1,
            "IL_MCC": il_mcc,
            "cgF1": 100.0 * pm_f1 * il_mcc,
            "queries": len(self.gt_presence),
            "positive_queries": sum(self.gt_presence),
        }


def evaluate_saco(
    pred_instances: Sequence[list[dict]],
    gt_instances: Sequence[list[dict]],
    *,
    pred_present: Sequence[bool] | None = None,
    gt_present: Sequence[bool] | None = None,
    iou_thresholds: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Evaluate a complete SA-Co/Gold query set."""
    if len(pred_instances) != len(gt_instances):
        raise ValueError("pred_instances and gt_instances must have the same length")
    if pred_present is not None and len(pred_present) != len(pred_instances):
        raise ValueError("pred_present must match the number of queries")
    if gt_present is not None and len(gt_present) != len(gt_instances):
        raise ValueError("gt_present must match the number of queries")

    accumulator = SACoAccumulator(iou_thresholds)
    for index, (pred_query, gt_query) in enumerate(
        zip(pred_instances, gt_instances, strict=True)
    ):
        accumulator.update(
            pred_query,
            gt_query,
            pred_present=None if pred_present is None else pred_present[index],
            gt_present=None if gt_present is None else gt_present[index],
        )
    return accumulator.compute()


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
                    best_iou = float(iou_matrix[i, j].item())
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


__all__ = [
    "SACO_IOU_THRESHOLDS",
    "SACoAccumulator",
    "compute_image_level_mcc",
    "compute_iou_box",
    "compute_iou_mask",
    "compute_iou_matrix_batched",
    "evaluate_instance",
    "evaluate_saco",
]
