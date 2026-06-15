"""Referring expression segmentation evaluation metrics.

Standard metrics:
- IoU
- Precision@threshold
- Precision, Recall, F-score
"""

from __future__ import annotations

import torch


def evaluate_referring(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute referring expression segmentation metrics.

    Args:
        pred_mask: Predicted binary mask [B, H, W].
        gt_mask: Ground truth binary mask [B, H, W].
        iou_threshold: Threshold for Precision@X.

    Returns:
        Dictionary with IoU, Precision@threshold, Recall, F-score.
    """
    pred_bool = pred_mask.bool()
    gt_bool = gt_mask.bool()

    intersection = (pred_bool & gt_bool).sum(dim=(-2, -1)).float()
    union = (pred_bool | gt_bool).sum(dim=(-2, -1)).float()

    iou = (intersection / union.clamp(min=1)).mean()
    prec_at_thresh = (iou >= iou_threshold).float().mean().item()
    iou_scalar = iou.item()

    tp = intersection
    fp = (pred_bool & ~gt_bool).sum(dim=(-2, -1)).float()
    fn = (~pred_bool & gt_bool).sum(dim=(-2, -1)).float()

    precision = (tp / (tp + fp).clamp(min=1)).mean().item()
    recall = (tp / (tp + fn).clamp(min=1)).mean().item()
    f_score = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "IoU": iou_scalar,
        f"Precision@{iou_threshold}": prec_at_thresh,
        "Precision": precision,
        "Recall": recall,
        "F-score": f_score,
    }


__all__ = ["evaluate_referring"]
