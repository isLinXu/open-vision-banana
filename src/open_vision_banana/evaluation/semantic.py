"""Semantic segmentation evaluation metrics.

Standard metrics:
- mIoU: Mean Intersection over Union
- Pixel Accuracy
- Per-class IoU
"""

from __future__ import annotations

import torch


from .benchmark import as_batch_hw_labels


def evaluate_semantic(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, float | list[float]]:
    """Compute semantic segmentation metrics.

    Args:
        pred: Predicted labels [B, H, W] or [B, 1, H, W] (integer class indices).
        gt: Ground truth labels [B, H, W] or [H, W] (integer class indices).
        num_classes: Total number of classes.
        ignore_index: Class index to ignore in evaluation.

    Returns:
        Dictionary with mIoU, pixel accuracy, and per-class IoU.
    """
    pred = as_batch_hw_labels(pred)
    gt = as_batch_hw_labels(gt)
    valid = gt != ignore_index
    pred_valid = pred[valid]
    gt_valid = gt[valid]

    flat_pred = pred_valid.long()
    flat_gt = gt_valid.long()
    indices = flat_gt * num_classes + flat_pred
    confusion_flat = torch.bincount(indices, minlength=num_classes * num_classes)
    confusion = confusion_flat[:num_classes * num_classes].reshape(num_classes, num_classes)

    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    denominator = tp + fp + fn
    ious = torch.where(denominator > 0, tp.float() / denominator.float(), torch.nan)

    miou = ious.nanmean().item()
    pixel_acc = (pred_valid == gt_valid).float().mean().item()

    return {
        "mIoU": miou,
        "Pixel Accuracy": pixel_acc,
        "per_class_IoU": ious.tolist(),
    }


__all__ = ["evaluate_semantic"]
