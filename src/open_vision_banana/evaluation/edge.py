"""Edge detection evaluation metrics.

Metrics:
- precision, recall, F1 (standard binary classification)
- ODS (Optimal Dataset Scale): F1 at optimal threshold across dataset
"""

from __future__ import annotations

import torch


def evaluate_edge(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> dict:
    """Evaluate edge detection results.

    Args:
        pred: Predicted edge mask [B, 1, H, W] (bool or float in [0, 1]).
        gt: Ground truth edge mask [B, 1, H, W] (bool).
        threshold: Threshold for converting float predictions to binary.

    Returns:
        Dict with keys: "precision", "recall", "f1", "ods_threshold", "ods_f1".
    """
    if pred.dtype != torch.bool:
        pred = pred > threshold
    if gt.dtype != torch.bool:
        gt = gt > 0.5

    pred = pred.bool()
    gt = gt.bool()

    tp = (pred & gt).sum().float()
    fp = (pred & ~gt).sum().float()
    fn = (~pred & gt).sum().float()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


__all__ = ["evaluate_edge"]
