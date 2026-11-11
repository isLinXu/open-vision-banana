"""Referring expression segmentation evaluation metrics.

Standard metrics:
- cIoU: cumulative intersection over cumulative union
- gIoU: mean per-query IoU (ReasonSeg naming, not box generalized IoU)
- Precision@threshold
- Precision, Recall, F-score
"""

from __future__ import annotations

import torch


class ReferringAccumulator:
    """Accumulate RefCOCOg/ReasonSeg metrics without batch-size bias."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = float(iou_threshold)
        self.intersection = 0.0
        self.union = 0.0
        self.sum_iou = 0.0
        self.samples = 0
        self.hits = 0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(
        self, pred_mask: torch.Tensor, gt_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        if valid_mask is not None and valid_mask.shape != gt_mask.shape:
            raise ValueError("valid_mask and gt_mask must have the same shape")
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"pred_mask and gt_mask must have the same shape, got "
                f"{tuple(pred_mask.shape)} and {tuple(gt_mask.shape)}"
            )
        if pred_mask.dim() == 2:
            pred_mask = pred_mask.unsqueeze(0)
            gt_mask = gt_mask.unsqueeze(0)

        pred_bool = pred_mask.bool()
        gt_bool = gt_mask.bool()
        if valid_mask is not None:
            valid = valid_mask.to(device=pred_bool.device).bool().reshape_as(pred_bool)
            if not bool(valid.flatten(1).any(dim=1).all()):
                raise ValueError("Referring sample has no valid pixels")
            pred_bool = pred_bool & valid
            gt_bool = gt_bool & valid
        intersection = (pred_bool & gt_bool).sum(dim=(-2, -1)).double()
        union = (pred_bool | gt_bool).sum(dim=(-2, -1)).double()
        per_sample_iou = torch.where(
            union > 0,
            intersection / union,
            torch.ones_like(union),
        )

        self.intersection += intersection.sum().item()
        self.union += union.sum().item()
        self.sum_iou += per_sample_iou.sum().item()
        self.samples += int(per_sample_iou.numel())
        self.hits += int((per_sample_iou >= self.iou_threshold).sum().item())
        self.tp += intersection.sum().item()
        self.fp += (pred_bool & ~gt_bool).sum().item()
        self.fn += (~pred_bool & gt_bool).sum().item()

    def compute(self) -> dict[str, float | int]:
        if self.samples == 0:
            return {}
        c_iou = self.intersection / self.union if self.union > 0 else 1.0
        g_iou = self.sum_iou / self.samples
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else 0.0
        f_score = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return {
            "IoU": g_iou,
            "cIoU": c_iou,
            "gIoU": g_iou,
            f"Precision@{self.iou_threshold}": self.hits / self.samples,
            "Precision": precision,
            "Recall": recall,
            "F-score": f_score,
            "samples": self.samples,
        }


def evaluate_referring(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    iou_threshold: float = 0.5,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute referring expression segmentation metrics.

    Args:
        pred_mask: Predicted binary mask [B, H, W].
        gt_mask: Ground truth binary mask [B, H, W].
        iou_threshold: Threshold for Precision@X.

    Returns:
        Dictionary with IoU, Precision@threshold, Recall, F-score.
    """
    accumulator = ReferringAccumulator(iou_threshold=iou_threshold)
    accumulator.update(pred_mask, gt_mask, valid_mask=valid_mask)
    return {
        key: float(value)
        for key, value in accumulator.compute().items()
        if key != "samples"
    }


__all__ = ["ReferringAccumulator", "evaluate_referring"]
