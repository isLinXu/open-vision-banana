"""Depth estimation evaluation metrics.

Standard metrics from the depth estimation literature:
- AbsRel: Absolute Relative Error
- SqRel: Squared Relative Error
- RMSE: Root Mean Squared Error
- RMSE_log: Root Mean Squared Log Error
- δ<1.25, δ<1.25², δ<1.25³: Threshold accuracy
"""

from __future__ import annotations

import torch


def _filter_valid(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter to valid, non-zero GT pixels."""
    if valid_mask is not None:
        pred, gt = pred[valid_mask], gt[valid_mask]
    nonzero = gt > 0
    return pred[nonzero], gt[nonzero]


def depth_abs_rel(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Absolute Relative Error: mean(|pred - gt| / gt)."""
    pred, gt = _filter_valid(pred, gt, valid_mask)
    return (pred - gt).abs().div(gt).mean().item()


def depth_sq_rel(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Squared Relative Error: mean(((pred - gt)² / gt))."""
    pred, gt = _filter_valid(pred, gt, valid_mask)
    return ((pred - gt) ** 2).div(gt).mean().item()


def depth_rmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Root Mean Squared Error."""
    pred, gt = _filter_valid(pred, gt, valid_mask)
    return torch.sqrt(((pred - gt) ** 2).mean()).item()


def depth_rmse_log(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Root Mean Squared Log Error."""
    if valid_mask is not None:
        pred, gt = pred[valid_mask], gt[valid_mask]
    nonzero = (gt > 0) & (pred > 0)
    pred, gt = pred[nonzero], gt[nonzero]
    return torch.sqrt(((pred.log() - gt.log()) ** 2).mean()).item()


def depth_delta(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float = 1.25,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Percentage of predictions within threshold × GT."""
    pred, gt = _filter_valid(pred, gt, valid_mask)
    ratio = torch.maximum(pred / gt, gt / pred)
    return (ratio < threshold).float().mean().item()


def evaluate_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute all standard depth evaluation metrics.

    Args:
        pred: Predicted depth [B, 1, H, W] in meters.
        gt: Ground truth depth [B, 1, H, W] in meters.
        valid_mask: Optional boolean mask for valid pixels.

    Returns:
        Dictionary of metric name → value.
    """
    pred = pred.squeeze(1)
    gt = gt.squeeze(1)

    return {
        "AbsRel": depth_abs_rel(pred, gt, valid_mask),
        "SqRel": depth_sq_rel(pred, gt, valid_mask),
        "RMSE": depth_rmse(pred, gt, valid_mask),
        "RMSE_log": depth_rmse_log(pred, gt, valid_mask),
        "δ<1.25": depth_delta(pred, gt, 1.25, valid_mask),
        "δ<1.25²": depth_delta(pred, gt, 1.25 ** 2, valid_mask),
        "δ<1.25³": depth_delta(pred, gt, 1.25 ** 3, valid_mask),
    }


__all__ = ["evaluate_depth"]
