"""Optical flow evaluation metrics.

Metrics:
- EPE (End-Point Error): ||pred - gt||_2 averaged over pixels
- AE (Angular Error): average angle between flow vectors
- F1 (percentage of pixels with EPE < 3px or relative < 5%)
"""

from __future__ import annotations

import math

import torch


def evaluate_flow(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict:
    """Evaluate optical flow results.

    Args:
        pred: Predicted flow [B, 2, H, W] (u, v).
        gt: Ground truth flow [B, 2, H, W] (u, v).
        valid_mask: Optional valid pixel mask [B, 1, H, W] (bool).
            If None, all pixels are considered valid.

    Returns:
        Dict with keys: "epe", "ae_deg", "f1".
    """
    if valid_mask is not None:
        pred = pred * valid_mask.float()
        gt = gt * valid_mask.float()
        denom = valid_mask.sum().float().clamp(min=1e-10)
    else:
        denom = pred.numel() / 2  # per-pixel count (B*H*W)
        denom = torch.tensor(denom, dtype=torch.float32, device=pred.device)

    diff = pred - gt
    epe = torch.sqrt(diff[:, 0:1] ** 2 + diff[:, 1:2] ** 2 + 1e-10)

    # End-Point Error
    mean_epe = epe.sum() / denom

    # Angular Error (in degrees)
    pred_norm = torch.sqrt(pred[:, 0:1] ** 2 + pred[:, 1:2] ** 2 + 1e-10)
    gt_norm = torch.sqrt(gt[:, 0:1] ** 2 + gt[:, 1:2] ** 2 + 1e-10)
    dot = pred[:, 0:1] * gt[:, 0:1] + pred[:, 1:2] * gt[:, 1:2]
    cos_angle = (dot / (pred_norm * gt_norm + 1e-10)).clamp(-1.0, 1.0)
    angle_rad = torch.acos(cos_angle)
    mean_ae = (angle_rad * 180.0 / math.pi).sum() / denom

    # F1: percentage of pixels with EPE < 3px or relative error < 5%
    gt_norm_safe = gt_norm.clamp(min=1e-10)
    relative_err = epe / (gt_norm_safe + 1e-10)
    correct = (epe < 3.0) | (relative_err < 0.05)
    f1 = correct.float().sum() / denom

    return {
        "epe": mean_epe.item(),
        "ae_deg": mean_ae.item(),
        "f1": f1.item(),
    }


__all__ = ["evaluate_flow"]
