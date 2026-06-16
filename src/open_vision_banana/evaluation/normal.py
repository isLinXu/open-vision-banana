"""Surface normal evaluation metrics.

Standard metrics from the normal estimation literature:
- Mean/Median Angular Error (degrees)
- Mean Cosine Similarity
- 11.25°, 22.5°, 30° accuracy thresholds
"""

from __future__ import annotations

import math

import torch


def evaluate_normals(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Compute surface normal evaluation metrics.

    Args:
        pred: Predicted normals [B, 3, H, W] (unit vectors).
        gt: Ground truth normals [B, 3, H, W] (unit vectors).

    Returns:
        Dictionary with angular error and cosine similarity metrics.
    """
    pred_norm = pred / pred.norm(dim=1, keepdim=True).clamp(min=1e-8)
    gt_norm = gt / gt.norm(dim=1, keepdim=True).clamp(min=1e-8)

    cos_sim = (pred_norm * gt_norm).sum(dim=1)
    cos_sim = cos_sim.clamp(-1.0, 1.0)

    angle_rad = torch.acos(cos_sim)
    angle_deg = angle_rad * (180.0 / math.pi)

    return {
        "Mean Angular Error (°)": angle_deg.mean().item(),
        "Median Angular Error (°)": angle_deg.median().item(),
        "Mean Cosine Similarity": cos_sim.mean().item(),
        "11.25° Accuracy": (angle_deg < 11.25).float().mean().item(),
        "22.5° Accuracy": (angle_deg < 22.5).float().mean().item(),
        "30° Accuracy": (angle_deg < 30.0).float().mean().item(),
    }


__all__ = ["evaluate_normals"]
