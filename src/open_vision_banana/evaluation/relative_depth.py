"""Relative-depth metrics for RGB labels without recoverable metric scale."""

from __future__ import annotations

import torch

from ..core.depth_mapper import DepthRGBMapper


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return one-based average ranks, assigning tied values their midrank."""
    values = values.flatten().to(dtype=torch.float64)
    order = torch.argsort(values)
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    midranks = (starts + ends - 1).to(dtype=torch.float64) / 2.0 + 1.0
    sorted_ranks = torch.repeat_interleave(midranks, counts)
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def evaluate_relative_depth_rgb(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    colormap: str,
) -> dict[str, float]:
    """Compare RGB outputs and their positions along a declared depth curve."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected prediction and target shaped [B, 3, H, W]")
    prediction = prediction.float().clamp(0, 1)
    target = target.float().clamp(0, 1)
    mapper = DepthRGBMapper(colormap=colormap)
    pred_position = mapper.rgb_to_normalized(prediction)
    target_position = mapper.rgb_to_normalized(target)
    pred_projected = mapper.normalized_to_rgb(pred_position)
    target_projected = mapper.normalized_to_rgb(target_position)

    position_error = pred_position - target_position
    pred_flat = pred_position.flatten()
    target_flat = target_position.flatten()
    pearson = 0.0
    if pred_flat.std() > 1e-8 and target_flat.std() > 1e-8:
        pearson = float(torch.corrcoef(torch.stack((pred_flat, target_flat)))[0, 1])
    pred_ranks = _average_ranks(pred_flat)
    target_ranks = _average_ranks(target_flat)
    spearman = 0.0
    if pred_ranks.std() > 1e-8 and target_ranks.std() > 1e-8:
        spearman = float(
            torch.corrcoef(torch.stack((pred_ranks, target_ranks)))[0, 1]
        )
    return {
        "rgb_mae": float((prediction - target).abs().mean()),
        "rgb_rmse": float((prediction - target).square().mean().sqrt()),
        "relative_position_mae": float(position_error.abs().mean()),
        "relative_position_rmse": float(position_error.square().mean().sqrt()),
        "relative_position_pearson_r": pearson,
        "relative_position_spearman_r": spearman,
        "prediction_curve_mae": float((prediction - pred_projected).abs().mean()),
        "target_curve_mae": float((target - target_projected).abs().mean()),
    }


__all__ = ["evaluate_relative_depth_rgb"]
