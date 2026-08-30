"""Depth estimation evaluation metrics.

Standard metrics from the depth estimation literature:
- AbsRel: Absolute Relative Error
- SqRel: Squared Relative Error
- RMSE: Root Mean Squared Error
- RMSE_log: Root Mean Squared Log Error
- δ<1.25, δ<1.25², δ<1.25³: Threshold accuracy

Quantization-aware evaluation (P3 from v0.5.0 deep analysis):
- Separates model prediction error from RGB encoding noise
- Provides encoding_noise_baseline() for dynamic calibration
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


# Skip median-scale when the prediction is degenerate. A 1e-6 floor
# turns near-zero RGB decodes into 10^6–10^7 scale factors and explodes
# AbsRel_aligned / RMSE_aligned. Those images stay in the raw paper metrics.
ALIGNED_MIN_PRED_MEDIAN = 1e-3
ALIGNED_MAX_SCALE = 50.0


def median_scale_align(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale each image so ``median(pred)`` matches ``median(gt)``.

    Returns ``(aligned, stable)`` where ``stable`` is a per-image bool mask.
    Images with a near-zero prediction median, or a scale outside
    ``[1/50, 50]``, are left unchanged and marked unstable.

    This is a diagnostic for relative-to-metric transfer. Paper-protocol
    metric depth must still be reported on the unaligned prediction.
    """
    pred = pred.float()
    gt = gt.float()
    squeezed = False
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)
        squeezed = True
    if valid_mask is None:
        mask = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0) & (pred > 0)
    else:
        mask = valid_mask.to(device=pred.device).bool()
        if mask.dim() == pred.dim() + 1 and mask.shape[1] == 1:
            mask = mask.squeeze(1)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).expand_as(pred)
        mask = mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0) & (pred > 0)

    aligned = pred.clone()
    stable = torch.zeros(pred.shape[0], dtype=torch.bool, device=pred.device)
    for index in range(pred.shape[0]):
        valid = mask[index]
        if not bool(valid.any()):
            continue
        pred_median = pred[index][valid].median()
        if float(pred_median) < ALIGNED_MIN_PRED_MEDIAN:
            continue
        gt_median = gt[index][valid].median()
        scale = float(gt_median / pred_median)
        if scale <= 0.0 or scale > ALIGNED_MAX_SCALE or scale < 1.0 / ALIGNED_MAX_SCALE:
            continue
        aligned[index] = pred[index] * scale
        stable[index] = True
    if squeezed:
        return aligned.squeeze(0), stable.squeeze(0)
    return aligned, stable


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


def encoding_noise_baseline(
    gt: torch.Tensor,
    colormap: str = "cube_edge",
    num_samples: int = 64,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Estimate the encoding noise floor from 8-bit RGB quantization.

    Simulates the depth→RGB→depth roundtrip using the specified colormap
    and quantizes to uint8 (round(rgb×255)/255). Returns the AbsRel
    introduced purely by the encoding layer — this is a lower bound on
    any end-to-end metric that uses RGB parameterization.

    Args:
        gt: Ground truth depth [B, 1, H, W] in meters.
        colormap: Colormap name from DepthRGBMapper registry.
        num_samples: Number of pixels to sample for speed (full tensor
            can be expensive at high resolution).
        valid_mask: Optional boolean mask for valid pixels.

    Returns:
        Estimated AbsRel (%) from quantization alone.
    """
    from ..core.depth_mapper import DepthRGBMapper

    mapper = DepthRGBMapper(colormap=colormap)
    pred = gt.squeeze(1)
    if valid_mask is not None:
        pred = pred[valid_mask]
    nonzero = pred > 0
    pred = pred[nonzero]
    if pred.numel() == 0:
        return 0.0

    # Sample for efficiency if tensor is large
    if pred.numel() > num_samples:
        idx = torch.randperm(pred.numel())[:num_samples]
        pred = pred.flatten()[idx]

    # Add batch dimension [1, 1, N]
    depth = pred.view(1, 1, -1)
    rgb = mapper.depth_to_rgb(depth)
    # Simulate uint8 quantization
    rgb_q = (rgb * 255.0).round() / 255.0
    recovered = mapper.rgb_to_depth(rgb_q).squeeze()
    return (recovered - pred).abs().div(pred).mean().item()


def evaluate_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    colormap: str | None = None,
) -> dict[str, float]:
    """Compute all standard depth evaluation metrics.

    When ``colormap`` is provided, also estimates the encoding noise
    baseline (AbsRel from 8-bit quantization) and reports
    ``AbsRel_model`` = ``AbsRel`` − ``AbsRel_encoding``.

    Args:
        pred: Predicted depth [B, 1, H, W] in meters.
        gt: Ground truth depth [B, 1, H, W] in meters.
        valid_mask: Optional boolean mask for valid pixels.
        colormap: Optional colormap name to estimate quantization noise.

    Returns:
        Dictionary of metric name → value. If colormap is provided,
        includes ``AbsRel_encoding`` and ``AbsRel_model``.
    """
    pred = pred.squeeze(1)
    gt = gt.squeeze(1)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=pred.device)
        if valid_mask.dim() == pred.dim() + 1 and valid_mask.shape[1] == 1:
            valid_mask = valid_mask.squeeze(1)

    result = {
        "AbsRel": depth_abs_rel(pred, gt, valid_mask),
        "SqRel": depth_sq_rel(pred, gt, valid_mask),
        "RMSE": depth_rmse(pred, gt, valid_mask),
        "RMSE_log": depth_rmse_log(pred, gt, valid_mask),
        "δ<1.25": depth_delta(pred, gt, 1.25, valid_mask),
        "δ<1.25²": depth_delta(pred, gt, 1.25 ** 2, valid_mask),
        "δ<1.25³": depth_delta(pred, gt, 1.25 ** 3, valid_mask),
    }

    if colormap is not None:
        enc = encoding_noise_baseline(gt, colormap=colormap, valid_mask=valid_mask)
        result["AbsRel_encoding"] = enc
        result["AbsRel_model"] = max(0.0, result["AbsRel"] - enc)

    return result


__all__ = ["evaluate_depth", "encoding_noise_baseline", "median_scale_align"]
