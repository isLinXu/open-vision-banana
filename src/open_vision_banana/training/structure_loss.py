"""Differentiable spatial losses for decoded diffusion x0 predictions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.depth_mapper import DepthRGBMapper


def _charbonnier(error: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Smooth L1-like penalty with an exact zero at zero error."""
    return (torch.sqrt(error.square() + epsilon**2) - epsilon).mean()


def decoded_x0_structure_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_weight: float = 1.0,
    scales: tuple[int, ...] = (1, 2, 4),
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Match decoded RGB values and spatial gradients at multiple scales.

    Both tensors use the same image range (normally ``[-1, 1]``). The loss is
    deliberately colormap-agnostic and fully differentiable through the VAE.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected prediction and target shaped [B, 3, H, W]")
    if gradient_weight < 0:
        raise ValueError("gradient_weight must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("scales must contain positive integers")

    total = prediction.new_zeros(())
    total_weight = 0.0
    for index, scale in enumerate(scales):
        if prediction.shape[-2] < scale or prediction.shape[-1] < scale:
            continue
        pred_level = prediction
        target_level = target
        if scale > 1:
            pred_level = F.avg_pool2d(pred_level, kernel_size=scale, stride=scale)
            target_level = F.avg_pool2d(target_level, kernel_size=scale, stride=scale)

        level_weight = 1.0 / (2**index)
        value_loss = _charbonnier(pred_level - target_level, epsilon)
        gradient_loss = prediction.new_zeros(())
        if pred_level.shape[-1] > 1:
            pred_dx = pred_level[..., :, 1:] - pred_level[..., :, :-1]
            target_dx = target_level[..., :, 1:] - target_level[..., :, :-1]
            gradient_loss = gradient_loss + _charbonnier(pred_dx - target_dx, epsilon)
        if pred_level.shape[-2] > 1:
            pred_dy = pred_level[..., 1:, :] - pred_level[..., :-1, :]
            target_dy = target_level[..., 1:, :] - target_level[..., :-1, :]
            gradient_loss = gradient_loss + _charbonnier(pred_dy - target_dy, epsilon)
        total = total + level_weight * (value_loss + gradient_weight * gradient_loss)
        total_weight += level_weight

    if total_weight == 0:
        raise ValueError("No requested scale fits the image dimensions")
    result = total / total_weight
    if not torch.isfinite(result):
        raise FloatingPointError("decoded x0 structure loss is non-finite")
    return result


def soft_colormap_position(
    rgb: torch.Tensor,
    *,
    colormap: str,
    temperature: float = 1e-3,
) -> torch.Tensor:
    """Softly project RGB pixels onto a registered depth-colormap curve.

    Unlike the exact evaluation inverse, this function avoids a hard nearest
    segment selection. Each pixel is analytically projected onto every curve
    segment, then the segment positions are averaged with distance-based
    softmax weights. The result is differentiable almost everywhere with
    respect to RGB, including across segment boundaries.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Expected RGB shaped [B, 3, H, W]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if colormap == "unknown":
        raise ValueError("colormap must identify a registered depth curve")

    keypoints = DepthRGBMapper(colormap).colormap_keypoints.to(
        device=rgb.device,
        dtype=torch.float32,
    )
    positions = keypoints[:, 0]
    colors = keypoints[:, 1:]
    segment_start = colors[:-1]
    segment_vector = colors[1:] - segment_start
    segment_length_sq = segment_vector.square().sum(-1).clamp_min(1e-10)

    batch, _, height, width = rgb.shape
    pixels = rgb.float().permute(0, 2, 3, 1).reshape(batch, height * width, 3)
    difference = pixels.unsqueeze(2) - segment_start.view(1, 1, -1, 3)
    local_position = (
        (difference * segment_vector.view(1, 1, -1, 3)).sum(-1)
        / segment_length_sq.view(1, 1, -1)
    ).clamp(0.0, 1.0)
    closest_rgb = segment_start.view(1, 1, -1, 3) + (
        local_position.unsqueeze(-1) * segment_vector.view(1, 1, -1, 3)
    )
    distance_sq = (pixels.unsqueeze(2) - closest_rgb).square().sum(-1)
    weights = torch.softmax(-distance_sq / temperature, dim=-1)
    segment_positions = positions[:-1].view(1, 1, -1) + local_position * (
        positions[1:] - positions[:-1]
    ).view(1, 1, -1)
    projected = (weights * segment_positions).sum(-1)
    return projected.reshape(batch, 1, height, width).clamp(0.0, 1.0)


def _scalar_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    loss = prediction.new_zeros(())
    if prediction.shape[-1] > 1:
        pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
        target_dx = target[..., :, 1:] - target[..., :, :-1]
        loss = loss + _charbonnier(pred_dx - target_dx, epsilon)
    if prediction.shape[-2] > 1:
        pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
        target_dy = target[..., 1:, :] - target[..., :-1, :]
        loss = loss + _charbonnier(pred_dy - target_dy, epsilon)
    return loss


def _deterministic_ordinal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Penalize wrong long-range ordering without consuming training RNG."""
    height, width = prediction.shape[-2:]
    candidates = (
        (0, width // 2),
        (height // 2, 0),
        (height // 2, width // 2),
        (height // 4, width // 4),
    )
    offsets = {
        (dy, dx)
        for dy, dx in candidates
        if (dy > 0 or dx > 0) and dy < height and dx < width
    }
    numerator = prediction.new_zeros(())
    denominator = prediction.new_zeros(())
    for dy, dx in sorted(offsets):
        end_y = height - dy if dy else height
        end_x = width - dx if dx else width
        pred_a = prediction[..., :end_y, :end_x]
        pred_b = prediction[..., dy:, dx:]
        target_a = target[..., :end_y, :end_x]
        target_b = target[..., dy:, dx:]
        target_delta = target_b - target_a
        valid = (target_delta.abs() >= margin).to(dtype=prediction.dtype)
        direction = target_delta.sign()
        predicted_delta = pred_b - pred_a
        penalty = F.softplus(-direction * predicted_delta / temperature)
        numerator = numerator + (penalty * valid).sum()
        denominator = denominator + valid.sum()
    return numerator / denominator.clamp_min(1.0)


def decoded_x0_scalar_geometry_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    colormap: str,
    gradient_weight: float = 1.0,
    ordinal_weight: float = 1.0,
    projection_temperature: float = 1e-3,
    ordinal_temperature: float = 5e-2,
    ordinal_margin: float = 2e-2,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Directly supervise scalar depth position, gradients, and ordering.

    ``prediction`` and ``target`` are decoded RGB tensors in ``[-1, 1]``.
    They are mapped to positions along the checkpoint's audited depth
    colormap before applying geometry-aligned losses.
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected prediction and target shaped [B, 3, H, W]")
    if gradient_weight < 0 or ordinal_weight < 0:
        raise ValueError("geometry component weights must be non-negative")
    if projection_temperature <= 0:
        raise ValueError("projection_temperature must be positive")
    if ordinal_temperature <= 0:
        raise ValueError("ordinal_temperature must be positive")
    if ordinal_margin < 0:
        raise ValueError("ordinal_margin must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    prediction_rgb = (prediction.float() + 1.0) / 2.0
    target_rgb = (target.float() + 1.0) / 2.0
    prediction_position = soft_colormap_position(
        prediction_rgb,
        colormap=colormap,
        temperature=projection_temperature,
    )
    target_position = soft_colormap_position(
        target_rgb,
        colormap=colormap,
        temperature=projection_temperature,
    )
    value_loss = _charbonnier(prediction_position - target_position, epsilon)
    gradient_loss = _scalar_gradient_loss(
        prediction_position,
        target_position,
        epsilon=epsilon,
    )
    ordinal_loss = _deterministic_ordinal_loss(
        prediction_position,
        target_position,
        margin=ordinal_margin,
        temperature=ordinal_temperature,
    )
    result = value_loss + gradient_weight * gradient_loss + ordinal_weight * ordinal_loss
    if not torch.isfinite(result):
        raise FloatingPointError("decoded x0 scalar geometry loss is non-finite")
    return result


__all__ = [
    "decoded_x0_scalar_geometry_loss",
    "decoded_x0_structure_loss",
    "soft_colormap_position",
]
