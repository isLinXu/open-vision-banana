"""Geometry-aware auxiliary objectives for SANA depth flow matching.

The SANA DC-AE represents a 512px output on a 16x16 latent grid.  Decoding an
intermediate prediction to RGB and backpropagating through the frozen decoder
would make a batch-one MPS run substantially more expensive.  This module
instead derives a target-depth edge map from the already available RGB target,
projects it onto the latent grid, and uses it to weight the flow residual and
its local spatial gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core.depth_mapper import DepthRGBMapper


def _validate_flow_tensors(
    predicted_velocity: torch.Tensor,
    noisy_latents: torch.Tensor,
    target_latents: torch.Tensor,
    target_rgb: torch.Tensor,
    sigma: torch.Tensor,
) -> None:
    if predicted_velocity.shape != noisy_latents.shape or predicted_velocity.shape != target_latents.shape:
        raise ValueError("predicted_velocity, noisy_latents, and target_latents must share a shape")
    if predicted_velocity.ndim != 4:
        raise ValueError("latent tensors must be shaped [B, C, H, W]")
    if target_rgb.ndim != 4 or target_rgb.shape[1] != 3:
        raise ValueError("target_rgb must be shaped [B, 3, H, W]")
    if target_rgb.shape[0] != predicted_velocity.shape[0]:
        raise ValueError("target_rgb and latent tensors must share a batch size")
    if sigma.ndim != 1 or sigma.shape[0] != predicted_velocity.shape[0]:
        raise ValueError("sigma must contain one value per latent batch item")


def depth_task_mask(task_values: object, batch_size: int, device: torch.device) -> torch.Tensor:
    """Return a boolean mask for depth rows in a mixed manifest batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if isinstance(task_values, str):
        tasks = [task_values]
    elif task_values is None:
        tasks = [""] * batch_size
    else:
        tasks = [str(task) for task in task_values]
    if len(tasks) != batch_size:
        raise ValueError(
            f"task labels must contain one entry per sample: {len(tasks)} != {batch_size}"
        )
    return torch.tensor([task == "depth" for task in tasks], device=device, dtype=torch.bool)


def depth_edge_weights(
    target_rgb: torch.Tensor,
    latent_size: tuple[int, int],
    *,
    colormap: str = "turbo",
    edge_threshold: float = 0.025,
    edge_weight: float = 2.0,
) -> torch.Tensor:
    """Build a positive per-cell weight map from decoded target-depth edges.

    ``edge_weight`` is an additive multiplier: zero returns unit weights, and
    a fully edge-covered latent cell has weight ``1 + edge_weight``.  The
    target is deliberately detached because it is annotation data, not a model
    output; the returned map therefore only controls gradients to the model.
    """
    if target_rgb.ndim != 4 or target_rgb.shape[1] != 3:
        raise ValueError("target_rgb must be shaped [B, 3, H, W]")
    if len(latent_size) != 2 or min(latent_size) < 1:
        raise ValueError("latent_size must contain two positive dimensions")
    if edge_threshold <= 0:
        raise ValueError("edge_threshold must be positive")
    if edge_weight < 0:
        raise ValueError("edge_weight must be non-negative")

    with torch.no_grad():
        normalized_depth = DepthRGBMapper(colormap=colormap).rgb_to_normalized(
            target_rgb.detach().float().clamp(0.0, 1.0)
        )
        gradient_x = F.pad(
            (normalized_depth[..., :, 1:] - normalized_depth[..., :, :-1]).abs(),
            (0, 1, 0, 0),
        )
        gradient_y = F.pad(
            (normalized_depth[..., 1:, :] - normalized_depth[..., :-1, :]).abs(),
            (0, 0, 0, 1),
        )
        edge_strength = (torch.maximum(gradient_x, gradient_y) / edge_threshold).clamp(0.0, 1.0)
        coverage = F.adaptive_avg_pool2d(edge_strength, latent_size)
    return 1.0 + edge_weight * coverage


def _weighted_mean(error: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    expanded_weights = weights.expand(-1, error.shape[1], -1, -1)
    return (error.square() * expanded_weights).sum() / expanded_weights.sum().clamp_min(1.0)


def depth_edge_endpoint_loss(
    predicted_velocity: torch.Tensor,
    noisy_latents: torch.Tensor,
    target_latents: torch.Tensor,
    target_rgb: torch.Tensor,
    sigma: torch.Tensor,
    *,
    colormap: str = "turbo",
    edge_threshold: float = 0.025,
    edge_weight: float = 2.0,
    gradient_weight: float = 0.25,
    sigma_floor: float = 0.10,
) -> torch.Tensor:
    """Penalize target-endpoint error and gradients near decoded depth edges.

    SANA flow prediction uses ``x0 = sample - sigma * velocity``.  Dividing
    the reconstructed endpoint residual by ``max(sigma, sigma_floor)`` keeps
    its magnitude comparable across timesteps while avoiding amplification near
    the clean endpoint.  The value term targets high-depth-edge cells; the
    gradient term aligns local latent transitions at those same boundaries.
    """
    _validate_flow_tensors(
        predicted_velocity, noisy_latents, target_latents, target_rgb, sigma
    )
    if gradient_weight < 0:
        raise ValueError("gradient_weight must be non-negative")
    if sigma_floor <= 0:
        raise ValueError("sigma_floor must be positive")

    sigma_view = sigma.float().view(-1, 1, 1, 1)
    endpoint = noisy_latents.float() - sigma_view * predicted_velocity.float()
    residual = (endpoint - target_latents.float()) / sigma_view.clamp_min(sigma_floor)
    weights = depth_edge_weights(
        target_rgb,
        tuple(target_latents.shape[-2:]),
        colormap=colormap,
        edge_threshold=edge_threshold,
        edge_weight=edge_weight,
    ).to(device=residual.device, dtype=residual.dtype)

    value_loss = _weighted_mean(residual, weights)
    gradient_loss = residual.new_zeros(())
    if residual.shape[-1] > 1:
        gradient_loss = gradient_loss + _weighted_mean(
            residual[..., :, 1:] - residual[..., :, :-1],
            (weights[..., :, 1:] + weights[..., :, :-1]) * 0.5,
        )
    if residual.shape[-2] > 1:
        gradient_loss = gradient_loss + _weighted_mean(
            residual[..., 1:, :] - residual[..., :-1, :],
            (weights[..., 1:, :] + weights[..., :-1, :]) * 0.5,
        )
    result = value_loss + gradient_weight * gradient_loss
    if not torch.isfinite(result):
        raise FloatingPointError("SANA depth edge endpoint loss is non-finite")
    return result


__all__ = ["depth_edge_endpoint_loss", "depth_edge_weights", "depth_task_mask"]
