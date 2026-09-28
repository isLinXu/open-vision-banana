"""Normal angular loss for SANA training.

Adds a surface-normal angular error auxiliary loss for normal-task samples.
When the model predicts flow velocity for a normal task, we reconstruct x0,
decode it through the VAE to RGB, convert RGB → normal vectors, and compute
the angular error against the ground-truth normal vectors.

This mirrors the pattern established by ``sana_depth_aux.py`` for depth tasks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normal_task_mask(task_values: object, batch_size: int, device: torch.device) -> torch.Tensor:
    """Return a boolean mask for normal rows in a mixed manifest batch."""
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
    return torch.tensor([task == "normal" for task in tasks], device=device, dtype=torch.bool)


def normal_angular_loss(
    predicted_velocity: torch.Tensor,
    noisy_latents: torch.Tensor,
    target_latents: torch.Tensor,
    target_rgb: torch.Tensor,
    sigma: torch.Tensor,
    vae_decode_fn,
    *,
    sigma_floor: float = 0.10,
) -> torch.Tensor:
    """Compute angular error loss between predicted and GT surface normals.

    Reconstructs x0 from the flow-matching velocity prediction, decodes it
    through the VAE to RGB, converts to normal vectors, and computes the
    angular error against ground-truth normals.

    Flow formulation (SANA):
        x0 = noisy_latents - sigma * predicted_velocity

    Args:
        predicted_velocity: Model's velocity prediction [B, C, H, W].
        noisy_latents: Noisy input latents [B, C, H, W].
        target_latents: Target latents (unused, kept for API symmetry) [B, C, H, W].
        target_rgb: Ground-truth target RGB [B, 3, H_img, W_img] in [0, 1].
        sigma: Flow timesteps [B] in [0, 1].
        vae_decode_fn: Callable that decodes latents to RGB in [-1, 1].
            Signature: ``vae_decode_fn(latents) -> images_in_minus1_to_1``.
        sigma_floor: Minimum sigma for numerical stability.

    Returns:
        Scalar angular loss tensor (mean of (1 - cos_sim)).
    """
    if predicted_velocity.shape != noisy_latents.shape:
        raise ValueError(
            f"Shape mismatch: predicted_velocity {tuple(predicted_velocity.shape)} "
            f"vs noisy_latents {tuple(noisy_latents.shape)}"
        )
    if predicted_velocity.ndim != 4:
        raise ValueError("Expected 4D tensors [B, C, H, W]")
    if sigma_floor <= 0:
        raise ValueError("sigma_floor must be positive")

    # Reconstruct x0 from flow prediction: x0 = sample - sigma * velocity
    sigma_view = sigma.float().view(-1, 1, 1, 1)
    pred_x0 = noisy_latents.float() - sigma_view * predicted_velocity.float()

    # Decode predicted x0 to RGB via VAE (differentiable)
    pred_images = vae_decode_fn(pred_x0.to(dtype=predicted_velocity.dtype))
    # pred_images in [-1, 1] → [0, 1]
    pred_rgb = (pred_images.float() + 1.0) / 2.0

    # Resize pred_rgb to match target_rgb spatial dims if needed
    if pred_rgb.shape[-2:] != target_rgb.shape[-2:]:
        pred_rgb = F.interpolate(
            pred_rgb, size=target_rgb.shape[-2:], mode="bilinear", align_corners=False
        )

    # Convert RGB → normal vectors
    # R = (1 - x) / 2, G = (1 + y) / 2, B = (1 + z) / 2
    pred_normal = _rgb_to_normal(pred_rgb)
    gt_normal = _rgb_to_normal(target_rgb.float())

    # Angular error: (1 - cos_sim)
    cos_sim = (pred_normal * gt_normal).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    angular_error = (1.0 - cos_sim).mean()

    if not torch.isfinite(angular_error):
        raise FloatingPointError("Normal angular loss is non-finite")

    return angular_error


def _rgb_to_normal(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB image to surface normal vectors.

    Args:
        rgb: [B, 3, H, W] in [0, 1].

    Returns:
        Normal vectors [B, 3, H, W] normalized to unit length.
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    x = 1.0 - 2.0 * r
    y = 2.0 * g - 1.0
    z = 2.0 * b - 1.0
    normal = torch.cat([x, y, z], dim=1)
    # Normalize to unit vectors
    norm = normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return normal / norm


__all__ = ["normal_angular_loss", "normal_task_mask"]
