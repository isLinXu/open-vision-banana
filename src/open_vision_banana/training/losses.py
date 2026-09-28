"""Unified loss computation for multi-task training.

Handles flow matching MSE loss with task weighting, Min-SNR, and
auxiliary losses (depth edge endpoint + normal angular) that are
selectively applied based on task type.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from .normal_aux_loss import normal_angular_loss, normal_task_mask
from .sana_depth_aux import depth_edge_endpoint_loss, depth_task_mask
from .sana_flow import flow_min_snr_weight

logger = logging.getLogger(__name__)


class TrainingLosses:
    """Unified loss computation for multi-task training.

    Combines:
      - Flow matching MSE loss (with task weighting + optional Min-SNR)
      - Depth edge auxiliary loss (for depth-task samples)
      - Normal angular auxiliary loss (for normal-task samples)
    """

    def __init__(
        self,
        task_weights: dict[str, float] | None = None,
        depth_edge_aux_weight: float = 0.0,
        normal_angular_loss_weight: float = 0.0,
        depth_edge_aux_gradient_weight: float = 0.25,
        depth_edge_aux_edge_weight: float = 2.0,
        depth_edge_aux_edge_threshold: float = 0.025,
        depth_edge_aux_sigma_floor: float = 0.10,
        min_snr: bool = True,
        min_snr_gamma: float = 5.0,
        family: str = "sana",
    ):
        self.task_weights = task_weights or {"depth": 1.0, "normal": 1.0, "semantic": 1.0}
        self.depth_edge_aux_weight = depth_edge_aux_weight
        self.normal_angular_loss_weight = normal_angular_loss_weight
        self.depth_edge_aux_gradient_weight = depth_edge_aux_gradient_weight
        self.depth_edge_aux_edge_weight = depth_edge_aux_edge_weight
        self.depth_edge_aux_edge_threshold = depth_edge_aux_edge_threshold
        self.depth_edge_aux_sigma_floor = depth_edge_aux_sigma_floor
        self.min_snr = min_snr
        self.min_snr_gamma = min_snr_gamma
        self.family = family

    def compute_flow_loss(
        self,
        predicted_velocity: torch.Tensor,
        velocity_target: torch.Tensor,
        timesteps: torch.Tensor,
        batch_tasks: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[float]]]:
        """Compute MSE flow loss with task weighting and optional Min-SNR.

        Args:
            predicted_velocity: Model prediction [B, C, H, W].
            velocity_target: Flow target [B, C, H, W].
            timesteps: Flow timesteps [B].
            batch_tasks: Task label per sample.
            device: Device for weight tensors.

        Returns:
            (flow_loss, per_sample_loss, task_loss_acc)
        """
        sq_error = (predicted_velocity.float() - velocity_target.float()).square()
        per_sample_loss = sq_error.flatten(1).mean(dim=1)

        # Task loss weighting
        weights = torch.tensor(
            [self.task_weights.get(t, 1.0) for t in batch_tasks],
            device=device,
            dtype=torch.float32,
        )
        per_sample_loss = per_sample_loss * weights

        # Min-SNR weighting
        if self.min_snr:
            snw = flow_min_snr_weight(timesteps, self.min_snr_gamma)
            flow_loss = (per_sample_loss * snw).mean()
        else:
            flow_loss = per_sample_loss.mean()

        # Per-task tracking
        task_loss_acc: dict[str, list[float]] = {}
        for t, loss_val in zip(
            batch_tasks, per_sample_loss.detach().float().tolist(), strict=False
        ):
            task_loss_acc.setdefault(t, []).append(float(loss_val))

        return flow_loss, per_sample_loss, task_loss_acc

    def compute_aux_losses(
        self,
        predicted_velocity: torch.Tensor,
        noisy_latents: torch.Tensor,
        target_latents: torch.Tensor,
        target_rgb: torch.Tensor,
        timesteps: torch.Tensor,
        batch_tasks: list[str],
        vae_decode_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute depth edge + normal angular auxiliary losses.

        Args:
            predicted_velocity: Model prediction [B, C, H, W].
            noisy_latents: Noisy input latents [B, C, H, W].
            target_latents: Target latents [B, C, H, W].
            target_rgb: Ground-truth RGB [B, 3, H, W] in [0, 1].
            timesteps: Flow timesteps [B].
            batch_tasks: Task label per sample.
            vae_decode_fn: Callable that decodes latents to images in [-1, 1].
            device: Device.

        Returns:
            (depth_aux_loss, normal_aux_loss)
        """
        batch_size = predicted_velocity.shape[0]

        # Depth edge auxiliary loss
        depth_aux_loss = predicted_velocity.new_zeros(())
        if self.depth_edge_aux_weight > 0:
            depth_mask = depth_task_mask(batch_tasks, batch_size, device)
            if depth_mask.any():
                depth_aux_loss = depth_edge_endpoint_loss(
                    predicted_velocity[depth_mask],
                    noisy_latents[depth_mask],
                    target_latents[depth_mask],
                    target_rgb[depth_mask],
                    timesteps[depth_mask],
                    edge_threshold=self.depth_edge_aux_edge_threshold,
                    edge_weight=self.depth_edge_aux_edge_weight,
                    gradient_weight=self.depth_edge_aux_gradient_weight,
                    sigma_floor=self.depth_edge_aux_sigma_floor,
                )

        # Normal angular auxiliary loss
        normal_aux_loss = predicted_velocity.new_zeros(())
        if self.normal_angular_loss_weight > 0:
            normal_mask = normal_task_mask(batch_tasks, batch_size, device)
            if normal_mask.any():
                normal_aux_loss = normal_angular_loss(
                    predicted_velocity[normal_mask],
                    noisy_latents[normal_mask],
                    target_latents[normal_mask],
                    target_rgb[normal_mask],
                    timesteps[normal_mask],
                    vae_decode_fn,
                )

        return depth_aux_loss, normal_aux_loss

    def total_loss(
        self,
        flow_loss: torch.Tensor,
        depth_aux_loss: torch.Tensor,
        normal_aux_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted sum of all loss components."""
        total = flow_loss.float()
        if self.depth_edge_aux_weight > 0:
            total = total + self.depth_edge_aux_weight * depth_aux_loss
        if self.normal_angular_loss_weight > 0:
            total = total + self.normal_angular_loss_weight * normal_aux_loss
        return total
