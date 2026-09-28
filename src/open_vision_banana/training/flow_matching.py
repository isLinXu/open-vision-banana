"""Unified flow-matching scheduler for multi-model training.

Abstracts the differences between SANA's native linear_flow (with shift) and
rectified-flow models (FLUX, SD3, Chroma, etc.) behind a single interface.

All families share the same core interpolation:
    z_t = (1 - t) * z_source + t * z_target
    v_target = z_target - z_source

SANA applies a flow_shift to the sampled sigma before interpolation.
"""

from __future__ import annotations

import torch

# Families that use rectified-flow parameterization (t=0 → source, t=1 → target)
_RECTIFIED_FLOW_FAMILIES = frozenset(
    {
        "flux",
        "flux2",
        "sd3",
        "chroma",
        "hidream-image",
    }
)

# Families that need 2x2 latent packing before transformer forward
_PACKING_FAMILIES = frozenset(
    {
        "flux",
        "flux2",
        "hidream-image",
    }
)


class FlowScheduler:
    """Unified flow matching scheduler for all model families."""

    @staticmethod
    def sample_timesteps(
        batch_size: int,
        device: torch.device,
        family: str,
        logit_normal: bool = True,
        flow_shift: float = 1.0,
    ) -> torch.Tensor:
        """Sample flow timesteps in [0, 1] for the given model family.

        For SANA (linear_flow with shift):
            sigma = shift * u / (1 + (shift-1)*u), where u ~ logit_normal or uniform
        For rectified-flow models:
            t = sigmoid(u) if logit_normal else uniform

        Returns:
            timesteps: (batch_size,) tensor in [0, 1]
        """
        if family == "sana":
            if logit_normal:
                u = torch.sigmoid(torch.randn(batch_size, device=device))
            else:
                u = torch.rand(batch_size, device=device)
            # Apply flow shift (SANA convention)
            sigma = flow_shift * u / (1.0 + (flow_shift - 1.0) * u)
            return sigma.float()

        # Rectified flow families (FLUX, SD3, Chroma, etc.)
        if logit_normal:
            u = torch.rand(batch_size, device=device)
            return torch.sigmoid(u)
        return torch.rand(batch_size, device=device)

    @staticmethod
    def create_noisy_latents(
        target_latents: torch.Tensor,
        source_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        family: str,
        flow_sigma: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create noisy latents and velocity target via flow interpolation.

        Convention: z_t = (1-t)*z_source + t*z_target, v = z_target - z_source

        For SANA (sigma parameterization):
            sample = sigma * source + (1-sigma) * target
            target = source - target
        For rectified-flow:
            z_t = (1-t) * source + t * target
            v = target - source

        When flow_sigma > 0, source is mixed with noise:
            source = (1-flow_sigma) * input_latents + flow_sigma * noise

        Returns:
            (noisy_latents, velocity_target)
        """
        t = timesteps
        t_expand = t.view(-1, *([1] * (target_latents.dim() - 1)))

        if family == "sana":
            # SANA: sample = sigma * source + (1-sigma) * target, v = source - target
            noisy = t_expand * source_latents + (1.0 - t_expand) * target_latents
            velocity = source_latents - target_latents
        else:
            # Rectified flow: z_t = (1-t)*source + t*target, v = target - source
            noisy = (1.0 - t_expand) * source_latents + t_expand * target_latents
            velocity = target_latents - source_latents

        return noisy, velocity

    @staticmethod
    def needs_latent_packing(family: str) -> bool:
        """Whether the model family requires 2x2 latent packing."""
        return family in _PACKING_FAMILIES

    @staticmethod
    def needs_guidance(family: str) -> bool:
        """Whether the model family needs guidance in forward pass."""
        return family in {"flux", "flux2", "qwen-image", "sana-sprint", "chroma", "hunyuan-image"}


# ── Latent packing utilities (FLUX, HiDream) ──────────────────────────────


def pack_latents(latents: torch.Tensor, batch_size: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, (H/2)*(W/2), C*4] for FLUX-style transformers."""
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(B, (H // 2) * (W // 2), C * 4)


def unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """[B, (H/2)*(W/2), C*4] -> [B, C, H, W]."""
    B, N, C = latents.shape
    latents = latents.view(B, height // 2, width // 2, C // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(B, C // 4, height, width)


def prepare_latent_image_ids(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create FLUX-style position IDs [H*W, 3]."""
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width)[None, :]
    return ids.reshape(height * width, 3).to(device=device, dtype=dtype)
