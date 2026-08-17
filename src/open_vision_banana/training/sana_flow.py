"""Native SANA flow-matching helpers.

SANA's ``DPMSolverMultistepScheduler`` uses shifted flow sigmas and native
timesteps in ``[0, num_train_timesteps]``.  Keeping those conventions in one
module prevents a training/inference coordinate mismatch.
"""

from __future__ import annotations

import torch


def _scheduler_value(scheduler, name: str, default: float) -> float:
    config = getattr(scheduler, "config", None)
    return float(getattr(config, name, default))


def vae_latent_scaling_factor(vae: torch.nn.Module) -> float:
    """Return the diffusion-latent scaling required by the loaded VAE."""
    config = getattr(vae, "config", None)
    return float(getattr(config, "scaling_factor", 1.0) or 1.0)


def vae_latent_shift_factor(vae: torch.nn.Module) -> float:
    """Return an optional VAE latent shift (SANA's DC-AE has no shift)."""
    config = getattr(vae, "config", None)
    return float(getattr(config, "shift_factor", 0.0) or 0.0)


def normalize_vae_latents(latents: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """Convert DC-AE encoder output into the transformer's latent space."""
    return (latents - vae_latent_shift_factor(vae)) * vae_latent_scaling_factor(vae)


def denormalize_vae_latents(latents: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """Convert transformer latents back into the VAE decoder's latent space."""
    return latents / vae_latent_scaling_factor(vae) + vae_latent_shift_factor(vae)


def shifted_flow_sigma(base_sigma: torch.Tensor, scheduler) -> torch.Tensor:
    """Apply the flow shift used by SANA's native scheduler.

    ``base_sigma`` is in [0, 1].  At 0 the sample is the clean target; at 1
    it is the source/noise endpoint.  The scheduler applies the same mapping
    when it creates its inference sigmas.
    """
    shift = _scheduler_value(scheduler, "flow_shift", 1.0)
    return shift * base_sigma / (1.0 + (shift - 1.0) * base_sigma)


def sample_sana_flow_timesteps(
    batch_size: int,
    device: torch.device,
    scheduler,
    *,
    logit_normal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample shifted sigmas and matching native SANA transformer timesteps."""
    if logit_normal:
        base_sigma = torch.sigmoid(torch.randn(batch_size, device=device))
    else:
        base_sigma = torch.rand(batch_size, device=device)

    sigma = shifted_flow_sigma(base_sigma, scheduler)
    timestep_count = _scheduler_value(scheduler, "num_train_timesteps", 1000.0)
    return sigma.float(), (sigma * timestep_count).float()


def prepare_sana_flow_batch(
    source_latents: torch.Tensor,
    target_latents: torch.Tensor,
    sigma: torch.Tensor,
    *,
    noise_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a native SANA flow sample and its ``flow_prediction`` target.

    The scheduler interprets a prediction as ``x0 = sample - sigma * v``.
    Therefore the path from source to target must be
    ``sample = sigma * source + (1 - sigma) * target`` and the model target
    must be ``source - target``.
    """
    if source_latents.shape != target_latents.shape:
        raise ValueError(
            "SANA source and target latent shapes must match: "
            f"{tuple(source_latents.shape)} != {tuple(target_latents.shape)}"
        )
    if sigma.ndim != 1 or sigma.shape[0] != source_latents.shape[0]:
        raise ValueError("sigma must contain one value per latent batch item")

    sigma_b = sigma.to(dtype=source_latents.dtype).view(
        source_latents.shape[0], *([1] * (source_latents.ndim - 1))
    )
    sample = sigma_b * source_latents + (1.0 - sigma_b) * target_latents
    if noise_scale > 0.0:
        noise_strength = noise_scale * (sigma_b * (1.0 - sigma_b)).sqrt()
        sample = sample + noise_strength * torch.randn_like(sample)
    return sample, source_latents - target_latents


def native_sana_timestep(timestep: torch.Tensor, batch_size: int, transformer) -> torch.Tensor:
    """Expand a scheduler timestep exactly as ``SanaPipeline`` does."""
    if not hasattr(transformer, "config") and hasattr(transformer, "base_model"):
        transformer = getattr(transformer.base_model, "model", transformer.base_model)
    scale = float(getattr(getattr(transformer, "config", None), "timestep_scale", 1.0))
    return timestep.expand(batch_size).float() * scale


def flow_min_snr_weight(sigma: torch.Tensor, gamma: float = 5.0) -> torch.Tensor:
    """Optional Min-SNR weighting in the scheduler's source-to-target frame."""
    sigma = sigma.float().clamp(1e-4, 1.0 - 1e-4)
    snr = ((1.0 - sigma) / sigma).square()
    return torch.clamp(snr, max=gamma) / (snr + 1.0)
