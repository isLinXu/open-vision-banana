"""Flow-GRPO pieces with *decoded task metrics* as the reward.

This is not the Qwen-Image-2.0-RL aesthetic RLHF suite. Those rewards would
destroy Vision Banana RGB contracts. The only implemented reward is
``decoded_metric`` against ground-truth RGB (depth δ1.25, mask IoU, colour MAE).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .opd import drift_update

_DEPTH_EPS = 1e-3


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO group-relative advantage: (r - mean) / std within the group."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO needs a 1-D reward group of size >= 2")
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp(min=1e-6)
    return (rewards - mean) / std


def gaussian_transition_log_prob(
    x_next: torch.Tensor,
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
    sigma: float,
    family: str,
) -> torch.Tensor:
    """Per-sample log N(x_next | Euler mean, (sigma √dt)² I), dropping the constant."""
    if sigma <= 0:
        raise ValueError("GRPO SDE sigma must be positive so the transition density exists")
    mean = drift_update(x_t, velocity, dt, family)
    scale = sigma * (dt ** 0.5)
    residual = (x_next - mean).float() / scale
    return -0.5 * residual.flatten(1).square().sum(dim=1)


def clipped_surrogate(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """Mean of the PPO/GRPO clipped surrogate (minimized: the negative of the paper)."""
    if clip_eps <= 0:
        raise ValueError("grpo_clip_eps must be positive")
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.minimum(unclipped, clipped).mean()


def sde_step(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
    sigma: float,
    family: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler–Maruyama step. Returns ``(x_next, eps)`` where eps ~ N(0, I)."""
    mean = drift_update(latents, velocity, dt, family)
    eps = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    scale = sigma * (dt ** 0.5)
    return mean + scale * eps, eps


@torch.no_grad()
def rollout_sde(
    predict_fn,
    noise: torch.Tensor,
    steps: int,
    family: str,
    sigma: float,
    text_cond: dict[str, Any],
    context_latents: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Stochastic rollout used as the old policy. Stores transitions for the ratio."""
    if steps < 1:
        raise ValueError("rollout_steps must be positive")
    dt = 1.0 / steps
    batch = noise.shape[0]
    device, dtype = noise.device, noise.dtype
    latents = noise
    time = torch.ones(batch, device=device, dtype=dtype) if family == "sana" else torch.zeros(
        batch, device=device, dtype=dtype
    )
    sign = -1.0 if family == "sana" else 1.0
    states = [latents]
    times = [time]
    next_states = []
    for _ in range(steps):
        velocity = predict_fn(latents, time, text_cond, batch, context_latents=context_latents)
        latents, _eps = sde_step(latents, velocity, dt, sigma, family, generator=generator)
        time = time + sign * dt
        next_states.append(latents)
        states.append(latents)
        times.append(time)
    return {
        "states": states,  # length steps+1
        "next_states": next_states,  # length steps
        "times": times,
        "dt": dt,
        "final": latents,
    }


def _per_sample_delta1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(gt) & (gt > _DEPTH_EPS)
    ratio = torch.maximum(pred.clamp(min=_DEPTH_EPS) / gt.clamp(min=_DEPTH_EPS), gt / pred.clamp(min=_DEPTH_EPS))
    hit = (ratio < 1.25) & valid
    denom = valid.flatten(1).sum(dim=1).clamp(min=1)
    return hit.flatten(1).float().sum(dim=1) / denom


def decoded_metric_reward(
    task: str,
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    colormap: str = "cube_edge",
) -> torch.Tensor:
    """Higher-is-better per-sample reward in [0, 1] from decoded GT metrics.

    ``pred_rgb`` / ``target_rgb`` are [B, 3, H, W] in [0, 1].
    """
    if task == "generation":
        raise ValueError(
            "decoded_metric GRPO has no GT for generation; keep generation on SFT/OPD"
        )
    pred = pred_rgb.float().clamp(0, 1)
    target = target_rgb.float().clamp(0, 1)
    if task == "depth":
        from ..core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper(colormap=colormap)
        return _per_sample_delta1(mapper.rgb_to_depth(pred), mapper.rgb_to_depth(target))
    if task == "normal":
        pred_n = F.normalize(pred * 2.0 - 1.0, dim=1, eps=1e-8)
        gt_n = F.normalize(target * 2.0 - 1.0, dim=1, eps=1e-8)
        cosine = (pred_n * gt_n).sum(dim=1).clamp(-1, 1)
        angle = torch.acos(cosine)
        return (1.0 - angle / torch.pi).flatten(1).mean(dim=1)
    if task == "referring":
        pred_m = pred.mean(dim=1) > 0.5
        gt_m = target.mean(dim=1) > 0.5
        inter = (pred_m & gt_m).flatten(1).float().sum(dim=1)
        union = (pred_m | gt_m).flatten(1).float().sum(dim=1).clamp(min=1)
        return inter / union
    # semantic / instance: colour fidelity as a stand-in for palette IoU
    return (1.0 - (pred - target).abs().mean(dim=(1, 2, 3))).clamp(0, 1)


def to_unit_rgb(images_minus_one_one: torch.Tensor) -> torch.Tensor:
    """VAE decode frame [-1, 1] → [0, 1]."""
    return (images_minus_one_one.float() * 0.5 + 0.5).clamp(0, 1)
