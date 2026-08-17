from __future__ import annotations

import torch
from diffusers import DPMSolverMultistepScheduler

from open_vision_banana.training.sana_flow import (
    denormalize_vae_latents,
    normalize_vae_latents,
    prepare_sana_flow_batch,
    sample_sana_flow_timesteps,
)


def _scheduler() -> DPMSolverMultistepScheduler:
    return DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        algorithm_type="dpmsolver++",
        use_flow_sigmas=True,
        flow_shift=3.0,
        final_sigmas_type="zero",
    )


def test_flow_endpoints_match_scheduler_semantics() -> None:
    source = torch.full((2, 1, 1, 1), 3.0)
    target = torch.full((2, 1, 1, 1), -2.0)

    at_target, velocity = prepare_sana_flow_batch(source, target, torch.zeros(2))
    at_source, _ = prepare_sana_flow_batch(source, target, torch.ones(2))

    assert torch.equal(at_target, target)
    assert torch.equal(at_source, source)
    assert torch.equal(velocity, source - target)


def test_perfect_flow_prediction_reaches_target_with_native_scheduler() -> None:
    torch.manual_seed(0)
    source = torch.randn(2, 32, 4, 4)
    target = torch.randn_like(source)
    scheduler = _scheduler()
    scheduler.set_timesteps(20, device="cpu")

    sample = source.clone()
    perfect_velocity = source - target
    for timestep in scheduler.timesteps:
        sample = scheduler.step(perfect_velocity, timestep, sample, return_dict=False)[0]

    assert torch.mean(torch.abs(sample - target)).item() < 0.005


def test_sampled_timesteps_are_shifted_native_sana_times() -> None:
    torch.manual_seed(0)
    sigma, timestep = sample_sana_flow_timesteps(
        128, torch.device("cpu"), _scheduler(), logit_normal=False
    )

    assert torch.all((sigma >= 0.0) & (sigma <= 1.0))
    assert torch.allclose(timestep, sigma * 1000.0)
    assert timestep.max().item() > 100.0


def test_vae_scaling_round_trip() -> None:
    class DummyConfig:
        scaling_factor = 0.41407

    class DummyVae:
        config = DummyConfig()

    latents = torch.randn(2, 32, 4, 4)
    assert torch.allclose(denormalize_vae_latents(normalize_vae_latents(latents, DummyVae()), DummyVae()), latents)
