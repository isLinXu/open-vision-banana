"""ModelBackend must hand each backbone the timestep range it was built for.

The unified trainer used to pass flow time sigma in [0, 1] straight to SANA's
transformer, which expects scheduler timesteps (sigma * 1000).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from open_vision_banana.training import model_backend
from open_vision_banana.training.model_backend import ModelBackend

B = 2
SIGMA = torch.tensor([0.25, 1.0])


def make_backend(family: str, *, timestep_scale=None, num_train_timesteps=None) -> ModelBackend:
    backend = object.__new__(ModelBackend)
    backend.family = family
    backend.device = torch.device("cpu")
    backend.dtype = torch.float32
    backend.model_config = SimpleNamespace(family=family)
    transformer_config = SimpleNamespace()
    if timestep_scale is not None:
        transformer_config.timestep_scale = timestep_scale
    backend.peft_model = SimpleNamespace(config=transformer_config)
    scheduler = None
    if num_train_timesteps is not None:
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=num_train_timesteps))
    backend.components = SimpleNamespace(scheduler=scheduler)
    return backend


def test_sana_gets_scheduler_timesteps():
    backend = make_backend("sana", num_train_timesteps=1000)
    torch.testing.assert_close(backend.model_timesteps(SIGMA, B), torch.tensor([250.0, 1000.0]))


def test_sana_defaults_to_1000_steps_and_applies_timestep_scale():
    backend = make_backend("sana", timestep_scale=0.5)
    torch.testing.assert_close(backend.model_timesteps(SIGMA, B), torch.tensor([125.0, 500.0]))


def test_sana_timesteps_are_computed_in_float32():
    backend = make_backend("sana")
    sigma = torch.tensor([0.7013], dtype=torch.bfloat16)
    out = backend.model_timesteps(sigma, 1)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, sigma.float() * 1000)


def test_flux_timesteps_are_unchanged():
    backend = make_backend("flux")
    assert backend.model_timesteps(SIGMA, B) is SIGMA


def test_forward_passes_model_timesteps_to_the_backbone(monkeypatch):
    seen = {}

    def fake_backbone_forward(peft_model, latents, timesteps, **kwargs):
        seen["timesteps"] = timesteps
        return latents

    monkeypatch.setattr(model_backend, "backbone_forward", fake_backbone_forward)
    backend = make_backend("sana")
    latents = torch.zeros(B, 4, 8, 8)
    text_cond = {
        "encoder_hidden_states": torch.zeros(B, 3, 5),
        "attention_mask": torch.ones(B, 3),
    }

    out = backend.forward(latents, SIGMA, text_cond, B)

    assert out is latents
    torch.testing.assert_close(seen["timesteps"], torch.tensor([250.0, 1000.0]))
