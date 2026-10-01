"""Validation must measure the training objective, and the same one at every call."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from open_vision_banana.training.config import TrainingConfig
from open_vision_banana.training.trainer import Trainer

B, C, H, W = 2, 4, 8, 8


def make_trainer(**config) -> tuple[Trainer, list[dict]]:
    flow_calls: list[dict] = []

    def prepare_flow_batch(target, source, batch_size, **kwargs):
        flow_calls.append(kwargs)
        noise = torch.randn_like(target)
        return {
            "timesteps": torch.rand(batch_size),
            "noisy_latents": noise,
            "velocity_target": target,
        }

    backend = SimpleNamespace(
        peft_model=torch.nn.Linear(1, 1),
        encode_image=lambda pixels: torch.zeros(pixels.shape[0], C, H, W),
        prepare_flow_batch=prepare_flow_batch,
        encode_text=lambda *a: {},
        forward=lambda noisy, *a: noisy,
    )
    losses = SimpleNamespace(
        compute_flow_loss=lambda pred, target, t, tasks, device: (
            (pred - target).square().mean() + t.mean(),
            None,
            None,
        )
    )
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(**config)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.backend = backend
    trainer.losses = losses
    batch = {
        "prompt": ["p"] * B,
        "target_rgb": torch.rand(B, 3, 16, 16),
        "image": torch.rand(B, 3, 16, 16),
        "task": ["depth"] * B,
    }
    trainer.val_loader = [batch, batch, batch]
    return trainer, flow_calls


def test_validation_uses_training_flow_settings():
    trainer, flow_calls = make_trainer(logit_normal=False, flow_shift=3.0, flow_sigma=0.1)
    trainer.validate()
    assert len(flow_calls) == 3
    for kwargs in flow_calls:
        assert kwargs["logit_normal"] is False
        assert kwargs["flow_shift"] == 3.0
        assert kwargs["flow_sigma"] == 0.1


def test_validation_loss_is_repeatable_and_leaves_training_rng_alone():
    trainer, _ = make_trainer(seed=7)
    torch.manual_seed(123)
    first = trainer.validate()
    after_first = torch.rand(3)
    second = trainer.validate()

    assert first == second  # same noise and timesteps at every evaluation
    torch.manual_seed(123)
    torch.testing.assert_close(after_first, torch.rand(3))  # as if validate() never drew
