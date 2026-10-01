"""The normal angular aux loss must reach the model through the VAE decode.

``ModelBackend.decode_latents`` used to run under ``torch.no_grad``; the unified trainer
hands it to the loss, which was then logged and added to the total with no gradient.
"""

from __future__ import annotations

import pytest
import torch

from open_vision_banana.training import model_backend
from open_vision_banana.training.flow_matching import FlowScheduler
from open_vision_banana.training.losses import TrainingLosses
from open_vision_banana.training.model_backend import ModelBackend

B, C, H, W = 2, 4, 8, 8
T = torch.tensor([0.3, 0.6])


@pytest.fixture
def backend(monkeypatch) -> ModelBackend:
    monkeypatch.setattr(
        model_backend, "registry_vae_decode", lambda vae, latents, cfg: torch.tanh(latents[:, :3])
    )
    backend = object.__new__(ModelBackend)
    backend.vae = None
    backend.model_config = None
    return backend


def test_decode_latents_keeps_the_graph(backend):
    latents = torch.randn(B, C, H, W, requires_grad=True)
    assert backend.decode_latents(latents).requires_grad


@pytest.mark.parametrize("family", ["sana", "flux"])
def test_normal_aux_loss_backpropagates_to_the_prediction(backend, family):
    torch.manual_seed(0)
    source, target = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    noisy, velocity = FlowScheduler.create_noisy_latents(target, source, None, T, family)
    rgb = (backend.decode_latents(target) + 1.0) / 2.0
    predicted = (velocity + 0.5 * torch.randn_like(velocity)).requires_grad_()
    losses = TrainingLosses(normal_angular_loss_weight=1.0, family=family)

    _, normal = losses.compute_aux_losses(
        predicted,
        noisy,
        target,
        rgb,
        T,
        ["normal"] * B,
        backend.decode_latents,
        torch.device("cpu"),
    )
    normal.backward()

    assert normal.item() > 0
    assert predicted.grad is not None and predicted.grad.abs().sum() > 0
