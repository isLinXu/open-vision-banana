"""SANA-frame loss terms must see flow (t, v) in SANA's frame.

The depth-edge / normal-angular aux losses rebuild the target endpoint as
``sample - sigma * v`` and flow_min_snr_weight takes SANA's sigma. For rectified
flow (FLUX) the raw (t, v) rebuilt the source, so a perfect prediction was penalised,
and the Min-SNR weighting was mirrored along the path.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch
from _paths import REPO_ROOT

from open_vision_banana.training.flow_matching import FlowScheduler
from open_vision_banana.training.losses import TrainingLosses
from open_vision_banana.training.sana_flow import flow_min_snr_weight

B, C, H, W = 2, 4, 8, 8
CPU = torch.device("cpu")
T = torch.tensor([0.2, 0.7])


def flow_sample(family: str):
    torch.manual_seed(0)
    source, target = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    noisy, velocity = FlowScheduler.create_noisy_latents(target, source, None, T, family)
    return noisy, velocity, target


def decode(latents: torch.Tensor) -> torch.Tensor:
    return torch.tanh(latents[:, :3])


def aux_losses(losses: TrainingLosses, family: str, tasks: list[str]):
    noisy, velocity, target = flow_sample(family)
    rgb = (decode(target) + 1.0) / 2.0
    return losses.compute_aux_losses(velocity, noisy, target, rgb, T, tasks, decode, CPU)


@pytest.mark.parametrize("family", ["sana", "flux"])
def test_endpoint_in_sana_frame_is_target(family):
    noisy, velocity, target = flow_sample(family)
    sigma, v = FlowScheduler.to_sana_frame(family, T, velocity)
    torch.testing.assert_close(noisy - sigma.view(-1, 1, 1, 1) * v, target)


@pytest.mark.parametrize("family", ["sana", "flux"])
def test_perfect_prediction_has_zero_aux_losses(family):
    losses = TrainingLosses(
        depth_edge_aux_weight=1.0, normal_angular_loss_weight=1.0, family=family
    )
    depth, normal = aux_losses(losses, family, ["depth", "normal"])
    assert depth.item() < 1e-6
    assert normal.item() < 1e-6


def test_family_mismatch_rebuilds_the_source():
    # A FLUX sample scored in SANA's frame: the old behaviour for every FLUX run.
    losses = TrainingLosses(depth_edge_aux_weight=1.0, family="sana")
    depth, _ = aux_losses(losses, "flux", ["depth", "depth"])
    assert depth.item() > 0.1


def test_aux_losses_skipped_when_disabled():
    depth, normal = aux_losses(TrainingLosses(family="flux"), "flux", ["depth", "normal"])
    assert depth.item() == 0 and normal.item() == 0


@pytest.mark.parametrize("family", ["sana", "flux"])
def test_min_snr_uses_sana_sigma(family):
    pred, target = torch.zeros(B, C, H, W), torch.ones(B, C, H, W)
    loss, per_sample, _ = TrainingLosses(min_snr=True, family=family).compute_flow_loss(
        pred, target, T, ["depth"] * B, CPU
    )
    sigma = T if family == "sana" else 1.0 - T
    torch.testing.assert_close(loss, (per_sample * flow_min_snr_weight(sigma)).mean())


def test_legacy_train_flux_mix_puts_source_at_t_equal_one():
    # train_flux.py scores its aux losses with (t, -v): its mix_rectified_flow weights
    # the source by t, the opposite of FlowScheduler's rectified convention.
    pytest.importorskip("wandb")
    path = REPO_ROOT / "scripts" / "train_flux.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    torch.manual_seed(0)
    source, target = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    noisy, velocity = module.mix_rectified_flow(target, None, T, input_latents=source)
    torch.testing.assert_close(noisy - T.view(-1, 1, 1, 1) * -velocity, target)
