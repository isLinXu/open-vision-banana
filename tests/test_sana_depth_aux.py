"""Unit tests for the lightweight SANA depth geometry auxiliary objective."""

from __future__ import annotations

import torch

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.training.sana_depth_aux import (
    depth_edge_endpoint_loss,
    depth_edge_weights,
    depth_task_mask,
)


def _turbo_target(values: torch.Tensor) -> torch.Tensor:
    return DepthRGBMapper("turbo").normalized_to_rgb(values.unsqueeze(0).unsqueeze(0))


def test_depth_edge_weights_emphasize_cells_covering_a_depth_boundary() -> None:
    normalized = torch.cat((torch.zeros(8, 4), torch.ones(8, 4)), dim=1)
    weights = depth_edge_weights(
        _turbo_target(normalized), (2, 2), edge_threshold=0.025, edge_weight=3.0
    )

    assert weights.shape == (1, 1, 2, 2)
    assert weights.min().item() >= 1.0
    assert weights.max().item() > weights.min().item()


def test_depth_edge_endpoint_loss_is_zero_for_a_perfect_flow_prediction() -> None:
    source = torch.randn(1, 4, 3, 3)
    target = torch.randn_like(source)
    sigma = torch.tensor([0.6])
    noisy = sigma.view(1, 1, 1, 1) * source + (1.0 - sigma).view(1, 1, 1, 1) * target
    target_rgb = _turbo_target(torch.linspace(0.0, 1.0, 8).repeat(8, 1))

    loss = depth_edge_endpoint_loss(
        source - target, noisy, target, target_rgb, sigma, gradient_weight=0.5
    )

    assert loss.item() < 1e-10


def test_depth_edge_endpoint_loss_backpropagates_to_velocity() -> None:
    source = torch.randn(1, 4, 3, 3)
    target = torch.randn_like(source)
    sigma = torch.tensor([0.7])
    noisy = sigma.view(1, 1, 1, 1) * source + (1.0 - sigma).view(1, 1, 1, 1) * target
    predicted_velocity = (source - target).detach().clone().requires_grad_(True)
    predicted_velocity.data[..., 1, 1] += 0.5
    target_rgb = _turbo_target(torch.cat((torch.zeros(8, 4), torch.ones(8, 4)), dim=1))

    loss = depth_edge_endpoint_loss(
        predicted_velocity,
        noisy,
        target,
        target_rgb,
        sigma,
        edge_weight=2.0,
        gradient_weight=0.5,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted_velocity.grad is not None
    assert predicted_velocity.grad.abs().sum().item() > 0.0


def test_depth_task_mask_only_selects_depth_rows() -> None:
    assert torch.equal(
        depth_task_mask(["depth", "semantic", "depth"], 3, torch.device("cpu")),
        torch.tensor([True, False, True]),
    )
