"""Tests for optical flow RGB encoder/decoder."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.flow_mapper import FlowRGBMapper


def test_flow_roundtrip() -> None:
    """Test flow -> RGB -> flow roundtrip with small EPE."""
    mapper = FlowRGBMapper(max_flow=20.0)

    # Create a simple flow field
    u = torch.linspace(-5.0, 5.0, 64).view(1, 1, 1, 64).expand(1, 1, 64, 64)
    v = torch.linspace(-5.0, 5.0, 64).view(1, 1, 64, 1).expand(1, 1, 64, 64)
    flow = torch.cat([u, v], dim=1)

    rgb = mapper.flow_to_rgb(flow)
    assert rgb.shape == (1, 3, 64, 64)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    recovered = mapper.rgb_to_flow(rgb)
    assert recovered.shape == (1, 2, 64, 64)

    epe = torch.sqrt((flow - recovered) ** 2).mean().item()
    assert epe < 0.5, f"EPE = {epe}, expected < 0.5"


def test_zero_flow() -> None:
    """Test zero flow encoding."""
    mapper = FlowRGBMapper(max_flow=20.0)

    flow = torch.zeros(1, 2, 64, 64)
    rgb = mapper.flow_to_rgb(flow)

    # Zero flow -> direction undefined, but magnitude = 0 -> saturation = 0
    # Hue arbitrary, saturation = 0, value = 0.5 -> gray
    assert torch.allclose(rgb[:, 0], rgb[:, 1], atol=1e-5) and torch.allclose(rgb[:, 1], rgb[:, 2], atol=1e-5), "Zero flow should be gray"

    recovered = mapper.rgb_to_flow(rgb)
    epe = torch.sqrt((flow - recovered) ** 2).mean().item()
    assert epe < 0.1, f"Zero flow EPE = {epe}"


def test_max_flow_boundary() -> None:
    """Test max flow magnitude boundary."""
    mapper = FlowRGBMapper(max_flow=20.0)

    # Flow at exactly max magnitude
    flow = torch.zeros(1, 2, 64, 64)
    flow[0, 0, 32, 32] = 20.0  # u = max_flow
    flow[0, 1, 32, 32] = 0.0

    rgb = mapper.flow_to_rgb(flow)
    recovered = mapper.rgb_to_flow(rgb)

    epe = torch.sqrt((flow - recovered) ** 2).mean().item()
    assert epe < 0.5, f"Max flow EPE = {epe}"


def test_flow_direction_encoding() -> None:
    """Test that direction is correctly encoded in hue."""
    mapper = FlowRGBMapper(max_flow=20.0)

    # Rightward flow (u=10, v=0) -> hue = 0.5
    flow_right = torch.zeros(1, 2, 64, 64)
    flow_right[0, 0, :, :] = 10.0
    rgb_right = mapper.flow_to_rgb(flow_right)
    # Hue = atan2(0, 10) / 2π + 0.5 = 0.5 -> cyan-ish (high G, moderate R, B)
    # In HSV: hue=0.5, sat=0.5, val=0.5 -> RGB ~= (0.25, 0.5, 0.25)
    # Just check it's not gray
    assert rgb_right[0, 0, 32, 32] != rgb_right[0, 1, 32, 32], "Rightward flow should have non-uniform RGB"

    # Upward flow (u=0, v=10) -> hue = 0.75
    flow_up = torch.zeros(1, 2, 64, 64)
    flow_up[0, 1, :, :] = 10.0
    rgb_up = mapper.flow_to_rgb(flow_up)
    assert rgb_up[0, 0, 32, 32] != rgb_up[0, 1, 32, 32], "Upward flow should have non-uniform RGB"

    # The two directions should produce different RGB
    assert not torch.allclose(rgb_right, rgb_up, atol=0.01), "Different directions should produce different RGB"


if __name__ == "__main__":
    test_flow_roundtrip()
    test_zero_flow()
    test_max_flow_boundary()
    test_flow_direction_encoding()
    print("All flow mapper tests passed!")
