"""Tests for depth edge detection RGB encoder/decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.depth_edge_mapper import DepthEdgeRGBMapper


def test_depth_edge_roundtrip() -> None:
    """Test depth -> edge RGB -> edge mask roundtrip."""
    mapper = DepthEdgeRGBMapper(grad_threshold=1.0)

    # Create a depth map with a sharp discontinuity
    depth = torch.ones(1, 1, 64, 64) * 5.0
    depth[0, 0, 20:44, 20:44] = 15.0  # sharp jump

    rgb = mapper.depth_to_edge_rgb(depth)
    assert rgb.shape == (1, 3, 64, 64)
    # Edges should be around the discontinuity boundary
    assert rgb.max() > 0

    edge_mask = mapper.rgb_to_depth_edge(rgb)
    assert edge_mask.shape == (1, 1, 64, 64)
    assert edge_mask.dtype == torch.bool


def test_depth_edge_on_smooth_surface() -> None:
    """Test that smooth depth has minimal edges (only boundary artifacts)."""
    mapper = DepthEdgeRGBMapper(grad_threshold=1.0)

    # Smooth depth (constant)
    depth = torch.ones(1, 1, 64, 64) * 5.0

    rgb = mapper.depth_to_edge_rgb(depth)
    edge_mask = mapper.rgb_to_depth_edge(rgb)
    # Sobel operator produces boundary artifacts; allow up to 10% edge pixels
    edge_fraction = edge_mask.float().mean().item()
    assert edge_fraction < 0.10, f"Smooth depth edge fraction = {edge_fraction}, expected < 0.10"


def test_depth_edge_threshold() -> None:
    """Test that higher threshold reduces edge count."""
    mapper_low = DepthEdgeRGBMapper(grad_threshold=0.5)
    mapper_high = DepthEdgeRGBMapper(grad_threshold=2.0)

    depth = torch.ones(1, 1, 64, 64) * 5.0
    depth[0, 0, 30:34, 30:34] = 20.0

    rgb_low = mapper_low.depth_to_edge_rgb(depth)
    rgb_high = mapper_high.depth_to_edge_rgb(depth)

    edge_low = mapper_low.rgb_to_depth_edge(rgb_low)
    edge_high = mapper_high.rgb_to_depth_edge(rgb_high)

    assert edge_low.sum() >= edge_high.sum(), "Lower threshold should detect more edges"


if __name__ == "__main__":
    test_depth_edge_roundtrip()
    test_depth_edge_on_smooth_surface()
    test_depth_edge_threshold()
    print("All depth edge mapper tests passed!")
