"""Tests for edge detection RGB encoder/decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.edge_mapper import EdgeRGBMapper


def test_edge_roundtrip() -> None:
    """Test edge mask -> RGB -> edge mask roundtrip."""
    mapper = EdgeRGBMapper()

    edge = torch.zeros(1, 1, 64, 64)
    edge[0, 0, 20:44, 20:44] = 1.0

    rgb = mapper.edge_to_rgb(edge)
    assert rgb.shape == (1, 3, 64, 64)
    # Edges should be white, background black
    assert (rgb[0, :, 20:44, 20:44] == 1.0).all()
    assert (rgb[0, :, :20, :20] == 0.0).all()

    recovered = mapper.rgb_to_edge(rgb)
    assert recovered.shape == (1, 1, 64, 64)
    assert recovered.dtype == torch.bool
    assert (recovered[0, 0, 20:44, 20:44]).all()
    assert not recovered[0, 0, :20, :20].any()


def test_edge_threshold_variation() -> None:
    """Test that different thresholds produce different results."""
    mapper_low = EdgeRGBMapper(threshold=0.3)
    mapper_high = EdgeRGBMapper(threshold=0.7)

    # Create a gray-ish edge (not pure white)
    rgb = torch.zeros(1, 3, 64, 64)
    rgb[0, :, 20:44, 20:44] = 0.5  # gray

    edge_low = mapper_low.rgb_to_edge(rgb)
    edge_high = mapper_high.rgb_to_edge(rgb)

    # Lower threshold should detect more edges
    assert edge_low.sum() >= edge_high.sum()


def test_edge_luminance_formula() -> None:
    """Test that luminance formula (0.299R + 0.587G + 0.114B) is used."""
    mapper = EdgeRGBMapper(threshold=0.5)

    # Pure green should have high luminance (> 0.5)
    rgb_green = torch.zeros(1, 3, 64, 64)
    rgb_green[0, 1, 20:44, 20:44] = 1.0

    edge = mapper.rgb_to_edge(rgb_green)
    assert edge[0, 0, 20:44, 20:44].all(), "Pure green should be detected as edge"

    # Pure blue has low luminance (~0.114)
    rgb_blue = torch.zeros(1, 3, 64, 64)
    rgb_blue[0, 2, 20:44, 20:44] = 1.0

    edge_blue = mapper.rgb_to_edge(rgb_blue)
    assert not edge_blue[0, 0, 20:44, 20:44].any(), "Pure blue should not be detected as edge at threshold 0.5"


if __name__ == "__main__":
    test_edge_roundtrip()
    test_edge_threshold_variation()
    test_edge_luminance_formula()
    print("All edge mapper tests passed!")
