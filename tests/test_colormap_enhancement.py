"""Tests for multi-colormap depth training enhancement."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.depth_mapper import (
    DepthRGBMapper,
    get_available_colormaps,
)


def test_available_colormaps() -> None:
    """Test that all expected colormaps are available."""
    names = get_available_colormaps()
    expected = ["cube_edge", "plasma", "inferno", "viridis", "grayscale", "rainbow"]
    assert sorted(names) == sorted(expected), f"Expected {expected}, got {names}"


def test_cube_edge_colormap_default() -> None:
    """Test that cube_edge is the default and roundtrip is accurate."""
    mapper = DepthRGBMapper()
    assert mapper.colormap == "cube_edge"
    assert mapper.colormap_keypoints.shape[0] == 7  # 7 keypoints (6 segments)

    depth = torch.rand(1, 1, 64, 64) * 50.0 + 1.0
    rgb = mapper.depth_to_rgb(depth)
    recovered = mapper.rgb_to_depth(rgb)
    abs_rel = ((depth - recovered).abs() / depth).mean().item()
    assert abs_rel < 0.005, f"cube_edge roundtrip abs_rel = {abs_rel}"


def test_plasma_colormap_roundtrip() -> None:
    """Test plasma colormap roundtrip accuracy."""
    mapper = DepthRGBMapper(colormap="plasma")
    assert mapper.colormap == "plasma"
    assert mapper.colormap_keypoints.shape[0] == 8  # 8 keypoints

    depth = torch.rand(1, 1, 64, 64) * 50.0 + 1.0
    rgb = mapper.depth_to_rgb(depth)
    recovered = mapper.rgb_to_depth(rgb)
    abs_rel = ((depth - recovered).abs() / depth).mean().item()
    assert abs_rel < 0.005, f"plasma roundtrip abs_rel = {abs_rel}"


def test_inferno_colormap_roundtrip() -> None:
    """Test inferno colormap roundtrip accuracy."""
    mapper = DepthRGBMapper(colormap="inferno")
    depth = torch.rand(1, 1, 64, 64) * 50.0 + 1.0
    rgb = mapper.depth_to_rgb(depth)
    recovered = mapper.rgb_to_depth(rgb)
    abs_rel = ((depth - recovered).abs() / depth).mean().item()
    assert abs_rel < 0.005, f"inferno roundtrip abs_rel = {abs_rel}"


def test_viridis_colormap_roundtrip() -> None:
    """Test viridis colormap roundtrip accuracy."""
    mapper = DepthRGBMapper(colormap="viridis")
    depth = torch.rand(1, 1, 64, 64) * 50.0 + 1.0
    rgb = mapper.depth_to_rgb(depth)
    recovered = mapper.rgb_to_depth(rgb)
    abs_rel = ((depth - recovered).abs() / depth).mean().item()
    assert abs_rel < 0.005, f"viridis roundtrip abs_rel = {abs_rel}"


def test_grayscale_colormap_roundtrip() -> None:
    """Test grayscale colormap roundtrip accuracy."""
    mapper = DepthRGBMapper(colormap="grayscale")
    depth = torch.rand(1, 1, 64, 64) * 50.0 + 1.0
    rgb = mapper.depth_to_rgb(depth)
    # Verify grayscale: R==G==B
    assert (rgb[:, 0] == rgb[:, 1]).all() and (rgb[:, 1] == rgb[:, 2]).all()
    recovered = mapper.rgb_to_depth(rgb)
    abs_rel = ((depth - recovered).abs() / depth).mean().item()
    assert abs_rel < 0.005, f"grayscale roundtrip abs_rel = {abs_rel}"


def test_set_colormap_switch() -> None:
    """Test switching colormap at runtime."""
    mapper = DepthRGBMapper(colormap="cube_edge")
    mapper.set_colormap("plasma")
    assert mapper.colormap == "plasma"
    mapper.set_colormap("grayscale")
    assert mapper.colormap == "grayscale"


def test_all_colormaps_quantized_roundtrip() -> None:
    """Test all colormaps with 8-bit quantization."""
    torch.manual_seed(0)
    depth = torch.rand(1, 1, 32, 32) * 49.0 + 1.0

    for name in get_available_colormaps():
        mapper = DepthRGBMapper(colormap=name)
        rgb = mapper.depth_to_rgb(depth)
        rgb_q = (rgb * 255).round() / 255.0
        recovered = mapper.rgb_to_depth(rgb_q)
        abs_rel = ((depth - recovered).abs() / depth).mean().item()
        assert abs_rel < 0.01, f"{name} quantized roundtrip abs_rel = {abs_rel}"


def test_rainbow_colormap_roundtrip() -> None:
    """Test rainbow colormap roundtrip accuracy."""
    mapper = DepthRGBMapper(colormap="rainbow")
    depth = torch.rand(1, 1, 64, 64) * 50.0 + 1.0
    rgb = mapper.depth_to_rgb(depth)
    recovered = mapper.rgb_to_depth(rgb)
    abs_rel = ((depth - recovered).abs() / depth).mean().item()
    assert abs_rel < 0.005, f"rainbow roundtrip abs_rel = {abs_rel}"


if __name__ == "__main__":
    test_available_colormaps()
    test_cube_edge_colormap_default()
    test_plasma_colormap_roundtrip()
    test_inferno_colormap_roundtrip()
    test_viridis_colormap_roundtrip()
    test_grayscale_colormap_roundtrip()
    test_set_colormap_switch()
    test_all_colormaps_quantized_roundtrip()
    print("All colormap enhancement tests passed!")
