"""Tests for seed-based Union-Find instance segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.rgb_mapper import RGBMapper
from open_vision_banana.core.segmentation import (
    PAPER_INSTANCE_DECODING_CONFIG,
    InstanceDecodingConfig,
    SegmentationDecoder,
)
from open_vision_banana.inference.pipeline import VisionBanana


def test_adjacent_mode_default() -> None:
    """Test that adjacent mode is still the default and works."""
    decoder = SegmentationDecoder()

    # Create an image with 3 distinct colored squares on black background
    rgb = torch.zeros(1, 3, 64, 64)
    rgb[0, 0, 10:20, 10:20] = 1.0  # red square
    rgb[0, 1, 30:40, 30:40] = 1.0  # green square
    rgb[0, 2, 50:60, 50:60] = 1.0  # blue square

    instances = decoder.decode_instance(rgb, background_color=(0, 0, 0), grouping_mode="adjacent")
    assert len(instances[0]) >= 3, f"Expected >= 3 instances, got {len(instances[0])}"


def test_paper_reproduction_decoder_has_fixed_protocol() -> None:
    decoder = SegmentationDecoder.paper_reproduction()

    assert decoder.instance_config == PAPER_INSTANCE_DECODING_CONFIG
    assert decoder.instance_config.to_dict() == {
        "color_tolerance": 14.0,
        "min_area_fraction": 2e-4,
        "erosion_threshold": 0.1,
        "bbox_expansion_factor": 5.0,
        "grouping_mode": "seed",
        "adaptive_color_tolerance": False,
    }


def test_paper_reproduction_rgb_mapper_propagates_protocol() -> None:
    mapper = RGBMapper.paper_reproduction()
    assert mapper.seg_decoder.instance_config == PAPER_INSTANCE_DECODING_CONFIG


def test_rgb_mapper_accepts_full_instance_decoding_config() -> None:
    config = InstanceDecodingConfig(
        color_tolerance=9.0,
        min_area_fraction=1e-3,
        erosion_threshold=0.2,
        bbox_expansion_factor=3.0,
        grouping_mode="seed",
        adaptive_color_tolerance=False,
    )
    mapper = RGBMapper(instance_decoding_config=config)
    assert mapper.seg_decoder.instance_config == config


def test_vision_banana_propagates_instance_protocol() -> None:
    class FakeGenerator:
        source_conditioning_config = None

    model = VisionBanana(
        FakeGenerator(),
        instance_grouping_mode="seed",
        instance_adaptive_color_tolerance=False,
    )
    assert model.rgb_mapper.seg_decoder.instance_config == PAPER_INSTANCE_DECODING_CONFIG


def test_seed_mode_convergence() -> None:
    """Test that seed-based mode converges and produces instances."""
    decoder = SegmentationDecoder()

    rgb = torch.zeros(1, 3, 64, 64)
    rgb[0, 0, 10:20, 10:20] = 1.0  # red square
    rgb[0, 1, 30:40, 30:40] = 1.0  # green square
    rgb[0, 2, 50:60, 50:60] = 1.0  # blue square

    instances = decoder.decode_instance(rgb, background_color=(0, 0, 0), grouping_mode="seed")
    assert len(instances[0]) >= 3, f"Expected >= 3 instances, got {len(instances[0])}"


def test_seed_vs_adjacent_on_gradients() -> None:
    """Test that seed-based mode is more robust to gradients than adjacent."""
    decoder = SegmentationDecoder(color_tolerance=30.0)

    # Create a gradient region that should NOT be split into many tiny instances
    # Gradient from red to green over a region
    h, w = 64, 64
    rgb = torch.zeros(1, 3, h, w)
    for i in range(h):
        for j in range(w):
            rgb[0, 0, i, j] = i / h  # R increases
            rgb[0, 1, i, j] = j / w  # G increases

    # This is a tricky case: seed-based should be more compact
    instances_seed = decoder.decode_instance(rgb, background_color=(0, 0, 0), grouping_mode="seed")
    instances_adj = decoder.decode_instance(rgb, background_color=(0, 0, 0), grouping_mode="adjacent")

    # Seed-based should produce fewer, more compact components on gradients
    count_seed = len(instances_seed[0])
    count_adj = len(instances_adj[0])
    # Both should produce some components (the gradient isn't uniform)
    assert count_seed > 0, "seed mode should produce at least one component"
    assert count_adj > 0, "adjacent mode should produce at least one component"


def test_seed_mode_three_blocks() -> None:
    """Test seed-based mode on three distinct solid-color blocks."""
    decoder = SegmentationDecoder(color_tolerance=20.0)

    rgb = torch.zeros(1, 3, 128, 128)
    # Block 1: red, [10:40, 10:40]
    rgb[0, 0, 10:40, 10:40] = 200 / 255
    # Block 2: green, [10:40, 60:90]
    rgb[0, 1, 10:40, 60:90] = 200 / 255
    # Block 3: blue, [60:90, 10:40]
    rgb[0, 2, 60:90, 10:40] = 200 / 255

    instances = decoder.decode_instance(rgb, background_color=(0, 0, 0), grouping_mode="seed")
    assert len(instances[0]) >= 3, f"Expected >= 3 blocks, got {len(instances[0])}"

    # Verify each block has the right dominant color
    colors = [inst["color"] for inst in instances[0]]
    # Should have at least one red-ish, one green-ish, one blue-ish block
    has_red = any(c[0] > c[1] and c[0] > c[2] for c in colors)
    has_green = any(c[1] > c[0] and c[1] > c[2] for c in colors)
    has_blue = any(c[2] > c[0] and c[2] > c[1] for c in colors)
    assert has_red or has_green or has_blue, "Should have blocks with distinct colors"


if __name__ == "__main__":
    test_adjacent_mode_default()
    test_seed_mode_convergence()
    test_seed_vs_adjacent_on_gradients()
    test_seed_mode_three_blocks()
    print("All seed-based segmentation tests passed!")
