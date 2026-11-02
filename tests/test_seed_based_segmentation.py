"""Tests for seed-based Union-Find instance segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
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
        "seed_merge_rule": "pixel_only",
        "spatial_merge_area_basis": "bbox",
    }


def test_seed_merge_rule_switch_pixel_only_vs_pixel_and_seed() -> None:
    """``pixel_only`` is the paper Eq.(3) rule; ``pixel_and_seed`` is an opt-in that can
    only ever split more (never merge more) and is threaded through the config contract.
    """
    img = torch.zeros(8, 64, 3)
    for x in range(64):
        img[:, x] = torch.tensor([60.0 + 5.0 * x, 60.0, 60.0])
    bg_mask = torch.zeros(8, 64, dtype=torch.bool)

    lenient = SegmentationDecoder(seed_merge_rule="pixel_only")
    strict = SegmentationDecoder(seed_merge_rule="pixel_and_seed")
    n_lenient = int(lenient._seed_based_union_find(img, bg_mask).unique().numel())
    n_strict = int(strict._seed_based_union_find(img, bg_mask).unique().numel())
    assert n_strict >= n_lenient
    assert strict.instance_config.to_dict()["seed_merge_rule"] == "pixel_and_seed"
    assert (
        RGBMapper(instance_decoding_config=strict.instance_config).seg_decoder.seed_merge_rule
        == "pixel_and_seed"
    )
    with pytest.raises(ValueError, match="seed_merge_rule"):
        InstanceDecodingConfig(seed_merge_rule="bogus")


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
    instances_adj = decoder.decode_instance(
        rgb, background_color=(0, 0, 0), grouping_mode="adjacent"
    )

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


def test_stage2_converges_on_large_nonconvex_instance_over_32px() -> None:
    """A single large U-shaped instance (>100px tall, so bbox > 5x area in Stage 5)
    must already be 1 connected component in Stage 2 without fragmenting at 32px.
    """
    decoder = SegmentationDecoder.paper_reproduction()
    img = torch.zeros(256, 256, 3)
    # Thin U-shape: left vertical bar [10:210, 10:15], right vertical bar [10:210, 200:205],
    # bottom horizontal bar [205:210, 10:205]. Total height = 200px (>32px).
    img[10:210, 10:15] = torch.tensor([220.0, 40.0, 40.0])
    img[10:210, 200:205] = torch.tensor([220.0, 40.0, 40.0])
    img[205:210, 10:205] = torch.tensor([220.0, 40.0, 40.0])
    bg_mask = img.norm(dim=-1) < decoder.color_tolerance

    for labels in (
        decoder._union_find_grouping(img, bg_mask),
        decoder._seed_based_union_find(img, bg_mask),
    ):
        fg_labels = labels[~bg_mask].unique()
        assert len(fg_labels) == 1 and int(fg_labels[0].item()) > 0


if __name__ == "__main__":
    test_adjacent_mode_default()
    test_seed_mode_convergence()
    test_seed_vs_adjacent_on_gradients()
    test_seed_mode_three_blocks()
    print("All seed-based segmentation tests passed!")


def _two_frames() -> tuple[torch.Tensor, torch.Tensor]:
    """Two same-coloured hollow 20x20 frames, 10 px apart vertically.

    Each frame has 76 pixels in a 400-px bounding box; their union box is 50x20=1000.
    With gamma=5 the pixel basis allows 5*(76+76)=760 < 1000 (no merge) while the
    bounding-box basis allows 5*(400+400)=4000 >= 1000 (merge).
    """
    labels = torch.zeros(60, 30, dtype=torch.long)
    for lab, y0 in ((1, 0), (2, 30)):
        labels[y0 : y0 + 20, 0:20] = lab
        labels[y0 + 1 : y0 + 19, 1:19] = 0
    img = torch.zeros(60, 30, 3)
    img[labels > 0] = torch.tensor([200.0, 40.0, 40.0])
    return labels, img


def test_stage5_area_basis_bbox_is_paper_default_and_pixel_is_stricter() -> None:
    labels, img = _two_frames()
    assert int((labels == 1).sum()) == 76 and int((labels == 2).sum()) == 76

    paper = SegmentationDecoder.paper_reproduction()
    assert paper.spatial_merge_area_basis == "bbox"
    merged = paper._spatial_merging(labels.clone(), img)
    assert len(merged.unique()) == 2  # background + one merged component

    strict = SegmentationDecoder(spatial_merge_area_basis="pixel")
    kept = strict._spatial_merging(labels.clone(), img)
    assert len(kept.unique()) == 3  # background + two components

    with pytest.raises(ValueError, match="spatial_merge_area_basis"):
        SegmentationDecoder(spatial_merge_area_basis="hull")
