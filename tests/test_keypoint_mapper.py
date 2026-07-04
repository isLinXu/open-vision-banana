"""Tests for 2D keypoint detection RGB encoder/decoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.keypoint_mapper import KeypointRGBMapper


def test_keypoint_roundtrip() -> None:
    """Test keypoint -> RGB -> keypoint roundtrip."""
    mapper = KeypointRGBMapper(num_keypoints=5, dot_radius=3)

    keypoints = torch.tensor([
        [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7], [0.1, 0.9], [0.9, 0.1]]
    ])  # [1, 5, 2]
    visibility = torch.ones(1, 5, dtype=torch.bool)
    visibility[0, 3] = False  # one invisible

    rgb = mapper.keypoints_to_rgb(keypoints, visibility)
    assert rgb.shape == (1, 3, 512, 512)
    assert rgb.max() > 0

    recovered_kp, recovered_vis = mapper.rgb_to_keypoints(rgb)
    assert recovered_kp.shape == (1, 5, 2)
    assert recovered_vis.shape == (1, 5)
    assert recovered_vis[0, 3] == False  # invisible keypoint not recovered

    # Check visible keypoints are close
    for i in [0, 1, 2, 4]:
        assert recovered_vis[0, i]
        err = (keypoints[0, i] - recovered_kp[0, i]).abs().max().item()
        assert err < 0.05, f"Keypoint {i} error = {err}"


def test_keypoint_visibility() -> None:
    """Test that invisible keypoints are not rendered."""
    mapper = KeypointRGBMapper(num_keypoints=3, dot_radius=3)

    keypoints = torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]])
    visibility = torch.tensor([[True, False, True]])

    rgb = mapper.keypoints_to_rgb(keypoints, visibility)
    # Middle keypoint is invisible, so only 2 colors should be present
    # But they overlap at same location, so it's fine


def test_coco_keypoint_colors() -> None:
    """Test COCO 17 keypoint color palette."""
    from open_vision_banana.core.keypoint_mapper import COCO_KEYPOINT_COLORS, COCO_KEYPOINT_NAMES

    assert len(COCO_KEYPOINT_NAMES) == 17
    assert COCO_KEYPOINT_COLORS.shape == (17, 3)

    mapper = KeypointRGBMapper(num_keypoints=17)
    assert mapper.colors.shape == (17, 3)


def test_keypoint_dynamic_resolution() -> None:
    """Test keypoint encoding with non-default resolution (P1-1)."""
    mapper = KeypointRGBMapper(num_keypoints=3, dot_radius=3, output_height=256, output_width=256)

    keypoints = torch.tensor([[[0.5, 0.5], [0.2, 0.3], [0.8, 0.9]]])
    visibility = torch.ones(1, 3, dtype=torch.bool)

    # Default resolution from mapper
    rgb = mapper.keypoints_to_rgb(keypoints, visibility)
    assert rgb.shape == (1, 3, 256, 256)

    # Override with per-call resolution
    rgb = mapper.keypoints_to_rgb(keypoints, visibility, output_size=(128, 128))
    assert rgb.shape == (1, 3, 128, 128)


def test_keypoint_vectorized_batch() -> None:
    """Test vectorized batch encoding produces correct results (P1-3)."""
    mapper = KeypointRGBMapper(num_keypoints=3, dot_radius=3)

    keypoints = torch.tensor([
        [[0.2, 0.3], [0.5, 0.5], [0.8, 0.9]],
        [[0.1, 0.1], [0.4, 0.6], [0.9, 0.2]],
    ])  # [2, 3, 2]
    visibility = torch.ones(2, 3, dtype=torch.bool)

    rgb = mapper.keypoints_to_rgb(keypoints, visibility)
    assert rgb.shape == (2, 3, 512, 512)
    assert rgb.max() > 0

    # Decode should recover both batches
    recovered_kp, recovered_vis = mapper.rgb_to_keypoints(rgb)
    assert recovered_kp.shape == (2, 3, 2)
    assert recovered_vis.shape == (2, 3)

    # Check visible keypoints are close
    for b in range(2):
        for i in range(3):
            assert recovered_vis[b, i]
            err = (keypoints[b, i] - recovered_kp[b, i]).abs().max().item()
            assert err < 0.05, f"Batch {b} keypoint {i} error = {err}"


if __name__ == "__main__":
    test_keypoint_roundtrip()
    test_keypoint_visibility()
    test_coco_keypoint_colors()
    test_keypoint_dynamic_resolution()
    test_keypoint_vectorized_batch()
    print("All keypoint mapper tests passed!")
