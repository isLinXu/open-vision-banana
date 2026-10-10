"""Paper-default depth curve stays metric; log_relative is an opt-in ablation."""

from __future__ import annotations

import math

import pytest
import torch

from open_vision_banana.core.depth_mapper import (
    DEFAULT_LOG_ALPHA,
    LOG_RELATIVE_CURVE,
    PAPER_DEPTH_CURVE,
    DepthRGBMapper,
    get_available_depth_curves,
)
from open_vision_banana.evaluation.evaluator import decode_depth_rgb
from open_vision_banana.training.trainer import drop_oversized_gradients


def test_default_mapper_is_paper_metric_power():
    mapper = DepthRGBMapper()
    assert mapper.curve == PAPER_DEPTH_CURVE
    assert mapper.is_metric is True
    assert get_available_depth_curves()[0] == PAPER_DEPTH_CURVE

    # Regression lock: 1 m under λ=-3, c=10/3.
    depth = torch.ones(1, 1, 2, 2)
    expected = 1.0 - (1.1 ** -2)
    assert abs(float(mapper.depth_to_normalized(depth).mean()) - expected) < 1e-6


def test_unknown_curve_is_rejected():
    with pytest.raises(ValueError, match="Unknown depth curve"):
        DepthRGBMapper(curve="wan_flow")


def test_log_relative_roundtrip_is_median_relative():
    mapper = DepthRGBMapper(curve=LOG_RELATIVE_CURVE, log_alpha=DEFAULT_LOG_ALPHA)
    assert mapper.is_metric is False
    # Median 4 m; values 2 / 4 / 8 stay inside the unclipped log range.
    depth = torch.tensor([[[[2.0, 4.0], [4.0, 8.0]]]])
    rgb = mapper.depth_to_rgb(depth)
    recovered = mapper.rgb_to_depth(rgb)
    assert recovered.shape == depth.shape
    rel = recovered / recovered.median()
    expected = depth / depth.median()
    assert torch.allclose(rel, expected, atol=5e-3)


def test_log_relative_median_maps_to_alpha_ln2():
    mapper = DepthRGBMapper(curve=LOG_RELATIVE_CURVE, log_alpha=0.4)
    depth = torch.full((1, 1, 3, 3), 5.0)
    normalized = mapper.depth_to_normalized(depth)
    assert torch.allclose(normalized, torch.full_like(normalized, 0.4 * math.log(2.0)), atol=1e-6)


def test_decode_depth_rgb_respects_mapper_curve():
    mapper = DepthRGBMapper(curve=LOG_RELATIVE_CURVE)
    depth = torch.tensor([[[[1.0, 3.0], [3.0, 9.0]]]])
    rgb = mapper.depth_to_rgb(depth)
    decoded = decode_depth_rgb(rgb, mapper)
    assert torch.allclose(decoded / decoded.median(), depth / depth.median(), atol=5e-3)


def test_drop_oversized_gradients_is_off_by_default():
    assert drop_oversized_gradients(1e9, 0.0) is False
    assert drop_oversized_gradients(2.0, 1.5) is True
    assert drop_oversized_gradients(1.0, 1.5) is False


def test_mixed_dataset_reencodes_depth_when_curve_is_log_relative():
    from open_vision_banana.data.mixed import MixedVisionDataset

    paper = DepthRGBMapper()
    depth = torch.tensor([[[1.0, 4.0], [4.0, 16.0]]])
    target = paper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)

    class _DepthDS:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "image": torch.zeros(3, 2, 2),
                "target_rgb": target.clone(),
                "gt_depth": depth.clone(),
                "task": "depth",
            }

    mixed = MixedVisionDataset(
        vision_datasets=[_DepthDS()],
        vision_ratio=1.0,
        seed=0,
        depth_curve=LOG_RELATIVE_CURVE,
    )
    sample = mixed[0]
    alt = DepthRGBMapper(curve=LOG_RELATIVE_CURVE)
    expected = alt.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
    assert torch.allclose(sample["target_rgb"], expected, atol=1e-5)
    assert not torch.allclose(sample["target_rgb"], target, atol=1e-4)
