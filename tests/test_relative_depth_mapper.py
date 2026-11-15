import numpy as np
import pytest
import torch

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.core.relative_depth_mapper import (
    DEPTH_REL_SENTINEL,
    RelativeDepthMapper,
    align_inverse_depth_affine,
    metric_target_inverse_depth,
)


def _depth(h=24, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return 0.5 + 9.0 * torch.rand(h, w, generator=g)


def test_encode_is_grey_near_bright_and_minmax():
    mapper = RelativeDepthMapper()
    depth = _depth()
    valid = torch.ones_like(depth, dtype=torch.bool)
    rgb = mapper.encode_depth(depth, valid)
    assert rgb.shape == (3, *depth.shape)
    assert torch.equal(rgb[0], rgb[1]) and torch.equal(rgb[1], rgb[2])
    near, far = depth.argmin(), depth.argmax()
    assert rgb[0].flatten()[near] == pytest.approx(1.0)
    assert rgb[0].flatten()[far] == pytest.approx(0.0)


def test_uint8_round_trip_within_half_level():
    mapper = RelativeDepthMapper()
    depth = _depth()
    valid = torch.rand(depth.shape, generator=torch.Generator().manual_seed(1)) > 0.1
    inverse = torch.where(valid, 1.0 / depth, torch.zeros_like(depth))
    normalized = mapper.normalize_inverse_depth(inverse, valid)
    stored = (mapper.encode(normalized, valid) * 255).round().to(torch.uint8)
    decoded = mapper.decode(stored.float() / 255)
    assert (decoded - normalized)[valid].abs().max() <= 0.5 / 255 + 1e-6
    sentinel = mapper.sentinel_mask(stored)
    assert torch.equal(sentinel, ~valid)
    assert mapper.off_contract(stored.float() / 255).eq(~valid).all()
    assert tuple(stored[:, ~valid][:, 0].tolist()) == DEPTH_REL_SENTINEL


def test_normalize_degenerate_and_batched():
    mapper = RelativeDepthMapper()
    inverse = torch.stack([torch.full((4, 4), 2.0), torch.arange(16.0).reshape(4, 4)])
    valid = torch.ones_like(inverse, dtype=torch.bool)
    valid[1, 0, 0] = False
    out = mapper.normalize_inverse_depth(inverse, valid)
    assert torch.allclose(out[0], torch.full((4, 4), 0.5))
    assert out[1, 0, 0] == 0 and out[1, 0, 1] == 0 and out[1, 3, 3] == 1
    empty = mapper.normalize_inverse_depth(inverse[:1], torch.zeros_like(valid[:1]))
    assert empty.abs().sum() == 0


def test_metric_target_decode_marks_holes_and_mask():
    depth = _depth(16, 16)
    depth[0, :4] = 0.0  # stored hole
    rgb = (DepthRGBMapper("cube_edge").depth_to_rgb(depth[None, None])[0] * 255).round().to(torch.uint8)
    target = rgb.permute(1, 2, 0).numpy()
    mask = np.ones((16, 16), np.uint8)
    mask[5, 5] = 0
    inverse, valid = metric_target_inverse_depth(target, mask)
    assert not valid[0, :4].any() and not valid[5, 5]
    assert valid.sum() == 256 - 5
    assert np.allclose(inverse[valid], 1.0 / depth.numpy()[valid], rtol=0.05)
    with pytest.raises(ValueError):
        metric_target_inverse_depth(target, np.ones((8, 8), np.uint8))


def test_affine_alignment_recovers_depth():
    gt = _depth()[None, None]
    valid = torch.ones_like(gt, dtype=torch.bool)
    prediction = 0.3 * (1.0 / gt) + 0.05  # affine in inverse depth
    aligned = align_inverse_depth_affine(prediction, gt, valid)
    assert torch.allclose(aligned, gt, rtol=1e-4)
    single = valid.clone().zero_()
    single[..., 0, 0] = True
    scale_only = align_inverse_depth_affine(prediction, gt, single)
    assert torch.isfinite(scale_only).all()
    none = align_inverse_depth_affine(prediction, gt, torch.zeros_like(valid), max_depth=10.0)
    assert torch.isfinite(none).all()
    with pytest.raises(ValueError):
        align_inverse_depth_affine(prediction, gt[..., :3], valid)


def test_evaluator_scores_relative_inverse_after_affine_fit():
    from types import SimpleNamespace

    from open_vision_banana.evaluation.benchmark import DepthAccumulator
    from open_vision_banana.evaluation.evaluator import VisionEvaluator

    gt = _depth(16, 16)[None, None]
    valid = torch.ones(1, 16, 16, dtype=torch.bool)
    mapper = RelativeDepthMapper()
    pred_rgb = mapper.encode_depth(gt[:, 0], valid)  # perfect relative prediction, [1, 3, H, W]
    fake = SimpleNamespace(device="cpu")
    acc = DepthAccumulator()
    batch = {"gt_depth_native": gt, "valid_mask_native": valid}
    metrics = VisionEvaluator._relative_inverse_depth_metrics(fake, pred_rgb, batch, "diode", True, acc)
    assert metrics["δ<1.25"] > 0.99
    plain = VisionEvaluator._relative_inverse_depth_metrics(
        fake, pred_rgb, {"gt_depth": gt, "valid_mask": valid}, "nyu", False, None
    )
    assert plain["δ<1.25"] > 0.99 and plain["AbsRel"] < 0.01
    with pytest.raises(ValueError):
        VisionEvaluator._relative_inverse_depth_metrics(fake, pred_rgb, {}, "nyu", True, None)
    with pytest.raises(ValueError):
        VisionEvaluator._relative_inverse_depth_metrics(fake, pred_rgb, {}, "nyu", False, None)


def test_suite_command_passes_depth_curve():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/run_external_eval_suite.py"
    spec = importlib.util.spec_from_file_location("suite", path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    config = {"data_roots": {"nyu": "/d"}, "reasonseg_annotation_mode": "x"}
    args = suite.command(config, {"name": "m", "depth_curve": "relative_inverse"}, "nyu", Path("o.json"))
    assert args[args.index("--depth-curve") + 1] == "relative_inverse"
    default = suite.command(config, {"name": "m"}, "nyu", Path("o.json"))
    assert default[default.index("--depth-curve") + 1] == "metric_power"
