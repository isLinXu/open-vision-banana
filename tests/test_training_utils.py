"""Tests for training utilities and evaluation enhancements."""

import pytest
import torch
import torch.nn as nn

from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.evaluation.referring import evaluate_referring
from open_vision_banana.training.utils import DDPWrapper, EMAModel, VisionAugmentation

# ──────────────────────────────────────────────────────────────
# EMAModel Tests
# ──────────────────────────────────────────────────────────────

class TestEMAModel:
    """Tests for Exponential Moving Average."""

    def test_shadow_tracks_online(self):
        model = nn.Linear(4, 2)
        ema = EMAModel(model, decay=0.9, warmup_steps=0)

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)

        for _ in range(50):
            ema.update(model)

        for name, _param in model.named_parameters():
            shadow_val = ema.shadow_params[name]
            assert abs(shadow_val.mean().item() - 1.0) < 0.1

    def test_apply_and_restore(self):
        model = nn.Linear(4, 2)
        ema = EMAModel(model, decay=0.99, warmup_steps=0)

        original_params = {n: p.data.clone() for n, p in model.named_parameters()}
        backup = ema.apply_shadow(model)
        ema.restore_shadow(model, backup)

        for name, param in model.named_parameters():
            assert torch.allclose(param.data, original_params[name])

    def test_delayed_update(self):
        model = nn.Linear(4, 2)
        ema = EMAModel(model, decay=0.9, warmup_steps=0, update_after_step=5)

        initial_shadow = {n: p.clone() for n, p in ema.shadow_params.items()}

        with torch.no_grad():
            for p in model.parameters():
                p.fill_(5.0)

        for _ in range(4):
            ema.update(model)

        for name in ema.shadow_params:
            assert torch.allclose(ema.shadow_params[name], initial_shadow[name])


# ──────────────────────────────────────────────────────────────
# VisionAugmentation Tests
# ──────────────────────────────────────────────────────────────

class TestVisionAugmentation:
    """Tests for task-aware augmentation."""

    def test_output_shapes_preserved(self):
        aug = VisionAugmentation(resolution=64)
        image = torch.randn(3, 64, 64)
        target = torch.rand(3, 64, 64)
        aug_img, aug_tgt = aug(image, target)
        assert aug_img.shape == image.shape
        assert aug_tgt.shape == target.shape

    def test_geometric_consistency(self):
        aug = VisionAugmentation(resolution=32, hflip_prob=1.0, crop_scale=(1.0, 1.0))
        image = torch.randn(3, 32, 32)
        target = torch.rand(3, 32, 32)
        aug_img, aug_tgt = aug(image, target)
        assert torch.allclose(aug_img, torch.flip(image, [-1]))
        assert torch.allclose(aug_tgt, torch.flip(target, [-1]))

    def test_no_augmentation_identity(self):
        aug = VisionAugmentation(resolution=32, hflip_prob=0.0, crop_scale=(1.0, 1.0))
        image = torch.randn(3, 32, 32)
        target = torch.rand(3, 32, 32)
        aug_img, aug_tgt = aug(image, target)
        assert torch.allclose(aug_img, image)
        assert torch.allclose(aug_tgt, target)


# ──────────────────────────────────────────────────────────────
# Referring Evaluation Tests
# ──────────────────────────────────────────────────────────────

class TestReferringMetrics:
    """Tests for referring expression segmentation metrics."""

    def test_perfect_prediction(self):
        pred = torch.ones(2, 16, 16, dtype=torch.bool)
        gt = torch.ones(2, 16, 16, dtype=torch.bool)
        metrics = evaluate_referring(pred, gt)
        assert metrics["IoU"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["Precision@0.5"] == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        pred = torch.zeros(2, 16, 16, dtype=torch.bool)
        pred[:, :8, :] = True
        gt = torch.zeros(2, 16, 16, dtype=torch.bool)
        gt[:, 8:, :] = True
        metrics = evaluate_referring(pred, gt)
        assert metrics["IoU"] == pytest.approx(0.0, abs=1e-4)

    def test_partial_overlap(self):
        pred = torch.zeros(2, 16, 16, dtype=torch.bool)
        pred[:, :12, :] = True
        gt = torch.zeros(2, 16, 16, dtype=torch.bool)
        gt[:, 4:, :] = True
        metrics = evaluate_referring(pred, gt)
        assert 0.0 < metrics["IoU"] < 1.0


# ──────────────────────────────────────────────────────────────
# SqRel Metric Fix Verification
# ──────────────────────────────────────────────────────────────

class TestSqRelMetric:
    """Verify SqRel is computed correctly (not as AbsRel copy)."""

    def test_sqrel_different_from_absrel(self):
        pred2 = torch.tensor([[[[3.0]]]])
        gt2 = torch.tensor([[[[1.0]]]])
        metrics2 = evaluate_depth(pred2, gt2)
        assert metrics2["SqRel"] > metrics2["AbsRel"], \
            "SqRel should be larger than AbsRel for errors > 1"


# ──────────────────────────────────────────────────────────────
# DDPWrapper Tests
# ──────────────────────────────────────────────────────────────

class TestDDPWrapper:
    """Tests for DDP wrapper (single-process mode)."""

    def test_single_process_passthrough(self):
        ddp = DDPWrapper(world_size=1, rank=0)
        model = nn.Linear(4, 2)
        wrapped = ddp.wrap(model)
        assert wrapped is model

    def test_is_main(self):
        ddp = DDPWrapper(world_size=1, rank=0)
        assert ddp.is_main() is True

        ddp2 = DDPWrapper(world_size=4, rank=2)
        assert ddp2.is_main() is False
