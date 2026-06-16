"""Tests for evaluation metrics across all vision tasks."""

import pytest
import torch

from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.evaluation.instance import compute_iou_mask
from open_vision_banana.evaluation.normal import evaluate_normals
from open_vision_banana.evaluation.semantic import evaluate_semantic


class TestDepthMetrics:
    """Tests for depth estimation metrics."""

    def test_perfect_depth(self):
        pred = torch.tensor([[[[5.0, 10.0, 15.0]]]])
        gt = torch.tensor([[[[5.0, 10.0, 15.0]]]])
        metrics = evaluate_depth(pred, gt)
        assert metrics["AbsRel"] == pytest.approx(0.0, abs=1e-4)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-4)

    def test_depth_delta_thresholds(self):
        pred = torch.tensor([[[[1.0]]]])
        gt = torch.tensor([[[[1.2]]]])
        metrics = evaluate_depth(pred, gt)
        assert metrics["δ<1.25"] > 0.0


class TestNormalMetrics:
    """Tests for surface normal metrics."""

    def test_perfect_normals(self):
        normal = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
        metrics = evaluate_normals(normal, normal)
        assert metrics["Mean Angular Error (°)"] == pytest.approx(0.0, abs=0.1)
        assert metrics["Mean Cosine Similarity"] == pytest.approx(1.0, abs=1e-4)

    def test_opposite_normals(self):
        n1 = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
        n2 = torch.tensor([[[[0.0]], [[0.0]], [[-1.0]]]])
        metrics = evaluate_normals(n1, n2)
        assert metrics["Mean Angular Error (°)"] > 170.0


class TestSemanticMetrics:
    """Tests for semantic segmentation metrics."""

    def test_perfect_segmentation(self):
        pred = torch.zeros(1, 4, 4, dtype=torch.long)
        gt = torch.zeros(1, 4, 4, dtype=torch.long)
        metrics = evaluate_semantic(pred, gt, num_classes=2)
        assert metrics["mIoU"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["Pixel Accuracy"] == pytest.approx(1.0, abs=1e-4)


class TestInstanceMetrics:
    """Tests for instance segmentation metrics."""

    def test_iou_mask_perfect(self):
        mask1 = torch.zeros(4, 4, dtype=torch.bool)
        mask1[:2, :2] = True
        iou = compute_iou_mask(mask1, mask1)
        assert iou == pytest.approx(1.0, abs=1e-4)

    def test_iou_mask_no_overlap(self):
        mask1 = torch.zeros(4, 4, dtype=torch.bool)
        mask1[:2, :2] = True
        mask2 = torch.zeros(4, 4, dtype=torch.bool)
        mask2[2:, 2:] = True
        iou = compute_iou_mask(mask1, mask2)
        assert iou == pytest.approx(0.0, abs=1e-4)
