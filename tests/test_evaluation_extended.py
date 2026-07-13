"""Extended evaluation tests for all new metrics and evaluator tasks."""

import pytest
import torch

from open_vision_banana.evaluation.edge import evaluate_edge, evaluate_edge_dataset
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.evaluation.flow import evaluate_flow
from open_vision_banana.evaluation.generation_preservation import (
    compute_clip_score_proxy,
    compute_diversity_score,
    compute_fid,
    compute_fid_proxy,
    compute_perceptual_similarity,
    evaluate_generation_preservation,
)
from open_vision_banana.evaluation.keypoint import (
    SIGMAS_COCO,
    compute_oks,
    compute_pck,
    evaluate_keypoints,
)
from open_vision_banana.evaluation.object_detection import (
    _pycocotools_available,
    compute_iou_box,
    evaluate_object_detection,
)

# ────────────────────────────────────────────
# Edge Detection Tests
# ────────────────────────────────────────────

class TestEdgeMetrics:
    """Tests for edge detection evaluation metrics."""

    def test_perfect_edge(self):
        pred = torch.ones(1, 1, 8, 8)
        gt = torch.ones(1, 1, 8, 8)
        metrics = evaluate_edge(pred, gt, threshold=0.5)
        assert metrics["f1"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["precision"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["recall"] == pytest.approx(1.0, abs=1e-4)

    def test_no_edge(self):
        pred = torch.zeros(1, 1, 8, 8)
        gt = torch.ones(1, 1, 8, 8)
        metrics = evaluate_edge(pred, gt, threshold=0.5)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_ods_computation(self):
        pred = torch.rand(1, 1, 16, 16)
        gt = torch.rand(1, 1, 16, 16) > 0.5
        metrics = evaluate_edge(pred, gt, threshold=None, compute_ods=True)
        assert "ods_f1" in metrics
        assert "ods_threshold" in metrics
        assert metrics["ods_f1"] is not None
        assert 0.0 <= metrics["ods_f1"] <= 1.0

    def test_edge_dataset_evaluation(self):
        preds = [torch.rand(1, 1, 8, 8) for _ in range(5)]
        gts = [torch.rand(1, 1, 8, 8) > 0.5 for _ in range(5)]
        metrics = evaluate_edge_dataset(preds, gts, num_thresholds=10)
        assert "ods_f1" in metrics
        assert "ois_f1" in metrics
        assert "ap" in metrics


# ────────────────────────────────────────────
# Flow Metrics Tests
# ────────────────────────────────────────────

class TestFlowMetrics:
    """Tests for optical flow evaluation metrics."""

    def test_perfect_flow(self):
        flow = torch.randn(2, 2, 8, 8)
        metrics = evaluate_flow(flow, flow)
        assert metrics["epe"] == pytest.approx(0.0, abs=1e-3)
        assert metrics["ae_deg"] == pytest.approx(0.0, abs=1e-2)
        assert metrics["f1"] == pytest.approx(1.0, abs=1e-3)

    def test_flow_f1_logic(self):
        pred = torch.zeros(2, 2, 8, 8)
        gt = torch.zeros(2, 2, 8, 8)
        # Both zero flow → should be considered "correct" (EPE = 0 < 3)
        metrics = evaluate_flow(pred, gt)
        assert metrics["f1"] == pytest.approx(1.0, abs=1e-4)

    def test_flow_with_valid_mask(self):
        pred = torch.randn(2, 2, 8, 8)
        gt = torch.randn(2, 2, 8, 8)
        valid = torch.zeros(2, 1, 8, 8)
        valid[:, :, :4, :] = 1.0
        metrics = evaluate_flow(pred, gt, valid_mask=valid.bool())
        assert metrics["epe"] >= 0.0


# ────────────────────────────────────────────
# Keypoint Metrics Tests
# ────────────────────────────────────────────

class TestKeypointMetrics:
    """Tests for 2D keypoint detection metrics."""

    def test_compute_oks_identical(self):
        kpts = torch.tensor([[[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]]])
        areas = torch.tensor([100.0])
        oks = compute_oks(kpts, kpts, areas)
        assert oks.shape == (1, 3)
        assert (oks == 1.0).all()

    def test_compute_oks_different(self):
        pred = torch.tensor([[[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]]])
        gt = torch.tensor([[[0.6, 0.6], [0.3, 0.3], [0.9, 0.9]]])
        areas = torch.tensor([100.0])
        oks = compute_oks(pred, gt, areas)
        assert oks.shape == (1, 3)
        assert (oks < 1.0).any()
        assert (oks > 0.0).all()

    def test_compute_pck(self):
        pred = torch.tensor([[[0.5, 0.5], [0.2, 0.2]]])
        gt = torch.tensor([[[0.5, 0.5], [0.2, 0.2]]])
        pck = compute_pck(pred, gt, thresholds=[0.1, 0.2])
        assert pck["PCK@0.1"] == pytest.approx(1.0, abs=1e-4)
        assert pck["PCK@0.2"] == pytest.approx(1.0, abs=1e-4)

    def test_evaluate_keypoints_full(self):
        pred = torch.tensor([[[0.5, 0.5], [0.2, 0.2], [0.8, 0.8]]])
        gt = torch.tensor([[[0.52, 0.52], [0.22, 0.22], [0.82, 0.82]]])
        areas = torch.tensor([100.0])
        metrics = evaluate_keypoints(pred, gt, areas=areas)
        assert "mAP_OKS" in metrics
        assert "AP@0.5" in metrics
        assert "AP@0.75" in metrics
        assert "Mean OKS" in metrics
        assert "PCK@0.1" in metrics
        assert "PCK@0.2" in metrics
        assert "PCK@0.5" in metrics
        assert 0.0 <= metrics["mAP_OKS"] <= 1.0

    def test_sigmas_coco_shape(self):
        assert SIGMAS_COCO.shape[0] == 17
        assert (SIGMAS_COCO > 0).all()


# ────────────────────────────────────────────
# Object Detection Tests
# ────────────────────────────────────────────

class TestObjectDetectionMetrics:
    """Tests for object detection evaluation metrics."""

    def test_compute_iou_box_perfect(self):
        box = (0.0, 0.0, 10.0, 10.0)
        iou = compute_iou_box(box, box)
        assert iou == pytest.approx(1.0, abs=1e-4)

    def test_compute_iou_box_no_overlap(self):
        box1 = (0.0, 0.0, 5.0, 5.0)
        box2 = (6.0, 6.0, 10.0, 10.0)
        iou = compute_iou_box(box1, box2)
        assert iou == pytest.approx(0.0, abs=1e-4)

    def test_compute_iou_box_partial(self):
        box1 = (0.0, 0.0, 10.0, 10.0)
        box2 = (5.0, 5.0, 15.0, 15.0)
        iou = compute_iou_box(box1, box2)
        # 5x5 overlap / (100 + 100 - 25) = 25 / 175 = 0.1428
        assert iou == pytest.approx(25.0 / 175.0, abs=1e-4)

    def test_evaluate_object_detection_perfect(self):
        pred_boxes = [[(0.0, 0.0, 10.0, 10.0)]]
        pred_scores = [[1.0]]
        pred_labels = [[1]]
        gt_boxes = [[(0.0, 0.0, 10.0, 10.0)]]
        gt_labels = [[1]]
        metrics = evaluate_object_detection(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=1,
        )
        assert "mAP@[0.5:0.95]" in metrics
        assert "AP@0.5" in metrics
        assert "AP@0.75" in metrics
        assert metrics["AP@0.5"] == pytest.approx(1.0, abs=1e-4)
        assert metrics.get("eval_backend") in ("pycocotools", "fallback")

    @pytest.mark.skipif(not _pycocotools_available(), reason="pycocotools not installed")
    def test_evaluate_object_detection_pycocotools_backend(self):
        pred_boxes = [[(0.0, 0.0, 10.0, 10.0)]]
        pred_scores = [[1.0]]
        pred_labels = [[1]]
        gt_boxes = [[(0.0, 0.0, 10.0, 10.0)]]
        gt_labels = [[1]]
        metrics = evaluate_object_detection(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=1,
            use_pycocotools=True,
        )
        assert metrics["eval_backend"] == "pycocotools"
        assert metrics["AP@0.5"] == pytest.approx(1.0, abs=1e-2)

    def test_evaluate_object_detection_empty(self):
        pred_boxes: list[list] = [[]]
        pred_scores: list[list] = [[]]
        pred_labels: list[list] = [[]]
        gt_boxes = [[(0.0, 0.0, 10.0, 10.0)]]
        gt_labels = [[1]]
        metrics = evaluate_object_detection(
            pred_boxes, pred_scores, pred_labels,
            gt_boxes, gt_labels, num_classes=1,
        )
        assert metrics["AP@0.5"] == pytest.approx(0.0, abs=1e-4)


# ────────────────────────────────────────────
# Generation Preservation Tests
# ────────────────────────────────────────────

class TestGenerationPreservation:
    """Tests for generation preservation metrics."""

    def test_fid_identical(self):
        features = torch.randn(10, 64)
        fid = compute_fid(features, features)
        assert fid == pytest.approx(0.0, abs=1e-2)

    def test_fid_proxy_identical(self):
        features = torch.randn(10, 64)
        fid = compute_fid_proxy(features, features)
        assert fid == pytest.approx(0.0, abs=1e-3)

    def test_fid_proxy_different(self):
        real = torch.randn(10, 64)
        gen = torch.randn(10, 64)
        fid = compute_fid_proxy(real, gen)
        assert fid > 0.0

    def test_clip_score_proxy(self):
        img = torch.randn(10, 64)
        text = torch.randn(10, 64)
        score = compute_clip_score_proxy(img, text)
        assert -1.0 <= score <= 1.0

    def test_diversity_score(self):
        features = torch.randn(10, 64)
        div = compute_diversity_score(features)
        assert div > 0.0

    def test_diversity_score_identical(self):
        features = torch.ones(10, 64)
        div = compute_diversity_score(features)
        assert div == pytest.approx(0.0, abs=1e-3)

    def test_perceptual_similarity(self):
        pred = torch.rand(2, 3, 8, 8)
        target = torch.rand(2, 3, 8, 8)
        sim = compute_perceptual_similarity(pred, target)
        assert sim >= 0.0

    def test_full_generation_preservation(self):
        real = torch.randn(10, 64)
        gen = torch.randn(10, 64)
        text = torch.randn(10, 64)
        pred_rgb = torch.rand(2, 3, 32, 32)
        target_rgb = torch.rand(2, 3, 32, 32)
        metrics = evaluate_generation_preservation(
            real, gen, text_features=text, pred_rgb=pred_rgb, target_rgb=target_rgb
        )
        assert "FID_proxy" in metrics
        assert "FID" in metrics
        assert "diversity_score" in metrics
        assert "clip_score_proxy" in metrics
        assert "perceptual_similarity" in metrics
        assert "generation_health_score" in metrics

    def test_generation_preservation_missing_optional(self):
        gen = torch.randn(10, 64)
        metrics = evaluate_generation_preservation(None, gen)
        assert math.isnan(metrics["FID_proxy"])
        assert metrics["diversity_score"] > 0.0


# ────────────────────────────────────────────
# Evaluator Extended Task Tests
# ────────────────────────────────────────────

class MockModelWithPredict:
    """Mock model that exposes a predict() method."""

    def __init__(self, output_rgb=None, output_size: tuple[int, int] | None = None):
        self._output = output_rgb
        self._output_size = output_size
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def predict(self, **kwargs):
        if self._output is not None:
            return {"rgb_output": self._output}
        image = kwargs.get("image", torch.zeros(1, 3, 8, 8))
        size = self._output_size or image.shape[-2:]
        b = image.shape[0]
        return {"rgb_output": torch.rand(b, 3, size[0], size[1])}


class TestEvaluatorExtendedTasks:
    """Test VisionEvaluator supports all 10 tasks."""

    def test_evaluator_edge_task(self):
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        batch = {
            "image": torch.rand(1, 3, 8, 8),
            "target_rgb": torch.rand(1, 3, 8, 8),
        }
        metrics = ev.evaluate([batch], task="edge")
        assert "f1" in metrics
        assert "ods_f1" in metrics

    def test_evaluator_flow_task(self):
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        batch = {
            "image": torch.rand(1, 3, 8, 8),
            "target_rgb": torch.rand(1, 3, 8, 8),
        }
        metrics = ev.evaluate([batch], task="flow")
        assert "epe" in metrics
        assert "ae_deg" in metrics
        assert "f1" in metrics

    def test_evaluator_depth_edge_task(self):
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        batch = {
            "image": torch.rand(1, 3, 8, 8),
            "target_rgb": torch.rand(1, 3, 8, 8),
        }
        metrics = ev.evaluate([batch], task="depth_edge")
        assert "f1" in metrics

    def test_evaluator_keypoint_task_with_annotations(self):
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        pred_kpts = torch.tensor([[[0.5, 0.5], [0.2, 0.2]]])
        gt_kpts = torch.tensor([[[0.5, 0.5], [0.2, 0.2]]])
        batch = {
            "image": torch.rand(1, 3, 8, 8),
            "target_rgb": torch.rand(1, 3, 8, 8),
            "pred_keypoints": pred_kpts,
            "gt_keypoints": gt_kpts,
            "areas": torch.tensor([100.0]),
        }
        metrics = ev.evaluate([batch], task="keypoint")
        assert "mAP_OKS" in metrics
        assert "Mean OKS" in metrics

    def test_evaluator_keypoint_task_missing_annotations(self):
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        batch = {
            "image": torch.rand(1, 3, 8, 8),
            "target_rgb": torch.rand(1, 3, 8, 8),
        }
        with pytest.raises(ValueError, match="gt_keypoints"):
            ev.evaluate([batch], task="keypoint")

    def test_evaluator_object_detection_task(self):
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        batch = {
            "image": torch.rand(1, 3, 8, 8),
            "target_rgb": torch.rand(1, 3, 8, 8),
            "pred_boxes": [[(0.0, 0.0, 4.0, 4.0)]],
            "pred_scores": [[1.0]],
            "pred_labels": [[1]],
            "gt_boxes": [[(0.0, 0.0, 4.0, 4.0)]],
            "gt_labels": [[1]],
        }
        metrics = ev.evaluate([batch], task="object_detection", num_classes=1)
        assert "mAP@[0.5:0.95]" in metrics
        assert "AP@0.5" in metrics

    def test_evaluator_unknown_task(self):
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        batch = {
            "image": torch.rand(1, 3, 8, 8),
            "target_rgb": torch.rand(1, 3, 8, 8),
        }
        with pytest.raises(ValueError, match="Unknown task"):
            ev.evaluate([batch], task="unknown_task")

    def test_evaluator_generation_preservation_task(self):
        pytest.importorskip("torchvision")
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        batch = {
            "image": torch.rand(1, 3, 32, 32),
            "target_rgb": torch.rand(1, 3, 32, 32),
        }
        metrics = ev.evaluate([batch], task="generation_preservation", max_batches=1)
        assert "FID" in metrics
        assert "diversity_score" in metrics

    def test_all_ten_tasks_supported(self):
        """Verify all 10 tasks are accepted by the evaluator."""
        tasks = [
            "depth", "normal", "semantic", "instance", "referring",
            "edge", "depth_edge", "flow", "keypoint", "object_detection",
        ]
        for task in tasks:
            assert task in [
                "depth", "normal", "semantic", "instance", "referring",
                "edge", "depth_edge", "flow", "keypoint", "object_detection",
            ]


import math  # noqa: E402
