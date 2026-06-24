"""Tests for evaluation/evaluator.py — VisionEvaluator class."""

import pytest
import torch

from open_vision_banana.evaluation.evaluator import VisionEvaluator

# ──────────────────────────────────────────────────────────────────
# Mock model helpers
# ──────────────────────────────────────────────────────────────────


class MockModelWithPredict:
    """Mock model that exposes a predict() method."""

    def __init__(self, output_rgb=None):
        self._output = output_rgb
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def predict(self, **kwargs):
        if self._output is None:
            # Return a default RGB tensor
            B = kwargs.get("image", torch.zeros(1, 3, 8, 8)).shape[0]
            return {"rgb_output": torch.rand(B, 3, 8, 8)}
        return {"rgb_output": self._output}


class MockModelDirect:
    """Mock model that does NOT expose predict() — uses direct call."""

    def __init__(self, output_rgb=None):
        self._output = output_rgb

    def eval(self):
        return self

    def __call__(self, image):
        if self._output is None:
            B = image.shape[0]
            return torch.rand(B, 3, 8, 8)
        return self._output


# ──────────────────────────────────────────────────────────────────
# Init tests
# ──────────────────────────────────────────────────────────────────


class TestVisionEvaluatorInit:
    """Test VisionEvaluator initialization."""

    def test_init_with_predict_model(self):
        """Should detect predict() on model."""
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        assert ev._has_predict is True

    def test_init_without_predict_model(self):
        """Should detect absence of predict() and warn."""
        model = MockModelDirect()
        ev = VisionEvaluator(model, device="cpu")
        assert ev._has_predict is False

    def test_device_stored(self):
        """Device should be stored."""
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        assert ev.device == "cpu"


# ──────────────────────────────────────────────────────────────────
# Evaluate tests — depth task
# ──────────────────────────────────────────────────────────────────


class TestEvaluateDepth:
    """Test evaluate() for depth estimation task."""

    def test_depth_with_predict_model(self):
        """Should use predict() path for depth task."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 8, 8) * 5 + 0.5
        target_rgb = mapper.depth_to_rgb(depth)
        pred_rgb = mapper.depth_to_rgb(depth + 0.1)

        model = MockModelWithPredict(output_rgb=pred_rgb)
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": target_rgb}]
        metrics = ev.evaluate(dataloader, task="depth")
        assert "AbsRel" in metrics
        assert metrics["AbsRel"] >= 0.0

    def test_depth_with_direct_model(self):
        """Should use direct call fallback for depth task."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 8, 8) * 5 + 0.5
        target_rgb = mapper.depth_to_rgb(depth)
        pred_rgb = mapper.depth_to_rgb(depth)

        model = MockModelDirect(output_rgb=pred_rgb)
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": target_rgb}]
        metrics = ev.evaluate(dataloader, task="depth")
        assert "AbsRel" in metrics
        assert metrics["AbsRel"] == pytest.approx(0.0, abs=1e-3)

    def test_depth_perfect_prediction(self):
        """Perfect prediction should yield zero error."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 8, 8) * 5 + 0.5
        rgb = mapper.depth_to_rgb(depth)

        model = MockModelWithPredict(output_rgb=rgb.clone())
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb}]
        metrics = ev.evaluate(dataloader, task="depth")
        assert metrics["AbsRel"] == pytest.approx(0.0, abs=1e-3)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-3)


# ──────────────────────────────────────────────────────────────────
# Evaluate tests — normal task
# ──────────────────────────────────────────────────────────────────


class TestEvaluateNormal:
    """Test evaluate() for normal estimation task."""

    def test_normal_with_predict_model(self):
        """Should work for normal task via predict()."""
        from open_vision_banana.core.normal_mapper import NormalRGBMapper

        mapper = NormalRGBMapper()
        normal = torch.randn(1, 3, 8, 8)
        normal = normal / normal.norm(dim=1, keepdim=True)
        target_rgb = mapper.normal_to_rgb(normal)
        pred_rgb = mapper.normal_to_rgb(normal)

        model = MockModelWithPredict(output_rgb=pred_rgb)
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": target_rgb}]
        metrics = ev.evaluate(dataloader, task="normal")
        assert "Mean Angular Error (°)" in metrics
        assert metrics["Mean Angular Error (°)"] < 5.0  # Should be close


# ──────────────────────────────────────────────────────────────────
# Evaluate tests — referring task
# ──────────────────────────────────────────────────────────────────


class TestEvaluateReferring:
    """Test evaluate() for referring segmentation task."""

    def test_referring_perfect(self):
        """Perfect prediction should yield high IoU."""
        mask = torch.zeros(1, 3, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0

        model = MockModelWithPredict(output_rgb=mask.clone())
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": mask}]
        metrics = ev.evaluate(dataloader, task="referring", expression="a cat")
        assert "IoU" in metrics
        assert metrics["IoU"] > 0.9

    def test_referring_requires_expression(self):
        """Should raise when expression is None for referring with predict model."""
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": torch.rand(1, 3, 8, 8)}]
        with pytest.raises(ValueError, match="expression required"):
            ev.evaluate(dataloader, task="referring")


# ──────────────────────────────────────────────────────────────────
# Evaluate tests — instance task
# ──────────────────────────────────────────────────────────────────


class TestEvaluateInstance:
    """Test evaluate() for instance segmentation task."""

    def test_instance_with_predict_model(self):
        """Should work for instance task via predict()."""
        rgb = torch.rand(1, 3, 8, 8)

        model = MockModelWithPredict(output_rgb=rgb.clone())
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb}]
        metrics = ev.evaluate(dataloader, task="instance", object_name="cat")
        assert "AP@0.5" in metrics


# ──────────────────────────────────────────────────────────────────
# Evaluate tests — semantic task
# ──────────────────────────────────────────────────────────────────


class TestEvaluateSemantic:
    """Test evaluate() for semantic segmentation task."""

    def test_semantic_with_class_color_map(self):
        """Should work with class_color_map provided."""
        rgb = torch.rand(1, 3, 8, 8)
        cmap = {"road": (128, 64, 128), "car": (0, 0, 142)}

        model = MockModelWithPredict(output_rgb=rgb.clone())
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb}]
        metrics = ev.evaluate(dataloader, task="semantic", class_color_map=cmap)
        assert "mIoU" in metrics

    def test_semantic_missing_color_map_raises(self):
        """Should raise when class_color_map is missing."""
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": torch.rand(1, 3, 8, 8)}]
        with pytest.raises(ValueError, match="class_color_map required"):
            ev.evaluate(dataloader, task="semantic")

    def test_semantic_color_map_from_batch(self):
        """Should read class_color_map from batch dict."""
        rgb = torch.rand(1, 3, 8, 8)
        cmap = {"road": (128, 64, 128), "car": (0, 0, 142)}

        model = MockModelWithPredict(output_rgb=rgb.clone())
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb, "class_color_map": cmap}]
        metrics = ev.evaluate(dataloader, task="semantic")
        assert "mIoU" in metrics


# ──────────────────────────────────────────────────────────────────
# Evaluate tests — edge cases
# ──────────────────────────────────────────────────────────────────


class TestEvaluateEdgeCases:
    """Test edge cases in evaluate()."""

    def test_empty_dataloader(self):
        """Empty dataloader should return empty dict."""
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")
        metrics = ev.evaluate([], task="depth")
        assert metrics == {}

    def test_max_batches(self):
        """max_batches should limit evaluation."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 8, 8) * 5 + 0.5
        rgb = mapper.depth_to_rgb(depth)

        model = MockModelWithPredict(output_rgb=rgb.clone())
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [
            {"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb},
            {"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb},
            {"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb},
        ]
        metrics = ev.evaluate(dataloader, task="depth", max_batches=2)
        assert "AbsRel" in metrics

    def test_unknown_task_raises(self):
        """Unknown task should raise ValueError."""
        model = MockModelWithPredict()
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": torch.rand(1, 3, 8, 8)}]
        with pytest.raises(ValueError, match="Unknown task"):
            ev.evaluate(dataloader, task="unknown_task")

    def test_multiple_batches_averaged(self):
        """Metrics should be averaged across batches."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth1 = torch.rand(1, 1, 8, 8) * 5 + 0.5
        depth2 = torch.rand(1, 1, 8, 8) * 5 + 0.5
        rgb1 = mapper.depth_to_rgb(depth1)
        rgb2 = mapper.depth_to_rgb(depth2)

        model = MockModelWithPredict(output_rgb=rgb1.clone())
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [
            {"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb1},
            {"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb2},
        ]
        metrics = ev.evaluate(dataloader, task="depth")
        assert "AbsRel" in metrics
        # The average of batch1 (zero error) and batch2 (some error) should be between them
        assert metrics["AbsRel"] >= 0.0


# ──────────────────────────────────────────────────────────────────
# evaluate_batch tests
# ──────────────────────────────────────────────────────────────────


class TestEvaluateBatch:
    """Test evaluate_batch() convenience method."""

    def test_depth_batch(self):
        """evaluate_batch should work for depth task."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth = torch.rand(2, 1, 8, 8) * 5 + 0.5
        target_rgb = mapper.depth_to_rgb(depth)
        pred_rgb = mapper.depth_to_rgb(depth)

        model = MockModelWithPredict(output_rgb=pred_rgb)
        ev = VisionEvaluator(model, device="cpu")

        image = torch.rand(2, 3, 8, 8)
        metrics = ev.evaluate_batch(image, target_rgb, task="depth")
        assert "AbsRel" in metrics

    def test_referring_batch(self):
        """evaluate_batch should work for referring task."""
        mask = torch.zeros(2, 3, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0

        model = MockModelWithPredict(output_rgb=mask.clone())
        ev = VisionEvaluator(model, device="cpu")

        image = torch.rand(2, 3, 8, 8)
        metrics = ev.evaluate_batch(image, mask, task="referring", expression="a cat")
        assert "IoU" in metrics


# ──────────────────────────────────────────────────────────────────
# _extract_rgb_from_result tests
# ──────────────────────────────────────────────────────────────────


class TestExtractRgbFromResult:
    """Test _extract_rgb_from_result static method."""

    def test_extract_visualization(self):
        """Should return 'visualization' key if present."""
        result = {"visualization": torch.rand(1, 3, 8, 8)}
        rgb = VisionEvaluator._extract_rgb_from_result(result, "depth", torch.rand(1, 3, 8, 8))
        assert rgb.shape == (1, 3, 8, 8)

    def test_extract_depth(self):
        """Should re-encode depth to RGB for depth task."""

        depth = torch.rand(1, 1, 8, 8) * 5 + 0.5
        result = {"depth": depth}
        image = torch.rand(1, 3, 8, 8)
        rgb = VisionEvaluator._extract_rgb_from_result(result, "depth", image)
        assert rgb.shape == (1, 3, 8, 8)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_extract_normal(self):
        """Should re-encode normal to RGB for normal task."""
        normal = torch.randn(1, 3, 8, 8)
        normal = normal / normal.norm(dim=1, keepdim=True)
        result = {"normal": normal}
        image = torch.rand(1, 3, 8, 8)
        rgb = VisionEvaluator._extract_rgb_from_result(result, "normal", image)
        assert rgb.shape == (1, 3, 8, 8)

    def test_extract_mask(self):
        """Should expand mask to 3-channel for segmentation tasks."""
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        result = {"mask": mask}
        image = torch.rand(1, 3, 8, 8)
        rgb = VisionEvaluator._extract_rgb_from_result(result, "instance", image)
        assert rgb.shape == (1, 3, 8, 8)

    def test_extract_fallback_zero(self):
        """Should return zero tensor when no known keys are present."""
        result = {"unknown_key": "value"}
        image = torch.rand(1, 3, 8, 8)
        rgb = VisionEvaluator._extract_rgb_from_result(result, "depth", image)
        assert rgb.shape == (1, 3, 8, 8)
        assert rgb.sum() == 0.0


# ──────────────────────────────────────────────────────────────────
# Direct model fallback tests
# ──────────────────────────────────────────────────────────────────


class TestDirectModelFallback:
    """Test the direct model(image) fallback path."""

    def test_dict_output_from_direct_model(self):
        """Should handle dict output from direct model call."""
        from open_vision_banana.core.depth_mapper import DepthRGBMapper

        mapper = DepthRGBMapper()
        depth = torch.rand(1, 1, 8, 8) * 5 + 0.5
        rgb = mapper.depth_to_rgb(depth)

        class DictModel:
            def eval(self):
                return self

            def __call__(self, image):
                return {"rgb_output": rgb.clone()}

        model = DictModel()
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb}]
        metrics = ev.evaluate(dataloader, task="depth")
        assert "AbsRel" in metrics

    def test_dict_output_first_value_fallback(self):
        """Should use first value when rgb_output key is missing."""
        rgb = torch.rand(1, 3, 8, 8)

        class DictModel:
            def eval(self):
                return self

            def __call__(self, image):
                return {"some_key": rgb.clone()}

        model = DictModel()
        ev = VisionEvaluator(model, device="cpu")

        dataloader = [{"image": torch.rand(1, 3, 8, 8), "target_rgb": rgb}]
        metrics = ev.evaluate(dataloader, task="depth")
        assert "AbsRel" in metrics
