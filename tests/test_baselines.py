"""Tests for SOTA baseline comparison utilities."""

from open_vision_banana.evaluation.baselines import SOTA_BASELINES, format_sota_comparison


class TestSOTABaselines:
    def test_depth_baselines_exist(self):
        assert "delta_1.25" in SOTA_BASELINES["depth"]
        assert "Vision Banana" in SOTA_BASELINES["depth"]["delta_1.25"]

    def test_format_comparison(self):
        text = format_sota_comparison(
            "depth",
            {"delta_1.25": 0.90, "abs_rel": 0.05, "rmse": 0.30},
        )
        assert "SOTA Comparison" in text
        assert "delta_1.25" in text
        assert "0.9000" in text

    def test_unknown_task(self):
        text = format_sota_comparison("unknown", {})
        assert "No SOTA baselines" in text
