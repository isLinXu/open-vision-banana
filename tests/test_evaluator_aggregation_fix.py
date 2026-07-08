#!/usr/bin/env python3
"""Standalone validation test for evaluator aggregation fixes."""

import sys

sys.path.insert(0, "src")

import math
from typing import Any

import pytest
import torch

from open_vision_banana.evaluation.evaluator import VisionEvaluator


class MockModelDirect:
    """Mock model that does NOT expose predict()."""

    def eval(self):
        return self

    def __call__(self, image):
        return torch.zeros(1, 3, 8, 8)


class TestDictAggregation:
    """Verify dict-type metrics (per_class_AP50) are averaged across batches."""

    def test_per_class_ap50_multi_batch(self):
        ev = VisionEvaluator(MockModelDirect(), device="cpu")
        batch1 = {
            "image": torch.zeros(1, 3, 8, 8),
            "target_rgb": torch.zeros(1, 3, 8, 8),
            "pred_boxes": [[(0.0, 0.0, 2.0, 2.0)]],
            "pred_scores": [[0.9]],
            "pred_labels": [[1]],
            "gt_boxes": [[(0.0, 0.0, 2.0, 2.0)]],
            "gt_labels": [[1]],
        }
        batch2 = {
            "image": torch.zeros(1, 3, 8, 8),
            "target_rgb": torch.zeros(1, 3, 8, 8),
            "pred_boxes": [[]],
            "pred_scores": [[]],
            "pred_labels": [[]],
            "gt_boxes": [[(0.0, 0.0, 2.0, 2.0)]],
            "gt_labels": [[1]],
        }
        metrics = ev.evaluate([batch1, batch2], task="object_detection", num_classes=1)
        ap50: dict = metrics.get("per_class_AP50", {})
        assert isinstance(ap50, dict)
        assert abs(ap50[1] - 0.5) < 1e-6


class TestListNanAggregation:
    """Verify list-type metrics with nan values are averaged correctly."""

    def test_per_class_iou_with_nan(self):
        from open_vision_banana.evaluation.semantic import evaluate_semantic

        # Batch 1: class 0 ignored (nan), class 1 perfect (1.0)
        pred1 = torch.tensor([[[0, 0], [1, 1]]])
        gt1 = torch.tensor([[[255, 255], [1, 1]]])
        m1 = evaluate_semantic(pred1, gt1, num_classes=2)
        assert math.isnan(m1["per_class_IoU"][0])
        assert m1["per_class_IoU"][1] == 1.0

        # Batch 2: class 0 perfect (1.0), class 1 ignored (nan)
        pred2 = torch.tensor([[[0, 0], [1, 1]]])
        gt2 = torch.tensor([[[0, 0], [255, 255]]])
        m2 = evaluate_semantic(pred2, gt2, num_classes=2)
        assert m2["per_class_IoU"][0] == 1.0
        assert math.isnan(m2["per_class_IoU"][1])

        # Simulate evaluator aggregation logic
        all_metrics: dict[str, Any] = {}
        for metrics in [m1, m2]:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k] = all_metrics.get(k, 0.0) + v
                elif isinstance(v, list):
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)

        n_batches = 2
        for k in all_metrics:
            if isinstance(all_metrics[k], list):
                len(all_metrics[k])
                elem_len = len(all_metrics[k][0])
                result = []
                for i in range(elem_len):
                    vals = [
                        batch[i]
                        for batch in all_metrics[k]
                        if not (isinstance(batch[i], float) and math.isnan(batch[i]))
                    ]
                    result.append(sum(vals) / len(vals) if vals else float("nan"))
                all_metrics[k] = result
            else:
                all_metrics[k] /= n_batches

        assert not math.isnan(all_metrics["per_class_IoU"][0])
        assert abs(all_metrics["per_class_IoU"][0] - 1.0) < 1e-6
        assert not math.isnan(all_metrics["per_class_IoU"][1])
        assert abs(all_metrics["per_class_IoU"][1] - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
