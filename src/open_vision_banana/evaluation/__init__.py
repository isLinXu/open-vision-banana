"""Evaluation sub-package — metrics for all vision tasks.

Provides:
- Depth metrics: AbsRel, SqRel, RMSE, RMSE_log, δ thresholds
- Normal metrics: Mean/Median Angular Error, Cosine Similarity
- Semantic metrics: mIoU, Pixel Accuracy
- Instance metrics: AP@0.5, AP@0.75, mAP@[0.5:0.95]
- Referring metrics: IoU, Precision@threshold
- VisionEvaluator: Unified evaluation runner
"""

from .depth import evaluate_depth
from .evaluator import VisionEvaluator
from .instance import compute_iou_box, compute_iou_mask, evaluate_instance
from .normal import evaluate_normals
from .referring import evaluate_referring
from .semantic import evaluate_semantic

__all__ = [
    "evaluate_depth",
    "evaluate_normals",
    "evaluate_semantic",
    "evaluate_instance",
    "evaluate_referring",
    "compute_iou_mask",
    "compute_iou_box",
    "VisionEvaluator",
]
