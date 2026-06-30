"""Evaluation sub-package — metrics for all vision tasks.

Provides:
- Depth metrics: AbsRel, SqRel, RMSE, RMSE_log, δ thresholds
- Normal metrics: Mean/Median Angular Error, Cosine Similarity
- Semantic metrics: mIoU, Pixel Accuracy
- Instance metrics: AP@0.5, AP@0.75, mAP@[0.5:0.95]
- Referring metrics: IoU, Precision@threshold
- Edge metrics: precision, recall, F1, ODS, OIS, AP
- Flow metrics: EPE, AE, F1
- Keypoint metrics: OKS, mAP, PCK
- Object detection metrics: mAP@[0.5:0.95], AP per class
- Generation preservation: FID-proxy, CLIP score, diversity

VisionEvaluator: Unified evaluation runner for all 10 tasks.
"""

from .depth import evaluate_depth
from .edge import evaluate_edge, evaluate_edge_dataset
from .evaluator import VisionEvaluator
from .flow import evaluate_flow
from .generation_preservation import evaluate_generation_preservation
from .instance import compute_iou_box, compute_iou_mask, evaluate_instance
from .keypoint import evaluate_keypoints
from .normal import evaluate_normals
from .object_detection import evaluate_object_detection
from .referring import evaluate_referring
from .semantic import evaluate_semantic

__all__ = [
    "evaluate_depth",
    "evaluate_normals",
    "evaluate_semantic",
    "evaluate_instance",
    "evaluate_referring",
    "evaluate_edge",
    "evaluate_edge_dataset",
    "evaluate_flow",
    "evaluate_keypoints",
    "evaluate_object_detection",
    "evaluate_generation_preservation",
    "compute_iou_mask",
    "compute_iou_box",
    "VisionEvaluator",
]
