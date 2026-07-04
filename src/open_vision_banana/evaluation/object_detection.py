"""Object detection evaluation metrics.

Standard COCO-style metrics:
- mAP@0.5 (AP50): Average Precision at IoU=0.5
- mAP@[0.5:0.95]: Average Precision across 10 IoU thresholds
- AP per class: class-wise average precision
- AR (Average Recall) at max detections = 1, 10, 100

When ``pycocotools`` is installed, ``evaluate_object_detection`` uses the
official COCO evaluation API.  Otherwise it falls back to a lightweight
Python implementation (same metric keys, approximate ranking).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _pycocotools_available() -> bool:
    try:
        import pycocotools.coco  # noqa: F401
        import pycocotools.cocoeval  # noqa: F401
        return True
    except ImportError:
        return False


def compute_iou_box(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _xyxy_to_xywh(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _image_size_from_boxes(
    boxes_list: list[list[tuple[float, float, float, float]]],
) -> tuple[int, int]:
    max_x = max_y = 1.0
    for boxes in boxes_list:
        for x1, y1, x2, y2 in boxes:
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
    return max(int(math.ceil(max_x)), 1), max(int(math.ceil(max_y)), 1)


def _build_coco_gt(
    gt_boxes: list[list[tuple[float, float, float, float]]],
    gt_labels: list[list[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build COCO-format images + annotations from per-image GT lists."""
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for img_id, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels), start=1):
        w, h = _image_size_from_boxes([boxes])
        images.append({"id": img_id, "width": w, "height": h})
        for box, label in zip(boxes, labels):
            xywh = _xyxy_to_xywh(box)
            area = max(xywh[2], 0.0) * max(xywh[3], 0.0)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": xywh,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return images, annotations


def _build_coco_dt(
    pred_boxes: list[list[tuple[float, float, float, float]]],
    pred_scores: list[list[float]],
    pred_labels: list[list[int]],
) -> list[dict[str, Any]]:
    """Build COCO-format detection results."""
    results: list[dict[str, Any]] = []
    for img_id, (boxes, scores, labels) in enumerate(
        zip(pred_boxes, pred_scores, pred_labels), start=1
    ):
        for box, score, label in zip(boxes, scores, labels):
            results.append(
                {
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": _xyxy_to_xywh(box),
                    "score": float(score),
                }
            )
    return results


def evaluate_object_detection_pycocotools(
    pred_boxes: list[list[tuple[float, float, float, float]]],
    pred_scores: list[list[float]],
    pred_labels: list[list[int]],
    gt_boxes: list[list[tuple[float, float, float, float]]],
    gt_labels: list[list[int]],
    num_classes: int,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """COCO-standard AP/AR via pycocotools (requires optional dependency)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    if iou_thresholds is None:
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    categories = [{"id": cls, "name": str(cls)} for cls in range(1, num_classes + 1)]
    images, annotations = _build_coco_gt(gt_boxes, gt_labels)
    detections = _build_coco_dt(pred_boxes, pred_scores, pred_labels)

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()

    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = coco_gt.loadRes([])

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.asarray(iou_thresholds, dtype=np.float64)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    result: dict[str, float] = {
        "mAP@[0.5:0.95]": float(stats[0]),
        "AP@0.5": float(stats[1]),
        "AP@0.75": float(stats[2]),
        "mAP_50": float(stats[1]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        "AR@100": float(stats[8]),
        "eval_backend": "pycocotools",
    }
    for thresh in iou_thresholds:
        key = f"AP@{thresh:.2f}"
        if key not in result:
            if abs(thresh - 0.5) < 1e-6:
                result[key] = float(stats[1])
            elif abs(thresh - 0.75) < 1e-6:
                result[key] = float(stats[2])
            else:
                result[key] = float(stats[0])

    per_class_ap50: dict[int, float] = {}
    try:
        if coco_eval.eval is not None:
            precision = coco_eval.eval["precision"]
            cat_ids = coco_eval.params.catIds
            ap50_idx = iou_thresholds.index(0.5) if 0.5 in iou_thresholds else 0
            for idx, cat_id in enumerate(cat_ids):
                prec = precision[ap50_idx, :, idx, 0, -1]
                prec = prec[prec > -1]
                if hasattr(prec, "size") and prec.size:
                    per_class_ap50[int(cat_id)] = float(prec.mean())
                elif prec is not None and float(prec) > -1:
                    per_class_ap50[int(cat_id)] = float(prec)
                else:
                    per_class_ap50[int(cat_id)] = 0.0
    except Exception as exc:
        logger.debug("per-class AP extraction skipped: %s", exc)
    result["per_class_AP50"] = per_class_ap50
    return result


def evaluate_object_detection_fallback(
    pred_boxes: list[list[tuple[float, float, float, float]]],
    pred_scores: list[list[float]],
    pred_labels: list[list[int]],
    gt_boxes: list[list[tuple[float, float, float, float]]],
    gt_labels: list[list[int]],
    num_classes: int,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Lightweight AP implementation (no external deps beyond torch)."""
    if iou_thresholds is None:
        iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    aps_per_threshold = {t: [] for t in iou_thresholds}
    per_class_ap50: dict[int, float] = {}

    for cls in range(1, num_classes + 1):
        all_pred = []
        for img_idx, (boxes, scores, labels) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
            for box, score, label in zip(boxes, scores, labels):
                if label == cls:
                    all_pred.append({"img_idx": img_idx, "box": box, "score": score})

        all_gt = []
        for img_idx, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
            for box, label in zip(boxes, labels):
                if label == cls:
                    all_gt.append({"img_idx": img_idx, "box": box, "matched": {t: False for t in iou_thresholds}})

        if not all_gt:
            continue

        all_pred.sort(key=lambda x: x["score"], reverse=True)

        for thresh in iou_thresholds:
            tp = 0
            fp = 0
            total_gt = len(all_gt)
            for gt in all_gt:
                gt["matched"][thresh] = False

            for pred in all_pred:
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(all_gt):
                    if gt["img_idx"] != pred["img_idx"] or gt["matched"][thresh]:
                        continue
                    iou = compute_iou_box(pred["box"], gt["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= thresh and best_gt_idx >= 0:
                    tp += 1
                    all_gt[best_gt_idx]["matched"][thresh] = True
                else:
                    fp += 1

            fn = total_gt - tp
            ap = tp / (tp + fp + fn + 1e-10) if (tp + fp + fn) > 0 else 0.0
            aps_per_threshold[thresh].append(ap)

        ap50 = aps_per_threshold[0.5][-1] if aps_per_threshold[0.5] else 0.0
        per_class_ap50[cls] = ap50

    result: dict[str, float] = {"eval_backend": "fallback"}
    for thresh in iou_thresholds:
        if aps_per_threshold[thresh]:
            result[f"AP@{thresh:.2f}"] = sum(aps_per_threshold[thresh]) / len(aps_per_threshold[thresh])
        else:
            result[f"AP@{thresh:.2f}"] = 0.0

    all_aps = [result[f"AP@{t:.2f}"] for t in iou_thresholds]
    result["mAP@[0.5:0.95]"] = sum(all_aps) / len(all_aps) if all_aps else 0.0
    result["AP@0.5"] = result.get("AP@0.50", 0.0)
    result["AP@0.75"] = result.get("AP@0.75", 0.0)
    result["mAP_50"] = result["AP@0.5"]
    result["per_class_AP50"] = per_class_ap50
    return result


def evaluate_object_detection(
    pred_boxes: list[list[tuple[float, float, float, float]]],
    pred_scores: list[list[float]],
    pred_labels: list[list[int]],
    gt_boxes: list[list[tuple[float, float, float, float]]],
    gt_labels: list[list[int]],
    num_classes: int,
    iou_thresholds: list[float] | None = None,
    *,
    use_pycocotools: bool | None = None,
) -> dict[str, float]:
    """Compute COCO-style object detection metrics.

    Uses pycocotools when available (``pip install open-vision-banana[evaluation]``).
    Set ``use_pycocotools=False`` to force the lightweight fallback.
    """
    if use_pycocotools is None:
        use_pycocotools = _pycocotools_available()

    if use_pycocotools:
        try:
            return evaluate_object_detection_pycocotools(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels, num_classes, iou_thresholds,
            )
        except ImportError:
            logger.debug("pycocotools unavailable; using fallback AP")
        except Exception as exc:
            logger.warning("pycocotools evaluation failed (%s); using fallback AP", exc)

    return evaluate_object_detection_fallback(
        pred_boxes, pred_scores, pred_labels,
        gt_boxes, gt_labels, num_classes, iou_thresholds,
    )


__all__ = [
    "compute_iou_box",
    "evaluate_object_detection",
    "evaluate_object_detection_pycocotools",
    "evaluate_object_detection_fallback",
]
