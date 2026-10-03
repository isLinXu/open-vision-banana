"""Task adapters reuse project decoders and dataset-level accumulators.

No generic evaluator defaults: decoding is determined by backend contracts.
New task handlers register here; the runner contains no task-specific branches.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

TASK_REGISTRY = {}


def register_task(name):
    def decorate(cls):
        if name in TASK_REGISTRY:
            raise ValueError("Task already registered: {}".format(name))
        TASK_REGISTRY[name] = cls
        return cls
    return decorate


def build_task(name, contract, dataset, options=None):
    if name not in TASK_REGISTRY:
        raise ValueError("Unknown task: {}; registered: {}".format(name, sorted(TASK_REGISTRY)))
    return TASK_REGISTRY[name](contract, dataset, options or {})


@dataclass
class Decoded:
    values: torch.Tensor
    rgb: torch.Tensor


def _rgb(prediction):
    rgb = prediction.rgb_output
    if rgb is None or rgb.ndim != 4 or tuple(rgb.shape[:2]) != (1, 3):
        raise ValueError("Task requires RGB output [1,3,H,W]")
    if not torch.isfinite(rgb).all().item():
        raise ValueError("Nonfinite prediction; metrics must not discard failed predictions")
    if rgb.min().item() < -1e-4 or rgb.max().item() > 1.0001:
        raise ValueError("RGB prediction outside declared [0,1] encoding")
    return rgb.float().clamp(0, 1)


def _metrics_module():
    if sys.version_info < (3, 10):
        raise RuntimeError("GT metrics require supported Python >=3.10; legacy mode is qualitative only")
    from open_vision_banana.evaluation import benchmark
    return benchmark


def clean_json(value):
    """Absent classes and empty valid sets are null, never non-standard JSON NaN."""
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class TaskHandler:
    def __init__(self, contract, dataset, options):
        if not isinstance(contract, dict) or not contract:
            raise ValueError("Missing task encoding contract")
        self.contract = contract
        self.dataset = dataset
        self.options = options
        if options:
            raise ValueError("Unknown task options: {}".format(sorted(options)))
        self.accumulator = None

    def start_metrics(self):
        raise NotImplementedError

    def update(self, values, gt, mask=None):
        raise NotImplementedError

    def compute(self):
        if self.accumulator is None:
            return {"status": "unavailable", "reason": "no_ground_truth", "metrics": None}
        result = self.accumulator.compute()
        if result.get("valid_pixels", 0) == 0:
            return {"status": "unavailable", "reason": "zero_valid_pixels", "metrics": None}
        return {"status": "measured", "aggregation": "dataset_global_pixels",
                "metrics": clean_json(result)}

    def validate_gt(self, gt, image_hw):
        if tuple(gt.shape[-2:]) != tuple(image_hw):
            raise ValueError("GT must be on the EXIF-oriented original image grid")


@register_task("semantic")
class SemanticTask(TaskHandler):
    def __init__(self, contract, dataset, options):
        super().__init__(contract, dataset, options)
        self.palette = contract.get("palette")
        self.num_classes = contract.get("num_classes")
        if not self.palette or len(self.palette) != self.num_classes:
            raise ValueError("Semantic palette length must equal num_classes")
        if not contract.get("label_space"):
            raise ValueError("Semantic label_space must be explicit")

    def decode(self, prediction):
        if prediction.labels is None:
            from open_vision_banana.core.segmentation import SegmentationDecoder
            labels = SegmentationDecoder().decode_semantic(_rgb(prediction), self.palette)
        else:
            labels = prediction.labels
            if labels.ndim == 3:
                labels = labels.unsqueeze(1)
            if labels.ndim != 4 or tuple(labels.shape[:2]) != (1, 1):
                raise ValueError("Semantic labels must be [1,1,H,W]")
            if labels.is_floating_point():
                raise ValueError("Semantic predictions must be integer class IDs")
        if (labels < 0).any().item() or (labels >= self.num_classes).any().item():
            raise ValueError("Predicted class ID outside declared label space")
        colors = torch.tensor(list(self.palette.values()), device=labels.device, dtype=torch.float32) / 255.0
        rgb = colors[labels[:, 0].long()].permute(0, 3, 1, 2)
        return Decoded(labels.long(), rgb)

    def start_metrics(self):
        if self.dataset.get("label_space") != self.contract["label_space"]:
            raise ValueError("GT label_space {} differs from model {}".format(
                self.dataset.get("label_space"), self.contract["label_space"]))
        self.accumulator = _metrics_module().SemanticAccumulator(
            self.num_classes, self.dataset.get("ignore_index", 255))

    def update(self, values, gt, mask=None):
        ignore = self.dataset.get("ignore_index", 255)
        valid = gt != ignore
        if mask is not None:
            valid &= mask.bool()
        if ((gt[valid] < 0) | (gt[valid] >= self.num_classes)).any().item():
            raise ValueError("GT class IDs outside declared label space")
        labels = F.interpolate(values.float(), size=gt.shape[-2:], mode="nearest").long().cpu()
        gt = gt.clone().cpu()
        if mask is not None:
            gt[~mask.cpu()] = ignore
        self.accumulator.update(labels, gt)


@register_task("depth")
class DepthTask(TaskHandler):
    def __init__(self, contract, dataset, options):
        super().__init__(contract, dataset, options)
        from open_vision_banana.core.depth_mapper import DepthRGBMapper
        if contract.get("representation") not in {"metric_power", "normalized_scalar"}:
            raise ValueError("Depth representation must be metric_power or normalized_scalar")
        required = {"colormap"} | ({"lam", "c"} if contract["representation"] == "metric_power" else set())
        if required - set(contract):
            raise ValueError("Depth contract missing explicit fields: {}".format(sorted(required - set(contract))))
        if contract["representation"] == "metric_power" and not (
                isinstance(contract["lam"], (float, int)) and math.isfinite(contract["lam"]) and contract["lam"] < -1
                and isinstance(contract["c"], (float, int)) and math.isfinite(contract["c"]) and contract["c"] > 0):
            raise ValueError("Metric power parameters require finite lam < -1 and c > 0")
        self.protocol_metadata = None
        self.mapper = DepthRGBMapper(colormap=contract["colormap"],
                                     lam=contract.get("lam", -3.0), c=contract.get("c", 10.0 / 3))

    def decode(self, prediction):
        rgb = _rgb(prediction)
        normalized = self.mapper.rgb_to_normalized(rgb)
        values = (self.mapper.normalized_to_depth(normalized)
                  if self.contract["representation"] == "metric_power" else normalized)
        return Decoded(values, rgb)

    def start_metrics(self):
        expected = "meters" if self.contract["representation"] == "metric_power" else "normalized"
        if self.dataset.get("depth_units") != expected:
            raise ValueError("Depth GT units must be {} for this contract".format(expected))
        if self.dataset.get("depth_protocol", "full") != "full" and expected != "meters":
            raise ValueError("NYU/KITTI/ETH3D protocols require metric depth")
        self.accumulator = _metrics_module().DepthAccumulator()

    def update(self, values, gt, mask=None):
        pred = F.interpolate(values.float(), size=gt.shape[-2:], mode="bilinear", align_corners=False).cpu()
        if not torch.isfinite(pred).all().item():
            raise ValueError("Nonfinite decoded depth")
        # Keep zero predictions in the denominator using an explicit epsilon,
        # rather than letting the project accumulator drop model failures.
        pred = pred.clamp_min(1e-6)
        protocol = self.dataset.get("depth_protocol", "full")
        if protocol != "full":
            from .protocols import depth_protocol
            pred, gt, mask, self.protocol_metadata = depth_protocol(protocol, pred, gt, mask)
        self.accumulator.update(pred, gt.cpu(), mask)

    def compute(self):
        result = super().compute()
        result["depth_units"] = self.dataset.get("depth_units")
        result["protocol"] = self.protocol_metadata or {"name": "full", "crop": None, "cap": None}
        result["aligned_metrics_scope"] = "diagnostic_only_per_image_median_scale_stability_filter"
        return result


@register_task("normal")
class NormalTask(TaskHandler):
    def __init__(self, contract, dataset, options):
        super().__init__(contract, dataset, options)
        if contract.get("encoding") not in {"paper_camera", "standard_xyz_to_rgb"}:
            raise ValueError("Unknown normal encoding")

    def decode(self, prediction):
        rgb = _rgb(prediction)
        if self.contract["encoding"] == "paper_camera":
            from open_vision_banana.core.normal_mapper import NormalRGBMapper
            values = NormalRGBMapper().rgb_to_normal(rgb)
        else:
            values = F.normalize(2 * rgb - 1, dim=1, eps=1e-8)
        return Decoded(values, rgb)

    def start_metrics(self):
        self.accumulator = _metrics_module().NormalAccumulator()

    def update(self, values, gt, mask=None):
        pred = F.interpolate(values.float(), size=gt.shape[-2:], mode="bilinear", align_corners=False).cpu()
        if not torch.isfinite(pred).all().item() or (pred.norm(dim=1) <= 1e-6).any().item():
            raise ValueError("Invalid predicted normal vectors")
        self.accumulator.update(pred, gt.cpu(), mask)
