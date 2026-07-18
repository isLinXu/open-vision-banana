"""Dataset-level accumulators for decoded benchmark metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class DepthAccumulator:
    """Accumulate depth metrics over pixels, not batches."""

    abs_rel_sum: float = 0.0
    sq_rel_sum: float = 0.0
    squared_sum: float = 0.0
    log_squared_sum: float = 0.0
    valid_pixels: int = 0
    delta_counts: list[int] | None = None

    def __post_init__(self) -> None:
        if self.delta_counts is None:
            self.delta_counts = [0, 0, 0]

    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        pred = pred.squeeze(1).float()
        gt = gt.squeeze(1).float()
        mask = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0) & (pred > 0)
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            if valid_mask.dim() == mask.dim() + 1 and valid_mask.shape[1] == 1:
                valid_mask = valid_mask.squeeze(1)
            mask &= valid_mask
        pred = pred[mask]
        gt = gt[mask]
        if pred.numel() == 0:
            return
        error = pred - gt
        ratio = torch.maximum(pred / gt, gt / pred)
        self.abs_rel_sum += torch.sum(error.abs() / gt).item()
        self.sq_rel_sum += torch.sum(error.square() / gt).item()
        self.squared_sum += torch.sum(error.square()).item()
        self.log_squared_sum += torch.sum((pred.log() - gt.log()).square()).item()
        self.valid_pixels += pred.numel()
        assert self.delta_counts is not None
        for index, threshold in enumerate((1.25, 1.25**2, 1.25**3)):
            self.delta_counts[index] += int((ratio < threshold).sum().item())

    def compute(self) -> dict[str, float | int]:
        n = max(self.valid_pixels, 1)
        deltas = self.delta_counts or [0, 0, 0]
        return {
            "AbsRel": self.abs_rel_sum / n,
            "SqRel": self.sq_rel_sum / n,
            "RMSE": math.sqrt(self.squared_sum / n),
            "RMSE_log": math.sqrt(self.log_squared_sum / n),
            "delta_1.25": deltas[0] / n,
            "delta_1.25^2": deltas[1] / n,
            "delta_1.25^3": deltas[2] / n,
            "valid_pixels": self.valid_pixels,
        }


@dataclass
class NormalAccumulator:
    """Accumulate surface-normal angular metrics over valid pixels."""

    angle_sum: float = 0.0
    angles: list[torch.Tensor] | None = None
    valid_pixels: int = 0

    def __post_init__(self) -> None:
        if self.angles is None:
            self.angles = []

    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        pred = pred.float()
        gt = gt.float()
        pred_norm = pred.norm(dim=1)
        gt_norm = gt.norm(dim=1)
        mask = torch.isfinite(pred).all(dim=1) & torch.isfinite(gt).all(dim=1)
        mask &= (pred_norm > 1e-6) & (gt_norm > 1e-6)
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            if valid_mask.dim() == mask.dim() + 1 and valid_mask.shape[1] == 1:
                valid_mask = valid_mask.squeeze(1)
            mask &= valid_mask
        pred = pred / pred_norm.unsqueeze(1).clamp_min(1e-8)
        gt = gt / gt_norm.unsqueeze(1).clamp_min(1e-8)
        cosine = (pred * gt).sum(dim=1).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine))[mask]
        if angles.numel() == 0:
            return
        self.angle_sum += angles.sum().item()
        self.valid_pixels += angles.numel()
        assert self.angles is not None
        self.angles.append(angles.detach().cpu())

    def compute(self) -> dict[str, float | int]:
        angles = torch.cat(self.angles) if self.angles else torch.empty(0)
        n = max(self.valid_pixels, 1)
        mean_angle = self.angle_sum / n
        median_angle = angles.median().item() if angles.numel() else float("nan")
        acc_11 = float((angles < 11.25).float().mean()) if angles.numel() else 0.0
        acc_22 = float((angles < 22.5).float().mean()) if angles.numel() else 0.0
        acc_30 = float((angles < 30).float().mean()) if angles.numel() else 0.0
        return {
            "mean_angle_deg": self.angle_sum / n,
            "median_angle_deg": median_angle,
            "accuracy_11.25": acc_11,
            "accuracy_22.5": acc_22,
            "accuracy_30": acc_30,
            "Mean Angular Error (°)": mean_angle,
            "Median Angular Error (°)": median_angle,
            "11.25° Accuracy": acc_11,
            "22.5° Accuracy": acc_22,
            "30° Accuracy": acc_30,
            "valid_pixels": self.valid_pixels,
        }


class SemanticAccumulator:
    """Accumulate a global Cityscapes-style confusion matrix."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        pred = pred.long().cpu()
        gt = gt.long().cpu()
        valid = (gt != self.ignore_index) & (gt >= 0) & (gt < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)
        indices = gt[valid] * self.num_classes + pred[valid]
        self.confusion += torch.bincount(
            indices, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float | int | list[float]]:
        tp = torch.diag(self.confusion).float()
        denominator = self.confusion.sum(0) + self.confusion.sum(1) - tp
        iou = torch.where(denominator > 0, tp / denominator, torch.nan)
        valid_pixels = int(self.confusion.sum().item())
        accuracy = tp.sum().item() / max(valid_pixels, 1)
        return {
            "mIoU": torch.nanmean(iou).item(),
            "pixel_accuracy": accuracy,
            "per_class_iou": iou.tolist(),
            "valid_pixels": valid_pixels,
        }


def benchmark_valid_mask(
    dataset: str,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Apply standard benchmark validity/crop rules when requested.

    NYU uses the Eigen crop; KITTI uses the Garg crop on the common 352x1216
    evaluation canvas. Callers may pass a precomputed mask to override the
    default validity rule.
    """
    mask = valid_mask.bool() if valid_mask is not None else torch.ones_like(gt, dtype=torch.bool)
    height, width = gt.shape[-2:]
    if dataset == "nyu" and height >= 400 and width >= 500:
        crop = torch.zeros_like(mask)
        y0, y1 = int(0.408 * height), int(0.992 * height)
        x0, x1 = int(0.035 * width), int(0.965 * width)
        crop[..., y0:y1, x0:x1] = True
        mask &= crop
    elif dataset == "kitti" and height >= 300 and width >= 1000:
        crop = torch.zeros_like(mask)
        y0, y1 = int(0.408 * height), int(0.992 * height)
        x0, x1 = int(0.036 * width), int(0.964 * width)
        crop[..., y0:y1, x0:x1] = True
        mask &= crop
    return mask


__all__ = [
    "DepthAccumulator",
    "NormalAccumulator",
    "SemanticAccumulator",
    "benchmark_valid_mask",
]
