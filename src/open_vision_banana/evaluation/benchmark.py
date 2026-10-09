"""Dataset-level accumulators for decoded benchmark metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .depth import median_scale_align


@dataclass
class DepthAccumulator:
    """Accumulate depth metrics over pixels, not batches."""

    abs_rel_sum: float = 0.0
    sq_rel_sum: float = 0.0
    squared_sum: float = 0.0
    log_squared_sum: float = 0.0
    valid_pixels: int = 0
    delta_counts: list[int] | None = None
    aligned_abs_rel_sum: float = 0.0
    aligned_sq_rel_sum: float = 0.0
    aligned_squared_sum: float = 0.0
    aligned_log_squared_sum: float = 0.0
    aligned_valid_pixels: int = 0
    aligned_delta_counts: list[int] = field(default_factory=lambda: [0, 0, 0])
    # Per-image aggregation (mean of per-image metrics, the convention used by
    # many depth papers). The paper does not state which one it uses, so both
    # are reported; pixel-weighted remains the headline.
    image_count: int = 0
    image_abs_rel_sum: float = 0.0
    image_rmse_sum: float = 0.0
    image_delta1_sum: float = 0.0

    def __post_init__(self) -> None:
        if self.delta_counts is None:
            self.delta_counts = [0, 0, 0]

    def _add(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        aligned: bool,
    ) -> None:
        if pred.numel() == 0:
            return
        error = pred - gt
        ratio = torch.maximum(pred / gt, gt / pred)
        if aligned:
            self.aligned_abs_rel_sum += torch.sum(error.abs() / gt).item()
            self.aligned_sq_rel_sum += torch.sum(error.square() / gt).item()
            self.aligned_squared_sum += torch.sum(error.square()).item()
            self.aligned_log_squared_sum += torch.sum((pred.log() - gt.log()).square()).item()
            self.aligned_valid_pixels += pred.numel()
            for index, threshold in enumerate((1.25, 1.25**2, 1.25**3)):
                self.aligned_delta_counts[index] += int((ratio < threshold).sum().item())
            return
        self.abs_rel_sum += torch.sum(error.abs() / gt).item()
        self.sq_rel_sum += torch.sum(error.square() / gt).item()
        self.squared_sum += torch.sum(error.square()).item()
        self.log_squared_sum += torch.sum((pred.log() - gt.log()).square()).item()
        self.valid_pixels += pred.numel()
        assert self.delta_counts is not None
        for index, threshold in enumerate((1.25, 1.25**2, 1.25**3)):
            self.delta_counts[index] += int((ratio < threshold).sum().item())

    def update(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        pred = pred.squeeze(1).float()
        gt = gt.squeeze(1).float()
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
        mask = torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0) & (pred > 0)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=mask.device).bool()
            if valid_mask.dim() == mask.dim() + 1 and valid_mask.shape[1] == 1:
                valid_mask = valid_mask.squeeze(1)
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(0).expand_as(mask)
            mask &= valid_mask
        self._add(pred[mask], gt[mask], aligned=False)
        for index in range(pred.shape[0]):
            image_mask = mask[index]
            if not bool(image_mask.any()):
                continue
            p_i, g_i = pred[index][image_mask], gt[index][image_mask]
            self.image_count += 1
            self.image_abs_rel_sum += torch.mean((p_i - g_i).abs() / g_i).item()
            self.image_rmse_sum += torch.sqrt(torch.mean((p_i - g_i).square())).item()
            self.image_delta1_sum += (torch.maximum(p_i / g_i, g_i / p_i) < 1.25).float().mean().item()
        aligned, stable = median_scale_align(pred, gt, mask)
        if stable.dim() == 0:
            stable = stable.view(1)
        pixel_stable = stable.view(-1, *([1] * (mask.dim() - 1))).expand_as(mask) & mask
        if bool(pixel_stable.any()):
            self._add(aligned[pixel_stable], gt[pixel_stable], aligned=True)

    def compute(self) -> dict[str, float | int]:
        n = max(self.valid_pixels, 1)
        aligned_n = max(self.aligned_valid_pixels, 1)
        deltas = self.delta_counts or [0, 0, 0]
        aligned_deltas = self.aligned_delta_counts
        return {
            "AbsRel": self.abs_rel_sum / n,
            "SqRel": self.sq_rel_sum / n,
            "RMSE": math.sqrt(self.squared_sum / n),
            "RMSE_log": math.sqrt(self.log_squared_sum / n),
            "delta_1.25": deltas[0] / n,
            "delta_1.25^2": deltas[1] / n,
            "delta_1.25^3": deltas[2] / n,
            "valid_pixels": self.valid_pixels,
            "AbsRel_per_image": self.image_abs_rel_sum / max(self.image_count, 1),
            "RMSE_per_image": self.image_rmse_sum / max(self.image_count, 1),
            "delta_1.25_per_image": self.image_delta1_sum / max(self.image_count, 1),
            "images": self.image_count,
            "AbsRel_aligned": self.aligned_abs_rel_sum / aligned_n,
            "delta_1.25_aligned": aligned_deltas[0] / aligned_n,
            "RMSE_aligned": math.sqrt(self.aligned_squared_sum / aligned_n),
            "aligned_valid_pixels": self.aligned_valid_pixels,
        }


@dataclass
class NormalAccumulator:
    """Accumulate surface-normal angular metrics over valid pixels."""

    angle_sum: float = 0.0
    angles: list[torch.Tensor] | None = None
    valid_pixels: int = 0
    image_count: int = 0
    image_mean_sum: float = 0.0

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
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            gt = gt.unsqueeze(0)
        pred_norm = pred.norm(dim=1)
        gt_norm = gt.norm(dim=1)
        mask = torch.isfinite(pred).all(dim=1) & torch.isfinite(gt).all(dim=1)
        mask &= (pred_norm > 1e-6) & (gt_norm > 1e-6)
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=mask.device).bool()
            if valid_mask.dim() == mask.dim() + 1 and valid_mask.shape[1] == 1:
                valid_mask = valid_mask.squeeze(1)
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(0)
            mask &= valid_mask
        pred = pred / pred_norm.unsqueeze(1).clamp_min(1e-8)
        gt = gt / gt_norm.unsqueeze(1).clamp_min(1e-8)
        cosine = (pred * gt).sum(dim=1).clamp(-1.0, 1.0)
        angle_map = torch.rad2deg(torch.acos(cosine))
        assert self.angles is not None
        for sample_angles, sample_mask in zip(angle_map, mask, strict=True):
            selected = sample_angles[sample_mask]
            if selected.numel() == 0:
                continue
            self.angle_sum += selected.sum().item()
            self.valid_pixels += int(selected.numel())
            self.image_count += 1
            self.image_mean_sum += selected.mean().item()
            self.angles.append(selected.detach().cpu())

    def compute(self) -> dict[str, float | int]:
        angles = torch.cat(self.angles) if self.angles else torch.empty(0)
        n = max(self.valid_pixels, 1)
        mean_angle = self.angle_sum / n
        median_angle = angles.median().item() if angles.numel() else float("nan")
        acc_11 = float((angles < 11.25).float().mean()) if angles.numel() else 0.0
        acc_22 = float((angles < 22.5).float().mean()) if angles.numel() else 0.0
        acc_30 = float((angles < 30).float().mean()) if angles.numel() else 0.0
        per_image = self.image_mean_sum / self.image_count if self.image_count else float("nan")
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
            "mean_angle_deg_per_image": per_image,
            "images": self.image_count,
        }


def as_batch_hw_labels(labels: torch.Tensor) -> torch.Tensor:
    """Coerce class maps to ``[B, H, W]``.

    Decoded semantic masks are ``[B, 1, H, W]``. Cityscapes GT after DataLoader
    collate is ``[B, H, W]``. In-place masks cannot broadcast those two.
    """
    labels = labels.long()
    while labels.dim() > 3:
        if labels.shape[1] != 1:
            raise ValueError(f"semantic labels must be [B, H, W], got {tuple(labels.shape)}")
        labels = labels.squeeze(1)
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    if labels.dim() != 3:
        raise ValueError(f"semantic labels must be [B, H, W], got {tuple(labels.shape)}")
    return labels


class SemanticAccumulator:
    """Accumulate a global Cityscapes-style confusion matrix."""

    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        pred = as_batch_hw_labels(pred).cpu()
        gt = as_batch_hw_labels(gt).cpu()
        if pred.shape != gt.shape:
            raise ValueError(
                f"pred/gt spatial shapes must match: {tuple(pred.shape)} vs {tuple(gt.shape)}"
            )
        valid = (gt != self.ignore_index) & (gt >= 0) & (gt < self.num_classes)
        valid = valid & (pred >= 0) & (pred < self.num_classes)
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


DEPTH_EVAL_CAPS = {
    "nyu": 10.0,
    "kitti": 80.0,
    "eth3d": 80.0,
}


_MIN_CROP_CANVAS = 64

# (top, bottom, left, right) as fractions of the original frame.
DEPTH_CROP_FRACTIONS = {
    # Eigen et al. NYU crop: rows 45:471, cols 41:601 of the 480x640 frame.
    "nyu": (45 / 480, 471 / 480, 41 / 640, 601 / 640),
    # Garg et al. KITTI crop (0.40810811 H, 0.99189189 H, 0.03594771 W, 0.96405229 W).
    "kitti": (0.40810811, 0.99189189, 0.03594771, 0.96405229),
}


def canonical_depth_dataset(dataset: str) -> str:
    """Map adapter keys such as nyu_depth onto the Eigen/Garg protocol names."""
    key = dataset.strip().lower()
    if key.startswith("nyu"):
        return "nyu"
    if key.startswith("kitti"):
        return "kitti"
    if key.startswith("eth3d"):
        return "eth3d"
    return key


def apply_depth_benchmark_protocol(
    dataset: str,
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Crop, mask, and cap depths the way NYU Eigen / KITTI Garg eval expect.

    Predictions that decode to tens or thousands of metres (cube-edge near
    white) must be clipped, or a few pixels dominate AbsRel/RMSE.
    """
    protocol = canonical_depth_dataset(dataset)
    mask = benchmark_valid_mask(protocol, gt, valid_mask)
    cap = DEPTH_EVAL_CAPS.get(protocol)
    if cap is not None:
        finite_gt = torch.isfinite(gt) & (gt > 1e-3) & (gt <= cap)
        mask = mask & finite_gt
        pred = pred.clamp(1e-3, cap)
        gt = gt.clamp(1e-3, cap)
    return pred, gt, mask


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
    mask = (
        valid_mask.to(device=gt.device).bool()
        if valid_mask is not None
        else torch.ones_like(gt, dtype=torch.bool)
    )
    # Dataset adapters expose masks as [B,H,W] while decoded depth/GT may be
    # [B,1,H,W]. Normalize before applying a crop; otherwise broadcasting a
    # [B,H,W] mask against [B,1,H,W] creates an accidental [B,B,H,W] mask
    # whenever batch_size > 1.
    if mask.dim() == gt.dim() - 1 and gt.dim() >= 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() == 2 and gt.dim() == 3:
        mask = mask.unsqueeze(0)
    if mask.shape != gt.shape:
        raise ValueError(
            "Depth validity mask shape must match ground truth after channel "
            f"normalization: mask={tuple(mask.shape)}, gt={tuple(gt.shape)}"
        )
    height, width = gt.shape[-2:]
    # Crops are defined as fractions of the original frame so they stay valid
    # when the benchmark is evaluated on a resized canvas (e.g. 512x512).
    # The size guard only skips degenerate toy tensors.
    if min(height, width) >= _MIN_CROP_CANVAS and dataset in DEPTH_CROP_FRACTIONS:
        top, bottom, left, right = DEPTH_CROP_FRACTIONS[dataset]
        crop = torch.zeros_like(mask)
        crop[
            ...,
            round(top * height) : round(bottom * height),
            round(left * width) : round(right * width),
        ] = True
        mask &= crop
    return mask


__all__ = [
    "DEPTH_CROP_FRACTIONS",
    "DEPTH_EVAL_CAPS",
    "DepthAccumulator",
    "NormalAccumulator",
    "SemanticAccumulator",
    "as_batch_hw_labels",
    "apply_depth_benchmark_protocol",
    "benchmark_valid_mask",
    "canonical_depth_dataset",
]
