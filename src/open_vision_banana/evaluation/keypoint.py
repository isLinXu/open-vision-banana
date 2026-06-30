"""2D keypoint detection evaluation metrics.

Standard metrics:
- OKS (Object Keypoint Similarity): area-normalized keypoint similarity
- mAP (mean Average Precision): COCO-style OKS-based AP
- PCK (Percentage of Correct Keypoints): distance threshold-based accuracy
- PCK@0.2, PCK@0.5, PCK@0.1 standard thresholds
"""

from __future__ import annotations

import math

import torch


SIGMAS_COCO = torch.tensor([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.072, 0.062, 0.079, 0.072,
    0.062, 0.107, 0.087, 0.089, 0.107, 0.087, 0.089,
], dtype=torch.float32)
"""COCO keypoint sigmas (17 keypoints) for OKS computation."""


def compute_oks(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    areas: torch.Tensor,
    sigmas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Object Keypoint Similarity (OKS) between predicted and GT keypoints.

    Args:
        pred_kpts: [B, N, 2] or [B, N, 3] — predicted keypoint coordinates (x, y, [visibility]).
        gt_kpts: [B, N, 2] or [B, N, 3] — ground truth keypoint coordinates (x, y, [visibility]).
        areas: [B] — object area (bounding box area in pixels²).
        sigmas: [N] — per-keypoint sigmas. Defaults to COCO sigmas.

    Returns:
        OKS scores [B, N].
    """
    if sigmas is None:
        sigmas = SIGMAS_COCO.to(pred_kpts.device)
        # Adapt sigmas to the actual number of keypoints if not COCO 17
        n_kpts = pred_kpts.shape[1]
        if n_kpts != sigmas.shape[0]:
            # Use uniform sigma based on mean of available sigmas, or fallback
            if n_kpts <= sigmas.shape[0]:
                sigmas = sigmas[:n_kpts]
            else:
                # Extend with the mean sigma for extra keypoints
                mean_sigma = sigmas.mean()
                extra = torch.full((n_kpts - sigmas.shape[0],), mean_sigma, device=sigmas.device, dtype=sigmas.dtype)
                sigmas = torch.cat([sigmas, extra])

    # Extract x, y coordinates
    pred_xy = pred_kpts[..., :2]
    gt_xy = gt_kpts[..., :2]

    # Visibility mask (if provided)
    if pred_kpts.shape[-1] == 3 and gt_kpts.shape[-1] == 3:
        vis_mask = (gt_kpts[..., 2] > 0).float()  # [B, N]
    else:
        vis_mask = torch.ones(pred_kpts.shape[0], pred_kpts.shape[1], device=pred_kpts.device)

    # Compute squared distance
    dx = pred_xy[..., 0] - gt_xy[..., 0]  # [B, N]
    dy = pred_xy[..., 1] - gt_xy[..., 1]  # [B, N]
    d2 = dx ** 2 + dy ** 2  # [B, N]

    # Area expansion: [B, 1]
    areas_exp = areas.unsqueeze(-1).clamp(min=1.0)

    # Sigma normalization: σ² * 2 * area
    sigmas_exp = sigmas.unsqueeze(0).expand(pred_kpts.shape[0], -1)  # [B, N]
    var = (sigmas_exp * 2) ** 2 * areas_exp  # [B, N]

    oks = torch.exp(-d2 / (2 * var + 1e-10)) * vis_mask  # [B, N]
    return oks


def compute_pck(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    thresholds: list[float] | None = None,
    normalization_factor: float | None = None,
) -> dict[str, float]:
    """Compute Percentage of Correct Keypoints (PCK).

    Args:
        pred_kpts: [B, N, 2] or [B, N, 3] — predicted keypoints.
        gt_kpts: [B, N, 2] or [B, N, 3] — ground truth keypoints.
        thresholds: List of distance thresholds (as fraction of normalization_factor).
            Defaults to [0.1, 0.2, 0.5].
        normalization_factor: Normalization distance (e.g., head size, torso length, or bbox diagonal).
            If None, uses the bounding box diagonal of each sample.

    Returns:
        Dict with PCK@threshold for each threshold.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.5]

    pred_xy = pred_kpts[..., :2]
    gt_xy = gt_kpts[..., :2]

    # Visibility mask
    if pred_kpts.shape[-1] == 3 and gt_kpts.shape[-1] == 3:
        vis_mask = (gt_kpts[..., 2] > 0).float()
    else:
        vis_mask = torch.ones(pred_kpts.shape[0], pred_kpts.shape[1], device=pred_kpts.device)

    # Compute distances
    dx = pred_xy[..., 0] - gt_xy[..., 0]
    dy = pred_xy[..., 1] - gt_xy[..., 1]
    distances = torch.sqrt(dx ** 2 + dy ** 2 + 1e-10)  # [B, N]

    # Normalization factor
    if normalization_factor is None:
        # Use bounding box diagonal as normalization
        x_range = gt_xy[..., 0].max(dim=1)[0] - gt_xy[..., 0].min(dim=1)[0]
        y_range = gt_xy[..., 1].max(dim=1)[0] - gt_xy[..., 1].min(dim=1)[0]
        normalization_factor = torch.sqrt(x_range ** 2 + y_range ** 2 + 1e-10)  # [B]
    else:
        normalization_factor = torch.full((pred_kpts.shape[0],), normalization_factor, device=pred_kpts.device)

    nf_exp = normalization_factor.unsqueeze(-1).clamp(min=1.0)  # [B, 1]

    results = {}
    for t in thresholds:
        correct = (distances < t * nf_exp).float() * vis_mask
        total_visible = vis_mask.sum()
        if total_visible > 0:
            results[f"PCK@{t}"] = correct.sum().item() / total_visible.item()
        else:
            results[f"PCK@{t}"] = 0.0

    return results


def evaluate_keypoints(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    areas: torch.Tensor | None = None,
    oks_thresholds: list[float] | None = None,
    pck_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """Compute comprehensive keypoint detection metrics.

    Args:
        pred_kpts: [B, N, 2] or [B, N, 3] — predicted keypoints (x, y, [visibility]).
        gt_kpts: [B, N, 2] or [B, N, 3] — ground truth keypoints.
        areas: [B] — object area for OKS. If None, uses bounding box area from GT.
        oks_thresholds: OKS thresholds for AP computation. Defaults to COCO standard [0.5, 0.55, ..., 0.95].
        pck_thresholds: PCK distance thresholds. Defaults to [0.1, 0.2, 0.5].

    Returns:
        Dict with:
        - ``mAP_OKS``: mean AP across OKS thresholds
        - ``AP@0.5``, ``AP@0.75``: AP at specific OKS thresholds
        - ``Mean OKS``: average OKS per keypoint
        - ``PCK@0.1``, ``PCK@0.2``, ``PCK@0.5``: PCK at various thresholds
    """
    if oks_thresholds is None:
        oks_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]

    # Compute areas if not provided
    if areas is None:
        gt_xy = gt_kpts[..., :2]
        x_min = gt_xy[..., 0].min(dim=1)[0]
        x_max = gt_xy[..., 0].max(dim=1)[0]
        y_min = gt_xy[..., 1].min(dim=1)[0]
        y_max = gt_xy[..., 1].max(dim=1)[0]
        areas = (x_max - x_min) * (y_max - y_min)  # [B]

    # Compute OKS per keypoint
    oks_per_kpt = compute_oks(pred_kpts, gt_kpts, areas)  # [B, N]
    mean_oks = oks_per_kpt.mean().item()

    # Compute AP per OKS threshold
    aps = {}
    for thresh in oks_thresholds:
        # Count keypoints with OKS >= threshold as correct
        correct = (oks_per_kpt >= thresh).float()
        if pred_kpts.shape[-1] == 3 and gt_kpts.shape[-1] == 3:
            vis_mask = (gt_kpts[..., 2] > 0).float()
        else:
            vis_mask = torch.ones_like(correct)
        tp = (correct * vis_mask).sum()
        total = vis_mask.sum().clamp(min=1.0)
        aps[f"AP@{thresh:.2f}"] = (tp / total).item()

    # mAP: mean across all OKS thresholds
    mAP = sum(aps.values()) / len(aps) if aps else 0.0

    # PCK
    pck_results = compute_pck(pred_kpts, gt_kpts, thresholds=pck_thresholds)

    result = {
        "mAP_OKS": mAP,
        "AP@0.5": aps.get("AP@0.50", 0.0),
        "AP@0.75": aps.get("AP@0.75", 0.0),
        "Mean OKS": mean_oks,
    }
    result.update(pck_results)
    return result


__all__ = [
    "compute_oks",
    "compute_pck",
    "evaluate_keypoints",
    "SIGMAS_COCO",
]