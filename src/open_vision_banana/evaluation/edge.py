"""Edge detection evaluation metrics.

Metrics:
- precision, recall, F1 (standard binary classification)
- ODS (Optimal Dataset Scale): F1 at optimal threshold across dataset
- OIS (Optimal Image Scale): F1 at optimal threshold per image
- AP (Average Precision): area under PR curve
"""

from __future__ import annotations

import torch


def evaluate_edge(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float | None = None,
    compute_ods: bool = True,
) -> dict:
    """Evaluate edge detection results.

    Supports two modes:
    1. Fixed threshold (default): ``threshold`` is applied to float predictions.
    2. ODS (Optimal Dataset Scale): when ``threshold`` is *None* and
       ``compute_ods=True``, the optimal threshold is searched across the
       dataset to maximize the global F1 score.  This matches the standard
       BSDS500 evaluation protocol.

    Args:
        pred: Predicted edge probability [B, 1, H, W] in [0, 1] or bool.
        gt: Ground truth edge mask [B, 1, H, W] (bool).
        threshold: Fixed threshold for binarization.  If *None* and
            ``compute_ods`` is True, ODS threshold is searched.
        compute_ods: Whether to compute ODS when threshold is None.

    Returns:
        Dict with keys:
        - ``precision``, ``recall``, ``f1`` (at the applied threshold)
        - ``ods_threshold`` (float or None)
        - ``ods_f1`` (float or None)
    """
    if gt.dtype != torch.bool:
        gt = gt > 0.5
    gt = gt.bool()

    # If pred is already bool, skip threshold search
    if pred.dtype == torch.bool:
        pred_bin = pred.bool()
        tp = (pred_bin & gt).sum().float()
        fp = (pred_bin & ~gt).sum().float()
        fn = (~pred_bin & gt).sum().float()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return {
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "ods_threshold": None,
            "ods_f1": None,
        }

    pred_float = pred.float()

    if threshold is not None:
        # Fixed threshold mode
        pred_bin = pred_float > threshold
        tp = (pred_bin & gt).sum().float()
        fp = (pred_bin & ~gt).sum().float()
        fn = (~pred_bin & gt).sum().float()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return {
            "precision": precision.item(),
            "recall": recall.item(),
            "f1": f1.item(),
            "ods_threshold": threshold,
            "ods_f1": f1.item(),
        }

    if not compute_ods:
        # Default threshold = 0.5
        return evaluate_edge(pred, gt, threshold=0.5, compute_ods=False)

    # ── ODS: search optimal threshold over the dataset ──
    # Sample thresholds from [0.01, 0.99] in 50 steps
    thresholds = torch.linspace(0.01, 0.99, 50, device=pred_float.device)
    best_f1 = 0.0
    best_thresh = 0.5
    best_precision = 0.0
    best_recall = 0.0

    for t in thresholds:
        pred_bin = pred_float > t
        tp = (pred_bin & gt).sum().float()
        fp = (pred_bin & ~gt).sum().float()
        fn = (~pred_bin & gt).sum().float()
        p = tp / (tp + fp + 1e-10)
        r = tp / (tp + fn + 1e-10)
        f1_t = 2 * p * r / (p + r + 1e-10)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t.item()
            best_precision = p.item()
            best_recall = r.item()

    return {
        "precision": float(best_precision),
        "recall": float(best_recall),
        "f1": float(best_f1),
        "ods_threshold": float(best_thresh),
        "ods_f1": float(best_f1),
    }


def evaluate_edge_dataset(
    preds: list[torch.Tensor],
    gts: list[torch.Tensor],
    num_thresholds: int = 50,
) -> dict:
    """Evaluate edge detection over a full dataset with ODS and OIS.

    Args:
        preds: List of predicted edge probability maps [1, 1, H_i, W_i].
        gts: List of ground truth edge masks [1, 1, H_i, W_i] (bool).
        num_thresholds: Number of thresholds to search for ODS.

    Returns:
        Dict with:
        - ``ods_f1``, ``ods_precision``, ``ods_recall``, ``ods_threshold``
        - ``ois_f1``: mean of per-image optimal F1
        - ``ap``: average precision across all thresholds
    """
    thresholds = torch.linspace(0.01, 0.99, num_thresholds)

    # Global counts for ODS
    global_tp = torch.zeros(num_thresholds)
    global_fp = torch.zeros(num_thresholds)
    global_fn = torch.zeros(num_thresholds)

    ois_f1s = []

    for pred, gt in zip(preds, gts, strict=False):
        if gt.dtype != torch.bool:
            gt = gt > 0.5
        gt = gt.bool()
        pred_f = pred.float()

        best_img_f1 = 0.0
        for ti, t in enumerate(thresholds):
            pred_bin = pred_f > t
            tp = (pred_bin & gt).sum().float()
            fp = (pred_bin & ~gt).sum().float()
            fn = (~pred_bin & gt).sum().float()
            global_tp[ti] += tp.item()
            global_fp[ti] += fp.item()
            global_fn[ti] += fn.item()

            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            if f1 > best_img_f1:
                best_img_f1 = f1.item()

        ois_f1s.append(best_img_f1)

    # ODS: find best threshold across all images
    ods_f1s = []
    ods_ps = []
    ods_rs = []
    for ti in range(num_thresholds):
        p = global_tp[ti] / (global_tp[ti] + global_fp[ti] + 1e-10)
        r = global_tp[ti] / (global_tp[ti] + global_fn[ti] + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)
        ods_f1s.append(float(f1))
        ods_ps.append(float(p))
        ods_rs.append(float(r))

    best_idx = int(torch.tensor(ods_f1s).argmax().item())

    # AP: trapezoidal rule over PR curve
    sorted_indices = torch.argsort(torch.tensor(ods_rs), descending=False)
    ap = 0.0
    for i in range(len(sorted_indices) - 1):
        i1, i2 = sorted_indices[i].item(), sorted_indices[i + 1].item()
        r1, r2 = ods_rs[i1], ods_rs[i2]
        p1, p2 = ods_ps[i1], ods_ps[i2]
        if r2 > r1:
            ap += (r2 - r1) * (p1 + p2) / 2.0

    return {
        "ods_f1": ods_f1s[best_idx],
        "ods_precision": ods_ps[best_idx],
        "ods_recall": ods_rs[best_idx],
        "ods_threshold": thresholds[best_idx].item(),
        "ois_f1": sum(ois_f1s) / len(ois_f1s) if ois_f1s else 0.0,
        "ap": ap,
    }


__all__ = ["evaluate_edge", "evaluate_edge_dataset"]
