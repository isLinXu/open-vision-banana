"""Trivial depth baselines evaluated under the exact benchmark protocol.

Purpose: calibrate what a metric (e.g. delta_1.25 on NYU) means before reading
it as evidence that a model learned geometry. A model that only reproduces a
dataset-level depth prior should score about the same as these baselines.

Baselines (all use ``apply_depth_benchmark_protocol`` + ``DepthAccumulator``,
i.e. the same crop, cap and pixel-weighted aggregation as model evaluation):

* ``constant@<m>``      -- one fixed depth in metres for every pixel.
* ``oracle_constant``   -- one scalar per *dataset*, set to the median valid GT
  depth (best-case constant; uses the evaluation GT, so it is an upper bound
  for any constant predictor, not a legitimate zero-shot method).
* ``oracle_image_median`` -- one scalar per *image*, the median valid GT depth
  of that image (upper bound for "knows the global scale, nothing else").
* ``prior_mean_map``    -- per-pixel mean depth from an independent prior set
  (e.g. the synthetic training data), resized to the evaluation canvas. This is
  a legitimate zero-shot baseline: it encodes the training distribution's
  layout (floor low / ceiling high) and nothing image-specific.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .benchmark import DepthAccumulator, apply_depth_benchmark_protocol


def _batch_tensors(batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    gt = batch["gt_depth"].float()
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    valid = batch.get("valid_mask")
    if valid is not None:
        valid = valid.bool()
        if valid.dim() == 3:
            valid = valid.unsqueeze(1)
    return gt, valid


def _valid_pixels(gt: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    mask = torch.isfinite(gt) & (gt > 0)
    if valid is not None:
        mask &= valid
    return mask


def _median(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.median().item())


def mean_depth_map(
    batches: Iterable[Mapping[str, Any]],
    size: tuple[int, int],
) -> torch.Tensor:
    """Per-pixel mean valid depth over *batches*, resized to *size* ``(H, W)``.

    Pixels never observed as valid fall back to the global mean so the map is
    strictly positive everywhere.
    """
    total = torch.zeros(1, 1, *size)
    count = torch.zeros(1, 1, *size)
    global_sum = 0.0
    global_count = 0
    for batch in batches:
        gt, valid = _batch_tensors(batch)
        mask = _valid_pixels(gt, valid)
        gt = torch.where(mask, gt, torch.zeros_like(gt))
        if gt.shape[-2:] != size:
            gt = F.interpolate(gt, size=size, mode="nearest")
            mask = F.interpolate(mask.float(), size=size, mode="nearest") > 0.5
        total += (gt * mask).sum(dim=0, keepdim=True)
        count += mask.float().sum(dim=0, keepdim=True)
        global_sum += float(gt[mask].sum().item())
        global_count += int(mask.sum().item())
    if global_count == 0:
        raise ValueError("No valid depth pixels in prior set")
    fallback = global_sum / global_count
    return torch.where(count > 0, total / count.clamp_min(1.0), torch.full_like(total, fallback))


def evaluate_trivial_baselines(
    batches: Sequence[Mapping[str, Any]],
    *,
    dataset_name: str,
    constants: Sequence[float] = (1.0, 2.0, 3.0, 5.0, 10.0),
    prior_map: torch.Tensor | None = None,
) -> dict[str, dict[str, float | int]]:
    """Evaluate trivial predictors on *batches* (each with ``gt_depth``/``valid_mask``).

    *batches* must be re-iterable (a list), because the dataset-level oracle
    needs one pass to find the median before scoring.
    """
    # Pass 1: dataset-level median of valid GT, after the benchmark crop/cap.
    protocol_values: list[torch.Tensor] = []
    for batch in batches:
        gt, valid = _batch_tensors(batch)
        _, gt_p, mask = apply_depth_benchmark_protocol(dataset_name, gt.clone(), gt, valid)
        mask = mask & _valid_pixels(gt_p, None)
        protocol_values.append(gt_p[mask].flatten())
    dataset_median = _median(torch.cat(protocol_values)) if protocol_values else float("nan")

    accumulators: dict[str, DepthAccumulator] = {f"constant@{c:g}m": DepthAccumulator() for c in constants}
    accumulators["oracle_constant"] = DepthAccumulator()
    accumulators["oracle_image_median"] = DepthAccumulator()
    if prior_map is not None:
        accumulators["prior_mean_map"] = DepthAccumulator()

    for batch in batches:
        gt, valid = _batch_tensors(batch)
        base_mask = _valid_pixels(gt, valid)

        def score(name: str, pred: torch.Tensor) -> None:
            pred_p, gt_p, mask_p = apply_depth_benchmark_protocol(dataset_name, pred, gt, valid)
            accumulators[name].update(pred_p, gt_p, mask_p)

        for c in constants:
            score(f"constant@{c:g}m", torch.full_like(gt, float(c)))
        score("oracle_constant", torch.full_like(gt, dataset_median))
        per_image = torch.stack(
            [
                torch.full_like(gt[i], _median(gt[i][base_mask[i]]) if bool(base_mask[i].any()) else 1.0)
                for i in range(gt.shape[0])
            ]
        )
        score("oracle_image_median", per_image)
        if prior_map is not None:
            prior = prior_map.to(gt.dtype)
            if prior.shape[-2:] != gt.shape[-2:]:
                prior = F.interpolate(prior, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            score("prior_mean_map", prior.expand_as(gt).clone())

    results = {name: acc.compute() for name, acc in accumulators.items()}
    results["oracle_constant"]["constant_value_m"] = dataset_median
    return results


__all__ = ["evaluate_trivial_baselines", "mean_depth_map"]
