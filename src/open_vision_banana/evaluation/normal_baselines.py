"""Trivial surface-normal baselines under the same accumulator as model eval.

Lower mean angular error is better. Oracles use the evaluation ground truth
and must not set the metric-gate bar (see ``evaluate_normal_gate``).

* ``facing_camera`` — every pixel is ``(0, 0, 1)``.
* ``axis_up`` — every pixel is ``(0, 1, 0)``.
* ``oracle_dataset_mean`` — one direction, the mean of all valid eval GT.
* ``oracle_image_mean`` — one direction per image, the mean of that image's
  valid GT. Beating this requires a spatial field.
* ``prior_mean_direction`` — one direction averaged from an independent
  training set, when that set is supplied.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from .benchmark import NormalAccumulator


def _as_bchw(normal: torch.Tensor) -> torch.Tensor:
    normal = normal.float()
    if normal.dim() == 3:
        normal = normal.unsqueeze(0)
    if normal.dim() != 4 or normal.shape[1] != 3:
        raise ValueError(f"Expected normals [B, 3, H, W], got {tuple(normal.shape)}")
    return normal


def _as_mask(valid: torch.Tensor | None, gt: torch.Tensor) -> torch.Tensor:
    if valid is None:
        return torch.ones(gt.shape[0], gt.shape[-2], gt.shape[-1], dtype=torch.bool, device=gt.device)
    mask = valid.bool()
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask.squeeze(1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    return mask


def _broadcast(direction: Sequence[float] | torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    vector = torch.as_tensor(direction, dtype=gt.dtype, device=gt.device).reshape(3)
    vector = vector / vector.norm().clamp_min(1e-8)
    return vector.view(1, 3, 1, 1).expand_as(gt).contiguous()


def mean_direction(batches: Iterable[Mapping[str, Any]]) -> torch.Tensor:
    """Unit mean of every valid normal in ``batches``."""
    total = torch.zeros(3)
    count = 0
    for batch in batches:
        gt = _as_bchw(batch["gt_normal"])
        mask = _as_mask(batch.get("valid_mask"), gt)
        for sample, sample_mask in zip(gt, mask, strict=True):
            selected = sample[:, sample_mask]
            if selected.numel() == 0:
                continue
            total = total + selected.sum(dim=1).cpu()
            count += int(selected.shape[1])
    if count == 0:
        raise ValueError("No valid normals to average")
    return total / total.norm().clamp_min(1e-8)


def _oracle_image_mean(gt: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    mask = _as_mask(valid, gt)
    frames = []
    for sample, sample_mask in zip(gt, mask, strict=True):
        selected = sample[:, sample_mask]
        if selected.numel() == 0:
            direction = torch.tensor([0.0, 0.0, 1.0], dtype=gt.dtype, device=gt.device)
        else:
            direction = selected.mean(dim=1)
            direction = direction / direction.norm().clamp_min(1e-8)
        frames.append(direction.view(3, 1, 1).expand_as(sample))
    return torch.stack(frames)


def evaluate_normal_baselines(
    batches: list[Mapping[str, Any]],
    *,
    prior_direction: torch.Tensor | None = None,
) -> dict[str, dict[str, float | int]]:
    """Score each trivial predictor with ``NormalAccumulator``."""
    dataset_direction = mean_direction(batches)

    def predict(name: str, gt: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if name == "facing_camera":
            return _broadcast((0.0, 0.0, 1.0), gt)
        if name == "axis_up":
            return _broadcast((0.0, 1.0, 0.0), gt)
        if name == "oracle_dataset_mean":
            return _broadcast(dataset_direction, gt)
        if name == "oracle_image_mean":
            return _oracle_image_mean(gt, valid)
        if name == "prior_mean_direction":
            if prior_direction is None:
                raise ValueError("prior_mean_direction requires prior_direction")
            return _broadcast(prior_direction, gt)
        raise KeyError(name)

    names = ["facing_camera", "axis_up", "oracle_dataset_mean", "oracle_image_mean"]
    if prior_direction is not None:
        names.append("prior_mean_direction")
    results: dict[str, dict[str, float | int]] = {}
    for name in names:
        accumulator = NormalAccumulator()
        for batch in batches:
            gt = _as_bchw(batch["gt_normal"])
            valid = batch.get("valid_mask")
            accumulator.update(predict(name, gt, valid), gt, valid)
        results[name] = accumulator.compute()
    return results
