"""Opt-in absolute scale supervision for cube-edge metric depth targets."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.clean_depth import _plane
from .structure_loss import soft_colormap_position


def load_native_metric_pair(
    depth_path: str, mask_path: str, units_per_metre: float, image_size: tuple[int, int]
) -> tuple[Image.Image, Image.Image]:
    """Load explicit native metres and validity, at the input's native geometry."""
    if not math.isfinite(units_per_metre) or units_per_metre <= 0:
        raise ValueError("metric_depth_scale must be finite and positive")
    depth = _plane(Path(depth_path)).astype(np.float32) / units_per_metre
    mask = _plane(Path(mask_path))
    if depth.shape != mask.shape or depth.shape[::-1] != image_size:
        raise ValueError("Native metric depth, validity mask and input geometry must match")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(mask) & (mask > 0)
    return Image.fromarray(np.where(valid, depth, 0)), Image.fromarray(valid)


def metric_scale_loss(
    decoded_rgb: torch.Tensor, metric_depth: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Huber on each image's 10% trimmed mean log-depth ratio.

    Input RGB is in the VAE's [-1, 1] range. The log inverse is evaluated in
    FP32 with a small headroom at white and a 1 mm log floor at black. Trim
    residuals per image so isolated color/edge errors do not set the scale.
    Validity comes from the native label and never from the prediction.
    """
    position = soft_colormap_position((decoded_rgb.float() + 1) / 2, colormap="cube_edge")
    if position.shape != metric_depth.shape or position.shape != valid_mask.shape:
        raise ValueError("Decoded RGB, native metric depth and valid mask geometry must match")
    # Algebraically d = 10 * ((1 - u)^(-1/2) - 1), with no subtraction near black.
    u = position * (1 - 1e-5)
    root = (1 - u).sqrt()
    log_prediction = (10 * u / (root * (1 + root)) + 1e-3).log()
    scale_errors = []
    for pred, gt, valid in zip(log_prediction, metric_depth.float(), valid_mask, strict=True):
        valid = valid.bool() & torch.isfinite(gt) & (gt > 0)
        residual = pred[valid] - (gt[valid] + 1e-3).log()
        if residual.numel():
            trim = residual.numel() // 10
            retained = residual.sort().values[trim : residual.numel() - trim] if trim else residual
            scale_errors.append(retained.mean())
    if not scale_errors:
        return decoded_rgb.sum() * 0.0
    errors = torch.stack(scale_errors)
    return F.smooth_l1_loss(errors, torch.zeros_like(errors), beta=math.log(1.25))
