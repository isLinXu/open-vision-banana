"""Camera-geometry helpers for turning renderer depth into paper-style z-depth.

The paper (Section 3.2) defines depth as "the physical metric distance from the
camera plane to the observed object", i.e. planar z-depth. Hypersim's
``frame.XXXX.depth_meters.hdf5`` instead stores the Euclidean distance from the
optical centre to the surface along each pixel's ray; at the corners of a
1024x768 frame that is ~25% larger than z. Encoding it unconverted bakes a
systematic, position-dependent bias into every training target.

The conversion only needs the focal length in pixels. Hypersim renders with a
60 degree horizontal field of view, which gives the ``fltFocal = 886.81`` constant
used by the official repository for 1024 px wide frames; the same rule scales to
other widths.
"""

from __future__ import annotations

import math

import torch

HYPERSIM_HORIZONTAL_FOV_DEG = 60.0


def hypersim_focal_px(width: int, fov_deg: float = HYPERSIM_HORIZONTAL_FOV_DEG) -> float:
    """Focal length in pixels for a frame ``width`` px wide and ``fov_deg`` horizontal FOV."""
    if width <= 0:
        raise ValueError("width must be positive")
    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def ray_distance_to_planar_depth(
    distance: torch.Tensor,
    focal_px: float | None = None,
    *,
    principal_point: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Convert per-pixel ray distance ``[..., H, W]`` to planar z-depth.

    ``z = d * f / sqrt((u - cx)^2 + (v - cy)^2 + f^2)`` with pixel-centre coordinates.
    ``focal_px`` defaults to the Hypersim 60 degree rule for the tensor width; the
    principal point defaults to the image centre. Non-finite / non-positive input
    is passed through unchanged so invalid markers survive.
    """
    if distance.ndim < 2:
        raise ValueError("distance must have at least 2 dimensions [..., H, W]")
    height, width = int(distance.shape[-2]), int(distance.shape[-1])
    focal = hypersim_focal_px(width) if focal_px is None else float(focal_px)
    if focal <= 0:
        raise ValueError("focal_px must be positive")
    cx, cy = principal_point if principal_point is not None else ((width - 1) / 2.0, (height - 1) / 2.0)
    us = torch.arange(width, dtype=torch.float32, device=distance.device) - cx
    vs = torch.arange(height, dtype=torch.float32, device=distance.device) - cy
    norm = torch.sqrt(vs.view(-1, 1) ** 2 + us.view(1, -1) ** 2 + focal**2)
    scale = (focal / norm).to(distance.dtype)
    valid = torch.isfinite(distance) & (distance > 0)
    return torch.where(valid, distance * scale, distance)


__all__ = ["HYPERSIM_HORIZONTAL_FOV_DEG", "hypersim_focal_px", "ray_distance_to_planar_depth"]
