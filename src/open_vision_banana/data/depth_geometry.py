"""Camera-geometry helpers: renderer depth -> paper z-depth, and depth -> paper normals.

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
from pathlib import Path

import numpy as np
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


# --------------------------------------------------------------------------- normals from depth
# VKITTI2 default pinhole (1242 x 375): intrinsic.txt lists the same values per frame.
VKITTI2_DEFAULT_INTRINSICS = (725.0087, 725.0087, 620.5, 187.0)
VKITTI2_DEPTH_CM_INVALID = 65535  # far plane / sky marker in the uint16 centimetre PNGs


def read_vkitti_intrinsics(path: str | Path) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """Parse VKITTI2 ``intrinsic.txt`` (``frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]``).

    Returns ``{(frame, camera): (fx, fy, cx, cy)}``; the header line is skipped.
    """
    table: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 6 or not parts[0].lstrip("-").isdigit():
            continue
        frame, camera = int(parts[0]), int(parts[1])
        table[(frame, camera)] = tuple(float(v) for v in parts[2:6])  # type: ignore[assignment]
    return table


def depth_to_normals(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Planar z-depth ``[H, W]`` (metres) -> unit normals ``[H, W, 3]`` in the paper's camera frame.

    Points are unprojected with the OpenCV pinhole convention (x right, y down,
    z forward), tangents are central differences, the normal is their cross
    product oriented towards the camera (``n . (-p) > 0``), and the result is
    rotated into the paper frame (+x right, +y up, +z out of the image plane
    towards the viewer): ``(x, y, z) -> (x, -y, -z)``. Pixels whose own depth or
    any 4-neighbour is invalid are reported invalid and set to zero.
    """
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"depth must be [H, W], got {z.shape}")
    h, w = z.shape
    ok = np.isfinite(z) & (z > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    points = np.stack([x, y, z], axis=-1)

    # central differences with one-sided fallback at the border
    def grad(arr: np.ndarray, axis: int) -> np.ndarray:
        return np.gradient(arr, axis=axis, edge_order=1)

    du = np.stack([grad(points[..., i], 1) for i in range(3)], axis=-1)
    dv = np.stack([grad(points[..., i], 0) for i in range(3)], axis=-1)
    normal = np.cross(du, dv)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.where(norm > 1e-12, normal / np.clip(norm, 1e-12, None), 0.0)
    # orient towards the camera: flip where n . p > 0 (p points from camera to surface)
    flip = (normal * points).sum(axis=-1) > 0
    normal[flip] *= -1.0
    # neighbour validity (gradients mix neighbouring pixels)
    nb_ok = ok.copy()
    nb_ok[1:, :] &= ok[:-1, :]
    nb_ok[:-1, :] &= ok[1:, :]
    nb_ok[:, 1:] &= ok[:, :-1]
    nb_ok[:, :-1] &= ok[:, 1:]
    nb_ok &= norm[..., 0] > 1e-12
    paper = np.stack([normal[..., 0], -normal[..., 1], -normal[..., 2]], axis=-1).astype(np.float32)
    paper[~nb_ok] = 0.0
    return paper, nb_ok


def load_vkitti_depth_png(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """VKITTI2 ``depth_*.png`` (uint16 centimetres, 65535 = far/sky) -> metres + valid mask."""
    from PIL import Image

    raw = np.asarray(Image.open(path)).astype(np.float64)
    valid = raw < VKITTI2_DEPTH_CM_INVALID
    return (raw / 100.0).astype(np.float32), valid
