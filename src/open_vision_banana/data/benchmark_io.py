"""Shared tensor I/O helpers for paper benchmark dataset adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def find_matching_file(
    root: str | Path,
    stem: str,
    suffixes: Sequence[str],
) -> Path | None:
    """Find a deterministic exact-stem match below ``root``."""
    root_path = Path(root)
    for suffix in suffixes:
        candidate = root_path / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    if not root_path.is_dir():
        return None
    suffix_set = {suffix.lower() for suffix in suffixes}
    matches = sorted(
        path
        for path in root_path.rglob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in suffix_set
    )
    return matches[0] if matches else None


def infer_planar_hw(num_pixels: int) -> tuple[int, int]:
    """Infer a landscape H×W for a raw float depth dump.

    ETH3D DSLR ground-truth maps are 3:2 float32 arrays with no header.
    """
    if num_pixels <= 0:
        raise ValueError(f"Cannot infer depth shape from {num_pixels} pixels")
    width = int(round((num_pixels * 1.5) ** 0.5))
    if width > 0 and num_pixels % width == 0:
        height = num_pixels // width
        return height, width
    width = int(num_pixels**0.5)
    while width > 1 and num_pixels % width != 0:
        width -= 1
    height = num_pixels // width
    if height > width:
        height, width = width, height
    return height, width


def load_eth3d_depth(path: str | Path) -> torch.Tensor:
    """Load official ETH3D geometric depth (raw float32 metres, inf = invalid)."""
    source = Path(path)
    raw = np.fromfile(source, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Empty ETH3D depth file: {source}")
    height, width = infer_planar_hw(int(raw.size))
    depth = np.ascontiguousarray(raw.reshape(height, width))
    return torch.from_numpy(depth).float().unsqueeze(0)


def load_scalar(path: str | Path, *, scale: float = 1.0) -> torch.Tensor:
    """Load a scalar map as a float tensor with shape ``[1, H, W]``."""
    source = Path(path)
    array = (
        np.load(source)
        if source.suffix.lower() == ".npy"
        else np.asarray(Image.open(source))
    )
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D scalar map, got shape {array.shape} from {source}")
    return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0) * scale


def resize_scalar(value: torch.Tensor, resolution: int | tuple[int, int]) -> torch.Tensor:
    """Resize a scalar map with nearest-neighbour interpolation."""
    if value.dim() == 2:
        value = value.unsqueeze(0)
    if value.dim() != 3 or value.shape[0] != 1:
        raise ValueError(f"Expected scalar tensor [1,H,W], got {tuple(value.shape)}")
    size = (resolution, resolution) if isinstance(resolution, int) else resolution
    return F.interpolate(value.unsqueeze(0).float(), size=size, mode="nearest").squeeze(0)


def load_normal(path: str | Path) -> torch.Tensor:
    """Load camera-space normals as ``[3, H, W]`` in ``[-1, 1]``."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        array = np.load(source)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        if tensor.dim() != 3:
            raise ValueError(f"Expected a 3-D normal map, got {tuple(tensor.shape)}")
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.shape[0] != 3:
            raise ValueError(f"Normal map must have three channels, got {tuple(tensor.shape)}")
    else:
        rgb = torch.from_numpy(np.array(Image.open(source).convert("RGB"), copy=True)).float()
        rgb = rgb.permute(2, 0, 1) / 255.0
        tensor = torch.stack((1.0 - 2.0 * rgb[0], 2.0 * rgb[1] - 1.0, 2.0 * rgb[2] - 1.0))
    return F.normalize(tensor, dim=0, eps=1e-8)


def resize_normal(value: torch.Tensor, resolution: int | tuple[int, int]) -> torch.Tensor:
    """Resize a normal map and restore unit length per pixel."""
    if value.dim() != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected normal tensor [3,H,W], got {tuple(value.shape)}")
    size = (resolution, resolution) if isinstance(resolution, int) else resolution
    resized = F.interpolate(
        value.unsqueeze(0).float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return F.normalize(resized, dim=0, eps=1e-8)


__all__ = [
    "find_matching_file",
    "infer_planar_hw",
    "load_eth3d_depth",
    "load_normal",
    "load_scalar",
    "resize_normal",
    "resize_scalar",
]
