"""Benchmark I/O utilities for loading and resizing depth/normal/scalar targets.

Provides helper functions used by dataset classes to load ground-truth
depth maps, surface normals, and other scalar fields from disk, with
optional resizing to a target resolution.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Image file extensions to search for
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
_NPY_EXTS = (".npy", ".npz")
_ALL_EXTS = _IMAGE_EXTS + _NPY_EXTS


def find_matching_file(
    directory: Path,
    stem: str,
    extensions: tuple[str, ...] = _ALL_EXTS,
    suffixes: tuple[str, ...] = ("", "_depth", "_normal", "_gt", "_target"),
) -> Path | None:
    """Find a file in *directory* matching *stem* with any of the given extensions.

    Searches for files named ``{stem}{suffix}{ext}`` for each combination of
    suffix and extension. Returns the first match, or None if not found.

    Args:
        directory: Directory to search in.
        stem: Base filename stem (without extension).
        extensions: Tuple of file extensions to try (e.g. (".png", ".npy")).
        suffixes: Tuple of suffixes to append to stem (e.g. ("", "_depth")).

    Returns:
        Path to the first matching file, or None.
    """
    directory = Path(directory)
    if not directory.exists():
        return None

    for suffix in suffixes:
        for ext in extensions:
            candidate = directory / f"{stem}{suffix}{ext}"
            if candidate.exists():
                return candidate

    # Fallback: regex search for any file starting with stem
    pattern = re.compile(rf"^{re.escape(stem)}.*$")
    for f in directory.iterdir():
        if f.is_file() and pattern.match(f.stem):
            return f

    return None


def load_scalar(
    path: Path,
    scale: float = 1.0,
) -> torch.Tensor:
    """Load a scalar field (e.g. depth map) from disk.

    Supports:
    - ``.npy`` / ``.npz``: Loaded as numpy array, converted to tensor.
    - Image files (``.png``, ``.jpg``, etc.): Loaded as grayscale, converted
      to float tensor and scaled.

    Args:
        path: Path to the file.
        scale: Multiplicative scale factor applied to the loaded values.
            For depth PNGs encoded in millimeters, use scale=0.001 to
            convert to meters.

    Returns:
        Tensor of shape [H, W] (2D, no batch/channel dims).
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in _NPY_EXTS:
        if ext == ".npz":
            data = np.load(str(path))
            # Try common keys, fall back to first array
            for key in ("depth", "target", "gt", "arr_0"):
                if key in data:
                    data = data[key]
                    break
            else:
                data = data[list(data.keys())[0]]
        else:
            data = np.load(str(path))
        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
    elif ext in _IMAGE_EXTS:
        img = Image.open(str(path))
        # Convert to grayscale if multi-channel
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img, dtype=np.float32)
        tensor = torch.from_numpy(arr)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Apply scale
    if scale != 1.0:
        tensor = tensor * scale

    return tensor


def load_normal(
    path: Path,
    normalize: bool = True,
) -> torch.Tensor:
    """Load a surface normal map from disk.

    Supports:
    - ``.npy`` / ``.npz``: Loaded as [H, W, 3] numpy array.
    - Image files (``.png``, ``.jpg``): Loaded as RGB, converted to float.

    Args:
        path: Path to the file.
        normalize: If True, normalize each pixel's vector to unit length.

    Returns:
        Tensor of shape [3, H, W] (CHW format).
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in _NPY_EXTS:
        if ext == ".npz":
            data = np.load(str(path))
            for key in ("normal", "target", "gt", "arr_0"):
                if key in data:
                    data = data[key]
                    break
            else:
                data = data[list(data.keys())[0]]
        else:
            data = np.load(str(path))
        arr = np.asarray(data, dtype=np.float32)
        # Ensure [H, W, 3]
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] != 3:
            if arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)
    elif ext in _IMAGE_EXTS:
        img = Image.open(str(path)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        # Convert [0,1] to [-1,1] range for normals
        arr = arr * 2.0 - 1.0
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    # Convert to CHW
    tensor = torch.from_numpy(arr)
    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)  # HWC → CHW

    if normalize:
        # Normalize each pixel to unit length
        norm = tensor.norm(dim=0, keepdim=True).clamp_min(1e-8)
        tensor = tensor / norm

    return tensor


def resize_scalar(
    tensor: torch.Tensor,
    size: int | tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a 2D scalar field (e.g. depth map) to target size.

    Args:
        tensor: 2D tensor [H, W].
        size: Target size as int (square) or (H, W) tuple.
        mode: Interpolation mode ("bilinear", "bicubic", "nearest").

    Returns:
        Resized tensor [H', W'].
    """
    if isinstance(size, int):
        size = (size, size)

    # Add batch and channel dims for interpolate: [1, 1, H, W]
    x = tensor.unsqueeze(0).unsqueeze(0).float()
    x = torch.nn.functional.interpolate(
        x, size=size, mode=mode, align_corners=(mode != "nearest")
    )
    return x.squeeze(0).squeeze(0)


def resize_normal(
    tensor: torch.Tensor,
    size: int | tuple[int, int],
    mode: str = "bilinear",
    renormalize: bool = True,
) -> torch.Tensor:
    """Resize a 3-channel normal map to target size.

    Args:
        tensor: 3D tensor [3, H, W].
        size: Target size as int (square) or (H, W) tuple.
        mode: Interpolation mode ("bilinear", "bicubic", "nearest").
        renormalize: If True, re-normalize vectors to unit length after resize.

    Returns:
        Resized tensor [3, H', W'].
    """
    if isinstance(size, int):
        size = (size, size)

    x = tensor.unsqueeze(0).float()  # [1, 3, H, W]
    x = torch.nn.functional.interpolate(
        x, size=size, mode=mode, align_corners=(mode != "nearest")
    )
    result = x.squeeze(0)

    if renormalize:
        norm = result.norm(dim=0, keepdim=True).clamp_min(1e-8)
        result = result / norm

    return result


__all__ = [
    "find_matching_file",
    "load_scalar",
    "load_normal",
    "resize_scalar",
    "resize_normal",
]
