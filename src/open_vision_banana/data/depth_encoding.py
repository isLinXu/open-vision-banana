"""Audit RGB depth targets and build prompts that match their actual colormap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..core.depth_mapper import DepthRGBMapper, get_available_colormaps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DepthEncodingAudit:
    detected_colormap: str
    mean_curve_mae: float
    candidate_curve_mae: dict[str, float]
    sample_count: int
    statistics_sample_count: int
    channel_mean: tuple[float, float, float]
    channel_std: tuple[float, float, float]

    @property
    def is_curve_consistent(self) -> bool:
        return self.mean_curve_mae < 0.05


def audit_depth_target_encoding(
    target_dir: str | Path,
    *,
    candidates: list[str] | None = None,
    max_images: int = 16,
    resolution: int = 64,
    allowed_names: set[str] | None = None,
) -> DepthEncodingAudit:
    """Identify which registered depth colormap best fits RGB target pixels."""
    directory = Path(target_dir)
    if max_images <= 0 or resolution <= 0:
        raise ValueError("max_images and resolution must be positive")
    allowed_stems = (
        {Path(name).stem for name in allowed_names}
        if allowed_names is not None
        else None
    )
    paths = [
        path
        for path in sorted(directory.iterdir())
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and (allowed_stems is None or path.stem in allowed_stems)
        )
    ]
    if not paths:
        raise FileNotFoundError(f"No depth targets found in {directory}")
    curve_paths = paths[:max_images]

    candidate_names = candidates or get_available_colormaps()
    if not candidate_names:
        raise ValueError("candidates must be non-empty")
    totals = {name: 0.0 for name in candidate_names}
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    with torch.no_grad():
        for path in paths:
            image = Image.open(path).convert("RGB").resize(
                (resolution, resolution),
                Image.Resampling.LANCZOS,
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
            flattened = array.reshape(-1, 3).astype(np.float64)
            channel_sum += flattened.sum(axis=0)
            channel_square_sum += np.square(flattened).sum(axis=0)
            pixel_count += flattened.shape[0]
            if path not in curve_paths:
                continue
            rgb = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
            for name in candidate_names:
                mapper = DepthRGBMapper(colormap=name)
                normalized = mapper.rgb_to_normalized(rgb)
                projected = mapper.normalized_to_rgb(normalized)
                totals[name] += float((rgb - projected).abs().mean())

    means = {name: total / len(curve_paths) for name, total in totals.items()}
    detected = min(means, key=means.get)
    channel_mean = channel_sum / pixel_count
    channel_variance = channel_square_sum / pixel_count - np.square(channel_mean)
    channel_std = np.sqrt(np.maximum(channel_variance, 0.0))
    return DepthEncodingAudit(
        detected_colormap=detected,
        mean_curve_mae=means[detected],
        candidate_curve_mae=means,
        sample_count=len(curve_paths),
        statistics_sample_count=len(paths),
        channel_mean=tuple(float(value) for value in channel_mean),
        channel_std=tuple(float(value) for value in channel_std),
    )


def relative_depth_prompt(colormap: str) -> str:
    """Return a truthful prompt for a relative-depth RGB target."""
    if colormap == "viridis":
        return (
            "Generate a relative depth map of this image using the viridis RGB "
            "palette, from dark purple and blue through green to yellow."
        )
    return (
        "Generate a relative depth map of this image using the "
        f"{colormap} RGB palette."
    )


__all__ = [
    "DepthEncodingAudit",
    "audit_depth_target_encoding",
    "relative_depth_prompt",
]
