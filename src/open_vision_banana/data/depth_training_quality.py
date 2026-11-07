"""Native depth admission and verified crop geometry for training only."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from .clean_depth import TRAIN_SPLITS, assess_depth_row, within_invalid_fraction

logger = logging.getLogger(__name__)
DEPTH_TASKS = frozenset({"depth", "metric_depth", "depth_estimation", "depth_edge"})


class DepthGeometryRejectedError(ValueError):
    """A native-clean frame has no clean geometry at the requested resolution."""


def check_training_dataset(dataset) -> None:
    """Reject unverified depth adapters at the legacy tuner boundary, not evaluation."""
    children = []
    for name in ("datasets", "vision_datasets", "task_datasets"):
        collection = getattr(dataset, name, ()) or ()
        children.extend(collection.values() if isinstance(collection, Mapping) else collection)
    for name in ("dataset", "generation_dataset"):
        child = getattr(dataset, name, None)
        if child is not None:
            children.append(child)
    for child in children:
        check_training_dataset(child)
    if children:
        return
    tasks = {getattr(dataset, "task", None), getattr(dataset, "TASK_TYPE", None)}
    for row in getattr(dataset, "samples", ()):
        if isinstance(row, dict):
            tasks.add(row.get("task"))
    if DEPTH_TASKS.intersection(tasks) and not getattr(dataset, "depth_quality_verified", False):
        require_depth_manifest(tasks, type(dataset).__name__)


def require_depth_manifest(tasks, route: str) -> None:
    if DEPTH_TASKS.intersection(tasks):
        raise ValueError(
            f"Depth quality cannot verify {route}. Use --pair-manifest with "
            "native metric_depth and/or valid_mask paths (maximum 2% holes)."
        )


def admit_depth_training_row(row, root, resolution, *, aspect_preserving_crop=False):
    """None rejects native-dirty frames; () selects verified full-frame fallback.

    Nonempty offsets identify every clean square crop after nearest resizing.
    Sample only these offsets, without rejecting a native-clean frame merely
    because some random crops would contain too many holes.
    """
    source_split = str(row.get("source_split", "train")).strip().lower()
    if source_split not in TRAIN_SPLITS:
        raise ValueError(
            f"Depth quality: source_split={source_split!r} cannot enter training "
            "even when the manifest split is train."
        )
    mask, stats = assess_depth_row(row, Path(root))
    if not stats["passes"]:
        logger.info(
            "Depth quality rejected %s: %s/%s native invalid pixels",
            row.get("input"),
            stats["invalid_pixels"],
            stats["total_pixels"],
        )
        return None
    r = resolution
    h, w = mask.shape
    size = (r, r)
    if aspect_preserving_crop:
        scale = r / float(min(w, h))
        size = (max(r, round(w * scale)), max(r, round(h * scale)))
    valid = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))
    projection = (~valid).sum(axis=0 if size[1] == r else 1, dtype=np.int64)
    integral = np.concatenate(([0], projection.cumsum()))
    counts = integral[r:] - integral[:-r]
    offsets = tuple(
        i for i, count in enumerate(counts) if within_invalid_fraction(int(count), r * r)
    )
    if offsets:
        return offsets
    full = np.asarray(Image.fromarray(mask).resize((r, r), Image.Resampling.NEAREST))
    if not within_invalid_fraction(int((~full).sum()), r * r):
        raise DepthGeometryRejectedError(
            f"Depth quality: {row.get('input')} has no clean crop or full-frame "
            "resize at this resolution; choose another resolution."
        )
    logger.warning(
        "Depth quality: %s uses verified full-frame resize; no clean square crop", row.get("input")
    )
    return ()
