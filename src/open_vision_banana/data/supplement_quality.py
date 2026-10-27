"""Deterministic removal of referring images that can lose their target in a crop."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from open_vision_banana.data.manifest import resolve_manifest_path
from open_vision_banana.res_preparation import _pixels


def minimum_crop_foreground(mask: Image.Image, resolution: int = 512) -> int:
    """Minimum foreground over every integer crop the existing loader can choose.

    Match the loader's nearest-neighbour resize and rounding exactly. The short
    side becomes resolution, so every crop can be checked with one sliding sum
    along the long side, including disconnected objects and thin resized masks.
    """
    width, height = mask.size
    scale = resolution / float(min(width, height))
    size = (max(resolution, round(width * scale)), max(resolution, round(height * scale)))
    rgb = np.asarray(mask.convert("RGB"))
    if not np.array_equal(rgb[..., 0], rgb[..., 1]) or not np.array_equal(rgb[..., 0], rgb[..., 2]):
        raise ValueError("Referring mask channels differ")
    if not np.isin(rgb[..., 0], [0, 255]).all():
        raise ValueError("Referring mask is not binary")
    resized = np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST)) == 255
    projection = resized.sum(axis=0 if size[1] == resolution else 1, dtype=np.int64)
    integral = np.concatenate(([0], projection.cumsum()))
    return int((integral[resolution:] - integral[:-resolution]).min())


def inspect_referring_row(row: dict, manifest: Path, resolution: int = 512) -> dict:
    image_path = resolve_manifest_path(row["input"], manifest)
    target_path = resolve_manifest_path(row["target"], manifest)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        pixels = _pixels(image)
        size = image.size
    with Image.open(target_path) as target:
        if target.size != size:
            raise ValueError("Referring image/mask dimensions differ")
        minimum = minimum_crop_foreground(target, resolution)
    coco_ids = set()
    for key in ("input", "image", "source_image"):
        match = re.fullmatch(
            r"(?:COCO_(?:train|val|test)20\d\d_)?0*(\d{1,12})\.(?:jpg|jpeg|png)",
            Path(str(row.get(key, ""))).name,
            re.IGNORECASE,
        )
        if match:
            coco_ids.add(int(match[1]))
    if "llmseg" in str(row.get("source_dataset", "")).lower() and row.get("image_id") is not None:
        coco_ids.add(int(row["image_id"]))
    return {
        "sample_id": row["sample_id"],
        "rgb_pixels_sha256": pixels,
        "coco_image_ids": sorted(coco_ids),
        "minimum_crop_foreground_pixels": minimum,
    }


def excluded_images(records: list[dict]) -> tuple[set[str], set[int]]:
    """Close rejection across identical pixels or COCO IDs, including shared sources."""
    bad_pixels = {
        r["rgb_pixels_sha256"] for r in records if not r["minimum_crop_foreground_pixels"]
    }
    bad_coco = {
        i for r in records if not r["minimum_crop_foreground_pixels"] for i in r["coco_image_ids"]
    }
    while True:
        previous = len(bad_pixels), len(bad_coco)
        for row in records:
            if row["rgb_pixels_sha256"] in bad_pixels or bad_coco.intersection(
                row["coco_image_ids"]
            ):
                bad_pixels.add(row["rgb_pixels_sha256"])
                bad_coco.update(row["coco_image_ids"])
        if previous == (len(bad_pixels), len(bad_coco)):
            return bad_pixels, bad_coco
