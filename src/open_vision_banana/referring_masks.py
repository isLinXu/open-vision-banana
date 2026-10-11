"""Strict native COCO masks shared by RES preparation and benchmark loading."""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _compressed_counts(encoded: str | bytes) -> list[int]:
    """Validate COCO's signed, differential 5-bit RLE before native decoding.

    COCO maskApi.c rleFrString defines the format. Checking the decoded run
    lengths prevents malformed input from under/over-filling the native buffer.
    Pixel rasterization itself always uses pycocotools.
    """
    if isinstance(encoded, str):
        try:
            encoded = encoded.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("COCO RLE counts must be ASCII") from exc
    counts: list[int] = []
    value = shift = 0
    for byte in encoded:
        code = byte - 48
        if not 0 <= code <= 63 or shift > 30:
            raise ValueError("Invalid compressed COCO RLE counts")
        value |= (code & 31) << shift
        shift += 5
        if code & 32:
            continue
        if code & 16:
            value -= 1 << shift
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("Negative COCO RLE run length")
        counts.append(value)
        value = shift = 0
    if shift or not counts:
        raise ValueError("Truncated or empty compressed COCO RLE counts")
    return counts


def decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    """Return a nonempty native ``[height, width]`` boolean COCO mask.

    Supports the union of polygons and compressed/uncompressed RLE. Dimensions
    must come from the verified source image, never a bounding box. Polygon
    vertices outside the canvas retain COCO's official clipping behavior.
    Missing, malformed, dimension-mismatched and empty masks raise ValueError.
    """
    if any(not isinstance(n, Integral) or isinstance(n, bool) or n <= 0 for n in (height, width)):
        raise ValueError("COCO mask dimensions must be positive integers")
    height, width = int(height), int(width)
    if height * width > np.iinfo(np.uint32).max:
        raise ValueError("COCO mask dimensions exceed the RLE run-length range")
    try:
        from pycocotools import mask as coco_mask
    except ImportError as exc:
        raise ImportError(
            "Strict RES masks require pycocotools; install open-vision-banana[evaluation]"
        ) from exc

    if isinstance(segmentation, list) and segmentation:
        polygons = []
        for polygon in segmentation:
            coordinates = np.asarray(polygon)
            if (
                coordinates.ndim != 1
                or coordinates.size < 6
                or coordinates.size % 2
                or coordinates.dtype.kind not in "iuf"
                or not np.isfinite(coordinates).all()
            ):
                raise ValueError("COCO polygons require at least three finite (x, y) points")
            if np.abs(coordinates.astype(np.float64)).max() > np.iinfo(np.int32).max / 10:
                raise ValueError("COCO polygon coordinates exceed the rasterizer range")
            polygons.append(coordinates.astype(np.float64).tolist())
        rle = coco_mask.merge(coco_mask.frPyObjects(polygons, height, width))
    elif isinstance(segmentation, dict):
        size = segmentation.get("size")
        if (
            not isinstance(size, list | tuple)
            or len(size) != 2
            or any(not isinstance(n, Integral) or isinstance(n, bool) for n in size)
            or tuple(size) != (height, width)
        ):
            raise ValueError("COCO RLE size must match the native image dimensions")
        counts = segmentation.get("counts")
        if isinstance(counts, str | bytes):
            runs = _compressed_counts(counts)
        elif isinstance(counts, list) and counts:
            if any(not isinstance(n, Integral) or isinstance(n, bool) or n < 0 for n in counts):
                raise ValueError("COCO RLE counts must be nonnegative integers")
            runs = counts
        else:
            raise ValueError("COCO RLE counts must be a nonempty list or compressed string")
        if sum(runs) != height * width:
            raise ValueError("COCO RLE runs must cover exactly the native image pixels")
        # Normalize all RLE forms through official uncompressed-RLE conversion.
        rle = coco_mask.frPyObjects({"size": [height, width], "counts": runs}, height, width)
    else:
        raise ValueError("Missing or invalid COCO polygon/RLE segmentation")

    try:
        decoded = coco_mask.decode(rle)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid COCO segmentation encoding") from exc
    if decoded.shape != (height, width):
        raise ValueError("Decoded COCO mask shape differs from native image dimensions")
    if not decoded.any():
        raise ValueError("COCO segmentation decodes to an empty foreground mask")
    return np.ascontiguousarray(decoded, dtype=np.bool_)
