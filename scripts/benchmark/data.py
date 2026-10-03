"""Deterministic data selection; metric GT never comes from model output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import checked_id


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_samples(spec):
    kind = spec["kind"]
    if kind == "synthetic":
        return [{"id": "synthetic-{:03d}".format(i), "synthetic": True,
                 "gt": {}} for i in range(min(spec.get("synthetic_count", 2),
                                               spec["max_samples"]))]
    if kind == "images":
        root = Path(spec["root"])
        if not root.is_dir():
            raise FileNotFoundError("Dataset root not found: {}".format(root))
        patterns = spec.get("include", ["*.jpg", "*.jpeg", "*.png"])
        paths = sorted({p.resolve() for pattern in patterns for p in root.glob(pattern)
                        if p.is_file()}, key=lambda p: str(p).lower())
        # An explicit filename is a contract: don't silently drop missing samples.
        for pattern in patterns:
            if not any(c in pattern for c in "*?[") and not (root / pattern).is_file():
                raise FileNotFoundError(root / pattern)
        rows = [{"id": checked_id(p.stem, "sample id"), "image": str(p), "gt": {}}
                for p in paths[:spec["max_samples"]]]
    else:
        manifest = Path(spec["manifest"])
        rows = []
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) - {"id", "image", "gt", "valid_masks"}:
                raise ValueError("Unknown manifest fields at line {}".format(line_number))
            checked_id(row.get("id"), "sample id")
            row["image"] = str((manifest.parent / row["image"]).resolve())
            for group in ("gt", "valid_masks"):
                row[group] = {k: str((manifest.parent / v).resolve())
                              for k, v in row.get(group, {}).items()}
            rows.append(row)
        rows = sorted(rows, key=lambda r: r["id"])[:spec["max_samples"]]
    if not rows:
        raise ValueError("Dataset selected zero samples")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs; use a manifest with distinct IDs")
    for row in rows:
        for path in [row["image"]] + list(row["gt"].values()) + list(row.get("valid_masks", {}).values()):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return rows


def resized_hw(height, width, size):
    """Fit max edge preserving aspect ratio, then floor dimensions to alignment.

    No padding/cropping. Alignment can slightly change the aspect ratio; actual
    dimensions are recorded. Small source images are deliberately upsampled.
    """
    scale = size["max_edge"] / float(max(height, width))
    alignment = size.get("alignment", 1)
    return tuple(max(alignment, int(round(n * scale)) // alignment * alignment)
                 for n in (height, width))


def load_image(row):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    if row.get("synthetic"):
        yy, xx = np.mgrid[0:48, 0:64]
        pixels = np.stack([xx / 63.0, yy / 47.0, (xx + yy) / 110.0], axis=-1)
        image = Image.fromarray(np.round(pixels * 255).astype("uint8"), "RGB")
    else:
        with Image.open(row["image"]) as stream:
            image = ImageOps.exif_transpose(stream).convert("RGB")
    pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return image, tensor


def load_target(path, task):
    import numpy as np
    import torch
    from PIL import Image

    path = Path(path)
    if task == "semantic":
        if path.suffix.lower() == ".npy":
            array = np.load(str(path), allow_pickle=False)
        else:
            with Image.open(path) as stream:
                array = np.array(stream, copy=True)
        if array.ndim != 2 or array.dtype.kind not in "iu":
            raise ValueError("Semantic GT must be an integer HxW label map, not RGB")
        return torch.from_numpy(array.astype(np.int64)).unsqueeze(0).unsqueeze(0)
    array = np.load(str(path), allow_pickle=False)
    if task == "normal":
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Normal GT must be float HxWx3 camera-space vectors in .npy")
        return torch.from_numpy(array.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    if array.ndim != 2:
        raise ValueError("Scalar GT must be HxW .npy")
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def load_valid_mask(path, spatial_shape):
    import numpy as np
    import torch
    from PIL import Image

    if not path:
        return None
    if Path(path).suffix.lower() == ".npy":
        array = np.load(str(path), allow_pickle=False)
    else:
        with Image.open(path) as stream:
            array = np.array(stream, copy=True)
    if tuple(array.shape) != tuple(spatial_shape):
        raise ValueError("Valid mask must match GT spatial shape")
    return torch.from_numpy(array.astype(bool)).unsqueeze(0).unsqueeze(0)
