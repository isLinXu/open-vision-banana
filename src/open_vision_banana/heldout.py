"""Held-out benchmark identity shared by the manifest builders and the RES source adapters.

``benchmark_leak`` names the evaluated benchmark a row's paths or source image name come from;
``coco_image_id_from_path`` recovers a COCO image id from a COCO-style file name, so a row can be
checked against an evaluation image-id list.
"""

from __future__ import annotations

import re
from pathlib import Path

# Paper zero-shot transfer protocol: no training data from any evaluated benchmark.
# Tables 2/4/5/6/7 evaluate Cityscapes val, RefCOCOg, ReasonSeg, NYU, iBims-1, ETH3D,
# DIODE, KITTI, nuScenes, ScanNet, and Virtual KITTI 2. Rows whose input/target
# paths come from these corpora are rejected unless the caller explicitly allows
# a benchmark (e.g. Cityscapes *train* when the run is not claimed as zero-shot).
ZERO_SHOT_BENCHMARK_PATTERNS: dict[str, tuple[str, ...]] = {
    # Table 2 evaluates Cityscapes val; its *train* split is only admissible in the
    # in-domain protocol (review item 1: the strict table used to omit it).
    "cityscapes": ("cityscapes", "leftimg8bit", "gtfine"),
    "vkitti": ("vkitti", "virtual_kitti", "virtualkitti"),
    "nyu": ("nyu_depth", "nyuv2", "nyu-depth", "nyu/"),
    "ibims": ("ibims",),
    "eth3d": ("eth3d",),
    "diode": ("diode",),
    "kitti": ("kitti_raw", "kitti/", "kitti_depth", "data_depth_annotated"),
    "nuscenes": ("nuscenes",),
    "scannet": ("scannet/", "scannetv2", "scannet_v2"),
    "refcocog": ("refcocog", "refcoco"),
    "reasonseg": ("reasonseg",),
    "sa_co": ("saco", "sa-co", "sa_co"),
}


def benchmark_leak(row: dict, allowed: set[str]) -> str | None:
    """Return the benchmark name a manifest row leaks from, or None."""
    haystack = " ".join(
        str(row.get(k, ""))
        for k in ("input", "target", "image", "source", "dataset", "source_image")
    ).lower()
    # An explicitly allowed corpus wins even if its path also contains a
    # shorter pattern (e.g. "virtual_kitti/" also contains "kitti/").
    for name in allowed:
        if any(p in haystack for p in ZERO_SHOT_BENCHMARK_PATTERNS[name]):
            return None
    for name, patterns in ZERO_SHOT_BENCHMARK_PATTERNS.items():
        if any(p in haystack for p in patterns):
            return name
    return None


# COCO file names only: COCO_train2014_000000581921.jpg, coco_000000581921.jpg (Rea2Seg
# spatial) or the 12-digit 000000581921.jpg.
# Short numeric names (Hypersim frame.0001.png, Cityscapes *_000019_leftImg8bit.png)
# deliberately do not parse, so an exclusion list cannot hit non-COCO sources.
_COCO_IMAGE_ID_RE = re.compile(
    r"^(?:COCO_(?:train|val|test)20\d{2}_|coco_)?(\d{12})\.(?:jpg|jpeg|png)$", re.IGNORECASE
)


def coco_image_id_from_path(path: str) -> int | None:
    """Return the COCO image id encoded in a COCO-style file name, or None."""
    match = _COCO_IMAGE_ID_RE.match(Path(str(path)).name)
    return int(match.group(1)) if match else None
