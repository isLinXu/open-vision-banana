#!/usr/bin/env python3
"""Build annotation-backed, leakage-resistant MAR-VB task pairs.

The v5_full pack flattens every image in an annotation's ``image`` list into
the task image directory. For depth/normal records that includes the target
image, so glob-and-stem pairing can accidentally train on target images as
contexts. This script follows annotation semantics instead and writes a JSONL
manifest without changing the source dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pack_v5_full import (  # noqa: E402
    V5_ROOT,
    get_dataset_name,
    get_task_family,
    resolve_path,
)

TASKS = ("depth", "normal", "seg", "structure")
TASK_TO_FAMILY = {
    "depth": "depth",
    "normal": "normal",
    "seg": "segmentation",
    "structure": "structure",
}
MAX_SEGMENTATION_SOURCE_PIXELS = 50_000_000


def _link_path(orig: str, dataset: str, family: str) -> Path | None:
    actual = resolve_path(orig, dataset)
    if actual is None:
        return None
    digest = hashlib.md5(orig.encode()).hexdigest()[:8]
    name = f"{dataset}_{digest}_{Path(actual).name}"
    return V5_ROOT / "images" / family / name


def _target_link_path(orig: str, dataset: str, family: str) -> Path | None:
    actual = resolve_path(orig, dataset)
    if actual is None:
        return None
    digest = hashlib.md5(orig.encode()).hexdigest()[:8]
    name = f"{dataset}_{digest}_{Path(actual).name}"
    return V5_ROOT / "targets" / family / name


def _is_color_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            # Palette images are valid RGB contexts after the loader converts
            # them; single-channel depth/gray images are not.
            return image.mode not in {"1", "L", "I", "F", "I;16"}
    except Exception:
        return False


def _depth_rows(ann_dir: Path) -> list[dict]:
    rows = []
    for ann in sorted(ann_dir.glob("*_sampled.jsonl")):
        if get_task_family(ann.name) != "depth":
            continue
        dataset = get_dataset_name(ann.name)
        for line_no, line in enumerate(ann.open(), 1):
            try:
                item = json.loads(line)
                images = item.get("image")
                if not isinstance(images, list) or not images:
                    continue
                source = images[0]
                inp = _link_path(source, dataset, "depth")
                if inp is None or not inp.exists() or not _is_color_image(inp):
                    continue
                # Targets were generated from each original RGB input and are
                # intentionally addressed by the input stem.
                target = V5_ROOT / "targets" / "depth" / f"{inp.stem}.png"
                if not target.exists():
                    continue
                rows.append({
                    "task": "depth", "input": str(inp), "target": str(target),
                    "source": str(source), "annotation": str(ann),
                    "line": line_no, "target_kind": "depth_pseudo_rgb",
                })
            except (json.JSONDecodeError, OSError):
                continue
    return rows


def _normal_rows(ann_dir: Path) -> list[dict]:
    rows = []
    for ann in sorted(ann_dir.glob("*_sampled.jsonl")):
        if get_task_family(ann.name) != "normal":
            continue
        dataset = get_dataset_name(ann.name)
        for line_no, line in enumerate(ann.open(), 1):
            try:
                item = json.loads(line)
                images = item.get("image")
                if not isinstance(images, list) or len(images) < 2:
                    continue
                source, target_source = images[0], images[1]
                inp = _link_path(source, dataset, "normal")
                target = _link_path(target_source, dataset, "normal")
                if (inp is None or target is None or not inp.exists()
                        or not target.exists() or not _is_color_image(inp)):
                    continue
                rows.append({
                    "task": "normal", "input": str(inp), "target": str(target),
                    "source": str(source), "target_source": str(target_source),
                    "annotation": str(ann), "line": line_no,
                    "target_kind": "source_normal_rgb",
                })
            except (json.JSONDecodeError, OSError):
                continue
    return rows


def _same_file(left: Path, right: Path) -> bool:
    """Return true when two manifest paths resolve to one physical file."""
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _pixel_count(path: Path) -> int | None:
    """Read dimensions without decoding an entire image into memory."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                return image.width * image.height
    except OSError:
        return None


def _seg_rows(ann_dir: Path) -> tuple[list[dict], Counter]:
    rows = []
    rejected: Counter = Counter()
    for ann in sorted(ann_dir.glob("*_sampled.jsonl")):
        if get_task_family(ann.name) != "segmentation":
            continue
        dataset = get_dataset_name(ann.name)
        for line_no, line in enumerate(ann.open(), 1):
            try:
                item = json.loads(line)
                source = item.get("image")
                target_source = item.get("seg") or item.get("mask")
                if not isinstance(source, str) or not isinstance(target_source, str):
                    continue
                inp = _link_path(source, dataset, "segmentation")
                target = _target_link_path(target_source, dataset, "segmentation")
                if (inp is None or target is None or not inp.exists()
                        or not target.exists() or not _is_color_image(inp)):
                    continue
                # The released v5 pack currently resolves every LoveDA
                # ``seg`` entry to its RGB context image.  Treating it as a
                # target silently turns the segmentation task into identity
                # reconstruction, so reject it rather than training on an
                # invalid supervision signal.
                if _same_file(inp, target):
                    rejected["input_target_same_file"] += 1
                    continue
                rows.append({
                    "task": "seg", "input": str(inp), "target": str(target),
                    "source": str(source), "target_source": str(target_source),
                    "annotation": str(ann), "line": line_no,
                    "target_kind": "source_segmentation_rgb",
                })
            except (json.JSONDecodeError, OSError):
                continue
    return rows, rejected


def _stem_paired_seg_rows() -> tuple[list[dict], Counter]:
    """Collect explicit RGB→mask pairs staged directly in v5_full.

    ``integrate_vb_seg_data.py`` links Vision-Banana semantic data into the
    segmentation directories but it intentionally has no matching SNVC JSONL
    annotation.  Its input and target names share a stem (the target's source
    name had a ``_rgb`` suffix stripped), so stem pairing is safe here.  It is
    deliberately kept separate from annotation pairs: generic stem matching
    remains unsafe for the depth and normal task families.
    """
    image_dir = V5_ROOT / "images" / "segmentation"
    target_dir = V5_ROOT / "targets" / "segmentation"
    rows = []
    rejected: Counter = Counter()
    if not image_dir.is_dir() or not target_dir.is_dir():
        rejected["segmentation_directories_missing"] += 1
        return rows, rejected

    image_by_stem = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    target_by_stem = {
        path.stem: path
        for path in target_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    for stem in sorted(image_by_stem.keys() & target_by_stem.keys()):
        inp, target = image_by_stem[stem], target_by_stem[stem]
        if _same_file(inp, target):
            rejected["input_target_same_file"] += 1
            continue
        inp_pixels, target_pixels = _pixel_count(inp), _pixel_count(target)
        if inp_pixels is None or target_pixels is None:
            rejected["unreadable_pair"] += 1
            continue
        if max(inp_pixels, target_pixels) > MAX_SEGMENTATION_SOURCE_PIXELS:
            rejected["source_too_large"] += 1
            continue
        rows.append({
            "task": "seg", "input": str(inp), "target": str(target),
            "source": inp.name, "target_source": target.name,
            "target_kind": "vision_banana_semantic_rgb",
        })
    rejected["unmatched_inputs"] = len(image_by_stem.keys() - target_by_stem.keys())
    rejected["unmatched_targets"] = len(target_by_stem.keys() - image_by_stem.keys())
    return rows, rejected


def _structure_rows() -> list[dict]:
    image_dir = V5_ROOT / "images" / "structure"
    target_dir = V5_ROOT / "targets" / "structure_aligned"
    rows = []
    for inp in sorted(image_dir.glob("*.png")):
        target = target_dir / f"{inp.stem}.png"
        if inp.exists() and target.exists() and _is_color_image(inp):
            rows.append({
                "task": "structure", "input": str(inp), "target": str(target),
                "source": inp.name, "target_kind": "aligned_edge_rgb",
            })
    return rows


def _assign_split(rows: list[dict], val_percent: int) -> None:
    # Use resolved input paths as the grouping key so the same RGB image cannot
    # land in train for one task and validation for another task.
    for row in rows:
        group = os.path.realpath(row["input"])
        bucket = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) % 100
        row["split"] = "val" if bucket < val_percent else "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/mar_vb_clean_pairs_v1/pairs.jsonl"),
    )
    parser.add_argument("--val-percent", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.val_percent <= 50:
        raise SystemExit("--val-percent must be between 1 and 50")

    ann_dir = V5_ROOT / "annotations"
    annotation_seg_rows, annotation_seg_rejected = _seg_rows(ann_dir)
    paired_seg_rows, paired_seg_rejected = _stem_paired_seg_rows()
    seg_rows = annotation_seg_rows + paired_seg_rows
    rows = _depth_rows(ann_dir) + _normal_rows(ann_dir) + seg_rows + _structure_rows()
    _assign_split(rows, args.val_percent)
    rows.sort(key=lambda r: (r["task"], r["split"], r["input"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = defaultdict(Counter)
    for row in rows:
        counts[row["task"]][row["split"]] += 1
    audit = {
        "manifest": str(args.output.resolve()),
        "source_root": str(V5_ROOT),
        "val_percent": args.val_percent,
        "total": len(rows),
        "counts": {task: dict(counts[task]) for task in TASKS},
        "target_kinds": dict(Counter(row["target_kind"] for row in rows)),
        "segmentation_sources": {
            "annotation_pairs": len(annotation_seg_rows),
            "stem_paired_semantic_rgb": len(paired_seg_rows),
        },
        "segmentation_rejected": {
            "annotation_pairs": dict(annotation_seg_rejected),
            "stem_paired": dict(paired_seg_rejected),
        },
        "input_depth_png": sum(Path(row["input"]).name.endswith("_depth.png") for row in rows),
        "max_segmentation_source_pixels": MAX_SEGMENTATION_SOURCE_PIXELS,
        "input_modes_checked": "non-single-channel via PIL metadata",
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
