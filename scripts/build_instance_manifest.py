#!/usr/bin/env python3
"""Build ``instance`` training rows from COCO or LVIS annotations.

Paper Section 3.1 (Instance Segmentation): the model is prompted with one target
class and a background colour, and must colour every instance of that class with
its own solid colour; masks are recovered with the Appendix A clustering. This
script renders that supervision from public annotations:

* one row per (image, category) pair that has at least ``--min-instances``
  annotated instances;
* target PNG: black background, each instance a distinct colour drawn from a
  palette whose pairwise RGB distance is far above the decoder tolerance tau=14;
  instances are painted largest-first so small objects stay on top of overlaps;
* prompt: ``PromptBuilder.build_instance_prompt(category)`` (Fig. 3 template);
* ambiguous supervision is skipped: COCO ``iscrowd`` regions of the category and
  LVIS categories listed in the image's ``not_exhaustive_category_ids`` (not every
  instance annotated), so the target never under-labels the prompted class.

Zero-shot protocol: RefCOCOg UMD val images are COCO train2014 files. Pass
``--exclude-image-ids refs(umd).p`` (same formats as
``build_qwen_baseline_manifest.py``) to drop them; the audit records how many.

Example::

    python scripts/build_instance_manifest.py \
        --annotations /data/coco/annotations/instances_train2017.json \
        --images-dir /data/coco/train2017 --data-root /data \
        --targets-dir /data/instance_targets/coco \
        --exclude-image-ids /data/refcocog/refs(umd).p \
        --output /data/qwen_baseline/instance_coco.jsonl --val-fraction 0.01

    python scripts/build_instance_manifest.py --format lvis \
        --annotations /data/lvis/lvis_v1_train.json --images-dir /data/coco \
        --data-root /data --targets-dir /data/instance_targets/lvis \
        --exclude-image-ids /data/refcocog/refs(umd).p --output /data/qwen_baseline/instance_lvis.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from open_vision_banana.inference.prompt import PromptBuilder  # noqa: E402
from scripts.build_qwen_baseline_manifest import (  # noqa: E402
    coco_image_id_from_path,
    load_excluded_image_ids,
)

BACKGROUND = (0, 0, 0)
# Minimum pairwise distance between instance colours and from the background. The
# decoder merges colours within tau=14 (Appendix A); generated images drift by a few
# levels, so the palette keeps a wide margin.
MIN_COLOR_DISTANCE = 60.0


def build_instance_palette(min_distance: float = MIN_COLOR_DISTANCE) -> list[tuple[int, int, int]]:
    """Greedy, deterministic palette: 4 levels per channel, far from black and each other."""
    levels = (255, 128, 192, 64)
    candidates = [c for c in itertools.product(levels, repeat=3)]
    # Saturated colours first so the most common small instance counts get the most
    # separable colours.
    candidates.sort(key=lambda c: (-(max(c) - min(c)), -sum(c)))
    palette: list[tuple[int, int, int]] = []
    for color in candidates:
        if _dist(color, BACKGROUND) < min_distance:
            continue
        if all(_dist(color, chosen) >= min_distance for chosen in palette):
            palette.append(color)
    return palette


def _dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True))))


PALETTE = build_instance_palette()


def ann_to_mask(ann: dict, height: int, width: int) -> np.ndarray:
    """Decode a COCO/LVIS segmentation (polygons or RLE) to a boolean mask."""
    from pycocotools import mask as mask_utils

    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def render_instance_target(
    anns: list[dict], height: int, width: int, rng: random.Random
) -> tuple[np.ndarray, int]:
    """Paint instances largest-first with distinct palette colours on black.

    Returns the HxWx3 uint8 target and the number of instances that kept at least
    one visible pixel after occlusion by smaller instances painted on top.
    """
    colors = PALETTE[: len(anns)]
    rng.shuffle(colors)
    target = np.zeros((height, width, 3), dtype=np.uint8)
    masks = [ann_to_mask(ann, height, width) for ann in anns]
    order = sorted(range(len(anns)), key=lambda i: -int(masks[i].sum()))
    for color, index in zip(colors, order, strict=True):
        target[masks[index]] = color
    visible = sum(
        1 for color in colors if np.any(np.all(target == np.asarray(color, np.uint8), axis=-1))
    )
    return target, visible


def _image_file_name(image: dict, fmt: str) -> str:
    if fmt == "lvis":
        # LVIS v1 images carry coco_url like .../train2017/000000391895.jpg
        url = image.get("coco_url") or image.get("file_name", "")
        name = Path(url).name
        subdir = Path(url).parent.name if "/" in url else ""
        return f"{subdir}/{name}" if subdir in {"train2017", "val2017"} else name
    return image["file_name"]


def build_instance_manifest(
    annotations: Path,
    images_dir: Path,
    targets_dir: Path,
    output: Path,
    *,
    fmt: str = "coco",
    data_root: Path | None = None,
    min_instances: int = 1,
    max_instances: int | None = None,
    max_pairs_per_image: int | None = None,
    max_rows: int | None = None,
    val_fraction: float = 0.0,
    exclude_image_ids: tuple[Path, ...] | list[Path] = (),
    seed: int = 42,
    require_images: bool = True,
) -> dict:
    """Render instance targets and write manifest rows; return an audit summary."""
    if fmt not in {"coco", "lvis"}:
        raise ValueError("fmt must be 'coco' or 'lvis'")
    if min_instances < 1:
        raise ValueError("min_instances must be >= 1")
    max_instances = len(PALETTE) if max_instances is None else min(max_instances, len(PALETTE))
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")

    rng = random.Random(seed)
    payload = json.loads(Path(annotations).read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in payload["categories"]}
    images = {img["id"]: img for img in payload["images"]}
    excluded_ids = load_excluded_image_ids(exclude_image_ids)

    by_image_cat: dict[tuple[int, int], list[dict]] = defaultdict(list)
    crowd_pairs: set[tuple[int, int]] = set()
    for ann in payload["annotations"]:
        key = (ann["image_id"], ann["category_id"])
        if ann.get("iscrowd", 0):
            crowd_pairs.add(key)
            continue
        if not ann.get("segmentation"):
            continue
        by_image_cat[key].append(ann)

    stats = defaultdict(int)
    rows: list[dict] = []
    pairs_per_image: dict[int, int] = defaultdict(int)
    anchor = data_root.resolve() if data_root else None
    targets_dir.mkdir(parents=True, exist_ok=True)

    for (image_id, cat_id), anns in sorted(by_image_cat.items()):
        image = images.get(image_id)
        if image is None:
            stats["skipped_unknown_image"] += 1
            continue
        if image_id in excluded_ids or (
            coco_image_id_from_path(_image_file_name(image, fmt)) in excluded_ids
        ):
            stats["rejected_excluded_image_id"] += 1
            continue
        if (image_id, cat_id) in crowd_pairs:
            stats["skipped_crowd_pairs"] += 1
            continue
        if fmt == "lvis" and cat_id in set(image.get("not_exhaustive_category_ids", [])):
            stats["skipped_not_exhaustive_pairs"] += 1
            continue
        if len(anns) < min_instances:
            stats["skipped_too_few_instances"] += 1
            continue
        if len(anns) > max_instances:
            stats["skipped_too_many_instances"] += 1
            continue
        if max_pairs_per_image is not None and pairs_per_image[image_id] >= max_pairs_per_image:
            stats["skipped_pair_cap"] += 1
            continue

        rel_image = Path(_image_file_name(image, fmt))
        image_path = images_dir / rel_image
        if require_images and not image_path.is_file():
            stats["skipped_missing_image"] += 1
            continue

        height, width = int(image["height"]), int(image["width"])
        target, visible = render_instance_target(anns, height, width, rng)
        if visible < min_instances:
            stats["skipped_fully_occluded"] += 1
            continue
        name = categories[cat_id]
        slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
        target_path = targets_dir / f"{image_id:012d}_{cat_id}_{slug}.png"
        Image.fromarray(target).save(target_path, compress_level=1)

        def rel(path: Path) -> str:
            if anchor is None:
                return str(path)
            try:
                return str(path.resolve().relative_to(anchor))
            except ValueError:
                return str(path)

        rows.append(
            {
                "task": "instance",
                "input": rel(image_path),
                "target": rel(target_path),
                "prompt": PromptBuilder.build_instance_prompt(name),
                "category": name,
                "num_instances": len(anns),
                "source_dataset": fmt,
                "image_id": image_id,
            }
        )
        pairs_per_image[image_id] += 1
        if max_rows is not None and len(rows) >= max_rows:
            break

    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_fraction))
    for index, row in enumerate(rows):
        row["split"] = "val" if index < n_val else "train"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    instance_hist: dict[str, int] = defaultdict(int)
    for row in rows:
        instance_hist[str(row["num_instances"])] += 1
    audit = {
        "output_manifest": str(output.resolve()),
        "format": fmt,
        "rows_written": len(rows),
        "train_rows": len(rows) - n_val,
        "val_rows": n_val,
        "images_used": len(pairs_per_image),
        "categories_used": len({row["category"] for row in rows}),
        "instances_per_row_histogram": dict(sorted(instance_hist.items(), key=lambda kv: int(kv[0]))),
        "palette_size": len(PALETTE),
        "excluded_image_id_count": len(excluded_ids),
        **{k: int(v) for k, v in sorted(stats.items())},
    }
    output.with_suffix(".instance_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--format", choices=("coco", "lvis"), default="coco")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--min-instances", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-pairs-per-image", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--exclude-image-ids", action="append", type=Path, default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-require-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_instance_manifest(
        args.annotations,
        args.images_dir,
        args.targets_dir,
        args.output,
        fmt=args.format,
        data_root=args.data_root,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        max_pairs_per_image=args.max_pairs_per_image,
        max_rows=args.max_rows,
        val_fraction=args.val_fraction,
        exclude_image_ids=args.exclude_image_ids,
        seed=args.seed,
        require_images=not args.no_require_images,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
