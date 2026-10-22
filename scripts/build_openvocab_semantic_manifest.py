#!/usr/bin/env python3
"""Build an open-vocabulary semantic-segmentation training manifest.

Turns any (image, integer-label-PNG) collection into prompt-defined RGB targets:
every sample gets its own class->color mapping and a matching prompt (JSON /
tuple / natural-language / hex styles), as in Fig. 2 of the paper. Several
variants per image re-sample the mapping so the model must read the prompt
instead of memorizing a palette.

Example (ADE20K SceneParse150: labels are 0=other, 1..150 = classes):

    python scripts/build_openvocab_semantic_manifest.py \
        --images data/ade20k/images/training \
        --labels data/ade20k/annotations/training \
        --classes configs/ade20k_150_classes.json \
        --source-name ade20k --label-offset 1 --ignore-index -1 \
        --out-dir data/openvocab_semantic_ade20k --variants 2

COCO-Stuff (stuffthingmaps PNG value = COCO id - 1, 255 = unlabelled, 11 unused ids):
        --images data/raw/coco2017/train2017 --labels data/raw/coco_stuff/train2017 \
        --classes configs/coco_stuff_182_labels.json --source-name coco_stuff \
        --label-offset 0 --ignore-index 255 --out-dir data/openvocab_semantic_cocostuff

``--classes`` is a JSON list of class names (index = label - label_offset); an
empty string keeps a gap in the id space and ignores those pixels.
Pixels whose (label - label_offset) is ``--ignore-index`` or out of range are
painted with the background color.

Zero-shot hygiene: benchmark sources (e.g. Cityscapes) are refused unless
``--allow-benchmark-source`` is passed, because the paper evaluates semantic
segmentation zero-shot.

The output ``pairs.jsonl`` is read by the unified trainer's
``ManifestVisionDataset`` (task ``semantic``; per-row ``prompt`` is honored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data.openvocab_semantic import sample_openvocab_semantic  # noqa: E402

BENCHMARK_SOURCES = frozenset({"cityscapes", "ade20k_val_bench", "refcocog", "sa-co", "saco"})
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def assert_not_benchmark(source_name: str, allow: bool) -> None:
    if source_name.strip().lower() in BENCHMARK_SOURCES and not allow:
        raise SystemExit(
            f"Refusing to train on benchmark source {source_name!r}: the paper evaluates "
            "zero-shot. Pass --allow-benchmark-source only for explicit in-domain gates "
            "(and never report those numbers as zero-shot)."
        )


def split_for(stem: str, val_fraction: float) -> str:
    """Deterministic split by image id so all variants of an image stay together."""
    bucket = int(hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    return "val" if bucket < val_fraction * 10_000 else "train"


def load_labels(
    path: Path,
    offset: int,
    ignore_index: int | None,
    num_classes: int,
    unused_indices: frozenset[int] = frozenset(),
) -> np.ndarray:
    with Image.open(path) as img:
        raw = np.asarray(img).astype(np.int64)
    if raw.ndim == 3:  # RGB-encoded label PNGs are not supported; need class indices
        raise ValueError(f"{path}: expected single-channel index PNG, got shape {raw.shape}")
    labels = raw - offset
    bad = (labels < 0) | (labels >= num_classes)
    if ignore_index is not None:
        bad |= labels == ignore_index
    # Class lists may keep gaps as "" so raw label ids stay addressable (COCO-Stuff's
    # 182-slot id space has 11 unused ids); those pixels are ignored.
    for index in unused_indices:
        bad |= labels == index
    labels[bad] = -1
    return labels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True, help="Dir of index PNGs named <image stem>.png")
    p.add_argument("--classes", type=Path, required=True, help="JSON list of class names")
    p.add_argument("--source-name", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--label-offset", type=int, default=0)
    p.add_argument("--ignore-index", type=int, default=None)
    p.add_argument("--variants", type=int, default=2, help="Mappings sampled per image")
    p.add_argument("--val-fraction", type=float, default=0.02)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--max-present", type=int, default=12)
    p.add_argument("--max-absent", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-benchmark-source", action="store_true")
    args = p.parse_args()

    assert_not_benchmark(args.source_name, args.allow_benchmark_source)
    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or not all(isinstance(n, str) for n in class_names):
        raise SystemExit("--classes must be a JSON list of strings ('' marks an unused label id)")
    unused_indices = frozenset(i for i, n in enumerate(class_names) if not n)

    images = sorted(f for f in args.images.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images under {args.images}")

    targets_dir = args.out_dir / "targets"
    targets_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "pairs.jsonl"
    style_counts: dict[str, int] = {}
    rows = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as out:
        for image_path in images:
            label_path = args.labels / f"{image_path.stem}.png"
            if not label_path.is_file():
                skipped += 1
                continue
            labels = load_labels(label_path, args.label_offset, args.ignore_index, len(class_names), unused_indices)
            split = split_for(image_path.stem, args.val_fraction)
            for variant in range(args.variants):
                # Seed from (global seed, image, variant): reproducible and order-independent.
                digest = hashlib.sha256(f"{args.seed}:{image_path.stem}:{variant}".encode()).hexdigest()
                rng = random.Random(int(digest[:16], 16))
                sample = sample_openvocab_semantic(
                    labels,
                    class_names,
                    rng,
                    ignore_index=-1,
                    max_present=args.max_present,
                    max_absent=args.max_absent,
                )
                target_path = targets_dir / f"{image_path.stem}_v{variant}.png"
                Image.fromarray(sample.target_rgb).save(target_path)
                record = {
                    "task": "semantic",
                    "split": split,
                    "source": args.source_name,
                    "input": os.path.relpath(image_path.resolve(), manifest_path.parent.resolve()),
                    "target": os.path.relpath(target_path.resolve(), manifest_path.parent.resolve()),
                    "prompt": sample.prompt,
                    "style": sample.style,
                    "mapping": {k: list(v) for k, v in sample.mapping.items()},
                    "absent_classes": list(sample.absent_classes),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                style_counts[sample.style] = style_counts.get(sample.style, 0) + 1
                rows += 1

    summary = {
        "source": args.source_name,
        "images": len(images) - skipped,
        "skipped_missing_label": skipped,
        "rows": rows,
        "variants": args.variants,
        "seed": args.seed,
        "styles": style_counts,
        "manifest": str(manifest_path),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
