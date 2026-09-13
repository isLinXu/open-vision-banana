#!/usr/bin/env python3
"""Build Cityscapes 19-class RGB training pairs and a paper-eval val layout.

The local Cityscapes dump stores a flattened parquet export::

    leftImg8bit/train/cityscapes_XXXXXX.png
    gtFine/train/cityscapes_XXXXXX.png   # RGB-encoded labelIds 0-33

Paper evaluation (``CityscapesSegDataset``) needs the official names::

    leftImg8bit/val/<city>/*_leftImg8bit.png
    gtFine/val/<city>/*_gtFine_labelIds.png

This script writes both a 512px 19-color training split and a held-out val
tree in that official layout. It does not load a model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from open_vision_banana.data.datasets import CityscapesSegDataset
from open_vision_banana.data.manifest import fingerprint_records
from open_vision_banana.inference.prompt import PromptBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PALETTE = np.array(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)

LABEL_ID_TO_TRAIN_ID = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}

COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
for _lid, _tid in LABEL_ID_TO_TRAIN_ID.items():
    COLOR_LUT[_lid] = PALETTE[_tid]


def label_ids_from_gt(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.array(image)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array.astype(np.uint8, copy=False)


def colorize_label_ids(label_ids: np.ndarray) -> np.ndarray:
    return COLOR_LUT[label_ids]


def discover_flat_pairs(source: Path) -> list[tuple[Path, Path]]:
    image_dir = source / "leftImg8bit" / "train"
    label_dir = source / "gtFine" / "train"
    pairs: list[tuple[Path, Path]] = []
    if not image_dir.is_dir() or not label_dir.is_dir():
        return pairs
    for image_path in sorted(image_dir.glob("cityscapes_*.png")):
        label_path = label_dir / image_path.name
        if label_path.is_file():
            pairs.append((image_path, label_path))
    return pairs


def _resize_image(image: Image.Image, resolution: int) -> Image.Image:
    if image.size == (resolution, resolution):
        return image
    return image.resize((resolution, resolution), Image.Resampling.LANCZOS)


def _resize_labels(label_ids: np.ndarray, resolution: int) -> np.ndarray:
    if label_ids.shape == (resolution, resolution):
        return label_ids
    label_image = Image.fromarray(label_ids, mode="L")
    resized = label_image.resize((resolution, resolution), Image.Resampling.NEAREST)
    return np.array(resized, dtype=np.uint8)


def write_training_split(
    pairs: list[tuple[Path, Path]],
    output: Path,
    *,
    resolution: int,
) -> int:
    image_dir = output / "images"
    target_dir = output / "targets"
    image_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for image_path, label_path in pairs:
        stem = image_path.stem
        with Image.open(image_path) as image:
            rgb = _resize_image(image.convert("RGB"), resolution)
            rgb.save(image_dir / f"{stem}.png")
        label_ids = _resize_labels(label_ids_from_gt(label_path), resolution)
        Image.fromarray(colorize_label_ids(label_ids), mode="RGB").save(
            target_dir / f"{stem}.png"
        )
        written += 1
    return written


def write_paper_val_layout(
    pairs: list[tuple[Path, Path]],
    paper_root: Path,
    *,
    city: str = "export",
) -> int:
    image_dir = paper_root / "leftImg8bit" / "val" / city
    label_dir = paper_root / "gtFine" / "val" / city
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, (image_path, label_path) in enumerate(pairs):
        stem = f"{city}_{index:06d}_000000"
        with Image.open(image_path) as image:
            image.convert("RGB").save(image_dir / f"{stem}_leftImg8bit.png")
        Image.fromarray(label_ids_from_gt(label_path), mode="L").save(
            label_dir / f"{stem}_gtFine_labelIds.png"
        )
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/cityscapes"),
        help="Cityscapes root with leftImg8bit/train and gtFine/train",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/cityscapes_19class_512"),
        help="512px 19-color RGB training split (images/ + targets/)",
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=Path("data/benchmarks/cityscapes"),
        help="Official-layout root for evaluate_paper_benchmarks.py",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--val-count", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = discover_flat_pairs(args.source)
    if not pairs:
        logger.error("No flattened Cityscapes pairs under %s", args.source)
        return 1
    val_count = min(args.val_count, len(pairs))
    train_pairs = pairs[:-val_count] if val_count else pairs
    val_pairs = pairs[-val_count:] if val_count else []

    train_written = write_training_split(
        train_pairs, args.train_output, resolution=args.resolution
    )
    val_written = write_paper_val_layout(val_pairs, args.paper_root)
    class_prompt = PromptBuilder.build_semantic_prompt(
        CityscapesSegDataset.CITYSCAPES_COLORS
    )
    manifest = {
        "schema_version": 2,
        "source": args.source.name,
        "source_kind": "cityscapes_flat_export",
        "pairs_found": len(pairs),
        "train_pairs": train_written,
        "val_pairs": val_written,
        "resolution": args.resolution,
        "palette_classes": 19,
        "train_output": ".",
        "paper_root": Path(os.path.relpath(args.paper_root, args.train_output)).as_posix(),
    }
    pairs_path = args.train_output / "pairs.jsonl"
    records: list[dict] = []
    with pairs_path.open("w", encoding="utf-8") as handle:
        for image_path, _label_path in train_pairs:
            stem = image_path.stem
            record = {
                "task": "cityscapes",
                "input": f"images/{stem}.png",
                "target": f"targets/{stem}.png",
                "split": "train",
                "target_kind": "cityscapes_19_palette",
                "prompt": class_prompt,
            }
            records.append(record)
            handle.write(json.dumps(record) + "\n")
    manifest["pair_manifest"] = "pairs.jsonl"
    manifest["dataset_fingerprint"] = fingerprint_records(records, args.train_output)
    manifest["fingerprint_algorithm"] = "sha256:canonical-pairs-v1"
    (args.train_output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    val_records = [
        {
            "task": "semantic",
            "input": f"leftImg8bit/val/export/export_{index:06d}_000000_leftImg8bit.png",
            "target": f"gtFine/val/export/export_{index:06d}_000000_gtFine_labelIds.png",
            "split": "val",
        }
        for index in range(val_written)
    ]
    paper_manifest = {
        "schema_version": 2,
        "source": args.source.name,
        "source_kind": "cityscapes_flat_export",
        "samples": val_written,
        "split": "val",
        "dataset_fingerprint": fingerprint_records(val_records, args.paper_root),
        "fingerprint_algorithm": "sha256:canonical-pairs-v1",
    }
    (args.paper_root / "manifest.json").write_text(
        json.dumps(paper_manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Wrote %d train pairs to %s and %d val pairs to %s",
        train_written,
        args.train_output,
        val_written,
        args.paper_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
