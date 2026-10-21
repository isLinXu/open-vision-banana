#!/usr/bin/env python3
"""Paint official Cityscapes 19-color targets and write a training manifest.

The SNVC ADE-150 palette is a different alphabet. Paper Cityscapes mIoU needs
the official trainId colors used by ``CityscapesSegDataset``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from open_vision_banana.data.datasets import CityscapesSegDataset
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT
from scripts.setup_cityscapes_official import (
    OFFICIAL_TRAIN_PAIRS,
    OFFICIAL_VAL_PAIRS,
    official_split_pairs,
    verify_official_train,
    verify_official_val,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cityscapes19_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cityscapes19"),
        help="Directory for colorized targets and pairs.jsonl.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
    )
    return parser.parse_args()


def labelids_to_trainid_rgb(label_path: Path, resolution: int) -> Image.Image:
    with Image.open(label_path) as label_img:
        label = np.asarray(label_img, dtype=np.int32)
    mapped = np.full(label.shape, 255, dtype=np.int32)
    for label_id, train_id in CityscapesSegDataset.CITYSCAPES_LABEL_TO_TRAIN.items():
        mapped[label == label_id] = train_id
    resized = np.array(
        Image.fromarray(mapped.astype(np.uint8), mode="L").resize(
            (resolution, resolution), Image.Resampling.NEAREST
        ),
        dtype=np.int32,
    )
    palette = np.zeros((256, 3), dtype=np.uint8)
    for train_id, color in enumerate(CityscapesSegDataset.CITYSCAPES_COLORS.values()):
        palette[train_id] = color
    rgb = palette[resized]
    return Image.fromarray(rgb, mode="RGB")


def main() -> int:
    args = parse_args()
    root = args.cityscapes_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = tuple(dict.fromkeys(args.splits))

    expected = {"train": OFFICIAL_TRAIN_PAIRS, "val": OFFICIAL_VAL_PAIRS}
    if "val" in splits:
        verify_official_val(root)
    if "train" in splits:
        verify_official_train(root)

    rows: list[dict[str, str | int]] = []
    for split in splits:
        pairs = official_split_pairs(root, split)
        target_dir = output / "targets" / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in pairs:
            target_path = target_dir / image_path.name.replace("leftImg8bit", "gtFine_color")
            if not target_path.is_file():
                labelids_to_trainid_rgb(label_path, args.resolution).save(target_path)
            rows.append(
                {
                    "input": str(image_path),
                    "target": str(target_path),
                    "task": "semantic",
                    "split": split,
                    "prompt": SEMANTIC_SEGMENTATION_PROMPT,
                    "palette": "cityscapes_19_official",
                }
            )
        logger.info("%s: wrote %d official 19-color pairs", split, len(pairs))
        if len(pairs) != expected[split]:
            raise SystemExit(f"{split} has {len(pairs)} pairs; expected {expected[split]}")

    manifest = output / "pairs.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "palette_source": "cityscapes_19_official",
        "n_classes": 19,
        "prompt": SEMANTIC_SEGMENTATION_PROMPT,
        "resolution": args.resolution,
        "splits": {split: expected[split] for split in splits},
        "note": "Do not mix ADE-150 / SNVC seg targets into this manifest.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d rows)", manifest, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
