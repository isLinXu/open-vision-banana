#!/usr/bin/env python3
"""Fix P0-3: regenerate Cityscapes segmentation training targets in the
official 19-color palette used by the evaluation decoder.

Root cause: the existing targets in data/vb_snvc_train/segmentation/targets
are grayscale class-ID masks saved as JPEG (edge artifacts produce values up
to 42). The eval decoder (decode_semantic + CITYSCAPES_COLORS) expects the
official Cityscapes color PNGs — 0/38 overlap between the two conventions
makes mIoU ~ 0 no matter how well the model fits the training targets.

This script rebuilds the segmentation split directly from the HF parquet
export of Cityscapes (authoritative labelIds 0-33):
  image                 -> 384x384 JPG  (LANCZOS)
  semantic_segmentation -> labelId -> trainId -> palette color,
                           resized with NEAREST, saved as lossless PNG
Filenames keep the existing convention: cityscapes_cityscapes_{i:06d}.jpg/.png

The previous class-ID directory is preserved at
data/vb_snvc_train/segmentation_classid_backup/
"""

from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = Path("/Users/gatilin/MyWork/datasets/cityscapes/parquet")
SEG_DIR = PROJECT_ROOT / "data" / "vb_snvc_train" / "segmentation"
BACKUP_DIR = PROJECT_ROOT / "data" / "vb_snvc_train" / "segmentation_classid_backup"
RES = 384

# Eval palette (scripts/eval_sd3_paper_benchmark.py CITYSCAPES_COLORS),
# ordered by trainId 0..18.
PALETTE = np.array([
    (128, 64, 128),    # 0 road
    (244, 35, 232),    # 1 sidewalk
    (70, 70, 70),      # 2 building
    (102, 102, 156),   # 3 wall
    (190, 153, 153),   # 4 fence
    (153, 153, 153),   # 5 pole
    (250, 170, 30),    # 6 traffic_light
    (220, 220, 0),     # 7 traffic_sign
    (107, 142, 35),    # 8 vegetation
    (152, 251, 152),   # 9 terrain
    (70, 130, 180),    # 10 sky
    (220, 20, 60),     # 11 person
    (255, 0, 0),       # 12 rider
    (0, 0, 142),       # 13 car
    (0, 0, 70),        # 14 truck
    (0, 60, 100),      # 15 bus
    (0, 80, 100),      # 16 train
    (0, 0, 230),       # 17 motorcycle
    (119, 11, 32),     # 18 bicycle
], dtype=np.uint8)

# Official Cityscapes labelId -> trainId mapping; everything else -> ignore.
LABEL_ID_TO_TRAIN_ID = {
    7: 0,   # road
    8: 1,   # sidewalk
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    17: 5,  # pole
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}

# 256-entry labelId -> RGB lookup (ignore classes map to black).
COLOR_LUT = np.zeros((256, 3), dtype=np.uint8)
for lid, tid in LABEL_ID_TO_TRAIN_ID.items():
    COLOR_LUT[lid] = PALETTE[tid]


def colorize(seg: np.ndarray) -> np.ndarray:
    """Map a labelId array [H, W] to palette RGB [H, W, 3]."""
    if seg.ndim == 3:
        seg = seg[:, :, 0]
    return COLOR_LUT[seg]


def main() -> int:
    shards = sorted(PARQUET_DIR.glob("*.parquet"))
    if not shards:
        print(f"[error] no parquet shards under {PARQUET_DIR}")
        return 1

    # Backup existing class-ID split once
    if SEG_DIR.exists() and not BACKUP_DIR.exists():
        shutil.move(str(SEG_DIR), str(BACKUP_DIR))
        print(f"Backed up old class-ID split -> {BACKUP_DIR}")

    img_dir = SEG_DIR / "images"
    tgt_dir = SEG_DIR / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for shard in shards:
        df = pd.read_parquet(shard)
        for _, row in df.iterrows():
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            seg_img = Image.open(io.BytesIO(row["semantic_segmentation"]["bytes"]))
            seg = np.array(seg_img)

            rgb = Image.fromarray(colorize(seg), "RGB")
            img = img.resize((RES, RES), Image.LANCZOS)
            rgb = rgb.resize((RES, RES), Image.NEAREST)  # keep exact palette colors

            stem = f"cityscapes_cityscapes_{count:06d}"
            img.save(img_dir / f"{stem}.jpg", quality=95)
            rgb.save(tgt_dir / f"{stem}.png")
            count += 1
        print(f"  {shard.name}: cumulative {count}")

    print(f"\nRegenerated {count} segmentation pairs at {SEG_DIR}")

    # Verify: every target color must exist in the eval palette (plus black ignore)
    sample = np.array(Image.open(tgt_dir / "cityscapes_cityscapes_000000.png").convert("RGB"))
    uniq = np.unique(sample.reshape(-1, 3), axis=0)
    valid = set(map(tuple, PALETTE.tolist())) | {(0, 0, 0)}
    bad = [c for c in map(tuple, uniq.tolist()) if c not in valid]
    print(f"Verification: {len(uniq)} unique colors in sample, invalid={bad}")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
