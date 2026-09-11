#!/usr/bin/env python3
"""Convert segmentation class-ID index targets to unified RGB palette images.

Problem:
  - 1426 targets are L mode (uint8, ADE20K 0-150 class IDs)
  - 3984 targets are I;16 mode (uint16, lllab 3490-31730 class IDs)
  - The two class-ID spaces are incompatible; .convert("RGB") truncates
    uint16 to uint8, destroying lllab labels.

Fix:
  - L-mode targets → apply ADE20K 150-class color palette → RGB PNG
  - I;16 targets → dropped (incompatible class space; would teach wrong semantics)
  - Also filter train_val_split.json to only keep valid (L-mode) files

Usage:
  python scripts/regen_seg_rgb_palette.py --data-dir /path/to/vb_snvc_train_v2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ADE20K 150-class color palette (standard)
ADE20K_PALETTE = np.array([
    [0, 0, 0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
    [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70],
    [8, 255, 51], [255, 6, 82], [143, 255, 140], [204, 255, 4], [255, 51, 7],
    [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
    [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6], [10, 255, 71],
    [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6],
    [255, 194, 7], [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
    [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0],
    [153, 255, 0], [0, 0, 255], [255, 71, 0], [0, 235, 255], [0, 173, 255],
    [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 224, 0], [255, 0, 152], [255, 0, 5],
    [255, 245, 0], [255, 184, 0], [0, 255, 61], [0, 255, 133], [255, 0, 0],
    [255, 41, 0], [255, 163, 0], [255, 204, 0], [245, 0, 255], [255, 0, 112],
    [255, 51, 0], [0, 255, 163], [255, 235, 0], [92, 0, 255], [255, 0, 235],
    [255, 0, 92], [255, 0, 20], [255, 255, 0], [51, 0, 255], [0, 255, 214],
    [255, 102, 0], [255, 0, 184], [255, 214, 0], [255, 0, 71], [0, 255, 245],
    [255, 184, 0], [255, 0, 255], [255, 0, 163], [255, 224, 0], [255, 0, 102],
    [255, 245, 0], [255, 214, 0], [0, 255, 122], [255, 0, 214], [255, 0, 41],
    [0, 255, 184], [255, 0, 245], [255, 0, 51], [255, 82, 0], [255, 0, 31],
    [255, 163, 0], [255, 0, 133], [255, 235, 0], [255, 0, 122], [255, 0, 173],
    [255, 0, 10], [255, 0, 61], [0, 255, 10], [255, 0, 82], [0, 255, 102],
    [255, 0, 204], [255, 0, 224], [255, 0, 71], [255, 0, 92], [255, 0, 153],
    [255, 0, 20], [255, 0, 41], [255, 0, 163], [255, 0, 112], [255, 0, 235],
    [255, 0, 10], [255, 0, 31], [255, 0, 184], [255, 0, 51], [255, 0, 133],
    [255, 0, 71], [255, 0, 102], [255, 0, 153], [255, 0, 214], [255, 0, 245],
    [255, 0, 92], [255, 0, 20], [255, 0, 41], [255, 0, 61], [255, 0, 82],
    [255, 0, 102], [255, 0, 122], [255, 0, 143], [255, 0, 163], [255, 0, 184],
    [255, 0, 204], [255, 0, 224], [255, 0, 245],
], dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Dataset root (vb_snvc_train_v2)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, no changes")
    args = parser.parse_args()

    root = Path(args.data_dir)
    seg_dir = root / "segmentation"
    tgt_dir = seg_dir / "targets"
    img_dir = seg_dir / "images"

    if not tgt_dir.exists():
        print(f"ERROR: {tgt_dir} not found")
        return

    all_targets = sorted(os.listdir(tgt_dir))
    print(f"Total segmentation targets: {len(all_targets)}")

    # Classify by mode
    l_mode = []
    i16_mode = []
    for t in all_targets:
        img = Image.open(tgt_dir / t)
        if img.mode == "L":
            l_mode.append(t)
        elif img.mode in ("I;16", "I"):
            i16_mode.append(t)
        else:
            print(f"  Unknown mode: {t} ({img.mode})")

    print(f"  L mode (ADE20K, keep): {len(l_mode)}")
    print(f"  I;16 mode (lllab, drop): {len(i16_mode)}")

    if args.dry_run:
        print("\n[Dry run] No changes made.")
        return

    # Backup original targets
    backup_dir = seg_dir / "_targets_original"
    if not backup_dir.exists():
        backup_dir.mkdir()
        print(f"\nBacking up original targets to {backup_dir}")
        for t in all_targets:
            shutil.copy2(tgt_dir / t, backup_dir / t)

    # Move I;16 targets (and their images) to quarantine
    quarantine_dir = seg_dir / "_quarantine_i16"
    quarantine_dir.mkdir(exist_ok=True)
    quarantine_img_dir = quarantine_dir / "targets"
    quarantine_img_dir.mkdir(exist_ok=True)
    quarantine_src_img = quarantine_dir / "images"
    quarantine_src_img.mkdir(exist_ok=True)

    moved = 0
    for t in i16_mode:
        shutil.move(str(tgt_dir / t), str(quarantine_img_dir / t))
        # Also move corresponding image
        img_path = img_dir / t.replace(".png", ".jpg")
        if not img_path.exists():
            img_path = img_dir / t
        if img_path.exists():
            shutil.move(str(img_path), str(quarantine_src_img / img_path.name))
        moved += 1
    print(f"Quarantined {moved} I;16 targets + images")

    # Convert L-mode targets to RGB palette
    converted = 0
    for t in l_mode:
        img = Image.open(tgt_dir / t)
        arr = np.array(img)  # uint8 class IDs
        # Clip to palette range
        arr = np.clip(arr, 0, len(ADE20K_PALETTE) - 1)
        rgb = ADE20K_PALETTE[arr]  # [H, W, 3]
        rgb_img = Image.fromarray(rgb, mode="RGB")
        rgb_img.save(tgt_dir / t)  # Overwrite as RGB PNG
        converted += 1
    print(f"Converted {converted} L-mode targets to ADE20K RGB palette")

    # Update train_val_split.json
    split_file = root / "train_val_split.json"
    if split_file.exists():
        with open(split_file) as f:
            split = json.load(f)

        # Get remaining valid segmentation filenames (images, not targets)
        remaining_imgs = set()
        for t in l_mode:
            img_name = t.replace(".png", ".jpg")
            if (img_dir / img_name).exists():
                remaining_imgs.add(img_name)
            elif (img_dir / t).exists():
                remaining_imgs.add(t)

        for split_key in ["train", "val"]:
            old = split[split_key].get("segmentation", [])
            new = [f for f in old if f in remaining_imgs]
            removed = len(old) - len(new)
            split[split_key]["segmentation"] = new
            print(f"  split[{split_key}][segmentation]: {len(old)} → {len(new)} ({removed} removed)")

        with open(split_file, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Updated {split_file}")

    # Write palette info
    palette_file = seg_dir / "palette_info.json"
    with open(palette_file, "w") as f:
        json.dump({
            "palette": "ADE20K_150",
            "num_classes": len(ADE20K_PALETTE),
            "converted_from": "L_mode_class_id",
            "dropped": f"{len(i16_mode)} I;16 (lllab) targets — incompatible class space",
            "note": "Targets are now RGB PNG images using ADE20K 150-class color palette",
        }, f, indent=2)
    print(f"Wrote {palette_file}")

    print("\nDone! Segmentation targets are now unified RGB palette images.")


if __name__ == "__main__":
    main()
