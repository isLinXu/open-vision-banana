#!/usr/bin/env python3
"""Integrate vision-banana seg pairs into v5_full, stripping _rgb from target stems.

Source: llab-vision-banana-dataset semantic-segmentation/processed/semantic_all/train/
- input/  9729 .jpg images
- target/ 9729 _rgb.png seg masks

Naming: input "pixabay_123.jpg" <-> target "pixabay_123_rgb.png"
Strip "_rgb" from target name to get matching stem "pixabay_123".
"""
import os, sys, shutil

VB_SRC = "/Users/gatilin/.cache/huggingface/hub/datasets--lllab--lllab-vision-banana-dataset/snapshots/f074bc5ea17e167d2a91561121f58f5a40eca882/semantic-segmentation/processed/semantic_all/train"
DST_IMG = "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full/images/segmentation"
DST_TGT = "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full/targets/segmentation"

SRC_IMG = os.path.join(VB_SRC, "input")
SRC_TGT = os.path.join(VB_SRC, "target")

inputs = sorted(os.listdir(SRC_IMG))
targets = sorted(os.listdir(SRC_TGT))

print(f"Source inputs: {len(inputs)}, Source targets: {len(targets)}")

# Build target stem->filename map (strip _rgb)
tgt_map = {}
for t in targets:
    stem = t.rsplit(".", 1)[0]  # e.g. "pixabay_123_rgb"
    if stem.endswith("_rgb"):
        stem = stem[:-4]  # "pixabay_123"
    else:
        print(f"  WARNING: target {t} doesn't end with _rgb")
    tgt_map[stem] = t

# Pair and symlink
copied = 0
skipped = 0
for img_name in inputs:
    stem = img_name.rsplit(".", 1)[0]
    tgt_name = tgt_map.get(stem)
    if tgt_name is None:
        print(f"  SKIP: no target for {img_name}")
        skipped += 1
        continue

    src_img = os.path.join(SRC_IMG, img_name)
    src_tgt = os.path.join(SRC_TGT, tgt_name)

    # Destination names: keep original image name, rename target to stem.png
    dst_img = os.path.join(DST_IMG, img_name)
    dst_tgt_name = stem + ".png"  # stripped _rgb
    dst_tgt = os.path.join(DST_TGT, dst_tgt_name)

    if os.path.exists(dst_img):
        if not os.path.islink(dst_img):
            os.remove(dst_img)
        else:
            os.unlink(dst_img)
    if os.path.exists(dst_tgt):
        if not os.path.islink(dst_tgt):
            os.remove(dst_tgt)
        else:
            os.unlink(dst_tgt)

    os.symlink(src_img, dst_img)
    os.symlink(src_tgt, dst_tgt)
    copied += 1

    if copied % 2000 == 0:
        print(f"  ... {copied}/{len(inputs)}")

print(f"\nDone: {copied} pairs linked, {skipped} skipped")

# Verify
v5_imgs = set(os.listdir(DST_IMG))
v5_tgts = set(os.listdir(DST_TGT))
img_stems = {f.rsplit(".", 1)[0] for f in v5_imgs}
tgt_stems = {f.rsplit(".", 1)[0] for f in v5_tgts}
paired = img_stems & tgt_stems
print(f"\nPost-integration:")
print(f"  Total seg images: {len(v5_imgs)}")
print(f"  Total seg targets: {len(v5_tgts)}")
print(f"  Stem-paired: {len(paired)}")
print(f"  Unpaired images: {len(img_stems - tgt_stems)}")
print(f"  Unpaired targets: {len(tgt_stems - img_stems)}")
