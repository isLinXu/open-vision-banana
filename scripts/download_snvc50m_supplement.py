#!/usr/bin/env python3
"""
SN-VC-50M 数据集补充下载脚本

在已有 26 个 JSONL 基础上，补充下载缺失的小型 JSONL 标注文件（<100MB），
解压已有数据集（FSC147），并尝试下载 TAS500。

策略：按任务族抽样下载，不全量下载。
"""

import json
import os
import sys
import random
import subprocess
import time
from pathlib import Path
from collections import Counter

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
SNVC_ROOT = DATASETS_ROOT / "snvc50m"
HF_BASE = "https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M/resolve/main"

# ============================================================
# Missing JSONL files to download (only < 100MB for efficiency)
# ============================================================

MISSING_JSONL = {
    "dense_geometric_prediction": [
        # scannetpp_normal is moderate but adds normal prediction diversity
        ("scannetpp_normal", "dense_geometric_prediction/scannetpp_total_unified_normal_edit_regenerated.jsonl", 338.5, 500),
    ],
    "segmentation": [
        ("51world_gcg", "segmentation/51World_total_unified_gcg_edit_regenerated.jsonl", 6.7, 500),
        ("51world_genseg", "segmentation/51world_total_unified_genseg_edit_regenerated.jsonl", 17.1, 1000),
        ("hypersim_gcg", "segmentation/Hypersim_total_unified_gcg_edit_regenerated.jsonl", 60.5, 1000),
        ("idd2_genseg", "segmentation/IDD-2_total_unified_genseg_edit_regenerated.jsonl", 23.7, 1000),
        ("iddav3_genseg", "segmentation/IDDAV3_total_unified_genseg_edit_regenerated.jsonl", 8.8, 500),
        ("iddav3_gcg", "segmentation/IDDAv3_total_unified_gcg_edit_regenerated.jsonl", 1.9, 500),
        ("kitti_gcg", "segmentation/KITTI_total_unified_gcg_edit_regenerated.jsonl", 6.9, 500),
        ("streethazards_gcg", "segmentation/StreetHazards_total_unified_gcg_edit_regenerated.jsonl", 1.5, 500),
        ("streethazards_genseg", "segmentation/StreetHazards_total_unified_genseg_edit_regenerated.jsonl", 6.7, 500),
        ("ttpla_genseg", "segmentation/TTPLA_total_unified_genseg_edit_regenerated.jsonl", 1.2, 500),
        ("trashcan_genseg", "segmentation/Trashcan_total_unified_genseg_edit_regenerated.jsonl", 5.1, 500),
        ("udd5_gcg", "segmentation/UDD5_total_unified_gcg_edit_regenerated.jsonl", 0.1, 100),
        ("udd5_genseg", "segmentation/UDD5_total_unified_genseg_edit_regenerated.jsonl", 0.1, 100),
        ("udd6_gcg", "segmentation/UDD6_total_unified_gcg_edit_regenerated.jsonl", 0.1, 100),
        ("udd6_genseg", "segmentation/UDD6_total_unified_genseg_edit_regenerated.jsonl", 0.1, 100),
        ("zerowaste_genseg", "segmentation/ZeroWaste_total_unified_genseg_edit_regenerated.jsonl", 2.5, 500),
        ("entityv2_gcg", "segmentation/entityv2_total_unified_gcg_edit_regenerated.jsonl", 7.9, 500),
        ("grandf_gcg", "segmentation/grandf_total_unified_gcg_edit_regenerated.jsonl", 0.9, 200),
        ("pidray_genseg", "segmentation/pidray_total_unified_genseg_edit_regenerated.jsonl", 19.1, 500),
        ("flickr_gcg", "segmentation/flickr_total_unified_gcg_edit_regenerated.jsonl", 84.6, 1000),
        ("vipseg_gcg", "segmentation/VIPSeg_total_unified_gcg_edit_regenerated.jsonl", 85.9, 1000),
    ],
    "structure_view_understanding": [
        ("aptv2_point", "structure_view_understanding/APTv2_total_unified_point_und.jsonl", 6.1, 500),
        ("egoobjects_point", "structure_view_understanding/EgoObjects_total_unified_point_und.jsonl", 29.6, 1000),
        ("fair1m_point", "structure_view_understanding/FAIR1M_total_unified_point_und.jsonl", 16.1, 500),
        ("nuimages_point", "structure_view_understanding/NuImages_total_unified_point_und.jsonl", 35.7, 1000),
        ("paco_point", "structure_view_understanding/PACO_total_unified_point_und.jsonl", 20.2, 500),
        ("v3detovd_point", "structure_view_understanding/V3Detovd_total_unified_point_und.jsonl", 32.3, 1000),
        ("pixmo_bbox", "structure_view_understanding/pixmo_total_unified_bbox_und_regenerated.jsonl", 0.3, 100),
    ],
    "multiview_visual_geometry": [
        ("scannetv2_recon", "multiview_visual_geometry/scannetv2_total_unified_recon_edit_regenerated.jsonl", 30.3, 500),
    ],
}


def download_jsonl(name: str, hf_path: str, size_mb: float, sample_size: int):
    """Download a JSONL file from HuggingFace and sample it."""
    out_path = SNVC_ROOT / f"{name}.jsonl"
    sampled_path = SNVC_ROOT / f"{name}_sampled.jsonl"

    if sampled_path.exists():
        count = sum(1 for _ in open(sampled_path))
        print(f"  [skip] {name}: already sampled ({count} lines)")
        return True

    url = f"{HF_BASE}/{hf_path}"
    print(f"  [download] {name} ({size_mb:.1f} MB) ...")

    result = subprocess.run(
        ["curl", "-L", "-s", "-o", str(out_path), url],
        capture_output=True, text=True, timeout=max(300, int(size_mb * 3))
    )

    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 100:
        print(f"  [error] {name}: download failed")
        out_path.unlink(missing_ok=True)
        return False

    actual_size = out_path.stat().st_size / 1024 / 1024
    print(f"  [ok] {name}: {actual_size:.1f} MB")

    # Sample
    with open(out_path) as f:
        lines = f.readlines()
    total = len(lines)

    if total <= sample_size:
        sampled = lines
        print(f"  [sample] {name}: keeping all {total} lines")
    else:
        random.seed(42)
        sampled = random.sample(lines, sample_size)
        print(f"  [sample] {name}: {total} -> {sample_size}")

    with open(sampled_path, "w") as f:
        f.writelines(sampled)

    # Remove full file to save space (keep only sampled)
    out_path.unlink(missing_ok=True)

    return True


def extract_fsc147():
    """Extract FSC147 images."""
    print("\n[Extract] FSC147...")
    zip_path = DATASETS_ROOT / "FSC147" / "images_384_VarV2.zip"
    img_dir = DATASETS_ROOT / "FSC147" / "images_384_VarV2"

    if img_dir.exists() and any(img_dir.iterdir()):
        count = len(list(img_dir.glob("*.jpg")))
        print(f"  [skip] FSC147 already extracted ({count} images)")
        return

    if not zip_path.exists():
        print("  [error] FSC147 zip not found")
        return

    print(f"  Extracting {zip_path.name}...")
    result = subprocess.run(
        ["unzip", "-q", "-o", str(zip_path), "-d", str(DATASETS_ROOT / "FSC147")],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode == 0 and img_dir.exists():
        count = len(list(img_dir.glob("*.jpg")))
        print(f"  [ok] FSC147: {count} images extracted")
    else:
        print(f"  [error] Extraction failed: {result.stderr[:200]}")


def try_tas500():
    """Try to download TAS500 from alternative sources."""
    print("\n[Download] TAS500...")
    tas_root = DATASETS_ROOT / "TAS500"
    tas_root.mkdir(parents=True, exist_ok=True)

    # Check if already has content
    if any(tas_root.iterdir()):
        print("  [skip] TAS500 has content")
        return

    # Try dataset-ninja
    urls = [
        ("https://huggingface.co/datasets/dataset-ninja/tas500/resolve/main/tas500.zip", "tas500.zip"),
        ("https://huggingface.co/datasets/dataset-ninja/tas500/resolve/main/train.zip", "train.zip"),
    ]

    for url, fname in urls:
        out_path = tas_root / fname
        print(f"  Trying {url}...")
        result = subprocess.run(
            ["curl", "-L", "-s", "-o", str(out_path), url],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"  [ok] Downloaded {fname} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
            subprocess.run(["unzip", "-q", "-o", str(out_path), "-d", str(tas_root)],
                         capture_output=True)
            out_path.unlink(missing_ok=True)
            return
        out_path.unlink(missing_ok=True)

    print("  [warn] TAS500 download failed (all sources). Will skip in training.")


def download_loveda_full():
    """Download LoveDA Train + Val if not present."""
    print("\n[Download] LoveDA (full)...")
    loveda_root = DATASETS_ROOT / "LoveDA"

    # Check current state
    current_count = len(list(loveda_root.rglob("*.png"))) if loveda_root.exists() else 0
    if current_count > 5000:
        print(f"  [skip] LoveDA already has {current_count} images")
        return

    # Download from HF mirror
    urls = [
        ("Train.zip", "https://huggingface.co/datasets/yangwang825/LoveDA/resolve/main/Train.zip"),
        ("Val.zip", "https://huggingface.co/datasets/yangwang825/LoveDA/resolve/main/Val.zip"),
    ]

    for fname, url in urls:
        out_path = loveda_root / fname
        if out_path.exists():
            print(f"  [skip] {fname} exists")
        else:
            print(f"  Downloading {fname}...")
            result = subprocess.run(
                ["curl", "-L", "-s", "-o", str(out_path), url],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
                print(f"  [warn] {fname} download failed")
                out_path.unlink(missing_ok=True)
                continue

        print(f"  Extracting {fname}...")
        subprocess.run(["unzip", "-q", "-o", str(out_path), "-d", str(loveda_root)],
                     capture_output=True)

    # Check result
    img_count = len(list(loveda_root.rglob("*.png")))
    mask_count = len(list(loveda_root.rglob("*.png")))  # masks are also png
    print(f"  LoveDA total PNG files: {img_count}")


def print_summary():
    """Print final summary of all JSONL files."""
    print("\n" + "=" * 60)
    print("SN-VC-50M Supplement Summary")
    print("=" * 60)

    all_sampled = sorted(SNVC_ROOT.glob("*_sampled.jsonl"))
    print(f"\nTotal sampled JSONL files: {len(all_sampled)}")

    total_samples = 0
    by_family = {"dense_geometric_prediction": 0, "segmentation": 0,
                 "structure_view_understanding": 0, "multiview_visual_geometry": 0}

    for f in all_sampled:
        count = sum(1 for _ in open(f))
        total_samples += count
        name = f.stem.replace("_sampled", "")

        # Classify
        if any(k in name for k in ["depth", "normal"]):
            by_family["dense_geometric_prediction"] += count
        elif any(k in name for k in ["recon", "wildrgbd"]):
            by_family["multiview_visual_geometry"] += count
        elif any(k in name for k in ["point", "bbox", "visual_und", "ocr"]):
            by_family["structure_view_understanding"] += count
        else:
            by_family["segmentation"] += count

        print(f"  {name}: {count}")

    print(f"\nTotal samples: {total_samples}")
    print(f"\nBy task family:")
    for family, count in by_family.items():
        print(f"  {family}: {count}")

    # Source dataset status
    print(f"\nSource dataset status:")
    checks = [
        ("COCO2017 train", DATASETS_ROOT / "coco2017" / "train2017"),
        ("COCO2014 train", DATASETS_ROOT / "coco2014" / "train2014"),
        ("Cityscapes images", DATASETS_ROOT / "cityscapes" / "leftImg8bit"),
        ("GroceryStore", DATASETS_ROOT / "GroceryStore" / "train"),
        ("FSC147 images", DATASETS_ROOT / "FSC147" / "images_384_VarV2"),
        ("LoveDA", DATASETS_ROOT / "LoveDA"),
        ("Hypersim scenes", DATASETS_ROOT / "hypersim" / "scenes"),
        ("TAS500", DATASETS_ROOT / "TAS500"),
    ]
    for name, path in checks:
        if path.exists():
            if path.is_dir():
                count = sum(1 for _ in path.rglob("*") if _.is_file())
                print(f"  [OK]   {name}: {count} files")
            else:
                print(f"  [OK]   {name}")
        else:
            print(f"  [MISS] {name}")


def main():
    print("=" * 60)
    print("SN-VC-50M Supplement Download")
    print("=" * 60)

    SNVC_ROOT.mkdir(parents=True, exist_ok=True)

    # Step 1: Download missing JSONL files
    print("\n[Step 1] Downloading missing JSONL annotations...")
    success = 0
    failed = 0
    for family, files in MISSING_JSONL.items():
        print(f"\n  Family: {family}")
        for name, path, size_mb, sample_size in files:
            ok = download_jsonl(name, path, size_mb, sample_size)
            if ok:
                success += 1
            else:
                failed += 1

    print(f"\n  JSONL download: {success} ok, {failed} failed")

    # Step 2: Extract FSC147
    print("\n[Step 2] Extracting datasets...")
    extract_fsc147()

    # Step 3: Download LoveDA full
    download_loveda_full()

    # Step 4: Try TAS500
    try_tas500()

    # Step 5: Summary
    print_summary()


if __name__ == "__main__":
    main()
