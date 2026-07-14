#!/usr/bin/env python3
"""
SN-VC-50M 数据集抽样下载与准备脚本

从 SenseNova-Vision-Corpus-50M 下载 JSONL 标注文件，
并准备对应的原始图片和目标图（深度/法线/分割等）。

策略：
- JSONL 标注：直接从 HF 下载（小文件）
- 原始图片：从各原始数据集官方源下载
- 目标图（depth/normal/seg masks）：8TB tar.gz 无法全量下载，
  改为从原始数据集标注生成或用预训练模型推理

任务覆盖：
1. Dense Geometric Prediction (depth/normal) — COCO 2017
2. Segmentation — Cityscapes, RefCOCOg, Hypersim, LOVEDA
3. Structure View Understanding — FSC147, DOTA, GroceryStore, HumanParts
"""

import json
import os
import sys
import random
import subprocess
import urllib.request
from pathlib import Path
from collections import defaultdict, Counter

# ============================================================
# Configuration
# ============================================================

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
SNVC_ROOT = DATASETS_ROOT / "snvc50m"
HF_BASE = "https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M/resolve/main"

# JSONL files to download, grouped by task family
JSONL_FILES = {
    "dense_geometric_prediction": {
        "coco_depth": "dense_geometric_prediction/coco_total_unified_depth_edit_regenerated.jsonl",
        "coco_normal": "dense_geometric_prediction/coco_total_unified_normal_edit_regenerated.jsonl",
    },
    "segmentation": {
        "cityscapes_genseg": "segmentation/cityscapes_total_unified_genseg_edit_regenerated.jsonl",
        "cityscapes_gcg": "segmentation/cityscapes_total_unified_gcg_edit_regenerated.jsonl",
        "refcocog_gcg": "segmentation/refcocog_total_unified_gcg_edit_regenerated.jsonl",
        "hypersim_genseg": "segmentation/hypersim_total_unified_genseg_edit_regenerated.jsonl",
        "loveda_genseg": "segmentation/LOVEDA_total_unified_genseg_edit_regenerated.jsonl",
        "loveda_gcg": "segmentation/LOVEDA_total_unified_gcg_edit_regenerated.jsonl",
        "tas500_genseg": "segmentation/TAS500_total_unified_genseg_edit_regenerated.jsonl",
        "tas500_gcg": "segmentation/TAS500_total_unified_gcg_edit_regenerated.jsonl",
    },
    "structure_view_understanding": {
        "fsc147_bbox": "structure_view_understanding/FSC147_total_unified_bbox_und.jsonl",
        "fsc147_visual": "structure_view_understanding/FSC147_total_unified_visual_und.jsonl",
        "dota_point": "structure_view_understanding/DOTAv2_total_unified_point_und.jsonl",
        "grocery_bbox": "structure_view_understanding/GroceryStore_total_unified_bbox_und.jsonl",
        "grocery_visual": "structure_view_understanding/GroceryStore_total_unified_visual_und.jsonl",
        "humanparts_point": "structure_view_understanding/HumanParts_total_unified_point_und.jsonl",
        "imagenetpart_point": "structure_view_understanding/ImageNetPart_total_unified_point_und.jsonl",
    },
}

# Sampling: how many samples to keep per JSONL
SAMPLE_SIZES = {
    "coco_depth": 5000,
    "coco_normal": 5000,
    "cityscapes_genseg": 2000,  # all (2975)
    "cityscapes_gcg": 2000,
    "refcocog_gcg": 5000,
    "hypersim_genseg": 3000,
    "loveda_genseg": 1500,
    "loveda_gcg": 1500,
    "tas500_genseg": 500,  # all
    "tas500_gcg": 500,
    "fsc147_bbox": 1000,
    "fsc147_visual": 1000,
    "dota_point": 1000,
    "grocery_bbox": 500,
    "grocery_visual": 500,
    "humanparts_point": 1000,
    "imagenetpart_point": 1000,
}


def download_jsonl(name: str, path: str, force: bool = False):
    """Download a JSONL file from HuggingFace."""
    out_path = SNVC_ROOT / f"{name}.jsonl"
    if out_path.exists() and not force:
        print(f"  [skip] {name}.jsonl already exists ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return out_path

    url = f"{HF_BASE}/{path}"
    print(f"  [download] {name}.jsonl from {url}")
    subprocess.run(["curl", "-L", "-o", str(out_path), url], check=True)
    return out_path


def sample_jsonl(name: str, sample_size: int):
    """Sample N lines from a JSONL file."""
    in_path = SNVC_ROOT / f"{name}.jsonl"
    out_path = SNVC_ROOT / f"{name}_sampled.jsonl"

    with open(in_path) as f:
        lines = f.readlines()

    total = len(lines)
    if total <= sample_size:
        print(f"  [sample] {name}: keeping all {total} lines")
        sampled = lines
    else:
        random.seed(42)
        sampled = random.sample(lines, sample_size)
        print(f"  [sample] {name}: {total} -> {sample_size}")

    with open(out_path, "w") as f:
        f.writelines(sampled)

    return out_path


def analyze_jsonl(name: str, max_lines: int = 100):
    """Analyze a JSONL file: show format, image paths, target paths."""
    in_path = SNVC_ROOT / f"{name}.jsonl"
    if not in_path.exists():
        return

    print(f"\n  --- {name} analysis ---")
    with open(in_path) as f:
        first = json.loads(f.readline())

    # Show structure
    for key in ["image", "seg", "conversations", "caption"]:
        if key in first:
            val = first[key]
            if isinstance(val, list):
                print(f"  {key}: list[{len(val)}] -> {val[:2]}")
            elif isinstance(val, str) and len(val) > 100:
                print(f"  {key}: {val[:100]}...")
            else:
                print(f"  {key}: {val}")

    # Count unique source datasets
    sources = Counter()
    with open(in_path) as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            d = json.loads(line)
            img = d.get("image", "")
            if isinstance(img, list):
                prefix = img[0].split("/")[0]
            else:
                prefix = img.split("/")[0] if img else "unknown"
            sources[prefix] += 1

    print(f"  Source datasets (first {max_lines}): {dict(sources)}")


def check_image_exists(jsonl_name: str, datasets_root: Path):
    """Check which images referenced in the JSONL already exist locally."""
    in_path = SNVC_ROOT / f"{jsonl_name}_sampled.jsonl"
    if not in_path.exists():
        in_path = SNVC_ROOT / f"{jsonl_name}.jsonl"

    if not in_path.exists():
        return

    found = 0
    missing = 0
    missing_paths = []

    with open(in_path) as f:
        for line in f:
            d = json.loads(line)
            imgs = d.get("image", "")
            if isinstance(imgs, list):
                img_path = imgs[0]
            else:
                img_path = imgs

            # Try to find the image relative to datasets root
            full_path = datasets_root / img_path
            if full_path.exists():
                found += 1
            else:
                missing += 1
                if len(missing_paths) < 3:
                    missing_paths.append(img_path)

    total = found + missing
    print(f"  [check] {jsonl_name}: {found}/{total} images found")
    if missing > 0:
        print(f"    Missing examples: {missing_paths}")


def main():
    print("=" * 60)
    print("SN-VC-50M Dataset Sampling & Preparation")
    print("=" * 60)

    SNVC_ROOT.mkdir(parents=True, exist_ok=True)

    # Step 1: Download JSONL files
    print("\n[Step 1] Downloading JSONL annotation files...")
    for family, files in JSONL_FILES.items():
        print(f"\n  Family: {family}")
        for name, path in files.items():
            try:
                download_jsonl(name, path)
            except Exception as e:
                print(f"  [error] {name}: {e}")

    # Step 2: Analyze formats
    print("\n[Step 2] Analyzing JSONL formats...")
    for family, files in JSONL_FILES.items():
        for name in files:
            analyze_jsonl(name)

    # Step 3: Sample
    print("\n[Step 3] Sampling JSONL files...")
    for name, size in SAMPLE_SIZES.items():
        jsonl_path = SNVC_ROOT / f"{name}.jsonl"
        if jsonl_path.exists():
            sample_jsonl(name, size)

    # Step 4: Check image availability
    print("\n[Step 4] Checking image availability...")
    for name in SAMPLE_SIZES:
        check_image_exists(name, DATASETS_ROOT)

    # Step 5: Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_samples = 0
    for name, size in SAMPLE_SIZES.items():
        sampled_path = SNVC_ROOT / f"{name}_sampled.jsonl"
        if sampled_path.exists():
            with open(sampled_path) as f:
                count = sum(1 for _ in f)
            total_samples += count
            print(f"  {name}: {count} samples")
    print(f"\nTotal sampled: {total_samples}")
    print(f"\nOutput: {SNVC_ROOT}")


if __name__ == "__main__":
    main()
