#!/usr/bin/env python3
"""
SN-VC-50M source dataset download script

Downloads the source image datasets referenced by the SN-VC-50M JSONL annotations.
Only the required subsets are downloaded, not everything.

Priority:
1. Cityscapes (leftImg8bit + gtFine): segmentation tasks
2. FSC147: counting/detection tasks
3. LOVEDA: remote-sensing segmentation
4. Path fixes: RefCOCOg (COCO 2014 already present, needs a symlink)
5. Hypersim subset (8 scenes present, download the missing scenes)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
SNVC_ROOT = DATASETS_ROOT / "snvc50m"


def fix_refcocog_paths():
    """Fix RefCOCOg image paths: JSONL uses 'coco2014/train2014/' but images are at 'coco2014/images/train2014/'"""
    print("\n[1] Fixing RefCOCOg image paths...")
    
    # Create symlink: datasets_root/coco2014/train2014 -> images/train2014
    target = DATASETS_ROOT / "coco2014" / "train2014"
    source = DATASETS_ROOT / "coco2014" / "images" / "train2014"
    
    if source.exists() and not target.exists():
        target.symlink_to(source)
        print(f"  Symlink: {target} -> {source}")
    elif target.exists():
        print(f"  Already exists: {target}")
    else:
        print(f"  [warn] Source not found: {source}")
    
    # Also fix val2014
    target_val = DATASETS_ROOT / "coco2014" / "val2014"
    source_val = DATASETS_ROOT / "coco2014" / "images" / "val2014"
    if source_val.exists() and not target_val.exists():
        target_val.symlink_to(source_val)
        print(f"  Symlink: {target_val} -> {source_val}")


def download_cityscapes():
    """Download Cityscapes leftImg8bit and gtFine from cityscapes.net."""
    print("\n[2] Downloading Cityscapes...")
    
    cs_root = DATASETS_ROOT / "cityscapes"
    cs_root.mkdir(parents=True, exist_ok=True)
    
    # Cityscapes requires login, try HF mirror
    # Check HF for cityscapes mirrors
    urls = [
        # leftImg8bit train
        ("leftImg8bit_trainvaltest.zip", 
         "https://huggingface.co/datasets/Antreas/Cityscapes/resolve/main/leftImg8bit_trainvaltest.zip"),
        # gtFine train
        ("gtFine_trainvaltest.zip",
         "https://huggingface.co/datasets/Antreas/Cityscapes/resolve/main/gtFine_trainvaltest.zip"),
    ]
    
    for fname, url in urls:
        out_path = cs_root / fname
        if out_path.exists():
            print(f"  [skip] {fname} already exists")
            continue
        print(f"  [download] {fname}...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(out_path), url],
            capture_output=True, text=True
        )
        if result.returncode != 0 or out_path.stat().st_size < 1000:
            print(f"  [warn] Download failed for {fname}, trying alternative...")
            out_path.unlink(missing_ok=True)
        else:
            print(f"  [ok] {fname} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Check if we got the files
    zip_files = list(cs_root.glob("*.zip"))
    if not zip_files:
        print("  [fallback] Cityscapes not available from HF, trying direct URLs...")
        # Try cityscapes.net direct (requires login, likely fails)
        print("  [note] Cityscapes requires registration. Skipping - will use JSONL annotations only.")
        return False
    return True


def download_fsc147():
    """Download FSC147 dataset from GitHub."""
    print("\n[3] Downloading FSC147...")
    
    fsc_root = DATASETS_ROOT / "FSC147"
    fsc_root.mkdir(parents=True, exist_ok=True)
    
    urls = [
        ("images_384_VarV2.zip",
         "https://github.com/cvlab-stonybrook/LearningToCountEverything/releases/download/v1.0/images_384_VarV2.zip"),
    ]
    
    for fname, url in urls:
        out_path = fsc_root / fname
        if out_path.exists():
            print(f"  [skip] {fname} already exists")
            continue
        print(f"  [download] {fname}...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(out_path), url],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
            print(f"  [warn] Download failed for {fname}")
            out_path.unlink(missing_ok=True)
        else:
            print(f"  [ok] {fname} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
            # Extract
            print(f"  [extract] {fname}...")
            subprocess.run(["unzip", "-q", "-o", str(out_path), "-d", str(fsc_root)], 
                         capture_output=True)
    
    # Check
    img_dir = fsc_root / "images_384_VarV2"
    if img_dir.exists():
        count = len(list(img_dir.glob("*.jpg")))
        print(f"  FSC147 images: {count}")
        return True
    print("  [warn] FSC147 images not found after download")
    return False


def download_loveda():
    """Download LoveDA dataset."""
    print("\n[4] Downloading LoveDA...")
    
    loveda_root = DATASETS_ROOT / "LoveDA"
    loveda_root.mkdir(parents=True, exist_ok=True)
    
    # LoveDA from Google Drive or HF
    urls = [
        ("Train.zip", "https://huggingface.co/datasets/yangwang825/LoveDA/resolve/main/Train.zip"),
        ("Val.zip", "https://huggingface.co/datasets/yangwang825/LoveDA/resolve/main/Val.zip"),
    ]
    
    for fname, url in urls:
        out_path = loveda_root / fname
        if out_path.exists():
            print(f"  [skip] {fname} already exists")
            continue
        print(f"  [download] {fname}...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(out_path), url],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
            print(f"  [warn] Download failed for {fname}")
            out_path.unlink(missing_ok=True)
        else:
            print(f"  [ok] {fname} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"  [extract] {fname}...")
            subprocess.run(["unzip", "-q", "-o", str(out_path), "-d", str(loveda_root)],
                         capture_output=True)
    
    # Check structure: LoveDA/Train/Rural/images_png/
    train_rural = loveda_root / "Train" / "Rural" / "images_png"
    if train_rural.exists():
        count = len(list(train_rural.glob("*.png")))
        print(f"  LoveDA Train/Rural images: {count}")
        return True
    print("  [warn] LoveDA not found after download")
    return False


def download_tas500():
    """Download TAS500 dataset."""
    print("\n[5] Downloading TAS500...")
    
    tas_root = DATASETS_ROOT / "TAS500"
    tas_root.mkdir(parents=True, exist_ok=True)
    
    url = "https://huggingface.co/datasets/dataset-ninja/tas500/resolve/main/tas500.zip"
    out_path = tas_root / "tas500.zip"
    
    if not out_path.exists():
        print(f"  [download] TAS500...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(out_path), url],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"  [ok] ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"  [extract]...")
            subprocess.run(["unzip", "-q", "-o", str(out_path), "-d", str(tas_root)],
                         capture_output=True)
        else:
            print(f"  [warn] Download failed")
            out_path.unlink(missing_ok=True)
    
    train_dir = tas_root / "train"
    if train_dir.exists():
        count = len(list(train_dir.glob("*.png")))
        print(f"  TAS500 train images: {count}")
        return True
    print("  [warn] TAS500 not found")
    return False


def download_hypersim_missing():
    """Download missing Hypersim scenes referenced in JSONL."""
    print("\n[6] Checking Hypersim scenes...")
    
    # Read sampled JSONL to find which scenes we need
    jsonl_path = SNVC_ROOT / "hypersim_genseg_sampled.jsonl"
    if not jsonl_path.exists():
        jsonl_path = SNVC_ROOT / "hypersim_genseg.jsonl"
    
    needed_scenes = set()
    existing_scenes = set()
    
    # Check existing scenes
    scenes_dir = DATASETS_ROOT / "hypersim" / "scenes"
    if scenes_dir.exists():
        existing_scenes = {d.name for d in scenes_dir.iterdir() if d.is_dir()}
    
    # Find needed scenes from JSONL
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                img = d.get("image", "")
                if img.startswith("ai_"):
                    scene = img.split("/")[0]
                    needed_scenes.add(scene)
    
    missing = needed_scenes - existing_scenes
    print(f"  Needed scenes: {len(needed_scenes)}")
    print(f"  Existing scenes: {len(existing_scenes)} ({sorted(existing_scenes)[:5]}...)")
    print(f"  Missing scenes: {len(missing)}")
    
    if missing:
        print(f"  Missing scene list: {sorted(missing)[:20]}...")
        # Hypersim scenes are large (several GB each), only download a few
        to_download = sorted(missing)[:5]  # limit to 5
        print(f"  Will try to download: {to_download}")
        
        # Hypersim download from HF: apple/ml-hypersim
        # Each scene has images at scenes/{scene}/images/scene_cam_XX_final_preview/
        for scene in to_download:
            print(f"  [note] Scene {scene} requires Hypersim registration. Skipping full download.")
            # Mark as unavailable - will filter these from training data
        return False
    return True


def check_grocery_store():
    """Download GroceryStore dataset."""
    print("\n[7] Downloading GroceryStore...")
    
    gs_root = DATASETS_ROOT / "GroceryStore"
    gs_root.mkdir(parents=True, exist_ok=True)
    
    url = "https://github.com/marcusklasson/GroceryStoreDataset/raw/master/dataset.zip"
    out_path = gs_root / "dataset.zip"
    
    if not out_path.exists():
        print(f"  [download] GroceryStore...")
        result = subprocess.run(
            ["curl", "-L", "-o", str(out_path), url],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
            print(f"  [ok] ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
            print(f"  [extract]...")
            subprocess.run(["unzip", "-q", "-o", str(out_path), "-d", str(gs_root)],
                         capture_output=True)
        else:
            print(f"  [warn] Download failed")
            out_path.unlink(missing_ok=True)
    
    train_dir = gs_root / "train"
    if train_dir.exists():
        count = sum(1 for _ in train_dir.rglob("*.jpg"))
        print(f"  GroceryStore train images: {count}")
        return True
    print("  [warn] GroceryStore not found")
    return False


def generate_coco_depth_normal_targets():
    """Generate depth/normal target images for COCO using the JSONL references.
    
    The SN-VC-50M JSONL references target images at coco2017/depth/ and coco2017/normal/.
    These are pre-computed targets inside the 8TB tar.gz that we can't download.
    
    Instead, we'll:
    1. Create empty placeholder directories
    2. Note that we need to generate targets with a depth estimation model
    """
    print("\n[8] Setting up COCO depth/normal target directories...")
    
    depth_dir = DATASETS_ROOT / "coco2017" / "depth"
    normal_dir = DATASETS_ROOT / "coco2017" / "normal"
    
    depth_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if any target images already exist
    depth_count = len(list(depth_dir.glob("*.png")))
    normal_count = len(list(normal_dir.glob("*.png")))
    
    print(f"  COCO depth targets: {depth_count}")
    print(f"  COCO normal targets: {normal_count}")
    
    if depth_count == 0:
        print("  [note] Depth targets need to be generated (use Depth Anything V2 or Marigold)")
    if normal_count == 0:
        print("  [note] Normal targets need to be generated (use Marigold or DSINE)")
    
    return depth_count, normal_count


def final_summary():
    """Print final availability summary."""
    print("\n" + "=" * 60)
    print("Data Availability Summary")
    print("=" * 60)
    
    checks = [
        ("COCO 2017 train images", DATASETS_ROOT / "coco2017" / "train2017"),
        ("COCO 2017 depth targets", DATASETS_ROOT / "coco2017" / "depth"),
        ("COCO 2017 normal targets", DATASETS_ROOT / "coco2017" / "normal"),
        ("COCO 2014 train images", DATASETS_ROOT / "coco2014" / "train2014"),
        ("Cityscapes", DATASETS_ROOT / "cityscapes"),
        ("FSC147", DATASETS_ROOT / "FSC147" / "images_384_VarV2"),
        ("LoveDA", DATASETS_ROOT / "LoveDA" / "Train"),
        ("TAS500", DATASETS_ROOT / "TAS500" / "train"),
        ("GroceryStore", DATASETS_ROOT / "GroceryStore" / "train"),
        ("Hypersim scenes", DATASETS_ROOT / "hypersim" / "scenes"),
    ]
    
    for name, path in checks:
        if path.exists():
            if path.is_dir():
                count = sum(1 for _ in path.rglob("*") if _.is_file())
                print(f"  [OK]   {name}: {count} files at {path}")
            else:
                print(f"  [OK]   {name}: {path}")
        else:
            print(f"  [MISS] {name}: {path}")
    
    # JSONL summary
    print(f"\n  JSONL files at: {SNVC_ROOT}")
    jsonl_count = len(list(SNVC_ROOT.glob("*.jsonl")))
    sampled_count = len(list(SNVC_ROOT.glob("*_sampled.jsonl")))
    print(f"  Total JSONL: {jsonl_count} (sampled: {sampled_count})")


def main():
    print("=" * 60)
    print("SN-VC-50M Source Dataset Downloader")
    print("=" * 60)
    
    # 1. Fix RefCOCOg paths (symlink)
    fix_refcocog_paths()
    
    # 2. Download Cityscapes
    download_cityscapes()
    
    # 3. Download FSC147
    download_fsc147()
    
    # 4. Download LoveDA
    download_loveda()
    
    # 5. Download TAS500
    download_tas500()
    
    # 6. Check Hypersim
    download_hypersim_missing()
    
    # 7. Download GroceryStore
    check_grocery_store()
    
    # 8. Setup COCO target dirs
    generate_coco_depth_normal_targets()
    
    # Summary
    final_summary()


if __name__ == "__main__":
    main()
