#!/usr/bin/env python3
"""
Download WildRGB-D tar files from yslan/3dscene_wildrgbd.
Only download the 15 categories referenced in JSONL.
Each category is a single .tar file containing all scenes.
"""
import json
import subprocess
import time
from pathlib import Path
from collections import defaultdict

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
WILDRGBD_DIR = DATASETS_ROOT / "WildRGB-D"
HF_BASE = "https://hf-mirror.com/datasets"
REPO_ID = "yslan/3dscene_wildrgbd"
JSONL_PATH = DATASETS_ROOT / "snvc50m" / "wildrgbd_total_unified_recon_edit_regenerated_sampled.jsonl"

# Categories referenced in JSONL
JSONL_CATEGORIES = [
    "TV", "apple", "backpack", "ball", "banana", "boat", "book",
    "bottle", "bowl", "box", "bucket", "bus", "cake", "car", "carrot"
]


def main():
    WILDRGBD_DIR.mkdir(parents=True, exist_ok=True)
    tar_dir = WILDRGBD_DIR / "tars"
    tar_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0
    skipped = 0

    for i, cat in enumerate(JSONL_CATEGORIES):
        tar_path = tar_dir / f"{cat}.tar"
        
        if tar_path.exists() and tar_path.stat().st_size > 1000:
            print(f"[{i+1}/{len(JSONL_CATEGORIES)}] {cat}.tar already exists ({tar_path.stat().st_size} bytes), skipping")
            skipped += 1
            continue

        url = f"{HF_BASE}/{REPO_ID}/resolve/main/{cat}.tar"
        print(f"[{i+1}/{len(JSONL_CATEGORIES)}] Downloading {cat}.tar...")
        
        try:
            result = subprocess.run(
                ["curl", "-s", "-L", "-o", str(tar_path), url, "--max-time", "600"],
                capture_output=True,
                timeout=620,
            )

            if tar_path.exists() and tar_path.stat().st_size > 1000:
                size_mb = tar_path.stat().st_size / (1024 * 1024)
                msg = f"  OK: {size_mb:.1f} MB"
                print(msg)
                downloaded += 1
            else:
                msg = f"  FAILED: file too small or missing"
                print(msg)
                failed += 1
                try:
                    tar_path.unlink(missing_ok=True)
                except (PermissionError, OSError):
                    pass
        except Exception as e:
            msg = f"  ERROR: {e}"
            print(msg)
            failed += 1
            try:
                tar_path.unlink(missing_ok=True)
            except (PermissionError, OSError):
                pass

    msg = f"\nCOMPLETE: downloaded={downloaded}, failed={failed}, skipped={skipped}"
    print(msg)
    
    # Extract tars
    print("\n=== Extracting tars ===")
    for cat in JSONL_CATEGORIES:
        tar_path = tar_dir / f"{cat}.tar"
        if not tar_path.exists() or tar_path.stat().st_size < 1000:
            continue
        
        extract_dir = WILDRGBD_DIR / cat
        if extract_dir.exists() and any(extract_dir.rglob("*.jpg")):
            print(f"  {cat}: already extracted, skipping")
            continue
        
        print(f"  Extracting {cat}.tar...")
        result = subprocess.run(
            ["tar", "xf", str(tar_path), "-C", str(WILDRGBD_DIR)],
            capture_output=True,
            timeout=300,
        )
        if result.returncode == 0:
            # Count extracted files
            jpg_count = len(list(extract_dir.rglob("*.jpg"))) if extract_dir.exists() else 0
            print(f"    OK: {jpg_count} jpg files")
        else:
            print(f"    FAILED: {result.stderr.decode()[:200]}")
    
    # Create .png symlinks for .jpg files
    print("\n=== Creating .png symlinks ===")
    symlink_count = 0
    for jpg_file in WILDRGBD_DIR.rglob("*.jpg"):
        png_file = jpg_file.with_suffix(".png")
        if not png_file.exists():
            try:
                png_file.symlink_to(jpg_file.name)
                symlink_count += 1
            except (OSError, FileExistsError):
                pass
    print(f"Created {symlink_count} .png symlinks")


if __name__ == "__main__":
    main()
