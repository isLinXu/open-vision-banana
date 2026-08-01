#!/usr/bin/env python3
"""
Download only the WildRGB-D images referenced in the JSONL file.
JSONL references .png files, but HF repo has .jpg files.
Strategy: Download .jpg and create .png symlink to .jpg.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
import urllib.request

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
WILDRGBD_DIR = DATASETS_ROOT / "WildRGB-D"
HF_BASE = "https://hf-mirror.com/datasets"
REPO_ID = "dwzdwz321/wildrgbd"
JSONL_PATH = DATASETS_ROOT / "snvc50m" / "wildrgbd_total_unified_recon_edit_regenerated_sampled.jsonl"


def main():
    # Read all image paths from JSONL
    image_paths = set()
    with open(JSONL_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            img = entry.get("image", "")
            if isinstance(img, list):
                img = img[0] if img else ""
            if img:
                image_paths.add(img)
    
    print(f"Total unique image paths in JSONL: {len(image_paths)}")
    
    WILDRGBD_DIR.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    for i, img_path in enumerate(sorted(image_paths)):
        # JSONL path: boat/scenes/scene_112/rgb/00000.png
        # HF repo path: boat/scenes/scene_112/rgb/00000.jpg
        png_path = WILDRGBD_DIR / img_path
        jpg_path = png_path.with_suffix(".jpg")
        
        # Check if already exists (either .png or .jpg)
        if png_path.exists() and png_path.stat().st_size > 100:
            skipped += 1
            continue
        if jpg_path.exists() and jpg_path.stat().st_size > 100:
            # Create .png symlink to .jpg
            png_path.parent.mkdir(parents=True, exist_ok=True)
            if not png_path.exists():
                png_path.symlink_to(jpg_path.name)
            skipped += 1
            continue
        
        # Download .jpg version
        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        hf_path = str(jpg_path.relative_to(WILDRGBD_DIR))
        url = f"{HF_BASE}/{REPO_ID}/resolve/main/{hf_path}"
        
        try:
            result = subprocess.run(
                ["curl", "-s", "-L", "-o", str(jpg_path), url, "--max-time", "30"],
                capture_output=True,
                timeout=35,
            )
            
            if jpg_path.exists() and jpg_path.stat().st_size > 100:
                # Create .png symlink to .jpg
                if not png_path.exists():
                    png_path.symlink_to(jpg_path.name)
                downloaded += 1
            else:
                failed += 1
                try:
                    jpg_path.unlink(missing_ok=True)
                except (PermissionError, OSError):
                    pass
        except Exception as e:
            failed += 1
            try:
                jpg_path.unlink(missing_ok=True)
            except (PermissionError, OSError):
                pass
        
        if (i + 1) % 50 == 0:
            msg = f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(image_paths)}: downloaded={downloaded}, failed={failed}, skipped={skipped}"
            print(msg)
            with open("/tmp/wildrgbd_dl.log", "a") as log:
                log.write(msg + "\n")
    
    msg = f"[{time.strftime('%H:%M:%S')}] COMPLETE: downloaded={downloaded}, failed={failed}, skipped={skipped}, total={len(image_paths)}"
    print(msg)
    with open("/tmp/wildrgbd_dl.log", "a") as log:
        log.write(msg + "\n")


if __name__ == "__main__":
    main()
