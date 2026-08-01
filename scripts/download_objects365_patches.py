#!/usr/bin/env python3
"""
Download Objects365 patches 14-44 from HF mirror.
Patches 0-13 are already downloaded and extracted.
"""
import json
import subprocess
import sys
import time
from pathlib import Path
import urllib.request

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
PATCHES_DIR = DATASETS_ROOT / "Objects365" / "patches"
HF_BASE = "https://hf-mirror.com/datasets"
REPO_ID = "guozonghao96/objects365"

def main():
    # Download patches 14-44
    patches_to_download = list(range(14, 45))
    
    print(f"Objects365 patch downloader")
    print(f"Repo: {REPO_ID}")
    print(f"Patches to download: {patches_to_download[0]}-{patches_to_download[-1]} ({len(patches_to_download)} patches)")
    print()
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    for patch_num in patches_to_download:
        patch_name = f"patch{patch_num}.tar.gz"
        local_path = PATCHES_DIR / patch_name
        
        if local_path.exists() and local_path.stat().st_size > 1_000_000:  # >1MB
            print(f"[SKIP] {patch_name} already exists ({local_path.stat().st_size / 1e9:.2f} GB)")
            skipped += 1
            continue
        
        url = f"{HF_BASE}/{REPO_ID}/resolve/main/{patch_name}"
        print(f"[DOWNLOAD] {patch_name} from {url}...")
        
        try:
            result = subprocess.run(
                ["curl", "-L", "-o", str(local_path), url, "--max-time", "600"],
                timeout=610,
            )
            
            if local_path.exists() and local_path.stat().st_size > 1_000_000:
                size_gb = local_path.stat().st_size / 1e9
                print(f"  OK: {size_gb:.2f} GB")
                downloaded += 1
                
                # Extract immediately
                print(f"  Extracting...")
                extract_dir = DATASETS_ROOT / "Objects365" / "extracted"
                extract_dir.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    ["tar", "-xzf", str(local_path), "-C", str(extract_dir), "--skip-old-files"],
                    timeout=300,
                )
                
                # Create symlink if not exists
                patch_dir = DATASETS_ROOT / "Objects365" / f"patch{patch_num}"
                if not patch_dir.exists():
                    patch_dir.symlink_to(extract_dir / f"patch{patch_num}")
                
                print(f"  Extracted and linked")
            else:
                print(f"  FAILED: too small or missing")
                failed += 1
                local_path.unlink(missing_ok=True)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
            failed += 1
            local_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
            local_path.unlink(missing_ok=True)
    
    print(f"\nSummary: downloaded={downloaded}, failed={failed}, skipped={skipped}")
    
    # Log
    with open("/tmp/objects365_patches_dl.log", "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] downloaded={downloaded}, failed={failed}, skipped={skipped}\n")


if __name__ == "__main__":
    main()
