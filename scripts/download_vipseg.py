#!/usr/bin/env python3
"""
Download VIPSeg dataset from HF mirror.
The previous VIPSeg.zip was incomplete (only 968M, should be ~10GB).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import urllib.request

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
VIPSEG_DIR = DATASETS_ROOT / "VIPSeg"
HF_BASE = "https://hf-mirror.com/datasets"


def get_file_list(repo_id: str) -> list[str]:
    """Get list of files from HF API."""
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    print(f"Fetching file list from {url}...")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    files = [s["rfilename"] for s in data.get("siblings", [])]
    return files


def main():
    # Try different repo IDs for VIPSeg
    repos = [
        "Qnancy/VIPSeg",
        "zhangtao-whu/VIPSeg",
    ]
    
    for repo_id in repos:
        print(f"\n{'='*60}")
        print(f"Trying repo: {repo_id}")
        print(f"{'='*60}")
        
        try:
            files = get_file_list(repo_id)
            print(f"Found {len(files)} files")
            
            # Filter for image files
            image_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
            print(f"Image files: {len(image_files)}")
            
            if not image_files:
                # Try to download the main zip
                zip_files = [f for f in files if f.endswith('.zip') or f.endswith('.tar.gz')]
                print(f"Archive files: {zip_files}")
                
                for zf in zip_files:
                    local_path = VIPSEG_DIR / zf
                    if local_path.exists() and local_path.stat().st_size > 9_000_000_000:  # >9GB
                        print(f"Already downloaded: {zf}")
                        continue
                    
                    url = f"{HF_BASE}/{repo_id}/resolve/main/{zf}"
                    print(f"Downloading {zf} from {url}...")
                    
                    # Use curl with resume support
                    result = subprocess.run(
                        ["curl", "-L", "-o", str(local_path), url, "--max-time", "600"],
                        timeout=610,
                    )
                    
                    if local_path.exists() and local_path.stat().st_size > 1_000_000_000:
                        print(f"Downloaded: {local_path.stat().st_size / 1e9:.2f} GB")
                    else:
                        print(f"Download failed or too small: {local_path.stat().st_size if local_path.exists() else 0}")
                
                continue
            
            # Download individual image files
            VIPSEG_DIR.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            failed = 0
            
            for i, fpath in enumerate(image_files[:100]):  # Limit to first 100 for testing
                local_path = DATASETS_ROOT / fpath
                if local_path.exists() and local_path.stat().st_size > 100:
                    downloaded += 1
                    continue
                
                local_path.parent.mkdir(parents=True, exist_ok=True)
                url = f"{HF_BASE}/{repo_id}/resolve/main/{fpath}"
                
                try:
                    result = subprocess.run(
                        ["curl", "-s", "-L", "-o", str(local_path), url, "--max-time", "30"],
                        capture_output=True,
                        timeout=35,
                    )
                    if local_path.exists() and local_path.stat().st_size > 100:
                        downloaded += 1
                    else:
                        failed += 1
                        local_path.unlink(missing_ok=True)
                except:
                    failed += 1
                    local_path.unlink(missing_ok=True)
                
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(image_files[:100])}: downloaded={downloaded}, failed={failed}")
            
            print(f"Result: downloaded={downloaded}, failed={failed}")
            
        except Exception as e:
            print(f"Error with {repo_id}: {e}")
            continue


if __name__ == "__main__":
    main()
