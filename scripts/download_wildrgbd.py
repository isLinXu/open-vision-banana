#!/usr/bin/env python3
"""
Download WildRGB-D dataset for multiview task.
Source: https://wildrgb.github.io/
HF mirror: try various repos
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


def get_file_list(repo_id: str) -> list[str]:
    """Get list of files from HF API."""
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    return [s["rfilename"] for s in data.get("siblings", [])]


def main():
    WILDRGBD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try different repos
    repos = [
        "wildrgb/wildrgb-d",
        "WildRGB/WildRGB-D",
        "wildrgbd/wildrgbd",
    ]
    
    for repo_id in repos:
        print(f"\nTrying {repo_id}...")
        try:
            files = get_file_list(repo_id)
            print(f"  Found {len(files)} files")
            
            # Filter for relevant files
            archives = [f for f in files if f.endswith(('.tar.gz', '.zip'))]
            images = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"  Archives: {len(archives)}, Images: {len(images)}")
            
            if archives:
                for arch in archives[:5]:
                    print(f"    {arch}")
            
            if images:
                # Download first 100 images as test
                downloaded = 0
                for fpath in images[:100]:
                    local_path = DATASETS_ROOT / fpath
                    if local_path.exists() and local_path.stat().st_size > 100:
                        downloaded += 1
                        continue
                    
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    url = f"{HF_BASE}/{repo_id}/resolve/main/{fpath}"
                    
                    try:
                        result = subprocess.run(
                            ["curl", "-s", "-L", "-o", str(local_path), url, "--max-time", "30"],
                            capture_output=True, timeout=35,
                        )
                        if local_path.exists() and local_path.stat().st_size > 100:
                            downloaded += 1
                        else:
                            local_path.unlink(missing_ok=True)
                    except:
                        local_path.unlink(missing_ok=True)
                
                print(f"  Downloaded: {downloaded}/{len(images[:100])}")
                return
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\nNo suitable repo found. WildRGB-D may need manual download from https://wildrgb.github.io/")


if __name__ == "__main__":
    main()
