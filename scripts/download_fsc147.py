#!/usr/bin/env python3
"""
Download missing FSC147 images from HF mirror.
Current: 1808 images, need ~6146 total.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import urllib.request

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
FSC147_DIR = DATASETS_ROOT / "FSC147" / "images_384_VarV2"
HF_BASE = "https://hf-mirror.com/datasets"

def get_file_list(repo_id: str, prefix: str = "images_384_VarV2/") -> list[str]:
    """Get list of files from HF API."""
    url = f"https://huggingface.co/api/datasets/{repo_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    files = [
        s["rfilename"]
        for s in data.get("siblings", [])
        if s.get("rfilename", "").startswith(prefix)
    ]
    return files


def main():
    repo_id = "nlphuji/fsc147"
    print(f"Fetching file list for {repo_id}...")
    files = get_file_list(repo_id)
    print(f"Found {len(files)} files in repo")

    FSC147_DIR.mkdir(parents=True, exist_ok=True)

    # Check which files already exist
    existing = 0
    to_download = []
    for f in files:
        local_path = DATASETS_ROOT / f
        if local_path.exists() and local_path.stat().st_size > 100:
            existing += 1
        else:
            to_download.append(f)

    print(f"Already downloaded: {existing}/{len(files)}")
    print(f"Need to download: {len(to_download)}")

    if not to_download:
        print("All files present!")
        return

    # Download using curl
    downloaded = 0
    failed = 0

    for i, fpath in enumerate(to_download):
        local_path = DATASETS_ROOT / fpath
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_BASE}/{repo_id}/resolve/main/{fpath}"

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

        if (i + 1) % 500 == 0:
            msg = f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(to_download)} processed, downloaded={downloaded}, failed={failed}"
            print(msg)
            with open("/tmp/fsc147_dl.log", "a") as log:
                log.write(msg + "\n")

    msg = f"[{time.strftime('%H:%M:%S')}] COMPLETE: downloaded={downloaded}, failed={failed}, total={len(to_download)}"
    print(msg)
    with open("/tmp/fsc147_dl.log", "a") as log:
        log.write(msg + "\n")


if __name__ == "__main__":
    main()
