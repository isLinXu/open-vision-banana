#!/usr/bin/env python3
"""Download BDD100K and VisDrone image files using HF API file listing + curl."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
import urllib.request

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
HF_BASE = "https://hf-mirror.com/datasets"


def get_file_list(repo_id: str, prefix: str = "data/") -> list[str]:
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


def download_files(repo_id: str, files: list[str], local_root: Path, log_file: str):
    """Download files using curl."""
    total = len(files)
    downloaded = 0
    failed = 0

    for i, fpath in enumerate(files):
        local_path = local_root / fpath
        if local_path.exists() and local_path.stat().st_size > 100:
            downloaded += 1
            continue

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
            msg = f"[{time.strftime('%H:%M:%S')}] {log_file}: {i + 1}/{total} processed, downloaded={downloaded}, failed={failed}"
            print(msg)
            with open(f"/tmp/{log_file}", "a") as f:
                f.write(msg + "\n")

    msg = f"[{time.strftime('%H:%M:%S')}] {log_file} COMPLETE: downloaded={downloaded}, failed={failed}, total={total}"
    print(msg)
    with open(f"/tmp/{log_file}", "a") as f:
        f.write(msg + "\n")


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "bdd100k"

    if dataset == "bdd100k":
        repo_id = "dgural/bdd100k"
        local_root = DATASETS_ROOT / "BDD100K"
        log_name = "bdd100k_dl.log"
    elif dataset == "visdrone":
        repo_id = "Voxel51/VisDrone2019-DET"
        local_root = DATASETS_ROOT / "VisDrone2019-DET"
        log_name = "visdrone_dl.log"
    else:
        print(f"Unknown dataset: {dataset}")
        sys.exit(1)

    print(f"Fetching file list for {repo_id}...")
    files = get_file_list(repo_id)
    print(f"Found {len(files)} files to download")

    download_files(repo_id, files, local_root, log_name)
