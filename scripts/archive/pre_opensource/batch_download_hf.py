#!/usr/bin/env python3
"""
Batch download all missing source datasets using huggingface_hub.
Uses parallel downloads with snapshot_download for maximum efficiency.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")

# ============================================================
# Download plan: (dataset_name, hf_repo_id, task, subfolder_filter, local_dir, description)
# subfolder_filter: only download files matching this prefix, or None for all
# ============================================================

DOWNLOAD_PLAN = [
    # --- Phase 1A: High priority ---
    {
        "name": "LoveDA Train+Val Images",
        "repo_id": "chloechia/loveda",
        "task": "segmentation",
        "patterns": ["urban:rural train images/", "urban:rural val images/"],
        "local_dir": "LoveDA",
        "repo_type": "dataset",
        "estimated_files": 4320,
    },
    {
        "name": "BDD100K Images",
        "repo_id": "dgural/bdd100k",
        "task": "keypoint",
        "patterns": ["data/"],
        "local_dir": "BDD100K",
        "repo_type": "dataset",
        "estimated_files": 20000,
    },
    {
        "name": "Objects365 patches (for depth)",
        "repo_id": "guozonghao96/objects365",
        "task": "depth",
        "patterns": [f"patch{i}.tar.gz" for i in range(45)],
        "local_dir": "objects365",
        "repo_type": "dataset",
        "estimated_files": 45,
    },
    # --- Phase 1B: Medium priority ---
    {
        "name": "VisDrone2019-DET train",
        "repo_id": "Voxel51/VisDrone2019-DET",
        "task": "keypoint",
        "patterns": ["data/"],
        "local_dir": "VisDrone2019-DET",
        "repo_type": "dataset",
        "estimated_files": 6471,
    },
    {
        "name": "DeepFashion2",
        "repo_id": "sahirp/deepfashion2",
        "task": "keypoint",
        "patterns": None,
        "local_dir": "DeepFashion2",
        "repo_type": "dataset",
        "estimated_files": 1000,
    },
    # --- Phase 1C: Small segmentation datasets ---
    {
        "name": "Flickr30k Images",
        "repo_id": "HuggingFaceM4/flickr30k",
        "task": "segmentation",
        "patterns": None,
        "local_dir": "flickr30k",
        "repo_type": "dataset",
        "estimated_files": 3,
    },
]


def download_dataset(entry: dict, max_retries: int = 3) -> dict:
    """Download a dataset from HuggingFace.
    Returns dict with status info."""
    name = entry["name"]
    repo_id = entry["repo_id"]
    repo_type = entry.get("repo_type", "dataset")
    patterns = entry.get("patterns")
    local_dir = DATASETS_ROOT / entry["local_dir"]

    print(f"\n{'=' * 60}")
    print(f"Downloading: {name}")
    print(f"  Repo: {repo_id} ({repo_type})")
    print(f"  Local: {local_dir}")
    print(f"  Patterns: {patterns[:3] if patterns else 'all'}...")
    print(f"{'=' * 60}")

    local_dir.mkdir(parents=True, exist_ok=True)

    # List files in repo
    try:
        all_files = list_repo_files(repo_id, repo_type=repo_type)
        print(f"  Repo has {len(all_files)} files total")
    except Exception as e:
        print(f"  [ERROR] Failed to list repo files: {e}")
        return {"name": name, "status": "error", "error": str(e)}

    # Filter by patterns
    if patterns:
        target_files = []
        for f in all_files:
            for pat in patterns:
                if f.startswith(pat) or f == pat:
                    target_files.append(f)
                    break
    else:
        target_files = all_files

    print(f"  Target files to download: {len(target_files)}")

    if len(target_files) == 0:
        print(f"  [SKIP] No matching files found")
        return {"name": name, "status": "skip", "reason": "no matching files"}

    # Check which files already exist
    existing = 0
    to_download = []
    for f in target_files:
        local_path = local_dir / f
        if local_path.exists() and local_path.stat().st_size > 100:
            existing += 1
        else:
            to_download.append(f)

    print(f"  Already downloaded: {existing}/{len(target_files)}")
    print(f"  Need to download: {len(to_download)}")

    if len(to_download) == 0:
        print(f"  [DONE] All files already present")
        return {
            "name": name,
            "status": "skip",
            "reason": "all present",
            "total": len(target_files),
            "existing": existing,
        }

    # Download in batches
    downloaded = 0
    failed = 0
    batch_size = 50  # Download in batches to manage memory

    for i in range(0, len(to_download), batch_size):
        batch = to_download[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(to_download) + batch_size - 1) // batch_size

        for f in batch:
            for attempt in range(max_retries):
                try:
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=f,
                        repo_type=repo_type,
                        local_dir=str(local_dir),
                        token=None,
                    )
                    downloaded += 1
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        failed += 1
                        if failed <= 5:
                            print(f"    [FAIL] {f}: {e}")
                    else:
                        time.sleep(2)

        print(f"  Batch {batch_num}/{total_batches}: downloaded={downloaded}, failed={failed}")

    print(f"\n  Result: downloaded={downloaded}, failed={failed}, existing={existing}")
    print(f"  Total available: {existing + downloaded}/{len(target_files)}")

    return {
        "name": name,
        "status": "done" if failed == 0 else "partial",
        "total": len(target_files),
        "downloaded": downloaded,
        "failed": failed,
        "existing": existing,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch download missing datasets")
    parser.add_argument("--task", help="Filter by task (depth, segmentation, keypoint, etc.)")
    parser.add_argument("--name", help="Filter by dataset name (substring match)")
    parser.add_argument("--list", action="store_true", help="List plan and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("Open-Vision-Banana Dataset Batch Downloader")
    print(f"  HF Endpoint: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
    print(f"  Datasets root: {DATASETS_ROOT}")
    print("=" * 60)

    plan = DOWNLOAD_PLAN
    if args.task:
        plan = [d for d in plan if d["task"] == args.task]
    if args.name:
        plan = [d for d in plan if args.name.lower() in d["name"].lower()]

    if args.list or not plan:
        print("\nDownload plan:")
        for i, d in enumerate(plan):
            print(
                f"  {i + 1}. [{d['task']:12s}] {d['name']} ({d['repo_id']}, ~{d['estimated_files']} files)"
            )
        if not plan:
            print("  (no matching datasets)")
        return

    print(f"\nDatasets to process: {len(plan)}")

    results = []
    for entry in plan:
        result = download_dataset(entry)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for r in results:
        status_icon = (
            "OK" if r["status"] == "done" else ("SKIP" if r["status"] == "skip" else "PARTIAL")
        )
        total = r.get("total", 0)
        avail = r.get("existing", 0) + r.get("downloaded", 0)
        print(f"  [{status_icon:7s}] {r['name']:<30s} {avail}/{total}")

    print("\nDone!")


if __name__ == "__main__":
    main()
