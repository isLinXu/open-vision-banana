#!/usr/bin/env python3
"""
Segmentation dataset image download script

Downloads, file by file from the HuggingFace mirror, the missing segmentation source images referenced by the SNVC50M JSONL.
Avoids the huggingface_hub Python API (OOM) and the hf download CLI (unstable).
Uses only per-file curl downloads.

Strategy:
  1. Read the JSONL to get the list of required files
  2. Get the repo file list with the HF API (urllib)
  3. Download each file with curl
  4. Skip files that already exist (size > 100 bytes)
  5. Skip repos that do not exist or are not accessible

Usage:
  python scripts/download_seg_datasets.py
  python scripts/download_seg_datasets.py --dataset TrashCan
  python scripts/download_seg_datasets.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
SNVC_ROOT = DATASETS_ROOT / "snvc50m"
HF_MIRROR = "https://hf-mirror.com/datasets"
HF_ORIGINAL = "https://huggingface.co/datasets"
# Use hf-mirror.com as primary (better access from China)
HF_BASE = HF_MIRROR

# ── Dataset configs ───────────────────────────────────────────────────────────

# Each config defines:
#   jsonl_name:  the _sampled.jsonl file (without extension)
#   hf_repo:    HuggingFace repo id (org/name)
#   local_dir:  local save directory under DATASETS_ROOT
#   jsonl_path_field:  field in JSONL that contains the image path
#   path_map:   function(jsonl_path) -> hf_path (for path translation)
#               if None, use jsonl_path directly
#   ext_change: optional extension change (e.g. .jpg -> .png for download)

DATASET_CONFIGS = [
    # 1. TAS500 - dataset-ninja/tas500 (PRIVATE/401)
    {
        "name": "TAS500",
        "jsonl_names": ["tas500_genseg_sampled", "tas500_gcg_sampled"],
        "hf_repo": "dataset-ninja/tas500",
        "local_dir": "TAS500",
        "description": "TAS500 aerial segmentation dataset",
    },
    # 2. StreetHazards - fedemarchits/streethazards (tar only, 10.7GB total)
    {
        "name": "StreetHazards",
        "jsonl_names": ["streethazards_genseg_sampled", "streethazards_gcg_sampled"],
        "hf_repo": "fedemarchits/streethazards",
        "local_dir": "StreetHazards",
        "description": "StreetHazards open-set semantic segmentation",
        "tar_only": True,
        "tar_urls": {
            "training": f"{HF_BASE}/fedemarchits/streethazards/resolve/main/streethazards_train.tar",
            "validation": f"{HF_BASE}/fedemarchits/streethazards/resolve/main/streethazards_test.tar",
        },
    },
    # 3. TrashCan - anyaeross/trashcan1 (100% file match, train/ dir)
    {
        "name": "TrashCan",
        "jsonl_names": ["trashcan_genseg_sampled"],
        "hf_repo": "anyaeross/trashcan1",
        "local_dir": "TrashCan",
        "description": "TrashCan underwater debris segmentation",
        # JSONL path: train/vid_xxx_framexxxxxxxx.jpg
        # HF path:   train/vid_xxx_framexxxxxxxx.jpg  (direct match)
    },
    # 4. TTPLA - V8heart/yolino-ttpla-benchmark (78.6% match, png vs jpg)
    {
        "name": "TTPLA",
        "jsonl_names": ["ttpla_genseg_sampled"],
        "hf_repo": "V8heart/yolino-ttpla-benchmark",
        "local_dir": "TTPLA",
        "description": "TTPLA transmission tower & power line segmentation",
        # JSONL path: data_original_size_v1/data_original_size/04_585.jpg
        # HF path:   images/{train,test,val}/04_585.png
        # Need filename-based matching with extension change
        "needs_filename_match": True,
    },
    # 5. ZeroWaste - No HF repo found
    {
        "name": "ZeroWaste",
        "jsonl_names": ["zerowaste_genseg_sampled"],
        "hf_repo": None,  # No known HF repo
        "local_dir": "ZeroWaste",
        "description": "ZeroWaste-f waste segmentation",
    },
]


# ── Utility functions ────────────────────────────────────────────────────────

def read_jsonl_image_paths(jsonl_path: Path) -> set[str]:
    """Read a JSONL file and extract all unique image paths."""
    paths = set()
    if not jsonl_path.exists():
        print(f"  [warn] JSONL not found: {jsonl_path}")
        return paths
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                img = d.get("image", "")
                if isinstance(img, list):
                    img = img[0] if img else ""
                if img:
                    paths.add(img)
            except json.JSONDecodeError:
                continue
    return paths


def get_hf_file_list(repo: str) -> Optional[list[dict]]:
    """Get file list from HuggingFace API using urllib.
    Returns list of {path, size} dicts, or None if repo inaccessible.
    """
    # Try both mirror and original
    for base in [HF_MIRROR, HF_ORIGINAL]:
        url = f"{base.replace('/datasets', '')}/api/datasets/{repo}/tree/main?recursive=true"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            files = []
            for item in data:
                if item.get("type") == "file":
                    files.append({
                        "path": item.get("path", ""),
                        "size": item.get("size", 0),
                    })
            return files
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(f"  [error] Repo {repo} requires authentication (401)")
                return None
            elif e.code == 404:
                print(f"  [error] Repo {repo} not found (404)")
                return None
            else:
                continue
        except Exception as e:
            continue
    print(f"  [error] Cannot access repo {repo}")
    return None


def get_hf_siblings(repo: str) -> Optional[list[str]]:
    """Get file list using the siblings API endpoint (alternative to tree API)."""
    for base in [HF_MIRROR, HF_ORIGINAL]:
        url = f"{base.replace('/datasets', '')}/api/datasets/{repo}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            siblings = data.get("siblings", [])
            return [s.get("rfilename", "") for s in siblings if s.get("rfilename")]
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(f"  [error] Repo {repo} requires authentication (401)")
                return None
            elif e.code == 404:
                print(f"  [error] Repo {repo} not found (404)")
                return None
            else:
                continue
        except Exception:
            continue
    print(f"  [error] Cannot access repo {repo}")
    return None


def download_file_curl(url: str, out_path: Path, timeout: int = 60, retries: int = 3) -> bool:
    """Download a single file using curl. Returns True on success."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        result = subprocess.run(
            [
                "curl", "-s", "-L",
                "--retry", "2",
                "--retry-delay", "3",
                "--max-time", str(timeout),
                "-o", str(out_path),
                "-w", "\\n%{http_code}",  # Append HTTP status code
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 30,
        )

        # Check HTTP status code from output
        output_lines = result.stdout.strip().split("\n")
        http_code = output_lines[-1] if output_lines else "000"

        if out_path.exists() and out_path.stat().st_size > 100 and http_code == "200":
            return True

        # Clean up failed download
        if out_path.exists() and out_path.stat().st_size < 100:
            out_path.unlink(missing_ok=True)

        if attempt < retries - 1:
            time.sleep(2)

    return False


def stream_extract_from_tar(
    tar_url: str, needed_files: set[str], dest_dir: Path, timeout: int = 7200
) -> int:
    """Stream-download a tar and extract only needed files.
    Returns count of successfully extracted files.
    """
    if not needed_files:
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0

    # Build tar extraction patterns
    # Each needed file becomes a --include pattern for tar
    # We'll extract them one batch at a time to avoid command line length limits
    batch_size = 50
    needed_list = sorted(needed_files)

    for i in range(0, len(needed_list), batch_size):
        batch = needed_list[i : i + batch_size]
        patterns = []
        for f in batch:
            # tar --wildcards uses glob patterns
            patterns.extend(["--wildcards", f"--include={f}"])

        # Skip files that already exist
        to_download = []
        for f in batch:
            local_path = dest_dir / f
            if not (local_path.exists() and local_path.stat().st_size > 100):
                to_download.append(f)

        if not to_download:
            extracted += len(batch)
            continue

        patterns = []
        for f in to_download:
            patterns.extend(["--wildcards", f"--include={f}"])

        print(f"    Streaming tar batch {i//batch_size + 1}/{(len(needed_list)-1)//batch_size + 1} "
              f"({len(to_download)} files)...", end=" ", flush=True)

        cmd = ["curl", "-s", "-L", tar_url, "--max-time", str(timeout)] + ["|"] + \
              ["tar", "-xf", "-", "-C", str(dest_dir)] + patterns

        # Use shell pipe approach
        shell_cmd = (
            f'curl -s -L "{tar_url}" --max-time {timeout} | '
            f'tar -xf - -C "{dest_dir}" {" ".join(patterns)}'
        )

        result = subprocess.run(
            ["bash", "-c", shell_cmd],
            capture_output=True,
            text=True,
            timeout=timeout + 60,
        )

        # Count how many files we actually got
        batch_extracted = 0
        for f in to_download:
            local_path = dest_dir / f
            if local_path.exists() and local_path.stat().st_size > 100:
                batch_extracted += 1
                extracted += 1

        print(f"got {batch_extracted}/{len(to_download)}")

    return extracted


# ── Per-dataset download functions ────────────────────────────────────────────

def download_standard_dataset(config: dict, dry_run: bool = False) -> dict:
    """Download a dataset where files are individually accessible on HF.
    Used for TrashCan (direct path match) and similar datasets.
    """
    name = config["name"]
    jsonl_names = config["jsonl_names"]
    hf_repo = config["hf_repo"]
    local_dir = config["local_dir"]
    dest = DATASETS_ROOT / local_dir

    stats = {"name": name, "total_needed": 0, "downloaded": 0, "skipped": 0, "failed": 0, "errors": []}

    if not hf_repo:
        msg = f"No HF repo configured for {name}"
        stats["errors"].append(msg)
        print(f"  [skip] {msg}")
        return stats

    # Collect all needed paths from JSONL files
    all_needed = set()
    for jl_name in jsonl_names:
        jl_path = SNVC_ROOT / f"{jl_name}.jsonl"
        paths = read_jsonl_image_paths(jl_path)
        all_needed.update(paths)
        print(f"  {jl_name}: {len(paths)} unique paths")

    stats["total_needed"] = len(all_needed)
    print(f"  Total unique files needed: {len(all_needed)}")

    if dry_run:
        print(f"  [DRY-RUN] Would download {len(all_needed)} files from {hf_repo}")
        return stats

    # Check if repo is accessible
    # Use siblings API first (returns complete file list, unlike tree API which may be truncated)
    print(f"  Checking repo {hf_repo}...")
    sibling_names = get_hf_siblings(hf_repo)
    if sibling_names is not None:
        available_paths = set(sibling_names)
        filename_to_path = {}
        # Also build a basename (no extension) -> path index for cross-extension matching
        basename_to_path = {}
        for p in sibling_names:
            fname = p.split("/")[-1]
            filename_to_path[fname] = p
            fname_base = fname.rsplit(".", 1)[0] if "." in fname else fname
            basename_to_path[fname_base] = p
    else:
        # Fall back to tree API
        print(f"  Trying tree API...")
        hf_files = get_hf_file_list(hf_repo)
        if hf_files is None:
            msg = f"Cannot access repo {hf_repo} (tried siblings + tree API)"
            stats["errors"].append(msg)
            return stats
        available_paths = set(f["path"] for f in hf_files)
        filename_to_path = {}
        basename_to_path = {}
        for f in hf_files:
            fname = f["path"].split("/")[-1]
            filename_to_path[fname] = f["path"]
            fname_base = fname.rsplit(".", 1)[0] if "." in fname else fname
            basename_to_path[fname_base] = f["path"]

    print(f"  Repo has {len(available_paths)} files")

    # Download each needed file
    for i, needed_path in enumerate(sorted(all_needed)):
        local_path = dest / needed_path

        # Skip if already exists
        if local_path.exists() and local_path.stat().st_size > 100:
            stats["skipped"] += 1
            continue

        # Determine HF path
        hf_path = None
        if needed_path in available_paths:
            # Direct path match
            hf_path = needed_path
        elif config.get("needs_filename_match"):
            # Try matching by basename (no extension) for cross-extension matching
            # e.g. TTPLA: JSONL has .jpg, HF has .png with different directory structure
            fname = needed_path.split("/")[-1]
            fname_base = fname.rsplit(".", 1)[0] if "." in fname else fname
            if fname_base in basename_to_path:
                hf_path = basename_to_path[fname_base]
        else:
            # Try matching by exact filename (different directory, same extension)
            fname = needed_path.split("/")[-1]
            if fname in filename_to_path:
                hf_path = filename_to_path[fname]

        if hf_path is None:
            stats["failed"] += 1
            if i < 5:
                print(f"    [not found] {needed_path}")
            continue

        # Download
        url = f"{HF_BASE}/{hf_repo}/resolve/main/{urllib.parse.quote(hf_path)}"
        if download_file_curl(url, local_path):
            stats["downloaded"] += 1
            if (stats["downloaded"] % 50) == 0:
                print(f"    Progress: {stats['downloaded']} downloaded, "
                      f"{stats['skipped']} skipped, {stats['failed']} failed")
        else:
            stats["failed"] += 1
            if stats["failed"] <= 5:
                print(f"    [failed] {needed_path} -> {hf_path}")

    return stats


def download_tar_dataset(config: dict, dry_run: bool = False) -> dict:
    """Download a dataset where files are only available as tar archives.
    Used for StreetHazards.
    """
    name = config["name"]
    jsonl_names = config["jsonl_names"]
    local_dir = config["local_dir"]
    tar_urls = config.get("tar_urls", {})
    dest = DATASETS_ROOT / local_dir

    stats = {"name": name, "total_needed": 0, "downloaded": 0, "skipped": 0, "failed": 0, "errors": []}

    # Collect all needed paths
    all_needed = set()
    for jl_name in jsonl_names:
        jl_path = SNVC_ROOT / f"{jl_name}.jsonl"
        paths = read_jsonl_image_paths(jl_path)
        all_needed.update(paths)
        print(f"  {jl_name}: {len(paths)} unique paths")

    stats["total_needed"] = len(all_needed)
    print(f"  Total unique files needed: {len(all_needed)}")

    if dry_run:
        for prefix, url in tar_urls.items():
            print(f"  [DRY-RUN] Would stream-extract from {url}")
            needed_for_prefix = {p for p in all_needed if p.startswith(prefix)}
            print(f"    {prefix}: {len(needed_for_prefix)} files")
        return stats

    # Group needed files by tar prefix
    for prefix, tar_url in tar_urls.items():
        needed_for_prefix = {p for p in all_needed if p.startswith(prefix)}
        if not needed_for_prefix:
            print(f"  No files needed from {prefix} tar")
            continue

        # Filter out already-downloaded files
        to_download = set()
        for f in needed_for_prefix:
            local_path = dest / f
            if local_path.exists() and local_path.stat().st_size > 100:
                stats["skipped"] += 1
            else:
                to_download.add(f)

        if not to_download:
            print(f"  All {len(needed_for_prefix)} files for {prefix} already exist")
            continue

        print(f"\n  Streaming {prefix} tar ({len(to_download)} files needed)...")
        print(f"  URL: {tar_url}")

        extracted = stream_extract_from_tar(tar_url, to_download, dest, timeout=7200)
        stats["downloaded"] += extracted
        stats["failed"] += len(to_download) - extracted

    return stats


# ── Main execution ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download missing segmentation source images from HuggingFace"
    )
    parser.add_argument(
        "--dataset",
        help="Download only specific dataset (TAS500, StreetHazards, TrashCan, TTPLA, ZeroWaste)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    args = parser.parse_args()

    print("=" * 70)
    print("Segmentation Dataset Image Downloader")
    print("=" * 70)
    print(f"  Datasets root: {DATASETS_ROOT}")
    print(f"  SNVC JSONL dir: {SNVC_ROOT}")
    print(f"  HF mirror: {HF_BASE}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # Select datasets to process
    configs = DATASET_CONFIGS
    if args.dataset:
        configs = [c for c in configs if c["name"].lower() == args.dataset.lower()]
        if not configs:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {[c['name'] for c in DATASET_CONFIGS]}")
            sys.exit(1)

    all_stats = []

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Processing: {config['name']}")
        print(f"  Description: {config.get('description', 'N/A')}")
        print(f"  HF repo: {config.get('hf_repo', 'N/A')}")
        print(f"  Local dir: {config['local_dir']}")
        print(f"{'=' * 70}")

        if config.get("tar_only"):
            stats = download_tar_dataset(config, dry_run=args.dry_run)
        elif config.get("hf_repo") is None:
            msg = f"No HF repo available for {config['name']}"
            print(f"  [skip] {msg}")
            stats = {
                "name": config["name"],
                "total_needed": 0,
                "downloaded": 0,
                "skipped": 0,
                "failed": 0,
                "errors": [msg],
            }
        else:
            stats = download_standard_dataset(config, dry_run=args.dry_run)

        all_stats.append(stats)

        # Print per-dataset summary
        print(f"\n  --- {config['name']} Summary ---")
        print(f"    Total needed:  {stats['total_needed']}")
        print(f"    Downloaded:    {stats['downloaded']}")
        print(f"    Skipped (existing): {stats['skipped']}")
        print(f"    Failed:        {stats['failed']}")
        if stats["errors"]:
            print(f"    Errors:")
            for e in stats["errors"]:
                print(f"      - {e}")

    # Print overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<20} {'Needed':>8} {'Downloaded':>12} {'Skipped':>10} {'Failed':>8}")
    print("-" * 60)

    total_needed = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for s in all_stats:
        print(
            f"{s['name']:<20} {s['total_needed']:>8} {s['downloaded']:>12} "
            f"{s['skipped']:>10} {s['failed']:>8}"
        )
        total_needed += s["total_needed"]
        total_downloaded += s["downloaded"]
        total_skipped += s["skipped"]
        total_failed += s["failed"]

    print("-" * 60)
    print(
        f"{'TOTAL':<20} {total_needed:>8} {total_downloaded:>12} "
        f"{total_skipped:>10} {total_failed:>8}"
    )
    print()


if __name__ == "__main__":
    main()
