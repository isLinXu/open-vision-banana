#!/usr/bin/env python3
"""
Download nuImages and EgoObjects images for SNVC-50M.

nuImages: Extract 1000 specific images from kd7/graid-nuimages parquet files on HuggingFace.
EgoObjects: Try multiple sources for EgoObjectsV1_images.zip.
"""

import json
import os
import sys
import urllib.request
import urllib.parse
import time
from pathlib import Path
from collections import defaultdict

import pyarrow.parquet as pq
import fsspec

# === Configuration ===
NUIMAGES_JSONL = "/Users/gatilin/MyWork/datasets/snvc50m/nuimages_point_sampled.jsonl"
EGOOBJECTS_JSONL = "/Users/gatilin/MyWork/datasets/snvc50m/egoobjects_point_sampled.jsonl"

NUIMAGES_TARGET = "/Users/gatilin/MyWork/datasets/nuimages"
EGOOBJECTS_TARGET = "/Users/gatilin/MyWork/datasets/EgoObjects/images"

GRAID_PARQUET_URL = "https://huggingface.co/datasets/kd7/graid-nuimages/resolve/main/data/train-{:05d}-of-00287.parquet"
NUM_PARQUETS = 287

# Progress file to track which parquets we've already scanned
PROGRESS_FILE = "/tmp/nuimages_download_progress.json"


def load_needed_images(jsonl_path, path_key=None):
    """Load set of needed image paths from JSONL."""
    needed = set()
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if path_key:
                needed.add(data[path_key])
            else:
                for key in ['image', 'image_path', 'file_name', 'filename', 'path', 'image_filename']:
                    if key in data:
                        needed.add(data[key])
                        break
    return needed


def load_progress():
    """Load download progress."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"scanned_parquets": [], "found": {}, "downloaded": []}


def save_progress(progress):
    """Save download progress."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def scan_parquet_for_matches(parquet_idx, needed_source_ids):
    """
    Scan a single parquet file for matching source_ids.
    Returns dict: {source_id: row_index} for all matches.
    """
    url = GRAID_PARQUET_URL.format(parquet_idx)
    matches = {}

    try:
        fs = fsspec.filesystem('https')
        with fs.open(url, 'rb') as f:
            pf = pq.ParquetFile(f)
            num_row_groups = pf.metadata.num_row_groups

            for rg_idx in range(num_row_groups):
                # Read only source_id and id columns (tiny)
                table = pf.read_row_group(rg_idx, columns=['source_id', 'id'])
                source_ids = table.column('source_id').to_pylist()
                ids = table.column('id').to_pylist()

                for sid, rid in zip(source_ids, ids):
                    if sid in needed_source_ids:
                        matches[sid] = {
                            'parquet_idx': parquet_idx,
                            'row_group': rg_idx,
                            'row_id': rid,
                            'global_id': rid
                        }

    except Exception as e:
        print(f"  Error scanning parquet {parquet_idx}: {e}")

    return matches


def download_image_from_parquet(parquet_idx, row_group, target_path):
    """Download a specific image from a parquet file by reading the image column."""
    url = GRAID_PARQUET_URL.format(parquet_idx)

    try:
        fs = fsspec.filesystem('https')
        with fs.open(url, 'rb') as f:
            pf = pq.ParquetFile(f)
            # Read the image column for the specific row group
            table = pf.read_row_group(row_group, columns=['image', 'source_id'])
            source_ids = table.column('source_id').to_pylist()
            images = table.column('image').to_pylist()

            for sid, img_data in zip(source_ids, images):
                if target_path.endswith(sid):
                    # img_data is a dict with 'bytes' key
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        img_bytes = img_data['bytes']
                    elif isinstance(img_data, (bytes, bytearray)):
                        img_bytes = img_data
                    else:
                        print(f"  Unexpected image data type: {type(img_data)}")
                        continue

                    # Create directory structure
                    # sid is like "samples/CAM_BACK/n015-...jpg"
                    # target is /Users/gatilin/MyWork/datasets/nuimages/samples/CAM_BACK/n015-...jpg
                    full_path = os.path.join(NUIMAGES_TARGET, sid)
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    with open(full_path, 'wb') as out:
                        out.write(img_bytes)

                    return True

    except Exception as e:
        print(f"  Error downloading from parquet {parquet_idx}: {e}")

    return False


def download_nuimages():
    """Main function to download nuImages."""
    print("=" * 60)
    print("Downloading nuImages from kd7/graid-nuimages (HuggingFace)")
    print("=" * 60)

    needed = load_needed_images(NUIMAGES_JSONL)
    print(f"Need {len(needed)} unique nuimages files")

    # Check which files we already have
    already_have = set()
    for sid in needed:
        full_path = os.path.join(NUIMAGES_TARGET, sid)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
            already_have.add(sid)
    print(f"Already downloaded: {len(already_have)}")
    still_needed = needed - already_have
    print(f"Still need: {len(still_needed)}")

    if not still_needed:
        print("All nuImages already downloaded!")
        return

    progress = load_progress()

    # Phase 1: Scan parquet files to find matches
    found_matches = {}
    # Restore previously found matches
    for sid, info in progress.get('found', {}).items():
        if sid in still_needed:
            found_matches[sid] = info

    scanned = set(progress.get('scanned_parquets', []))

    print(f"\nPhase 1: Scanning {NUM_PARQUETS} parquet files for source_id matches...")
    print(f"Already scanned: {len(scanned)} parquets, found {len(found_matches)} matches so far")

    for pidx in range(NUM_PARQUETS):
        if pidx in scanned:
            continue

        # Check if we've found all needed files
        if len(found_matches) >= len(still_needed):
            # Still need to verify all are found
            missing = still_needed - set(found_matches.keys())
            if not missing:
                print(f"\nAll {len(still_needed)} matches found!")
                break

        matches = scan_parquet_for_matches(pidx, still_needed)
        if matches:
            found_matches.update(matches)
            print(f"  Parquet {pidx:03d}: Found {len(matches)} new matches (total: {len(found_matches)}/{len(still_needed)})")

        scanned.add(pidx)
        progress['scanned_parquets'] = list(scanned)
        progress['found'] = found_matches
        save_progress(progress)

        if (pidx + 1) % 10 == 0:
            print(f"  Scanned {pidx+1}/{NUM_PARQUETS} parquets, found {len(found_matches)} matches so far...")

    print(f"\nPhase 1 complete: Found {len(found_matches)} matches out of {len(still_needed)} needed")

    # Check for missing files
    missing = still_needed - set(found_matches.keys())
    if missing:
        print(f"\nWARNING: {len(missing)} files not found in any parquet:")
        for m in sorted(missing)[:5]:
            print(f"  {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    # Phase 2: Download images for found matches
    print(f"\nPhase 2: Downloading {len(found_matches)} images...")

    # Group by parquet for efficient downloading
    by_parquet = defaultdict(list)
    for sid, info in found_matches.items():
        by_parquet[info['parquet_idx']].append((sid, info))

    downloaded = set(progress.get('downloaded', []))
    count = 0

    for pidx in sorted(by_parquet.keys()):
        items = by_parquet[pidx]
        print(f"\n  Parquet {pidx:03d}: {len(items)} images to download")

        # Group by row group within this parquet
        by_rg = defaultdict(list)
        for sid, info in items:
            by_rg[info['row_group']].append((sid, info))

        for rg_idx in sorted(by_rg.keys()):
            rg_items = by_rg[rg_idx]
            # Download all images in this row group at once
            url = GRAID_PARQUET_URL.format(pidx)
            try:
                fs = fsspec.filesystem('https')
                with fs.open(url, 'rb') as f:
                    pf = pq.ParquetFile(f)
                    table = pf.read_row_group(rg_idx, columns=['image', 'source_id'])
                    source_ids = table.column('source_id').to_pylist()
                    images = table.column('image').to_pylist()

                    for sid, img_data in zip(source_ids, images):
                        if sid in [s for s, _ in rg_items]:
                            # Extract image bytes
                            if isinstance(img_data, dict) and 'bytes' in img_data:
                                img_bytes = img_data['bytes']
                            elif isinstance(img_data, (bytes, bytearray)):
                                img_bytes = img_data
                            else:
                                continue

                            # Save image
                            full_path = os.path.join(NUIMAGES_TARGET, sid)
                            os.makedirs(os.path.dirname(full_path), exist_ok=True)
                            with open(full_path, 'wb') as out:
                                out.write(img_bytes)

                            downloaded.add(sid)
                            count += 1
                            if count % 10 == 0:
                                print(f"    Downloaded {count}/{len(found_matches)}...")
                                progress['downloaded'] = list(downloaded)
                                save_progress(progress)

            except Exception as e:
                print(f"    Error downloading from parquet {pidx}, row group {rg_idx}: {e}")

        progress['downloaded'] = list(downloaded)
        save_progress(progress)

    print(f"\nnuImages download complete: {len(downloaded)}/{len(still_needed)} images downloaded")

    # Final verification
    actual_files = 0
    for sid in needed:
        full_path = os.path.join(NUIMAGES_TARGET, sid)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
            actual_files += 1
    print(f"Verified: {actual_files}/{len(needed)} files exist on disk")


def try_egoobjects_sources():
    """Try multiple sources for EgoObjects images."""
    print("\n" + "=" * 60)
    print("Searching for EgoObjects images download source")
    print("=" * 60)

    # List of URLs to try
    urls_to_try = [
        # Meta AI / FAIR public files
        "https://dl.fbaipublicfiles.com/EgoObjects/EgoObjectsV1_images.zip",
        "https://dl.fbaipublicfiles.com/egoobjects/EgoObjectsV1_images.zip",
        "https://dl.fbaipublicfiles.com/EgoObjects/v1/EgoObjectsV1_images.zip",
        # CVDL / detectron2 patterns
        "https://dl.fbaipublicfiles.com/detectron2/EgoObjects/EgoObjectsV1_images.zip",
        # Possible S3 patterns
        "https://s3.amazonaws.com/egoobjects/EgoObjectsV1_images.zip",
        "https://egoobjects.s3.amazonaws.com/EgoObjectsV1_images.zip",
        # CVDL pattern
        "https://cvdl.s3.amazonaws.com/EgoObjects/EgoObjectsV1_images.zip",
    ]

    for url in urls_to_try:
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=15) as resp:
                status = resp.status
                size = resp.headers.get('Content-Length', 'unknown')
                print(f"  [OK] {url} -> status={status}, size={size}")
                return url
        except urllib.error.HTTPError as e:
            print(f"  [{e.code}] {url}")
        except Exception as e:
            print(f"  [ERR] {url} -> {e}")

    return None


def download_egoobjects():
    """Download EgoObjects images."""
    needed = load_needed_images(EGOOBJECTS_JSONL)
    print(f"Need {len(needed)} unique EgoObjects images")

    # Check which we already have
    already_have = set()
    for img_path in needed:
        # img_path is like "images/EF2C6FFB...jpg"
        filename = os.path.basename(img_path)
        full_path = os.path.join(EGOOBJECTS_TARGET, filename)
        if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
            already_have.add(img_path)
    print(f"Already downloaded: {len(already_have)}")
    still_needed = needed - already_have
    print(f"Still need: {len(still_needed)}")

    if not still_needed:
        print("All EgoObjects already downloaded!")
        return

    # Try to find a download source
    download_url = try_egoobjects_sources()

    if download_url:
        print(f"\nFound download URL: {download_url}")
        print(f"Downloading EgoObjectsV1_images.zip (~40GB)...")
        zip_path = "/tmp/EgoObjectsV1_images.zip"
        os.system(f"wget -c -O {zip_path} '{download_url}'")

        # Extract only needed files
        print(f"Extracting {len(still_needed)} needed images from zip...")
        os.makedirs(EGOOBJECTS_TARGET, exist_ok=True)

        needed_filenames = set()
        for img_path in still_needed:
            needed_filenames.add(os.path.basename(img_path))

        # Use Python zipfile to extract only needed files
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for info in zf.infolist():
                fname = os.path.basename(info.filename)
                if fname in needed_filenames:
                    target = os.path.join(EGOOBJECTS_TARGET, fname)
                    with zf.open(info) as src, open(target, 'wb') as dst:
                        dst.write(src.read())
                    print(f"  Extracted: {fname}")

        # Clean up zip
        os.remove(zip_path)
    else:
        print("\nERROR: Could not find a downloadable source for EgoObjects images.")
        print("The official download page (ai.meta.com/datasets/egoobjects-downloads)")
        print("requires a web browser to access.")
        print("\nAlternative options:")
        print("1. Download EgoObjectsV1_images.zip manually from:")
        print("   https://ai.meta.com/datasets/egoobjects-downloads/")
        print("2. Place it at /Users/gatilin/MyWork/datasets/EgoObjects/EgoObjectsV1_images.zip")
        print("3. Re-run this script to extract only needed images.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['nuimages', 'egoobjects', 'both'], default='both')
    args = parser.parse_args()

    if args.dataset in ('nuimages', 'both'):
        download_nuimages()

    if args.dataset in ('egoobjects', 'both'):
        download_egoobjects()
