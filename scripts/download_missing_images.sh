#!/usr/bin/env bash
#===============================================================================
# SNVC-50M Missing Image Download Script
# Run on macOS (local machine with full network access)
#
# Usage:
#   chmod +x download_missing_images.sh
#   ./download_missing_images.sh                  # download all datasets
#   ./download_missing_images.sh hypersim          # download specific dataset only
#   ./download_missing_images.sh hypersim loveda   # download multiple
#
# Prerequisites:
#   - Python 3.8+ with: pip install huggingface_hub Pillow requests
#   - wget, curl, unzip, tar (pre-installed on macOS)
#   - ~50GB free disk space
#   - Optional: set HF_TOKEN env var for faster HF downloads
#
# This script is safe to re-run: it skips already-downloaded files.
#===============================================================================

set -euo pipefail

DATASETS_ROOT="/Users/gatilin/MyWork/datasets"
SNVC50M_DIR="$DATASETS_ROOT/snvc50m"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/download_missing_images.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
ok()  { echo -e "${GREEN}[OK]${NC} $1" | tee -a "$LOG_FILE"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"; }
err() { echo -e "${RED}[ERR]${NC} $1" | tee -a "$LOG_FILE"; }
info(){ echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"; }

# All datasets that can be downloaded
ALL_DATASETS=(
    hypersim
    loveda
    grandf
    object365
    streethazards
    imagenet
    entityv2
    ttpla
    aptv2
    idd
    nuimages
    egoobjects
    scannetpp
    scannetv2
    udd
    pidray
    51world
)

# Determine which datasets to process
if [ "$#" -gt 0 ]; then
    DATASETS_TO_RUN=("$@")
else
    DATASETS_TO_RUN=("${ALL_DATASETS[@]}")
fi

info "SNVC-50M Missing Image Downloader"
info "Datasets root: $DATASETS_ROOT"
info "Datasets to process: ${DATASETS_TO_RUN[*]}"
info "Log file: $LOG_FILE"
echo ""

#===============================================================================
# 1. HYPERSIM - Download 163 missing scenes from HF omrastogi/Hypersim-Processed
#    Source: https://huggingface.co/datasets/omrastogi/Hypersim-Processed
#    Files: train/<scene>/rgb_cam_XX_frYYYY.png -> convert to JPG tonemap
#===============================================================================
download_hypersim() {
    info "=== HYPERSIM ==="
    info "Downloading 163 missing scenes from omrastogi/Hypersim-Processed (HF)"

    local HF_REPO="omrastogi/Hypersim-Processed"
    local SCENES_DIR="$DATASETS_ROOT/hypersim/scenes_v2"
    local TMP_DIR="$SCRIPT_DIR/tmp_hypersim"

    mkdir -p "$TMP_DIR"

    python3 << 'PYEOF'
import json, os, sys
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
import io

HF_REPO = "omrastogi/Hypersim-Processed"
DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
SCENES_DIR = os.path.join(DATASETS_ROOT, "hypersim", "scenes_v2")
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")

# Find missing scenes
all_paths = set()
for f in ['hypersim_gcg_sampled.jsonl', 'hypersim_genseg.jsonl', 'hypersim_genseg_sampled.jsonl']:
    path = os.path.join(SNVC50M_DIR, f)
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                d = json.loads(line)
                all_paths.add(d['image'])

missing_scenes = {}
for p in all_paths:
    full = os.path.join(SCENES_DIR, p)
    if not os.path.exists(full):
        scene = p.split('/')[0]
        if scene not in missing_scenes:
            missing_scenes[scene] = []
        missing_scenes[scene].append(p)

print(f"Missing: {sum(len(v) for v in missing_scenes.values())} files in {len(missing_scenes)} scenes")

# List all files in the HF repo
print("Listing HF repo files...")
try:
    repo_files = list_repo_files(HF_REPO, repo_type="dataset")
    print(f"Repo has {len(repo_files)} files")
except Exception as e:
    print(f"Error listing repo: {e}")
    sys.exit(1)

# Build mapping: scene -> list of HF file paths
# HF files are like: train/ai_004_007/rgb_cam_01_fr0000.png
# We need: ai_004_007/images/scene_cam_01_final_preview/frame.0000.tonemap.jpg
hf_by_scene = {}
for rf in repo_files:
    if not rf.endswith('.png'):
        continue
    parts = rf.split('/')
    if len(parts) >= 3:
        split = parts[0]  # train/val/test
        scene = parts[1]
        filename = parts[2]
        if scene not in hf_by_scene:
            hf_by_scene[scene] = {'split': split, 'files': []}
        hf_by_scene[scene]['files'].append(filename)

# Download and convert for each missing scene
downloaded = 0
errors = 0
cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_hypersim")
os.makedirs(cache_dir, exist_ok=True)

for scene in sorted(missing_scenes.keys()):
    if scene not in hf_by_scene:
        print(f"  {scene}: NOT in HF repo, skipping")
        errors += len(missing_scenes[scene])
        continue

    split = hf_by_scene[scene]['split']
    hf_files = hf_by_scene[scene]['files']

    # Map HF filenames to expected paths
    # HF: rgb_cam_01_fr0000.png
    # Expected: ai_004_007/images/scene_cam_01_final_preview/frame.0000.tonemap.jpg
    needed = {}
    for expected_path in missing_scenes[scene]:
        # Parse expected path: scene/images/cam_dir/frame.XXXX.tonemap.jpg
        parts = expected_path.split('/')
        cam_dir = parts[2]  # scene_cam_01_final_preview
        frame_file = parts[3]  # frame.0000.tonemap.jpg
        frame_num = frame_file.split('.')[1]  # 0000
        cam_num = cam_dir.split('_')[2]  # 01

        hf_filename = f"rgb_cam_{cam_num}_fr{frame_num}.png"
        hf_path = f"{split}/{scene}/{hf_filename}"
        needed[hf_path] = expected_path

    scene_ok = 0
    for hf_path, expected_path in needed.items():
        local_full = os.path.join(SCENES_DIR, expected_path)
        if os.path.exists(local_full):
            scene_ok += 1
            continue

        try:
            # Download PNG
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=hf_path,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            # Convert PNG to JPG
            img = Image.open(downloaded_path)
            img = img.convert('RGB')
            os.makedirs(os.path.dirname(local_full), exist_ok=True)
            img.save(local_full, 'JPEG', quality=95)
            scene_ok += 1
            downloaded += 1
        except Exception as e:
            print(f"    Error: {hf_path}: {e}")
            errors += 1

    print(f"  {scene}: {scene_ok}/{len(needed)} downloaded")

print(f"\nHypersim complete: {downloaded} downloaded, {errors} errors")
PYEOF
    ok "Hypersim download complete"
}

#===============================================================================
# 2. LOVEDA - Download missing Train/Rural images 718-999
#    Source: Christie510/rs_seg_benchmarks_LoveDA (HF) or original LoveDA GitHub
#===============================================================================
download_loveda() {
    info "=== LOVEDA ==="
    info "Downloading missing LoveDA Train/Rural images (718-999)"

    local TARGET_DIR="$DATASETS_ROOT/LoveDA/urban_rural_train_images"
    mkdir -p "$TARGET_DIR"

    python3 << 'PYEOF'
import os, sys
from huggingface_hub import hf_hub_download
import zipfile, shutil

HF_REPO = "Christie510/rs_seg_benchmarks_LoveDA"
TARGET_DIR = "/Users/gatilin/MyWork/datasets/LoveDA/urban_rural_train_images"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_loveda")
os.makedirs(CACHE_DIR, exist_ok=True)

# Find which file numbers are missing
missing_nums = []
for i in range(718, 1000):
    path = os.path.join(TARGET_DIR, f"{i}.png")
    if not os.path.exists(path):
        missing_nums.append(i)

print(f"Missing LoveDA files: {len(missing_nums)} (range {missing_nums[0]}-{missing_nums[-1]})")

# Method 1: Try downloading the full LoveDA.zip from Christie510
print("Downloading LoveDA.zip from HF (Christie510/rs_seg_benchmarks_LoveDA)...")
try:
    zip_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="LoveDA.zip",
        repo_type="dataset",
        cache_dir=CACHE_DIR,
    )
    print(f"Downloaded to: {zip_path}")

    # Extract only the Train/Rural images we need
    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_names = zf.namelist()

        # Find Train/Rural image paths
        # LoveDA.zip structure: LoveDA/Train/Rural/images_png/XXX.png
        extracted = 0
        for num in missing_nums:
            patterns = [
                f"LoveDA/Train/Rural/images_png/{num}.png",
                f"Train/Rural/images_png/{num}.png",
                f"LoveDA/Train/Rural/images_png/{num:04d}.png",
            ]
            for pattern in patterns:
                if pattern in all_names:
                    data = zf.read(pattern)
                    out_path = os.path.join(TARGET_DIR, f"{num}.png")
                    with open(out_path, 'wb') as f:
                        f.write(data)
                    extracted += 1
                    break

        print(f"Extracted {extracted}/{len(missing_nums)} missing files")

except Exception as e:
    print(f"HF download failed: {e}")
    print("Trying alternative: direct GitHub download...")

    # Method 2: Try Junjue-Wang/LoveDA GitHub release
    import subprocess
    # The LoveDA dataset is hosted on OneDrive/Google Drive
    # Try the GitHub release page
    try:
        result = subprocess.run([
            "wget", "-q", "--show-progress",
            "https://github.com/Junjue-Wang/LoveDA/raw/main/Train/Rural/images_png.zip",
            "-O", os.path.join(CACHE_DIR, "rural_images.zip")
        ], timeout=300)
        if result.returncode == 0:
            with zipfile.ZipFile(os.path.join(CACHE_DIR, "rural_images.zip"), 'r') as zf:
                for num in missing_nums:
                    try:
                        data = zf.read(f"images_png/{num}.png")
                        with open(os.path.join(TARGET_DIR, f"{num}.png"), 'wb') as f:
                            f.write(data)
                    except KeyError:
                        pass
            print("GitHub download successful")
    except Exception as e2:
        print(f"GitHub download also failed: {e2}")
        print("LoveDA download failed. Please download manually from:")
        print("  https://github.com/Junjue-Wang/LoveDA")

# Verify
still_missing = sum(1 for i in range(718, 1000)
                    if not os.path.exists(os.path.join(TARGET_DIR, f"{i}.png")))
print(f"Still missing: {still_missing} files")
PYEOF
    ok "LoveDA download complete"
}

#===============================================================================
# 3. GRANDF - Download train images from MBZUAI/GranD-f (HF)
#    Source: https://huggingface.co/datasets/MBZUAI/GranD-f
#===============================================================================
download_grandf() {
    info "=== GRANDF ==="
    info "Downloading GranDf train images"

    local TARGET_DIR="$DATASETS_ROOT/grandf"
    mkdir -p "$TARGET_DIR"

    python3 << 'PYEOF'
import os, sys, json
from huggingface_hub import hf_hub_download, list_repo_files

HF_REPO = "MBZUAI/GranD-f"
TARGET_DIR = "/Users/gatilin/MyWork/datasets/grandf"
SNVC50M_DIR = "/Users/gatilin/MyWork/datasets/snvc50m"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_grandf")
os.makedirs(CACHE_DIR, exist_ok=True)

# Get expected sa_ IDs
expected_files = set()
with open(os.path.join(SNVC50M_DIR, "grandf_gcg_sampled.jsonl")) as f:
    for line in f:
        d = json.loads(f.readline())
        fname = os.path.basename(d['image'])
        expected_files.add(fname)

# Also check which we already have
train_dir = os.path.join(TARGET_DIR, "train")
os.makedirs(train_dir, exist_ok=True)
existing = set(os.listdir(train_dir)) if os.path.exists(train_dir) else set()
missing = expected_files - existing
print(f"GranDf: {len(missing)} missing train images")

# List HF repo files
print(f"Listing {HF_REPO} files...")
try:
    repo_files = list_repo_files(HF_REPO, repo_type="dataset")
    print(f"Repo has {len(repo_files)} files")

    # Look for image files or archives
    image_files = [f for f in repo_files if f.endswith(('.jpg', '.png', '.zip', '.tar', '.tar.gz'))]
    print(f"Image/archive files: {len(image_files)}")

    # Check for train images archive
    train_archives = [f for f in image_files if 'train' in f.lower() and 'image' in f.lower()]
    print(f"Train image archives: {train_archives[:5]}")

    if train_archives:
        # Download and extract
        for archive in train_archives:
            print(f"Downloading {archive}...")
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=archive,
                repo_type="dataset",
                cache_dir=CACHE_DIR,
            )
            print(f"Downloaded to: {path}")

            # Extract
            import zipfile, tarfile
            if archive.endswith('.zip'):
                with zipfile.ZipFile(path, 'r') as zf:
                    for name in zf.namelist():
                        if name.endswith('.jpg') and os.path.basename(name) in missing:
                            data = zf.read(name)
                            out_path = os.path.join(train_dir, os.path.basename(name))
                            with open(out_path, 'wb') as f:
                                f.write(data)
                            print(f"  Extracted: {os.path.basename(name)}")
            elif archive.endswith('.tar.gz') or archive.endswith('.tar'):
                with tarfile.open(path, 'r:*') as tf:
                    for member in tf.getmembers():
                        if member.name.endswith('.jpg') and os.path.basename(member.name) in missing:
                            f_obj = tf.extractfile(member)
                            if f_obj:
                                data = f_obj.read()
                                out_path = os.path.join(train_dir, os.path.basename(member.name))
                                with open(out_path, 'wb') as f:
                                    f.write(data)
    else:
        # Try downloading individual images
        print("No train archive found. Trying individual files...")
        # Check if images are stored as individual files
        for sa_file in sorted(missing)[:5]:
            matches = [f for f in repo_files if sa_file in f]
            print(f"  {sa_file}: {len(matches)} matches")

except Exception as e:
    print(f"Error: {e}")
    print("GranDf download failed. The images may need to be downloaded manually.")
    print("Try: https://huggingface.co/datasets/MBZUAI/GranD-f")
PYEOF
    ok "GranDf download complete"
}

#===============================================================================
# 4. OBJECT365 - Download missing individual images
#    Source: Original Objects365 v2 from their website or HF mirror
#===============================================================================
download_object365() {
    info "=== OBJECT365 ==="
    info "Object365: 1881 individual images missing across 40 patches"
    info "These are specific images not included in the downloaded patches."
    info "Attempting to download from Objects365 official source..."

    python3 << 'PYEOF'
import os, json

DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")
EXTRACTED_DIR = os.path.join(DATASETS_ROOT, "Objects365", "extracted")

# Get missing files
missing = []
with open(os.path.join(SNVC50M_DIR, "object365_total_unified_depth_edit_regenerated_sampled.jsonl")) as f:
    for line in f:
        d = json.loads(f.readline())
        img = d['image'][0] if isinstance(d['image'], list) else d['image']
        full = os.path.join(EXTRACTED_DIR, img)
        if not os.path.exists(full):
            missing.append(img)

print(f"Object365: {len(missing)} missing images")

# Objects365 images can be downloaded from:
# 1. https://www.objects365.org/ (requires registration)
# 2. HF: Object365 data might be available as a mirror

# Write the missing list for manual download
missing_list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "object365_missing_files.txt")
with open(missing_list_path, 'w') as f:
    for m in sorted(missing):
        f.write(m + '\n')

print(f"Missing files list written to: {missing_list_path}")
print("\nObject365 images cannot be auto-downloaded (requires registration at objects365.org).")
print("Options:")
print("  1. Download full patches from https://www.objects365.org/download.html")
print("  2. Check HF mirrors: search 'objects365' on huggingface.co")
print("  3. The missing 1881 images are spread across 40 patches (~47 per patch avg)")
print("     Most patches already have 5000-24000 images, so the missing rate is <1%")
PYEOF
    warn "Object365: 1881 images need manual download. See object365_missing_files.txt"
}

#===============================================================================
# 5. STREETAZARDS - Download from StreetHazards official source
#    Source: https://github.com/rolandgassler/streethazards or HF
#===============================================================================
download_streethazards() {
    info "=== STREETAZARDS ==="
    info "Downloading missing StreetHazards images"

    python3 << 'PYEOF'
import os, json, sys
from huggingface_hub import list_repo_files, hf_hub_download

DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")
SH_DIR = os.path.join(DATASETS_ROOT, "streethazards_hf_v2", "extracted_v2", "train", "images")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_streethazards")
os.makedirs(CACHE_DIR, exist_ok=True)

# Get missing files
missing = set()
for jsonl in ['streethazards_gcg_sampled.jsonl', 'streethazards_genseg_sampled.jsonl']:
    with open(os.path.join(SNVC50M_DIR, jsonl)) as f:
        for line in f:
            d = json.loads(f.readline())
            p = d['image']
            full = os.path.join(SH_DIR, p)
            if not os.path.exists(full):
                missing.add(p)

print(f"StreetHazards: {len(missing)} missing files")

# Try HF datasets
HF_CANDIDATES = [
    "hpa/StreetHazards",
    "RolandGassler/streethazards",
    "datasets/streethazards",
]

for repo in HF_CANDIDATES:
    try:
        files = list_repo_files(repo, repo_type="dataset")
        print(f"  Found repo: {repo} with {len(files)} files")
        # Download relevant images
        png_files = [f for f in files if f.endswith('.png') and ('training' in f or 'validation' in f)]
        print(f"  PNG files: {len(png_files)}")

        if png_files:
            for hf_path in png_files:
                # Check if this file is needed
                fname = os.path.basename(hf_path)
                # Check if any missing path matches
                needed = any(os.path.basename(m) == fname for m in missing)
                if not needed:
                    continue

                downloaded = hf_hub_download(
                    repo_id=repo,
                    filename=hf_path,
                    repo_type="dataset",
                    cache_dir=CACHE_DIR,
                )

                # Find the target path
                for m in missing:
                    if os.path.basename(m) == fname:
                        target = os.path.join(SH_DIR, m)
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        import shutil
                        shutil.copy2(downloaded, target)
                        print(f"  Downloaded: {fname}")
                        break
            break
    except Exception as e:
        print(f"  {repo}: {e}")

# Try direct GitHub download
print("\nTrying GitHub sources...")
import subprocess
urls = [
    "https://raw.githubusercontent.com/rolandgassler/StreetHazards/master/train/images/training/t1-3/",
    "https://raw.githubusercontent.com/hendrydong/Street-Hazards/main/train/images/training/t1-3/",
]

# Write missing list for manual download
missing_list = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streethazards_missing_files.txt")
with open(missing_list, 'w') as f:
    for m in sorted(missing):
        f.write(m + '\n')
print(f"\nMissing files list: {missing_list}")
print("If auto-download fails, get the data from:")
print("  https://github.com/rolandgassler/StreetHazards")
PYEOF
    warn "StreetHazards: some files may need manual download"
}

#===============================================================================
# 6. IMAGENET - Download missing class images
#    Source: image-net.org (requires registration) or HF: ImageNetV2
#===============================================================================
download_imagenet() {
    info "=== IMAGENET ==="
    info "Downloading 365 missing ImageNet images"

    python3 << 'PYEOF'
import os, json, sys
from huggingface_hub import hf_hub_download

DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")
IMAGENET_DIR = os.path.join(DATASETS_ROOT, "ImageNet")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_imagenet")
os.makedirs(CACHE_DIR, exist_ok=True)

# Get missing files
missing = []
with open(os.path.join(SNVC50M_DIR, "imagenetpart_point_sampled.jsonl")) as f:
    for line in f:
        d = json.loads(f.readline())
        p = d['image']
        full = os.path.join(IMAGENET_DIR, p)
        if not os.path.exists(full):
            missing.append(p)

print(f"ImageNet: {len(missing)} missing images")

# Group by class (wnid)
from collections import defaultdict
by_class = defaultdict(list)
for p in missing:
    parts = p.split('/')
    wnid = parts[1]  # train/n02132136/n02132136_10837.JPEG
    fname = parts[2]
    by_class[wnid].append(fname)

print(f"Across {len(by_class)} classes")

# Try downloading from image-net.org (requires no auth for individual images via wget)
# URL format: https://www.image-net.org/data/winter21_whole/n02132136.tar
# Or individual: https://image-net.org/data/winter21_whole/n02132136/n02132136_10837.JPEG

import urllib.request
import ssl

# Disable SSL verification for image-net.org (sometimes has cert issues)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

downloaded = 0
errors = 0

for wnid in sorted(by_class.keys()):
    fnames = by_class[wnid]
    for fname in fnames:
        target_path = os.path.join(IMAGENET_DIR, 'train', wnid, fname)
        if os.path.exists(target_path):
            continue

        # Try downloading from image-net.org
        urls = [
            f"https://image-net.org/data/winter21_whole/{wnid}/{fname}",
            f"https://image-net.org/data/ILSVRC/2012/orig/train/{wnid}/{fname}",
        ]

        success = False
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                resp = urllib.request.urlopen(req, timeout=15, context=ctx)
                data = resp.read()
                if len(data) > 1000:  # Valid image should be > 1KB
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, 'wb') as f:
                        f.write(data)
                    downloaded += 1
                    success = True
                    break
            except Exception as e:
                continue

        if not success:
            errors += 1

    print(f"  {wnid}: {len(fnames)} needed, downloaded so far: {downloaded}")

print(f"\nImageNet: {downloaded} downloaded, {errors} failed")
if errors > 0:
    print("For failed downloads, try:")
    print("  1. Register at https://www.image-net.org/")
    print("  2. Download full class tarballs")
PYEOF
    ok "ImageNet download complete"
}

#===============================================================================
# 7. ENTITYV2 - Download missing images from various sources
#===============================================================================
download_entityv2() {
    info "=== ENTITYV2 ==="
    info "Downloading 369 missing EntityV2 images"

    python3 << 'PYEOF'
import os, json
from huggingface_hub import hf_hub_download, list_repo_files

DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")
ENTITY_DIR = os.path.join(DATASETS_ROOT, "entityv2")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_entityv2")
os.makedirs(CACHE_DIR, exist_ok=True)

# Get missing files
missing = []
with open(os.path.join(SNVC50M_DIR, "entityv2_gcg_sampled.jsonl")) as f:
    for line in f:
        d = json.loads(f.readline())
        p = d['image']
        full = os.path.join(ENTITY_DIR, p)
        if not os.path.exists(full):
            missing.append(p)

print(f"EntityV2: {len(missing)} missing images")

# EntityV2 images come from: COCO, OpenImages, Unsplash, Flickr2k, Pixabay
# Try HF repo
HF_REPOS = ["dylank444/entityv2", "entityv2/dataset"]

for repo in HF_REPOS:
    try:
        files = list_repo_files(repo, repo_type="dataset")
        print(f"Found HF repo: {repo} with {len(files)} files")
        # Download matching files
        for m in missing:
            fname = os.path.basename(m)
            matches = [f for f in files if f.endswith(fname)]
            if matches:
                path = hf_hub_download(repo_id=repo, filename=matches[0],
                                       repo_type="dataset", cache_dir=CACHE_DIR)
                import shutil
                target = os.path.join(ENTITY_DIR, m)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(path, target)
        break
    except Exception as e:
        print(f"  {repo}: not found")

# Write missing list for manual download
missing_list = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entityv2_missing_files.txt")
with open(missing_list, 'w') as f:
    for m in sorted(missing):
        f.write(m + '\n')
print(f"\nMissing files list: {missing_list}")
print("EntityV2 images come from multiple sources (COCO, OpenImages, Unsplash, etc.)")
print("If auto-download fails, check: https://github.com/zyong812/EntityV2")
PYEOF
    warn "EntityV2: may need manual download"
}

#===============================================================================
# 8. TTPLA - Download missing original-size images
#    Source: https://github.com/He tara/TTPLA
#===============================================================================
download_ttpla() {
    info "=== TTPLA ==="
    info "Downloading 260 missing TTPLA original-size images"

    python3 << 'PYEOF'
import os, json, subprocess

DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")
TTPLA_DIR = os.path.join(DATASETS_ROOT, "TTPLA", "data_original_size_v1", "data_original_size")

# Get missing files
missing = []
with open(os.path.join(SNVC50M_DIR, "ttpla_genseg_sampled.jsonl")) as f:
    for line in f:
        d = json.loads(f.readline())
        p = d['image']
        fname = os.path.basename(p)
        full = os.path.join(TTPLA_DIR, fname)
        if not os.path.exists(full):
            missing.append(fname)

print(f"TTPLA: {len(missing)} missing original images")

# TTPLA original images are available at:
# https://github.com/r3md3n/ttpla_dataset
# Or: https://github.com/He-Tara/TTPLA
GITHUB_BASE = "https://raw.githubusercontent.com/r3md3n/ttpla_dataset/main/data_original_size"

import urllib.request
downloaded = 0
errors = 0

for fname in sorted(missing):
    target = os.path.join(TTPLA_DIR, fname)
    if os.path.exists(target):
        continue

    urls = [
        f"{GITHUB_BASE}/{fname}",
        f"https://raw.githubusercontent.com/He-Tara/TTPLA/master/data_original_size/{fname}",
    ]

    for url in urls:
        try:
            urllib.request.urlretrieve(url, target)
            if os.path.getsize(target) > 1000:
                downloaded += 1
                break
        except Exception:
            if os.path.exists(target):
                os.remove(target)
            continue
    else:
        errors += 1

print(f"TTPLA: {downloaded} downloaded, {errors} failed")
if errors > 0:
    print("For manual download: https://github.com/r3md3n/ttpla_dataset")
PYEOF
    ok "TTPLA download complete"
}

#===============================================================================
# 9. APTV2 - Download missing video frames
#===============================================================================
download_aptv2() {
    info "=== APTV2 ==="
    info "Downloading 88 missing APTv2 images"

    python3 << 'PYEOF'
import os, json, urllib.request

DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")

# Get missing files
missing = []
with open(os.path.join(SNVC50M_DIR, "aptv2_point_sampled.jsonl")) as f:
    for line in f:
        d = json.loads(f.readline())
        p = d['image']
        full = os.path.join(DATASETS_ROOT, p)
        if not os.path.exists(full):
            missing.append(p)

print(f"APTv2: {len(missing)} missing images")

# APTv2 is available at: https://github.com/yfZhang0429/APTv2
# Try GitHub raw download
GITHUB_BASE = "https://raw.githubusercontent.com/yfZhang0429/APTv2/main"

downloaded = 0
errors = 0

for p in sorted(missing):
    target = os.path.join(DATASETS_ROOT, p)
    if os.path.exists(target):
        continue

    url = f"{GITHUB_BASE}/{p}"
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        urllib.request.urlretrieve(url, target)
        if os.path.getsize(target) > 1000:
            downloaded += 1
        else:
            os.remove(target)
            errors += 1
    except Exception:
        if os.path.exists(target):
            os.remove(target)
        errors += 1

print(f"APTv2: {downloaded} downloaded, {errors} failed")
if errors > 0:
    print("For manual download: https://github.com/yfZhang0429/APTv2")
PYEOF
    ok "APTv2 download complete"
}

#===============================================================================
# 10. IDD - Download train split
#     Source: https://idd.insaan.iiit.ac.in/
#===============================================================================
download_idd() {
    info "=== IDD ==="
    info "Downloading IDD train data"

    python3 << 'PYEOF'
import os, json, subprocess

DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
IDD_DIR = os.path.join(DATASETS_ROOT, "IDD")

# IDD has multiple sub-datasets:
# 1. idd2_genseg / IDD-1: needs leftImg8bit/train/ (segmentation images)
# 2. iddav3: needs T02_CN_M/RGB/ (different IDD version - IDD-Fork)

# Method 1: Download IDD Segmentation train data from official site
print("=== IDD Segmentation (idd2, IDD-1) ===")
print("The local idd-segmentation.tar.gz only contains test/ split.")
print("Downloading train/ split from official source...")

# Try downloading from IIIT Hyderabad
# The IDD segmentation data is at: https://idd.insaan.iiit.ac.in/dataset/download/

# Try HF mirror
from huggingface_hub import list_repo_files, hf_hub_download

HF_CANDIDATES = [
    "segments/idd-segmentation",
    "idd_dataset/segmentation",
]

for repo in HF_CANDIDATES:
    try:
        files = list_repo_files(repo, repo_type="dataset")
        print(f"Found: {repo} with {len(files)} files")
        train_files = [f for f in files if 'train' in f.lower() and f.endswith('.png')]
        print(f"Train PNG files: {len(train_files)}")
        if train_files:
            # Download train data
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_idd")
            os.makedirs(cache_dir, exist_ok=True)
            for tf in train_files[:5]:
                print(f"  Sample: {tf}")
            break
    except Exception as e:
        print(f"  {repo}: not found")

print("\n=== IDD-Fork v3 (iddav3) ===")
print("iddav3 uses T02_CN_M/RGB/ format from IDD-Fork v3")
print("This is a different version from standard IDD.")
print("Check: https://github.com/VainF/IDD-Fork")

# Write missing lists
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")
for jsonl in ['idd2_genseg_sampled.jsonl', 'iddav3_gcg_sampled.jsonl', 'iddav3_genseg_sampled.jsonl',
              'IDD-1_total_unified_genseg_edit_regenerated_sampled.jsonl']:
    path = os.path.join(SNVC50M_DIR, jsonl)
    if not os.path.exists(path):
        continue
    missing = []
    with open(path) as f:
        for line in f:
            d = json.loads(f.readline())
            p = d['image']
            # Apply path fixes
            if p.startswith('leftImg8bit/'):
                fixed = os.path.join(IDD_DIR, 'extracted', 'IDD_Segmentation',
                                     p.replace('.jpg', '.png'))
            else:
                fixed = os.path.join(IDD_DIR, p)
            if not os.path.exists(fixed):
                missing.append(p)

    if missing:
        list_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"idd_missing_{jsonl.replace('.jsonl','')}.txt")
        with open(list_file, 'w') as f:
            for m in missing:
                f.write(m + '\n')
        print(f"\n{jsonl}: {len(missing)} missing -> {list_file}")

print("\nIDD download instructions:")
print("  1. Segmentation train: https://idd.insaan.iiit.ac.in/dataset/download/")
print("     Register and download leftImg8bit_train.zip")
print("  2. IDD-Fork v3: https://github.com/VainF/IDD-Fork")
PYEOF
    warn "IDD: needs manual download from official site"
}

#===============================================================================
# 11-17. Datasets with no auto-download source
#===============================================================================
download_nuimages() {
    info "=== NUIMAGES ==="
    info "nuimages: 1000 samples at 0%"
    info "Source: https://www.nuscenes.org/download (requires registration)"
    info "Download 'nuimages-v1.0-all' and extract to $DATASETS_ROOT/nuimages/"
    warn "nuimages requires manual download from nuscenes.org"
}

download_egoobjects() {
    info "=== EGOOBJECTS ==="
    info "EgoObjects: 1000 samples at 0% (only annotations downloaded)"
    info "Source: https://github.com/EGO4D/ego-objects"
    info "Need to download image tarballs (not annotation tarballs)"
    python3 << 'PYEOF'
import os
DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
EGO_DIR = os.path.join(DATASETS_ROOT, "EgoObjects")
# List existing tar files
tars = [f for f in os.listdir(EGO_DIR) if f.endswith('.tar')]
print(f"Existing tar files: {len(tars)}")
print("These contain JSON annotations, NOT images.")
print("Image tarballs are named like: egoobjects_part1_XXXX_images.tar")
print("Download from: https://github.com/EGO4D/ego-objects/releases")
PYEOF
    warn "EgoObjects: need image tarballs (not annotation tarballs)"
}

download_scannetpp() {
    info "=== SCANNET++ ==="
    info "ScanNet++: 1500 samples at 0%"
    info "Source: https://kaldir.vc.in.tum.de/scannetpp/ (requires registration)"
    warn "ScanNet++ requires manual download (academic license)"
}

download_scannetv2() {
    info "=== SCANNET v2 ==="
    info "ScanNet v2: 500 samples at 0%"
    info "Source: http://www.scan-net.org/ (requires registration)"
    warn "ScanNet v2 requires manual download (academic license)"
}

download_udd() {
    info "=== UDD5/UDD6 ==="
    info "UDD5/UDD6: 301 samples at 0%"
    info "Source: https://xinyu-yi.github.io/Underwater-Image-Dataset/"
    python3 << 'PYEOF'
import urllib.request, os
# Try downloading UDD6 from GitHub
urls = {
    "udd5": "https://github.com/qbgeobotics/UDD6/releases/download/v1.0/udd5_train.zip",
    "udd6": "https://github.com/qbgeobotics/UDD6/releases/download/v1.0/udd6_train.zip",
}
# Also try the official site
print("UDD Dataset download:")
print("  Official: https://xinyu-yi.github.io/Underwater-Image-Dataset/")
print("  Try: https://github.com/danielgordon10/UDD6")
PYEOF
    warn "UDD5/UDD6: try GitHub or official site"
}

download_pidray() {
    info "=== PIDRAY ==="
    info "PIDray: 500 samples at 0%"
    info "Source: https://github.com/bywang2019/PIDray"
    info "Images are on Baidu Netdisk (code: 8vz6)"
    info "Alternative: ask authors for direct link"
    python3 << 'PYEOF'
import os
# Try downloading from GitHub (annotations only, but check)
print("PIDray images are hosted on Baidu Netdisk.")
print("If you have access to Baidu Netdisk:")
print("  URL: https://pan.baidu.com/s/1qPUNWYItg08pht0kFn8qxg")
print("  Code: 8vz6")
print("")
print("Alternative: try HF lutao2021/PIDray")
print("  The train split (76913 images) may be available as a dataset")
PYEOF
    warn "PIDray: Baidu Netdisk or HF lutao2021/PIDray"
}

download_51world() {
    info "=== 51WORLD ==="
    info "51world: 1500 samples at 0% (wrong data format)"
    info "JSONL expects scene/image/ format but downloaded data has camera_front/ format"
    info "Need to re-download or convert the data format"
    python3 << 'PYEOF'
import os, json
DATASETS_ROOT = "/Users/gatilin/MyWork/datasets"
SNVC50M_DIR = os.path.join(DATASETS_ROOT, "snvc50m")

# Check what format JSONL expects
with open(os.path.join(SNVC50M_DIR, "51world_gcg_sampled.jsonl")) as f:
    d = json.loads(f.readline())
    print(f"Expected path: {d['image']}")

print("The 51world data format mismatch:")
print("  Expected: 10-vehicle/image/006287.png")
print("  Have:     image/camera_front/17004.jpg")
print("")
print("This requires either:")
print("  1. Download the correct 51world format from the original source")
print("  2. Convert existing camera data to the expected scene format")
print("  Source: https://www.51world.com/products/51world-data-platform.html")
PYEOF
    warn "51world: data format mismatch, needs format conversion or re-download"
}

#===============================================================================
# MAIN
#===============================================================================
echo ""
info "Starting download for ${#DATASETS_TO_RUN[@]} datasets..."
echo ""

FAILED=()
for ds in "${DATASETS_TO_RUN[@]}"; do
    echo ""
    log "Processing: $ds"
    case "$ds" in
        hypersim)      download_hypersim ;;
        loveda)        download_loveda ;;
        grandf)        download_grandf ;;
        object365)     download_object365 ;;
        streethazards) download_streethazards ;;
        imagenet)      download_imagenet ;;
        entityv2)      download_entityv2 ;;
        ttpla)         download_ttpla ;;
        aptv2)         download_aptv2 ;;
        idd)           download_idd ;;
        nuimages)      download_nuimages ;;
        egoobjects)    download_egoobjects ;;
        scannetpp)     download_scannetpp ;;
        scannetv2)     download_scannetv2 ;;
        udd)           download_udd ;;
        pidray)        download_pidray ;;
        51world)       download_51world ;;
        *)
            err "Unknown dataset: $ds"
            err "Available: ${ALL_DATASETS[*]}"
            exit 1
            ;;
    esac
done

echo ""
info "==============================="
info "Download Summary"
info "==============================="
echo ""
echo "Auto-downloadable datasets (with HF/GitHub access):"
echo "  hypersim, loveda, grandf, streethazards, imagenet, entityv2, ttpla, aptv2"
echo ""
echo "Manual download required:"
echo "  object365   -> https://www.objects365.org/download.html"
echo "  idd         -> https://idd.insaan.iiit.ac.in/dataset/download/"
echo "  nuimages    -> https://www.nuscenes.org/download"
echo "  egoobjects  -> https://github.com/EGO4D/ego-objects/releases"
echo "  scannetpp   -> https://kaldir.vc.in.tum.de/scannetpp/"
echo "  scannetv2   -> http://www.scan-net.org/"
echo "  udd         -> https://xinyu-yi.github.io/Underwater-Image-Dataset/"
echo "  pidray      -> Baidu Netdisk (code: 8vz6)"
echo "  51world     -> format mismatch, needs conversion"
echo ""
echo "After downloads complete, re-run availability check:"
echo "  cd /Users/gatilin/PycharmProjects/open-vision-banana-git-v260722"
echo "  python3 scripts/check_v3_availability.py"
echo ""
ok "Script finished. Check $LOG_FILE for details."
