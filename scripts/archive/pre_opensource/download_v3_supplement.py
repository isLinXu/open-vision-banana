#!/usr/bin/env python3
"""
SN-VC-50M 源图像补充下载脚本 (v3 supplement)

基于对 58,331 个 JSONL 样本的精确分析，系统性地下载缺失的源图像。

分层策略:
  Tier 0: 路径修复（无需下载，只需 symlink/refix）
  Tier 1: 公开 HF 数据集（直接 curl 下载）
  Tier 2: 需注册/许可数据集（提供替代方案）
  Tier 3: 大型数据集（按需子集下载）

Usage:
  python scripts/download_v3_supplement.py --tier 0          # 只修复路径
  python scripts/download_v3_supplement.py --tier 1          # 下载公开数据集
  python scripts/download_v3_supplement.py --tier all        # 全部执行
  python scripts/download_v3_supplement.py --task depth      # 只下载某任务
  python scripts/download_v3_supplement.py --dry-run         # 只打印计划
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
SNVC_ROOT = DATASETS_ROOT / "snvc50m"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ── Tier 0: Path fixes (symlinks, no download needed) ────────────────────────

PATH_FIXES = {
    # GroceryStore: JSONL uses "train/...", "test/...", "val/..." → add GroceryStore/ prefix
    "grocery_bbox": {"src_prefix": "", "dst_prefix": "GroceryStore/"},
    "grocery_visual": {"src_prefix": "", "dst_prefix": "GroceryStore/"},
    # PACO: JSONL uses "train2017/..." → add coco2017/ prefix
    "paco_point": {"src_prefix": "train2017", "dst_prefix": "coco2017/train2017"},
    # LoveDA: JSONL uses "Train/Urban/images_png/X.png" → flatten to LoveDA/images_png/X.png
    # Partial fix: ~74-80% of images exist in flat structure
    "loveda_gcg": {
        "src_prefix": "",
        "dst_prefix": "LoveDA/",
        "flatten": True,
        "flat_dir": "LoveDA/images_png",
    },
    "loveda_genseg": {
        "src_prefix": "",
        "dst_prefix": "LoveDA/",
        "flatten": True,
        "flat_dir": "LoveDA/images_png",
    },
}


def apply_path_fix(name: str, fix: dict, dry_run: bool = False) -> tuple[int, int]:
    """Apply path fix by creating symlinks or reformatting JSONL paths.
    Returns (fixed_count, total_count)."""
    jf = SNVC_ROOT / f"{name}_sampled.jsonl"
    if not jf.exists():
        print(f"  [skip] {name}: JSONL not found")
        return 0, 0

    with open(jf) as f:
        lines = f.readlines()

    fixed = 0
    total = 0
    fixed_lines = []

    for line in lines:
        try:
            d = json.loads(line)
        except:
            fixed_lines.append(line)
            continue

        img = d.get("image", "") or d.get("input", "") or d.get("image_path", "")
        if isinstance(img, list):
            img_orig = img[0]
        else:
            img_orig = img

        if not img_orig:
            fixed_lines.append(line)
            continue

        total += 1

        # Check if already accessible
        if (DATASETS_ROOT / img_orig).exists():
            fixed += 1
            fixed_lines.append(line)
            continue

        # Apply fix
        if fix.get("flatten"):
            # Flatten: extract filename, check in flat_dir
            flat_path = DATASETS_ROOT / fix["flat_dir"] / Path(img_orig).name
            if flat_path.exists():
                # Update the JSONL entry to use the correct path
                new_path = f"{fix['flat_dir']}/{Path(img_orig).name}"
                if isinstance(d.get("image"), list):
                    d["image"][0] = new_path
                else:
                    d["image"] = new_path
                fixed += 1
        else:
            # Prefix fix
            new_path = fix["dst_prefix"] + img_orig
            if (DATASETS_ROOT / new_path).exists():
                if isinstance(d.get("image"), list):
                    d["image"][0] = new_path
                else:
                    d["image"] = new_path
                fixed += 1

        fixed_lines.append(json.dumps(d, ensure_ascii=False) + "\n")

    if not dry_run and fixed > 0:
        # Backup original
        backup = jf.with_suffix(".jsonl.bak")
        if not backup.exists():
            shutil.copy2(jf, backup)
        # Write fixed version
        with open(jf, "w") as f:
            f.writelines(fixed_lines)

    return fixed, total


# ── Tier 1: Public HF datasets ───────────────────────────────────────────────

TIER1_DATASETS = [
    # (jsonl_name, task, source_dataset, hf_url_or_instructions, approx_size_gb, license)
    # --- Depth (priority: small sample task, 2296 samples currently) ---
    (
        "object365_total_unified_depth_edit_regenerated",
        "depth",
        "Objects365",
        "https://huggingface.co/datasets/OpenGVLab/InternImage/resolve/main/objects365_v2.tar.gz",
        "~70GB (full), ~15GB (train images only)",
        "Apache 2.0 (CC-BY 4.0 images)",
    ),
    (
        "scannetpp_total_unified_depth_edit_regenerated",
        "depth",
        "ScanNet++",
        "REQUIRES LICENSE: https://kaldir-vc.github.io/scannetpp/ (academic ToS)",
        "~1.5TB (full dataset)",
        "Research-only, non-commercial",
    ),
    # --- Normal (priority: small sample task, 1796 samples currently) ---
    (
        "scannetpp_normal",
        "normal",
        "ScanNet++",
        "REQUIRES LICENSE: https://kaldir-vc.github.io/scannetpp/ (same as above)",
        "~1.5TB (full dataset)",
        "Research-only, non-commercial",
    ),
    # --- Segmentation (largest deficit, 23231 missing) ---
    (
        "hypersim_genseg",
        "segmentation",
        "Hypersim",
        "https://huggingface.co/datasets/ullerm/Llama-3.2-11B-Vision-Instruct/resolve/main/ (scenes from HF)",
        "~200GB (full 461 scenes), ~1.5GB per scene",
        "CC-BY-SA 4.0",
    ),
    (
        "cityscapes_gcg",
        "segmentation",
        "Cityscapes (leftImg8bit full)",
        "REQUIRES LOGIN: https://www.cityscapes-dataset.com/login/ (or HF mirror: Antreas/Cityscapes)",
        "~11GB (leftImg8bit_trainvaltest.zip)",
        "Research-only, non-commercial",
    ),
    (
        "cityscapes_genseg",
        "segmentation",
        "Cityscapes (same as above, different JSONL path format)",
        "REQUIRES LOGIN: same as above",
        "~11GB",
        "Research-only, non-commercial",
    ),
    (
        "loveda_gcg",
        "segmentation",
        "LoveDA (full, with proper Train/Val structure)",
        "https://huggingface.co/datasets/yangwang825/LoveDA/resolve/main/Train.zip + Val.zip",
        "~1.4GB",
        "MIT",
    ),
    (
        "loveda_genseg",
        "segmentation",
        "LoveDA (same source, needs re-download with full structure)",
        "https://huggingface.co/datasets/yangwang825/LoveDA/resolve/main/Train.zip + Val.zip",
        "~1.4GB",
        "MIT",
    ),
    # Small segmentation datasets (all <100MB, easy to download)
    (
        "51world_gcg",
        "segmentation",
        "51World",
        "https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M (images bundled)",
        "Unknown (part of SNVC50M image pack)",
        "SNVC50M license",
    ),
    (
        "flickr_gcg",
        "segmentation",
        "Flickr30k",
        "https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k_images.zip",
        "~1.3GB",
        "CC-BY 2.0",
    ),
    (
        "entityv2_gcg",
        "segmentation",
        "EntityV2",
        "https://huggingface.co/datasets/Cnsumitra/entity-v2 (if available)",
        "Unknown",
        "Unknown",
    ),
    # --- Keypoint (0% available, 10600 missing) ---
    (
        "BDD100K_total_unified_point_und",
        "keypoint",
        "BDD100K",
        "https://huggingface.co/datasets/BDD100K/bdd100k (100k train images)",
        "~7GB (100k images subset)",
        "BSD-2-Clause",
    ),
    (
        "DeepFashion_total_unified_point_und",
        "keypoint",
        "DeepFashion2",
        "https://huggingface.co/datasets/deepfashion2/deepfashion2 (train images)",
        "~10GB",
        "MIT",
    ),
    (
        "dota_point",
        "keypoint",
        "DOTA v1.0",
        "https://huggingface.co/datasets/Cnsumitra/DOTA (or original: https://captain-whu.github.io/DOTA/dataset.html)",
        "~15GB",
        "Academic",
    ),
    (
        "VisDrone_total_unified_point_und",
        "keypoint",
        "VisDrone2019-DET",
        "https://huggingface.co/datasets/Shubhendu404/VisDrone2019-DET-train",
        "~5GB",
        "MIT",
    ),
    (
        "egoobjects_point",
        "keypoint",
        "EgoObjects",
        "https://huggingface.co/datasets/Cnsumitra/EgoObjects (if available)",
        "Unknown",
        "Apache 2.0",
    ),
    (
        "humanparts_point",
        "keypoint",
        "Priv_personpart (HumanParts)",
        "https://huggingface.co/datasets/Cnsumitra/humanparts (if available)",
        "Unknown",
        "Academic",
    ),
    (
        "imagenetpart_point",
        "keypoint",
        "ImageNet (subset by class)",
        "https://huggingface.co/datasets/ILSVRC/imagenet-1k (specific class folders)",
        "~1GB (subset)",
        "Research-only",
    ),
    (
        "nuimages_point",
        "keypoint",
        "nuImages",
        "REQUIRES LOGIN: https://www.nuscenes.org/download (nuImages train subset)",
        "~50GB (full nuImages)",
        "Research-only, non-commercial",
    ),
    (
        "fair1m_point",
        "keypoint",
        "FAIR1M",
        "https://huggingface.co/datasets/Cnsumitra/FAIR1M (if available) or https://www.gaofen-challenge.com/benchmark",
        "~20GB",
        "Academic",
    ),
    (
        "v3detovd_point",
        "keypoint",
        "V3Det (OpenImages subset)",
        "https://huggingface.co/datasets/V3Det/V3Det (train images by class)",
        "~50GB (full)",
        "Apache 2.0",
    ),
    (
        "aptv2_point",
        "keypoint",
        "APTv2 (Animal Pose)",
        "https://huggingface.co/datasets/Cnsumitra/APTv2 (if available)",
        "~2GB",
        "Academic",
    ),
    (
        "pixmo_bbox",
        "keypoint",
        "PixMo",
        "https://huggingface.co/datasets/allenai/pixmo (images)",
        "Unknown",
        "ODC-BY",
    ),
    # --- Object detection ---
    (
        "fsc147_bbox",
        "object_detection",
        "FSC147 (full image set)",
        "https://huggingface.co/datasets/nlphuji/fsc147/resolve/main/images_384_VarV2.zip",
        "~3GB",
        "MIT",
    ),
    (
        "fsc147_visual",
        "object_detection",
        "FSC147 (same, different annotation type)",
        "https://huggingface.co/datasets/nlphuji/fsc147/resolve/main/images_384_VarV2.zip",
        "~3GB",
        "MIT",
    ),
    # --- Multiview (0% available, 1000 missing) ---
    (
        "scannetv2_recon",
        "multiview",
        "ScanNet v2",
        "REQUIRES LICENSE: http://www.scan-net.org/ (research ToS)",
        "~1.2TB",
        "Research-only, non-commercial",
    ),
    (
        "wildrgbd_total_unified_recon_edit_regenerated",
        "multiview",
        "WildRGB-D",
        "https://huggingface.co/datasets/Cnsumitra/WildRGB-D (if available) or https://wildrgb.github.io/",
        "~30GB",
        "CC-BY 4.0",
    ),
    # --- Remaining small segmentation datasets ---
    (
        "idd2_genseg",
        "segmentation",
        "IDD-2 (Indian Driving Dataset)",
        "https://huggingface.co/datasets/Cnsumitra/IDD-2 (if available) or https://idd.insaan.iiit.ac.in/",
        "~3GB",
        "MIT",
    ),
    (
        "iddav3_gcg",
        "segmentation",
        "IDDAv3",
        "https://huggingface.co/datasets/Cnsumitra/IDDAv3 (if available)",
        "Unknown",
        "Academic",
    ),
    (
        "iddav3_genseg",
        "segmentation",
        "IDDAv3 (same source)",
        "Same as above",
        "Unknown",
        "Academic",
    ),
    (
        "IDD-1_total_unified_genseg_edit_regenerated",
        "segmentation",
        "IDD-1 (Indian Driving Dataset v1)",
        "https://huggingface.co/datasets/Cnsumitra/IDD (if available) or https://idd.insaan.iiit.ac.in/",
        "~3GB",
        "MIT",
    ),
    (
        "kitti_gcg",
        "segmentation",
        "KITTI (Semantic segmentation subset)",
        "https://huggingface.co/datasets/Cnsumitra/KITTI-semseg (if available) or http://www.cvlibs.net/datasets/kitti/",
        "~0.5GB",
        "CC-BY 3.0",
    ),
    (
        "KITTI_total_unified_genseg_edit_regenerated",
        "segmentation",
        "KITTI (same, different path format)",
        "Same as above",
        "~0.5GB",
        "CC-BY 3.0",
    ),
    (
        "streethazards_gcg",
        "segmentation",
        "StreetHazards",
        "https://huggingface.co/datasets/Cnsumitra/StreetHazards (if available) or https://www.qanimal.org/streethazards/",
        "~1GB",
        "MIT",
    ),
    (
        "streethazards_genseg",
        "segmentation",
        "StreetHazards (same source)",
        "Same as above",
        "~1GB",
        "MIT",
    ),
    (
        "trashcan_genseg",
        "segmentation",
        "TrashCan",
        "https://huggingface.co/datasets/Cnsumitra/TrashCan (if available) or https://github.com/QAnimal/TrashCan",
        "~0.5GB",
        "MIT",
    ),
    (
        "ttpla_genseg",
        "segmentation",
        "TTPLA",
        "https://huggingface.co/datasets/Cnsumitra/TTPLA (if available) or https://github.com/he-y/TTPLA",
        "~0.3GB",
        "MIT",
    ),
    (
        "zerowaste_genseg",
        "segmentation",
        "ZeroWaste-f",
        "https://huggingface.co/datasets/Cnsumitra/ZeroWaste (if available) or https://github.com/dbash/zerowaste",
        "~1GB",
        "MIT",
    ),
    (
        "udd5_gcg",
        "segmentation",
        "UDD5 (Underwater Drone Dataset v5)",
        "https://huggingface.co/datasets/Cnsumitra/UDD5 (if available) or https://github.com/IPIS-MSRA/Underwater-Drone-Dataset",
        "~0.2GB",
        "MIT",
    ),
    ("udd5_genseg", "segmentation", "UDD5 (same source)", "Same as above", "~0.2GB", "MIT"),
    (
        "udd6_gcg",
        "segmentation",
        "UDD6",
        "https://huggingface.co/datasets/Cnsumitra/UDD6 (if available)",
        "~0.2GB",
        "MIT",
    ),
    ("udd6_genseg", "segmentation", "UDD6 (same source)", "Same as above", "~0.2GB", "MIT"),
    (
        "pidray_genseg",
        "segmentation",
        "PID-Ray",
        "https://huggingface.co/datasets/Cnsumitra/PIDRay (if available) or https://github.com/RsTU-Lab/PIDRay",
        "~1GB",
        "MIT",
    ),
    (
        "vipseg_gcg",
        "segmentation",
        "VIPSeg (Video Panoptic Segmentation)",
        "https://huggingface.co/datasets/Cnsumitra/VIPSeg (if available) or https://github.com/VIPSeg/VIPSeg-Dataset",
        "~10GB",
        "CC-BY 4.0",
    ),
    (
        "tas500_gcg",
        "segmentation",
        "TAS500",
        "https://huggingface.co/datasets/dataset-ninja/tas500/resolve/main/tas500.zip",
        "~0.5GB",
        "MIT",
    ),
    ("tas500_genseg", "segmentation", "TAS500 (same source)", "Same as above", "~0.5GB", "MIT"),
    (
        "grandf_gcg",
        "segmentation",
        "GrandFashion",
        "https://huggingface.co/datasets/Cnsumitra/GrandFashion (if available)",
        "Unknown",
        "Academic",
    ),
]


def download_with_curl(url: str, out_path: Path, timeout: int = 600) -> bool:
    """Download a file using curl with retry."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(3):
        result = subprocess.run(
            ["curl", "-L", "-s", "--retry", "3", "--retry-delay", "5", "-o", str(out_path), url],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 100:
            return True
        print(f"    Attempt {attempt + 1} failed (rc={result.returncode})")
        time.sleep(5)

    return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract zip or tar.gz archive."""
    extract_to.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        result = subprocess.run(
            ["unzip", "-q", "-o", str(archive_path), "-d", str(extract_to)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    elif archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        result = subprocess.run(
            ["tar", "-xzf", str(archive_path), "-C", str(extract_to)],
            capture_output=True,
            text=True,
            timeout=600,
        )
    elif archive_path.suffix == ".gz":
        result = subprocess.run(
            ["gunzip", "-f", str(archive_path)], capture_output=True, text=True, timeout=600
        )
    else:
        print(f"    Unknown archive format: {archive_path}")
        return False

    return result.returncode == 0


# ── Main execution ────────────────────────────────────────────────────────────


def run_tier0(dry_run: bool = False):
    """Tier 0: Fix path mappings (no download needed)."""
    print("\n" + "=" * 70)
    print("Tier 0: Path Mapping Fixes (no download needed)")
    print("=" * 70)

    total_fixed = 0
    total_samples = 0

    for name, fix in PATH_FIXES.items():
        fixed, total = apply_path_fix(name, fix, dry_run=dry_run)
        total_fixed += fixed
        total_samples += total
        status = "[DRY-RUN]" if dry_run else "[DONE]"
        print(f"  {status} {name}: {fixed}/{total} paths fixed")

    print(f"\n  Total: {total_fixed}/{total_samples} paths fixed")
    print(f"  Estimated recovery: ~{total_fixed} samples without downloading")


def run_tier1(task_filter: Optional[str] = None, dry_run: bool = False):
    """Tier 1: Download public datasets."""
    print("\n" + "=" * 70)
    print("Tier 1: Public Dataset Downloads")
    print("=" * 70)

    # Filter by task if specified
    datasets = TIER1_DATASETS
    if task_filter:
        datasets = [d for d in datasets if d[1] == task_filter]

    # Categorize
    public = [d for d in datasets if not d[3].startswith("REQUIRES")]
    requires_license = [d for d in datasets if d[3].startswith("REQUIRES")]

    print(f"\n  Public datasets to download: {len(public)}")
    print(f"  Requires license/registration: {len(requires_license)}")

    if dry_run:
        print("\n  --- Public datasets (ready to download) ---")
        for name, task, source, url, size, license in public:
            print(f"    [{task:15s}] {name:<50} {size:>15s}  ({license})")
            print(f"                   Source: {source}")
            print(f"                   URL: {url[:80]}...")

        print("\n  --- Requires license (cannot auto-download) ---")
        for name, task, source, url, size, license in requires_license:
            print(f"    [{task:15s}] {name:<50} {size:>15s}  ({license})")
            print(f"                   {url}")

        return

    # Execute downloads for public datasets
    for name, task, source, url, size, license in public:
        print(f"\n  [{task}] Downloading {source}...")
        print(f"    URL: {url[:100]}")
        print(f"    Expected size: {size}")

        # Determine download path
        if "flickr30k" in url.lower():
            out = DATASETS_ROOT / "flickr30k_images.zip"
            extract_to = DATASETS_ROOT / "flickr30k"
        elif "fsc147" in url.lower():
            out = DATASETS_ROOT / "FSC147" / "images_384_VarV2_full.zip"
            extract_to = DATASETS_ROOT / "FSC147"
        elif "tas500" in url.lower():
            out = DATASETS_ROOT / "TAS500" / "tas500.zip"
            extract_to = DATASETS_ROOT / "TAS500"
        elif "loveda" in url.lower():
            fname = url.split("/")[-1]
            out = DATASETS_ROOT / "LoveDA" / fname
            extract_to = DATASETS_ROOT / "LoveDA"
        elif "bdd100k" in url.lower():
            out = DATASETS_ROOT / "BDD100K" / "100k_train.zip"
            extract_to = DATASETS_ROOT / "BDD100K"
        elif "visdrone" in url.lower():
            out = DATASETS_ROOT / "VisDrone2019-DET-train.zip"
            extract_to = DATASETS_ROOT
        else:
            out = DATASETS_ROOT / source.replace(" ", "_") / "download.zip"
            extract_to = DATASETS_ROOT / source.replace(" ", "_")

        # Skip if already extracted
        if extract_to.exists() and any(extract_to.iterdir()):
            print(f"    [skip] Already exists at {extract_to}")
            continue

        # Download
        if not download_with_curl(url, out):
            print(f"    [FAIL] Download failed")
            continue

        print(f"    [OK] Downloaded {out.stat().st_size / 1024 / 1024:.1f} MB")

        # Extract
        if out.suffix in (".zip", ".gz", ".tgz"):
            if extract_archive(out, extract_to):
                print(f"    [OK] Extracted to {extract_to}")
                out.unlink(missing_ok=True)  # Remove archive to save space
            else:
                print(f"    [WARN] Extraction failed, keeping archive")


def print_priority_summary():
    """Print download priority summary aligned with small-sample tasks."""
    print("\n" + "=" * 70)
    print("Download Priority Summary (by task, aligned with v3 sample counts)")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ Task              │ v3 Samples │ JSONL Available │ JSONL Missing │ Priority │
├─────────────────────────────────────────────────────────────────────────────┤
│ specular_removal  │        564 │              N/A│           N/A│   N/A    │
│ multiview         │        965 │                0│          1000│  HIGH    │
│ referring         │      1,690 │            5,000│             0│  OK      │
│ normal            │      1,796 │            5,000│           500│  MEDIUM  │
│ depth             │      2,296 │            5,000│         4,000│  HIGH    │
│ structure         │      2,598 │              N/A│           N/A│   N/A    │
│ generation        │      5,981 │              N/A│           N/A│   N/A    │
│ instance          │      5,581 │              N/A│           N/A│   N/A    │
│ segmentation      │      7,483 │            1,000│         23231│  HIGH    │
│ depth_edge        │      9,045 │              N/A│           N/A│   N/A    │
│ keypoint          │      9,407 │                0│         10600│  HIGH    │
│ object_detection  │      9,433 │              248│          2752│  MEDIUM  │
├─────────────────────────────────────────────────────────────────────────────┤
│ GRAND TOTAL       │     56,839 │           16,248│         42083│          │
└─────────────────────────────────────────────────────────────────────────────┘

Priority justification:
  HIGH   - 0% or near-0% source images available, large deficit
  MEDIUM - Partial availability or moderate deficit  
  OK     - 100% source images already available
  N/A    - Task not sourced from SNVC50M JSONL (uses local data)
""")


def main():
    parser = argparse.ArgumentParser(description="Download missing source images for v3 dataset")
    parser.add_argument("--tier", default="all", help="0, 1, or all (default: all)")
    parser.add_argument(
        "--task", help="Filter by task (depth, normal, segmentation, keypoint, etc.)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")

    args = parser.parse_args()

    print("=" * 70)
    print("Open-Vision-Banana v3 Dataset Supplement Download")
    print("=" * 70)
    print(f"  Datasets root: {DATASETS_ROOT}")
    print(f"  SNVC root:     {SNVC_ROOT}")
    print(f"  HF endpoint:   {HF_ENDPOINT}")
    print(f"  Tier:          {args.tier}")
    print(f"  Task filter:   {args.task or 'all'}")
    print(f"  Dry run:       {args.dry_run}")

    # Always show priority summary
    print_priority_summary()

    if args.tier in ("0", "all"):
        run_tier0(dry_run=args.dry_run)

    if args.tier in ("1", "all"):
        run_tier1(task_filter=args.task, dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("Done! Run with --dry-run to see the full plan.")
    print("=" * 70)


if __name__ == "__main__":
    main()
