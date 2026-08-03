#!/usr/bin/env python3
"""
Comprehensive check of SNVC50M JSONL source image availability.
Scans all JSONL files, checks which image paths exist in the datasets root,
and reports per-task and per-dataset availability statistics.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

DATASETS_ROOT = Path("/Users/gatilin/MyWork/datasets")
JSONL_DIR = DATASETS_ROOT / "snvc50m"

# Task family mapping based on JSONL filename patterns
TASK_MAP = {
    "depth": ["coco_depth", "object365_total_unified_depth", "scannetpp_total_unified_depth"],
    "normal": ["coco_normal", "scannetpp_normal"],
    "segmentation": [
        "loveda_gcg",
        "loveda_genseg",
        "cityscapes_gcg",
        "cityscapes_genseg",
        "tas500_gcg",
        "tas500_genseg",
        "idd2_genseg",
        "iddav3_gcg",
        "iddav3_genseg",
        "IDD-1_total_unified_genseg",
        "PSG_total_unified_gcg",
        "trashcan_genseg",
        "ttpla_genseg",
        "udd5_gcg",
        "udd5_genseg",
        "udd6_gcg",
        "udd6_genseg",
        "streethazards_gcg",
        "streethazards_genseg",
        "zerowaste_genseg",
        "hypersim_gcg",
        "hypersim_genseg",
        "flickr_gcg",
        "vipseg_gcg",
        "51world_gcg",
        "51world_genseg",
        "entityv2_gcg",
        "grandf_gcg",
        "KITTI_total_unified_genseg",
        "pidray_genseg",
        "pixmo_bbox",
    ],
    "keypoint": [
        "BDD100K_total_unified_point",
        "VisDrone_total_unified_point",
        "DeepFashion_total_unified_point",
        "dota_point",
        "fair1m_point",
        "humanparts_point",
        "paco_point",
        "egoobjects_point",
        "aptv2_point",
        "nuimages_point",
        "imagenetpart_point",
        "v3detovd_point",
        "grocery_bbox",
        "grocery_visual",
        "fsc147_bbox",
        "fsc147_visual",
    ],
    "multiview": ["wildrgbd_total_unified_recon", "scannetv2_recon"],
}

# Path prefix fixes (Tier 0 + Tier 1)
PATH_FIXES = {
    "grocery_bbox": lambda p: f"GroceryStore/{p}" if not p.startswith("GroceryStore/") else p,
    "grocery_visual": lambda p: f"GroceryStore/{p}" if not p.startswith("GroceryStore/") else p,
    "paco_point": lambda p: f"coco2017/{p}" if not p.startswith("coco2017/") else p,
    "loveda_gcg": lambda p: (
        p.replace("LoveDA/Train/", "LoveDA/images_png/")
        .replace("LoveDA/Val/", "LoveDA/images_png/")
        .replace("Train/Rural/images_png/", "LoveDA/urban_rural_train_images/")
        .replace("Train/Urban/images_png/", "LoveDA/urban_rural_train_images/")
        .replace("Val/Rural/images_png/", "LoveDA/urban_rural_val_images/")
        .replace("Val/Urban/images_png/", "LoveDA/urban_rural_val_images/")
        if p.startswith(("LoveDA/", "Train/", "Val/"))
        else f"LoveDA/images_png/{p.split('/')[-1]}"
    ),
    "loveda_genseg": lambda p: (
        p.replace("LoveDA/Train/", "LoveDA/images_png/")
        .replace("LoveDA/Val/", "LoveDA/images_png/")
        .replace("Train/Rural/images_png/", "LoveDA/urban_rural_train_images/")
        .replace("Train/Urban/images_png/", "LoveDA/urban_rural_train_images/")
        .replace("Val/Rural/images_png/", "LoveDA/urban_rural_val_images/")
        .replace("Val/Urban/images_png/", "LoveDA/urban_rural_val_images/")
        if p.startswith(("LoveDA/", "Train/", "Val/"))
        else f"LoveDA/images_png/{p.split('/')[-1]}"
    ),
    "object365_total_unified_depth_edit_regenerated": lambda p: (
        f"Objects365/extracted/{p}" if not p.startswith("Objects365/") else p
    ),
    "dota_point": lambda p: f"DOTA/{p}" if not p.startswith("DOTA/") else p,
    "BDD100K_total_unified_point_und": lambda p: (
        p.replace("BDD100K/100k/", "BDD100K/bdd100k/images/100k/")
        if p.startswith("BDD100K/100k/")
        else p
    ),
    "VisDrone_total_unified_point_und": lambda p: p,  # Already has VisDrone2019-DET-train/ prefix
    # Also check VisDrone2019-DET (without -train)
    # Will be handled by fallback below
    "fsc147_bbox": lambda p: p,  # Already has FSC147/ prefix
    "fsc147_visual": lambda p: p,  # Already has FSC147/ prefix
    "flickr_gcg": lambda p: (
        p.replace("flickr30k/", "flickr30k/") if p.startswith("flickr30k/") else p
    ),
    "vipseg_gcg": lambda p: f"VIPSeg/VIPSeg/imgs/{p}" if not p.startswith("VIPSeg/") else p,
    "DeepFashion_total_unified_point_und": lambda p: (
        f"DeepFashion2/{p}" if not p.startswith("DeepFashion2/") else p
    ),
    "wildrgbd_total_unified_recon_edit_regenerated": lambda p: (
        f"WildRGB-D/{p.replace('.png', '.jpg')}"
    ),  # HF repo has .jpg not .png, needs WildRGB-D/ prefix
    "scannetv2_recon": lambda p: f"scannetv2/{p}" if not p.startswith("scannetv2/") else p,
    "cityscapes_gcg": lambda p: f"Cityscapes/{p}" if not p.startswith("Cityscapes/") else p,
    "cityscapes_genseg": lambda p: (
        f"Cityscapes/leftImg8bit/train/{p}"
        if not p.startswith("Cityscapes/") and not p.startswith("leftImg8bit/")
        else (f"Cityscapes/{p}" if not p.startswith("Cityscapes/") else p)
    ),
    "grandf_gcg": lambda p: f"grandf/{p}" if not p.startswith("grandf/") else p,
    "ttpla_genseg": lambda p: f"TTPLA/{p}" if not p.startswith("TTPLA/") else p,
    "trashcan_genseg": lambda p: f"TrashCan/{p}" if not p.startswith("TrashCan/") else p,
    "tas500_gcg": lambda p: (
        f"TAS500/extracted/tas500v1.1/{p}" if not p.startswith("TAS500/") else p
    ),
    "tas500_genseg": lambda p: (
        f"TAS500/extracted/tas500v1.1/{p}" if not p.startswith("TAS500/") else p
    ),
    "zerowaste_genseg": lambda p: f"zerowaste/{p}" if not p.startswith("zerowaste/") else p,
    "streethazards_gcg": lambda p: (
        f"streethazards_hf_v2/extracted_v2/train/images/{p}"
        if p.startswith("training/") or p.startswith("validation/")
        else (f"streethazards_hf_v2/extracted_v2/test/images/{p}" if p.startswith("test/") else p)
    ),
    "streethazards_genseg": lambda p: (
        f"streethazards_hf_v2/extracted_v2/train/images/{p}"
        if p.startswith("training/") or p.startswith("validation/")
        else (f"streethazards_hf_v2/extracted_v2/test/images/{p}" if p.startswith("test/") else p)
    ),
    "egoobjects_point": lambda p: f"EgoObjects/{p}" if not p.startswith("EgoObjects/") else p,
    "aptv2_point": lambda p: p if p.startswith("APTv2/") else p,
    "idd2_genseg": lambda p: f"IDD/{p}" if not p.startswith("IDD/") else p,
    "iddav3_gcg": lambda p: f"IDD/{p}" if not p.startswith("IDD/") else p,
    "iddav3_genseg": lambda p: f"IDD/{p}" if not p.startswith("IDD/") else p,
    "IDD-1_total_unified_genseg_edit_regenerated": lambda p: (
        f"IDD/{p}" if not p.startswith("IDD/") else p
    ),
    "KITTI_total_unified_genseg_edit_regenerated": lambda p: (
        f"KITTI/tracking_extract/{p}" if not p.startswith("KITTI/") else p
    ),
    "kitti_gcg": lambda p: f"KITTI/tracking_extract/{p}" if not p.startswith("KITTI/") else p,
    "entityv2_gcg": lambda p: f"entityv2/{p}" if not p.startswith("entityv2/") else p,
    "pixmo_bbox": lambda p: f"pixmo/{p}" if not p.startswith("pixmo/") else p,
    "hypersim_gcg": lambda p: f"hypersim/scenes_v2/{p}" if not p.startswith("hypersim/") else p,
    "hypersim_genseg": lambda p: f"hypersim/scenes_v2/{p}" if not p.startswith("hypersim/") else p,
    "fair1m_point": lambda p: f"Fair1M/{p}" if not p.startswith("Fair1M/") else p,
    "v3detovd_point": lambda p: (
        f"V3Det/extracted/{p}"
        if not p.startswith("V3Det/")
        else (
            f"V3Det/extracted/{p[len('V3Det/') :].replace('extracted/', '', 1)}"
            if "extracted" not in p
            else p
        )
    ),
    "pidray_genseg": lambda p: f"pidray/{p}" if not p.startswith("pidray/") else p,
    "nuimages_point": lambda p: f"nuimages/{p}" if not p.startswith("nuimages/") else p,
    "imagenetpart_point": lambda p: f"ImageNet/{p}" if not p.startswith("ImageNet/") else p,
    "humanparts_point": lambda p: f"HumanParts/{p}" if not p.startswith("HumanParts/") else p,
}


def get_image_path_from_jsonl(entry: dict, jsonl_name: str) -> str | None:
    """Extract the source image path from a JSONL entry."""
    # Try common field names
    for field in ["image_path", "image", "source_image", "file_path", "path", "img_path"]:
        if field in entry:
            val = entry[field]
            # Handle list type - take first element
            if isinstance(val, list):
                if len(val) > 0:
                    val = val[0]
                else:
                    continue
            # Handle dict type
            if isinstance(val, dict):
                return val.get("path") or val.get("file_path") or val.get("filename")
            if isinstance(val, str):
                return val
    # Some entries have nested structures
    if "image" in entry and isinstance(entry["image"], dict):
        return entry["image"].get("path") or entry["image"].get("file_path")
    return None


def check_jsonl_availability(jsonl_path: Path) -> dict:
    """Check availability of images referenced in a JSONL file."""
    name = jsonl_path.stem
    total = 0
    available = 0
    missing_paths = []
    found_paths_sample = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            img_path = get_image_path_from_jsonl(entry, name)
            if not img_path:
                continue

            total += 1

            # Apply path fix if available
            base_name = name
            # Remove _sampled suffix for fix lookup
            fix_key = base_name.replace("_sampled", "")
            if fix_key in PATH_FIXES:
                img_path = PATH_FIXES[fix_key](img_path)

            # Check if file exists
            full_path = DATASETS_ROOT / img_path
            if full_path.exists():
                available += 1
                if len(found_paths_sample) < 3:
                    found_paths_sample.append(img_path)
            else:
                # For hypersim, check both scenes/ and scenes_v2/ directories
                if fix_key.startswith("hypersim"):
                    # Try scenes/ if current is scenes_v2/
                    alt_hypersim = img_path.replace("hypersim/scenes_v2/", "hypersim/scenes/", 1)
                    alt_path_hs = DATASETS_ROOT / alt_hypersim
                    if alt_path_hs.exists():
                        available += 1
                        if len(found_paths_sample) < 3:
                            found_paths_sample.append(alt_hypersim)
                        continue
                    # Try scenes_v2/ if current is scenes/
                    alt_hypersim2 = img_path.replace("hypersim/scenes/", "hypersim/scenes_v2/", 1)
                    alt_path_hs2 = DATASETS_ROOT / alt_hypersim2
                    if alt_path_hs2.exists():
                        available += 1
                        if len(found_paths_sample) < 3:
                            found_paths_sample.append(alt_hypersim2)
                        continue
                # Try alternate extensions (.png -> .jpg, .jpg -> .png)
                alt_path = None
                if img_path.endswith(".png"):
                    alt_path = DATASETS_ROOT / (img_path[:-4] + ".jpg")
                elif img_path.endswith(".jpg"):
                    alt_path = DATASETS_ROOT / (img_path[:-4] + ".png")
                elif img_path.endswith(".jpeg"):
                    alt_path = DATASETS_ROOT / (img_path[:-5] + ".jpg")
                if alt_path and alt_path.exists():
                    available += 1
                    if len(found_paths_sample) < 3:
                        found_paths_sample.append(img_path)
                else:
                    # Try VisDrone2019-DET fallback (without -train)
                    if img_path.startswith("VisDrone2019-DET-train/"):
                        alt_visdrone = DATASETS_ROOT / img_path.replace(
                            "VisDrone2019-DET-train/", "VisDrone2019-DET/"
                        )
                        if alt_visdrone.exists():
                            available += 1
                            if len(found_paths_sample) < 3:
                                found_paths_sample.append(
                                    str(alt_visdrone.relative_to(DATASETS_ROOT))
                                )
                            continue
                    # Try .PNG for ZeroWaste
                    if img_path.endswith(".PNG"):
                        alt_png = DATASETS_ROOT / (img_path[:-4] + ".png")
                        if alt_png.exists():
                            available += 1
                            continue
                    # Try .JPG for UDD
                    if img_path.endswith(".JPG"):
                        alt_jpg = DATASETS_ROOT / (img_path[:-4] + ".jpg")
                        if alt_jpg.exists():
                            available += 1
                            continue
                    if len(missing_paths) < 5:
                        missing_paths.append(img_path)

    return {
        "name": name,
        "total": total,
        "available": available,
        "missing": total - available,
        "availability_rate": f"{available / total * 100:.1f}%" if total > 0 else "N/A",
        "missing_samples": missing_paths,
        "found_samples": found_paths_sample,
    }


def main():
    print("=" * 80)
    print(f"SNVC50M JSONL Source Image Availability Check")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets root: {DATASETS_ROOT}")
    print(f"JSONL dir: {JSONL_DIR}")
    print("=" * 80)

    # Get all JSONL files (prefer _sampled versions, fall back to full)
    all_jsonls = sorted(JSONL_DIR.glob("*.jsonl"))

    # Deduplicate: prefer _sampled over full
    jsonl_map = {}
    for jf in all_jsonls:
        stem = jf.stem
        base = stem.replace("_sampled", "")
        if base not in jsonl_map:
            jsonl_map[base] = jf
        elif "_sampled" in stem:
            jsonl_map[base] = jf

    jsonl_files = sorted(jsonl_map.values())
    print(f"\nFound {len(jsonl_files)} unique JSONL files to check\n")

    # Check each JSONL
    results = []
    for jf in jsonl_files:
        print(f"  Checking {jf.name}...", end=" ", flush=True)
        result = check_jsonl_availability(jf)
        results.append(result)
        print(f"{result['available']}/{result['total']} ({result['availability_rate']})")

    # Summary by task
    print("\n" + "=" * 80)
    print("SUMMARY BY TASK")
    print("=" * 80)

    task_stats = defaultdict(lambda: {"total": 0, "available": 0, "datasets": []})

    for r in results:
        name = r["name"]
        # Determine task
        task = "other"
        base_name = name.replace("_sampled", "")
        for t, patterns in TASK_MAP.items():
            for pat in patterns:
                if base_name.startswith(pat) or base_name == pat:
                    task = t
                    break
            if task != "other":
                break

        task_stats[task]["total"] += r["total"]
        task_stats[task]["available"] += r["available"]
        task_stats[task]["datasets"].append(r)

    print(f"\n{'Task':<20} {'Available':>10} {'Total':>10} {'Rate':>10} {'Datasets':>10}")
    print("-" * 65)
    for task in ["depth", "normal", "segmentation", "keypoint", "multiview", "other"]:
        if task not in task_stats:
            continue
        s = task_stats[task]
        rate = f"{s['available'] / s['total'] * 100:.1f}%" if s["total"] > 0 else "N/A"
        print(
            f"{task:<20} {s['available']:>10} {s['total']:>10} {rate:>10} {len(s['datasets']):>10}"
        )

    total_all = sum(s["total"] for s in task_stats.values())
    avail_all = sum(s["available"] for s in task_stats.values())
    print("-" * 65)
    print(f"{'TOTAL':<20} {avail_all:>10} {total_all:>10} {avail_all / total_all * 100:.1f}%")

    # Per-dataset details for high-priority tasks
    print("\n" + "=" * 80)
    print("DETAIL: HIGH PRIORITY TASKS (segmentation, keypoint, depth, multiview)")
    print("=" * 80)

    for task in ["depth", "segmentation", "keypoint", "multiview"]:
        if task not in task_stats:
            continue
        print(f"\n--- {task.upper()} ---")
        print(f"  {'Dataset':<50} {'Avail':>8} {'Total':>8} {'Rate':>8}")
        for d in sorted(task_stats[task]["datasets"], key=lambda x: x["missing"], reverse=True):
            if d["total"] == 0:
                continue
            print(
                f"  {d['name']:<50} {d['available']:>8} {d['total']:>8} {d['availability_rate']:>8}"
            )
            if d["missing"] > 0 and d["missing_samples"]:
                print(f"    Missing samples:")
                for s in d["missing_samples"][:3]:
                    print(f"      - {s}")

    # Save full report
    report_path = Path(
        "/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722/docs/v3_availability_report.md"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# v3 Source Image Availability Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Datasets root**: `{DATASETS_ROOT}`\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"| Task | Available | Total | Rate | Datasets |\n")
        f.write(f"|------|-----------|-------|------|----------|\n")
        for task in ["depth", "normal", "segmentation", "keypoint", "multiview", "other"]:
            if task not in task_stats:
                continue
            s = task_stats[task]
            rate = f"{s['available'] / s['total'] * 100:.1f}%" if s["total"] > 0 else "N/A"
            f.write(
                f"| {task} | {s['available']} | {s['total']} | {rate} | {len(s['datasets'])} |\n"
            )
        f.write(
            f"| **TOTAL** | **{avail_all}** | **{total_all}** | **{avail_all / total_all * 100:.1f}%** | |\n"
        )

        f.write(f"\n## Per-Dataset Details\n\n")
        for task in ["depth", "normal", "segmentation", "keypoint", "multiview", "other"]:
            if task not in task_stats:
                continue
            f.write(f"### {task.upper()}\n\n")
            f.write(f"| Dataset | Available | Total | Rate | Missing |\n")
            f.write(f"|---------|-----------|-------|------|---------|\n")
            for d in sorted(task_stats[task]["datasets"], key=lambda x: x["missing"], reverse=True):
                if d["total"] == 0:
                    continue
                f.write(
                    f"| {d['name']} | {d['available']} | {d['total']} | {d['availability_rate']} | {d['missing']} |\n"
                )
            f.write("\n")

    print(f"\nFull report saved to: {report_path}")

    # Also save JSON results
    json_path = Path(
        "/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722/docs/v3_availability_report.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_samples": total_all,
                "total_available": avail_all,
                "overall_rate": f"{avail_all / total_all * 100:.1f}%",
                "task_summary": {
                    t: {"available": s["available"], "total": s["total"]}
                    for t, s in task_stats.items()
                },
                "datasets": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
