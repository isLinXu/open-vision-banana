#!/usr/bin/env python3
"""
Dataset cleaning and optimization pipeline

Features:
1. Validate and clean the manifest
2. Regenerate high-quality normal pseudo-labels
3. Improve the segmentation pseudo-labels
4. Produce the final clean manifest

Usage:
    python scripts/dataset_cleaning_pipeline.py \
        --manifest output/snvc_v5full_pairs/pairs_5task_v3.jsonl \
        --output-dir output/snvc_v5full_pairs/cleaned/ \
        --normal-dir output/normal_pseudo_v2/ \
        --seg-dir output/seg_pseudo_v2/
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_clean_manifest(manifest: Path, output: Path) -> dict:
    """Run the manifest cleaning"""
    logger.info("=" * 60)
    logger.info("Step 1: clean the manifest")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "clean_dataset.py"),
        "--manifest", str(manifest),
        "--output", str(output),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Cleaning failed: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    # parse the output
    stats = {}
    for line in result.stdout.split("\n"):
        if "Kept:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                stats["kept"] = parts[1].strip()
        if "Skipped missing:" in line:
            parts = line.split(":")
            if len(parts) > 1:
                stats["skipped_missing"] = parts[1].strip()
    
    logger.info(f"Cleaning done: {output}")
    return {"success": True, "stats": stats}


def run_regen_normals(manifest: Path, output_dir: Path) -> dict:
    """Regenerate the normal pseudo-labels"""
    logger.info("=" * 60)
    logger.info("Step 2: regenerate the normal pseudo-labels")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "regen_normal_pseudo.py"),
        "--manifest", str(manifest),
        "--output-dir", str(output_dir),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Generation failed: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    logger.info(f"Normal pseudo-labels generated: {output_dir}")
    return {"success": True}


def run_improve_seg(manifest: Path, output_dir: Path) -> dict:
    """Improve the segmentation pseudo-labels"""
    logger.info("=" * 60)
    logger.info("Step 3: improve the segmentation pseudo-labels")
    logger.info("=" * 60)
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "improve_seg_pseudo.py"),
        "--manifest", str(manifest),
        "--output-dir", str(output_dir),
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Generation failed: {result.stderr}")
        return {"success": False, "error": result.stderr}
    
    logger.info(f"Seg pseudo-labels generated: {output_dir}")
    return {"success": True}


def generate_final_manifest(
    clean_manifest: Path,
    normal_dir: Path,
    seg_dir: Path,
    output: Path,
) -> dict:
    """Produce the final manifest"""
    logger.info("=" * 60)
    logger.info("Step 4: produce the final manifest")
    logger.info("=" * 60)
    
    # load the cleaned manifest
    rows = []
    with open(clean_manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    logger.info(f"Base samples: {len(rows)}")
    
    # add the normal samples
    normal_added = 0
    depth_rows = [r for r in rows if r.get("task") == "depth"]
    
    for row in depth_rows:
        depth_target = row.get("target", "")
        if not depth_target:
            continue
        
        depth_name = Path(depth_target).stem
        normal_path = normal_dir / f"{depth_name}_normal.png"
        
        if normal_path.exists():
            normal_row = row.copy()
            normal_row["task"] = "normal"
            normal_row["target"] = str(normal_path)
            rows.append(normal_row)
            normal_added += 1
    
    logger.info(f"Normal samples added: {normal_added}")
    
    # add the seg samples
    seg_added = 0
    input_rows = [r for r in rows if r.get("task") == "depth"]  # reuse the depth inputs
    
    for row in input_rows:
        input_path = row.get("input", "")
        if not input_path:
            continue
        
        input_name = Path(input_path).stem
        seg_path = seg_dir / f"{input_name}_seg.png"
        
        if seg_path.exists():
            seg_row = row.copy()
            seg_row["task"] = "seg"
            seg_row["target"] = str(seg_path)
            rows.append(seg_row)
            seg_added += 1
    
    logger.info(f"Seg samples added: {seg_added}")
    
    # write the final manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # statistics
    task_counts = {}
    for row in rows:
        task = row.get("task", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    
    logger.info(f"Final manifest: {output}")
    logger.info(f"  total samples: {len(rows)}")
    logger.info(f"  task distribution: {task_counts}")
    
    return {
        "total": len(rows),
        "by_task": task_counts,
        "normal_added": normal_added,
        "seg_added": seg_added,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="original manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--normal-dir", type=Path, required=True, help="normal output directory")
    parser.add_argument("--seg-dir", type=Path, required=True, help="seg output directory")
    parser.add_argument("--skip-clean", action="store_true", help="skip the cleaning step")
    parser.add_argument("--skip-normal", action="store_true", help="skip the normal generation")
    parser.add_argument("--skip-seg", action="store_true", help="skip the seg generation")
    
    args = parser.parse_args()
    
    # create the output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.normal_dir.mkdir(parents=True, exist_ok=True)
    args.seg_dir.mkdir(parents=True, exist_ok=True)
    
    # step 1: clean the manifest
    clean_manifest_path = args.output_dir / "pairs_clean.jsonl"
    if not args.skip_clean:
        run_clean_manifest(args.manifest, clean_manifest_path)
    else:
        logger.info("Skipping the cleaning step")
        clean_manifest_path = args.manifest
    
    # step 2: regenerate the normals
    if not args.skip_normal:
        run_regen_normals(clean_manifest_path, args.normal_dir)
    else:
        logger.info("Skipping the normal generation")
    
    # step 3: improve the seg labels
    if not args.skip_seg:
        run_improve_seg(clean_manifest_path, args.seg_dir)
    else:
        logger.info("Skipping the seg generation")
    
    # step 4: produce the final manifest
    final_manifest = args.output_dir / "pairs_final.jsonl"
    generate_final_manifest(
        clean_manifest_path,
        args.normal_dir,
        args.seg_dir,
        final_manifest,
    )
    
    logger.info("=" * 60)
    logger.info("Pipeline done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
