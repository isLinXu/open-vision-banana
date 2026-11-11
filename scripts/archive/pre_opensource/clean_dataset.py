#!/usr/bin/env python3
"""
Open-Vision-Banana dataset quality cleaning tool

Features:
1. Check that every file path in the manifest exists
2. Check image quality (size, format, validity)
3. Detect abnormal target images (all black, all white, low contrast)
4. Produce a cleaned manifest

Usage:
    python scripts/clean_dataset.py --manifest output/snvc_v5full_pairs/pairs_5task_v3.jsonl --output output/snvc_v5full_pairs/pairs_5task_v3_clean.jsonl
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_file_exists(path: str) -> bool:
    """Check whether a file exists"""
    return Path(path).exists()


def check_image_valid(path: str, min_size: int = 64) -> tuple[bool, str]:
    """
    Check whether an image is valid
    
    Returns:
        (is_valid, error_message)
    """
    try:
        img = Image.open(path)
        img.verify()
        
        # reopen to get the size (the image must be reopened after verify)
        img = Image.open(path)
        w, h = img.size
        
        if w < min_size or h < min_size:
            return False, f"too small: {w}x{h}"
        
        return True, ""
    except Exception as e:
        return False, str(e)


def check_image_quality(path: str) -> dict:
    """
    Check image quality metrics
    
    Returns:
        dict with quality metrics
    """
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        
        # basic statistics
        mean = arr.mean()
        std = arr.std()
        
        # detect all black / all white
        is_all_black = mean < 5
        is_all_white = mean > 250
        
        # detect low contrast
        low_contrast = std < 10
        
        # detect single channel (grayscale)
        channel_std = arr.std(axis=(0, 1))
        is_grayscale = channel_std.max() < 5
        
        return {
            "mean": float(mean),
            "std": float(std),
            "is_all_black": bool(is_all_black),
            "is_all_white": bool(is_all_white),
            "low_contrast": bool(low_contrast),
            "is_grayscale": bool(is_grayscale),
            "size": list(arr.shape[:2]),
        }
    except Exception as e:
        return {"error": str(e)}


def validate_manifest(manifest_path: Path, sample_rate: float = 1.0) -> dict:
    """
    Validate a manifest file
    
    Args:
        manifest_path: manifest file path
        sample_rate: sampling rate (1.0 = check everything)
    
    Returns:
        validation statistics
    """
    logger.info(f"Loading manifest: {manifest_path}")
    
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    logger.info(f"Total samples: {len(rows)}")
    
    # group by task
    task_counts = {}
    for row in rows:
        task = row.get("task", "unknown")
        task_counts[task] = task_counts.get(task, 0) + 1
    
    logger.info(f"Task distribution: {task_counts}")
    
    # validation results
    results = {
        "total": len(rows),
        "valid": 0,
        "missing_input": 0,
        "missing_target": 0,
        "invalid_image": 0,
        "quality_issues": 0,
        "errors": [],
    }
    
    # sampled check
    if sample_rate < 1.0:
        import random
        random.seed(42)
        rows = random.sample(rows, int(len(rows) * sample_rate))
        logger.info(f"Sampled check: {len(rows)} samples ({sample_rate*100:.0f}%)")
    
    for i, row in enumerate(rows):
        input_path = row.get("input", "")
        target_path = row.get("target", "")
        task = row.get("task", "unknown")
        
        # check that the files exist
        input_exists = check_file_exists(input_path)
        target_exists = check_file_exists(target_path)
        
        if not input_exists:
            results["missing_input"] += 1
            if len(results["errors"]) < 10:
                results["errors"].append(f"[{i}] input missing: {input_path}")
            continue
        
        if not target_exists:
            results["missing_target"] += 1
            if len(results["errors"]) < 10:
                results["errors"].append(f"[{i}] target missing: {target_path}")
            continue
        
        # check image validity
        input_valid, input_err = check_image_valid(input_path)
        target_valid, target_err = check_image_valid(target_path)
        
        if not input_valid or not target_valid:
            results["invalid_image"] += 1
            if len(results["errors"]) < 10:
                results["errors"].append(f"[{i}] invalid image: input={input_err}, target={target_err}")
            continue
        
        # check the target image quality
        quality = check_image_quality(target_path)
        
        has_quality_issue = False
        if quality.get("is_all_black"):
            has_quality_issue = True
        if quality.get("is_all_white"):
            has_quality_issue = True
        if quality.get("low_contrast") and task != "depth":
            # the depth task allows low contrast
            has_quality_issue = True
        
        if has_quality_issue:
            results["quality_issues"] += 1
            if len(results["errors"]) < 10:
                results["errors"].append(f"[{i}] quality issue: {quality}")
        
        results["valid"] += 1
        
        if (i + 1) % 1000 == 0:
            logger.info(f"  progress: {i+1}/{len(rows)}")
    
    return results


def clean_manifest(
    manifest_path: Path,
    output_path: Path,
    skip_missing: bool = True,
    skip_invalid: bool = True,
    skip_quality_issues: bool = True,
) -> dict:
    """
    Clean the manifest and write a clean version
    
    Args:
        manifest_path: original manifest
        output_path: output clean manifest
        skip_missing: skip missing files
        skip_invalid: skip invalid images
        skip_quality_issues: skip quality issues
    
    Returns:
        cleaning statistics
    """
    logger.info(f"Cleaning manifest: {manifest_path}")
    
    rows = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    
    logger.info(f"Original samples: {len(rows)}")
    
    clean_rows = []
    stats = {
        "total": len(rows),
        "kept": 0,
        "skipped_missing": 0,
        "skipped_invalid": 0,
        "skipped_quality": 0,
        "by_task": {},
    }
    
    for i, row in enumerate(rows):
        input_path = row.get("input", "")
        target_path = row.get("target", "")
        task = row.get("task", "unknown")
        
        # check that the files exist
        if skip_missing:
            if not check_file_exists(input_path) or not check_file_exists(target_path):
                stats["skipped_missing"] += 1
                continue
        
        # check image validity
        if skip_invalid:
            input_valid, _ = check_image_valid(input_path)
            target_valid, _ = check_image_valid(target_path)
            if not input_valid or not target_valid:
                stats["skipped_invalid"] += 1
                continue
        
        # check the target image quality
        if skip_quality_issues:
            quality = check_image_quality(target_path)
            
            has_issue = False
            if quality.get("is_all_black"):
                has_issue = True
            if quality.get("is_all_white"):
                has_issue = True
            if quality.get("low_contrast") and task != "depth":
                has_issue = True
            
            if has_issue:
                stats["skipped_quality"] += 1
                continue
        
        clean_rows.append(row)
        stats["kept"] += 1
        stats["by_task"][task] = stats["by_task"].get(task, 0) + 1
        
        if (i + 1) % 1000 == 0:
            logger.info(f"  progress: {i+1}/{len(rows)}, kept: {stats['kept']}")
    
    # write the clean manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in clean_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    logger.info(f"Cleaning done: {output_path}")
    logger.info(f"  Kept: {stats['kept']}/{stats['total']} ({stats['kept']/stats['total']*100:.1f}%)")
    logger.info(f"  Skipped missing: {stats['skipped_missing']}")
    logger.info(f"  Skipped invalid: {stats['skipped_invalid']}")
    logger.info(f"  Skipped quality: {stats['skipped_quality']}")
    logger.info(f"  Task distribution: {stats['by_task']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="original manifest file")
    parser.add_argument("--output", type=Path, required=True, help="output clean manifest")
    parser.add_argument("--validate-only", action="store_true", help="validate only, do not write a cleaned file")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="sampling rate (0.0-1.0)")
    parser.add_argument("--keep-missing", action="store_true", help="keep samples with missing files")
    parser.add_argument("--keep-invalid", action="store_true", help="keep invalid images")
    parser.add_argument("--keep-quality-issues", action="store_true", help="keep samples with quality issues")
    
    args = parser.parse_args()
    
    if args.validate_only:
        results = validate_manifest(args.manifest, args.sample_rate)
        logger.info(f"\nValidation results:")
        logger.info(f"  total samples: {results['total']}")
        logger.info(f"  valid: {results['valid']}")
        logger.info(f"  missing input: {results['missing_input']}")
        logger.info(f"  missing target: {results['missing_target']}")
        logger.info(f"  invalid images: {results['invalid_image']}")
        logger.info(f"  quality issues: {results['quality_issues']}")
        if results["errors"]:
            logger.info(f"\nError examples (first 10):")
            for err in results["errors"]:
                logger.info(f"  {err}")
    else:
        stats = clean_manifest(
            args.manifest,
            args.output,
            skip_missing=not args.keep_missing,
            skip_invalid=not args.keep_invalid,
            skip_quality_issues=not args.keep_quality_issues,
        )


if __name__ == "__main__":
    main()
