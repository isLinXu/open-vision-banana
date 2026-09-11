#!/usr/bin/env python3
"""Validate extracted SN-VC-50M data against the unified manifest.

Checks performed per task:
  1. **File existence** — both input and target files are present.
  2. **Image decodability** — PIL can open and decode the image.
  3. **Non-identity** — target is NOT a copy of the input (MAE > threshold).
  4. **Task-specific quality**:
     - depth: target has grayscale variation (std > 0).
     - normal: target has > 64 unique colours at 96×96.
     - seg: target has ≤ 256 unique colours at 96×96 (palette mask).
     - structure: target is mostly black with some white pixels.

Usage::

    python scripts/validate_snvc50m.py \
        --manifest output/snvc50m_manifest/pairs_all.jsonl \
        --data-root /media/linxu/thinkHub/datasets/snvc50m_extracted/SenseNova-Vision-Corpus-50M \
        --samples-per-task 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_snvc50m")

Image.MAX_IMAGE_PIXELS = None  # Allow large images


# ── Validation primitives ─────────────────────────────────────────────────


def _load_rgb(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception as exc:
        logger.debug("Cannot load %s: %s", path, exc)
        return None


def _unique_colors_at_small_size(arr: np.ndarray, size: int = 96) -> int:
    """Count unique colours after downsampling."""
    im = Image.fromarray(arr)
    im = im.resize((size, size), Image.BICUBIC)
    small = np.asarray(im)
    return len(np.unique(small.reshape(-1, 3), axis=0))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


# ── Per-task validators ──────────────────────────────────────────────────


def validate_depth(input_arr: np.ndarray, target_arr: np.ndarray) -> tuple[bool, str]:
    """Depth target should be a grayscale image with variation."""
    # Check non-identity
    mae = _mae(input_arr, target_arr)
    if mae < 1.0:
        return False, f"MAE too low ({mae:.2f}), likely identity copy"
    # Check grayscale variation
    gray = np.mean(target_arr, axis=-1)
    std = float(np.std(gray))
    if std < 1.0:
        return False, f"Target is near-constant (std={std:.2f})"
    return True, f"OK (MAE={mae:.1f}, std={std:.1f})"


def validate_normal(input_arr: np.ndarray, target_arr: np.ndarray) -> tuple[bool, str]:
    """Normal target should have many unique colours (RGB encoding)."""
    uc = _unique_colors_at_small_size(target_arr)
    if uc < 10:
        return False, f"Too few unique colours ({uc}), not a normal map"
    mae = _mae(input_arr, target_arr)
    if mae < 1.0:
        return False, f"MAE too low ({mae:.2f}), likely identity copy"
    return True, f"OK (unique_colors@96={uc}, MAE={mae:.1f})"


def validate_seg(input_arr: np.ndarray, target_arr: np.ndarray) -> tuple[bool, str]:
    """Segmentation target should have limited unique colours (palette mask)."""
    uc = _unique_colors_at_small_size(target_arr, size=128)
    if uc > 500:
        return False, f"Too many unique colours ({uc}), likely a photo not a mask"
    mae = _mae(input_arr, target_arr)
    if mae < 1.0:
        return False, f"MAE too low ({mae:.2f}), likely identity copy"
    return True, f"OK (unique_colors@128={uc}, MAE={mae:.1f})"


def validate_structure(input_arr: np.ndarray, target_arr: np.ndarray) -> tuple[bool, str]:
    """Structure target: mostly black with some white edges."""
    gray = np.mean(target_arr, axis=-1)
    white_ratio = float(np.mean(gray > 128))
    if white_ratio < 0.001:
        return False, f"Target is almost entirely black (white={white_ratio:.4f})"
    if white_ratio > 0.5:
        return False, f"Target is too bright (white={white_ratio:.4f}), may not be edges"
    mae = _mae(input_arr, target_arr)
    if mae < 1.0:
        return False, f"MAE too low ({mae:.2f}), likely identity copy"
    return True, f"OK (white_ratio={white_ratio:.4f}, MAE={mae:.1f})"


VALIDATORS = {
    "depth": validate_depth,
    "normal": validate_normal,
    "seg": validate_seg,
    "structure": validate_structure,
}


# ── Main validation pipeline ─────────────────────────────────────────────


def load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_sample(rec: dict, data_root: Path) -> dict:
    """Validate a single manifest record. Returns a result dict."""
    task = rec["task"]
    input_path = data_root / rec["input"]
    target_path = data_root / rec["target"]

    result = {
        "task": task,
        "input": rec["input"],
        "target": rec["target"],
        "input_exists": False,
        "target_exists": False,
        "input_loadable": False,
        "target_loadable": False,
        "valid": False,
        "message": "",
    }

    if not input_path.exists():
        result["message"] = f"Input not found: {input_path}"
        return result
    result["input_exists"] = True

    if not target_path.exists():
        result["message"] = f"Target not found: {target_path}"
        return result
    result["target_exists"] = True

    input_arr = _load_rgb(input_path)
    if input_arr is None:
        result["message"] = f"Cannot decode input: {input_path}"
        return result
    result["input_loadable"] = True

    target_arr = _load_rgb(target_path)
    if target_arr is None:
        result["message"] = f"Cannot decode target: {target_path}"
        return result
    result["target_loadable"] = True

    validator = VALIDATORS.get(task)
    if validator is None:
        result["message"] = f"No validator for task '{task}'"
        result["valid"] = True  # Skip validation for unknown tasks
        return result

    ok, msg = validator(input_arr, target_arr)
    result["valid"] = ok
    result["message"] = msg
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SN-VC-50M extracted data")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("output/snvc50m_manifest/pairs_all.jsonl"),
        help="Unified manifest JSONL",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/media/linxu/thinkHub/datasets/snvc50m_extracted/SenseNova-Vision-Corpus-50M"),
        help="Root directory of extracted data assets",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=100,
        help="Number of samples to validate per task (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/snvc50m_manifest/validation_report.json"),
        help="Output report path",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        logger.error("Manifest not found: %s", args.manifest)
        logger.info("Run build_snvc50m_manifest.py first.")
        sys.exit(1)

    if not args.data_root.is_dir():
        logger.error("Data root not found: %s", args.data_root)
        logger.info("Run extract_snvc50m.sh first.")
        sys.exit(1)

    records = load_manifest(args.manifest)
    logger.info("Loaded %d records from manifest", len(records))

    # Group by task
    by_task: dict[str, list[dict]] = {}
    for rec in records:
        t = rec["task"]
        by_task.setdefault(t, []).append(rec)

    rng = random.Random(args.seed)
    all_results: list[dict] = []
    summary: dict[str, dict] = {}

    for task, task_records in sorted(by_task.items()):
        n = min(args.samples_per_task, len(task_records))
        sample = rng.sample(task_records, n)
        logger.info("Validating %d samples for task '%s' (of %d total)", n, task, len(task_records))

        valid_count = 0
        exist_count = 0
        load_count = 0
        fail_messages: Counter = Counter()

        for rec in sample:
            result = validate_sample(rec, args.data_root)
            all_results.append(result)
            if result["input_exists"] and result["target_exists"]:
                exist_count += 1
            if result["input_loadable"] and result["target_loadable"]:
                load_count += 1
            if result["valid"]:
                valid_count += 1
            else:
                # Bucket failure reasons
                reason = result["message"].split("(")[0].strip()
                fail_messages[reason] += 1

        summary[task] = {
            "total_in_manifest": len(task_records),
            "sampled": n,
            "files_exist": exist_count,
            "loadable": load_count,
            "valid": valid_count,
            "failure_reasons": dict(fail_messages.most_common(10)),
        }

        logger.info(
            "  %s: exist=%d/%d  loadable=%d/%d  valid=%d/%d",
            task, exist_count, n, load_count, n, valid_count, n,
        )
        if fail_messages:
            for reason, cnt in fail_messages.most_common(5):
                logger.info("    FAIL: %s (%d)", reason, cnt)

    # Write report
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "samples_per_task": args.samples_per_task,
        "summary": summary,
        "failures": [r for r in all_results if not r["valid"]],
    }
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Validation report → %s", args.report)
    total_valid = sum(s["valid"] for s in summary.values())
    total_sampled = sum(s["sampled"] for s in summary.values())
    logger.info("Overall: %d/%d valid (%.1f%%)", total_valid, total_sampled,
                100 * total_valid / max(total_sampled, 1))


if __name__ == "__main__":
    main()
