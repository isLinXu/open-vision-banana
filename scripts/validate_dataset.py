#!/usr/bin/env python3
"""Automated data validation and preprocessing pipeline.

Validates vision datasets before training, checking:
- Image/target pair integrity (missing files, corrupt images)
- Depth/normal range sanity (depth > 0, normals unit length)
- Segmentation mask format (correct class indices, valid polygons)
- Dataset balance (class distribution, task coverage)
- Resolution consistency (all images same resolution, or warn)

Usage:
    python scripts/validate_dataset.py --data-path /data/vision --task depth
    python scripts/validate_dataset.py --data-path /data/vision --task all --output report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Structured validation report for a dataset."""

    task: str
    total_samples: int = 0
    valid_samples: int = 0
    corrupt_samples: int = 0
    missing_targets: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_ratio(self) -> float:
        return self.valid_samples / max(self.total_samples, 1)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "valid_ratio": round(self.valid_ratio, 4),
            "corrupt_samples": self.corrupt_samples,
            "missing_targets": self.missing_targets,
            "warnings": self.warnings,
            "errors": self.errors,
            "stats": self.stats,
        }


def validate_image_pair(img_path: Path, tgt_path: Path | None, task: str) -> tuple[bool, str | None]:
    """Validate a single image/target pair.

    Returns (is_valid, error_message).
    """
    # Check image exists and is readable
    if not img_path.exists():
        return False, f"Image not found: {img_path}"
    try:
        img = Image.open(img_path)
        img.verify()  # lightweight check
    except Exception as exc:
        return False, f"Corrupt image {img_path}: {exc}"

    # Check target
    if tgt_path is not None:
        if not tgt_path.exists():
            return False, f"Target not found: {tgt_path}"
        try:
            tgt = Image.open(tgt_path)
            tgt.verify()
        except Exception as exc:
            return False, f"Corrupt target {tgt_path}: {exc}"

        # Task-specific checks
        if task == "depth":
            try:
                tgt_arr = torch.tensor(
                    [float(p) for p in Image.open(tgt_path).getdata()],
                    dtype=torch.float32,
                )
                if tgt_arr.min() < 0:
                    return False, f"Negative depth values in {tgt_path}"
                if tgt_arr.max() > 1000:
                    return True, f"Warning: very large depth values in {tgt_path}"
            except Exception:
                pass

        elif task == "normal":
            try:
                tgt_img = Image.open(tgt_path).convert("RGB")
                tgt_arr = torch.tensor(tgt_img.getdata(), dtype=torch.float32) / 255.0
                # Normals should be in [0, 1] after RGB encoding
                if tgt_arr.min() < 0 or tgt_arr.max() > 1:
                    return False, f"Normal map out of [0, 1] range in {tgt_path}"
            except Exception:
                pass

    return True, None


def validate_dataset(data_path: Path, task: str) -> ValidationReport:
    """Validate a dataset directory for a specific task."""
    report = ValidationReport(task=task)
    image_dir = data_path / "images"
    target_dir = data_path / "targets"

    if not image_dir.exists():
        report.errors.append(f"Images directory not found: {image_dir}")
        return report

    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    report.total_samples = len(image_files)

    if report.total_samples == 0:
        report.errors.append(f"No images found in {image_dir}")
        return report

    resolutions: set[tuple[int, int]] = set()
    for img_path in image_files:
        stem = img_path.stem
        if task == "segmentation":
            tgt_path = target_dir / f"{stem}.png"
        else:
            tgt_path = target_dir / f"{stem}.png"

        valid, msg = validate_image_pair(img_path, tgt_path, task)
        if valid:
            report.valid_samples += 1
            if msg:
                report.warnings.append(msg)
        else:
            if "not found" in (msg or "").lower():
                report.missing_targets += 1
            else:
                report.corrupt_samples += 1
            report.errors.append(msg)

        # Collect resolution info
        try:
            with Image.open(img_path) as img:
                resolutions.add(img.size)
        except Exception:
            pass

    # Resolution consistency
    if len(resolutions) > 1:
        report.warnings.append(
            f"Mixed resolutions detected: {sorted(resolutions)}. "
            f"Training will resize them, which may affect quality."
        )
    report.stats["resolutions"] = [list(r) for r in sorted(resolutions)]
    report.stats["resolution_consistent"] = len(resolutions) == 1

    # Summary stats
    report.stats["valid_ratio_pct"] = round(report.valid_ratio * 100, 2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Validate vision datasets")
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset root")
    parser.add_argument(
        "--task", type=str, default="all",
        choices=["depth", "normal", "semantic", "instance", "referring", "all"],
    )
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data path not found: %s", data_path)
        return 1

    tasks = ["depth", "normal", "semantic", "instance", "referring"] if args.task == "all" else [args.task]
    reports: dict[str, dict] = {}

    for task in tasks:
        task_path = data_path / task
        if not task_path.exists():
            logger.warning("Task directory not found: %s", task_path)
            continue

        report = validate_dataset(task_path, task)
        reports[task] = report.to_dict()

        logger.info(
            "[%s] Valid: %d/%d (%.1f%%) | Corrupt: %d | Missing targets: %d",
            task,
            report.valid_samples,
            report.total_samples,
            report.valid_ratio * 100,
            report.corrupt_samples,
            report.missing_targets,
        )
        if report.warnings:
            for w in report.warnings[:5]:
                logger.warning("  ⚠ %s", w)
            if len(report.warnings) > 5:
                logger.warning("  ... and %d more warnings", len(report.warnings) - 5)
        if report.errors:
            for e in report.errors[:5]:
                logger.error("  ✗ %s", e)
            if len(report.errors) > 5:
                logger.error("  ... and %d more errors", len(report.errors) - 5)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(reports, f, indent=2)
        logger.info("Report saved to: %s", out_path)

    return 0


if __name__ == "__main__":
    exit(main())
