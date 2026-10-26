#!/usr/bin/env python3
"""Prepare the pinned LLM-Seg40K COCO subset for the existing pair-manifest loader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_vision_banana.llmseg import build_llmseg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Official train.json")
    parser.add_argument("--validation-annotations", type=Path, required=True)
    parser.add_argument(
        "--image-root", type=Path, required=True, help="COCO train2017 or train2014 images"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-image-ids", type=Path, required=True, help="RES evaluation_image_ids.json"
    )
    parser.add_argument(
        "--exclude", nargs=2, type=Path, action="append", default=[], metavar=("MANIFEST", "ROOT")
    )
    parser.add_argument("--merge", nargs=2, type=Path, metavar=("MANIFEST", "ROOT"))
    parser.add_argument(
        "--reject-invalid",
        action="store_true",
        help="Audit and omit invalid/duplicate/held-out incoming rows",
    )
    args = parser.parse_args()
    blocked = json.loads(args.exclude_image_ids.read_text(encoding="utf-8"))
    report = build_llmseg(
        args.annotations,
        args.validation_annotations,
        args.image_root,
        args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        exclude_image_ids=blocked,
        exclude=args.exclude,
        merge=tuple(args.merge) if args.merge else None,
        reject_invalid=args.reject_invalid,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
