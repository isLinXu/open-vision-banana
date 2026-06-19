#!/usr/bin/env python3
"""Validate Vision Banana training dataset format.

Checks paired ``images/`` + ``targets/`` directories for structure,
size consistency, and value-range issues before training.

Usage:
    python scripts/validate_data.py --data-path my_tiny_data/
    python scripts/validate_data.py --data-path my_tiny_data/ --strict
    python scripts/validate_data.py --data-path my_tiny_data/ --min-pairs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from open_vision_banana.data.validation import validate_vision_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Vision Banana dataset format (images/ + targets/)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Root directory containing images/ and targets/",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=1,
        help="Minimum number of valid pairs required (default: 1)",
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=64,
        help="Minimum image width/height in pixels (default: 64)",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=None,
        help="Warn when resolution exceeds this value",
    )
    parser.add_argument(
        "--require-square",
        action="store_true",
        help="Warn on non-square images",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit code 1)",
    )
    parser.add_argument(
        "--no-value-check",
        action="store_true",
        help="Skip pixel value range checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)

    report = validate_vision_dataset(
        data_path,
        min_pairs=args.min_pairs,
        require_square=args.require_square,
        min_resolution=args.min_resolution,
        max_resolution=args.max_resolution,
        check_value_range=not args.no_value_check,
    )

    print(report.summary())

    if args.strict and report.warnings:
        return 1
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
