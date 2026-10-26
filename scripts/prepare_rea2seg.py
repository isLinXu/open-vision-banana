#!/usr/bin/env python3
"""Convert local Rea2Seg Parquet (or an existing HF cache) into training pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_vision_banana.rea2seg import (
    CATEGORIES,
    DATASET,
    build_rea2seg,
    cached_hf_shards,
    parquet_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--parquet", nargs="+", type=Path, help="Local train shards, in sorted order"
    )
    inputs.add_argument("--dataset", help="HF dataset ID; reads an already downloaded cache only")
    parser.add_argument(
        "--revision", help="Source revision; required for HF, recorded for local shards"
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--max-samples", type=int, help="Positive cap after category filtering")
    parser.add_argument(
        "--reject-invalid",
        action="store_true",
        help="Audit and omit invalid/duplicate/held-out incoming rows",
    )
    parser.add_argument("--exclude-image-ids", type=Path, help="RES evaluation_image_ids.json")
    parser.add_argument(
        "--merge",
        nargs=2,
        type=Path,
        metavar=("MANIFEST", "ROOT"),
        help="Existing train/val manifest and its asset root; keep its splits",
    )
    parser.add_argument(
        "--exclude",
        nargs=2,
        type=Path,
        action="append",
        default=[],
        metavar=("MANIFEST", "ROOT"),
        help="Known eval manifest/root; repeatable",
    )
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.dataset and not args.revision:
        parser.error("--dataset requires --revision")
    paths, revision = args.parquet, args.revision
    if args.dataset:
        paths, revision = cached_hf_shards(args.dataset, args.revision)
    report = build_rea2seg(
        parquet_rows(paths),
        args.output_dir,
        source=args.dataset or DATASET,
        revision=revision,
        categories=args.categories,
        max_samples=args.max_samples,
        merge=tuple(args.merge) if args.merge else None,
        exclude=args.exclude,
        reject_invalid=args.reject_invalid,
        exclude_image_ids=json.loads(args.exclude_image_ids.read_text(encoding="utf-8"))
        if args.exclude_image_ids
        else (),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
