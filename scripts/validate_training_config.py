#!/usr/bin/env python3
"""Validate a training config and print a reproducibility manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_vision_banana.config import VisionBananaConfig
from open_vision_banana.reproducibility import build_experiment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML training configuration")
    parser.add_argument("--data-root", help="Override data.root for this validation")
    parser.add_argument("--generation-path", help="Override data.generation_path")
    parser.add_argument(
        "--hash-mode",
        choices=("metadata", "content"),
        default="metadata",
        help="Fast metadata fingerprint or full content SHA-256",
    )
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Validate config structure even when configured data is unavailable",
    )
    parser.add_argument("--output", help="Optional JSON manifest output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = VisionBananaConfig.from_yaml(config_path, strict=True)
    if args.data_root:
        config.data.root = args.data_root
    if args.generation_path:
        config.data.generation_path = args.generation_path
    config.validate()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = build_experiment_manifest(
        config,
        repo_root=repo_root,
        fingerprint_mode=args.hash_mode,
        require_data=not args.allow_missing_data,
    )
    rendered = json.dumps(manifest, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
