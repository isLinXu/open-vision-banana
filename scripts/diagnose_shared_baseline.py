#!/usr/bin/env python3
"""Audit a shared-baseline training config and JSONL manifest for contract mismatches.

Checks:
1. Whether the model's LoRA target_modules adapt attention Q/K/V layers (required
   for prompt-conditioned semantic color legends in multi-task runs).
2. Whether text_encoder_max_length is large enough for 19-class Cityscapes or
   open-vocabulary RGB legends without truncation.
3. Whether depth targets in the manifest match the configured depth_curve and
   depth_colormap (detects grayscale or relative depth targets paired with
   metric_power decoding).
4. Whether depth prompts are uniform and match metric vs relative supervision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.validation import validate_manifest_contract  # noqa: E402
from open_vision_banana.training.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Training YAML config")
    parser.add_argument("--pair-manifest", type=Path, required=True, help="Path to pairs.jsonl")
    parser.add_argument("--data-dir", type=Path, default=None, help="Optional root for relative paths")
    parser.add_argument("--model-name", default=None, help="Override model name")
    parser.add_argument("--depth-colormap", default="cube_edge")
    parser.add_argument("--depth-curve", default=None)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_name = args.model_name or "sana-1600m"
    lora_targets = None
    depth_curve = args.depth_curve or "metric_power"
    data_dir = args.data_dir

    if args.config is not None:
        cfg = load_config(config_path=args.config)
        if args.model_name is None:
            model_name = cfg.model_name
        lora_targets = cfg.lora_targets
        if args.depth_curve is None:
            depth_curve = cfg.depth_curve
        if data_dir is None and cfg.data_dir:
            data_dir = Path(cfg.data_dir)

    report = validate_manifest_contract(
        args.pair_manifest,
        data_root=data_dir,
        model_name=model_name,
        lora_targets=lora_targets,
        depth_colormap=args.depth_colormap,
        depth_curve=depth_curve,
        max_target_samples_per_task=args.max_samples,
    )
    payload = report.to_dict()
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(report.summary())
    print(json.dumps(payload["stats"], indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
