#!/usr/bin/env python3
"""Evaluate a source-conditioned SSD checkpoint on v3 relative-depth labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.conditioning import (  # noqa: E402
    SourceConditioningConfig,
    calibrate_source_conditioned_output,
)
from open_vision_banana.data.depth_encoding import relative_depth_prompt  # noqa: E402
from open_vision_banana.evaluation.relative_depth import (  # noqa: E402
    evaluate_relative_depth_rgb,
)
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402


def depth_names(data_dir: Path, split: str) -> list[str]:
    split_path = data_dir / "train_val_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing v3 split manifest: {split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    names = payload.get(split, {}).get("depth", [])
    if not names:
        raise ValueError(f"No depth {split} samples declared")
    return list(names)


def validation_depth_names(data_dir: Path) -> list[str]:
    """Backward-compatible accessor for the official validation split."""
    return depth_names(data_dir, "val")


def load_target_tensor(path: Path, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(
        (resolution, resolution),
        Image.Resampling.LANCZOS,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def evaluate_prediction_variants(
    raw_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    config: SourceConditioningConfig,
    colormap: str,
) -> dict[str, float]:
    """Report model-native metrics separately from configured post-processing."""
    metrics = {
        f"raw_{key}": value
        for key, value in evaluate_relative_depth_rgb(
            raw_prediction,
            target,
            colormap=colormap,
        ).items()
    }
    if config.output_calibration != "none":
        calibrated = calibrate_source_conditioned_output(raw_prediction, config)
        metrics.update(
            {
                f"calibrated_{key}": value
                for key, value in evaluate_relative_depth_rgb(
                    calibrated,
                    target,
                    colormap=colormap,
                ).items()
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="segmind/SSD-1B")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--colormap", default=None)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Manifest split to evaluate (default: val).",
    )
    args = parser.parse_args()
    if args.num_samples <= 0 or args.resolution <= 0:
        parser.error("--num-samples and --resolution must be positive")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generator = BaseGenerator(
        model_name=args.model,
        lora_path=str(args.checkpoint.expanduser().resolve()),
        device=device,
        dtype=torch.bfloat16,
    )
    config = generator.source_conditioning_config
    if config is None:
        raise ValueError("Checkpoint is not source-conditioned")
    colormap = args.colormap or config.target_colormap
    if colormap == "unknown":
        raise ValueError("Checkpoint has no target colormap; pass --colormap")
    prompt = config.recommended_prompt or relative_depth_prompt(colormap)
    # Generate exactly once per sample with post-processing disabled.  The
    # configured calibration is then applied to that same tensor so raw model
    # quality cannot be hidden by a second stochastic denoising pass.
    generator.source_conditioning_config = replace(
        config,
        output_calibration="none",
    )

    data_dir = args.data_dir.expanduser().resolve()
    image_dir = data_dir / "depth" / "images"
    target_dir = data_dir / "depth" / "targets"
    targets = {path.stem: path for path in target_dir.glob("*") if path.is_file()}
    results = []
    for index, name in enumerate(depth_names(data_dir, args.split)[: args.num_samples]):
        image_path = image_dir / name
        target_path = targets.get(Path(name).stem)
        if not image_path.is_file() or target_path is None:
            continue
        source = Image.open(image_path).convert("RGB").resize(
            (args.resolution, args.resolution),
            Image.Resampling.LANCZOS,
        )
        raw_prediction = generator.generate(
            prompt,
            image=source,
            seed=args.seed + index,
        )
        target = load_target_tensor(target_path, args.resolution)
        results.append(
            {
                "sample": name,
                **evaluate_prediction_variants(
                    raw_prediction,
                    target,
                    config=config,
                    colormap=colormap,
                ),
            }
        )
    if not results:
        raise RuntimeError(f"No {args.split} samples were evaluated")
    metric_names = [key for key in results[0] if key != "sample"]
    aggregate = {
        key: {
            "mean": float(np.mean([sample[key] for sample in results])),
            "std": float(np.std([sample[key] for sample in results])),
        }
        for key in metric_names
    }
    report = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "data_dir": str(data_dir),
        "split": args.split,
        "label_semantics": "relative_depth_rgb_without_metric_scale",
        "colormap": colormap,
        "output_calibration": config.output_calibration,
        "num_samples": len(results),
        "aggregate": aggregate,
        "samples": results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
