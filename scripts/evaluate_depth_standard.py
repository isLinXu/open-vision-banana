#!/usr/bin/env python3
"""Standard depth evaluation script using proper depth estimation metrics.

Replaces the archived Pearson-r correlation evaluation (which used 5 samples
in RGB space — no statistical significance) with the standard depth estimation
metrics from the literature:

  - AbsRel: Absolute Relative Error (lower is better)
  - SqRel:  Squared Relative Error (lower is better)
  - RMSE:   Root Mean Squared Error (lower is better)
  - RMSE_log: Root Mean Squared Log Error (lower is better)
  - δ<1.25, δ<1.25², δ<1.25³: Threshold accuracy (higher is better)

The script:
1. Loads depth evaluation dataset (RGB visualizations + GT depth)
2. Runs the trained model (img2img) to generate predictions
3. Decodes predicted RGB → depth using the same colormap as training
4. Computes standard metrics against GT depth
5. Reports per-sample and aggregate metrics with proper statistics

Usage:
    python scripts/evaluate_depth_standard.py \
        --lora-path checkpoints/depth_lora \
        --data-dir /path/to/vb_snvc_train_v2 \
        --num-samples 50 \
        --output results/depth_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── Project path setup ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.inference.generator import VisionGenerator


def load_depth_eval_samples(
    data_dir: Path, num_samples: int = 50
) -> list[dict]:
    """Load depth evaluation samples: input image + GT depth path."""
    depth_dir = data_dir / "depth"
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    samples = []
    image_files = sorted(depth_dir.glob("*.jpg")) + sorted(
        depth_dir.glob("*.png")
    )

    for img_path in image_files[:num_samples]:
        depth_path = img_path.with_suffix(".png")
        # Look for depth GT in subdirectory or with different naming
        depth_gt = depth_dir / f"{img_path.stem}_depth.png"
        if not depth_gt.exists():
            depth_gt = depth_dir / f"{img_path.stem}.npy"
        if not depth_gt.exists():
            # Skip if no GT depth file
            continue

        samples.append({
            "image_path": str(img_path),
            "depth_gt_path": str(depth_gt),
        })

    return samples


def evaluate_depth_model(
    lora_path: str | None,
    data_dir: str,
    num_samples: int = 50,
    colormap: str = "cube_edge",
    model_name: str = "stabilityai/sdxl-base-1.0",
    device: str = "mps",
    strength: float = 0.85,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
) -> dict:
    """Run standard depth evaluation on a trained model.

    Returns dict with per-sample and aggregate metrics.
    """
    mapper = DepthRGBMapper(colormap=colormap)

    generator = VisionGenerator(
        model_name=model_name,
        lora_path=lora_path,
        device=device,
    )

    data_path = Path(data_dir)
    samples = load_depth_eval_samples(data_path, num_samples)

    if not samples:
        warnings.warn(
            f"No depth evaluation samples found in {data_path}. "
            "Ensure depth/ subdirectory contains image + depth GT pairs."
        )
        return {"error": "no_samples", "num_samples": 0}

    print(f"Evaluating {len(samples)} samples with colormap={colormap}")

    per_sample = []
    all_preds = []
    all_gts = []

    for i, sample in enumerate(samples):
        try:
            # Load input image
            image = Image.open(sample["image_path"]).convert("RGB")

            # Generate depth visualization via img2img
            prompt = "depth estimation, depth map visualization"
            result = generator.generate(
                prompt=prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            if not result or len(result) == 0:
                print(f"  [{i+1}/{len(samples)}] No output generated, skipping")
                continue

            # Get predicted RGB → decode to depth
            pred_rgb = result[0]  # PIL Image
            pred_rgb_tensor = torch.from_numpy(
                np.array(pred_rgb).transpose(2, 0, 1)
            ).float() / 255.0
            pred_rgb_tensor = pred_rgb_tensor.unsqueeze(0).to(device)

            # Decode RGB → depth
            pred_depth = mapper.rgb_to_depth(pred_rgb_tensor).squeeze(0)

            # Load GT depth
            gt_path = Path(sample["depth_gt_path"])
            if gt_path.suffix == ".npy":
                gt_depth = torch.from_numpy(np.load(str(gt_path)))
            else:
                # Assume depth encoded in image — decode with same mapper
                gt_img = Image.open(str(gt_path)).convert("RGB")
                gt_rgb = torch.from_numpy(
                    np.array(gt_img).transpose(2, 0, 1)
                ).float() / 255.0
                gt_rgb = gt_rgb.unsqueeze(0).to(device)
                gt_depth = mapper.rgb_to_depth(gt_rgb).squeeze(0)

            # Ensure same spatial size
            if pred_depth.shape != gt_depth.shape:
                # Resize GT to match prediction
                from torch.nn.functional import interpolate
                gt_depth = interpolate(
                    gt_depth.unsqueeze(0).unsqueeze(0).float(),
                    size=pred_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

            # Compute standard metrics
            metrics = evaluate_depth(
                pred=pred_depth.unsqueeze(0).unsqueeze(0),
                gt=gt_depth.unsqueeze(0).unsqueeze(0),
                colormap=colormap,
            )

            per_sample.append({
                "image": sample["image_path"],
                **metrics,
            })
            all_preds.append(pred_depth.flatten())
            all_gts.append(gt_depth.flatten())

            print(
                f"  [{i+1}/{len(samples)}] "
                f"AbsRel={metrics['AbsRel']:.4f} "
                f"RMSE={metrics['RMSE']:.4f} "
                f"δ<1.25={metrics['δ<1.25']:.4f}"
            )

        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] Error: {e}")
            continue

    if not per_sample:
        return {"error": "all_failed", "num_samples": 0}

    # ── Aggregate statistics ─────────────────────────────────────
    # Use proper statistics: mean ± std over all evaluated samples
    metric_names = [
        "AbsRel", "SqRel", "RMSE", "RMSE_log",
        "δ<1.25", "δ<1.25²", "δ<1.25³",
        "AbsRel_encoding", "AbsRel_model",
    ]

    aggregate = {}
    for m in metric_names:
        values = [s[m] for s in per_sample if m in s]
        if values:
            aggregate[f"{m}_mean"] = float(np.mean(values))
            aggregate[f"{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            aggregate[f"{m}_median"] = float(np.median(values))

    # Also compute pixel-level aggregate metrics (all pixels pooled)
    if all_preds and all_gts:
        pooled_pred = torch.cat(all_preds)
        pooled_gt = torch.cat(all_gts)
        nonzero = pooled_gt > 0
        if nonzero.any():
            pooled_metrics = evaluate_depth(
                pred=pooled_pred[nonzero].unsqueeze(0).unsqueeze(0),
                gt=pooled_gt[nonzero].unsqueeze(0).unsqueeze(0),
            )
            aggregate["pooled"] = pooled_metrics

    return {
        "num_samples": len(per_sample),
        "colormap": colormap,
        "per_sample": per_sample,
        "aggregate": aggregate,
        "note": (
            "Standard depth estimation metrics (Eigen et al. 2014). "
            "AbsRel/RMSE lower is better; δ<1.25 higher is better. "
            "AbsRel_model = AbsRel - AbsRel_encoding (quantization noise floor)."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standard depth evaluation with proper metrics"
    )
    parser.add_argument(
        "--lora-path", type=str, default=None,
        help="Path to trained LoRA weights"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v2",
        help="Dataset directory with depth/ subdirectory",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of samples to evaluate (>= 30 for statistical significance)",
    )
    parser.add_argument(
        "--colormap", type=str, default="cube_edge",
        choices=["cube_edge", "spectral", "turbo", "adaptive"],
        help="Depth colormap used for encoding/decoding",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: stdout)",
    )
    args = parser.parse_args()

    results = evaluate_depth_model(
        lora_path=args.lora_path,
        data_dir=args.data_dir,
        num_samples=max(args.num_samples, 30),  # Enforce minimum sample size
        colormap=args.colormap,
        model_name=args.model_name,
        device=args.device,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Depth Evaluation Results (Standard Metrics)")
    print("=" * 60)

    if "aggregate" in results:
        agg = results["aggregate"]
        print(f"\nSamples evaluated: {results['num_samples']}")
        print(f"Colormap: {results['colormap']}")
        print(f"\n{'Metric':<20} {'Mean':>10} {'± Std':>10} {'Median':>10}")
        print("-" * 52)
        for m in ["AbsRel", "SqRel", "RMSE", "RMSE_log"]:
            if f"{m}_mean" in agg:
                print(
                    f"{m:<20} {agg[f'{m}_mean']:>10.4f} "
                    f"±{agg[f'{m}_std']:>9.4f} {agg[f'{m}_median']:>10.4f}"
                )
        for m in ["δ<1.25", "δ<1.25²", "δ<1.25³"]:
            if f"{m}_mean" in agg:
                print(
                    f"{m:<20} {agg[f'{m}_mean']:>10.4f} "
                    f"±{agg[f'{m}_std']:>9.4f} {agg[f'{m}_median']:>10.4f}"
                )
        if "AbsRel_model_mean" in agg:
            print(f"\n{'AbsRel_model':<20} {agg['AbsRel_model_mean']:>10.4f} "
                  f"(encoding noise: {agg.get('AbsRel_encoding_mean', 0):.4f})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
