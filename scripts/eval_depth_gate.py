#!/usr/bin/env python3
"""Phase 2 depth gate evaluation — load trained LoRA checkpoint and evaluate.

Loads FLUX.1 + trained LoRA adapter, runs img2img inference on held-out
depth samples, decodes RGB→depth, and computes standard depth metrics.

Usage:
    python scripts/eval_depth_gate.py \
        --lora-dir output/flux_depth_gate/checkpoint-200 \
        --data data/vb_snvc_train_v3/depth \
        --n-samples 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.inference.generator import BaseGenerator

PROMPT = "depth estimation, depth map visualization"


def load_v3_depth_samples(data_dir: Path, split: str = "val", max_samples: int = 50):
    """Load depth samples from v3 format directory."""
    img_dir = data_dir / split / "input"
    tgt_dir = data_dir / split / "target"
    if not img_dir.exists():
        # Fallback: try flat images/targets layout
        img_dir = data_dir / "images"
        tgt_dir = data_dir / "targets"
    if not img_dir.exists():
        print(f"WARNING: No depth data found at {data_dir}")
        return []

    items = []
    for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
        tgt_path = tgt_dir / (img_path.stem + ".png")
        if tgt_path.exists():
            items.append({"image": img_path, "target": tgt_path})
        if len(items) >= max_samples:
            break
    return items


@torch.no_grad()
def flux_predict(generator, image: Image.Image, steps: int, strength: float):
    out = generator.generate(
        prompt=PROMPT, image=image,
        strength=strength, num_inference_steps=steps, guidance_scale=0.0,
    )
    # out is torch.Tensor [B, 3, H, W] in [0, 1]
    if out is None:
        return None
    return out  # already [1,3,H,W] in [0,1]


def main():
    ap = argparse.ArgumentParser(description="Evaluate depth gate LoRA checkpoint")
    ap.add_argument("--model", default="/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell")
    ap.add_argument("--lora-dir", type=str, required=True,
                    help="Path to saved LoRA adapter directory")
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data/vb_snvc_train_v3/depth"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output/depth_gate_eval"))
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--infer-steps", type=int, default=4)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    mapper = DepthRGBMapper(colormap="cube_edge")

    # ── 1) Load FLUX.1 + LoRA ──────────────────────────────────────────
    lora_path = Path(args.lora_dir)
    if not lora_path.exists():
        print(f"ERROR: LoRA directory not found: {lora_path}")
        return

    print(f"[1/3] Loading FLUX.1 + LoRA on {args.device} …", flush=True)
    print(f"      LoRA: {lora_path}", flush=True)
    t0 = time.time()
    gen = BaseGenerator(
        model_name=args.model,
        lora_path=str(lora_path),
        device=args.device,
        dtype=torch.bfloat16,
    )
    # Keep text encoders and VAE in float32 for stability
    for _m in ("text_encoder", "text_encoder_2", "vae"):
        _mod = getattr(gen.pipe, _m, None)
        if _mod is not None:
            _mod.to(torch.float32)
    print(f"      FLUX + LoRA loaded in {time.time()-t0:.1f}s", flush=True)

    # ── 2) Load held-out samples ───────────────────────────────────────
    data_dir = Path(args.data)
    # Try val split first, then train split
    items = load_v3_depth_samples(data_dir, split="val", max_samples=args.n_samples)
    split_used = "val"
    if len(items) < 5:
        items = load_v3_depth_samples(data_dir, split="train", max_samples=args.n_samples)
        split_used = "train"
    print(f"[2/3] {len(items)} samples from {split_used} split", flush=True)

    if not items:
        print("ERROR: No depth samples found. Check --data path.")
        return

    # ── 3) Run inference + evaluate ────────────────────────────────────
    print(f"[3/3] Inference ({args.infer_steps} steps, strength={args.strength}, res={args.res}) …",
          flush=True)

    per_sample = []
    all_metrics_lora = []
    all_pred_depths = []
    all_gt_depths = []
    
    for i, item in enumerate(items):
        image = Image.open(item["image"]).convert("RGB")
        image = image.resize((args.res, args.res), Image.LANCZOS)
    
        # Load GT depth target (RGB encoded)
        tgt_rgb = torch.from_numpy(
            np.array(Image.open(item["target"]).convert("RGB")).astype(np.float32)
            .transpose(2, 0, 1)
        ) / 255.0
        gt_depth = mapper.rgb_to_depth(tgt_rgb.unsqueeze(0)).squeeze(0)  # [H,W]
    
        # LoRA prediction
        pred_rgb_lora = flux_predict(gen, image, args.infer_steps, args.strength)
        if pred_rgb_lora is None:
            print(f"  [{i+1}/{len(items)}] generation failed, skipping")
            continue
    
        pred_depth_lora = mapper.rgb_to_depth(pred_rgb_lora).squeeze(0)
        all_pred_depths.append(pred_depth_lora.flatten())
        all_gt_depths.append(gt_depth.flatten())
    
        m_lora = evaluate_depth(
            pred_depth_lora.unsqueeze(0).unsqueeze(0),
            gt_depth.unsqueeze(0).unsqueeze(0),
        )
        all_metrics_lora.append(m_lora)

        # Save visualization
        vis_dir = OUT / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        pred_rgb_pil = Image.fromarray(
            (pred_rgb_lora[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        )
        pred_rgb_pil.save(vis_dir / f"{i:03d}_pred_lora.png")

        print(f"  [{i+1}/{len(items)}] AbsRel={m_lora['AbsRel']:.3f} "
              f"RMSE={m_lora['RMSE']:.3f} δ1={m_lora['δ<1.25']:.3f}", flush=True)

        per_sample.append({
            "index": i,
            "image": str(item["image"]),
            "metrics": {k: float(v) for k, v in m_lora.items()},
        })

    # ── Aggregate ──────────────────────────────────────────────────────
    if not all_metrics_lora:
        print("ERROR: No successful predictions")
        return

    agg = {
        k: float(np.mean([m[k] for m in all_metrics_lora]))
        for k in all_metrics_lora[0].keys()
    }

    print(f"\n{'='*60}")
    print(f"Depth Gate Evaluation — {len(all_metrics_lora)} samples ({split_used} split)")
    print(f"{'='*60}")
    print(f"LoRA checkpoint: {lora_path}")
    print(f"Inference: steps={args.infer_steps}, strength={args.strength}, res={args.res}")
    print(f"\n{'Metric':<20} {'Mean':>10} {'± Std':>10}")
    print(f"{'-'*40}")
    for k, v in agg.items():
        std = float(np.std([m[k] for m in all_metrics_lora], ddof=1))
        print(f"{k:<20} {v:>10.4f} ±{std:>9.4f}")

    # Pearson correlation (key gate criterion)
    if all_pred_depths:
        preds_cat = torch.cat(all_pred_depths)
        gts_cat = torch.cat(all_gt_depths)
        mean_p = preds_cat.mean()
        mean_g = gts_cat.mean()
        cov = ((preds_cat - mean_p) * (gts_cat - mean_g)).mean()
        r = cov / (preds_cat.std() * gts_cat.std() + 1e-8)
        print(f"\nPearson r (global): {r.item():.4f}")
        agg["pearson_r"] = r.item()

    # Save results
    result = {
        "lora_checkpoint": str(lora_path),
        "split": split_used,
        "n_samples": len(all_metrics_lora),
        "inference": {
            "steps": args.infer_steps,
            "strength": args.strength,
            "resolution": args.res,
        },
        "aggregate": agg,
        "per_sample": per_sample,
    }
    result_path = OUT / "depth_gate_eval.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
