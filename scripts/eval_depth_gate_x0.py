#!/usr/bin/env python3
"""Phase 2 depth gate — single-step x0 inference.

Based on the learned protocol from previous experiments:
  - Bypass scheduler iteration entirely
  - Use single forward pass at t=0 to predict velocity
  - x0_pred = input_latents + v_pred (single-step)

This avoids the divergence issue with multi-step Euler integration
when the velocity field hasn't fully converged.

Usage:
    python scripts/eval_depth_gate_x0.py \
        --lora-dir output/flux_depth_gate/checkpoint-200 \
        --data data/vb_snvc_train_v3/depth \
        --n-samples 10
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
from open_vision_banana.model_registry import resolve_model_config, vae_decode, vae_encode


def load_v3_depth_samples(data_dir: Path, max_samples: int = 50):
    """Load depth samples from v3 format directory."""
    img_dir = data_dir / "images"
    tgt_dir = data_dir / "targets"
    prm_dir = data_dir / "prompts"
    if not img_dir.exists():
        return []
    items = []
    for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
        tgt_path = tgt_dir / (img_path.stem + ".png")
        if tgt_path.exists():
            prm_path = prm_dir / (img_path.stem + ".txt")
            prompt = prm_path.read_text().strip() if prm_path.exists() else None
            items.append({"image": img_path, "target": tgt_path, "prompt": prompt})
        if len(items) >= max_samples:
            break
    return items


@torch.no_grad()
def single_step_x0(
    pipe,
    model_config,
    input_image: Image.Image,
    prompt: str,
    resolution: int = 512,
) -> torch.Tensor:
    """Single-step x0 prediction at t=0.

    1. VAE encode input → input_latents
    2. Predict velocity v at t=0 (noisy = input_latents)
    3. x0_pred = input_latents + v
    4. VAE decode x0_pred → RGB
    """
    device = next(pipe.transformer.parameters()).device
    model_dtype = next(pipe.transformer.parameters()).dtype

    # Preprocess input
    img_resized = input_image.resize((resolution, resolution), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img_resized).astype(np.float32)).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # VAE encode
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype
    input_pixels = img_tensor * 2.0 - 1.0  # [0,1] → [-1,1]
    input_latents = vae_encode(vae, input_pixels.to(vae_dtype), model_config)

    # Encode text
    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    orig_dtypes = []
    for enc in encoders:
        if enc is not None:
            orig_dtypes.append(enc.dtype)
            enc.to(torch.float32)
        else:
            orig_dtypes.append(None)
    try:
        text_outputs = pipe.encode_prompt(
            prompt=[prompt], prompt_2=[prompt], device=device,
            num_images_per_prompt=1, max_sequence_length=512,
        )
    finally:
        for enc, dt in zip(encoders, orig_dtypes):
            if enc is not None and dt is not None:
                enc.to(dt)

    prompt_embeds = text_outputs[0]
    pooled_prompt_embeds = text_outputs[1]
    text_ids = text_outputs[2] if len(text_outputs) > 2 else None

    # Pack latents for FLUX
    B, C, H, W = input_latents.shape
    z_packed = input_latents.view(B, C, H // 2, 2, W // 2, 2)
    z_packed = z_packed.permute(0, 2, 4, 1, 3, 5)
    z_packed = z_packed.reshape(B, (H // 2) * (W // 2), C * 4)

    # Position IDs
    img_ids = torch.zeros(H // 2, W // 2, 3, device=device, dtype=model_dtype)
    img_ids[..., 1] += torch.arange(H // 2, device=device)[:, None]
    img_ids[..., 2] += torch.arange(W // 2, device=device)[None, :]
    img_ids = img_ids.reshape(-1, 3)

    # Single forward at t=0
    t_tensor = torch.tensor([0.0], device=device, dtype=model_dtype)
    out = pipe.transformer(
        hidden_states=z_packed.to(model_dtype),
        timestep=t_tensor,
        encoder_hidden_states=prompt_embeds.to(model_dtype),
        pooled_projections=(
            pooled_prompt_embeds.to(model_dtype)
            if pooled_prompt_embeds is not None
            else torch.zeros(B, 768, device=device, dtype=model_dtype)
        ),
        txt_ids=text_ids.to(model_dtype) if text_ids is not None else None,
        img_ids=img_ids,
        return_dict=False,
    )
    v = out[0] if isinstance(out, (tuple, list)) else out.sample

    # Unpack velocity: [B, seq_len, C*4] → [B, C, H, W]
    # C is the original latent channels (16 for FLUX), not packed channels (64)
    C_latent = C  # 16
    v = v.view(B, H // 2, W // 2, C_latent, 2, 2)
    v = v.permute(0, 3, 1, 4, 2, 5)
    v = v.reshape(B, C_latent, H, W)

    # Single-step x0 prediction
    x0_pred = input_latents + v  # x0 = input + v (at t=0)

    # VAE decode
    decoded = vae_decode(
        vae,
        x0_pred.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype),
        model_config,
    )
    rgb = (decoded + 1.0) / 2.0
    return rgb.clamp(0, 1)


def main():
    ap = argparse.ArgumentParser(description="Depth gate eval (single-step x0)")
    ap.add_argument("--model", default="/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell")
    ap.add_argument("--lora-dir", type=str, required=True)
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data/vb_snvc_train_v3/depth"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output/depth_gate_x0_eval"))
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    mapper = DepthRGBMapper(colormap="cube_edge")

    # Load model
    lora_path = Path(args.lora_dir)
    if not lora_path.exists():
        print(f"ERROR: LoRA dir not found: {lora_path}")
        return

    print(f"[1/3] Loading FLUX.1 + LoRA on {args.device} …", flush=True)
    t0 = time.time()
    gen = BaseGenerator(
        model_name=args.model, lora_path=str(lora_path),
        device=args.device, dtype=torch.bfloat16,
    )
    pipe = gen._t2i_pipe
    model_config = resolve_model_config(args.model)
    print(f"      Loaded in {time.time() - t0:.1f}s", flush=True)

    # Load samples
    data_dir = Path(args.data)
    items = load_v3_depth_samples(data_dir, max_samples=args.n_samples)
    print(f"[2/3] {len(items)} samples", flush=True)
    if not items:
        print("ERROR: No depth samples found")
        return

    # Evaluate
    DEFAULT_PROMPT = "Generate a depth map of this image. Use a power-transform color encoding where R,G,B encode metric depth values."
    print(f"[3/3] Single-step x0 inference …", flush=True)

    per_sample = []
    all_metrics = []
    all_pred_depths = []
    all_gt_depths = []

    for i, item in enumerate(items):
        image = Image.open(item["image"]).convert("RGB")
        prompt = item.get("prompt") or DEFAULT_PROMPT

        # GT depth
        tgt_rgb = torch.from_numpy(
            np.array(Image.open(item["target"]).convert("RGB")).astype(np.float32)
            .transpose(2, 0, 1)
        ) / 255.0
        gt_depth = mapper.rgb_to_depth(tgt_rgb.unsqueeze(0)).squeeze(0)

        # Single-step x0 prediction
        t_step = time.time()
        pred_rgb = single_step_x0(pipe, model_config, image, prompt, resolution=args.res)
        elapsed = time.time() - t_step

        pred_depth = mapper.rgb_to_depth(pred_rgb).squeeze(0)
        gt_depth = gt_depth.to(pred_depth.device)

        all_pred_depths.append(pred_depth.flatten())
        all_gt_depths.append(gt_depth.flatten())

        m = evaluate_depth(
            pred_depth.unsqueeze(0).unsqueeze(0),
            gt_depth.unsqueeze(0).unsqueeze(0),
        )
        all_metrics.append(m)

        # Save visualization
        vis_dir = OUT / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        pred_pil = Image.fromarray(
            (pred_rgb[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        )
        pred_pil.save(vis_dir / f"{i:03d}_pred.png")
        gt_vis = Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        )
        gt_vis.save(vis_dir / f"{i:03d}_gt.png")

        print(f"  [{i+1}/{len(items)}] AbsRel={m['AbsRel']:.3f} "
              f"RMSE={m['RMSE']:.3f} δ1={m['δ<1.25']:.3f} ({elapsed:.1f}s)", flush=True)

        per_sample.append({
            "index": i,
            "image": str(item["image"]),
            "metrics": {k: float(v) for k, v in m.items()},
        })

    # Aggregate
    if not all_metrics:
        print("ERROR: No successful predictions")
        return

    agg = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0].keys()}

    print(f"\n{'=' * 60}")
    print(f"Depth Gate Eval (Single-step x0) — {len(all_metrics)} samples")
    print(f"{'=' * 60}")
    print(f"LoRA: {lora_path}")
    print(f"\n{'Metric':<20} {'Mean':>10} {'± Std':>10}")
    print(f"{'-' * 40}")
    for k, v in agg.items():
        std = float(np.std([m[k] for m in all_metrics], ddof=1))
        print(f"{k:<20} {v:>10.4f} ±{std:>9.4f}")

    # Pearson correlation
    if all_pred_depths:
        preds_cat = torch.cat(all_pred_depths)
        gts_cat = torch.cat(all_gt_depths)
        mean_p, mean_g = preds_cat.mean(), gts_cat.mean()
        cov = ((preds_cat - mean_p) * (gts_cat - mean_g)).mean()
        r = cov / (preds_cat.std() * gts_cat.std() + 1e-8)
        print(f"\nPearson r (global): {r.item():.4f}")
        agg["pearson_r"] = r.item()

    result = {
        "lora_checkpoint": str(lora_path),
        "n_samples": len(all_metrics),
        "inference": {"method": "single_step_x0", "resolution": args.res},
        "aggregate": agg,
        "per_sample": per_sample,
    }
    result_path = OUT / "depth_gate_x0_eval.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults → {result_path}")


if __name__ == "__main__":
    main()
