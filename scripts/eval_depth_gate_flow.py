#!/usr/bin/env python3
"""Phase 2 depth gate evaluation — flow-matching matched inference.

Uses the SAME flow matching Euler integration as training:
    z_{t+dt} = z_t + dt * v_θ(z_t, t, c)
where v_θ predicts the velocity from input_latents → target_latents.

This avoids the schedule mismatch between training (custom flow matching)
and inference (diffusers img2img pipeline).

Usage:
    python scripts/eval_depth_gate_flow.py \
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


def load_v3_depth_samples(data_dir: Path, split: str = "val", max_samples: int = 50):
    """Load depth samples from v3 format directory."""
    img_dir = data_dir / "images"
    tgt_dir = data_dir / "targets"
    prm_dir = data_dir / "prompts"
    if not img_dir.exists():
        print(f"WARNING: No images/ at {data_dir}")
        return []
    items = []
    for img_path in sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg")):
        tgt_path = tgt_dir / (img_path.stem + ".png")
        if tgt_path.exists():
            # Load per-sample prompt if available
            prm_path = prm_dir / (img_path.stem + ".txt")
            prompt = prm_path.read_text().strip() if prm_path.exists() else None
            items.append({"image": img_path, "target": tgt_path, "prompt": prompt})
        if len(items) >= max_samples:
            break
    return items


@torch.no_grad()
def flow_match_inference(
    pipe,
    model_config,
    input_image: Image.Image,
    prompt: str,
    n_steps: int = 20,
    resolution: int = 512,
) -> torch.Tensor:
    """Run flow matching Euler integration from input_latents.

    Matches training: z_{t+dt} = z_t + dt * v_θ(z_t, t, c)
    where v_θ ≈ target_latents - input_latents.
    """
    device = pipe.transformer.device if hasattr(pipe.transformer, 'device') else next(pipe.transformer.parameters()).device
    model_dtype = next(pipe.transformer.parameters()).dtype

    # Resize and preprocess input
    img_resized = input_image.resize((resolution, resolution), Image.LANCZOS)
    img_tensor = torch.from_numpy(np.array(img_resized).astype(np.float32)).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W] in [0, 1]

    # ── VAE encode input image → input_latents ──────────────────────
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype

    input_pixels = img_tensor * 2.0 - 1.0  # [0,1] → [-1,1]
    with torch.no_grad():
        input_latents = vae_encode(vae, input_pixels.to(vae_dtype), model_config)

    # ── Encode text (FLUX: dual CLIP+T5) ────────────────────────────
    # Temporarily cast text encoders to float32 for MPS stability
    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    original_dtypes = []
    for enc in encoders:
        if enc is not None:
            original_dtypes.append(enc.dtype)
            enc.to(torch.float32)
        else:
            original_dtypes.append(None)

    try:
        with torch.no_grad():
            text_outputs = pipe.encode_prompt(
                prompt=[prompt],
                prompt_2=[prompt],
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
    finally:
        for enc, orig_dt in zip(encoders, original_dtypes):
            if enc is not None and orig_dt is not None:
                enc.to(orig_dt)

    prompt_embeds = text_outputs[0]       # [1, seq_len, D_t5]
    pooled_prompt_embeds = text_outputs[1] # [1, D_clip]
    text_ids = text_outputs[2] if len(text_outputs) > 2 else None  # [seq_len, 3] | None

    # ── Euler integration: z starts at input_latents, integrates toward target ──
    z = input_latents.clone()
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step * dt
        t_tensor = torch.tensor([t_val], device=device, dtype=model_dtype)

        # Pack latents for FLUX: [B, 16, H, W] -> [B, (H/2)*(W/2), 64]
        latent_h, latent_w = z.shape[-2], z.shape[-1]
        latent_channels = z.shape[1]

        if latent_channels == 16:
            z_packed = _pack_latents(z.detach(), 1, latent_channels, latent_h, latent_w)
            packed_h, packed_w = latent_h // 2, latent_w // 2
            img_ids = _prepare_latent_image_ids(1, packed_h, packed_w, device, z.dtype)
        else:
            z_packed = z.detach()
            img_ids = None

        # Forward pass through transformer
        with torch.no_grad():
            out = pipe.transformer(
                hidden_states=z_packed.to(model_dtype),
                timestep=t_tensor,
                encoder_hidden_states=prompt_embeds.to(model_dtype),
                pooled_projections=pooled_prompt_embeds.to(model_dtype) if pooled_prompt_embeds is not None else torch.zeros(1, 768, device=device, dtype=model_dtype),
                txt_ids=text_ids.to(model_dtype) if text_ids is not None else None,
                img_ids=img_ids.to(model_dtype) if img_ids is not None else None,
                return_dict=False,
            )
        v = out[0] if isinstance(out, (tuple, list)) else out.sample

        # Unpack if FLUX: [B, (H/2)*(W/2), 64] -> [B, 16, H, W]
        if latent_channels == 16:
            v = _unpack_latents(v, latent_h, latent_w)

        # Euler step
        z = z + dt * v

    # ── VAE decode → RGB ────────────────────────────────────────────
    decoded = vae_decode(vae, z.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype), model_config)
    rgb = (decoded + 1.0) / 2.0  # [-1,1] → [0,1]
    return rgb.clamp(0, 1)


def _pack_latents(latents, batch_size, num_channels, height, width):
    """Pack VAE latents for FLUX transformer. [B, C, H, W] → [B, (H/2)*(W/2), C*4]"""
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


def _unpack_latents(latents, height, width):
    """Unpack FLUX transformer output. [B, (H/2)*(W/2), C*4] → [B, C, H, W]"""
    batch_size, num_patches, channels = latents.shape
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, height, width)
    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """Build FLUX latent position IDs [H*W, 3]."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids.to(device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser(description="Depth gate eval (flow matching inference)")
    ap.add_argument("--model", default="/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell")
    ap.add_argument("--lora-dir", type=str, required=True)
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data/vb_snvc_train_v3/depth"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output/depth_gate_flow_eval"))
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--n-steps", type=int, default=20,
                    help="Euler integration steps (more = better quality)")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    mapper = DepthRGBMapper(colormap="cube_edge")

    # ── 1) Load FLUX.1 + LoRA via BaseGenerator ─────────────────────
    lora_path = Path(args.lora_dir)
    if not lora_path.exists():
        print(f"ERROR: LoRA dir not found: {lora_path}")
        return

    print(f"[1/3] Loading FLUX.1 + LoRA on {args.device} …", flush=True)
    t0 = time.time()
    gen = BaseGenerator(
        model_name=args.model,
        lora_path=str(lora_path),
        device=args.device,
        dtype=torch.bfloat16,
    )
    pipe = gen._t2i_pipe
    model_config = resolve_model_config(args.model)
    print(f"      Loaded in {time.time()-t0:.1f}s", flush=True)

    # ── 2) Load depth samples ──────────────────────────────────────────
    data_dir = Path(args.data)
    items = load_v3_depth_samples(data_dir, split="val", max_samples=args.n_samples)
    split_used = "val"
    if len(items) < 3:
        items = load_v3_depth_samples(data_dir, split="train", max_samples=args.n_samples)
        split_used = "train"
    print(f"[2/3] {len(items)} samples ({split_used} split), {args.n_steps} Euler steps", flush=True)

    if not items:
        print("ERROR: No depth samples found")
        return

    # ── 3) Run flow matching inference ─────────────────────────────────
    DEFAULT_PROMPT = "Generate a depth map of this image. Use a power-transform color encoding where R,G,B encode metric depth values."
    print(f"[3/3] Flow matching inference …", flush=True)

    per_sample = []
    all_metrics = []
    all_pred_depths = []
    all_gt_depths = []

    for i, item in enumerate(items):
        image = Image.open(item["image"]).convert("RGB")

        # Load GT depth target (RGB encoded)
        tgt_rgb = torch.from_numpy(
            np.array(Image.open(item["target"]).convert("RGB")).astype(np.float32)
            .transpose(2, 0, 1)
        ) / 255.0
        gt_depth = mapper.rgb_to_depth(tgt_rgb.unsqueeze(0)).squeeze(0)  # [H,W]

        # Use per-sample prompt or default
        prompt = item.get("prompt") or DEFAULT_PROMPT

        # Flow matching inference
        t_step = time.time()
        pred_rgb = flow_match_inference(
            pipe, model_config, image, prompt,
            n_steps=args.n_steps, resolution=args.res,
        )  # [1, 3, H, W] in [0, 1]
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

        # Also save GT for comparison
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

    # ── Aggregate ──────────────────────────────────────────────────────
    if not all_metrics:
        print("ERROR: No successful predictions")
        return

    agg = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0].keys()
    }

    print(f"\n{'='*60}")
    print(f"Depth Gate Eval (Flow Matching) — {len(all_metrics)} samples")
    print(f"{'='*60}")
    print(f"LoRA: {lora_path}")
    print(f"Euler steps: {args.n_steps}, resolution: {args.res}")
    print(f"\n{'Metric':<20} {'Mean':>10} {'± Std':>10}")
    print(f"{'-'*40}")
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

    # Save results
    result = {
        "lora_checkpoint": str(lora_path),
        "split": split_used,
        "n_samples": len(all_metrics),
        "inference": {"method": "flow_matching_euler", "steps": args.n_steps, "resolution": args.res},
        "aggregate": agg,
        "per_sample": per_sample,
    }
    result_path = OUT / "depth_gate_flow_eval.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults → {result_path}")


if __name__ == "__main__":
    main()
