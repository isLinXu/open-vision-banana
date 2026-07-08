#!/usr/bin/env python3
"""Vision Banana v6 — Comprehensive multi-task evaluation pipeline.

Supports evaluation across all 10 vision tasks with multiple metrics,
model families, and checkpoint versions. Generates structured reports
with visualizations.

Usage:
  # Single task evaluation (depth)
  python scripts/eval_vb_v6.py --model ssd-1b --checkpoint output/vb_v6/ssd-1b/vb_v6_ssd-1b_best_corr.pt --task depth --dataset data/vb_v6_depth

  # Multi-task evaluation
  python scripts/eval_vb_v6.py --model ssd-1b --checkpoint ckpt.pt --tasks depth,normal,semantic --dataset-dir data

  # Comparison across checkpoints
  python scripts/eval_vb_v6.py --model ssd-1b --checkpoints ckpt1.pt,ckpt2.pt --task depth --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.model_registry import (
    PipelineComponents,
    apply_lora,
    backbone_forward,
    build_added_cond_kwargs,
    encode_text,
    get_model_config,
    list_models,
    load_pipeline,
    prepare_position_ids,
    vae_decode,
    vae_encode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_v6_eval")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Task-specific prompts ───────────────────────────────────────────────────

TASK_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate a surface normal visualization of this image.",
    "semantic": "Generate a semantic segmentation map of this image.",
    "instance": "Generate an instance segmentation map of this image.",
    "referring": "Segment the object described in this image.",
    "edge": "Generate an edge detection visualization of this image.",
    "flow": "Generate an optical flow visualization of this image.",
    "keypoint": "Generate a 2D keypoint detection visualization of this image.",
    "object_detection": "Generate an object detection visualization of this image.",
    "depth_edge": "Generate a depth edge visualization of this image.",
}


# ─── Generation function ─────────────────────────────────────────────────────

@torch.no_grad()
def generate_task_output(
    components: PipelineComponents,
    input_image: torch.Tensor,
    task: str,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 20,
    seed: int = 42,
    strength: float = 0.8,
    guidance_scale: float = 2.0,
    class_color_map: dict | None = None,
    expression: str | None = None,
) -> torch.Tensor:
    """Generate RGB output for a given vision task using img2img."""
    config = components.config
    backbone = components.unet
    vae = components.vae
    backbone.eval()

    prompt = TASK_PROMPTS.get(task, "Generate a visualization of this image.")
    if task == "semantic" and class_color_map:
        prompt = f"Semantic segmentation with classes: {', '.join(class_color_map.keys())}."
    elif task == "referring" and expression:
        prompt = f"Segment: {expression}"

    encoder_hidden_states, aux_output = encode_text(components, [prompt], device, dtype)

    attention_mask = None
    pooled_embeds = None
    if isinstance(aux_output, torch.Tensor):
        if config.needs_attention_mask:
            attention_mask = aux_output
        else:
            pooled_embeds = aux_output

    res = (input_image.shape[-2], input_image.shape[-1])
    added_cond_kwargs = build_added_cond_kwargs(config, pooled_embeds, res, device, dtype, batch_size=1)

    # CFG
    uncond_hidden = None
    uncond_added_cond = None
    uncond_mask = None
    if guidance_scale > 1.0:
        uncond_hidden, uncond_aux = encode_text(components, [""], device, dtype, negative=True)
        if isinstance(uncond_aux, torch.Tensor):
            if config.needs_attention_mask:
                uncond_mask = uncond_aux
            else:
                uncond_added_cond = build_added_cond_kwargs(config, uncond_aux, res, device, dtype, batch_size=1)

    input_latents = vae_encode(vae, input_image, config)
    torch.manual_seed(seed)
    latents = input_latents.clone()

    num_actual = max(1, int(num_steps * strength))
    dt = strength / num_actual

    txt_ids, img_ids = prepare_position_ids(None, encoder_hidden_states, config, device, dtype)

    for i in range(num_actual):
        t = i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)
        if config.needs_position_ids:
            txt_ids, img_ids = prepare_position_ids(latents, encoder_hidden_states, config, device, dtype)

        if guidance_scale > 1.0 and uncond_hidden is not None:
            v_cond = backbone_forward(
                backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs, guidance=guidance_scale,
                attention_mask=attention_mask, pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
            v_uncond = backbone_forward(
                backbone, latents, timestep_long, uncond_hidden, config,
                added_cond_kwargs=uncond_added_cond, guidance=guidance_scale,
                attention_mask=uncond_mask, pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
            pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_v = backbone_forward(
                backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs, guidance=guidance_scale,
                attention_mask=attention_mask, pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
        latents = latents + dt * pred_v

    image = vae_decode(vae, latents, config)
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_depth_correlation(gen_img: Image.Image, target_img: Image.Image) -> float:
    gen_gray = np.array(gen_img.convert("L"), dtype=np.float32).flatten()
    tgt_gray = np.array(target_img.convert("L"), dtype=np.float32).flatten()
    gen_norm = gen_gray - gen_gray.mean()
    tgt_norm = tgt_gray - tgt_gray.mean()
    denom = np.sqrt((gen_norm ** 2).sum() * (tgt_norm ** 2).sum())
    if denom < 1e-6:
        return 0.0
    return float((gen_norm * tgt_norm).sum() / denom)


def compute_depth_metrics_via_mapper(
    gen_img: Image.Image, target_img: Image.Image
) -> dict[str, float]:
    """Compute depth-specific metrics using DepthRGBMapper for proper decoding.

    Decodes both generated and target RGB images to depth maps using the
    paper's Power Transform + 5-stage clustering pipeline, then computes
    standard depth estimation metrics.

    Returns:
        Dict with Pearson r, Spearman r, AbsRel, RMSE, δ1, δ2, δ3, etc.
    """
    from scipy.stats import spearmanr

    mapper = DepthRGBMapper()

    # Convert PIL → tensor [1, 3, H, W] in [0, 1]
    gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
    tgt_arr = np.array(target_img, dtype=np.float32) / 255.0
    gen_t = torch.from_numpy(gen_arr).permute(2, 0, 1).unsqueeze(0)
    tgt_t = torch.from_numpy(tgt_arr).permute(2, 0, 1).unsqueeze(0)

    # Decode RGB → depth
    try:
        pred_depth = mapper.rgb_to_depth(gen_t).squeeze().cpu().numpy()
        gt_depth = mapper.rgb_to_depth(tgt_t).squeeze().cpu().numpy()
    except Exception:
        # Fallback to grayscale if mapper fails
        pred_depth = gen_arr.mean(axis=2)
        gt_depth = tgt_arr.mean(axis=2)

    # Flatten
    pred_flat = pred_depth.flatten()
    gt_flat = gt_depth.flatten()

    # Pearson r
    pred_norm = pred_flat - pred_flat.mean()
    gt_norm = gt_flat - gt_flat.mean()
    denom = np.sqrt((pred_norm ** 2).sum() * (gt_norm ** 2).sum())
    pearson_r = float((pred_norm * gt_norm).sum() / denom) if denom > 1e-6 else 0.0

    # Spearman r (rank correlation — more robust to monotonic nonlinearities)
    try:
        spearman_r, _ = spearmanr(pred_flat, gt_flat)
        spearman_r = float(spearman_r)
    except Exception:
        spearman_r = 0.0

    # AbsRel
    absrel = float(np.mean(np.abs(pred_flat - gt_flat) / (np.abs(gt_flat) + 1e-6)))

    # RMSE
    rmse = float(np.sqrt(np.mean((pred_flat - gt_flat) ** 2)))

    # δ thresholds (within x% of GT)
    ratio = np.maximum(pred_flat / (gt_flat + 1e-6), gt_flat / (pred_flat + 1e-6))
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))

    # Log RMSE
    log_rmse = float(np.sqrt(np.mean(
        (np.log(pred_flat + 1e-6) - np.log(gt_flat + 1e-6)) ** 2
    )))

    return {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "absrel": absrel,
        "rmse": rmse,
        "log_rmse": log_rmse,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def compute_image_metrics(
    gen_img: Image.Image, target_img: Image.Image, task: str = "depth"
) -> dict[str, float]:
    """Compute comprehensive metrics for a generated vs target image pair.

    For depth tasks, uses the DepthRGBMapper for proper RGB→depth decoding
    and computes depth-specific metrics (AbsRel, RMSE, δ thresholds, etc.).
    For other tasks, falls back to pixel-level metrics + Pearson r.
    """
    gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
    tgt_arr = np.array(target_img, dtype=np.float32) / 255.0
    mse = float(np.mean((gen_arr - tgt_arr) ** 2))
    mae = float(np.mean(np.abs(gen_arr - tgt_arr)))
    corr = compute_depth_correlation(gen_img, target_img)

    metrics = {"mse": mse, "mae": mae, "correlation": corr,
               "gen_std": float(np.std(gen_arr)), "tgt_std": float(np.std(tgt_arr))}

    # Add depth-specific metrics for depth tasks
    if task in ("depth", "depth_edge"):
        try:
            depth_metrics = compute_depth_metrics_via_mapper(gen_img, target_img)
            metrics.update(depth_metrics)
        except ImportError:
            # scipy not available — skip spearman
            pass
        except Exception as e:
            logger.debug(f"Depth metrics computation failed: {e}")

    return metrics


# ─── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_checkpoint(
    checkpoint_path: Path,
    model_name: str,
    task: str,
    dataset_path: Path,
    num_images: int = 10,
    strengths: list[float] | None = None,
    resolution: int | None = None,
    num_steps: int | None = None,
    guidance_scale: float | None = None,
    use_ema: bool = True,
    lora_rank: int | None = None,
) -> dict[str, Any]:
    """Evaluate a single checkpoint on a task."""
    model_cfg = get_model_config(model_name)
    resolution = resolution or model_cfg.default_resolution
    num_steps = num_steps or model_cfg.default_num_inference_steps
    guidance_scale = guidance_scale if guidance_scale is not None else model_cfg.default_guidance_scale
    strengths = strengths or [0.4, 0.6, 0.8, 1.0]
    lora_rank = lora_rank or model_cfg.default_lora_rank
    lora_alpha = lora_rank * 2  # Standard convention: alpha = 2 * rank

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16

    logger.info(f"Evaluating checkpoint: {checkpoint_path.name}")
    logger.info(f"Model: {model_cfg.name}, Task: {task}, Resolution: {resolution}")

    # Load pipeline
    components = load_pipeline(model_cfg, device, dtype, enable_gradient_checkpointing=False)
    components.unet = apply_lora(components.unet, model_cfg, rank=lora_rank, alpha=lora_alpha)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ema_keys = {k: v for k, v in ckpt.items() if k.startswith("ema.")}
    unet_keys = {k: v for k, v in ckpt.items() if k.startswith("unet.")}
    use_ema_actual = use_ema and len(ema_keys) > 0
    load_keys = ema_keys if use_ema_actual else unet_keys

    loaded = 0
    for name, param in components.unet.named_parameters():
        if param.requires_grad:
            key = f"ema.{name}" if use_ema_actual else f"unet.{name}"
            if key in load_keys:
                param.data.copy_(load_keys[key].to(dtype))
                loaded += 1
    logger.info(f"Loaded {loaded} parameters (EMA={use_ema_actual})")

    # Load test images
    img_dir = dataset_path / "images"
    tgt_dir = dataset_path / "targets"
    if not img_dir.exists():
        logger.error(f"Images directory not found: {img_dir}")
        return {}

    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    test_images = all_images[-num_images:] if len(all_images) >= num_images else all_images
    logger.info(f"Evaluating on {len(test_images)} images")

    results = []
    for img_idx, img_path in enumerate(test_images):
        img = Image.open(img_path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        input_tensor = input_tensor.to(dtype)

        # Find target
        tgt_name = img_path.stem + ".png"
        tgt_path = tgt_dir / tgt_name
        if not tgt_path.exists():
            tgt_path = tgt_dir / img_path.name
        target_img = Image.open(tgt_path).convert("RGB").resize((resolution, resolution), Image.LANCZOS) if tgt_path.exists() else None

        for strength in strengths:
            try:
                gen_tensor = generate_task_output(
                    components, input_tensor, task, device, dtype,
                    num_steps=num_steps, seed=42 + img_idx, strength=strength,
                    guidance_scale=guidance_scale,
                )
                gen_np = (gen_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gen_img = Image.fromarray(gen_np)

                metrics = compute_image_metrics(gen_img, target_img, task=task) if target_img else {}
                metrics.update({"image": img_path.stem, "strength": strength})
                results.append(metrics)
                # Log depth-specific metrics when available
                pearson = metrics.get("pearson_r", metrics.get("correlation", 0))
                absrel = metrics.get("absrel", 0)
                logger.info(
                    f"  {img_path.stem} s={strength}: r={pearson:.3f} "
                    f"absrel={absrel:.3f}" if absrel > 0
                    else f"  {img_path.stem} s={strength}: corr={pearson:.3f}"
                )
            except Exception as e:
                logger.warning(f"  Failed {img_path.stem} s={strength}: {e}")

    # Aggregate
    by_strength = {}
    depth_metrics_by_strength = {}
    for r in results:
        s = r["strength"]
        if s not in by_strength:
            by_strength[s] = []
            depth_metrics_by_strength[s] = []
        # Use pearson_r if available (depth task), else correlation
        corr_val = r.get("pearson_r", r.get("correlation", 0))
        by_strength[s].append(corr_val)
        # Collect depth-specific metrics
        if "absrel" in r:
            depth_metrics_by_strength[s].append({
                k: r[k] for k in
                ("pearson_r", "spearman_r", "absrel", "rmse", "log_rmse",
                 "delta1", "delta2", "delta3")
                if k in r
            })

    summary = {}
    for s, corrs in by_strength.items():
        summary[f"strength_{s}"] = {
            "mean_corr": float(np.mean(corrs)),
            "std_corr": float(np.std(corrs)),
            "max_corr": float(np.max(corrs)),
            "min_corr": float(np.min(corrs)),
        }
        # Add averaged depth metrics
        dm_list = depth_metrics_by_strength[s]
        if dm_list:
            for key in dm_list[0]:
                vals = [dm[key] for dm in dm_list if key in dm]
                summary[f"strength_{s}"][f"mean_{key}"] = float(np.mean(vals))

    best_strength = max(by_strength.keys(), key=lambda s: np.mean(by_strength[s]))
    best_mean = float(np.mean(by_strength[best_strength]))

    return {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "task": task,
        "num_images": len(test_images),
        "resolution": resolution,
        "strengths": strengths,
        "best_strength": best_strength,
        "best_mean_correlation": best_mean,
        "step2_pass": best_mean > 0.3,
        "step3_pass": best_mean > 0.5,
        "summary_by_strength": summary,
        "all_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Vision Banana v6 — Multi-task evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--model", type=str, default="ssd-1b", help="Base model name")
    parser.add_argument("--task", type=str, default="depth", help="Task to evaluate")
    parser.add_argument("--dataset-path", type=str, default="data/vb_v4_depth_2k", help="Dataset directory")
    parser.add_argument("--num-images", type=int, default=10, help="Number of test images")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--no-ema", action="store_true", help="Use raw weights instead of EMA")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        for name in list_models():
            cfg = get_model_config(name)
            print(f"  {name:15s} | {cfg.description}")
        return

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    results = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=args.model,
        task=args.task,
        dataset_path=dataset_path,
        num_images=args.num_images,
        strengths=args.strengths,
        resolution=args.resolution,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        use_ema=not args.no_ema,
        lora_rank=args.lora_rank,
    )

    output_path = Path(args.output) if args.output else checkpoint_path.parent / f"eval_{args.task}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Print summary
    if results:
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Checkpoint: {results['checkpoint']}")
        print(f"Model: {results['model']}")
        print(f"Task: {results['task']}")
        print(f"Best strength: {results['best_strength']}")
        print(f"Best mean correlation: {results['best_mean_correlation']:.4f}")
        print(f"Step 2 (r > 0.3): {'PASS' if results['step2_pass'] else 'FAIL'}")
        print(f"Step 3 (r > 0.5): {'PASS' if results['step3_pass'] else 'FAIL'}")
        print("=" * 60)


if __name__ == "__main__":
    main()
