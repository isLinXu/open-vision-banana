#!/usr/bin/env python3
"""Vision Banana v5 — Multi-backbone evaluation & visualization.

Loads a trained v5 LoRA checkpoint and runs comprehensive evaluation across
any supported base model:
  - Pearson r depth correlation on held-out images
  - Multi-strength sweep
  - Side-by-side visualization (input | generated | target)
  - Success criteria: Pearson r > 0.3 (Step 2), > 0.5 (Step 3)

Usage:
  python scripts/eval_vb_v5.py --model ssd-1b --checkpoint output/vb_v5/ssd-1b/vb_v5_ssd-1b_final.pt
  python scripts/eval_vb_v5.py --model sd15 --checkpoint path/to/ckpt --num-images 10
  python scripts/eval_vb_v5.py --list-models
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
logger = logging.getLogger("vb_v5_eval")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEPTH_DATA_DIR = DATA_DIR / "vb_v4_depth_2k"

DEPTH_PROMPT = "Generate a metric depth visualization of this image."


def compute_depth_correlation(gen_img, target_img):
    """Pearson r between generated and target grayscale."""
    gen_gray = np.array(gen_img.convert("L"), dtype=np.float32).flatten()
    tgt_gray = np.array(target_img.convert("L"), dtype=np.float32).flatten()
    gen_norm = gen_gray - gen_gray.mean()
    tgt_norm = tgt_gray - tgt_gray.mean()
    denom = np.sqrt((gen_norm ** 2).sum() * (tgt_norm ** 2).sum())
    if denom < 1e-6:
        return 0.0
    return float((gen_norm * tgt_norm).sum() / denom)


def compute_structural_similarity(gen_gray, tgt_gray):
    """Lightweight SSIM-like metric (no scikit-image dependency)."""
    gen = gen_gray.astype(np.float32)
    tgt = tgt_gray.astype(np.float32)
    mu_g, mu_t = gen.mean(), tgt.mean()
    var_g, var_t = gen.var(), tgt.var()
    cov_gt = ((gen - mu_g) * (tgt - mu_t)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu_g * mu_t + c1) * (2 * cov_gt + c2)) / \
           ((mu_g ** 2 + mu_t ** 2 + c1) * (var_g + var_t + c2))
    return float(ssim)


def compute_metrics(gen_img, target_img):
    """Compute comprehensive depth metrics."""
    gen_gray = np.array(gen_img.convert("L"), dtype=np.float32)
    tgt_gray = np.array(target_img.convert("L"), dtype=np.float32)
    gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
    tgt_arr = np.array(target_img, dtype=np.float32) / 255.0

    corr = compute_depth_correlation(gen_img, target_img)
    ssim = compute_structural_similarity(gen_gray, tgt_gray)
    mse = float(np.mean((gen_arr - tgt_arr) ** 2))
    mae = float(np.mean(np.abs(gen_arr - tgt_arr)))

    from scipy.ndimage import sobel
    gen_grad = sobel(gen_gray)
    tgt_grad = sobel(tgt_gray)
    grad_corr = compute_depth_correlation(
        Image.fromarray(gen_grad.astype(np.uint8)),
        Image.fromarray(tgt_grad.astype(np.uint8)),
    )

    return {
        "pearson_r": corr,
        "ssim": ssim,
        "mse": mse,
        "mae": mae,
        "gradient_corr": grad_corr,
        "gen_std": float(np.std(gen_arr)),
        "tgt_std": float(np.std(tgt_arr)),
    }


def create_comparison_figure(input_img, gen_img, target_img, metrics, title, save_path):
    """Create side-by-side comparison: input | generated | target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_img)
    axes[0].set_title("Input Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(gen_img)
    axes[1].set_title(
        f"Generated Depth\nPearson r={metrics['pearson_r']:.3f} | SSIM={metrics['ssim']:.3f}",
        fontsize=14,
    )
    axes[1].axis("off")

    axes[2].imshow(target_img)
    axes[2].set_title("Target Depth (Depth-Anything)", fontsize=14)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_summary_plot(all_results, save_path):
    """Create summary plot of Pearson r across strength values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strengths = sorted(set(r["strength"] for r in all_results))
    by_strength = {}
    for s in strengths:
        corrs = [r["metrics"]["pearson_r"] for r in all_results if r["strength"] == s]
        by_strength[s] = corrs

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(strengths))
    bp = ax.boxplot(
        [by_strength[s] for s in strengths],
        positions=positions,
        labels=[f"s={s}" for s in strengths],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")

    ax.axhline(y=0.3, color="orange", linestyle="--", label="Step 2 threshold (r>0.3)")
    ax.axhline(y=0.5, color="green", linestyle="--", label="Step 3 target (r>0.5)")
    ax.set_ylabel("Pearson Correlation r", fontsize=14)
    ax.set_xlabel("img2img Strength", fontsize=14)
    ax.set_title("Depth Structure Correlation vs. img2img Strength", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


@torch.no_grad()
def generate_depth_img2img(
    components: PipelineComponents,
    input_image: torch.Tensor,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 30,
    seed: int = 42,
    strength: float = 1.0,
    guidance_scale: float = 2.0,
):
    """TRUE img2img depth generation — works across all model families."""
    config = components.config
    backbone = components.unet
    vae = components.vae
    backbone.eval()

    # Encode text
    encoder_hidden_states, aux_output = encode_text(
        components, [prompt], device, dtype
    )

    # Handle aux_output (pooled_embeds, attention_mask, or HunyuanDiT tuple)
    attention_mask = None
    pooled_embeds = None
    if isinstance(aux_output, tuple):
        attention_mask = aux_output  # HunyuanDiT: (t5_hidden, clip_mask, t5_mask)
    elif isinstance(aux_output, torch.Tensor):
        if config.needs_attention_mask:
            attention_mask = aux_output
        else:
            pooled_embeds = aux_output

    # Build added_cond_kwargs
    res = (input_image.shape[-2], input_image.shape[-1])
    added_cond_kwargs = build_added_cond_kwargs(
        config, pooled_embeds, res, device, dtype, batch_size=1
    )

    # CFG setup
    uncond_hidden = None
    uncond_added_cond = None
    uncond_mask = None
    if guidance_scale > 1.0:
        uncond_hidden, uncond_aux = encode_text(
            components, [""], device, dtype, negative=True
        )
        if isinstance(uncond_aux, torch.Tensor):
            if config.needs_attention_mask:
                uncond_mask = uncond_aux
            else:
                uncond_added_cond = build_added_cond_kwargs(
                    config, uncond_aux, res, device, dtype, batch_size=1
                )

    # TRUE img2img: encode INPUT image as starting latent
    input_latents = vae_encode(vae, input_image, config)

    torch.manual_seed(seed)
    latents = input_latents.clone()

    # Flow matching: integrate from t=0 (input) toward t=strength (target)
    num_actual = max(1, int(num_steps * strength))
    dt = strength / num_actual

    for i in range(num_actual):
        t = i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)

        # Prepare position IDs if needed
        if config.needs_position_ids:
            txt_ids, img_ids = prepare_position_ids(
                latents, encoder_hidden_states, config, device, dtype
            )

        if guidance_scale > 1.0 and uncond_hidden is not None:
            v_cond = backbone_forward(
                backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs,
                guidance=guidance_scale,
                attention_mask=attention_mask,
                pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
            v_uncond = backbone_forward(
                backbone, latents, timestep_long, uncond_hidden, config,
                added_cond_kwargs=uncond_added_cond,
                guidance=guidance_scale,
                attention_mask=uncond_mask,
                pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
            pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_v = backbone_forward(
                backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs,
                guidance=guidance_scale,
                attention_mask=attention_mask,
                pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
        latents = latents + dt * pred_v

    image = vae_decode(vae, latents, config)
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Vision Banana v5 (multi-backbone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models (23): sd15, sd2, ssd-1b, sdxl, sdxl-turbo, flux-dev, flux-schnell,
  qwen-image, z-image, sana-600m, sana-1600m, sana-sprint, lumina2, chroma, flux2-dev,
  cogview4, aura-flow, sd3, pixart-alpha, pixart-sigma, kolors, hunyuandit,
  kandinsky3, sana-sprint
Use --list-models to see all options.
        """,
    )
    parser.add_argument("--checkpoint", type=str, required=False,
                        help="Path to .pt checkpoint")
    parser.add_argument("--model", type=str, default="ssd-1b",
                        help="Base model name (default: ssd-1b)")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models and exit")
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=None,
                        help="Eval resolution (default: model-specific)")
    parser.add_argument("--strengths", type=float, nargs="+",
                        default=[0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Inference steps (default: model-specific)")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG scale (default: model-specific)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 80)
        for name in list_models():
            cfg = get_model_config(name)
            print(f"  {name:15s} | {cfg.description}")
            print(f"  {'':15s} | family={cfg.family}, memory~{cfg.memory_budget_gb}GB, "
                  f"encoders={cfg.num_text_encoders}")
            print()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required (unless --list-models)")

    model_cfg = get_model_config(args.model)
    resolution = args.resolution or model_cfg.default_resolution
    num_steps = args.num_steps or model_cfg.default_num_inference_steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else model_cfg.default_guidance_scale

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("Vision Banana v5 — Multi-backbone Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Model: {model_cfg.model_id} ({model_cfg.name})")
    logger.info(f"Family: {model_cfg.family}, Encoders: {model_cfg.num_text_encoders}")
    logger.info(f"Device: {device}, Resolution: {resolution}")
    logger.info(f"Steps: {num_steps}, Guidance: {guidance_scale}")
    logger.info(f"Strengths: {args.strengths}")
    logger.info("=" * 60)

    # ─── Load model ───
    components = load_pipeline(model_cfg, device, dtype, enable_gradient_checkpointing=False)

    # Apply LoRA
    components.unet = apply_lora(
        components.unet, model_cfg,
        rank=model_cfg.default_lora_rank,
        alpha=model_cfg.default_lora_alpha,
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path.name}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    loaded_keys = []
    for name, param in components.unet.named_parameters():
        if param.requires_grad:
            key = f"unet.{name}"
            if key in ckpt:
                param.data.copy_(ckpt[key].to(dtype))
                loaded_keys.append(key)
    logger.info(f"Loaded {len(loaded_keys)} LoRA parameters from checkpoint")

    # ─── Collect test images ───
    all_images = sorted((DEPTH_DATA_DIR / "images").glob("*.jpg"))
    test_images = all_images[-args.num_images:] if len(all_images) >= args.num_images else all_images
    logger.info(f"Evaluating on {len(test_images)} test images")

    # ─── Run evaluation ───
    all_results = []
    summary_by_strength = {s: [] for s in args.strengths}

    for img_idx, img_path in enumerate(test_images):
        img = Image.open(img_path).convert("RGB").resize(
            (resolution, resolution), Image.LANCZOS
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        input_tensor = input_tensor.to(dtype)

        tgt_name = img_path.stem + ".png"
        tgt_path = DEPTH_DATA_DIR / "targets" / tgt_name
        if not tgt_path.exists():
            tgt_path = DEPTH_DATA_DIR / "targets" / img_path.name
        target_img = Image.open(tgt_path).convert("RGB").resize(
            (resolution, resolution), Image.LANCZOS
        )

        best_corr = -2.0
        best_result = None

        for strength in args.strengths:
            try:
                gen_img = generate_depth_img2img(
                    components,
                    input_tensor, DEPTH_PROMPT, device, dtype,
                    num_steps=num_steps,
                    seed=args.seed + img_idx,
                    strength=strength,
                    guidance_scale=guidance_scale,
                )

                metrics = compute_metrics(gen_img, target_img)
                all_results.append({
                    "image": img_path.stem,
                    "strength": strength,
                    "metrics": metrics,
                })
                summary_by_strength[strength].append(metrics["pearson_r"])

                if metrics["pearson_r"] > best_corr:
                    best_corr = metrics["pearson_r"]
                    best_result = {
                        "strength": strength,
                        "gen_img": gen_img,
                        "metrics": metrics,
                    }

                logger.info(
                    f"  {img_path.stem} s={strength}: "
                    f"r={metrics['pearson_r']:.3f} SSIM={metrics['ssim']:.3f} "
                    f"grad={metrics['gradient_corr']:.3f}"
                )

            except Exception as e:
                logger.warning(f"  Failed {img_path.stem} s={strength}: {e}")

        if best_result is not None:
            create_comparison_figure(
                img, best_result["gen_img"], target_img,
                best_result["metrics"],
                f"Best strength={best_result['strength']} ({model_cfg.name})",
                output_dir / "comparisons" / f"compare_{img_idx}_{img_path.stem}.png",
            )

    # ─── Summary ───
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)

    for strength in args.strengths:
        corrs = summary_by_strength[strength]
        if corrs:
            logger.info(
                f"  Strength {strength}: "
                f"mean r={np.mean(corrs):.4f} ± {np.std(corrs):.4f} "
                f"(max={np.max(corrs):.4f}, min={np.min(corrs):.4f})"
            )

    best_overall = max(
        (r["metrics"]["pearson_r"] for r in all_results),
        default=-1.0,
    )
    best_mean = max(
        (np.mean(summary_by_strength[s]) for s in args.strengths if summary_by_strength[s]),
        default=-1.0,
    )
    best_strength = max(
        args.strengths,
        key=lambda s: np.mean(summary_by_strength[s]) if summary_by_strength[s] else -1,
    )

    logger.info(f"\n  Best individual Pearson r: {best_overall:.4f}")
    logger.info(f"  Best mean Pearson r: {best_mean:.4f} (at strength={best_strength})")

    step2_pass = best_mean > 0.3
    step3_pass = best_mean > 0.5
    logger.info(f"\n  Step 2 (r > 0.3): {'PASS' if step2_pass else 'FAIL'}")
    logger.info(f"  Step 3 (r > 0.5): {'PASS' if step3_pass else 'FAIL'}")

    if step3_pass:
        logger.info("  ✓ SUCCESS: Model generates structurally coherent depth maps!")
    elif step2_pass:
        logger.info("  ~ PARTIAL: Basic depth structure detected, needs longer training.")
    else:
        logger.info("  ✗ FAIL: Model has not learned depth structure yet.")

    # Save results JSON
    results_json = {
        "checkpoint": str(checkpoint_path),
        "model": model_cfg.model_id,
        "model_name": model_cfg.name,
        "model_family": model_cfg.family,
        "num_images": len(test_images),
        "resolution": resolution,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "strengths": args.strengths,
        "summary_by_strength": {
            str(s): {
                "mean": float(np.mean(c)) if c else None,
                "std": float(np.std(c)) if c else None,
                "max": float(np.max(c)) if c else None,
                "min": float(np.min(c)) if c else None,
            }
            for s, c in summary_by_strength.items()
        },
        "best_overall_pearson_r": float(best_overall),
        "best_mean_pearson_r": float(best_mean),
        "best_strength": float(best_strength),
        "step2_pass": step2_pass,
        "step3_pass": step3_pass,
        "all_results": [
            {
                "image": r["image"],
                "strength": r["strength"],
                "pearson_r": r["metrics"]["pearson_r"],
                "ssim": r["metrics"]["ssim"],
                "mse": r["metrics"]["mse"],
                "gradient_corr": r["metrics"]["gradient_corr"],
            }
            for r in all_results
        ],
    }
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    try:
        create_summary_plot(all_results, output_dir / "strength_sweep.png")
        logger.info(f"Summary plot saved to {output_dir / 'strength_sweep.png'}")
    except Exception as e:
        logger.warning(f"Could not create summary plot: {e}")

    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
