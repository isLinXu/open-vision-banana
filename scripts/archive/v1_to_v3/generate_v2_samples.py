#!/usr/bin/env python3
"""Generate depth visualization samples from trained v2 LoRA checkpoint.

Improvements over v1:
  1. 50-step Euler integration (vs 30) for better ODE solving
  2. CFG guidance scale sweep [1.0, 2.0, 4.0]
  3. Strength sweep [0.5, 0.7, 0.9]
  4. Proper depth metrics (MSE, MAE, std, correlation)
  5. Load best_val checkpoint (by validation MSE) or best (by training loss)
  6. Generation preservation test
  7. Save metrics JSON for report generation
"""

from __future__ import annotations

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
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("vb_v2_gen")

MODEL_ID = "segmind/tiny-sd"
VISION_DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_nyu_depth_300")
OUTPUT_BASE = Path("/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_paper_aligned_v2")
OUTPUT_DIR = OUTPUT_BASE / "final_samples"
RESOLUTION = 256
DEPTH_PROMPT = "Generate a metric depth visualization of this image."

TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "proj_in", "proj_out",
    "ff.net.0.proj", "ff.net.2",
]


def load_model_and_lora(device, ckpt_name="vb_v2_best_val.pt"):
    """Load tiny-sd and apply LoRA with trained v2 weights."""
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model

    ckpt_path = OUTPUT_BASE / ckpt_name
    if not ckpt_path.exists():
        # Fallback to best (training loss) or final
        for alt in ["vb_v2_best.pt", "vb_v2_final.pt"]:
            alt_path = OUTPUT_BASE / alt
            if alt_path.exists():
                ckpt_path = alt_path
                logger.info(f"Using fallback checkpoint: {alt}")
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in {OUTPUT_BASE}")

    logger.info("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer

    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", use_rslora=True,
    )
    unet = get_peft_model(unet, lora_config)

    logger.info(f"Loading checkpoint: {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    lora_state = {}
    ema_state = {}
    for k, v in ckpt.items():
        if k.startswith("unet."):
            lora_state[k[len("unet."):]] = v
        elif k.startswith("ema."):
            ema_state[k[len("ema."):]] = v

    logger.info(f"LoRA keys: {len(lora_state)}, EMA keys: {len(ema_state)}")
    unet.load_state_dict(ema_state if ema_state else lora_state, strict=False)
    logger.info(f"Weights loaded ({'EMA' if ema_state else 'direct LoRA'})")

    unet.eval()
    return unet, vae, text_encoder, tokenizer


@torch.no_grad()
def generate_depth_flow(
    unet, vae, text_encoder, tokenizer,
    input_image, prompt, device, vae_scaling,
    num_steps=50, seed=42, strength=0.7, guidance_scale=2.0,
):
    """Generate depth visualization using flow matching img2img with CFG."""
    unet.eval()

    # Encode text
    tokenized = tokenizer([prompt], padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt").input_ids.to(device)
    encoder_hidden_states = text_encoder(tokenized)[0]

    # Encode input image
    input_latents = vae.encode(input_image).latent_dist.mode()
    input_latents = input_latents * vae_scaling

    # Partial noise
    torch.manual_seed(seed)
    noise = torch.randn_like(input_latents)
    latents = (1.0 - strength) * input_latents + strength * noise

    # CFG: unconditional embedding
    if guidance_scale > 1.0:
        uncond_tokens = tokenizer([""], padding="max_length", max_length=77,
                                   truncation=True, return_tensors="pt").input_ids.to(device)
        uncond_hidden = text_encoder(uncond_tokens)[0]
    else:
        uncond_hidden = None

    num_actual = max(1, int(num_steps * strength))
    dt = strength / num_actual
    for i in range(num_actual):
        t = strength - i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)

        if guidance_scale > 1.0 and uncond_hidden is not None:
            v_cond = unet(latents, timestep_long, encoder_hidden_states).sample
            v_uncond = unet(latents, timestep_long, uncond_hidden).sample
            pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_v = unet(latents, timestep_long, encoder_hidden_states).sample

        latents = latents - dt * pred_v

    latents = latents / vae_scaling
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    arr = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def generate_text2image_flow(
    unet, vae, text_encoder, tokenizer,
    prompt, device, vae_scaling,
    num_steps=50, seed=42, guidance_scale=2.0,
):
    """Generate image from text using flow matching."""
    unet.eval()

    tokenized = tokenizer([prompt], padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt").input_ids.to(device)
    encoder_hidden_states = text_encoder(tokenized)[0]

    if guidance_scale > 1.0:
        uncond_tokens = tokenizer([""], padding="max_length", max_length=77,
                                   truncation=True, return_tensors="pt").input_ids.to(device)
        uncond_hidden = text_encoder(uncond_tokens)[0]
    else:
        uncond_hidden = None

    torch.manual_seed(seed)
    latents = torch.randn(1, 4, 32, 32, device=device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = 1.0 - i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)

        if guidance_scale > 1.0 and uncond_hidden is not None:
            v_cond = unet(latents, timestep_long, encoder_hidden_states).sample
            v_uncond = unet(latents, timestep_long, uncond_hidden).sample
            pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_v = unet(latents, timestep_long, encoder_hidden_states).sample

        latents = latents - dt * pred_v

    latents = latents / vae_scaling
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    arr = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_metrics(gen_img, target_img):
    """Compute comprehensive depth metrics."""
    gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
    tgt_arr = np.array(target_img, dtype=np.float32) / 255.0

    mse = np.mean((gen_arr - tgt_arr) ** 2)
    mae = np.mean(np.abs(gen_arr - tgt_arr))

    # Grayscale conversion for depth correlation
    gen_gray = np.mean(gen_arr, axis=2).flatten()
    tgt_gray = np.mean(tgt_arr, axis=2).flatten()
    if np.std(gen_gray) > 1e-6 and np.std(tgt_gray) > 1e-6:
        corr = float(np.corrcoef(gen_gray, tgt_gray)[0, 1])
    else:
        corr = 0.0

    return {
        "mse": float(mse),
        "mae": float(mae),
        "correlation": corr,
        "gen_std": float(np.std(gen_arr)),
        "tgt_std": float(np.std(tgt_arr)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    unet, vae, text_encoder, tokenizer = load_model_and_lora(device)
    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)

    # ── Depth generation comparison ─────────────────────────────────
    val_image_paths = sorted((VISION_DATA_DIR / "images").glob("*.jpg"))[:6]
    val_target_paths = sorted((VISION_DATA_DIR / "targets").glob("*.jpg"))[:6]

    all_results = []

    # Strength × guidance_scale grid for first 3 samples
    logger.info("=== Depth generation grid search ===")
    for vi, (img_path, tgt_path) in enumerate(zip(val_image_paths[:3], val_target_paths[:3], strict=False)):
        img = Image.open(img_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        tgt = Image.open(tgt_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        for strength in [0.5, 0.7, 0.9]:
            for gs in [1.0, 2.0, 4.0]:
                gen_img = generate_depth_flow(
                    unet, vae, text_encoder, tokenizer,
                    input_tensor, DEPTH_PROMPT, device, vae_scaling,
                    num_steps=50, seed=42 + vi, strength=strength,
                    guidance_scale=gs,
                )

                metrics = compute_metrics(gen_img, tgt)

                # Comparison image
                comparison = Image.new("RGB", (RESOLUTION * 3, RESOLUTION + 30), (40, 40, 40))
                comparison.paste(img, (0, 30))
                comparison.paste(gen_img, (RESOLUTION, 30))
                comparison.paste(tgt, (RESOLUTION * 2, 30))

                from PIL import ImageDraw
                draw = ImageDraw.Draw(comparison)
                draw.text((10, 5), "Input", fill="white")
                draw.text((RESOLUTION + 10, 5), f"Gen (s={strength}, gs={gs})", fill="white")
                draw.text((RESOLUTION * 2 + 10, 5), "Target", fill="white")

                fname = f"grid_{vi}_s{strength}_gs{gs}.png"
                comparison.save(OUTPUT_DIR / fname)

                result = {
                    "index": vi,
                    "strength": strength,
                    "guidance_scale": gs,
                    "file": fname,
                    **metrics,
                }
                all_results.append(result)
                logger.info(
                    f"  Sample {vi} s={strength} gs={gs}: "
                    f"MSE={metrics['mse']:.4f} MAE={metrics['mae']:.4f} "
                    f"corr={metrics['correlation']:.3f} "
                    f"std={metrics['gen_std']:.4f}/{metrics['tgt_std']:.4f}"
                )

    # Best params for remaining samples
    logger.info("=== Best params for remaining samples ===")
    best_params = min(all_results, key=lambda x: x["mse"])
    best_s = best_params["strength"]
    best_gs = best_params["guidance_scale"]
    logger.info(f"Best params: strength={best_s}, guidance_scale={best_gs}")

    for vi in range(3, 6):
        img_path = val_image_paths[vi]
        tgt_path = val_target_paths[vi]
        img = Image.open(img_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        tgt = Image.open(tgt_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        gen_img = generate_depth_flow(
            unet, vae, text_encoder, tokenizer,
            input_tensor, DEPTH_PROMPT, device, vae_scaling,
            num_steps=50, seed=42 + vi, strength=best_s,
            guidance_scale=best_gs,
        )

        metrics = compute_metrics(gen_img, tgt)

        comparison = Image.new("RGB", (RESOLUTION * 3, RESOLUTION + 30), (40, 40, 40))
        comparison.paste(img, (0, 30))
        comparison.paste(gen_img, (RESOLUTION, 30))
        comparison.paste(tgt, (RESOLUTION * 2, 30))

        from PIL import ImageDraw
        draw = ImageDraw.Draw(comparison)
        draw.text((10, 5), "Input", fill="white")
        draw.text((RESOLUTION + 10, 5), f"Gen (s={best_s}, gs={best_gs})", fill="white")
        draw.text((RESOLUTION * 2 + 10, 5), "Target", fill="white")

        fname = f"best_{vi}.png"
        comparison.save(OUTPUT_DIR / fname)

        all_results.append({
            "index": vi,
            "strength": best_s,
            "guidance_scale": best_gs,
            "file": fname,
            **metrics,
        })
        logger.info(
            f"  Sample {vi}: MSE={metrics['mse']:.4f} "
            f"corr={metrics['correlation']:.3f}"
        )

    # ── Generation preservation test ────────────────────────────────
    logger.info("=== Generation preservation test ===")
    gen_prompts = [
        "a photo of a zebra",
        "a photo with bowl, spoon and wine glass",
        "a beautiful landscape with mountains",
    ]
    gen_results = []
    for pi, prompt in enumerate(gen_prompts):
        gen_img = generate_text2image_flow(
            unet, vae, text_encoder, tokenizer,
            prompt, device, vae_scaling,
            num_steps=50, seed=100 + pi, guidance_scale=2.0,
        )
        fname = f"gen_preservation_{pi}.png"
        gen_img.save(OUTPUT_DIR / fname)
        gen_results.append({"prompt": prompt, "file": fname})

    # ── Summary statistics ──────────────────────────────────────────
    depth_results = [r for r in all_results if "mse" in r]
    avg_mse = np.mean([r["mse"] for r in depth_results])
    avg_corr = np.mean([r["correlation"] for r in depth_results])
    avg_gen_std = np.mean([r["gen_std"] for r in depth_results])
    avg_tgt_std = np.mean([r["tgt_std"] for r in depth_results])

    logger.info("=" * 50)
    logger.info("Summary:")
    logger.info(f"  Avg MSE: {avg_mse:.4f}")
    logger.info(f"  Avg Correlation: {avg_corr:.3f}")
    logger.info(f"  Avg Gen Std: {avg_gen_std:.4f}")
    logger.info(f"  Avg Target Std: {avg_tgt_std:.4f}")
    logger.info(f"  Best params: s={best_s}, gs={best_gs}")
    logger.info("=" * 50)

    with open(OUTPUT_DIR / "generation_results_v2.json", "w") as f:
        json.dump({
            "checkpoint": "vb_v2_best_val.pt",
            "depth_samples": all_results,
            "gen_preservation": gen_results,
            "summary": {
                "avg_mse": float(avg_mse),
                "avg_correlation": float(avg_corr),
                "avg_gen_std": float(avg_gen_std),
                "avg_tgt_std": float(avg_tgt_std),
                "best_strength": best_s,
                "best_guidance_scale": best_gs,
            },
        }, f, indent=2)

    logger.info(f"All samples saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
