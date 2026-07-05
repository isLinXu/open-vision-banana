#!/usr/bin/env python3
"""Generate depth visualization samples from trained paper-aligned LoRA checkpoint.

Loads the best checkpoint and generates comparison images using flow matching.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("vb_generate")

MODEL_ID = "segmind/tiny-sd"
VISION_DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_nyu_depth_300")
CKPT_PATH = Path("/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_paper_aligned/vb_paper_best.pt")
OUTPUT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_paper_aligned/final_samples")
RESOLUTION = 256
DEPTH_PROMPT = "Generate a metric depth visualization of this image."

TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "proj_in", "proj_out",
    "ff.net.0.proj", "ff.net.2",
]


def load_model_and_lora(device):
    """Load tiny-sd and apply LoRA with trained weights."""
    from diffusers import StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model

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

    # Load trained weights
    logger.info(f"Loading checkpoint from {CKPT_PATH}...")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    # Load LoRA weights (skip EMA keys)
    lora_state = {}
    ema_state = {}
    for k, v in ckpt.items():
        if k.startswith("unet."):
            lora_state[k[len("unet."):]] = v
        elif k.startswith("ema."):
            # EMA keys: ema.base_model.model... → base_model.model...
            ema_state[k[len("ema."):]] = v

    logger.info(f"LoRA keys: {len(lora_state)}, EMA keys: {len(ema_state)}")

    # Load EMA weights (paper uses EMA for inference)
    unet.load_state_dict(ema_state if ema_state else lora_state, strict=False)
    logger.info(f"Weights loaded ({'EMA' if ema_state else 'direct LoRA'})")

    unet.eval()
    return unet, vae, text_encoder, tokenizer


@torch.no_grad()
def generate_depth_flow(
    unet, vae, text_encoder, tokenizer,
    input_image, prompt, device, vae_scaling,
    num_steps=30, seed=42, strength=0.6,
):
    """Generate depth visualization using flow matching img2img."""
    unet.eval()

    # Encode text
    tokenized = tokenizer(
        [prompt], padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    encoder_hidden_states = text_encoder(tokenized)[0]

    # Encode input image
    input_latents = vae.encode(input_image).latent_dist.mode()
    input_latents = input_latents * vae_scaling

    # Partial noise
    torch.manual_seed(seed)
    noise = torch.randn_like(input_latents)
    latents = (1.0 - strength) * input_latents + strength * noise

    # Euler integration from t=strength → 0
    num_actual = max(1, int(num_steps * strength))
    dt = strength / num_actual
    for i in range(num_actual):
        t = strength - i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)
        pred_v = unet(latents, timestep_long, encoder_hidden_states).sample
        latents = latents - dt * pred_v

    # Decode
    latents = latents / vae_scaling
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    arr = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def generate_text2image_flow(
    unet, vae, text_encoder, tokenizer,
    prompt, device, vae_scaling,
    num_steps=30, seed=42,
):
    """Generate image from text using flow matching (tests generation preservation)."""
    unet.eval()

    tokenized = tokenizer(
        [prompt], padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    encoder_hidden_states = text_encoder(tokenized)[0]

    torch.manual_seed(seed)
    latents = torch.randn(1, 4, 32, 32, device=device)

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = 1.0 - i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)
        pred_v = unet(latents, timestep_long, encoder_hidden_states).sample
        latents = latents - dt * pred_v

    latents = latents / vae_scaling
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    arr = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    unet, vae, text_encoder, tokenizer = load_model_and_lora(device)
    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)

    # ── Depth generation comparison ─────────────────────────────────
    val_image_paths = sorted((VISION_DATA_DIR / "images").glob("*.jpg"))[:6]
    val_target_paths = sorted((VISION_DATA_DIR / "targets").glob("*.jpg"))[:6]

    results = []
    for vi, (img_path, tgt_path) in enumerate(zip(val_image_paths, val_target_paths)):
        logger.info(f"Generating depth sample {vi+1}/6: {img_path.name}")

        # Load input
        img = Image.open(img_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        input_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        # Load target
        tgt = Image.open(tgt_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        # Generate with multiple strengths
        for strength in [0.4, 0.6, 0.8]:
            gen_img = generate_depth_flow(
                unet, vae, text_encoder, tokenizer,
                input_tensor, DEPTH_PROMPT, device, vae_scaling,
                num_steps=30, seed=42 + vi, strength=strength,
            )

            # Create comparison: input | generated | target
            comparison = Image.new("RGB", (RESOLUTION * 3, RESOLUTION + 30), (40, 40, 40))
            comparison.paste(img, (0, 30))
            comparison.paste(gen_img, (RESOLUTION, 30))
            comparison.paste(tgt, (RESOLUTION * 2, 30))

            # Add labels
            from PIL import ImageDraw
            draw = ImageDraw.Draw(comparison)
            draw.text((10, 5), "Input", fill="white")
            draw.text((RESOLUTION + 10, 5), f"Generated (s={strength})", fill="white")
            draw.text((RESOLUTION * 2 + 10, 5), "Target", fill="white")

            fname = f"depth_compare_{vi}_s{strength}.png"
            comparison.save(OUTPUT_DIR / fname)
            results.append({"index": vi, "strength": strength, "file": fname})

            # Compute metrics
            gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
            tgt_arr = np.array(tgt, dtype=np.float32) / 255.0
            mse = np.mean((gen_arr - tgt_arr) ** 2)
            mae = np.mean(np.abs(gen_arr - tgt_arr))
            logger.info(f"  s={strength}: MSE={mse:.4f}, MAE={mae:.4f}")

    # ── Generation preservation test ────────────────────────────────
    gen_prompts = [
        "a photo of a zebra",
        "a photo with bowl, spoon and wine glass",
        "a beautiful landscape with mountains",
    ]
    for pi, prompt in enumerate(gen_prompts):
        logger.info(f"Generating text2image sample {pi+1}/{len(gen_prompts)}: {prompt}")
        gen_img = generate_text2image_flow(
            unet, vae, text_encoder, tokenizer,
            prompt, device, vae_scaling,
            num_steps=30, seed=100 + pi,
        )
        gen_img.save(OUTPUT_DIR / f"gen_preservation_{pi}.png")

    # Save results
    with open(OUTPUT_DIR / "generation_results.json", "w") as f:
        json.dump({
            "checkpoint": str(CKPT_PATH),
            "depth_samples": results,
            "gen_prompts": gen_prompts,
        }, f, indent=2)

    logger.info(f"All samples saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
