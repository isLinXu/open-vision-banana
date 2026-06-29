#!/usr/bin/env python3
"""v3 evaluation — generate depth maps from crashed v3 checkpoints and compute metrics.

Uses the same flow-matching inference as training script.
Tests 3 checkpoints: best_corr (step 200), best (step ~20), step600.
"""

from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# Config (must match training)
MODEL_ID = "segmind/tiny-sd"
OUTPUT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_paper_aligned_v3")
VISION_DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_distill_428")
RESOLUTION = 256
LORA_RANK = 64
LORA_ALPHA = 128
USE_RSLORA = True
SEED = 42
DEPTH_PROMPT = "Generate a metric depth visualization of this image."
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Checkpoints to evaluate
CKPTS = {
    "best_corr_step200": OUTPUT_DIR / "vb_v3_best_corr.pt",
    "best_step20": OUTPUT_DIR / "vb_v3_best.pt",
    "step600": OUTPUT_DIR / "vb_v3_step600.pt",
}

def load_model():
    print("Loading model components...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(DEVICE, dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(DEVICE, dtype=torch.float32)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(DEVICE, dtype=torch.float32)
    vae_scaling = vae.config.scaling_factor
    print(f"VAE scaling factor: {vae_scaling}")

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none", use_rslora=USE_RSLORA,
    )
    unet = get_peft_model(unet, lora_config)
    return vae, unet, text_encoder, tokenizer, vae_scaling


def load_checkpoint(unet, ckpt_path, use_ema=True):
    """Load LoRA weights from checkpoint. use_ema=True loads ema.base_model.* keys."""
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    prefix = "ema." if use_ema else "unet."
    lora_sd = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            clean_k = k[len(prefix):]
            # For ema keys: ema.base_model.model... -> base_model.model...
            lora_sd[clean_k] = v
    if not lora_sd:
        print(f"  WARNING: no {prefix} keys found, trying unet. prefix")
        for k, v in sd.items():
            if k.startswith("unet."):
                lora_sd[k[5:]] = v
    missing, unexpected = unet.load_state_dict(lora_sd, strict=False)
    print(f"  Loaded {len(lora_sd)} keys (missing={len(missing)}, unexpected={len(unexpected)})")
    return len(lora_sd)


@torch.no_grad()
def generate_depth_flow(unet, vae, text_encoder, tokenizer,
    input_image, prompt, device, vae_scaling,
    num_steps=30, seed=42, strength=0.6, guidance_scale=2.0):
    unet.eval()
    tokenized = tokenizer([prompt], padding="max_length", max_length=77,
                          truncation=True, return_tensors="pt").input_ids.to(device)
    encoder_hidden_states = text_encoder(tokenized)[0]
    input_latents = vae.encode(input_image).latent_dist.mode()
    input_latents = input_latents * vae_scaling
    torch.manual_seed(seed)
    noise = torch.randn_like(input_latents)
    latents = (1.0 - strength) * input_latents + strength * noise
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
    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def compute_depth_correlation(gen_img, target_img):
    gen_gray = np.array(gen_img.convert("L"), dtype=np.float32).flatten()
    tgt_gray = np.array(target_img.convert("L"), dtype=np.float32).flatten()
    gen_norm = gen_gray - gen_gray.mean()
    tgt_norm = tgt_gray - tgt_gray.mean()
    denom = np.sqrt((gen_norm ** 2).sum() * (tgt_norm ** 2).sum())
    if denom < 1e-6:
        return 0.0
    return float((gen_norm * tgt_norm).sum() / denom)


def compute_metrics(gen_img, target_img):
    gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
    tgt_arr = np.array(target_img, dtype=np.float32) / 255.0
    mse = float(np.mean((gen_arr - tgt_arr) ** 2))
    mae = float(np.mean(np.abs(gen_arr - tgt_arr)))
    gen_std = float(np.std(gen_arr))
    tgt_std = float(np.std(tgt_arr))
    corr = compute_depth_correlation(gen_img, target_img)
    return {"mse": mse, "mae": mae, "gen_std": gen_std, "tgt_std": tgt_std, "correlation": corr}


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Find validation images (first 6 NYU images)
    val_dir = VISION_DATA_DIR / "images"
    tgt_dir = VISION_DATA_DIR / "targets"
    image_files = sorted(val_dir.glob("*.jpg"))[:6]
    print(f"Found {len(image_files)} validation images")

    # Load model once
    vae, unet, text_encoder, tokenizer, vae_scaling = load_model()

    # Results storage
    all_results = {}

    for ckpt_name, ckpt_path in CKPTS.items():
        if not ckpt_path.exists():
            print(f"\n[SKIP] {ckpt_name}: file not found")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name} ({ckpt_path.name})")
        print(f"{'='*60}")

        load_checkpoint(unet, ckpt_path, use_ema=True)

        ckpt_results = []
        for vi, img_path in enumerate(image_files):
            img = Image.open(img_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            arr = np.array(img, dtype=np.float32) / 255.0
            inp = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE) * 2.0 - 1.0

            # Target depth
            tgt_name = img_path.stem + ".png"
            tgt_path = tgt_dir / tgt_name
            if not tgt_path.exists():
                tgt_path = tgt_dir / img_path.name
            if not tgt_path.exists():
                print(f"  [SKIP] target not found for {img_path.name}")
                continue
            target = Image.open(tgt_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

            # Sweep strengths
            best_corr = -2.0
            best_result = None
            for strength in [0.4, 0.6, 0.8]:
                t0 = time.time()
                gen = generate_depth_flow(
                    unet, vae, text_encoder, tokenizer,
                    inp, DEPTH_PROMPT, DEVICE, vae_scaling,
                    num_steps=30, seed=SEED+vi, strength=strength, guidance_scale=2.0
                )
                m = compute_metrics(gen, target)
                m["strength"] = strength
                m["gen_time_s"] = round(time.time() - t0, 1)
                print(f"  S{vi} s={strength}: corr={m['correlation']:.4f} mse={m['mse']:.4f} "
                      f"std={m['gen_std']:.4f}/{m['tgt_std']:.4f} ({m['gen_time_s']}s)")
                if m["correlation"] > best_corr:
                    best_corr = m["correlation"]
                    best_result = m
                    best_img = gen

            # Save best image
            img_out = OUTPUT_DIR / f"eval_{ckpt_name}_s{vi}.png"
            best_img.save(img_out)
            ckpt_results.append(best_result)

        # Summary
        if ckpt_results:
            avg_corr = np.mean([r["correlation"] for r in ckpt_results])
            avg_mse = np.mean([r["mse"] for r in ckpt_results])
            avg_gen_std = np.mean([r["gen_std"] for r in ckpt_results])
            avg_tgt_std = np.mean([r["tgt_std"] for r in ckpt_results])
            print(f"\n  Summary {ckpt_name}:")
            print(f"    Avg corr: {avg_corr:.4f}")
            print(f"    Avg MSE:  {avg_mse:.4f}")
            print(f"    Avg gen std: {avg_gen_std:.4f} (target: {avg_tgt_std:.4f})")
            all_results[ckpt_name] = {
                "avg_correlation": float(avg_corr),
                "avg_mse": float(avg_mse),
                "avg_gen_std": float(avg_gen_std),
                "avg_tgt_std": float(avg_tgt_std),
                "per_sample": ckpt_results,
            }

    # Save results JSON
    results_path = OUTPUT_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("CHECKPOINT COMPARISON")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<25} {'Corr':>8} {'MSE':>8} {'GenStd':>8} {'TgtStd':>8}")
    print("-" * 60)
    for name, r in all_results.items():
        print(f"{name:<25} {r['avg_correlation']:>8.4f} {r['avg_mse']:>8.4f} "
              f"{r['avg_gen_std']:>8.4f} {r['avg_tgt_std']:>8.4f}")


if __name__ == "__main__":
    main()
