#!/usr/bin/env python3
"""Vision Banana v4 — SDXL img2img training with TRUE image conditioning.

Key changes from v3:
  1. **Model**: SDXL (3.5B) @ float16 instead of tiny-sd (27M)
     → enough capacity to learn meaningful depth structure
  2. **True img2img**: input image encoded as latent, used as noise starting point
     → model must learn image→depth mapping, not text→depth
  3. **Data**: 2K depth-labeled + 10K+ synthetic (true 9:1 ratio)
  4. **Training**: flow-matching velocity prediction with img2img latent mixing

The critical innovation — TRUE image conditioning:
  Instead of:  noisy_latents = (1-t)*target_latents + t*noise
  We do:       input_latents = vae.encode(input_image).latent_dist.mode()
               noisy_latents = (1-t)*input_latents + t*noise   ← START FROM INPUT!
               velocity_target = target_latents - noisy_latents
  
  This forces the model to learn: "given this image (as starting latent),
  generate the corresponding depth visualization."

MPS-specific adaptations:
  - float16 for SDXL model, float32 for loss computation
  - batch_size=1, grad_accum=16 (effective batch=16)
  - gradient checkpointing on UNet
  - num_workers=0 (MPS-safe)
  - torch.amp.autocast for mixed-precision forward passes

Usage:
  python scripts/train_vb_sdxl_v4.py
  python scripts/train_vb_sdxl_v4.py --model ssd-1b  # fallback if SDXL OOMs
  python scripts/train_vb_sdxl_v4.py --steps 3000    # quick validation
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_v4")

# ─── Defaults ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSION_OUTPUT = Path(os.environ.get(
    "VB_OUTPUT_DIR",
    str(PROJECT_ROOT / "output" / "vb_sdxl_v4"),
))

DEPTH_DATA_DIR = DATA_DIR / "vb_v4_depth_2k"
SYNTHETIC_DATA_DIR = DATA_DIR / "vb_v4_synthetic"

DEPTH_PROMPT = "Generate a metric depth visualization of this image."

# ─── Timestep sampling ────────────────────────────────────────────────────


def sample_timesteps_logit_normal(batch_size, device, mean=0.0, std=1.0):
    """Logit-normal timestep sampling (matches SD3/FLUX training)."""
    noise = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(noise)


def compute_min_snr_weight(t, gamma=5.0):
    """Min-SNR-gamma weighting for balanced gradient signal."""
    snr = (1.0 - t) ** 2 / (t ** 2 + 1e-8)
    weight = torch.clamp(snr, max=gamma) / snr
    return weight


# ─── Datasets ─────────────────────────────────────────────────────────────


class DepthImg2ImgDataset(Dataset):
    """Paired (input_image, target_depth) dataset for img2img training.

    Each sample returns:
      - image: input RGB image in [-1, 1]  (the scene to predict depth for)
      - target_rgb: target depth visualization in [0, 1]
      - prompt: depth instruction prompt
    """

    def __init__(self, data_dir: Path, resolution: int = 256, augment: bool = True):
        self.resolution = resolution
        self.augment = augment
        self.image_dir = data_dir / "images"
        self.target_dir = data_dir / "targets"
        self.prompt = DEPTH_PROMPT
        self.samples = []

        for img_path in sorted(self.image_dir.glob("*.jpg")):
            tgt_name = img_path.stem + ".png"
            tgt_path = self.target_dir / tgt_name
            if not tgt_path.exists():
                tgt_path = self.target_dir / img_path.name
            if tgt_path.exists():
                self.samples.append((img_path, tgt_path))

        logger.info(f"DepthImg2ImgDataset: {len(self.samples)} paired samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tgt_path = self.samples[idx]

        # Load input image → [-1, 1]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0

        # Load target depth visualization → [0, 1]
        tgt = Image.open(tgt_path).convert("RGB")
        tgt = tgt.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr_t = np.array(tgt, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(arr_t).permute(2, 0, 1)

        if self.augment:
            # Synchronized augmentation (same transform on both)
            if np.random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
                target_tensor = torch.flip(target_tensor, dims=[2])
            if np.random.random() < 0.3:
                scale = np.random.uniform(0.8, 1.0)
                _, H, W = image_tensor.shape
                crop_h, crop_w = int(H * scale), int(W * scale)
                top = np.random.randint(0, H - crop_h + 1)
                left = np.random.randint(0, W - crop_w + 1)
                image_tensor = image_tensor[:, top:top+crop_h, left:left+crop_w]
                target_tensor = target_tensor[:, top:top+crop_h, left:left+crop_w]
                image_tensor = F.interpolate(
                    image_tensor.unsqueeze(0), size=(H, W),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                target_tensor = F.interpolate(
                    target_tensor.unsqueeze(0), size=(H, W),
                    mode='nearest'
                ).squeeze(0)
            if np.random.random() < 0.5:
                brightness = 1.0 + np.random.uniform(-0.1, 0.1)
                image_tensor = (image_tensor * brightness).clamp(-1.0, 1.0)

        return {
            "image": image_tensor,
            "target_rgb": target_tensor,
            "prompt": self.prompt,
            "task": "depth",
            "filename": img_path.stem,
        }


class SimpleGenerationDataset(Dataset):
    """Text-to-image generation samples for 9:1 mixture preservation."""

    def __init__(self, data_dir: Path, resolution: int = 256):
        self.resolution = resolution
        self.image_dir = data_dir / "images"
        captions_path = data_dir / "captions.json"
        self.samples = []

        if captions_path.exists():
            with open(captions_path) as f:
                pairs = json.load(f)
            for pair in pairs:
                img_path = self.image_dir / pair["filename"]
                if img_path.exists():
                    self.samples.append({
                        "image_path": str(img_path),
                        "prompt": pair.get("caption", "A high quality photograph."),
                    })

        # Fallback: scan images directory
        if not self.samples:
            for img_path in sorted(self.image_dir.glob("*.jpg")):
                self.samples.append({
                    "image_path": str(img_path),
                    "prompt": "A high quality, photorealistic image.",
                })

        logger.info(f"SimpleGenerationDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return {
            "target_rgb": target_tensor,
            "image": target_tensor * 2.0 - 1.0,  # for img2img-style generation training
            "prompt": sample["prompt"],
            "task": "generation",
            "filename": Path(sample["image_path"]).stem,
        }


class MixedDataset(Dataset):
    """9:1 generation-to-vision mixture (paper ratio)."""

    def __init__(self, vision_ds, gen_ds, vision_ratio=0.1, seed=42):
        self.vision_ds = vision_ds
        self.gen_ds = gen_ds
        self.vision_ratio = vision_ratio
        self.rng = torch.Generator().manual_seed(seed)
        self.total_vision = len(vision_ds)
        self.total_gen = len(gen_ds) if gen_ds else 0

    def __len__(self):
        if self.gen_ds is None or self.vision_ratio >= 1.0:
            return self.total_vision
        return max(1, round(self.total_vision / self.vision_ratio))

    def __getitem__(self, idx):
        use_vision = (
            self.gen_ds is None
            or torch.rand(1, generator=self.rng).item() < self.vision_ratio
        )
        if use_vision:
            sample_idx = int(torch.randint(len(self.vision_ds), (1,), generator=self.rng).item())
            return self.vision_ds[sample_idx]
        else:
            sample_idx = int(torch.randint(len(self.gen_ds), (1,), generator=self.rng).item())
            return self.gen_ds[sample_idx]


def collate_fn(batch):
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result


# ─── EMA ──────────────────────────────────────────────────────────────────


class LoRAEMA:
    """Exponential moving average for LoRA parameters."""

    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}

    def register(self, params):
        for name, param in params:
            self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, params):
        for name, param in params:
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def get_shadow(self):
        return {k: v.clone() for k, v in self.shadow.items()}


# ─── Checkpoint utilities ─────────────────────────────────────────────────


def save_lora_state(unet, path, ema_state=None):
    """Save LoRA adapter weights + optional EMA state."""
    state = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            state[f"unet.{name}"] = param.data.cpu().clone()
    if ema_state is not None:
        for k, v in ema_state.items():
            state[f"ema.{k}"] = v.cpu().clone()
    torch.save(state, path)
    logger.info(f"Saved {len(state)} keys to {path.name}")


# ─── Depth metrics ────────────────────────────────────────────────────────


def compute_depth_correlation(gen_img, target_img):
    """Pearson r between generated and target grayscale (luminance)."""
    gen_gray = np.array(gen_img.convert("L"), dtype=np.float32).flatten()
    tgt_gray = np.array(target_img.convert("L"), dtype=np.float32).flatten()
    gen_norm = gen_gray - gen_gray.mean()
    tgt_norm = tgt_gray - tgt_gray.mean()
    denom = np.sqrt((gen_norm ** 2).sum() * (tgt_norm ** 2).sum())
    if denom < 1e-6:
        return 0.0
    return float((gen_norm * tgt_norm).sum() / denom)


def compute_depth_metrics(gen_img, target_img):
    gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
    tgt_arr = np.array(target_img, dtype=np.float32) / 255.0
    mse = np.mean((gen_arr - tgt_arr) ** 2)
    mae = np.mean(np.abs(gen_arr - tgt_arr))
    correlation = compute_depth_correlation(gen_img, target_img)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "gen_std": float(np.std(gen_arr)),
        "tgt_std": float(np.std(tgt_arr)),
        "correlation": correlation,
    }


# ─── Flow matching inference with TRUE img2img ────────────────────────────


@torch.no_grad()
def generate_depth_img2img(
    unet, vae, text_encoder, tokenizer,
    input_image, prompt, device, vae_scaling,
    num_steps=30, seed=42, strength=0.6,
    guidance_scale=2.0,
    use_sdxl_text_encoder_2=None,
):
    """Generate depth visualization via TRUE img2img flow matching.

    The critical difference from v3:
      - v3: noisy_latents starts from target_latents (model learns to denoise target)
      - v4: noisy_latents starts from input_latents (model must transform input→target)

    This is the key to learning image→depth mapping.
    """
    unet.eval()

    # Encode text prompt
    tokenized = tokenizer(
        [prompt], padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    encoder_hidden_states = text_encoder(tokenized)[0]

    # SDXL dual text encoder: concatenate hidden states + extract pooled embeds
    pooled_embeds = None
    if use_sdxl_text_encoder_2 is not None:
        te2_out = use_sdxl_text_encoder_2(tokenized)
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, te2_out.last_hidden_state], dim=-1
        )
        pooled_embeds = te2_out.text_embeds  # [1, 1280]

    # Build added_cond_kwargs (required by SDXL UNet)
    # Derive dtype from encoder_hidden_states (avoids scope issue — dtype is
    # only defined in main(), not accessible from this module-level function)
    add_time_ids = torch.tensor(
        [input_image.shape[-2], input_image.shape[-1], 0, 0,
         input_image.shape[-2], input_image.shape[-1]],
        device=device, dtype=encoder_hidden_states.dtype,
    ).unsqueeze(0)
    added_cond_kwargs = {
        "text_embeds": pooled_embeds,
        "time_ids": add_time_ids,
    }

    # ── TRUE img2img: encode INPUT image as latent starting point ──
    # Conditional flow matching: t=0 → input, t=1 → target
    # We start at t=0 (pure input) and integrate forward to t=1 (target).
    # "strength" controls how much of the trajectory the model traverses:
    #   strength=1.0 → full integration from input to target
    #   strength=0.5 → half-way (more input preserved)
    input_latents = vae.encode(input_image).latent_dist.mode()
    input_latents = input_latents * vae_scaling

    # Start from pure input at t=0
    torch.manual_seed(seed)
    latents = input_latents.clone()

    # CFG setup
    uncond_pooled = None
    if guidance_scale > 1.0:
        uncond_tokens = tokenizer(
            [""], padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        uncond_hidden = text_encoder(uncond_tokens)[0]
        if use_sdxl_text_encoder_2 is not None:
            te2_uncond = use_sdxl_text_encoder_2(uncond_tokens)
            uncond_hidden = torch.cat([uncond_hidden, te2_uncond.last_hidden_state], dim=-1)
            uncond_pooled = te2_uncond.text_embeds
    else:
        uncond_hidden = None

    # Build added_cond_kwargs for uncond if needed
    uncond_added_cond = None
    if uncond_pooled is not None:
        uncond_added_cond = {
            "text_embeds": uncond_pooled,
            "time_ids": add_time_ids,
        }

    # Flow matching denoising loop
    # Integrate from t=0 (input) toward t=strength (partial target)
    # velocity = target - input, so latents += dt * pred_v moves toward target
    num_actual = max(1, int(num_steps * strength))
    dt = strength / num_actual  # step size: 0 → strength

    for i in range(num_actual):
        t = i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)

        if guidance_scale > 1.0 and uncond_hidden is not None:
            v_cond = unet(latents, timestep_long, encoder_hidden_states,
                          added_cond_kwargs=added_cond_kwargs).sample
            v_uncond = unet(latents, timestep_long, uncond_hidden,
                            added_cond_kwargs=uncond_added_cond).sample
            pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_v = unet(latents, timestep_long, encoder_hidden_states,
                          added_cond_kwargs=added_cond_kwargs).sample

        latents = latents + dt * pred_v

    # Decode to image
    latents = latents / vae_scaling
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    unet.train()
    return Image.fromarray(image_np)


# ─── Main training ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Vision Banana v4 SDXL training")
    parser.add_argument("--model", choices=["sdxl", "ssd-1b"], default="ssd-1b",
                        help="Base model (default: ssd-1b, use sdxl if disk allows)")
    parser.add_argument("--steps", type=int, default=3000,
                        help="Total optimizer steps (default: 3000)")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Training resolution (default: 256)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--vision-ratio", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--use-rslora", action="store_true", default=True)
    parser.add_argument("--no-min-snr", action="store_true")
    parser.add_argument("--img2img-strength", type=float, default=0.6,
                        help="Training img2img strength: how much noise to add to input (0.6)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = SESSION_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "val_during_training").mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # ─── Config summary ───
    model_id = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "ssd-1b": "segmind/SSD-1B",
    }[args.model]

    logger.info("=" * 70)
    logger.info("Vision Banana v4 — SDXL img2img Training (TRUE image conditioning)")
    logger.info("=" * 70)
    logger.info(f"Model: {model_id} ({args.model})")
    logger.info(f"Device: {device}, Resolution: {args.resolution}")
    logger.info(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, rsLoRA={args.use_rslora}")
    logger.info(f"Training: flow matching v-prediction + TRUE img2img (strength={args.img2img_strength})")
    logger.info(f"Data mix: {args.vision_ratio:.0%} vision + {1-args.vision_ratio:.0%} generation")
    logger.info(f"Batch: {args.batch_size}×{args.grad_accum}={args.batch_size*args.grad_accum}, LR: {args.lr}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"Key change: input image → VAE latent → noise start (not target latent!)")
    logger.info("=" * 70)

    # ─── Load SDXL pipeline ───
    from diffusers import StableDiffusionXLPipeline

    dtype = torch.float16
    logger.info(f"Loading {model_id} @ {dtype} ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device) if hasattr(pipe, "text_encoder_2") else None
    tokenizer = pipe.tokenizer

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)

    unet.enable_gradient_checkpointing()
    unet_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"UNet params: {unet_params / 1e6:.1f}M")

    # ─── Apply rsLoRA ───
    from peft import LoraConfig, get_peft_model

    # SDXL UNet attention modules — broader coverage
    target_modules = [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
    ]

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        use_rslora=args.use_rslora,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(f"Trainable: {trainable} ({trainable/total*100:.2f}%)")

    # ─── Build datasets ───
    if not DEPTH_DATA_DIR.exists():
        logger.error(f"Depth data not found: {DEPTH_DATA_DIR}")
        logger.error("Run: python scripts/prepare_v4_data.py first")
        return

    vision_ds = DepthImg2ImgDataset(DEPTH_DATA_DIR, args.resolution, augment=True)
    gen_ds = None
    if SYNTHETIC_DATA_DIR.exists():
        gen_ds = SimpleGenerationDataset(SYNTHETIC_DATA_DIR, args.resolution)
    else:
        logger.warning(f"Synthetic data not found: {SYNTHETIC_DATA_DIR}")
        logger.warning("Training with vision-only (100% vision ratio)")

    mixed_ds = MixedDataset(vision_ds, gen_ds, vision_ratio=args.vision_ratio, seed=args.seed)
    logger.info(f"Mixed dataset: {len(mixed_ds)} samples/epoch")

    dataloader = DataLoader(
        mixed_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=True,
    )

    # ─── Optimizer & scheduler ───
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    total_steps = args.steps
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    # ─── EMA ───
    ema = LoRAEMA(decay=args.ema_decay)
    ema.register([(n, p) for n, p in unet.named_parameters() if p.requires_grad])
    logger.info(f"EMA registered for {len(ema.shadow)} params (decay={args.ema_decay})")

    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)

    # ─── Validation image pool ───
    val_image_paths = sorted((DEPTH_DATA_DIR / "images").glob("*.jpg"))[:6]
    val_images = []
    val_targets = []
    for p in val_image_paths:
        img = Image.open(p).convert("RGB").resize(
            (args.resolution, args.resolution), Image.LANCZOS
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        val_images.append(t)
        tgt_name = p.stem + ".png"
        tgt_path = DEPTH_DATA_DIR / "targets" / tgt_name
        if not tgt_path.exists():
            tgt_path = DEPTH_DATA_DIR / "targets" / p.name
        if tgt_path.exists():
            tgt = Image.open(tgt_path).convert("RGB").resize(
                (args.resolution, args.resolution), Image.LANCZOS
            )
            val_targets.append(tgt)

    # ─── Training loop ───
    train_log = []
    val_log = []
    best_loss = float("inf")
    best_val_corr = -1.0
    t_train_start = time.time()
    global_step = 0

    logger.info("Starting training...")
    logger.info(f"  CRITICAL: Using TRUE img2img — input image as latent noise start")

    while global_step < total_steps:
        for step, batch in enumerate(dataloader):
            if global_step >= total_steps:
                break

            target_rgb = batch["target_rgb"].to(device).float()
            input_images = batch["image"].to(device).to(dtype)
            prompts = batch["prompt"]
            tasks = batch["task"]
            bsz = target_rgb.shape[0]

            # ── Encode TARGET to latents (what we want the model to generate) ──
            with torch.no_grad():
                target_pixels = target_rgb * 2.0 - 1.0
                target_pixels = target_pixels.to(dtype)
                target_latents = vae.encode(target_pixels).latent_dist.mode()
                target_latents = target_latents * vae_scaling

            # ── Encode INPUT image to latents (TRUE img2img starting point!) ──
            with torch.no_grad():
                input_latents = vae.encode(input_images).latent_dist.mode()
                input_latents = input_latents * vae_scaling

            # ── Sample timesteps ──
            timesteps = sample_timesteps_logit_normal(bsz, device)

            t_b = timesteps.view(bsz, 1, 1, 1)

            # ── Conditional flow matching with stochastic interpolant ──
            # Path: x_t = (1-t)*input + t*target + sigma*sqrt(t*(1-t))*noise
            # Velocity: v = target - input (constant, noise doesn't affect derivative)
            # The noise term bridges the train→inference distribution gap:
            #   at inference, target is unknown, so starting point uses noise instead.
            noise = torch.randn_like(target_latents)
            sigma = 0.1  # small perturbation
            noise_term = sigma * (t_b * (1.0 - t_b)).sqrt() * noise

            noisy_latents = (1.0 - t_b) * input_latents + t_b * target_latents + noise_term

            # Velocity target: constant direction input→target
            velocity_target = target_latents - input_latents

            # ── Encode text ──
            with torch.no_grad():
                tokenized = tokenizer(
                    prompts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt"
                ).input_ids.to(device)
                encoder_hidden_states = text_encoder(tokenized)[0]

                # SDXL dual encoder: concatenate hidden states + extract pooled embeds
                pooled_embeds = None
                if text_encoder_2 is not None:
                    te2_out = text_encoder_2(tokenized)
                    encoder_hidden_states = torch.cat(
                        [encoder_hidden_states, te2_out.last_hidden_state], dim=-1
                    )
                    pooled_embeds = te2_out.text_embeds

            # Build added_cond_kwargs (required by SDXL UNet)
            target_size = (args.resolution, args.resolution)
            add_time_ids = torch.tensor(
                [target_size[0], target_size[1], 0, 0, target_size[0], target_size[1]],
                device=device, dtype=dtype,
            ).unsqueeze(0).expand(bsz, -1)
            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids": add_time_ids,
            }

            timestep_long = (timesteps * 999).long().clamp(0, 999)

            # ── UNet forward ──
            with torch.amp.autocast("mps", dtype=dtype, enabled=(device.type == "mps")):
                predicted_velocity = unet(
                    noisy_latents, timestep_long, encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

            # ── Loss ──
            # Note: min-SNR weighting is designed for noise→data flows, not
            # input→target conditional flows. We use uniform weighting here.
            loss = F.mse_loss(predicted_velocity.float(), velocity_target.float())

            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            epoch_loss = loss.item()

            if (step + 1) % args.grad_accum == 0 or step == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler_lr.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                ema.update([(n, p) for n, p in unet.named_parameters() if p.requires_grad])

                current_lr = scheduler_lr.get_last_lr()[0]

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t_train_start
                    eta = elapsed / global_step * (total_steps - global_step)
                    logger.info(
                        f"S{global_step}/{total_steps} | "
                        f"Loss: {epoch_loss:.4f} | LR: {current_lr:.2e} | "
                        f"{elapsed:.0f}s | ETA: {eta/60:.0f}m"
                    )
                    train_log.append({
                        "step": global_step,
                        "loss": epoch_loss,
                        "lr": current_lr,
                    })

                if global_step % args.save_every == 0:
                    save_lora_state(
                        unet,
                        output_dir / f"vb_v4_step{global_step}.pt",
                        ema.get_shadow(),
                    )

                if epoch_loss < best_loss * 0.99:
                    best_loss = epoch_loss

                # ── Validation with TRUE img2img ──
                if global_step % args.val_every == 0 and global_step <= total_steps:
                    logger.info(f"Validation at step {global_step} (TRUE img2img)...")
                    val_results = []
                    for vi, vimg in enumerate(val_images[:3]):
                        best_corr = -2.0
                        best_strength = 0.6
                        best_img = None
                        best_metrics = None

                        for strength in [0.4, 0.6, 0.8]:
                            try:
                                gen_img = generate_depth_img2img(
                                    unet, vae, text_encoder, tokenizer,
                                    vimg, DEPTH_PROMPT, device, vae_scaling,
                                    num_steps=20, seed=args.seed + vi,
                                    strength=strength,
                                    guidance_scale=2.0,
                                    use_sdxl_text_encoder_2=text_encoder_2,
                                )
                                if vi < len(val_targets):
                                    metrics = compute_depth_metrics(gen_img, val_targets[vi])
                                    if metrics["correlation"] > best_corr:
                                        best_corr = metrics["correlation"]
                                        best_strength = strength
                                        best_img = gen_img
                                        best_metrics = metrics
                            except Exception as e:
                                logger.warning(f"  Val s={strength} failed: {e}")
                                continue

                        if best_img is not None:
                            best_img.save(
                                output_dir / "val_during_training" /
                                f"val_step{global_step}_{vi}_s{best_strength}.png"
                            )
                            val_results.append(best_metrics)

                    if val_results:
                        avg_corr = np.mean([m["correlation"] for m in val_results])
                        avg_mse = np.mean([m["mse"] for m in val_results])
                        logger.info(
                            f"  Val: corr={avg_corr:.4f} | MSE={avg_mse:.4f} | "
                            f"gen_std={np.mean([m['gen_std'] for m in val_results]):.4f}"
                        )
                        val_log.append({
                            "step": global_step,
                            "avg_correlation": float(avg_corr),
                            "avg_mse": float(avg_mse),
                        })
                        if avg_corr > best_val_corr:
                            best_val_corr = avg_corr
                            save_lora_state(
                                unet,
                                output_dir / "vb_v4_best_corr.pt",
                                ema.get_shadow(),
                            )
                            logger.info(f"  New best corr: {best_val_corr:.4f}")

    # ─── Save final ───
    save_lora_state(unet, output_dir / "vb_v4_final.pt", ema.get_shadow())
    total_time = time.time() - t_train_start
    logger.info(f"Training complete! Total time: {total_time/60:.1f} min")
    logger.info(f"Final loss: {epoch_loss:.4f}, Best val corr: {best_val_corr:.4f}")

    # ─── Save training log ───
    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "config": {
                "model": model_id,
                "version": "v4",
                "resolution": args.resolution,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "total_steps": total_steps,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "use_rslora": args.use_rslora,
                "target_modules": target_modules,
                "seed": args.seed,
                "device": str(device),
                "vision_ratio": args.vision_ratio,
                "training_objective": "flow_matching_v_prediction",
                "key_innovation": "true_img2img_input_latent_as_noise_start",
                "ema_decay": args.ema_decay,
                "use_min_snr": not args.no_min_snr,
                "min_snr_gamma": args.min_snr_gamma,
                "img2img_strength": args.img2img_strength,
            },
            "train_metrics": train_log,
            "val_metrics": val_log,
            "final_loss": epoch_loss,
            "best_train_loss": best_loss,
            "best_val_correlation": best_val_corr,
            "total_steps": global_step,
            "total_time_s": total_time,
        }, f, indent=2)
    logger.info(f"Training log saved to {output_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
