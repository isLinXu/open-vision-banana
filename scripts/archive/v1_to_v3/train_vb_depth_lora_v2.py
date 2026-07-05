#!/usr/bin/env python3
"""Vision Banana LoRA training v2 — optimized depth task on tiny-sd, MPS-compatible.

Key optimizations over v1:
  1. Expanded LoRA target modules: add proj_in, proj_out, ff
  2. Data augmentation: random horizontal flip (applied consistently to both image and target)
  3. Min-SNR weighted loss for faster convergence
  4. Text prompt properly encoded (not zero placeholder)
  5. VAE encoding with deterministic mode (mode instead of sample) for target
  6. More epochs (15) with cosine schedule + longer warmup
  7. EMA on LoRA weights for stable inference
  8. Periodic validation generation during training
  9. Better checkpoint strategy (keep best + last + periodic)

Architecture (same as v1):
  - Input image → VAE encode → input_latents [B, 4, 32, 32]
  - Depth target → VAE encode → target_latents [B, 4, 32, 32]
  - noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
  - UNet input = concat(input_latents, noisy_latents) [B, 8, 32, 32]
  - ChannelAdapter (8→4 1x1 conv) → UNet → predict noise
  - Loss: Min-SNR weighted MSE on noise prediction
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# HuggingFace mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_train_v2")

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "segmind/tiny-sd"
DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_depth_300")
OUTPUT_DIR = Path(
    "/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_depth_lora_v2"
)
RESOLUTION = 256
BATCH_SIZE = 1
GRAD_ACCUM = 8          # effective batch size = 8
LEARNING_RATE = 3e-5    # slightly lower for larger dataset + more epochs
EPOCHS = 15
LORA_RANK = 64
LORA_ALPHA = 128
USE_RSLORA = True
SAVE_EVERY = 20         # save checkpoint every N optimizer steps
LOG_EVERY = 5
VAL_EVERY = 40          # generate validation samples every N steps
SEED = 42
NUM_WORKERS = 0

# Min-SNR config
MIN_SNR_GAMMA = 5.0     # Min-SNR gamma parameter

# EMA config
USE_EMA = True
EMA_DECAY = 0.999

DEPTH_PROMPT = "Generate a metric depth visualization of this image."

# Expanded target modules for better coverage
TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "proj_in", "proj_out",
    "ff.net.0.proj", "ff.net.2",
]

# Augmentation
AUG_FLIP_PROB = 0.5


# ─── Dataset ─────────────────────────────────────────────────────────────
class VisionBananaDepthDatasetV2(Dataset):
    """Paired (image, depth_rgb) dataset with augmentation support."""

    def __init__(self, data_dir: Path, resolution: int = 256, augment: bool = True):
        self.resolution = resolution
        self.augment = augment
        self.image_dir = data_dir / "images"
        self.target_dir = data_dir / "targets"

        self.samples = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            tgt_path = self.target_dir / img_path.name
            if tgt_path.exists():
                self.samples.append((img_path, tgt_path))

        logger.info(f"Dataset: {len(self.samples)} paired samples (augment={augment})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, tgt_path = self.samples[idx]

        # Load input image → [3, H, W] in [-1, 1]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0

        # Load depth target → [3, H, W] in [-1, 1]
        tgt = Image.open(tgt_path).convert("RGB")
        tgt = tgt.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr_t = np.array(tgt, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(arr_t).permute(2, 0, 1) * 2.0 - 1.0

        # Consistent horizontal flip augmentation
        if self.augment and np.random.random() < AUG_FLIP_PROB:
            image_tensor = torch.flip(image_tensor, dims=[2])
            target_tensor = torch.flip(target_tensor, dims=[2])

        return {
            "image": image_tensor,
            "target_rgb": target_tensor,
            "prompt": DEPTH_PROMPT,
            "filename": img_path.stem,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "target_rgb": torch.stack([b["target_rgb"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
        "filename": [b["filename"] for b in batch],
    }


# ─── Channel Adapter ────────────────────────────────────────────────────
class ChannelAdapter(nn.Module):
    """Learnable 8→4 channel adapter with residual connection."""

    def __init__(self, in_channels: int = 8, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        # Initialize close to identity for the noisy_latents half (channels 4:8)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, out_channels:, 0, 0] = torch.eye(out_channels)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─── EMA ─────────────────────────────────────────────────────────────────
class LoRAEMA:
    """Exponential moving average for LoRA + adapter weights."""

    def __init__(self, decay: float = 0.999):
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


# ─── Min-SNR weighting ──────────────────────────────────────────────────
def compute_min_snr_weight(
    scheduler, timesteps: torch.Tensor, gamma: float = 5.0, device=None
) -> torch.Tensor:
    """Compute Min-SNR weights for each timestep.

    SNR(t) = alpha_cumprod(t) / (1 - alpha_cumprod(t))
    weight = min(SNR, gamma) / SNR

    This downweights high-noise (high-SNR) steps that contribute less to learning.
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    alpha_t = alphas_cumprod[timesteps]
    snr = alpha_t / (1 - alpha_t + 1e-8)  # SNR at each timestep
    # Min-SNR: clip SNR at gamma
    snr_clipped = torch.clamp(snr, max=gamma)
    # Weight = min(SNR, gamma) / SNR
    weights = snr_clipped / (snr + 1e-8)
    return weights


# ─── Training ────────────────────────────────────────────────────────────
def save_trainable_state(unet, adapter, path: Path, ema_state: dict = None):
    """Save LoRA adapter weights + channel adapter weights + optional EMA."""
    state = {}

    for name, param in unet.named_parameters():
        if param.requires_grad:
            state[f"unet.{name}"] = param.data.cpu().clone()

    for name, param in adapter.named_parameters():
        state[f"adapter.{name}"] = param.data.cpu().clone()

    if ema_state is not None:
        for k, v in ema_state.items():
            state[f"ema.{k}"] = v.cpu().clone()

    torch.save(state, path)
    logger.info(f"Saved {len(state)} trainable params to {path.name}")


@torch.no_grad()
def validate_sample(
    unet, vae, scheduler, adapter, tokenizer, text_encoder,
    image_tensor, device, prompt, vae_scaling, num_steps=20, seed=42,
) -> Image.Image:
    """Generate a single depth RGB sample for validation during training."""
    unet.eval()
    adapter.eval()

    # Encode input image
    input_latents = vae.encode(image_tensor).latent_dist.mode()
    input_latents = input_latents * vae_scaling

    # Encode text prompt
    tokenized = tokenizer(
        [prompt], padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    encoder_hidden_states = text_encoder(tokenized)[0]

    # Start from random noise
    torch.manual_seed(seed)
    latents = torch.randn(1, 4, 32, 32, device=device)
    scheduler.set_timesteps(num_steps, device=device)

    for t in scheduler.timesteps:
        concat = torch.cat([input_latents, latents], dim=1)
        adapted = adapter(concat)
        noise_pred = unet(adapted, t.unsqueeze(0), encoder_hidden_states).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae_scaling
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    unet.train()
    adapter.train()
    return Image.fromarray(image_np)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val_during_training").mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Config: rank={LORA_RANK}, alpha={LORA_ALPHA}, rsLoRA={USE_RSLORA}")
    logger.info(f"Resolution: {RESOLUTION}, Batch: {BATCH_SIZE}×{GRAD_ACCUM}, LR: {LEARNING_RATE}")
    logger.info(f"Epochs: {EPOCHS}, Output: {OUTPUT_DIR}")
    logger.info(f"Min-SNR gamma: {MIN_SNR_GAMMA}, EMA: {USE_EMA} (decay={EMA_DECAY})")
    logger.info(f"Target modules: {TARGET_MODULES}")

    # ── Load pipeline ───────────────────────────────────────────────────
    from diffusers import StableDiffusionPipeline

    logger.info("Loading model (segmind/tiny-sd)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )

    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.enable_gradient_checkpointing()

    unet_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"UNet params: {unet_params / 1e6:.1f}M")

    # ── Apply rsLoRA ────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none",
        use_rslora=USE_RSLORA,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Channel adapter ─────────────────────────────────────────────────
    adapter = ChannelAdapter(in_channels=8, out_channels=4).to(device)
    adapter_params = sum(p.numel() for p in adapter.parameters())
    logger.info(f"Channel adapter params: {adapter_params} ({adapter_params / 1e3:.1f}K)")

    # ── Build dataset & dataloader ──────────────────────────────────────
    dataset = VisionBananaDepthDatasetV2(DATA_DIR, RESOLUTION, augment=True)
    if len(dataset) == 0:
        raise RuntimeError(f"No paired samples found in {DATA_DIR}")

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=True,
    )
    logger.info(f"DataLoader: {len(dataloader)} batches/epoch × {EPOCHS} epochs")

    # ── Optimizer & scheduler ───────────────────────────────────────────
    trainable_params = (
        [p for p in unet.parameters() if p.requires_grad]
        + list(adapter.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params, lr=LEARNING_RATE, weight_decay=0.01,
    )

    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    warmup_steps = min(40, total_steps // 5)  # longer warmup for larger dataset

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    # ── EMA ─────────────────────────────────────────────────────────────
    ema = None
    if USE_EMA:
        ema = LoRAEMA(decay=EMA_DECAY)
        named_params = (
            [(f"unet.{n}", p) for n, p in unet.named_parameters() if p.requires_grad]
            + [(f"adapter.{n}", p) for n, p in adapter.named_parameters()]
        )
        ema.register(named_params)
        logger.info(f"EMA registered for {len(named_params)} params")

    # ── Validation sample pool ──────────────────────────────────────────
    val_image_paths = sorted((DATA_DIR / "images").glob("*.jpg"))[:3]
    val_target_paths = sorted((DATA_DIR / "targets").glob("*.jpg"))[:3]
    val_images_tensor = []
    for p in val_image_paths:
        img = Image.open(p).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        val_images_tensor.append(t)

    # ── Training loop ───────────────────────────────────────────────────
    train_log = []
    global_step = 0
    best_loss = float("inf")

    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)
    num_train_timesteps = scheduler.config.num_train_timesteps

    # Move alphas_cumprod to device for Min-SNR
    scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_snr_weight_sum = 0.0
        t_epoch_start = time.time()

        unet.train()
        adapter.train()

        for step, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            targets = batch["target_rgb"].to(device)
            prompts = batch["prompt"]

            bsz = images.shape[0]

            # ── Encode images and targets to latents ─────────────────
            with torch.no_grad():
                # Use mode (deterministic) for input, sample for target
                input_latents = vae.encode(images).latent_dist.mode()
                input_latents = input_latents * vae_scaling

                # Use mode for target too — more stable training signal
                target_latents = vae.encode(targets).latent_dist.mode()
                target_latents = target_latents * vae_scaling

            # ── Encode text prompt ────────────────────────────────────
            with torch.no_grad():
                tokenized = tokenizer(
                    prompts, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt",
                ).input_ids.to(device)
                encoder_hidden_states = text_encoder(tokenized)[0]

            # ── Sample noise & timestep ───────────────────────────────
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=device).long()

            # ── Min-SNR weights ───────────────────────────────────────
            snr_weights = compute_min_snr_weight(
                scheduler, timesteps, gamma=MIN_SNR_GAMMA, device=device
            )  # [B]

            # ── Add noise to target latents ───────────────────────────
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

            # ── Concatenate input + noisy target → 8 channels ─────────
            concat_latents = torch.cat([input_latents, noisy_latents], dim=1)
            adapted_latents = adapter(concat_latents)

            # ── UNet forward: predict noise ───────────────────────────
            predicted_noise = unet(
                adapted_latents, timesteps, encoder_hidden_states,
            ).sample

            # ── Min-SNR weighted MSE loss ─────────────────────────────
            # Per-sample MSE, then weight by SNR
            per_sample_loss = F.mse_loss(
                predicted_noise, noise, reduction="none"
            ).mean(dim=[1, 2, 3])  # [B]
            loss = (per_sample_loss * snr_weights).mean()

            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()

            epoch_loss += loss.item()
            epoch_snr_weight_sum += snr_weights.mean().item()
            epoch_steps += 1

            # ── Optimizer step ────────────────────────────────────────
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler_lr.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # EMA update
                if ema is not None:
                    named_params = (
                        [(f"unet.{n}", p) for n, p in unet.named_parameters() if p.requires_grad]
                        + [(f"adapter.{n}", p) for n, p in adapter.named_parameters()]
                    )
                    ema.update(named_params)

                current_lr = scheduler_lr.get_last_lr()[0]
                running_loss = epoch_loss / epoch_steps

                if global_step % LOG_EVERY == 0:
                    elapsed = time.time() - t_epoch_start
                    logger.info(
                        f"Epoch {epoch+1}/{EPOCHS} | Step {global_step}/{total_steps} | "
                        f"Loss: {running_loss:.4f} | LR: {current_lr:.2e} | "
                        f"{elapsed:.0f}s"
                    )
                    train_log.append({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss": running_loss,
                        "lr": current_lr,
                    })

                # Save periodic checkpoints
                if global_step % SAVE_EVERY == 0:
                    ckpt_path = OUTPUT_DIR / f"vb_lora_v2_step{global_step}.pt"
                    save_trainable_state(
                        unet, adapter, ckpt_path,
                        ema.get_shadow() if ema else None,
                    )

                # Save best
                if running_loss < best_loss:
                    best_loss = running_loss
                    save_trainable_state(
                        unet, adapter,
                        OUTPUT_DIR / "vb_lora_v2_best.pt",
                        ema.get_shadow() if ema else None,
                    )

                # Validation generation
                if global_step % VAL_EVERY == 0:
                    logger.info(f"Generating validation samples at step {global_step}...")
                    for vi, vimg in enumerate(val_images_tensor):
                        val_img = validate_sample(
                            unet, vae, scheduler, adapter, tokenizer, text_encoder,
                            vimg, device, DEPTH_PROMPT, vae_scaling,
                            num_steps=20, seed=SEED + vi,
                        )
                        val_img.save(
                            OUTPUT_DIR / "val_during_training" / f"val_step{global_step}_{vi}.png"
                        )
                    logger.info(f"Validation samples saved")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        avg_snr = epoch_snr_weight_sum / max(epoch_steps, 1)
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} done | Avg loss: {avg_loss:.4f} | "
            f"Avg SNR weight: {avg_snr:.3f} | "
            f"Steps: {global_step} | Time: {time.time() - t_epoch_start:.0f}s"
        )

    # ── Save final weights ─────────────────────────────────────────────
    final_path = OUTPUT_DIR / "vb_lora_v2_final.pt"
    save_trainable_state(unet, adapter, final_path, ema.get_shadow() if ema else None)

    # ── Save training log ──────────────────────────────────────────────
    log_path = OUTPUT_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID,
                "task": "depth",
                "resolution": RESOLUTION,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "lr": LEARNING_RATE,
                "epochs": EPOCHS,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "use_rslora": USE_RSLORA,
                "target_modules": TARGET_MODULES,
                "seed": SEED,
                "device": str(device),
                "prompt": DEPTH_PROMPT,
                "min_snr_gamma": MIN_SNR_GAMMA,
                "use_ema": USE_EMA,
                "ema_decay": EMA_DECAY,
                "augmentation": "random_horizontal_flip",
                "dataset_size": len(dataset),
            },
            "metrics": train_log,
            "final_loss": avg_loss,
            "total_steps": global_step,
            "best_loss": best_loss,
        }, f, indent=2)
    logger.info(f"Training log saved to {log_path}")
    logger.info(f"Training complete! Final loss: {avg_loss:.4f}, Best: {best_loss:.4f}")


if __name__ == "__main__":
    main()
