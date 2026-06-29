#!/usr/bin/env python3
"""Vision Banana LoRA training — depth task on tiny-sd, MPS-compatible.

Key differences from the previous generic T2I LoRA training:
  1. Data: (image, depth_rgb) paired samples, not (image, caption)
  2. LoRA: rsLoRA rank=64, alpha=128 (not rank=8, alpha=16)
  3. Conditioning: image latents concatenated with noisy target latents
  4. Prompt: task-specific depth instruction prompt
  5. Loss: MSE on noise prediction for the depth RGB target

Architecture:
  - Input image → VAE encode → input_latents [B, 4, 32, 32]
  - Depth target → VAE encode → target_latents [B, 4, 32, 32]
  - noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
  - UNet input = concat(input_latents, noisy_latents) [B, 8, 32, 32]
  - UNet predicts noise, loss = MSE(predicted, noise)

  Since tiny-sd UNet expects 4-channel input, we use a learnable conv2d
  adapter (4→4) on the concatenated 8-channel input, keeping the UNet frozen.
  The LoRA adapters + the conv adapter are the only trainable parameters.

MPS-specific:
  - float32 only
  - No GradScaler (MPS doesn't support CUDA amp)
  - num_workers=0
  - Gradient checkpointing enabled
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

# HuggingFace mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Add project to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_train")

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "segmind/tiny-sd"
DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vision_banana_depth")
OUTPUT_DIR = Path(
    "/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_depth_lora"
)
RESOLUTION = 256          # tiny-sd native resolution
BATCH_SIZE = 1            # conservative for 32GB MPS with 8-channel input
GRAD_ACCUM = 8            # effective batch size = 8
LEARNING_RATE = 5e-5      # paper recommended LR for rsLoRA
EPOCHS = 10
LORA_RANK = 64            # rsLoRA rank from paper
LORA_ALPHA = 128          # rsLoRA alpha = 2 * rank
USE_RSLORA = True         # rsLoRA scaling
SAVE_EVERY = 10           # save checkpoint every N optimizer steps
LOG_EVERY = 5
SEED = 42
NUM_WORKERS = 0           # MPS-safe

DEPTH_PROMPT = "Generate a metric depth visualization of this image."

# Target modules — expanded for better coverage
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


# ─── Dataset ─────────────────────────────────────────────────────────────
class VisionBananaDepthDataset(Dataset):
    """Paired (image, depth_rgb) dataset for Vision Banana depth training."""

    def __init__(self, data_dir: Path, resolution: int = 256):
        self.resolution = resolution
        self.image_dir = data_dir / "images"
        self.target_dir = data_dir / "targets"

        self.samples = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            tgt_path = self.target_dir / img_path.name
            if tgt_path.exists():
                self.samples.append((img_path, tgt_path))

        logger.info(f"Dataset: {len(self.samples)} paired samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, tgt_path = self.samples[idx]

        # Load input image → [3, H, W] in [-1, 1] (SD convention)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0

        # Load depth target → [3, H, W] in [-1, 1] (for VAE encoding)
        tgt = Image.open(tgt_path).convert("RGB")
        tgt = tgt.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr_t = np.array(tgt, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(arr_t).permute(2, 0, 1) * 2.0 - 1.0

        return {
            "image": image_tensor,
            "target_rgb": target_tensor,
            "prompt": DEPTH_PROMPT,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "target_rgb": torch.stack([b["target_rgb"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
    }


# ─── Channel Adapter ────────────────────────────────────────────────────
class ChannelAdapter(nn.Module):
    """Learnable 8→4 channel adapter for concatenated input.

    Since SD 1.x UNet expects 4-channel latent input, but we concatenate
    input_latents (4ch) + noisy_target_latents (4ch) = 8ch, we need this
    adapter to project back to 4 channels before the UNet.

    This is the ONLY non-LoRA trainable parameter.
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        # Initialise close to identity for the noisy_latents half (channels 4:8)
        # weight shape: [out_ch, in_ch, 1, 1]
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, out_channels:, 0, 0] = torch.eye(out_channels)
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ─── Training ────────────────────────────────────────────────────────────
def save_trainable_state(unet, adapter, path: Path):
    """Save LoRA adapter weights + channel adapter weights."""
    state = {}

    # Save LoRA weights
    for name, param in unet.named_parameters():
        if param.requires_grad:
            state[f"unet.{name}"] = param.data.cpu().clone()

    # Save channel adapter
    for name, param in adapter.named_parameters():
        state[f"adapter.{name}"] = param.data.cpu().clone()

    torch.save(state, path)
    logger.info(f"Saved {len(state)} trainable params to {path.name}")


def main():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Config: rank={LORA_RANK}, alpha={LORA_ALPHA}, rsLoRA={USE_RSLORA}")
    logger.info(f"Resolution: {RESOLUTION}, Batch: {BATCH_SIZE}×{GRAD_ACCUM}, LR: {LEARNING_RATE}")
    logger.info(f"Epochs: {EPOCHS}, Output: {OUTPUT_DIR}")

    # ── Load pipeline ───────────────────────────────────────────────────
    from diffusers import StableDiffusionPipeline

    logger.info("Loading model (segmind/tiny-sd)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )

    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler

    # Freeze base model
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()

    unet_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"UNet params: {unet_params / 1e6:.1f}M")

    # ── Apply rsLoRA ────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
        use_rslora=USE_RSLORA,
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ── Channel adapter ─────────────────────────────────────────────────
    adapter = ChannelAdapter(in_channels=8, out_channels=4).to(device)
    adapter_params = sum(p.numel() for p in adapter.parameters())
    logger.info(f"Channel adapter params: {adapter_params} ({adapter_params / 1e3:.1f}K)")

    # ── Build dataset & dataloader ──────────────────────────────────────
    dataset = VisionBananaDepthDataset(DATA_DIR, RESOLUTION)
    if len(dataset) == 0:
        raise RuntimeError(f"No paired samples found in {DATA_DIR}")

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"DataLoader: {len(dataloader)} batches/epoch × {EPOCHS} epochs")

    # ── Optimizer & scheduler ───────────────────────────────────────────
    trainable_params = (
        [p for p in unet.parameters() if p.requires_grad]
        + list(adapter.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    warmup_steps = min(20, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    # ── Training loop ───────────────────────────────────────────────────
    train_log = []
    global_step = 0
    best_loss = float("inf")

    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)
    num_train_timesteps = scheduler.config.num_train_timesteps

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        unet.train()
        adapter.train()

        for step, batch in enumerate(dataloader):
            images = batch["image"].to(device)        # [B, 3, 256, 256] in [-1, 1]
            targets = batch["target_rgb"].to(device)   # [B, 3, 256, 256] in [-1, 1]
            prompts = batch["prompt"]

            bsz = images.shape[0]

            # ── Encode images and targets to latents ─────────────────
            with torch.no_grad():
                # Input image latents
                input_latents = vae.encode(images).latent_dist.sample()
                input_latents = input_latents * vae_scaling  # [B, 4, 32, 32]

                # Target (depth RGB) latents
                target_latents = vae.encode(targets).latent_dist.sample()
                target_latents = target_latents * vae_scaling  # [B, 4, 32, 32]

            # ── Encode text prompt ────────────────────────────────────
            with torch.no_grad():
                tokenized = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(device)
                encoder_hidden_states = text_encoder(tokenized)[0]  # [B, 77, 768]

            # ── Sample noise & timestep ───────────────────────────────
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0, num_train_timesteps, (bsz,), device=device
            ).long()

            # ── Add noise to target latents ───────────────────────────
            noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)

            # ── Concatenate input + noisy target → 8 channels ─────────
            concat_latents = torch.cat([input_latents, noisy_latents], dim=1)
            adapted_latents = adapter(concat_latents)  # [B, 4, 32, 32]

            # ── UNet forward: predict noise ───────────────────────────
            predicted_noise = unet(
                adapted_latents,
                timesteps,
                encoder_hidden_states,
            ).sample  # [B, 4, 32, 32]

            # ── MSE loss ──────────────────────────────────────────────
            loss = F.mse_loss(predicted_noise, noise)
            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()

            epoch_loss += loss.item()
            epoch_steps += 1

            # ── Optimizer step ────────────────────────────────────────
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler_lr.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

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

                if global_step % SAVE_EVERY == 0 and running_loss < best_loss:
                    best_loss = running_loss
                    save_trainable_state(
                        unet, adapter,
                        OUTPUT_DIR / f"vb_lora_step{global_step}.pt",
                    )

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} done | Avg loss: {avg_loss:.4f} | "
            f"Steps: {global_step} | Time: {time.time() - t_epoch_start:.0f}s"
        )

    # ── Save final weights ─────────────────────────────────────────────
    final_path = OUTPUT_DIR / "vb_lora_final.pt"
    save_trainable_state(unet, adapter, final_path)

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
