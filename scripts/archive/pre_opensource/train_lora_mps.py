#!/usr/bin/env python3
"""LoRA fine-tuning of tiny-stable-diffusion on COCO128 — MPS-compatible.

This is a REAL training script (not a smoke test) that:
  1. Loads segmind/tiny-sd (323M UNet) on Apple MPS
  2. Builds a COCO128 image-caption dataset (128 samples)
  3. Applies LoRA (rank=8) to UNet attention layers (to_q, to_v)
  4. Trains with standard SD noise-prediction MSE loss
  5. Saves LoRA weights + training log

MPS-specific notes:
  - float32 only (MPS lacks reliable bf16 autocast for training)
  - torch.amp.GradScaler NOT used (MPS doesn't support CUDA amp)
  - num_workers=0 (MPS DataLoader can hang with workers)
  - torch.manual_seed() instead of torch.Generator(device='mps')
  - Gradient checkpointing enabled to fit 32GB unified memory
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# HuggingFace mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lora_train")

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "segmind/tiny-sd"
DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/coco128")
OUTPUT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/lora_output")
RESOLUTION = 256          # tiny-sd native resolution
BATCH_SIZE = 2            # conservative for 32GB MPS
GRAD_ACCUM = 4            # effective batch size = 8
LEARNING_RATE = 1e-4
EPOCHS = 5
LORA_RANK = 8
LORA_ALPHA = 16
SAVE_EVERY = 20           # save checkpoint every N steps
LOG_EVERY = 5
SEED = 42
NUM_WORKERS = 0           # MPS-safe


# ─── Dataset ─────────────────────────────────────────────────────────────
class COCO128Dataset(Dataset):
    """COCO128 image-caption dataset for SD training."""

    def __init__(self, data_dir: Path, resolution: int = 256, tokenizer=None):
        self.resolution = resolution
        self.tokenizer = tokenizer

        captions_path = data_dir / "captions.json"
        with open(captions_path) as f:
            self.pairs = json.load(f)

        self.image_dir = data_dir / "images"
        logger.info(f"Dataset: {len(self.pairs)} samples from {self.image_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        img_path = self.image_dir / pair["filename"]

        # Load + resize image → tensor [3, H, W] in [-1, 1]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        pixel_values = pixel_values * 2.0 - 1.0  # → [-1, 1]

        # Tokenize caption → [seq_len] (tokenizer will handle padding)
        caption = pair["caption"]
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)  # [77]
        else:
            tokens = torch.zeros(77, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens,
            "caption": caption,
        }


def collate_fn(batch: list[dict]) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


# ─── Training ────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Resolution: {RESOLUTION}, Batch: {BATCH_SIZE}×{GRAD_ACCUM}, LR: {LEARNING_RATE}")
    logger.info(f"LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}")
    logger.info(f"Epochs: {EPOCHS}, Output: {output_dir}")

    # ── Load pipeline components ──────────────────────────────────────
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

    # Enable gradient checkpointing on UNet
    unet.enable_gradient_checkpointing()

    logger.info(f"UNet params: {sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M")

    # ── Apply LoRA ────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable} ({trainable / sum(p.numel() for p in unet.parameters()) * 100:.2f}%)")

    # ── Build dataset & dataloader ────────────────────────────────────
    dataset = COCO128Dataset(DATA_DIR, RESOLUTION, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"DataLoader: {len(dataloader)} batches/epoch × {EPOCHS} epochs")

    # ── Optimizer & scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    warmup_steps = min(50, total_steps // 5)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}")

    # ── Training loop ────────────────────────────────────────────────
    train_log = []
    global_step = 0
    best_loss = float("inf")

    # VAE scaling factor
    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch_start = time.time()

        unet.train()
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)  # [B, 3, H, W] in [-1, 1]
            input_ids = batch["input_ids"].to(device)  # [B, 77]

            # ── Encode images to latents with VAE ────────────────────
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae_scaling  # [B, 4, H/8, W/8]

            # ── Encode text ──────────────────────────────────────────
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]  # [B, 77, 768]

            # ── Sample noise & timestep ─────────────────────────────
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            # ── Add noise (forward diffusion) ───────────────────────
            # SD scheduler: noisy = alpha_t * latent + sigma_t * noise
            # But diffusers schedulers have different add_noise implementations.
            # Use the scheduler's add_noise method:
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # ── Predict noise (UNet forward) ────────────────────────
            # SD 1.x UNet: predicts noise ε
            predicted_noise = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
            ).sample  # [B, 4, H/8, W/8]

            # ── MSE loss ────────────────────────────────────────────
            loss = F.mse_loss(predicted_noise, noise)
            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()

            epoch_loss += loss.item()
            epoch_steps += 1

            # ── Optimizer step (gradient accumulation) ──────────────
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in unet.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
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
                    save_lora(unet, output_dir / f"lora_step{global_step}.pt")
                    logger.info(f"  Saved checkpoint: lora_step{global_step}.pt (loss={running_loss:.4f})")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} done | Avg loss: {avg_loss:.4f} | "
            f"Steps: {global_step} | Time: {time.time() - t_epoch_start:.0f}s"
        )

    # ── Save final LoRA weights ──────────────────────────────────────
    save_lora(unet, output_dir / "lora_final.pt")
    logger.info(f"Final LoRA saved to {output_dir / 'lora_final.pt'}")

    # ── Save training log ────────────────────────────────────────────
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID,
                "resolution": RESOLUTION,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "lr": LEARNING_RATE,
                "epochs": EPOCHS,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "seed": SEED,
                "device": str(device),
            },
            "metrics": train_log,
            "final_loss": avg_loss,
            "total_steps": global_step,
        }, f, indent=2)
    logger.info(f"Training log saved to {log_path}")
    logger.info(f"Training complete! Final loss: {avg_loss:.4f}, Best: {best_loss:.4f}")


def save_lora(peft_model, path: Path):
    """Save only LoRA adapter weights."""
    lora_state = {}
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu().clone()
    torch.save(lora_state, path)


if __name__ == "__main__":
    main()
