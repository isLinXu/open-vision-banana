#!/usr/bin/env python3
"""Paper-aligned LoRA training v3 — Depth-Anything distillation.

Key improvements over v2:
  1. Distilled dataset: 428 Depth-Anything depth maps as targets (clean signal)
  2. Longer training: 10 epochs (~2140 optimizer steps)
  3. Depth structure correlation (Pearson r) as core validation metric
  4. Multi-strength validation sweep: 0.4 / 0.6 / 0.8
  5. Better LR schedule: warmup + cosine decay matched to step count

Inherited from v2: logit-normal timestep, Min-SNR-gamma, EMA 0.999, augmentation
"""

from __future__ import annotations

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("vb_v3")

# Config
MODEL_ID = "segmind/tiny-sd"
VISION_DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_distill_428")
GENERATION_DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/coco128")
OUTPUT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_paper_aligned_v3")
RESOLUTION = 256
BATCH_SIZE = 1
GRAD_ACCUM = 2
LEARNING_RATE = 1e-4
EPOCHS = 10
LORA_RANK = 64
LORA_ALPHA = 128
USE_RSLORA = True
VISION_RATIO = 0.1
MAX_SAMPLES_PER_EPOCH = 428
SAVE_EVERY = 100
LOG_EVERY = 20
VAL_EVERY = 200
SEED = 42
NUM_WORKERS = 0
EMA_DECAY = 0.999
USE_MIN_SNR = True
MIN_SNR_GAMMA = 5.0
USE_LOGIT_NORMAL = True
LOGIT_MEAN = 0.0
LOGIT_STD = 1.0
DEPTH_PROMPT = "Generate a metric depth visualization of this image."
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"]

# ─── Timestep sampling ────────────────────────────────────────────────────
def sample_timesteps_logit_normal(batch_size, device, mean=0.0, std=1.0):
    noise = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(noise)

def compute_min_snr_weight(t, gamma=5.0):
    snr = (1.0 - t) ** 2 / (t ** 2 + 1e-8)
    weight = torch.clamp(snr, max=gamma) / snr
    return weight


# ─── Datasets ─────────────────────────────────────────────────────────────
class VisionDepthDataset(Dataset):
    def __init__(self, data_dir, resolution=256, augment=True):
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
        logger.info(f"VisionDepthDataset (distilled): {len(self.samples)} paired samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tgt_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0
        tgt = Image.open(tgt_path).convert("RGB")
        tgt = tgt.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr_t = np.array(tgt, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(arr_t).permute(2, 0, 1)
        if self.augment:
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
                image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
                target_tensor = F.interpolate(target_tensor.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
            if np.random.random() < 0.5:
                brightness = 1.0 + np.random.uniform(-0.1, 0.1)
                image_tensor = (image_tensor * brightness).clamp(-1.0, 1.0)
        return {"image": image_tensor, "target_rgb": target_tensor, "prompt": self.prompt, "task": "depth", "filename": img_path.stem}


class GenerationDataset(Dataset):
    def __init__(self, data_dir, resolution=256):
        self.resolution = resolution
        self.image_dir = data_dir / "images"
        captions_path = data_dir / "captions.json"
        with open(captions_path) as f:
            self.pairs = json.load(f)
        self.samples = []
        for pair in self.pairs:
            img_path = self.image_dir / pair["filename"]
            if img_path.exists():
                self.samples.append({"image_path": str(img_path), "prompt": pair["caption"]})
        logger.info(f"GenerationDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        target_tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return {"target_rgb": target_tensor, "prompt": sample["prompt"], "task": "generation", "filename": Path(sample["image_path"]).stem}


class MixedDataset(Dataset):
    def __init__(self, vision_ds, gen_ds, vision_ratio=0.1, seed=42):
        self.vision_ds = vision_ds
        self.gen_ds = gen_ds
        self.vision_ratio = vision_ratio
        self.rng = torch.Generator().manual_seed(seed)
        self.total_vision = len(vision_ds)
        self.total_gen = len(gen_ds)

    def __len__(self):
        if self.gen_ds is None or self.vision_ratio >= 1.0:
            return self.total_vision
        total = max(1, round(self.total_vision / self.vision_ratio))
        return min(total, MAX_SAMPLES_PER_EPOCH)

    def __getitem__(self, idx):
        use_vision = (self.gen_ds is None or torch.rand(1, generator=self.rng).item() < self.vision_ratio)
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
    state = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            state[f"unet.{name}"] = param.data.cpu().clone()
    if ema_state is not None:
        for k, v in ema_state.items():
            state[f"ema.{k}"] = v.cpu().clone()
    torch.save(state, path)
    logger.info(f"Saved {len(state)} keys to {path.name}")

def cleanup_old_checkpoints(output_dir, keep_latest=1):
    ckpts = sorted(output_dir.glob("vb_v3_step*.pt"), key=lambda p: p.stat().st_mtime)
    if len(ckpts) > keep_latest:
        for old in ckpts[:-keep_latest]:
            old.unlink()
            logger.info(f"Cleaned up old checkpoint: {old.name}")


# ─── v3: Depth structure correlation ──────────────────────────────────────
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


def compute_depth_metrics_v3(gen_img, target_img):
    gen_arr = np.array(gen_img, dtype=np.float32) / 255.0
    tgt_arr = np.array(target_img, dtype=np.float32) / 255.0
    mse = np.mean((gen_arr - tgt_arr) ** 2)
    mae = np.mean(np.abs(gen_arr - tgt_arr))
    gen_std = np.std(gen_arr)
    tgt_std = np.std(tgt_arr)
    correlation = compute_depth_correlation(gen_img, target_img)
    return {"mse": float(mse), "mae": float(mae), "gen_std": float(gen_std),
            "tgt_std": float(tgt_std), "correlation": correlation}


# ─── Flow matching inference ──────────────────────────────────────────────
@torch.no_grad()
def generate_depth_sample_flow(unet, vae, text_encoder, tokenizer,
    input_image, prompt, device, vae_scaling,
    num_steps=30, seed=42, strength=0.6, guidance_scale=1.0):
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
    unet.train()
    return Image.fromarray(image_np)


# ─── Main training ────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val_during_training").mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("Vision Banana — Paper-Aligned Training v3 (Depth-Anything Distillation)")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_ID} (proxy for FLUX.1)")
    logger.info(f"Device: {device}, Resolution: {RESOLUTION}")
    logger.info(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, rsLoRA={USE_RSLORA}")
    logger.info("Training objective: Flow matching (v-prediction)")
    logger.info(f"Data mix: {VISION_RATIO:.0%} vision (distilled) + {1-VISION_RATIO:.0%} generation")
    logger.info(f"Batch: {BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, LR: {LEARNING_RATE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info("v3 improvements:")
    logger.info("  - Distilled dataset: 428 Depth-Anything depth maps (clean signal)")
    logger.info("  - 10 epochs (~2140 optimizer steps, 3.5x v2)")
    logger.info("  - Depth structure correlation (Pearson r) validation")
    logger.info("  - Multi-strength validation sweep: 0.4 / 0.6 / 0.8")
    logger.info("=" * 60)

    # Load pipeline
    from diffusers import StableDiffusionPipeline
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
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    unet_params = sum(p.numel() for p in unet.parameters())
    logger.info(f"UNet params: {unet_params / 1e6:.1f}M")

    # Apply rsLoRA
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none",
        use_rslora=USE_RSLORA,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(f"Trainable: {trainable} ({trainable/total*100:.2f}%)")

    # Build datasets
    vision_ds = VisionDepthDataset(VISION_DATA_DIR, RESOLUTION, augment=True)
    gen_ds = GenerationDataset(GENERATION_DATA_DIR, RESOLUTION)
    mixed_ds = MixedDataset(vision_ds, gen_ds, vision_ratio=VISION_RATIO, seed=SEED)
    logger.info(f"Mixed dataset: {len(mixed_ds)} samples/epoch")
    dataloader = DataLoader(mixed_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=True)
    logger.info(f"DataLoader: {len(dataloader)} batches/epoch x {EPOCHS} epochs")

    # Optimizer & scheduler
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    logger.info(f"Total optimizer steps: {total_steps}, warmup: {warmup_steps}")
    logger.info(f"Estimated time: ~{total_steps * 8 / 60:.0f} minutes")

    # EMA
    ema = LoRAEMA(decay=EMA_DECAY)
    ema.register([(n, p) for n, p in unet.named_parameters() if p.requires_grad])
    logger.info(f"EMA registered for {len(ema.shadow)} params (decay={EMA_DECAY})")

    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)

    # Validation image pool — use distilled dataset images + targets
    val_image_paths = sorted((VISION_DATA_DIR / "images").glob("nyu_*.jpg"))[:6]
    val_images = []
    val_targets = []
    for p in val_image_paths:
        img = Image.open(p).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        val_images.append(t)
        tgt_name = p.stem + ".png"
        tgt_path = VISION_DATA_DIR / "targets" / tgt_name
        if not tgt_path.exists():
            tgt_path = VISION_DATA_DIR / "targets" / p.name
        if tgt_path.exists():
            tgt = Image.open(tgt_path).convert("RGB").resize((RESOLUTION, RESOLUTION), Image.LANCZOS)
            val_targets.append(tgt)

    # Training loop
    train_log = []
    val_log = []
    best_loss = float("inf")
    best_val_corr = -1.0  # v3: track correlation instead of MSE
    t_train_start = time.time()
    global_step = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_vision_count = 0
        epoch_gen_count = 0
        t_epoch_start = time.time()
        unet.train()

        for step, batch in enumerate(dataloader):
            target_rgb = batch["target_rgb"].to(device)
            prompts = batch["prompt"]
            tasks = batch["task"]
            bsz = target_rgb.shape[0]

            with torch.no_grad():
                target_pixels = target_rgb * 2.0 - 1.0
                target_latents = vae.encode(target_pixels).latent_dist.mode()
                target_latents = target_latents * vae_scaling

            if USE_LOGIT_NORMAL:
                timesteps = sample_timesteps_logit_normal(bsz, device, LOGIT_MEAN, LOGIT_STD)
            else:
                timesteps = torch.rand(bsz, device=device)

            noise = torch.randn_like(target_latents)
            t_b = timesteps.view(bsz, 1, 1, 1)
            noisy_latents = (1.0 - t_b) * target_latents + t_b * noise
            velocity_target = target_latents - noise

            with torch.no_grad():
                tokenized = tokenizer(prompts, padding="max_length", max_length=77,
                                      truncation=True, return_tensors="pt").input_ids.to(device)
                encoder_hidden_states = text_encoder(tokenized)[0]

            timestep_long = (timesteps * 999).long().clamp(0, 999)
            predicted_velocity = unet(noisy_latents, timestep_long, encoder_hidden_states).sample

            if USE_MIN_SNR:
                snr_weights = compute_min_snr_weight(timesteps, MIN_SNR_GAMMA)
                snr_weights = snr_weights.view(bsz, 1, 1, 1)
                loss_per_sample = F.mse_loss(predicted_velocity, velocity_target, reduction="none")
                loss = (loss_per_sample * snr_weights).mean()
            else:
                loss = F.mse_loss(predicted_velocity, velocity_target)

            loss_scaled = loss / GRAD_ACCUM
            loss_scaled.backward()
            epoch_loss += loss.item()
            epoch_steps += 1
            for task in tasks:
                if task == "depth":
                    epoch_vision_count += 1
                else:
                    epoch_gen_count += 1

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler_lr.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                ema.update([(n, p) for n, p in unet.named_parameters() if p.requires_grad])

                current_lr = scheduler_lr.get_last_lr()[0]
                running_loss = epoch_loss / epoch_steps

                if global_step % LOG_EVERY == 0:
                    elapsed = time.time() - t_epoch_start
                    eta = (time.time() - t_train_start) / global_step * (total_steps - global_step)
                    logger.info(f"E{epoch+1}/{EPOCHS} | S{global_step}/{total_steps} | "
                                f"Loss: {running_loss:.4f} | LR: {current_lr:.2e} | "
                                f"V/G: {epoch_vision_count}/{epoch_gen_count} | "
                                f"{elapsed:.0f}s | ETA: {eta/60:.0f}m")
                    train_log.append({"step": global_step, "epoch": epoch+1, "loss": running_loss,
                                      "lr": current_lr, "vision_count": epoch_vision_count,
                                      "gen_count": epoch_gen_count})

                if global_step % SAVE_EVERY == 0:
                    ckpt_path = OUTPUT_DIR / f"vb_v3_step{global_step}.pt"
                    save_lora_state(unet, ckpt_path, ema.get_shadow())
                    cleanup_old_checkpoints(OUTPUT_DIR, keep_latest=1)

                if running_loss < best_loss * 0.99:
                    best_loss = running_loss
                    if global_step % SAVE_EVERY == 0 or global_step <= 20:
                        save_lora_state(unet, OUTPUT_DIR / "vb_v3_best.pt", ema.get_shadow())

                # v3: Multi-strength validation with correlation metric
                if global_step % VAL_EVERY == 0 and global_step <= total_steps:
                    logger.info(f"Validation at step {global_step} (correlation sweep)...")
                    val_results = []
                    for vi, vimg in enumerate(val_images[:3]):
                        best_corr = -2.0
                        best_strength = 0.6
                        best_img = None
                        for strength in [0.4, 0.6, 0.8]:
                            try:
                                gen_img = generate_depth_sample_flow(
                                    unet, vae, text_encoder, tokenizer,
                                    vimg, DEPTH_PROMPT, device, vae_scaling,
                                    num_steps=20, seed=SEED+vi, strength=strength,
                                    guidance_scale=2.0,
                                )
                                if vi < len(val_targets):
                                    metrics = compute_depth_metrics_v3(gen_img, val_targets[vi])
                                    if metrics["correlation"] > best_corr:
                                        best_corr = metrics["correlation"]
                                        best_strength = strength
                                        best_img = gen_img
                                        best_metrics = metrics
                            except Exception as e:
                                logger.warning(f"  Val s={strength} failed: {e}")
                                continue
                        if best_img is not None:
                            best_img.save(OUTPUT_DIR / "val_during_training" / f"val_step{global_step}_{vi}_s{best_strength}.png")
                            val_results.append(best_metrics)

                    if val_results:
                        avg_corr = np.mean([m["correlation"] for m in val_results])
                        avg_mse = np.mean([m["mse"] for m in val_results])
                        avg_gen_std = np.mean([m["gen_std"] for m in val_results])
                        avg_tgt_std = np.mean([m["tgt_std"] for m in val_results])
                        logger.info(f"  Val: corr={avg_corr:.4f} | MSE={avg_mse:.4f} | "
                                    f"gen_std={avg_gen_std:.4f} | tgt_std={avg_tgt_std:.4f}")
                        val_log.append({"step": global_step, "avg_correlation": float(avg_corr),
                                        "avg_mse": float(avg_mse), "avg_gen_std": float(avg_gen_std),
                                        "avg_tgt_std": float(avg_tgt_std)})
                        if avg_corr > best_val_corr:
                            best_val_corr = avg_corr
                            save_lora_state(unet, OUTPUT_DIR / "vb_v3_best_corr.pt", ema.get_shadow())
                            logger.info(f"  New best corr: {best_val_corr:.4f}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - t_epoch_start
        logger.info(f"Epoch {epoch+1}/{EPOCHS} done | Avg loss: {avg_loss:.4f} | "
                    f"V/G: {epoch_vision_count}/{epoch_gen_count} | "
                    f"Steps: {global_step} | Time: {elapsed:.0f}s")

    # Save final
    save_lora_state(unet, OUTPUT_DIR / "vb_v3_final.pt", ema.get_shadow())
    total_time = time.time() - t_train_start
    logger.info(f"Training complete! Total time: {total_time/60:.1f} min")
    logger.info(f"Final loss: {avg_loss:.4f}, Best train: {best_loss:.4f}, Best val corr: {best_val_corr:.4f}")

    # Save training log
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID, "version": "v3", "resolution": RESOLUTION,
                "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM, "lr": LEARNING_RATE,
                "epochs": EPOCHS, "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
                "use_rslora": USE_RSLORA, "target_modules": TARGET_MODULES, "seed": SEED,
                "device": str(device), "vision_ratio": VISION_RATIO,
                "training_objective": "flow_matching_v_prediction",
                "ema_decay": EMA_DECAY, "use_logit_normal": USE_LOGIT_NORMAL,
                "use_min_snr": USE_MIN_SNR, "min_snr_gamma": MIN_SNR_GAMMA,
                "dataset": "Depth-Anything distilled (428 images)",
                "total_steps": total_steps,
                "improvements_over_v2": [
                    "Depth-Anything distilled targets (clean structural signal)",
                    "10 epochs / ~2140 steps (3.5x v2)",
                    "Pearson correlation depth structure metric",
                    "Multi-strength validation sweep (0.4/0.6/0.8)",
                ],
            },
            "train_metrics": train_log, "val_metrics": val_log,
            "final_loss": avg_loss, "best_train_loss": best_loss,
            "best_val_correlation": best_val_corr,
            "total_steps": global_step, "total_time_s": total_time,
        }, f, indent=2)
    logger.info(f"Training log saved to {OUTPUT_DIR / 'training_log.json'}")


if __name__ == "__main__":
    main()
