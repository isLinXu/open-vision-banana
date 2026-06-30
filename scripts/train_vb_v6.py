#!/usr/bin/env python3
"""Vision Banana v6 — Optimized multi-backbone training with deep optimizations.

Key improvements over v5:
  1. MixedVisionDataset (multi-dataset, multi-colormap support)
  2. AdvancedAugmentation (color jitter, gaussian noise, cutout)
  3. MemoryManager (adaptive MPS/CUDA memory clearing)
  4. SchedulerFactory (cosine_restarts, polynomial, one_cycle options)
  5. create_optimized_dataloader (device-aware num_workers/pin_memory)
  6. torch.compile support (CUDA only)
  7. Multi-task vision data mixing (depth + normal + semantic in one run)
  8. Richer validation with multiple metrics

Usage:
  python scripts/train_vb_v6.py --model ssd-1b --scheduler cosine_restarts
  python scripts/train_vb_v6.py --model ssd-1b --vision-tasks depth,normal --multi-task
  python scripts/train_vb_v6.py --model ssd-1b --augment advanced --compile
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
from torch.utils.data import Dataset

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_vision_banana.model_registry import (
    ModelConfig,
    PipelineComponents,
    apply_lora,
    auto_select_model,
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
from open_vision_banana.data.mixed import MixedVisionDataset
from open_vision_banana.data.datasets import build_dataset
from open_vision_banana.training.optimization import (
    AdvancedAugmentation,
    MemoryManager,
    SchedulerFactory,
    create_optimized_dataloader,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_v6")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSION_OUTPUT = Path(os.environ.get("VB_OUTPUT_DIR", str(PROJECT_ROOT / "output" / "vb_v6")))

DEPTH_PROMPT = "Generate a metric depth visualization of this image."


# ─── Datasets with Advanced Augmentation ───────────────────────────────────


class AugmentedVisionDataset(Dataset):
    """Wrap a vision dataset with advanced augmentation."""

    def __init__(self, base_dataset, augmenter: AdvancedAugmentation | None = None):
        self.base = base_dataset
        self.augmenter = augmenter or AdvancedAugmentation()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if self.augmenter is not None and "image" in sample and "target_rgb" in sample:
            image = sample["image"]
            target = sample["target_rgb"]
            # Ensure image is in [-1, 1] and target in [0, 1]
            image_aug, target_aug = self.augmenter(image, target)
            sample = dict(sample)
            sample["image"] = image_aug
            sample["target_rgb"] = target_aug
        return sample


class GenerationDataset(Dataset):
    """Text-to-image generation samples for mixture preservation."""

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

        if not self.samples:
            for img_path in sorted(self.image_dir.glob("*.jpg")):
                self.samples.append({
                    "image_path": str(img_path),
                    "prompt": "A high quality, photorealistic image.",
                })

        logger.info(f"GenerationDataset: {len(self.samples)} samples")

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
            "image": target_tensor * 2.0 - 1.0,
            "prompt": sample["prompt"],
            "task": "generation",
            "filename": Path(sample["image_path"]).stem,
        }


def collate_fn(batch):
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            result[key] = [b[key] for b in batch]
    return result


# ─── EMA ────────────────────────────────────────────────────────────────────


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


# ─── Depth metrics ──────────────────────────────────────────────────────────


def compute_depth_correlation(gen_img, target_img):
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


# ─── Timestep & Flow matching ───────────────────────────────────────────────


def sample_timesteps_logit_normal(batch_size, device, mean=0.0, std=1.0):
    noise = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(noise)


@torch.no_grad()
def generate_depth_img2img(components, input_image, prompt, device, dtype, num_steps=30, seed=42, strength=0.6, guidance_scale=2.0):
    config = components.config
    backbone = components.unet
    vae = components.vae
    backbone.eval()

    encoder_hidden_states, aux_output = encode_text(components, [prompt], device, dtype)
    attention_mask = aux_output if isinstance(aux_output, torch.Tensor) and config.needs_attention_mask else None
    pooled_embeds = aux_output if isinstance(aux_output, torch.Tensor) and not config.needs_attention_mask else None

    res = (input_image.shape[-2], input_image.shape[-1])
    added_cond_kwargs = build_added_cond_kwargs(config, pooled_embeds, res, device, dtype, batch_size=1)
    txt_ids, img_ids = prepare_position_ids(None, encoder_hidden_states, config, device, dtype)

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

    for i in range(num_actual):
        t = i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)
        if config.needs_position_ids:
            txt_ids, img_ids = prepare_position_ids(latents, encoder_hidden_states, config, device, dtype)

        if guidance_scale > 1.0 and uncond_hidden is not None:
            v_cond = backbone_forward(backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs, guidance=guidance_scale,
                attention_mask=attention_mask, pooled_embeds=pooled_embeds, txt_ids=txt_ids, img_ids=img_ids)
            v_uncond = backbone_forward(backbone, latents, timestep_long, uncond_hidden, config,
                added_cond_kwargs=uncond_added_cond, guidance=guidance_scale,
                attention_mask=uncond_mask, pooled_embeds=pooled_embeds, txt_ids=txt_ids, img_ids=img_ids)
            pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_v = backbone_forward(backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs, guidance=guidance_scale,
                attention_mask=attention_mask, pooled_embeds=pooled_embeds, txt_ids=txt_ids, img_ids=img_ids)
        latents = latents + dt * pred_v

    image = vae_decode(vae, latents, config)
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    backbone.train()
    return Image.fromarray(image_np)


# ─── Main Training ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Vision Banana v6 — Optimized training")
    parser.add_argument("--model", type=str, default="ssd-1b")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--auto-select", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--vision-ratio", type=float, default=0.1)
    parser.add_argument("--vision-tasks", type=str, default="depth",
                        help="Comma-separated vision tasks: depth,normal,semantic")
    parser.add_argument("--multi-task", action="store_true",
                        help="Enable multi-task vision dataset mixing")
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-rslora", action="store_true", default=True)
    parser.add_argument("--img2img-strength", type=float, default=0.6)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "cosine_restarts", "polynomial", "constant", "one_cycle"])
    parser.add_argument("--augment", type=str, default="advanced",
                        choices=["none", "basic", "advanced"])
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (CUDA only)")
    parser.add_argument("--memory-threshold", type=float, default=0.85,
                        help="Memory clear threshold (0-1)")
    parser.add_argument("--colormaps", type=str, default="cube_edge,rainbow,viridis",
                        help="Comma-separated colormaps for depth training")
    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        print("-" * 80)
        for name in list_models():
            cfg = get_model_config(name)
            print(f"  {name:15s} | {cfg.description}")
            print(f"  {'':15s} | family={cfg.family}, params~{cfg.memory_budget_gb}GB")
            print()
        return

    if args.auto_select:
        if torch.backends.mps.is_available():
            mem_gb = 32.0
        elif torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        else:
            mem_gb = 8.0
        selected = auto_select_model(mem_gb)
        logger.info(f"Auto-selected: {selected} (memory: {mem_gb:.0f}GB)")
        args.model = selected

    model_cfg = get_model_config(args.model)
    resolution = args.resolution or model_cfg.default_resolution
    total_steps = args.steps or model_cfg.default_steps
    lora_rank = args.lora_rank or model_cfg.default_lora_rank
    lora_alpha = args.lora_alpha or model_cfg.default_lora_alpha

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = SESSION_OUTPUT / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "val_during_training").mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16

    logger.info("=" * 70)
    logger.info("Vision Banana v6 — Optimized Training")
    logger.info("=" * 70)
    logger.info(f"Model: {model_cfg.model_id} ({model_cfg.name})")
    logger.info(f"Family: {model_cfg.family}, Encoders: {model_cfg.num_text_encoders}")
    logger.info(f"Device: {device}, Dtype: {dtype}, Resolution: {resolution}")
    logger.info(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, rsLoRA={args.use_rslora}")
    logger.info(f"Scheduler: {args.scheduler}, Augment: {args.augment}")
    logger.info(f"Multi-task: {args.multi_task}, Vision tasks: {args.vision_tasks}")
    logger.info(f"Colormaps: {args.colormaps}")
    logger.info("=" * 70)

    # ── Memory Manager ──
    mem_manager = MemoryManager(device, threshold_gb=args.memory_threshold)

    # ── Load pipeline ──
    components = load_pipeline(model_cfg, device, dtype, enable_gradient_checkpointing=True)

    # ── torch.compile (CUDA only) ──
    if args.compile and device.type == "cuda":
        try:
            components.unet = torch.compile(components.unet, mode="reduce-overhead")
            logger.info("torch.compile enabled for UNet")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # ── Apply LoRA ──
    components.unet = apply_lora(components.unet, model_cfg, rank=lora_rank, alpha=lora_alpha, use_rslora=args.use_rslora)
    trainable = sum(p.numel() for p in components.unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in components.unet.parameters())
    logger.info(f"Trainable: {trainable} ({trainable/total*100:.2f}%)")

    unet = components.unet
    vae = components.vae

    # ── Build multi-task vision datasets ──
    vision_tasks = [t.strip() for t in args.vision_tasks.split(",")]
    vision_datasets = []

    for task in vision_tasks:
        task_dir = DATA_DIR / f"vb_v6_{task}"
        if task_dir.exists():
            try:
                ds = build_dataset(task, root_dir=str(task_dir), split="train", resolution=resolution)
                if len(ds) > 0:
                    vision_datasets.append(ds)
                    logger.info(f"  {task}: {len(ds)} samples")
                else:
                    logger.warning(f"  {task}: empty dataset")
            except Exception as e:
                logger.warning(f"  {task}: failed to load ({e})")
        else:
            logger.warning(f"  {task}: data dir not found ({task_dir})")

    # Fallback to v4 depth data if no task data found
    if not vision_datasets:
        depth_dir = DATA_DIR / "vb_v4_depth_2k"
        if depth_dir.exists():
            from open_vision_banana.data.datasets import HypersimDepthDataset
            ds = HypersimDepthDataset(depth_dir, split="train", resolution=resolution)
            if len(ds) > 0:
                vision_datasets.append(ds)
                logger.info(f"  Fallback depth: {len(ds)} samples")

    if not vision_datasets:
        logger.error("No vision datasets available!")
        return

    # ── Augmentation ──
    if args.augment == "advanced":
        augmenter = AdvancedAugmentation(
            flip_prob=0.5, scale_range=(0.8, 1.0), scale_prob=0.3,
            brightness_range=(-0.1, 0.1), brightness_prob=0.5,
            color_jitter_prob=0.3, color_jitter_strength=0.1,
            gaussian_noise_prob=0.2, gaussian_noise_std=0.02,
            cutout_prob=0.1, cutout_size=0.2,
        )
    elif args.augment == "basic":
        augmenter = AdvancedAugmentation(
            flip_prob=0.5, scale_range=(0.8, 1.0), scale_prob=0.3,
            brightness_range=(-0.1, 0.1), brightness_prob=0.5,
            color_jitter_prob=0.0, gaussian_noise_prob=0.0, cutout_prob=0.0,
        )
    else:
        augmenter = None

    # Wrap datasets with augmentation
    augmented_vision = [AugmentedVisionDataset(ds, augmenter) for ds in vision_datasets]

    # ── Generation dataset ──
    gen_ds = None
    gen_dir = DATA_DIR / "vb_v4_synthetic"
    if gen_dir.exists():
        gen_ds = GenerationDataset(gen_dir, resolution)
    else:
        logger.warning(f"Generation data not found: {gen_dir}")

    # ── Multi-colormap config ──
    colormaps = [c.strip() for c in args.colormaps.split(",")]

    # ── Mixed dataset ──
    mixed_ds = MixedVisionDataset(
        vision_datasets=augmented_vision,
        generation_dataset=gen_ds,
        vision_ratio=args.vision_ratio,
        seed=args.seed,
        colormaps=colormaps,
    )
    logger.info(f"Mixed dataset: {len(mixed_ds)} effective samples/epoch")

    # ── Optimized DataLoader ──
    dataloader = create_optimized_dataloader(
        dataset=mixed_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        device=device,
        drop_last=True,
    )

    # ── Optimizer & Scheduler ──
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = SchedulerFactory.create(
        optimizer, total_steps=total_steps,
        warmup_ratio=0.1, scheduler_type=args.scheduler,
        min_lr_ratio=0.01,
    )
    logger.info(f"Optimizer: AdamW, LR={args.lr}, Scheduler={args.scheduler}")

    # ── EMA ──
    ema = LoRAEMA(decay=args.ema_decay)
    ema.register([(n, p) for n, p in unet.named_parameters() if p.requires_grad])
    logger.info(f"EMA: {len(ema.shadow)} params, decay={args.ema_decay}")

    # ── Validation images ──
    val_image_paths = []
    for ds in vision_datasets:
        img_dir = Path(ds.root_dir) / "images"
        if img_dir.exists():
            val_image_paths.extend(sorted(img_dir.glob("*.jpg"))[:3])
    val_image_paths = val_image_paths[:6]

    val_images = []
    val_targets = []
    for p in val_image_paths:
        img = Image.open(p).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        t = t.to(dtype)
        val_images.append(t)
        # Find target
        for suffix in [".png", ".jpg"]:
            tgt_path = Path(p).parent.parent / "targets" / (Path(p).stem + suffix)
            if tgt_path.exists():
                tgt = Image.open(tgt_path).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
                val_targets.append(tgt)
                break

    # ── Training loop ──
    train_log = []
    val_log = []
    best_loss = float("inf")
    best_val_corr = -1.0
    t_train_start = time.time()
    global_step = 0

    logger.info("Starting v6 optimized training...")
    logger.info("  TRUE img2img + flow matching v-prediction + multi-colormap")

    while global_step < total_steps:
        for step, batch in enumerate(dataloader):
            if global_step >= total_steps:
                break

            target_rgb = batch["target_rgb"].to(device).float()
            input_images = batch["image"].to(device).to(dtype)
            prompts = batch["prompt"]
            bsz = target_rgb.shape[0]

            # Encode target and input
            with torch.no_grad():
                target_pixels = (target_rgb * 2.0 - 1.0).to(dtype)
                target_latents = vae_encode(vae, target_pixels, model_cfg)
                input_latents = vae_encode(vae, input_images, model_cfg)

            # Sample timesteps
            timesteps = sample_timesteps_logit_normal(bsz, device)
            t_b = timesteps.view(bsz, 1, 1, 1)

            # Flow matching with stochastic interpolant
            noise = torch.randn_like(target_latents)
            sigma = 0.1
            noise_term = sigma * (t_b * (1.0 - t_b)).sqrt() * noise
            noisy_latents = (1.0 - t_b) * input_latents + t_b * target_latents + noise_term
            velocity_target = target_latents - input_latents

            # Encode text
            with torch.no_grad():
                encoder_hidden_states, aux_output = encode_text(components, prompts, device, dtype)

            attention_mask = None
            pooled_embeds = None
            if isinstance(aux_output, torch.Tensor):
                if model_cfg.needs_attention_mask:
                    attention_mask = aux_output
                else:
                    pooled_embeds = aux_output

            added_cond_kwargs = build_added_cond_kwargs(
                model_cfg, pooled_embeds, (resolution, resolution), device, dtype, batch_size=bsz
            )
            timestep_long = (timesteps * 999).long().clamp(0, 999)
            txt_ids, img_ids = prepare_position_ids(noisy_latents, encoder_hidden_states, model_cfg, device, dtype)

            # Forward
            with torch.amp.autocast("mps", dtype=dtype, enabled=(device.type == "mps")):
                predicted_velocity = backbone_forward(
                    unet, noisy_latents, timestep_long, encoder_hidden_states, model_cfg,
                    added_cond_kwargs=added_cond_kwargs, attention_mask=attention_mask,
                    pooled_embeds=pooled_embeds, txt_ids=txt_ids, img_ids=img_ids,
                )

            loss = F.mse_loss(predicted_velocity.float(), velocity_target.float())
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            epoch_loss = loss.item()

            if (step + 1) % args.grad_accum == 0 or step == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                ema.update([(n, p) for n, p in unet.named_parameters() if p.requires_grad])

                current_lr = scheduler.get_last_lr()[0]

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t_train_start
                    eta = elapsed / global_step * (total_steps - global_step)
                    logger.info(
                        f"S{global_step}/{total_steps} | Loss: {epoch_loss:.4f} | "
                        f"LR: {current_lr:.2e} | {elapsed:.0f}s | ETA: {eta/60:.0f}m"
                    )
                    train_log.append({"step": global_step, "loss": epoch_loss, "lr": current_lr})

                if global_step % args.save_every == 0:
                    save_lora_state(unet, output_dir / f"vb_v6_{model_cfg.name}_step{global_step}.pt", ema.get_shadow())

                if epoch_loss < best_loss * 0.99:
                    best_loss = epoch_loss

                # ── Validation ──
                if global_step % args.val_every == 0:
                    logger.info(f"Validation at step {global_step} ...")
                    val_results = []
                    for vi, vimg in enumerate(val_images[:3]):
                        best_corr = -2.0
                        best_strength = 0.6
                        best_img = None
                        best_metrics = None
                        for strength in [0.4, 0.6, 0.8]:
                            try:
                                gen_img = generate_depth_img2img(
                                    components, vimg, DEPTH_PROMPT, device, dtype,
                                    num_steps=20, seed=args.seed + vi, strength=strength,
                                    guidance_scale=model_cfg.default_guidance_scale,
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
                            best_img.save(output_dir / "val_during_training" / f"val_step{global_step}_{vi}_s{best_strength}.png")
                            val_results.append(best_metrics)

                    if val_results:
                        avg_corr = np.mean([m["correlation"] for m in val_results])
                        avg_mse = np.mean([m["mse"] for m in val_results])
                        logger.info(f"  Val: corr={avg_corr:.4f} | MSE={avg_mse:.4f}")
                        val_log.append({"step": global_step, "avg_correlation": float(avg_corr), "avg_mse": float(avg_mse)})
                        if avg_corr > best_val_corr:
                            best_val_corr = avg_corr
                            save_lora_state(unet, output_dir / f"vb_v6_{model_cfg.name}_best_corr.pt", ema.get_shadow())
                            logger.info(f"  New best corr: {best_val_corr:.4f}")

                # ── Memory check ──
                if mem_manager.maybe_clear():
                    logger.debug(f"Memory cleared at step {global_step}")

    # ── Save final ──
    final_path = output_dir / f"vb_v6_{model_cfg.name}_final.pt"
    save_lora_state(unet, final_path, ema.get_shadow())
    total_time = time.time() - t_train_start
    logger.info(f"Training complete! Time: {total_time/60:.1f} min")
    logger.info(f"Final loss: {epoch_loss:.4f}, Best val corr: {best_val_corr:.4f}")

    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "config": {
                "model": model_cfg.model_id, "model_name": model_cfg.name,
                "version": "v6", "resolution": resolution,
                "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                "lr": args.lr, "total_steps": total_steps,
                "lora_rank": lora_rank, "lora_alpha": lora_alpha,
                "scheduler": args.scheduler, "augment": args.augment,
                "multi_task": args.multi_task, "vision_tasks": vision_tasks,
                "colormaps": colormaps, "vision_ratio": args.vision_ratio,
                "use_rslora": args.use_rslora, "ema_decay": args.ema_decay,
                "device": str(device), "dtype": str(dtype),
                "training_objective": "flow_matching_v_prediction",
                "key_innovation": "optimized_multi_task_multi_colormap",
            },
            "train_metrics": train_log, "val_metrics": val_log,
            "final_loss": epoch_loss, "best_train_loss": best_loss,
            "best_val_correlation": best_val_corr,
            "total_steps": global_step, "total_time_s": total_time,
        }, f, indent=2)
    logger.info(f"Log saved to {output_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
