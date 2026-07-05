#!/usr/bin/env python3
"""Vision Banana v7 — Training script with all bottleneck fixes.

.. deprecated::
    Use ``scripts/train.py`` instead.  v7-specific features (Min-SNR, logit-normal
    timesteps, img2img flow matching, SchedulerFactory, multi-colormap) have been
    ported to ``InstructionTuner`` and ``scripts/train.py``.

Key improvements over v6:
  1. Default img2img_strength = 0.3 (was 0.6, reduces texture leakage)
  2. Default colormaps = cube_edge (single colormap, lower learning difficulty)
  3. Default steps = 3000 (was 500,严重不足)
  4. Default grad-accum = 8 (effective batch = 8)
  5. --data-dir and --auto-select-data for flexible dataset discovery
  6. Per-sample validation logging (each sample's Pearson r, MSE, MAE, best strength)
  7. img2img_strength and colormap recorded in training_log.json
  8. FIX: EMA weight swap bug in validation (wrong key prefix check)
  9. Multi-dataset training: combines vb_v4_depth_2k + nyu + distill (1284 samples)
  10. Pure flow matching (sigma=0) — v6 had sigma=0.1 adding noise to velocity target
  11. Cosine scheduler (no restarts) — avoids loss spikes from cosine_restarts
  12. EMA decay 0.9995 (faster adaptation for short training)

Usage:
  python scripts/train_vb_v7.py --model ssd-1b --auto-select-data
  python scripts/train_vb_v7.py --model sana-600m --data-dir ./data/vb_v4_depth_2k
  python scripts/train_vb_v7.py --model ssd-1b --resume output/vb_v6/ssd-1b/vb_v6_ssd-1b_best_corr.pt
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
from torch.utils.data import Dataset, ConcatDataset

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
from open_vision_banana.data.datasets import PreEncodedDepthDataset, build_dataset
from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.training.optimization import (
    AdvancedAugmentation,
    MemoryManager,
    SchedulerFactory,
    create_optimized_dataloader,
)
from open_vision_banana.training.utils import EMAModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_v7")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSION_OUTPUT = Path(os.environ.get("VB_OUTPUT_DIR", str(PROJECT_ROOT / "output" / "vb_v7")))

DEPTH_PROMPT = "Generate a metric depth visualization of this image."


# ─── Dataset Utilities ─────────────────────────────────────────────────────


def discover_depth_datasets(data_dir: Path, pattern: str = "vb_*") -> list[Path]:
    """Discover depth datasets under data_dir matching pattern.
    
    A valid depth dataset must have both images/ and targets/ subdirectories
    with at least one image file.
    """
    candidates = []
    for candidate in sorted(data_dir.glob(pattern)):
        if candidate.is_dir():
            img_dir = candidate / "images"
            tgt_dir = candidate / "targets"
            if img_dir.exists() and tgt_dir.exists():
                img_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
                if img_count > 0:
                    candidates.append((candidate, img_count))
    # Sort by image count descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def auto_select_depth_datasets(data_dir: Path) -> list[Path]:
    """Auto-select all valid depth datasets, combining them for maximum diversity.
    
    Returns a list of dataset paths. If no datasets found, returns empty list.
    """
    discovered = discover_depth_datasets(data_dir)
    if not discovered:
        return []
    paths = [p for p, _ in discovered]
    logger.info(f"Auto-selected {len(paths)} depth dataset(s): {[p.name for p in paths]}")
    for p, count in discovered:
        logger.info(f"  {p.name}: {count} samples")
    return paths


class AugmentedVisionDataset(Dataset):
    """Wrap a vision dataset with advanced augmentation."""

    def __init__(self, base_dataset, augmenter: AdvancedAugmentation | None = None):
        self.base = base_dataset
        self.augmenter = augmenter

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if self.augmenter is not None and "image" in sample and "target_rgb" in sample:
            image = sample["image"]
            target = sample["target_rgb"]
            image_aug, target_aug = self.augmenter(image, target)
            sample = dict(sample)
            sample["image"] = image_aug
            sample["target_rgb"] = target_aug
        return sample


class GenerationDataset(Dataset):
    """Text-to-image generation samples for mixture preservation."""

    def __init__(self, data_dir: Path, resolution: int = 384):
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
    """Collate function that handles mixed sample types."""
    common_keys = set(batch[0].keys())
    for s in batch[1:]:
        common_keys &= set(s.keys())

    result = {}
    for key in common_keys:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


# ─── EMA (unified EMAModel — P2-18) ───────────────────────────────────────


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
    """Pearson r on metric depth recovered via DepthRGBMapper (P1-17)."""
    from open_vision_banana.core.depth_mapper import DepthRGBMapper

    mapper = DepthRGBMapper()

    def _pil_to_depth(pil_img):
        arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return mapper.rgb_to_depth(tensor).flatten().numpy()

    gen_depth = _pil_to_depth(gen_img)
    tgt_depth = _pil_to_depth(target_img)
    gen_norm = gen_depth - gen_depth.mean()
    tgt_norm = tgt_depth - tgt_depth.mean()
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


def compute_min_snr_weight(timesteps, gamma=5.0):
    """Compute Min-SNR weighting for flow matching timesteps."""
    t = timesteps.clamp(1e-4, 1.0 - 1e-4)
    snr = (t ** 2) / ((1.0 - t) ** 2 + 1e-10)
    min_snr = torch.clamp(snr, max=gamma)
    return min_snr / (snr + 1e-10)


@torch.no_grad()
def generate_depth_img2img(components, input_image, prompt, device, dtype, num_steps=30, seed=42, strength=0.3, guidance_scale=2.0):
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
    import warnings

    warnings.warn(
        "scripts/train_vb_v7.py is deprecated. "
        "Use scripts/train.py — v7 features are now integrated into InstructionTuner.",
        DeprecationWarning,
        stacklevel=1,
    )
    parser = argparse.ArgumentParser(description="Vision Banana v7 — Optimized training (deprecated)")
    parser.add_argument("--model", type=str, default="ssd-1b")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--auto-select", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--vision-ratio", type=float, default=0.1)
    parser.add_argument("--vision-tasks", type=str, default="depth")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--use-rslora", action="store_true", default=True)
    parser.add_argument("--img2img-strength", type=float, default=0.3,
                        help="img2img strength for validation (default 0.3, reduced from v6's 0.6 to reduce texture leakage)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "cosine_restarts", "polynomial", "constant", "one_cycle"])
    parser.add_argument("--augment", type=str, default="advanced",
                        choices=["none", "basic", "advanced"])
    parser.add_argument("--memory-threshold", type=float, default=0.85)
    parser.add_argument("--colormaps", type=str, default="cube_edge",
                        help="Comma-separated colormaps (default: cube_edge only for lower learning difficulty)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--resume-step", type=int, default=0,
                        help="Step number to resume from (for scheduler continuity)")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Noise sigma for flow matching (0=pure, v6 used 0.1)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data directory (default: project_root/data)")
    parser.add_argument("--auto-select-data", action="store_true",
                        help="Auto-discover and select all available depth datasets under data-dir")
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
    resolution = args.resolution
    total_steps = args.steps
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = SESSION_OUTPUT / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "val_during_training").mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16

    logger.info("=" * 70)
    logger.info("Vision Banana v7 — Optimized Training")
    logger.info("=" * 70)
    logger.info(f"Model: {model_cfg.model_id} ({model_cfg.name})")
    logger.info(f"Family: {model_cfg.family}, Encoders: {model_cfg.num_text_encoders}")
    logger.info(f"Device: {device}, Dtype: {dtype}, Resolution: {resolution}")
    logger.info(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, rsLoRA={args.use_rslora}")
    logger.info(f"Scheduler: {args.scheduler}, Augment: {args.augment}")
    logger.info(f"Sigma: {args.sigma} (pure flow matching)")
    logger.info(f"EMA decay: {args.ema_decay}")
    logger.info(f"Colormaps: {args.colormaps}")
    logger.info(f"img2img strength: {args.img2img_strength}")
    logger.info(f"grad-accum: {args.grad_accum} (effective batch = {args.batch_size * args.grad_accum})")
    logger.info("=" * 70)

    # ── Memory Manager ──
    mem_manager = MemoryManager(device, threshold_gb=args.memory_threshold)

    # ── Load pipeline ──
    components = load_pipeline(model_cfg, device, dtype, enable_gradient_checkpointing=True)

    # ── Apply LoRA ──
    components.unet = apply_lora(components.unet, model_cfg, rank=lora_rank, alpha=lora_alpha, use_rslora=args.use_rslora)
    trainable = sum(p.numel() for p in components.unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in components.unet.parameters())
    logger.info(f"Trainable: {trainable} ({trainable/total*100:.2f}%)")

    unet = components.unet
    vae = components.vae

    # ── Resume from checkpoint ──
    resume_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"Resuming from: {resume_path}")
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            ema_keys = {k[4:]: v for k, v in ckpt.items() if k.startswith("ema.")}
            unet_keys = {k[5:]: v for k, v in ckpt.items() if k.startswith("unet.")}
            loaded = 0
            source_keys = ema_keys if ema_keys else unet_keys
            prefix = "ema." if ema_keys else "unet."
            for name, param in unet.named_parameters():
                if param.requires_grad and name in source_keys:
                    try:
                        param.data.copy_(source_keys[name].to(dtype))
                        loaded += 1
                    except Exception as e:
                        logger.debug(f"  Skip {name}: shape mismatch ({param.shape} vs {source_keys[name].shape})")
            logger.info(f"Resumed: loaded {loaded} parameters from {prefix}* keys")
            import re
            m = re.search(r"step(\d+)", resume_path.stem)
            if m:
                resume_step = int(m.group(1))
                logger.info(f"Resuming from step {resume_step}")
            elif args.resume_step > 0:
                resume_step = args.resume_step
                logger.info(f"Resuming from step {resume_step} (via --resume-step)")
        else:
            logger.warning(f"Resume checkpoint not found: {resume_path}")

    # ── Determine data directory ──
    data_dir = DATA_DIR
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
    logger.info(f"Data directory: {data_dir}")

    # ── Build vision datasets ──
    vision_datasets = []
    vision_dataset_paths = []

    if args.auto_select_data:
        # Auto-discover and combine all available depth datasets
        auto_paths = auto_select_depth_datasets(data_dir)
        if not auto_paths:
            logger.warning("Auto-select found no datasets, falling back to known paths")
        for ds_path in auto_paths:
            try:
                ds = PreEncodedDepthDataset(ds_path, split="train", resolution=resolution)
                if len(ds) > 0:
                    vision_datasets.append(ds)
                    vision_dataset_paths.append(ds_path)
                    logger.info(f"  Auto: {ds_path.name}: {len(ds)} samples")
            except Exception as e:
                logger.warning(f"  Auto: {ds_path.name}: failed to load ({e})")
    else:
        # Manual: try known dataset paths in priority order
        for name in ["vb_v4_depth_2k", "vb_nyu_depth_300", "vb_distill_428", "vb_depth_300", "vision_banana_depth"]:
            ds_path = data_dir / name
            if ds_path.exists():
                try:
                    ds = PreEncodedDepthDataset(ds_path, split="train", resolution=resolution)
                    if len(ds) > 0:
                        vision_datasets.append(ds)
                        vision_dataset_paths.append(ds_path)
                        logger.info(f"  {name}: {len(ds)} samples")
                except Exception as e:
                    logger.warning(f"  {name}: failed to load ({e})")

    # Fallback: if still no datasets, try vb_* wildcard
    if not vision_datasets:
        logger.warning("No known datasets found, trying vb_* wildcard fallback...")
        for ds_path, count in discover_depth_datasets(data_dir):
            try:
                ds = PreEncodedDepthDataset(ds_path, split="train", resolution=resolution)
                if len(ds) > 0:
                    vision_datasets.append(ds)
                    vision_dataset_paths.append(ds_path)
                    logger.info(f"  Fallback: {ds_path.name}: {len(ds)} samples")
            except Exception as e:
                logger.warning(f"  Fallback: {ds_path.name}: failed ({e})")

    if not vision_datasets:
        logger.error("No vision datasets available!")
        return

    total_vision = sum(len(ds) for ds in vision_datasets)
    logger.info(f"Total vision samples: {total_vision}")

    # ── Augmentation ──
    if args.augment == "advanced":
        augmenter = AdvancedAugmentation(
            flip_prob=0.5, scale_range=(0.8, 1.0), scale_prob=0.3,
            brightness_range=(-0.1, 0.1), brightness_prob=0.5,
            color_jitter_prob=0.0,  # Disabled: preserve RGB encoding
            gaussian_noise_prob=0.15, gaussian_noise_std=0.01,
            cutout_prob=0.1, cutout_size=0.15,
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
    gen_dir = data_dir / "vb_v4_synthetic"
    if gen_dir.exists():
        gen_ds = GenerationDataset(gen_dir, resolution)
    else:
        logger.warning(f"Generation data not found: {gen_dir}")

    # ── Mixed dataset ──
    from open_vision_banana.data.mixed import MixedVisionDataset
    colormaps = [c.strip() for c in args.colormaps.split(",")]
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
        warmup_ratio=0.05,
        scheduler_type=args.scheduler,
        min_lr_ratio=0.01,
    )
    logger.info(f"Optimizer: AdamW, LR={args.lr}, Scheduler={args.scheduler}")

    # ── EMA ──
    ema = EMAModel(unet, decay=args.ema_decay, warmup_steps=0)
    logger.info(f"EMA: {len(ema.shadow_params)} params, decay={args.ema_decay}")

    # ── Validation images ──
    val_image_paths = []
    val_targets = []
    for ds in vision_datasets:
        root = getattr(ds, "root_dir", None)
        if root is not None:
            img_dir = Path(root) / "images"
            if img_dir.exists():
                for p in sorted(img_dir.glob("*.jpg"))[:2]:
                    val_image_paths.append(p)
                    for suffix in [".png", ".jpg"]:
                        tgt_path = Path(p).parent.parent / "targets" / (Path(p).stem + suffix)
                        if tgt_path.exists():
                            val_targets.append(tgt_path)
                            break
    val_image_paths = val_image_paths[:6]
    val_targets = val_targets[:6]

    val_images = []
    val_target_imgs = []
    for p, tp in zip(val_image_paths, val_targets):
        img = Image.open(p).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        t = t.to(dtype)
        val_images.append(t)
        tgt = Image.open(tp).convert("RGB").resize((resolution, resolution), Image.LANCZOS)
        val_target_imgs.append(tgt)

    logger.info(f"Validation: {len(val_images)} images with GT targets")

    # ── Training loop ──
    train_log = []
    val_log = []
    best_loss = float("inf")
    best_val_corr = -1.0
    t_train_start = time.time()
    global_step = resume_step

    logger.info("Starting v7 optimized training...")
    logger.info(f"  img2img_strength={args.img2img_strength}, colormaps={colormaps}")
    logger.info("  Pure flow matching (sigma=0) + cosine scheduler + multi-dataset")

    while global_step < total_steps:
        for step, batch in enumerate(dataloader):
            if global_step >= total_steps:
                break

            target_rgb = batch["target_rgb"].to(device).float()
            input_images = batch["image"].to(device).to(dtype)
            prompts = batch["prompt"]
            bsz = target_rgb.shape[0]

            # Multi-colormap training: re-encode depth targets with per-sample colormap
            colormaps_in_batch = batch.get("colormap", None)
            if colormaps_in_batch is not None and len(colormaps_in_batch) > 0:
                unique_cmaps = set(colormaps_in_batch)
                if len(unique_cmaps) > 1 or list(unique_cmaps)[0] != "cube_edge":
                    mapper = DepthRGBMapper()
                    for i in range(bsz):
                        cmap = colormaps_in_batch[i] if i < len(colormaps_in_batch) else "cube_edge"
                        if cmap != mapper.colormap:
                            tgt_single = target_rgb[i:i+1]
                            depth = mapper.rgb_to_depth(tgt_single)
                            mapper.set_colormap(cmap)
                            re_encoded = mapper.depth_to_rgb(depth)
                            target_rgb[i] = re_encoded.squeeze(0)
                            mapper.set_colormap("cube_edge")

            # Encode target and input
            with torch.no_grad():
                target_pixels = (target_rgb * 2.0 - 1.0).to(dtype)
                target_latents = vae_encode(vae, target_pixels, model_cfg)
                input_latents = vae_encode(vae, input_images, model_cfg)

            # Sample timesteps
            timesteps = sample_timesteps_logit_normal(bsz, device)
            t_b = timesteps.view(bsz, 1, 1, 1)

            # Pure flow matching
            noise = torch.randn_like(target_latents)
            if args.sigma > 0:
                noise_term = args.sigma * (t_b * (1.0 - t_b)).sqrt() * noise
            else:
                noise_term = 0.0

            noisy_latents = (1.0 - t_b) * input_latents + t_b * target_latents + noise_term
            velocity_target = target_latents - input_latents

            # Encode text
            with torch.no_grad():
                encoder_hidden_states, aux_output = encode_text(components, prompts, device, dtype)
                uncond_hidden, uncond_aux = encode_text(components, [""] * bsz, device, dtype, negative=True)

            # CFG dropout
            cfg_dropout_rate = 0.1
            if bsz > 1:
                drop_mask = torch.rand(bsz, device=device) < cfg_dropout_rate
                drop_exp = drop_mask.view(bsz, 1, 1)
                encoder_hidden_states = torch.where(drop_exp, uncond_hidden, encoder_hidden_states)
            elif torch.rand(1).item() < cfg_dropout_rate:
                encoder_hidden_states = uncond_hidden

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
            with torch.amp.autocast(device.type, dtype=dtype, enabled=(device.type in ("mps", "cuda"))):
                predicted_velocity = backbone_forward(
                    unet, noisy_latents, timestep_long, encoder_hidden_states, model_cfg,
                    added_cond_kwargs=added_cond_kwargs, attention_mask=attention_mask,
                    pooled_embeds=pooled_embeds, txt_ids=txt_ids, img_ids=img_ids,
                )

            # Min-SNR weighted loss
            snr_weights = compute_min_snr_weight(timesteps, gamma=5.0)
            per_sample_loss = F.mse_loss(
                predicted_velocity.float(), velocity_target.float(),
                reduction="none"
            ).mean(dim=[1, 2, 3])
            loss = (per_sample_loss * snr_weights).mean()
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()
            epoch_loss = loss.item()

            if (step + 1) % args.grad_accum == 0 or step == len(dataloader) - 1:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                ema.update(unet)

                current_lr = scheduler.get_last_lr()[0]

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t_train_start
                    eta = elapsed / max(1, global_step - resume_step) * (total_steps - global_step)
                    avg_t = timesteps.mean().item()
                    avg_snr_w = snr_weights.mean().item()
                    logger.info(
                        f"S{global_step}/{total_steps} | Loss: {epoch_loss:.4f} | "
                        f"LR: {current_lr:.2e} | avg_t={avg_t:.2f} | snr_w={avg_snr_w:.2f} | "
                        f"{elapsed:.0f}s | ETA: {eta/60:.0f}m"
                    )
                    train_log.append({
                        "step": global_step, "loss": epoch_loss, "lr": current_lr,
                        "avg_timestep": avg_t, "avg_snr_weight": avg_snr_w,
                    })

                if global_step % args.save_every == 0:
                    save_lora_state(unet, output_dir / f"vb_v7_{model_cfg.name}_step{global_step}.pt", ema.shadow_params)

                if epoch_loss < best_loss * 0.99:
                    best_loss = epoch_loss

                # ── Validation (with EMA weights) ──
                if global_step % args.val_every == 0:
                    # FIX: v6/v7 bug — ema.shadow_params returns keys WITHOUT "ema." prefix.
                    # The old code checked f"ema.{name}" in ema_shadow which NEVER matched,
                    # so validation always used training weights instead of EMA weights.
                    ema_shadow = ema.shadow_params
                    original_weights = {}
                    with torch.no_grad():
                        for name, param in unet.named_parameters():
                            # BUG FIX: check "name in ema_shadow" not "f'ema.{name}' in ema_shadow"
                            if param.requires_grad and name in ema_shadow:
                                original_weights[name] = param.data.clone()
                                param.data.copy_(ema_shadow[name].to(dtype))

                    logger.info(f"Validation at step {global_step} (EMA weights) ...")
                    val_results = []
                    val_per_sample = []
                    for vi, (vimg, vtarget) in enumerate(zip(val_images[:4], val_target_imgs[:4])):
                        best_corr = -2.0
                        best_strength = args.img2img_strength
                        best_img = None
                        best_metrics = None
                        # v7: finer strength grid (5 values) centered on default strength
                        for strength in [0.2, 0.3, 0.5, 0.7, 0.9]:
                            try:
                                gen_img = generate_depth_img2img(
                                    components, vimg, DEPTH_PROMPT, device, dtype,
                                    num_steps=25, seed=args.seed + vi, strength=strength,
                                    guidance_scale=model_cfg.default_guidance_scale,
                                )
                                metrics = compute_depth_metrics(gen_img, vtarget)
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
                            val_per_sample.append({
                                "sample_idx": vi,
                                "filename": val_image_paths[vi].stem if vi < len(val_image_paths) else f"val_{vi}",
                                "correlation": best_metrics["correlation"],
                                "mse": best_metrics["mse"],
                                "mae": best_metrics["mae"],
                                "gen_std": best_metrics["gen_std"],
                                "tgt_std": best_metrics["tgt_std"],
                                "best_strength": best_strength,
                            })
                            logger.info(
                                f"  Val[{vi}] corr={best_metrics['correlation']:.4f} "
                                f"mse={best_metrics['mse']:.4f} mae={best_metrics['mae']:.4f} "
                                f"strength={best_strength}"
                            )

                    # Restore original training weights after validation
                    with torch.no_grad():
                        for name, param in unet.named_parameters():
                            if name in original_weights:
                                param.data.copy_(original_weights[name])

                    if val_results:
                        avg_corr = np.mean([m["correlation"] for m in val_results])
                        avg_mse = np.mean([m["mse"] for m in val_results])
                        avg_mae = np.mean([m["mae"] for m in val_results])
                        logger.info(f"  Val avg: corr={avg_corr:.4f} | MSE={avg_mse:.4f} | MAE={avg_mae:.4f}")
                        val_log.append({
                            "step": global_step,
                            "avg_correlation": float(avg_corr),
                            "avg_mse": float(avg_mse),
                            "avg_mae": float(avg_mae),
                            "per_sample": val_per_sample,
                            "num_samples": len(val_results),
                        })
                        if avg_corr > best_val_corr:
                            best_val_corr = avg_corr
                            save_lora_state(unet, output_dir / f"vb_v7_{model_cfg.name}_best_corr.pt", ema.shadow_params)
                            logger.info(f"  New best corr: {best_val_corr:.4f}")

                # ── Memory check ──
                if mem_manager.maybe_clear():
                    logger.debug(f"Memory cleared at step {global_step}")

    # ── Save final ──
    final_path = output_dir / f"vb_v7_{model_cfg.name}_final.pt"
    save_lora_state(unet, final_path, ema.shadow_params)
    total_time = time.time() - t_train_start
    logger.info(f"Training complete! Time: {total_time/60:.1f} min")
    logger.info(f"Final loss: {epoch_loss:.4f}, Best val corr: {best_val_corr:.4f}")

    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "config": {
                "model": model_cfg.model_id, "model_name": model_cfg.name,
                "version": "v7", "resolution": resolution,
                "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                "lr": args.lr, "total_steps": total_steps,
                "lora_rank": lora_rank, "lora_alpha": lora_alpha,
                "scheduler": args.scheduler, "augment": args.augment,
                "vision_tasks": [args.vision_tasks], "colormaps": colormaps,
                "vision_ratio": args.vision_ratio,
                "use_rslora": args.use_rslora, "ema_decay": args.ema_decay,
                "device": str(device), "dtype": str(dtype),
                "sigma": args.sigma,
                "img2img_strength": args.img2img_strength,
                "training_objective": "flow_matching_v_prediction_pure",
                "key_innovation": "multi_dataset_pure_flow_cosine_ema_warmstart",
                "total_vision_samples": total_vision,
                "resume_from": args.resume,
                "data_dir": str(data_dir),
                "vision_dataset_paths": [str(p) for p in vision_dataset_paths],
            },
            "train_metrics": train_log,
            "val_metrics": val_log,
            "final_loss": epoch_loss,
            "best_train_loss": best_loss,
            "best_val_correlation": best_val_corr,
            "total_steps": global_step,
            "total_time_s": total_time,
        }, f, indent=2)
    logger.info(f"Log saved to {output_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
