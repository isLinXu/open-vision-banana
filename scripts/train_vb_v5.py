#!/usr/bin/env python3
"""Vision Banana v5 — Multi-backbone training with TRUE image conditioning.

Supports 23 base model families across UNet and Transformer architectures:
  - SD 1.5   (860M UNet, single text encoder)
  - SD 2.1   (866M UNet, single text encoder)
  - SSD-1B   (1.3B UNet, dual text encoder)  ← default
  - SDXL     (2.6B UNet, dual text encoder)
  - SDXL Turbo (2.6B, 4-step generation)
  - FLUX.1 dev/schnell (12B transformer)
  - Qwen-Image (transformer, text rendering, CFG)
  - Z-Image-Turbo (6B transformer, single-stream DiT)
  - SANA-600M/1600M (transformer, linear DiT, efficient)
  - Lumina-Image-2.0 (2B transformer, flow DiT)
  - Chroma1-HD (8.9B transformer, FLUX.1-schnell base)
  - FLUX.2-dev (next-gen transformer)
  - CogView4-6B (transformer, Apache-2.0)
  - AuraFlow (transformer, MMDiT, Apache-2.0)
  - SD3 (2B MMDiT, triple text encoder)
  - PixArt-Alpha/Sigma (600M DiT, T5 encoder)
  - Kolors (2.6B UNet, ChatGLM encoder)
  - HunyuanDiT (1.5B DiT, bilingual CLIP+mT5)
  - Kandinsky3 (3B UNet, FLAN-UL2 encoder)
  - SanaSprint (2-4 step ultra-fast generation)

Key innovations (inherited from v4):
  1. TRUE img2img: input image encoded as latent starting point
  2. Conditional flow matching: velocity = target - input
  3. Uniform weighting (min-SNR unsuitable for input→target flow)
  4. Stochastic interpolant: x_t = (1-t)*input + t*target + sigma*sqrt(t(1-t))*noise

Architecture differences handled by model_registry:
  - UNet vs Transformer backbone
  - Single vs dual text encoders
  - added_cond_kwargs (SDXL+ only)
  - VAE scaling/shift factors per model
  - VAE downsample factor (8x or 16x)
  - LoRA target modules per family
  - Attention masks for DiT models
  - Position IDs for FLUX/Chroma/FLUX.2

Usage:
  python scripts/train_vb_v5.py --model ssd-1b           # default
  python scripts/train_vb_v5.py --model sd15 --steps 500 # lightweight
  python scripts/train_vb_v5.py --model sana-600m        # efficient DiT
  python scripts/train_vb_v5.py --model qwen-image       # Qwen-Image
  python scripts/train_vb_v5.py --list-models            # list all options
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_v5")

# ─── Defaults ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SESSION_OUTPUT = Path(os.environ.get(
    "VB_OUTPUT_DIR",
    str(PROJECT_ROOT / "output" / "vb_v5"),
))

DEPTH_DATA_DIR = DATA_DIR / "vb_v4_depth_2k"
SYNTHETIC_DATA_DIR = DATA_DIR / "vb_v4_synthetic"

DEPTH_PROMPT = "Generate a metric depth visualization of this image."


# ─── Datasets (reused from v4, unchanged) ──────────────────────────────────


class DepthImg2ImgDataset(Dataset):
    """Paired (input_image, target_depth) dataset for img2img training."""

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
            "image": target_tensor * 2.0 - 1.0,
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


# ─── Timestep sampling ────────────────────────────────────────────────────


def sample_timesteps_logit_normal(batch_size, device, mean=0.0, std=1.0):
    """Logit-normal timestep sampling (matches SD3/FLUX training)."""
    noise = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(noise)


# ─── Flow matching inference ──────────────────────────────────────────────


@torch.no_grad()
def generate_depth_img2img(
    components: PipelineComponents,
    input_image: torch.Tensor,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    num_steps: int = 30,
    seed: int = 42,
    strength: float = 1.0,
    guidance_scale: float = 2.0,
):
    """Generate depth visualization via TRUE img2img flow matching.

    Works across all model families via the unified backbone_forward helper.
    """
    config = components.config
    backbone = components.unet
    vae = components.vae
    backbone.eval()

    # Encode text
    encoder_hidden_states, aux_output = encode_text(
        components, [prompt], device, dtype
    )

    # aux_output can be pooled_embeds (SDXL), attention_mask (DiT), tuple (HunyuanDiT), or None
    attention_mask = None
    pooled_embeds = None
    if isinstance(aux_output, tuple):
        attention_mask = aux_output  # HunyuanDiT: (t5_hidden, clip_mask, t5_mask)
    elif isinstance(aux_output, torch.Tensor) and config.needs_attention_mask:
        attention_mask = aux_output
    elif isinstance(aux_output, torch.Tensor):
        pooled_embeds = aux_output
    if aux_output is None:
        pooled_embeds = None
        attention_mask = None

    # Build added_cond_kwargs if needed
    res = (input_image.shape[-2], input_image.shape[-1])
    added_cond_kwargs = build_added_cond_kwargs(
        config, pooled_embeds, res, device, dtype, batch_size=1
    )

    # Prepare position IDs if needed
    txt_ids, img_ids = prepare_position_ids(
        None, encoder_hidden_states, config, device, dtype
    )

    # CFG setup
    uncond_hidden = None
    uncond_added_cond = None
    uncond_mask = None
    if guidance_scale > 1.0:
        uncond_hidden, uncond_aux = encode_text(
            components, [""], device, dtype, negative=True
        )
        if isinstance(uncond_aux, torch.Tensor):
            if config.needs_attention_mask:
                uncond_mask = uncond_aux
            else:
                uncond_added_cond = build_added_cond_kwargs(
                    config, uncond_aux, res, device, dtype, batch_size=1
                )

    # TRUE img2img: encode INPUT image as latent starting point
    input_latents = vae_encode(vae, input_image, config)

    torch.manual_seed(seed)
    latents = input_latents.clone()

    # Flow matching: integrate from t=0 (input) toward t=strength (target)
    num_actual = max(1, int(num_steps * strength))
    dt = strength / num_actual

    for i in range(num_actual):
        t = i * dt
        timestep_long = torch.tensor([int(t * 999)], device=device).clamp(0, 999)

        # Update position IDs with current latents
        if config.needs_position_ids:
            txt_ids, img_ids = prepare_position_ids(
                latents, encoder_hidden_states, config, device, dtype
            )

        if guidance_scale > 1.0 and uncond_hidden is not None:
            v_cond = backbone_forward(
                backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs,
                guidance=guidance_scale,
                attention_mask=attention_mask,
                pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
            v_uncond = backbone_forward(
                backbone, latents, timestep_long, uncond_hidden, config,
                added_cond_kwargs=uncond_added_cond,
                guidance=guidance_scale,
                attention_mask=uncond_mask,
                pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )
            pred_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            pred_v = backbone_forward(
                backbone, latents, timestep_long, encoder_hidden_states, config,
                added_cond_kwargs=added_cond_kwargs,
                guidance=guidance_scale,
                attention_mask=attention_mask,
                pooled_embeds=pooled_embeds,
                txt_ids=txt_ids, img_ids=img_ids,
            )

        latents = latents + dt * pred_v

    # Decode
    image = vae_decode(vae, latents, config)
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    backbone.train()
    return Image.fromarray(image_np)


# ─── Main training ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Vision Banana v5 — Multi-backbone training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models (23 total):
  UNet-based:
    sd15         Stable Diffusion 1.5 (860M, ~4GB)
    sd2          Stable Diffusion 2.1 (866M, ~5GB)
    ssd-1b       SSD-1B distilled SDXL (1.3B, ~5.5GB)  [default]
    sdxl         SDXL Base (2.6B, ~7GB)
    sdxl-turbo   SDXL Turbo (2.6B, ~7GB, 4-step)
    kolors       Kolors (2.6B, ChatGLM encoder, ~18GB)
    kandinsky3   Kandinsky3 (3B, FLAN-UL2, ~12GB)
  Transformer-based (DiT):
    flux-dev     FLUX.1-dev (12B, ~24GB)
    flux-schnell FLUX.1-schnell (12B, ~24GB, 4-step)
    qwen-image   Qwen-Image (text rendering, ~20GB)
    z-image      Z-Image-Turbo 6B (single-stream DiT, ~24GB)
    sana-600m    SANA-600M (linear DiT, ~6GB)
    sana-1600m   SANA-1600M (linear DiT, ~10GB)
    sana-sprint  SanaSprint (2-4 step ultra-fast, ~8GB)
    lumina2      Lumina-Image-2.0 (2B flow DiT, ~12GB)
    chroma       Chroma1-HD (8.9B, FLUX base, ~20GB)
    flux2-dev    FLUX.2-dev (next-gen, ~30GB)
    cogview4     CogView4-6B (Apache-2.0, ~24GB)
    aura-flow    AuraFlow (MMDiT, Apache-2.0, ~15GB)
    sd3          Stable Diffusion 3 (2B MMDiT, triple encoder, ~20GB)
    pixart-alpha PixArt-Alpha (600M DiT, T5 encoder, ~10GB)
    pixart-sigma PixArt-Sigma (600M DiT, 4K-capable, ~10GB)
    hunyuandit   HunyuanDiT (1.5B, bilingual CLIP+mT5, ~15GB)

Use --list-models to see all options with details.
        """,
    )
    parser.add_argument("--model", type=str, default="ssd-1b",
                        help="Base model name (default: ssd-1b)")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available models and exit")
    parser.add_argument("--auto-select", action="store_true",
                        help="Auto-select model based on available memory")
    parser.add_argument("--steps", type=int, default=None,
                        help="Total optimizer steps (default: model-specific)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Training resolution (default: model-specific)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=None,
                        help="LoRA rank (default: model-specific)")
    parser.add_argument("--lora-alpha", type=int, default=None,
                        help="LoRA alpha (default: model-specific)")
    parser.add_argument("--vision-ratio", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-rslora", action="store_true", default=True)
    parser.add_argument("--img2img-strength", type=float, default=0.6)
    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 80)
        for name in list_models():
            cfg = get_model_config(name)
            print(f"  {name:15s} | {cfg.description}")
            print(f"  {'':15s} | family={cfg.family}, params~{cfg.memory_budget_gb}GB, "
                  f"encoders={cfg.num_text_encoders}")
            print()
        return

    # Resolve model config
    if args.auto_select:
        # Estimate available memory
        if torch.backends.mps.is_available():
            mem_gb = 32.0  # M1 Pro 32GB assumption
        elif torch.cuda.is_available():
            mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        else:
            mem_gb = 8.0
        selected = auto_select_model(mem_gb)
        logger.info(f"Auto-selected model: {selected} (memory budget: {mem_gb:.0f}GB)")
        args.model = selected

    model_cfg = get_model_config(args.model)

    # Apply model-specific defaults
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
    logger.info("Vision Banana v5 — Multi-backbone Training")
    logger.info("=" * 70)
    logger.info(f"Model: {model_cfg.model_id} ({model_cfg.name})")
    logger.info(f"Family: {model_cfg.family}, Encoders: {model_cfg.num_text_encoders}")
    logger.info(f"Device: {device}, Dtype: {dtype}, Resolution: {resolution}")
    logger.info(f"LoRA: rank={lora_rank}, alpha={lora_alpha}, rsLoRA={args.use_rslora}")
    logger.info(f"Training: flow matching v-prediction + TRUE img2img")
    logger.info(f"Data mix: {args.vision_ratio:.0%} vision + {1-args.vision_ratio:.0%} generation")
    logger.info(f"Batch: {args.batch_size}×{args.grad_accum}={args.batch_size*args.grad_accum}, LR: {args.lr}")
    logger.info(f"Steps: {total_steps}")
    logger.info(f"added_cond_kwargs: {model_cfg.needs_added_cond_kwargs}")
    logger.info("=" * 70)

    # ─── Load pipeline ───
    components = load_pipeline(model_cfg, device, dtype, enable_gradient_checkpointing=True)

    # ─── Apply LoRA ───
    components.unet = apply_lora(
        components.unet, model_cfg, rank=lora_rank, alpha=lora_alpha,
        use_rslora=args.use_rslora,
    )
    trainable = sum(p.numel() for p in components.unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in components.unet.parameters())
    logger.info(f"Trainable: {trainable} ({trainable/total*100:.2f}%)")

    unet = components.unet
    vae = components.vae

    # ─── Build datasets ───
    if not DEPTH_DATA_DIR.exists():
        logger.error(f"Depth data not found: {DEPTH_DATA_DIR}")
        logger.error("Run: python scripts/prepare_v4_data.py first")
        return

    vision_ds = DepthImg2ImgDataset(DEPTH_DATA_DIR, resolution, augment=True)
    gen_ds = None
    if SYNTHETIC_DATA_DIR.exists():
        gen_ds = SimpleGenerationDataset(SYNTHETIC_DATA_DIR, resolution)
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

    # ─── Validation image pool ───
    val_image_paths = sorted((DEPTH_DATA_DIR / "images").glob("*.jpg"))[:6]
    val_images = []
    val_targets = []
    for p in val_image_paths:
        img = Image.open(p).convert("RGB").resize(
            (resolution, resolution), Image.LANCZOS
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0
        t = t.to(dtype)
        val_images.append(t)
        tgt_name = p.stem + ".png"
        tgt_path = DEPTH_DATA_DIR / "targets" / tgt_name
        if not tgt_path.exists():
            tgt_path = DEPTH_DATA_DIR / "targets" / p.name
        if tgt_path.exists():
            tgt = Image.open(tgt_path).convert("RGB").resize(
                (resolution, resolution), Image.LANCZOS
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
            bsz = target_rgb.shape[0]

            # ── Encode TARGET to latents ──
            with torch.no_grad():
                target_pixels = (target_rgb * 2.0 - 1.0).to(dtype)
                target_latents = vae_encode(vae, target_pixels, model_cfg)

            # ── Encode INPUT image to latents ──
            with torch.no_grad():
                input_latents = vae_encode(vae, input_images, model_cfg)

            # ── Sample timesteps ──
            timesteps = sample_timesteps_logit_normal(bsz, device)
            t_b = timesteps.view(bsz, 1, 1, 1)

            # ── Conditional flow matching with stochastic interpolant ──
            noise = torch.randn_like(target_latents)
            sigma = 0.1
            noise_term = sigma * (t_b * (1.0 - t_b)).sqrt() * noise

            noisy_latents = (1.0 - t_b) * input_latents + t_b * target_latents + noise_term

            # Velocity target: constant direction input→target
            velocity_target = target_latents - input_latents

            # ── Encode text ──
            with torch.no_grad():
                encoder_hidden_states, aux_output = encode_text(
                    components, prompts, device, dtype
                )

            # Handle aux_output (pooled_embeds for SDXL, attention_mask for DiT,
            # tuple for HunyuanDiT: (t5_hidden, clip_mask, t5_mask))
            attention_mask = None
            pooled_embeds = None
            if isinstance(aux_output, tuple):
                # HunyuanDiT: pass tuple directly, backbone_forward handles it
                attention_mask = aux_output
            elif isinstance(aux_output, torch.Tensor):
                if model_cfg.needs_attention_mask:
                    attention_mask = aux_output
                else:
                    pooled_embeds = aux_output

            # Build added_cond_kwargs if model needs it
            added_cond_kwargs = build_added_cond_kwargs(
                model_cfg, pooled_embeds,
                (resolution, resolution), device, dtype, batch_size=bsz
            )

            timestep_long = (timesteps * 999).long().clamp(0, 999)

            # ── Prepare position IDs if needed ──
            txt_ids, img_ids = prepare_position_ids(
                noisy_latents, encoder_hidden_states, model_cfg, device, dtype
            )

            # ── Backbone forward (unified) ──
            with torch.amp.autocast("mps", dtype=dtype, enabled=(device.type == "mps")):
                predicted_velocity = backbone_forward(
                    unet, noisy_latents, timestep_long, encoder_hidden_states,
                    model_cfg,
                    added_cond_kwargs=added_cond_kwargs,
                    attention_mask=attention_mask,
                    pooled_embeds=pooled_embeds,
                    txt_ids=txt_ids, img_ids=img_ids,
                )

            # ── Loss ──
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
                        output_dir / f"vb_v5_{model_cfg.name}_step{global_step}.pt",
                        ema.get_shadow(),
                    )

                if epoch_loss < best_loss * 0.99:
                    best_loss = epoch_loss

                # ── Validation ──
                if global_step % args.val_every == 0 and global_step <= total_steps:
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
                                    components,
                                    vimg, DEPTH_PROMPT, device, dtype,
                                    num_steps=20, seed=args.seed + vi,
                                    strength=strength,
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
                                output_dir / f"vb_v5_{model_cfg.name}_best_corr.pt",
                                ema.get_shadow(),
                            )
                            logger.info(f"  New best corr: {best_val_corr:.4f}")

    # ─── Save final ───
    final_path = output_dir / f"vb_v5_{model_cfg.name}_final.pt"
    save_lora_state(unet, final_path, ema.get_shadow())
    total_time = time.time() - t_train_start
    logger.info(f"Training complete! Total time: {total_time/60:.1f} min")
    logger.info(f"Final loss: {epoch_loss:.4f}, Best val corr: {best_val_corr:.4f}")

    # ─── Save training log ───
    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "config": {
                "model": model_cfg.model_id,
                "model_name": model_cfg.name,
                "model_family": model_cfg.family,
                "version": "v5",
                "resolution": resolution,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "total_steps": total_steps,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "use_rslora": args.use_rslora,
                "target_modules": model_cfg.lora_target_modules,
                "seed": args.seed,
                "device": str(device),
                "dtype": str(dtype),
                "vision_ratio": args.vision_ratio,
                "training_objective": "flow_matching_v_prediction",
                "key_innovation": "true_img2img_input_latent_as_noise_start",
                "ema_decay": args.ema_decay,
                "img2img_strength": args.img2img_strength,
                "needs_added_cond_kwargs": model_cfg.needs_added_cond_kwargs,
                "num_text_encoders": model_cfg.num_text_encoders,
                "cross_attention_dim": model_cfg.cross_attention_dim,
                "vae_scaling_factor": model_cfg.vae_scaling_factor,
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
