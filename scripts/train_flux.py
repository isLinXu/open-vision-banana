#!/usr/bin/env python3
"""Train Vision Banana with FLUX.1-dev (12B transformer) via LoRA.

Uses 8-bit quantization for the base transformer to fit within 24GB VRAM,
with manifest-based multi-task data loading (depth + normal + segmentation).

Usage:
    WANDB_API_KEY=... python scripts/train_flux.py \
        --config configs/flux_3task_full_clean.yaml --wandb
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import wandb
from open_vision_banana.data.manifest import resolve_manifest_path
from open_vision_banana.model_registry import (
    apply_lora,
    get_model_config,
    load_pipeline,
    vae_encode,
    vae_decode,
)
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.training.optimization import SchedulerFactory
from open_vision_banana.training.sana_flow import flow_min_snr_weight
from open_vision_banana.training.sana_depth_aux import depth_edge_endpoint_loss, depth_task_mask
from open_vision_banana.training.normal_aux_loss import normal_angular_loss, normal_task_mask
from open_vision_banana.training.utils import EMAModel

logger = logging.getLogger(__name__)

# ── Config loading ──────────────────────────────────────────────────────

def load_config_defaults(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    import yaml
    with config_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    model = raw.get("model", {}) or {}
    lora = raw.get("lora", {}) or {}
    data = raw.get("data", {}) or {}
    training = raw.get("training", {}) or {}
    scheduler = raw.get("scheduler", {}) or {}
    flow = raw.get("flow", {}) or {}
    loss = raw.get("loss", {}) or {}
    ckpt = raw.get("checkpoint", {}) or {}
    ema = raw.get("ema", {}) or {}
    wandb_section = raw.get("wandb", {}) or {}
    d = {
        "model_name": model.get("name", "flux-dev"),
        "dtype": model.get("dtype", "bfloat16"),
        "resolution": data.get("resolution", model.get("resolution", 512)),
        "lora_rank": lora.get("rank", 32),
        "lora_alpha": lora.get("alpha", 64),
        "rslora": lora.get("rs_lora", True),
        "lora_targets": lora.get("target_modules"),
        "data_dir": data.get("dir"),
        "pair_manifest": data.get("pair_manifest"),
        "num_workers": data.get("num_workers", 0),
        "tasks": data.get("tasks"),
        "steps": training.get("steps", 5000),
        "batch_size": training.get("batch_size", 1),
        "grad_accum": training.get("gradient_accumulation", 4),
        "lr": training.get("lr", 3e-5),
        "max_grad_norm": training.get("max_grad_norm", 1.0),
        "seed": training.get("seed", 42),
        "scheduler": scheduler.get("type", "cosine"),
        "warmup_ratio": scheduler.get("warmup_ratio", 0.05),
        "img2img_flow": flow.get("img2img_flow", True),
        "flow_sigma": flow.get("sigma", 0.0),
        "logit_normal": flow.get("logit_normal_timesteps", True),
        "min_snr": flow.get("min_snr", True),
        "min_snr_gamma": flow.get("min_snr_gamma", 5.0),
        "cfg_dropout_rate": flow.get("cfg_dropout_rate", 0.1),
        "depth_edge_aux_weight": loss.get("depth_edge_aux_weight", 0.0),
        "depth_edge_aux_gradient_weight": loss.get("depth_edge_aux_gradient_weight", 0.25),
        "depth_edge_aux_edge_weight": loss.get("depth_edge_aux_edge_weight", 2.0),
        "depth_edge_aux_edge_threshold": loss.get("depth_edge_aux_edge_threshold", 0.025),
        "depth_edge_aux_sigma_floor": loss.get("depth_edge_aux_sigma_floor", 0.10),
        "normal_angular_loss_weight": loss.get("normal_angular_loss_weight", 0.0),
        "output_dir": ckpt.get("output_dir", "output/flux_3task"),
        "save_every": ckpt.get("save_every", 500),
        "eval_every": ckpt.get("eval_every", 250),
        "log_every": ckpt.get("log_every", 25),
        "ema_decay": ema.get("decay", 0.9999),
        "ema_warmup": ema.get("warmup_steps", 500),
        "wandb": wandb_section.get("enabled", False),
        "wandb_project": wandb_section.get("project", "open-vision-banana"),
        "wandb_run_name": wandb_section.get("run_name"),
        "resume": ckpt.get("resume"),
        "resume_weights_only": ckpt.get("resume_weights_only", False),
    }
    return {k: v for k, v in d.items() if v is not None}


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    cfg = load_config_defaults(config_args.config)

    p = argparse.ArgumentParser(description="Train Vision Banana with FLUX.1-dev")
    p.add_argument("--config", type=Path, default=config_args.config)
    p.add_argument("--data-dir", default=cfg.get("data_dir"))
    p.add_argument("--pair-manifest", type=Path, default=cfg.get("pair_manifest"))
    p.add_argument("--tasks", nargs="+", default=cfg.get("tasks"))
    p.add_argument("--resolution", type=int, default=cfg.get("resolution", 512))
    p.add_argument("--model-name", default=cfg.get("model_name", "flux-dev"))
    p.add_argument("--dtype", default=cfg.get("dtype", "bfloat16"))
    p.add_argument("--lora-rank", type=int, default=cfg.get("lora_rank", 32))
    p.add_argument("--lora-alpha", type=int, default=cfg.get("lora_alpha", 64))
    p.add_argument("--no-rslora", action="store_true", default=not cfg.get("rslora", True))
    p.add_argument("--steps", type=int, default=cfg.get("steps", 5000))
    p.add_argument("--lr", type=float, default=cfg.get("lr", 3e-5))
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 1))
    p.add_argument("--grad-accum", type=int, default=cfg.get("grad_accum", 4))
    p.add_argument("--max-grad-norm", type=float, default=cfg.get("max_grad_norm", 1.0))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    p.add_argument("--num-workers", type=int, default=cfg.get("num_workers", 0))
    p.add_argument("--scheduler", default=cfg.get("scheduler", "cosine"))
    p.add_argument("--warmup-ratio", type=float, default=cfg.get("warmup_ratio", 0.05))
    p.add_argument("--no-img2img-flow", action="store_true", default=not cfg.get("img2img_flow", True))
    p.add_argument("--flow-sigma", type=float, default=cfg.get("flow_sigma", 0.0))
    p.add_argument("--no-logit-normal", action="store_true", default=not cfg.get("logit_normal", True))
    p.add_argument("--no-min-snr", action="store_true", default=not cfg.get("min_snr", True))
    p.add_argument("--min-snr-gamma", type=float, default=cfg.get("min_snr_gamma", 5.0))
    p.add_argument("--cfg-dropout-rate", type=float, default=cfg.get("cfg_dropout_rate", 0.1))
    p.add_argument("--depth-edge-aux-weight", type=float, default=cfg.get("depth_edge_aux_weight", 0.0))
    p.add_argument("--depth-edge-aux-gradient-weight", type=float, default=cfg.get("depth_edge_aux_gradient_weight", 0.25))
    p.add_argument("--depth-edge-aux-edge-weight", type=float, default=cfg.get("depth_edge_aux_edge_weight", 2.0))
    p.add_argument("--depth-edge-aux-edge-threshold", type=float, default=cfg.get("depth_edge_aux_edge_threshold", 0.025))
    p.add_argument("--depth-edge-aux-sigma-floor", type=float, default=cfg.get("depth_edge_aux_sigma_floor", 0.10))
    p.add_argument("--normal-angular-loss-weight", type=float, default=cfg.get("normal_angular_loss_weight", 0.0))
    p.add_argument("--output-dir", default=cfg.get("output_dir", "output/flux_3task"))
    p.add_argument("--save-every", type=int, default=cfg.get("save_every", 500))
    p.add_argument("--eval-every", type=int, default=cfg.get("eval_every", 250))
    p.add_argument("--log-every", type=int, default=cfg.get("log_every", 25))
    p.add_argument("--ema-decay", type=float, default=cfg.get("ema_decay", 0.9999))
    p.add_argument("--ema-warmup", type=int, default=cfg.get("ema_warmup", 500))
    p.add_argument("--device", default=cfg.get("device"))
    p.add_argument("--max-samples-per-task", type=int, default=None)
    p.add_argument("--wandb", action="store_true", default=cfg.get("wandb", False))
    p.add_argument("--wandb-project", default=cfg.get("wandb_project", "open-vision-banana"))
    p.add_argument("--wandb-run-name", default=cfg.get("wandb_run_name"))
    p.add_argument("--resume", default=cfg.get("resume"),
                   help="Path to a checkpoint dir or lora.pt to resume training from")
    p.add_argument("--resume-weights-only", action="store_true",
                   default=cfg.get("resume_weights_only", False),
                   help="Load only LoRA weights; reset optimizer, scheduler and step count")
    return p.parse_args()


def get_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Dataset ─────────────────────────────────────────────────────────────

TASK_ALIASES = {
    "segmentation": "semantic",
    "seg": "semantic",
    "detection": "object_detection",
    "bbox": "object_detection",
    "point": "keypoint",
    "gcg": "referring",
}

# Tasks whose RGB targets are discrete colour codes (palette / outlines / dots);
# their targets must be resized with NEAREST so colours are never blurred.
DISCRETE_TARGET_TASKS = {"semantic", "object_detection", "keypoint", "referring", "structure", "edge", "depth_edge", "instance"}

TASK_PROMPTS = {
    "depth": "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map.",
    "normal": "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
    "semantic": "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
    "structure": "Generate a binary structural edge map. Use a black background and thin white edges aligned to visible scene boundaries.",
    "object_detection": "Generate an object detection map. Draw one thin colored rectangle per detected object on a black background, using one distinct solid color per object category and no other pixels.",
    "keypoint": "Generate a keypoint map. Draw small solid colored dots at each detected landmark on a black background, using one distinct color per point category and no other pixels.",
    "referring": "Generate a grounded segmentation mask. Highlight the referred region with a solid palette color on a black background and sharp object boundaries.",
    "edge": "Generate an edge detection map. Draw thin white edges on a black background tracing the visible object boundaries of the input image, with no filled regions.",
    "depth_edge": "Generate a depth-edge map. Draw thin white edges on a black background only at depth discontinuities, suppressing flat-region texture edges.",
    "instance": "Generate an instance segmentation map. Color every object instance with one distinct solid palette color, so different instances of the same class get different colors, on a black background.",
    "colorization": "Colorize this grayscale image. Produce realistic natural colors matching the input structure exactly.",
    "super_resolution": "Restore this low-resolution blurred image. Reconstruct a sharp high-resolution version preserving all detail.",
    "deblurring": "Remove motion and defocus blur. Reconstruct the sharp clear photo that matches the input layout exactly.",
    "inpainting": "Complete the missing gray-masked regions. Reconstruct plausible natural content that seamlessly fills the holes.",
}


class ManifestVisionDataset(Dataset):
    """Load input/target pairs from a JSONL manifest for multi-task training."""

    def __init__(self, manifest_path, resolution, split, tasks, max_samples=None, data_root=None):
        self.resolution = resolution
        self.samples = []
        path_anchor = data_root.resolve() if data_root else manifest_path
        with Path(manifest_path).open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_task = row.get("task")
                mapped = TASK_ALIASES.get(raw_task, raw_task)
                if mapped not in tasks:
                    continue
                if row.get("split") != split:
                    continue
                img_path = resolve_manifest_path(row.get("input", ""), path_anchor)
                tgt_path = resolve_manifest_path(row.get("target", ""), path_anchor)
                if not img_path.is_file() or not tgt_path.is_file():
                    continue
                prompt = row.get("prompt") or TASK_PROMPTS.get(mapped, TASK_PROMPTS.get(raw_task, ""))
                self.samples.append({
                    "image_path": str(img_path), "target_path": str(tgt_path),
                    "task": mapped, "prompt": prompt,
                })
        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        last_err = None
        for attempt, cur in enumerate(self._candidates(idx)):
            try:
                return self._load(cur)
            except Exception as e:
                last_err = e
                if attempt == 0:
                    logger.warning("Skipping unreadable sample idx=%s (%s)", cur, e)
        raise RuntimeError(f"All candidates failed (last: {last_err})")

    def _candidates(self, idx):
        yield idx
        rng = random.Random(idx)
        n = len(self.samples)
        for _ in range(3):
            yield rng.randrange(n) if n else idx

    def _load(self, idx):
        s = self.samples[idx]
        with Image.open(s["image_path"]) as img:
            img.load()
            img = img.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            inp = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        resample = Image.Resampling.NEAREST if s["task"] in DISCRETE_TARGET_TASKS else Image.Resampling.LANCZOS
        with Image.open(s["target_path"]) as img:
            img.load()
            img = img.convert("RGB").resize((self.resolution, self.resolution), resample)
            tgt = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        return {
            "prompt": s["prompt"], "image": inp * 2.0 - 1.0,
            "target_rgb": tgt, "task": s["task"],
        }


# ── FLUX helpers ────────────────────────────────────────────────────────

def _pack_latents(latents, batch_size, num_channels, height, width):
    """[B, C, H, W] → [B, (H/2)*(W/2), C*4]"""
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)


def _unpack_latents(latents, height, width):
    """[B, (H/2)*(W/2), C*4] → [B, C, H, W]"""
    B, N, C = latents.shape
    latents = latents.view(B, height // 2, width // 2, C // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(B, C // 4, height, width)


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """FLUX position IDs [H*W, 3]."""
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width)[None, :]
    return ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def flux_encode_text(pipe, prompts, device, dtype, max_length=512):
    """FLUX dual text encoding via pipeline.encode_prompt (CLIP + T5)."""
    with torch.no_grad():
        outputs = pipe.encode_prompt(
            prompt=prompts, prompt_2=prompts,
            device=device, num_images_per_prompt=1,
            max_sequence_length=max_length,
        )

    # encode_prompt returns a 3-tuple:
    #   prompt_embeds:       (B, seq_len, hidden)
    #   pooled_prompt_embeds: (B, 768)  — from CLIP
    #   text_ids:            (seq_len, 1, 3) or (seq_len, 3)
    prompt_embeds = outputs[0]
    pooled = outputs[1] if len(outputs) > 1 else None
    text_ids = outputs[2] if len(outputs) > 2 else None

    return {
        "prompt_embeds": prompt_embeds.to(device=device, dtype=dtype),
        "pooled_prompt_embeds": pooled.to(device=device, dtype=dtype) if pooled is not None else None,
        "text_ids": text_ids.to(device=device, dtype=dtype) if text_ids is not None else None,
    }


def flux_vae_encode(vae, images, config, dtype):
    """Encode images to FLUX VAE latents with proper normalization."""
    return vae_encode(vae, images, config).to(dtype)


def flux_vae_decode(vae, latents, config):
    """Decode FLUX VAE latents to images."""
    return vae_decode(vae, latents, config)


def sample_rectified_flow_timesteps(batch_size, device, logit_normal=True):
    """Sample timesteps for rectified flow matching."""
    if logit_normal:
        u = torch.rand(batch_size, device=device)
        t = torch.sigmoid(u)
    else:
        t = torch.rand(batch_size, device=device)
    return t


def mix_rectified_flow(z_0, noise, t, input_latents=None, use_img2img=True, flow_sigma=0.0):
    """Create noisy latents via rectified flow interpolation.

    z_t = (1-t)*z_0 + t*z_source,  v_target = z_0 - z_source

    t = 0 is the target z_0 and t = 1 the source (the opposite of FlowScheduler's
    rectified-flow convention), so t is already SANA's sigma here.
    """
    t_expand = t.view(-1, *([1] * (z_0.dim() - 1)))
    if use_img2img and input_latents is not None:
        z_source = input_latents
        if flow_sigma > 0:
            z_source = (1 - flow_sigma) * z_source + flow_sigma * noise
    else:
        z_source = noise
    z_t = (1 - t_expand) * z_0 + t_expand * z_source
    v_target = z_0 - z_source
    return z_t, v_target


# ── Checkpoint ──────────────────────────────────────────────────────────

def save_checkpoint(output_dir, peft_model, optimizer, scheduler, step, val_loss, best_val_loss, args, ema=None, suffix=""):
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lora_state = {n: p.data.cpu().clone() for n, p in peft_model.named_parameters() if p.requires_grad}
    ckpt = {
        "lora_state_dict": lora_state, "step": step, "val_loss": val_loss,
        "best_val_loss": best_val_loss, "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if ema is not None and getattr(ema, "shadow_params", None):
        ckpt["ema_state_dict"] = {n: t.cpu().clone() for n, t in ema.shadow_params.items()}
    torch.save(ckpt, str(ckpt_dir / "lora.pt"))
    try:
        peft_model.save_pretrained(str(ckpt_dir / "lora_adapter"))
    except Exception as e:
        logger.warning("Could not save LoRA adapter: %s", e)
    meta = {"step": step, "val_loss": val_loss, "best_val_loss": best_val_loss,
            "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha, "model_name": args.model_name}
    (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler()])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)
    logging.getLogger().addHandler(fh)

    device = torch.device(args.device or auto_device())
    dtype = get_dtype(args.dtype)

    logger.info("=" * 70)
    logger.info("Vision Banana — FLUX.1-dev Training (8-bit quantized)")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name} | Device: {device} | dtype: {dtype}")
    logger.info(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, rsLoRA={not args.no_rslora}")
    logger.info(f"Steps: {args.steps} | LR: {args.lr} | eff_batch: {args.batch_size * args.grad_accum}")
    logger.info(f"Output: {output_dir}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load FLUX pipeline with 8-bit transformer ─────────────────────
    logger.info("Loading FLUX.1-dev (8-bit quantized transformer)...")
    from diffusers import FluxPipeline, FluxTransformer2DModel
    from transformers import T5EncoderModel, CLIPTextModel, BitsAndBytesConfig

    model_path = os.environ.get(
        "OPEN_VISION_BANANA_MODEL_PATH",
        "/media/linxu/thinkHub/ckpts/black-forest-labs/FLUX.1-dev",
    )
    # Check if local path exists
    if not Path(model_path).is_dir():
        model_path = "black-forest-labs/FLUX.1-dev"

    # Load transformer in 4-bit to fit 24GB VRAM (leaves room for T5 + training buffers)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer",
        torch_dtype=dtype, quantization_config=quant_config,
    )

    # Load T5 text encoder in 8-bit (fits thanks to 4-bit transformer)
    logger.info("Loading T5 text encoder in 8-bit...")
    t5_quant = BitsAndBytesConfig(load_in_8bit=True)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder_2",
        torch_dtype=dtype, quantization_config=t5_quant,
    )

    # Load full pipeline with quantized models
    pipe = FluxPipeline.from_pretrained(
        model_path, transformer=transformer, text_encoder_2=text_encoder_2,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    # Remove accelerate hooks from transformer and submodules.
    # The hooks' pre_forward corrupts FLUX-specific kwargs (pooled_projections).
    # Since all components are already on CUDA, hooks are unnecessary.
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(pipe.transformer, recurse=True)
    remove_hook_from_module(pipe.vae, recurse=True)
    remove_hook_from_module(pipe.text_encoder, recurse=True)
    remove_hook_from_module(pipe.text_encoder_2, recurse=True)
    logger.info("Removed accelerate hooks from all pipeline components.")

    # All components on CUDA (4-bit transformer + 8-bit T5 fits in 24GB).
    # VAE and CLIP stay on CUDA for fast encoding.
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    torch.cuda.empty_cache()
    logger.info(f"All components on CUDA. GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

    # Enable gradient checkpointing
    if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
        pipe.transformer.enable_gradient_checkpointing()

    # Extract components
    backbone = pipe.transformer
    vae = pipe.vae
    model_cfg = get_model_config(args.model_name)

    backbone_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"Backbone params: {backbone_params / 1e6:.1f}M")
    logger.info(f"VAE: scaling={model_cfg.vae_scaling_factor}, shift={model_cfg.vae_shift_factor}")

    # ── Apply LoRA ────────────────────────────────────────────────────
    logger.info("Applying LoRA adapters...")
    target_modules = getattr(args, 'lora_targets', None) or model_cfg.lora_target_modules
    peft_model = apply_lora(
        backbone, config=model_cfg,
        rank=args.lora_rank, alpha=args.lora_alpha,
        use_rslora=not args.no_rslora,
    )
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── EMA ───────────────────────────────────────────────────────────
    ema = EMAModel(peft_model, decay=args.ema_decay, warmup_steps=args.ema_warmup)

    # ── Build datasets ────────────────────────────────────────────────
    logger.info("Building datasets...")
    data_root = Path(args.data_dir) if args.data_dir else None
    manifest_path = args.pair_manifest
    if manifest_path is not None:
        manifest_path = manifest_path.expanduser().resolve()
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(f"Pair manifest not found: {manifest_path}")

    all_tasks = {"depth", "normal", "semantic"}
    tasks_to_load = set()
    if args.tasks:
        for t in args.tasks:
            tasks_to_load.add(TASK_ALIASES.get(t, t))
    else:
        tasks_to_load = all_tasks

    train_datasets = [
        ManifestVisionDataset(manifest_path, args.resolution, "train", {t},
                              args.max_samples_per_task, data_root=data_root)
        for t in sorted(tasks_to_load)
    ]
    train_datasets = [ds for ds in train_datasets if len(ds) > 0]
    val_datasets = [
        ManifestVisionDataset(manifest_path, args.resolution, "val", {t}, data_root=data_root)
        for t in sorted(tasks_to_load)
    ]
    val_datasets = [ds for ds in val_datasets if len(ds) > 0]

    from torch.utils.data import ConcatDataset, DataLoader
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds = ConcatDataset(val_datasets) if val_datasets else None

    def collate_fn(batch):
        return {
            "prompt": [b["prompt"] for b in batch],
            "image": torch.stack([b["image"] for b in batch]),
            "target_rgb": torch.stack([b["target_rgb"] for b in batch]),
            "task": [b["task"] for b in batch],
        }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate_fn)

    for ds in train_datasets:
        logger.info(f"  train/{ds.samples[0]['task'] if ds.samples else '?'}: {len(ds)} samples")
    logger.info(f"  Total train: {len(train_ds)} | Val: {len(val_ds) if val_ds else 0}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999),
    )
    total_steps = args.steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    lr_scheduler = SchedulerFactory.create(
        optimizer, total_steps=total_steps,
        warmup_steps=warmup_steps,
        scheduler_type=args.scheduler,
    )

    # ── W&B ───────────────────────────────────────────────────────────
    if args.wandb:
        run_name = args.wandb_run_name or f"flux-dev-3task-{random.randint(1000,9999)}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args), dir=str(output_dir))

    # ── Task loss weights ─────────────────────────────────────────────
    task_loss_weights = {"depth": 2.5, "normal": 2.0, "semantic": 0.5}

    # ── Training loop ─────────────────────────────────────────────────
    logger.info("Starting training...")
    model_dtype = dtype
    vae_downsample = model_cfg.vae_downsample_factor  # 16 for FLUX VAE (but actually 8)
    # FLUX VAE has 8x spatial downsample, 16 latent channels
    vae_spatial = 8  # standard VAE downsampling
    max_length = 512  # FLUX text encoder max length

    train_metrics = {"loss": 0.0, "flow_loss": 0.0, "depth_edge_aux_loss": 0.0,
                     "normal_angular_aux_loss": 0.0, "steps": 0}
    best_val_loss = float("inf")
    global_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    # -- Resume from checkpoint (warm-start multi-task training) ----------
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "lora.pt"
        if resume_path.is_file():
            logger.info(f"Resuming from {resume_path}")
            ckpt = torch.load(str(resume_path), map_location="cpu")
            peft_model.load_state_dict(ckpt.get("lora_state_dict", ckpt), strict=False)
            if args.resume_weights_only:
                logger.info("  Loaded LoRA weights only; optimizer/scheduler/step reset")
            else:
                global_step = ckpt.get("step", 0)
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                ema_ckpt = ckpt.get("ema_state_dict")
                if ema_ckpt and getattr(ema, "shadow_params", None):
                    for _n, _t in ema_ckpt.items():
                        if _n in ema.shadow_params:
                            ema.shadow_params[_n] = _t.to(ema.shadow_params[_n].device)
                    ema.step = ckpt.get("step", 0)
                    logger.info("  Restored EMA shadow weights")
                logger.info(f"  Resumed at step {global_step}, best_val_loss={best_val_loss:.4f}")
        else:
            logger.warning(f"Resume checkpoint not found: {resume_path}")

    while global_step < total_steps:
        updates_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
        epoch = global_step // updates_per_epoch
        peft_model.train()

        for batch in train_loader:
            if global_step >= total_steps:
                break

            prompts = batch["prompt"]
            batch_size = len(prompts)
            target_rgb = batch["target_rgb"].to(device=device, dtype=model_dtype)
            input_images = batch["image"].to(device=device, dtype=model_dtype)

            # ── VAE encode target & input ─────────────────────────────
            with torch.no_grad():
                target_pixels = target_rgb * 2.0 - 1.0  # [0,1] → [-1,1]
                target_latents = flux_vae_encode(vae, target_pixels, model_cfg, model_dtype)
                input_pixels = input_images  # already [-1,1]
                input_latents = flux_vae_encode(vae, input_pixels, model_cfg, model_dtype)
                if input_latents.shape != target_latents.shape:
                    input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])

            # ── Rectified flow matching ───────────────────────────────
            timesteps = sample_rectified_flow_timesteps(
                batch_size, device, logit_normal=not args.no_logit_normal)
            timesteps = timesteps.to(model_dtype)

            noise = torch.randn_like(target_latents)
            if not args.no_img2img_flow:
                source_latents = input_latents
                if args.flow_sigma > 0:
                    source_latents = (1 - args.flow_sigma) * source_latents + args.flow_sigma * noise
            else:
                source_latents = noise

            noisy_latents, velocity_target = mix_rectified_flow(
                target_latents, noise, timesteps,
                input_latents=source_latents if not args.no_img2img_flow else None,
                use_img2img=not args.no_img2img_flow,
                flow_sigma=args.flow_sigma,
            )

            # ── Text encoding ─────────────────────────────────────────
            with torch.no_grad():
                text_cond = flux_encode_text(pipe, prompts, device, model_dtype, max_length)
                prompt_embeds = text_cond["prompt_embeds"]
                pooled_embeds = text_cond["pooled_prompt_embeds"]
                text_ids = text_cond["text_ids"]

                # CFG dropout
                if args.cfg_dropout_rate > 0:
                    drop_mask = torch.rand(batch_size, device=device) < args.cfg_dropout_rate
                    if drop_mask.any():
                        uncond = flux_encode_text(pipe, [""] * batch_size, device, model_dtype, max_length)
                        drop = drop_mask.view(-1, *([1] * (prompt_embeds.dim() - 1)))
                        prompt_embeds = torch.where(drop, uncond["prompt_embeds"], prompt_embeds)
                        if pooled_embeds is not None and uncond["pooled_prompt_embeds"] is not None:
                            drop_p = drop_mask.view(-1, *([1] * (pooled_embeds.dim() - 1)))
                            pooled_embeds = torch.where(drop_p, uncond["pooled_prompt_embeds"], pooled_embeds)

            # ── Pack latents for FLUX ─────────────────────────────────
            latent_h, latent_w = noisy_latents.shape[-2], noisy_latents.shape[-1]
            latent_channels = noisy_latents.shape[1]
            packed_noisy = _pack_latents(noisy_latents, batch_size, latent_channels, latent_h, latent_w)
            packed_h, packed_w = latent_h // 2, latent_w // 2
            img_ids = _prepare_latent_image_ids(batch_size, packed_h, packed_w, device, model_dtype)

            # ── Forward pass ──────────────────────────────────────────
            forward_kwargs = {
                "hidden_states": packed_noisy.to(model_dtype),
                "timestep": timesteps.to(model_dtype),
                "encoder_hidden_states": prompt_embeds.to(model_dtype),
            }
            if pooled_embeds is not None:
                forward_kwargs["pooled_projections"] = pooled_embeds.to(model_dtype)
            else:
                forward_kwargs["pooled_projections"] = torch.zeros(batch_size, 768, device=device, dtype=model_dtype)
            if text_ids is not None:
                forward_kwargs["txt_ids"] = text_ids.to(model_dtype)
            forward_kwargs["img_ids"] = img_ids.to(model_dtype)

            # FLUX.1-dev uses CombinedTimestepGuidanceTextProjEmbeddings
            # which requires (timestep, guidance, pooled_projection).
            # Pass default guidance=50.0 (scaled to 50000 by transformer).
            if "guidance" not in forward_kwargs:
                forward_kwargs["guidance"] = torch.full(
                    (batch_size,), 50.0, device=device, dtype=model_dtype)

            predicted = peft_model(**forward_kwargs)
            if hasattr(predicted, "sample"):
                predicted = predicted.sample

            # Unpack output if packed: [B, (H/2)*(W/2), C*4] → [B, C, H, W]
            if predicted.dim() == 3:
                predicted = _unpack_latents(predicted, latent_h, latent_w)

            # ── Loss computation ──────────────────────────────────────
            sq_error = (predicted.float() - velocity_target.float()).square()
            per_sample_loss = sq_error.flatten(1).mean(dim=1)

            # Task loss weighting
            batch_tasks = batch.get("task", ["depth"] * batch_size)
            weights = torch.tensor(
                [task_loss_weights.get(t, 1.0) for t in batch_tasks],
                device=device, dtype=torch.float32,
            )
            per_sample_loss = per_sample_loss * weights
            flow_loss = per_sample_loss.mean()

            # Min-SNR weighting
            if not args.no_min_snr:
                snw = flow_min_snr_weight(timesteps, args.min_snr_gamma)
                flow_loss = (per_sample_loss * snw).mean()

            # Per-task tracking
            task_loss_acc = train_metrics.setdefault("task_loss", {})
            for _t, _c in zip(batch_tasks, per_sample_loss.detach().float().tolist()):
                _ent = task_loss_acc.setdefault(_t, [0.0, 0])
                _ent[0] += float(_c)
                _ent[1] += 1

            # ── Auxiliary losses ──────────────────────────────────────
            # They rebuild the target endpoint as sample - sigma * v with SANA's
            # v = source - target. mix_rectified_flow already weights the source by t,
            # but its v = target - source, so only the velocity is negated here.
            depth_aux_loss = flow_loss.new_zeros(())
            if args.depth_edge_aux_weight:
                depth_mask = depth_task_mask(batch.get("task"), batch_size, device)
                if depth_mask.any():
                    depth_aux_loss = depth_edge_endpoint_loss(
                        -predicted[depth_mask], noisy_latents[depth_mask],
                        target_latents[depth_mask], target_rgb[depth_mask],
                        timesteps[depth_mask],
                        edge_threshold=args.depth_edge_aux_edge_threshold,
                        edge_weight=args.depth_edge_aux_edge_weight,
                        gradient_weight=args.depth_edge_aux_gradient_weight,
                        sigma_floor=args.depth_edge_aux_sigma_floor,
                    )

            normal_aux_loss = flow_loss.new_zeros(())
            if args.normal_angular_loss_weight:
                normal_mask = normal_task_mask(batch.get("task"), batch_size, device)
                if normal_mask.any():
                    def _diff_decode(latents):
                        return flux_vae_decode(vae, latents, model_cfg)
                    normal_aux_loss = normal_angular_loss(
                        -predicted[normal_mask], noisy_latents[normal_mask],
                        target_latents[normal_mask], target_rgb[normal_mask],
                        timesteps[normal_mask], _diff_decode,
                    )

            total_loss = (flow_loss.float()
                          + args.depth_edge_aux_weight * depth_aux_loss
                          + args.normal_angular_loss_weight * normal_aux_loss)
            if not torch.isfinite(total_loss).all():
                raise FloatingPointError(f"Non-finite loss at step {global_step}")

            loss = total_loss / args.grad_accum
            loss.backward()

            train_metrics["loss"] += total_loss.item()
            train_metrics["flow_loss"] += flow_loss.item()
            train_metrics["depth_edge_aux_loss"] += depth_aux_loss.item()
            train_metrics["normal_angular_aux_loss"] += normal_aux_loss.item()
            train_metrics["steps"] += 1

            micro_step += 1
            if micro_step % args.grad_accum != 0:
                continue

            # ── Optimizer step ────────────────────────────────────────
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(peft_model)
            global_step += 1

            # ── Logging ──────────────────────────────────────────────
            if global_step % args.log_every == 0:
                avg_loss = train_metrics["loss"] / max(1, train_metrics["steps"])
                avg_flow = train_metrics["flow_loss"] / max(1, train_metrics["steps"])
                avg_depth_aux = train_metrics["depth_edge_aux_loss"] / max(1, train_metrics["steps"])
                avg_normal_aux = train_metrics["normal_angular_aux_loss"] / max(1, train_metrics["steps"])
                lr = lr_scheduler.get_last_lr()[0]
                task_str = " ".join(f"{t}={v[0]/max(1,v[1]):.4f}" for t, v in sorted(task_loss_acc.items()))
                logger.info(
                    f"  [step {global_step}/{args.steps}] loss={avg_loss:.4f} flow={avg_flow:.4f} "
                    f"depth_aux={avg_depth_aux:.4f} normal_aux={avg_normal_aux:.4f} "
                    f"lr={lr:.2e} epoch={epoch} | {task_str}"
                )
                if args.wandb:
                    wandb.log({
                        "train/loss": avg_loss, "train/flow_loss": avg_flow,
                        "train/depth_edge_aux_loss": avg_depth_aux,
                        "train/normal_angular_aux_loss": avg_normal_aux,
                        "train/lr": lr, "train/epoch": epoch,
                    }, step=global_step)
                    for t, v in sorted(task_loss_acc.items()):
                        wandb.log({f"train/task_{t}_loss": v[0]/max(1,v[1])}, step=global_step)
                train_metrics = {"loss": 0.0, "flow_loss": 0.0, "depth_edge_aux_loss": 0.0,
                                 "normal_angular_aux_loss": 0.0, "steps": 0}

            # ── Save checkpoint ──────────────────────────────────────
            if global_step % args.save_every == 0:
                save_checkpoint(output_dir, peft_model, optimizer, lr_scheduler,
                                global_step, None, best_val_loss, args, ema=ema)
                logger.info(f"  Saved checkpoint at step {global_step}")

            # ── Validation ────────────────────────────────────────────
            if val_loader is not None and global_step % args.eval_every == 0:
                ema_backup = ema.apply_shadow(peft_model)
                try:
                    val_loss = _validate(peft_model, vae, pipe, model_cfg, val_loader,
                                         device, model_dtype, args)
                finally:
                    ema.restore_shadow(peft_model, ema_backup)
                logger.info(f"  [val] step={global_step} val_loss={val_loss:.4f}")
                if args.wandb:
                    wandb.log({"val/loss": val_loss}, step=global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(output_dir, peft_model, optimizer, lr_scheduler,
                                    global_step, val_loss, best_val_loss, args, ema=ema, suffix="_best")
                    logger.info(f"  New best val_loss={best_val_loss:.4f}")

    # ── Final save ────────────────────────────────────────────────────
    save_checkpoint(output_dir, peft_model, optimizer, lr_scheduler,
                    global_step, None, best_val_loss, args, ema=ema, suffix="_final")
    logger.info("Training complete! Steps=%d, Best val_loss=%.4f", global_step, best_val_loss)
    if args.wandb:
        wandb.finish()


@torch.no_grad()
def _validate(peft_model, vae, pipe, model_cfg, val_loader, device, model_dtype, args):
    peft_model.eval()
    # Use peft_model directly so LoRA adapters participate in forward pass
    total_loss = 0.0
    n_batches = 0
    max_val_batches = 20
    task_weights = {"depth": 2.5, "normal": 2.0, "semantic": 0.5}

    for i, batch in enumerate(val_loader):
        if i >= max_val_batches:
            break
        prompts = batch["prompt"]
        batch_size = len(prompts)
        target_rgb = batch["target_rgb"].to(device=device, dtype=model_dtype)
        target_pixels = target_rgb * 2.0 - 1.0
        target_latents = flux_vae_encode(vae, target_pixels, model_cfg, model_dtype)

        input_images = batch["image"].to(device=device, dtype=model_dtype)
        input_latents = flux_vae_encode(vae, input_images, model_cfg, model_dtype)
        if input_latents.shape != target_latents.shape:
            input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])

        timesteps = sample_rectified_flow_timesteps(batch_size, device, logit_normal=True).to(model_dtype)
        noise = torch.randn_like(target_latents)
        noisy_latents, velocity_target = mix_rectified_flow(
            target_latents, noise, timesteps, input_latents=input_latents, use_img2img=True)

        text_cond = flux_encode_text(pipe, prompts, device, model_dtype, 512)
        prompt_embeds = text_cond["prompt_embeds"]
        pooled_embeds = text_cond["pooled_prompt_embeds"]
        text_ids = text_cond["text_ids"]

        latent_h, latent_w = noisy_latents.shape[-2:]
        latent_c = noisy_latents.shape[1]
        packed = _pack_latents(noisy_latents, batch_size, latent_c, latent_h, latent_w)
        img_ids = _prepare_latent_image_ids(batch_size, latent_h//2, latent_w//2, device, model_dtype)

        fwd = {
            "hidden_states": packed.to(model_dtype),
            "timestep": timesteps.to(model_dtype),
            "encoder_hidden_states": prompt_embeds.to(model_dtype),
            "pooled_projections": (pooled_embeds.to(model_dtype) if pooled_embeds is not None
                                   else torch.zeros(batch_size, 768, device=device, dtype=model_dtype)),
            "img_ids": img_ids.to(model_dtype),
        }
        if text_ids is not None:
            fwd["txt_ids"] = text_ids.to(model_dtype)
        fwd["guidance"] = torch.full((batch_size,), 50.0, device=device, dtype=model_dtype)

        pred = peft_model(**fwd)
        if hasattr(pred, "sample"):
            pred = pred.sample
        if pred.dim() == 3:
            pred = _unpack_latents(pred, latent_h, latent_w)

        sq_error = (pred.float() - velocity_target.float()).square()
        per_sample = sq_error.flatten(1).mean(dim=1)
        tasks = batch.get("task", ["depth"] * batch_size)
        w = torch.tensor([task_weights.get(t, 1.0) for t in tasks], device=device)
        flow_loss = (per_sample * w).mean()
        total_loss += flow_loss.item()
        n_batches += 1

    peft_model.train()
    return total_loss / max(1, n_batches)


if __name__ == "__main__":
    main()
