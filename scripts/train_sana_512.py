#!/usr/bin/env python3
"""Train Open-Vision-Banana with Sana_600M_512px_diffusers.

Uses the diffusers-format Sana-600M 512px model as the backbone for
LoRA-based instruction tuning on the vb_snvc_train_v3 multi-task dataset.

Key characteristics of Sana_600M_512px_diffusers:
  - Backbone: SanaTransformer (linear DiT, ~590M params)
  - VAE: AutoencoderDC (DC-AE, 32x downsample, 32 latent channels)
  - Text encoder: Gemma-2-2B-IT (max_length=300, dim=2048)
  - Resolution: 512x512
  - Flow-matching scheduler (linear_flow, flow_shift=3.0)

The script reuses the project's InstructionTuner and MixedVisionDataset
infrastructure, with Sana-specific adjustments for VAE encoding and
attention masks.

Usage:
    # Full multi-task training (all 12 tasks)
    python scripts/train_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --steps 10000 --lr 5e-5 \
        --output-dir output/sana_512_vb

    # Single-task debug (depth only)
    python scripts/train_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --tasks depth --steps 2000 \
        --output-dir output/sana_512_depth_debug

    # With generation preservation (9:1 mix)
    python scripts/train_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --vision-ratio 0.9 \
        --output-dir output/sana_512_mixed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Use HF mirror for models
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data import (
    MixedVisionDataset,
    build_v3_datasets,
    load_v3_train_val_split,
)
from open_vision_banana.data.split import split_vision_datasets
from open_vision_banana.model_registry import (
    PipelineComponents,
    apply_lora,
    get_model_config,
    load_pipeline,
)
from open_vision_banana.training.optimization import (
    SchedulerFactory,
    sample_timesteps_logit_normal,
    compute_min_snr_weight,
)
from open_vision_banana.training.utils import EMAModel, VisionAugmentation
from open_vision_banana.training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

# Default LoRA targets for Sana transformer blocks
SANA_LORA_TARGETS = [
    "linear",  # attention linear proj
    "proj_out",  # output projection
    "conv_inverted",  # GLUMBConv inverted conv
    "conv_point",  # GLUMBConv pointwise conv
]

# Default task weights (sqrt balancing for imbalanced tasks)
DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "depth": 2.5,
    "normal": 2.0,
    "edge": 1.5,
    "semantic": 0.5,
    "generation": 0.5,
    "instance": 1.0,
    "keypoint": 1.0,
    "object_detection": 1.0,
    "depth_edge": 1.0,
    "referring": 1.5,
    "multiview": 2.0,
    "specular_removal": 2.0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Open-Vision-Banana with Sana_600M_512px_diffusers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────
    p.add_argument("--data-dir", required=True, help="Path to vb_snvc_train_v3 dataset")
    p.add_argument(
        "--tasks", nargs="+", default=None, help="Subset of tasks to train (default: all)"
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution (default 512, matching model native)",
    )
    p.add_argument(
        "--vision-ratio",
        type=float,
        default=1.0,
        help="Ratio of vision vs generation data (default 1.0 = all vision)",
    )

    # ── Model ────────────────────────────────────────────────────────
    p.add_argument(
        "--model-name",
        default="sana-600m-512px",
        help="Model registry name (default: sana-600m-512px)",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-rslora", action="store_true", help="Disable rsLoRA scaling")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    # ── Training ─────────────────────────────────────────────────────
    p.add_argument("--steps", type=int, default=10000, help="Total training steps")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)

    # ── Scheduler ────────────────────────────────────────────────────
    p.add_argument(
        "--scheduler",
        default="cosine",
        choices=["cosine", "constant", "cosine_restarts", "one_cycle"],
    )
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=None)

    # ── Loss ─────────────────────────────────────────────────────────
    p.add_argument("--no-min-snr", action="store_true", help="Disable Min-SNR loss weighting")
    p.add_argument("--min-snr-gamma", type=float, default=5.0)
    p.add_argument(
        "--no-logit-normal-timesteps",
        action="store_true",
        help="Use uniform timesteps instead of logit-normal",
    )
    p.add_argument(
        "--cfg-dropout-rate", type=float, default=0.1, help="CFG dropout rate for text conditioning"
    )
    p.add_argument(
        "--flow-sigma", type=float, default=0.0, help="Flow-matching noise sigma (0 = pure flow)"
    )
    p.add_argument(
        "--no-img2img-flow",
        action="store_true",
        help="Disable img2img-style flow matching (use noise→target instead)",
    )

    # ── Evaluation & Saving ──────────────────────────────────────────
    p.add_argument("--output-dir", default="output/sana_512_vb")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--ema-warmup", type=int, default=500)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)

    # ── Resume ───────────────────────────────────────────────────────
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # ── Device ───────────────────────────────────────────────────────
    p.add_argument("--device", default=None, help="Device (auto-detect if not set)")

    # ── Task weights ─────────────────────────────────────────────────
    p.add_argument(
        "--task-weights",
        default="sqrt",
        help="Task weight mode: sqrt, uniform, or comma-separated floats",
    )

    return p.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype_str
    ]


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_text_sana(
    components: PipelineComponents,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text prompts for Sana using Gemma-2-2B-IT.

    Returns (encoder_hidden_states, attention_mask).
    """
    tokenizer = components.tokenizer
    text_encoder = components.text_encoder

    # Tokenize with padding to max_length
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)

    # Cast text encoder to float32 for stability (Gemma-2B is large)
    original_dtype = text_encoder.dtype
    text_encoder.to(torch.float32)

    with torch.no_grad():
        outputs = text_encoder(input_ids, attention_mask=attention_mask)
        # Gemma returns last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            encoder_hidden_states = outputs.last_hidden_state
        else:
            encoder_hidden_states = outputs[0]

    text_encoder.to(original_dtype)
    return encoder_hidden_states.to(dtype), attention_mask.to(dtype)


def vae_encode_sana(
    vae: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode images to latents using DC-AE.

    DC-AE: 32x downsample, 32 latent channels.
    Input: [B, 3, H, W] in [-1, 1]
    Output: [B, 32, H/32, W/32]

    DC-AE's encode() returns EncoderOutput(latent=...) — NOT latent_dist.
    """
    original_dtype = vae.dtype
    vae.to(dtype)

    with torch.no_grad():
        enc_out = vae.encode(images.to(dtype))
        # DC-AE returns EncoderOutput with .latent attribute
        if hasattr(enc_out, "latent_dist"):
            latents = enc_out.latent_dist.mode()
        elif hasattr(enc_out, "latent"):
            latents = enc_out.latent
        elif hasattr(enc_out, "sample"):
            latents = enc_out.sample
        elif isinstance(enc_out, torch.Tensor):
            latents = enc_out
        else:
            # Fallback: try common attribute names
            for attr in ("latents", "encoded", "z"):
                if hasattr(enc_out, attr):
                    latents = getattr(enc_out, attr)
                    break
            else:
                raise TypeError(
                    f"VAE encode returned {type(enc_out).__name__} with no known "
                    f"latent attribute. Available: {dir(enc_out)}"
                )

    vae.to(original_dtype)
    return latents


def vae_decode_sana(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode latents to images using DC-AE.

    Input: [B, 32, H/32, W/32]
    Output: [B, 3, H, W] in [-1, 1]
    """
    original_dtype = vae.dtype
    vae.to(dtype)

    with torch.no_grad():
        dec_out = vae.decode(latents.to(dtype))
        # DC-AE decode may return DecoderOutput or tensor
        if hasattr(dec_out, "sample"):
            images = dec_out.sample
        elif isinstance(dec_out, torch.Tensor):
            images = dec_out
        else:
            # Try common attribute names
            for attr in ("image", "images", "decoded", "x"):
                if hasattr(dec_out, attr):
                    images = getattr(dec_out, attr)
                    break
            else:
                raise TypeError(
                    f"VAE decode returned {type(dec_out).__name__} with no known "
                    f"sample attribute. Available: {dir(dec_out)}"
                )

    vae.to(original_dtype)
    return images


def flexible_collate(batch: list[dict]) -> dict:
    """Custom collate that handles inconsistent keys across samples.

    Different vision tasks may return different optional keys (e.g. only
    depth tasks include 'colormap').  ``default_collate`` requires every
    sample to share the exact same key set, so we:
      1. Compute the intersection of keys across the batch.
      2. Collate only those common keys with ``default_collate``.
      3. Any non-common key is dropped (the training loop does not use
         task-specific extras like 'colormap').
    """
    all_keys = set(batch[0].keys())
    for sample in batch[1:]:
        all_keys &= set(sample.keys())
    common = [{k: s[k] for k in all_keys} for s in batch]
    from torch.utils.data._utils.collate import default_collate

    return default_collate(common)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    device = torch.device(args.device or auto_device())
    dtype = get_dtype(args.dtype)

    logger.info("=" * 70)
    logger.info("Open-Vision-Banana — Sana_600M_512px Training")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {device} | dtype: {dtype}")
    logger.info(
        f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, rsLoRA={not args.no_rslora}"
    )
    logger.info(
        f"Training: steps={args.steps}, lr={args.lr}, effective_batch={args.batch_size * args.grad_accum}"
    )
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Resolution: {args.resolution}")
    logger.info(f"Output: {args.output_dir}")

    # ── Set seed ─────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load model ───────────────────────────────────────────────────
    logger.info("Loading Sana_600M_512px_diffusers...")
    model_cfg = get_model_config(args.model_name)
    logger.info(f"  model_id: {model_cfg.model_id}")
    logger.info(f"  family: {model_cfg.family}")
    logger.info(f"  VAE downsample: {model_cfg.vae_downsample_factor}x")
    logger.info(f"  text_encoder_max_length: {model_cfg.text_encoder_max_length}")

    components = load_pipeline(
        model_cfg,
        device=device,
        dtype=dtype,
        enable_gradient_checkpointing=args.gradient_checkpointing,
    )

    # Sana-specific dtype adjustments (per model card recommendations):
    #   - transformer stays in fp16/bf16 (training dtype)
    #   - text_encoder (Gemma-2B) in bfloat16 for stability
    #   - VAE (DC-AE) in bfloat16 or float32
    if dtype == torch.float16:
        logger.info("Applying Sana dtype adjustments: text_encoder→bf16, vae→bf16")
        components.text_encoder.to(torch.bfloat16)
        components.vae.to(torch.bfloat16)

    backbone = components.unet  # transformer
    vae = components.vae

    backbone_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"  Backbone params: {backbone_params / 1e6:.1f}M")

    # ── Apply LoRA ───────────────────────────────────────────────────
    logger.info("Applying LoRA adapters...")
    peft_model = apply_lora(
        backbone,
        config=model_cfg,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        use_rslora=not args.no_rslora,
    )

    # ── EMA ──────────────────────────────────────────────────────────
    ema = EMAModel(peft_model, decay=args.ema_decay, warmup_steps=args.ema_warmup)

    # ── Build datasets ───────────────────────────────────────────────
    logger.info("Building datasets...")
    data_root = Path(args.data_dir)

    v3_split = load_v3_train_val_split(data_root)
    is_v3 = v3_split is not None or (data_root / "manifest.json").exists()

    if not is_v3:
        logger.error("Dataset must be in v3 format (with train_val_split.json or manifest.json)")
        return

    logger.info(f"Detected v3-format dataset at {data_root}")

    # Determine which tasks to load
    all_tasks = [
        "depth",
        "normal",
        "segmentation",
        "structure",
        "generation",
        "instance",
        "keypoint",
        "object_detection",
        "depth_edge",
        "referring",
        "multiview",
        "specular_removal",
    ]
    tasks_to_load = args.tasks if args.tasks else all_tasks

    # Filter v3 split to selected tasks
    train_vision_datasets = build_v3_datasets(
        data_root,
        resolution=args.resolution,
        split="train",
        use_split_file=v3_split is not None,
    )
    # Filter by requested tasks
    train_vision_datasets = [
        ds
        for ds in train_vision_datasets
        if ds.task in tasks_to_load or ds.dir_name in tasks_to_load
    ]

    for ds in train_vision_datasets:
        logger.info(f"  [train] {ds.dir_name}/{ds.task}: {len(ds)} samples")

    if not train_vision_datasets:
        logger.error("No training datasets loaded. Check --data-dir and --tasks")
        return

    # Build validation set
    val_dataset = None
    if v3_split is not None:
        val_datasets = build_v3_datasets(
            data_root,
            resolution=args.resolution,
            split="val",
            use_split_file=True,
        )
        val_datasets = [
            ds for ds in val_datasets if ds.task in tasks_to_load or ds.dir_name in tasks_to_load
        ]
        if val_datasets:
            from torch.utils.data import ConcatDataset

            val_dataset = ConcatDataset(val_datasets)
            logger.info(f"  [val] Validation set: {len(val_dataset)} samples")

    # Generation dataset (for 9:1 mix)
    generation_dataset = None
    gen_dir = data_root / "generation"
    if gen_dir.is_dir() and args.vision_ratio < 1.0:
        from open_vision_banana.data import build_generation_dataset

        gen_split_filter = None
        if v3_split is not None:
            gen_filenames = v3_split.get("train", {}).get("generation")
            if gen_filenames is not None:
                gen_split_filter = (set(gen_filenames), "train")
        generation_dataset = build_generation_dataset(
            str(gen_dir),
            resolution=args.resolution,
            split_filter=gen_split_filter,
        )
        if len(generation_dataset) > 0:
            logger.info(f"  Generation data: {len(generation_dataset)} samples")
        else:
            generation_dataset = None

    # Task weights — pass directly to MixedVisionDataset (supports "sqrt", "uniform", or list)
    task_weights = args.task_weights
    if task_weights == "uniform":
        task_weights = None
    elif task_weights not in ("sqrt", None):
        # Parse comma-separated floats
        try:
            weights_list = [float(w.strip()) for w in task_weights.split(",")]
            if len(weights_list) != len(train_vision_datasets):
                logger.warning(
                    f"task_weights count ({len(weights_list)}) != "
                    f"num datasets ({len(train_vision_datasets)}); ignoring"
                )
                task_weights = None
            else:
                task_weights = weights_list
        except ValueError:
            logger.warning(f"Invalid task_weights value: {args.task_weights}; using sqrt")
            task_weights = "sqrt"

    if task_weights:
        logger.info(f"  Task weights mode: {task_weights}")

    mixed_ds = MixedVisionDataset(
        vision_datasets=train_vision_datasets,
        generation_dataset=generation_dataset,
        vision_ratio=args.vision_ratio if generation_dataset else 1.0,
        seed=args.seed,
        task_weights=task_weights,
    )
    logger.info(f"  Mixed dataset total: {len(mixed_ds)} samples")

    # ── DataLoader ───────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    persistent = args.num_workers > 0

    train_loader = DataLoader(
        mixed_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=persistent,
        collate_fn=flexible_collate,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, 4),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=persistent,
            collate_fn=flexible_collate,
        )

    # ── Optimizer & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    warmup_steps = args.warmup_steps or int(args.steps * args.warmup_ratio)
    lr_scheduler = SchedulerFactory.create(
        optimizer=optimizer,
        total_steps=args.steps,
        warmup_steps=warmup_steps,
        scheduler_type=args.scheduler,
    )

    # ── Training state ───────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dump = {
        "model": args.model_name,
        "model_id": model_cfg.model_id,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "rslora": not args.no_rslora,
        "lr": args.lr,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "resolution": args.resolution,
        "dtype": args.dtype,
        "flow_sigma": args.flow_sigma,
        "cfg_dropout_rate": args.cfg_dropout_rate,
        "use_img2img_flow": not args.no_img2img_flow,
        "use_min_snr": not args.no_min_snr,
        "use_logit_normal": not args.no_logit_normal_timesteps,
        "ema_decay": args.ema_decay,
        "tasks": [d.task for d in train_vision_datasets],
        "seed": args.seed,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config_dump, indent=2, ensure_ascii=False)
    )

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    # Resume
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            logger.info(f"Resuming from {resume_path}")
            ckpt = torch.load(str(resume_path), map_location=device)
            peft_model.load_state_dict(ckpt.get("lora_state_dict", ckpt), strict=False)
            global_step = ckpt.get("step", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            logger.info(f"  Resumed at step {global_step}, best_val_loss={best_val_loss:.4f}")

    # ── Training loop ────────────────────────────────────────────────
    logger.info("Starting training...")
    model_dtype = next(peft_model.parameters()).dtype
    vae_downsample = model_cfg.vae_downsample_factor  # 32 for DC-AE
    max_length = model_cfg.text_encoder_max_length  # 300

    # Adjust for device limitations
    use_amp = device.type == "cuda" and dtype != torch.float32
    amp_dtype = torch.bfloat16 if device.type == "mps" else dtype
    scaler = torch.amp.GradScaler("cuda") if use_amp and dtype == torch.float16 else None

    train_metrics = {"loss": 0.0, "steps": 0}

    while global_step < args.steps:
        epoch = start_epoch + (global_step // max(1, len(train_loader)))
        peft_model.train()

        for batch in train_loader:
            if global_step >= args.steps:
                break

            prompts = batch.get("prompt", [""] * args.batch_size)
            if isinstance(prompts, str):
                prompts = [prompts]

            # Load images and targets
            # image: [B, 3, H, W] in [-1, 1] (input image)
            # target_rgb: [B, 3, H, W] in [0, 1] (target visualization)
            input_images = batch["image"].to(device=device, dtype=model_dtype)
            target_rgb = batch["target_rgb"].to(device=device, dtype=model_dtype)

            # Sana expects images in [-1, 1] for VAE encode
            target_images = target_rgb * 2.0 - 1.0  # [0,1] → [-1,1]

            # ── Encode target to latents ──────────────────────────────
            with torch.no_grad():
                target_latents = vae_encode_sana(vae, target_images, device, model_dtype)

                # Encode input image for img2img flow
                input_latents = None
                if not args.no_img2img_flow:
                    input_latents = vae_encode_sana(vae, input_images, device, model_dtype)
                    # Ensure same spatial dims
                    if input_latents.shape != target_latents.shape:
                        input_latents = F.adaptive_avg_pool2d(
                            input_latents, target_latents.shape[-2:]
                        )

            batch_size = target_latents.shape[0]

            # ── Sample timesteps ──────────────────────────────────────
            if not args.no_logit_normal_timesteps:
                timesteps = sample_timesteps_logit_normal(batch_size, device).to(model_dtype)
            else:
                timesteps = torch.rand(batch_size, device=device).to(model_dtype)

            noise = torch.randn_like(target_latents)
            t_b = timesteps.view(batch_size, *([1] * (target_latents.dim() - 1)))

            # ── Flow matching ─────────────────────────────────────────
            if not args.no_img2img_flow and input_latents is not None:
                # img2img flow: input → target
                if args.flow_sigma > 0:
                    noise_term = args.flow_sigma * (t_b * (1.0 - t_b)).sqrt() * noise
                else:
                    noise_term = torch.zeros_like(input_latents)
                noisy_latents = (1.0 - t_b) * input_latents + t_b * target_latents + noise_term
                velocity_target = target_latents - input_latents
            else:
                # Standard flow: noise → target
                noisy_latents = (1.0 - t_b) * target_latents + t_b * noise
                velocity_target = target_latents - noise

            # ── Text encoding ─────────────────────────────────────────
            with torch.no_grad():
                prompt_embeds, attention_mask = encode_text_sana(
                    components, prompts, device, model_dtype, max_length
                )

                # CFG dropout
                if args.cfg_dropout_rate > 0:
                    drop_mask = torch.rand(batch_size, device=device) < args.cfg_dropout_rate
                    if drop_mask.any():
                        uncond_embeds, uncond_mask = encode_text_sana(
                            components, [""] * batch_size, device, model_dtype, max_length
                        )
                        drop = drop_mask.view(-1, *([1] * (prompt_embeds.dim() - 1)))
                        prompt_embeds = torch.where(drop, uncond_embeds, prompt_embeds)
                        if attention_mask is not None:
                            drop_mask_2d = drop_mask.view(-1, 1)
                            attention_mask = torch.where(drop_mask_2d, uncond_mask, attention_mask)

            # ── Forward pass ──────────────────────────────────────────
            autocast_ctx = torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp or device.type == "mps",
            )

            with autocast_ctx:
                predicted_velocity = peft_model(
                    noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )[0]

                # Velocity loss (flow matching)
                loss = F.mse_loss(predicted_velocity.float(), velocity_target.float())

                # Min-SNR weighting
                if not args.no_min_snr:
                    snr_weights = compute_min_snr_weight(
                        timesteps,
                        args.min_snr_gamma,
                    )
                    if snr_weights is not None:
                        loss = (loss * snr_weights.view(-1, *([1] * (loss.dim() - 1)))).mean()

                loss = loss / args.grad_accum

            # ── Backward ──────────────────────────────────────────────
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_metrics["loss"] += loss.item() * args.grad_accum
            train_metrics["steps"] += 1

            # ── Optimizer step (with gradient accumulation) ───────────
            if (global_step + 1) % args.grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in peft_model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in peft_model.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                ema.update(peft_model.parameters())

            global_step += 1

            # ── Logging ──────────────────────────────────────────────
            if global_step % args.log_every == 0:
                avg_loss = train_metrics["loss"] / max(1, train_metrics["steps"])
                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"  [step {global_step}/{args.steps}] "
                    f"loss={avg_loss:.4f} lr={lr:.2e} "
                    f"epoch={epoch} "
                    f"latent_shape={list(target_latents.shape[1:])}"
                )
                train_metrics = {"loss": 0.0, "steps": 0}

            # ── Save checkpoint ──────────────────────────────────────
            if global_step % args.save_every == 0:
                save_checkpoint(
                    output_dir,
                    peft_model,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    avg_loss if train_metrics["steps"] > 0 else None,
                    best_val_loss,
                    args,
                )
                logger.info(f"  Saved checkpoint at step {global_step}")

            # ── Validation ───────────────────────────────────────────
            if val_loader is not None and global_step % args.eval_every == 0:
                val_loss = validate(
                    peft_model,
                    vae,
                    components,
                    val_loader,
                    device,
                    model_dtype,
                    args,
                    vae_downsample,
                    max_length,
                )
                logger.info(f"  [val] step={global_step} val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir,
                        peft_model,
                        optimizer,
                        lr_scheduler,
                        global_step,
                        val_loss,
                        best_val_loss,
                        args,
                        suffix="_best",
                    )
                    logger.info(f"  New best val_loss={best_val_loss:.4f}")

    # ── Final save ──────────────────────────────────────────────────
    save_checkpoint(
        output_dir,
        peft_model,
        optimizer,
        lr_scheduler,
        global_step,
        None,
        best_val_loss,
        args,
        suffix="_final",
    )
    logger.info("Training complete!")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Best val_loss: {best_val_loss:.4f}")
    logger.info(f"  Checkpoints saved to: {output_dir}")


def save_checkpoint(
    output_dir: Path,
    peft_model,
    optimizer,
    lr_scheduler,
    step: int,
    val_loss: float | None,
    best_val_loss: float,
    args,
    suffix: str = "",
):
    """Save LoRA checkpoint with training state."""
    ckpt_dir = output_dir / f"checkpoint-{step}{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter weights
    lora_state = {}
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu().clone()

    ckpt = {
        "lora_state_dict": lora_state,
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    }

    torch.save(ckpt, str(ckpt_dir / "lora.pt"))

    # Also save as safetensors if possible
    try:
        peft_model.save_pretrained(str(ckpt_dir / "lora_adapter"))
        logger.info(f"  LoRA adapter saved to {ckpt_dir / 'lora_adapter'}")
    except Exception as e:
        logger.warning(f"  Could not save LoRA adapter via save_pretrained: {e}")

    # Save metadata
    meta = {
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "model_name": args.model_name,
    }
    (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


@torch.no_grad()
def validate(
    peft_model,
    vae,
    components: PipelineComponents,
    val_loader,
    device: torch.device,
    model_dtype: torch.dtype,
    args,
    vae_downsample: int,
    max_length: int,
) -> float:
    """Run validation and return average loss."""
    peft_model.eval()
    total_loss = 0.0
    num_batches = 0
    max_val_batches = 20  # Limit val batches for speed

    for i, batch in enumerate(val_loader):
        if i >= max_val_batches:
            break

        prompts = batch.get("prompt", [""])
        if isinstance(prompts, str):
            prompts = [prompts]

        target_rgb = batch["target_rgb"].to(device=device, dtype=model_dtype)
        target_images = target_rgb * 2.0 - 1.0

        target_latents = vae_encode_sana(vae, target_images, device, model_dtype)

        input_latents = None
        if not args.no_img2img_flow:
            input_images = batch["image"].to(device=device, dtype=model_dtype)
            input_latents = vae_encode_sana(vae, input_images, device, model_dtype)
            if input_latents.shape != target_latents.shape:
                input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])

        batch_size = target_latents.shape[0]

        if not args.no_logit_normal_timesteps:
            timesteps = sample_timesteps_logit_normal(batch_size, device).to(model_dtype)
        else:
            timesteps = torch.rand(batch_size, device=device).to(model_dtype)

        noise = torch.randn_like(target_latents)
        t_b = timesteps.view(batch_size, *([1] * (target_latents.dim() - 1)))

        if not args.no_img2img_flow and input_latents is not None:
            noisy_latents = (1.0 - t_b) * input_latents + t_b * target_latents
            velocity_target = target_latents - input_latents
        else:
            noisy_latents = (1.0 - t_b) * target_latents + t_b * noise
            velocity_target = target_latents - noise

        prompt_embeds, attention_mask = encode_text_sana(
            components, prompts, device, model_dtype, max_length
        )

        predicted_velocity = peft_model(
            noisy_latents,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=attention_mask,
            timestep=timesteps,
            return_dict=False,
        )[0]

        loss = F.mse_loss(predicted_velocity.float(), velocity_target.float())
        total_loss += loss.item()
        num_batches += 1

    peft_model.train()
    return total_loss / max(1, num_batches)


if __name__ == "__main__":
    main()
