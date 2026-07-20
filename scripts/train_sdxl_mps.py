#!/usr/bin/env python3
"""SDXL/SSD-1B LoRA fine-tuning on Apple MPS — full multi-task training.

Trains SSD-1B (SDXL-distilled, 1.3B UNet) or full SDXL (2.6B UNet) with LoRA on:
  - Depth estimation (COCO pseudo + SNVC)
  - Normal estimation (SNVC)
  - Semantic segmentation (ADE20K-style)
  - Edge detection (BSDS500-style)
  - Text-to-image generation (COCO captions)

Uses the unified InstructionTuner + BaseGenerator infrastructure, which handles
SDXL-specific concerns:
  - Dual text encoder (CLIP-L + OpenCLIP-bigG) → concatenated hidden states + pooled embeds
  - added_cond_kwargs (pooled text_embeds + time_ids) required by SDXL UNet
  - scheduler-aware SDXL epsilon/v-prediction objective

Usage:
  python scripts/train_sdxl_mps.py                              # full training, defaults
  python scripts/train_sdxl_mps.py --model sdxl                 # full SDXL (2.6B)
  python scripts/train_sdxl_mps.py --max-steps 50 --dry-run    # smoke test
  python scripts/train_sdxl_mps.py --resume                     # auto-resume
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset  # noqa: E402
from open_vision_banana.training.utils import enable_gradient_checkpointing  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_sdxl")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed all RNGs used by the training and sampling pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="SDXL/SSD-1B LoRA training on MPS")
    parser.add_argument("--model", type=str, default="ssd-1b",
                        choices=["ssd-1b", "sdxl"],
                        help="Base model (default: ssd-1b, use sdxl for full 2.6B)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps")
    parser.add_argument("--save-every", type=int, default=200, help="Save every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--rank", type=int, default=128, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=256, help="LoRA alpha")
    parser.add_argument("--resolution", type=int, default=256, help="Training resolution")
    parser.add_argument("--max-per-task", type=int, default=300,
                        help="Max samples per vision task (balanced sampling)")
    parser.add_argument("--max-generation", type=int, default=3000,
                        help="Max generation samples")
    parser.add_argument("--balance-data", action="store_true", default=True,
                        help="Balance task sampling (oversample minority tasks)")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint dir")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed")
    parser.add_argument("--dry-run", action="store_true", help="Load model and exit")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "depth", "normal", "segmentation", "edge", "generation"],
                        help="Which task to train on")
    args = parser.parse_args()

    set_seed(args.seed)

    device = get_device()
    logger.info(f"Device: {device}")

    # ─── Load SDXL/SSD-1B model ───────────────────────────────────────────
    # Use registered model name → resolve to HF repo ID for BaseGenerator.
    # SSD-1B is preferred on MPS (5.5GB vs 7GB for full SDXL).
    # Map short names to HF repo IDs so diffusers can use the local cache.
    _MODEL_ID_MAP = {
        "ssd-1b": "segmind/SSD-1B",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    model_name = _MODEL_ID_MAP.get(args.model, args.model)
    logger.info(f"Loading {args.model} → {model_name} pipeline (bfloat16)...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=model_name,
        device=str(device),
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Verify model config
    mc = generator.model_config
    logger.info(f"Model family: {mc.family}")
    logger.info(f"Pipeline class: {mc.pipeline_class}")
    logger.info(f"Cross attention dim: {mc.cross_attention_dim}")
    logger.info(f"VAE scaling: {mc.vae_scaling_factor}")
    logger.info(f"VAE latent channels: {mc.vae_latent_channels}")
    logger.info(f"LoRA target modules: {mc.lora_target_modules}")
    logger.info(f"Text encoder max length: {mc.text_encoder_max_length}")
    logger.info(f"Needs added_cond_kwargs: {mc.needs_added_cond_kwargs}")
    logger.info(f"Needs pooled projections: {mc.needs_pooled_projections}")
    logger.info(f"Memory budget: {mc.memory_budget_gb}GB")

    if args.dry_run:
        logger.info("Dry run complete — model loaded successfully")
        return

    # ─── Setup LoRA training ─────────────────────────────────────────────
    logger.info(f"Setting up LoRA: rank={args.rank}, alpha={args.alpha}, lr={args.lr}")

    tuner = InstructionTuner(
        model=generator,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        use_rslora=True,  # Enable rsLoRA for better high-rank training
        lr=args.lr,
        use_ema=not args.no_ema,
        ema_decay=0.9999,
        use_min_snr=True,
        use_logit_normal_timesteps=False,
        use_img2img_flow=False,
        cfg_dropout_rate=0.1,
        warmup_ratio=0.05,
    )

    # Enable gradient checkpointing for UNet on MPS.
    # diffusers UNet uses `_gradient_checkpointing_func` (set by
    # `enable_gradient_checkpointing()`), not the raw `gradient_checkpointing`
    # flag that utils.enable_gradient_checkpointing toggles.
    unet = getattr(generator.pipe, "unet", None)
    if unet is not None and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on UNet (diffusers native)")
    else:
        enable_gradient_checkpointing(tuner.peft_model)
        logger.info("Gradient checkpointing enabled (generic fallback)")

    # ─── Load training data from ALL available sources ───────────────────
    res = args.resolution
    # Multiple data directories — merge all available data
    data_dirs = [
        PROJECT_ROOT / "data" / "vb_zimage_train",
        PROJECT_ROOT / "data" / "vb_snvc_train",
    ]

    # Generation dataset class (defined at function scope for reuse)
    import json as _json_mod
    from PIL import Image as _PIL_Image  # noqa: I001

    class GenerationDataset(VisionTaskDataset):
        """Text-to-image dataset using COCO captions."""
        def __init__(self, data_path, resolution=256, max_samples=None):
            self.samples = []
            self.resolution = resolution
            self.task = None
            prompts_file = Path(data_path) / "prompts.jsonl"
            prompts_map = {}
            if prompts_file.exists():
                with open(prompts_file) as f:
                    for line in f:
                        item = _json_mod.loads(line)
                        prompts_map[item["image"]] = item["prompt"]
            img_dir = Path(data_path) / "images"
            for img_path in sorted(img_dir.glob("*.jpg")):
                prompt = prompts_map.get(img_path.name, "a photograph")
                self.samples.append({
                    "prompt": prompt,
                    "image_path": str(img_path),
                })
            if max_samples is not None and len(self.samples) > max_samples:
                self.samples = self.samples[:max_samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            from torchvision import transforms
            s = self.samples[idx]
            img = _PIL_Image.open(s["image_path"]).convert("RGB").resize(
                (self.resolution, self.resolution), _PIL_Image.LANCZOS
            )
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            img_tensor = transform(img)
            # P0-2 fix: text-to-image generation must NOT pass the input image
            # as conditioning. Previously image == target_rgb, which made the
            # img2img flow velocity_target = target - input = 0, teaching the
            # model to output zero velocity (i.e. "generate nothing") for ~22%
            # of all training steps. Returning image=None routes generation
            # through the pure t2i branch: noisy = (1-t)*target + t*noise,
            # velocity_target = target - noise (standard flow matching).
            return {
                "prompt": s["prompt"], "image": None, "target_rgb": img_tensor,
                "target_rgb_range": "-1_1",
            }

    datasets = []

    def _load_task_dirs(task_name, subdir, vision_task, max_per_task=None):
        """Load and merge a task from all data directories."""
        merged = []
        for d in data_dirs:
            task_dir = d / subdir
            if (task_dir / "images").exists():
                ds = VisionTaskDataset(data_path=str(task_dir), resolution=res, task=vision_task)
                if ds.samples:
                    if max_per_task is not None and len(ds.samples) > max_per_task:
                        # Cap samples per task for balance
                        ds.samples = ds.samples[:max_per_task]
                    merged.append(ds)
                    logger.info(f"  {task_name} from {d.name}: {len(ds)} samples")
        return merged

    def _load_generation_dirs():
        """Load and merge generation data from all directories."""
        merged = []
        for d in data_dirs:
            gen_dir = d / "generation"
            if (gen_dir / "images").exists():
                ds = GenerationDataset(str(gen_dir), resolution=res, max_samples=args.max_generation)
                if ds.samples:
                    merged.append(ds)
                    logger.info(f"  generation from {d.name}: {len(ds)} samples")
        return merged

    # Depth task
    if args.task in ("all", "depth"):
        datasets.extend(_load_task_dirs("depth", "depth", "depth", max_per_task=args.max_per_task))

    # Normal task
    if args.task in ("all", "normal"):
        datasets.extend(_load_task_dirs("normal", "normal", "normal", max_per_task=args.max_per_task))

    # Segmentation task
    # P0-3 fix: only load the Cityscapes-formatted split (vb_snvc_train), whose
    # targets are regenerated in the official 19-color palette by
    # scripts/regen_cityscapes_seg_color.py — the same convention the eval
    # decoder (decode_semantic + CITYSCAPES_COLORS) uses.
    # The vb_zimage_train ADE20K split is excluded: its 150-class space
    # conflicts with Cityscapes' 19-class space, so the same prompt would
    # receive two contradictory color semantics (supervision conflict).
    if args.task in ("all", "segmentation"):
        snvc_seg = PROJECT_ROOT / "data" / "vb_snvc_train" / "segmentation"
        if (snvc_seg / "images").exists():
            ds = VisionTaskDataset(data_path=str(snvc_seg), resolution=res, task="semantic")
            if ds.samples:
                if len(ds.samples) > args.max_per_task:
                    ds.samples = ds.samples[:args.max_per_task]
                datasets.append(ds)
                logger.info(f"  segmentation(cityscapes-color) from vb_snvc_train: {len(ds)} samples")

    # Edge task (edge from vb_zimage_train, structure from vb_snvc_train)
    if args.task in ("all", "edge"):
        datasets.extend(_load_task_dirs("edge", "edge", "edge", max_per_task=args.max_per_task))
        # Also load 'structure' dir from vb_snvc_train as edge-like data
        for d in data_dirs:
            struct_dir = d / "structure"
            if (struct_dir / "images").exists():
                ds = VisionTaskDataset(data_path=str(struct_dir), resolution=res, task="edge")
                if ds.samples:
                    if len(ds.samples) > args.max_per_task:
                        ds.samples = ds.samples[:args.max_per_task]
                    datasets.append(ds)
                    logger.info(f"  structure(edge) from {d.name}: {len(ds)} samples")

    # Generation task
    if args.task in ("all", "generation"):
        datasets.extend(_load_generation_dirs())

    if not datasets:
        logger.error("No training data found! Run data preparation scripts first.")
        sys.exit(1)

    # Combine all datasets with balanced sampling
    from torch.utils.data import ConcatDataset
    if len(datasets) > 1:
        if args.balance_data:
            # WeightedRandomSampler: oversample minority tasks to balance
            from torch.utils.data import WeightedRandomSampler
            dataset_sizes = [len(d) for d in datasets]
            max_size = max(dataset_sizes)
            weights = []
            for _ds, sz in zip(datasets, dataset_sizes, strict=True):
                w = max_size / sz  # minority tasks get higher weight
                weights.extend([w] * sz)
            combined = ConcatDataset(datasets)
            sampler_generator = torch.Generator().manual_seed(args.seed)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(combined),
                replacement=True,
                generator=sampler_generator,
            )
            logger.info("Data balanced: " + ", ".join(
                f"ds{i}={max_size / sz:.1f}x" for i, sz in enumerate(dataset_sizes) if sz > 0
            ))
            total_samples = len(combined)
        else:
            combined = ConcatDataset(datasets)
            sampler = None
            total_samples = len(combined)
    else:
        combined = datasets[0]
        sampler = None
        total_samples = len(combined)
    logger.info(f"Total training samples: {total_samples}")

    # ─── Training ────────────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "output" / f"{args.model}_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(combined)
    steps_per_epoch = max(
        1,
        (total_samples + args.batch_size * args.grad_accum - 1)
        // (args.batch_size * args.grad_accum),
    )
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(200, max(1, total_steps // 10))

    # Limit dataset to max_steps * grad_accum samples for early stopping (debug mode)
    if args.max_steps is not None:
        max_samples = args.max_steps * args.grad_accum
        if total_samples > max_samples:
            from torch.utils.data import Subset
            subset_generator = torch.Generator().manual_seed(args.seed + 1)
            indices = torch.randperm(total_samples, generator=subset_generator)[:max_samples].tolist()
            combined = Subset(combined, indices)
            total_samples = len(combined)
            sampler = None
            logger.info(f"Limited dataset to {total_samples} samples for {args.max_steps} steps")

    logger.info(f"Starting training: {args.epochs} epochs, {total_samples} samples, "
                f"batch_size={args.batch_size}, grad_accum={args.grad_accum}, "
                f"resolution={args.resolution}, total_steps={total_steps}")

    # ─── Resume from checkpoint ──────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    resume_checkpoint = None
    if args.resume or args.resume_from:
        import glob
        if args.resume_from:
            resume_dir = Path(args.resume_from)
        else:
            ckpt_dirs = sorted(glob.glob(str(output_dir / "checkpoint-*")),
                               key=lambda x: int(x.split("-")[-1]))
            resume_dir = Path(ckpt_dirs[-1]) if ckpt_dirs else None

        if resume_dir and resume_dir.exists():
            logger.info(f"Resuming from {resume_dir}")
            meta_file = resume_dir / "metadata.json"
            if meta_file.exists():
                import json as _json
                meta = _json.loads(meta_file.read_text())
                start_step = meta.get("step", 0)
                start_epoch = meta.get("epoch", 0)
                logger.info(f"Resumed at step={start_step}, epoch={start_epoch}")
            resume_checkpoint = resume_dir
        else:
            raise FileNotFoundError(
                f"Resume requested but no checkpoint found: {resume_dir}"
            )

    result = tuner.train(
        train_dataset=combined,
        val_dataset=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        scheduler_type="cosine",
        warmup_steps=warmup_steps,
        output_dir=str(output_dir),
        save_every=args.save_every,
        eval_every=10000,
        use_amp=True,
        num_workers=0,  # MPS must be 0
        log_every=args.log_every,
        start_step=start_step,
        start_epoch=start_epoch,
        train_sampler=sampler,
        resume_checkpoint=resume_checkpoint,
        restore_rng_on_resume=False,  # Old checkpoints may lack rng_state.pt
    )

    logger.info(f"Training complete! Result: {result}")
    logger.info(f"Checkpoints saved to: {output_dir}")

    # Save final LoRA weights as .pt for evaluation
    final_lora_path = output_dir / "lora_final.pt"
    lora_state = tuner.peft_model.state_dict()
    lora_only = {k: v for k, v in lora_state.items() if "lora_" in k}
    torch.save(lora_only, final_lora_path)
    logger.info(f"Final LoRA weights saved to: {final_lora_path} ({len(lora_only)} keys)")


if __name__ == "__main__":
    main()
