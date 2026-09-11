#!/usr/bin/env python3
"""Z-Image LoRA fine-tuning on Apple MPS — expanded multi-task training.

Trains Z-Image (6B) with LoRA rank=128 on:
  - Depth estimation (COCO pseudo + Hypersim GT + existing data)
  - Semantic segmentation (ADE20K)
  - Edge detection (BSDS500)
  - Text-to-image generation (COCO 8K captions)

Usage:
  python scripts/train_zimage_mps.py [--epochs 3] [--lr 2e-5] [--max-steps 100]
  python scripts/train_zimage_mps.py --resume  # auto-resume from latest checkpoint
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset
from open_vision_banana.training.utils import enable_gradient_checkpointing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_zimage")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Z-Image LoRA training on MPS")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps (debug)")
    parser.add_argument("--save-every", type=int, default=50, help="Save every N steps")
    parser.add_argument("--log-every", type=int, default=5, help="Log every N steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--rank", type=int, default=128, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=256, help="LoRA alpha")
    parser.add_argument("--resolution", type=int, default=384, help="Training resolution")
    parser.add_argument("--balance-data", action="store_true", default=True,
                        help="Balance task sampling (oversample minority tasks)")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint dir")
    parser.add_argument("--dry-run", action="store_true", help="Load model and exit")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "depth", "segmentation", "edge", "generation"],
                        help="Which task to train on")
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")

    # ─── Load Z-Image model ──────────────────────────────────────────────
    model_path = "/Users/gatilin/MyWork/models/ms_cache/models/models/Tongyi-MAI--Z-Image/snapshots/master"
    if not Path(model_path).exists():
        logger.error(f"Z-Image model not found at {model_path}")
        sys.exit(1)

    logger.info(f"Loading Z-Image pipeline from {model_path} (bfloat16)...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=model_path,
        device=device,
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Verify model config
    mc = generator.model_config
    logger.info(f"Model family: {mc.family}")
    logger.info(f"Pipeline class: {mc.pipeline_class}")
    logger.info(f"Cross attention dim: {mc.cross_attention_dim}")
    logger.info(f"VAE scaling: {mc.vae_scaling_factor}, shift: {mc.vae_shift_factor}")
    logger.info(f"VAE latent channels: {mc.vae_latent_channels}")
    logger.info(f"LoRA target modules: {mc.lora_target_modules}")

    if args.dry_run:
        logger.info("Dry run complete — model loaded successfully")
        return

    # ─── Setup LoRA training ─────────────────────────────────────────────
    logger.info(f"Setting up LoRA: rank={args.rank}, alpha={args.alpha}, rsLora=True, lr={args.lr}")

    tuner = InstructionTuner(
        model=generator,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        use_rslora=True,
        lr=args.lr,
        use_ema=not args.no_ema,
        ema_decay=0.9999,
        use_min_snr=True,
        use_logit_normal_timesteps=True,
        use_img2img_flow=True,
        cfg_dropout_rate=0.1,
        warmup_ratio=0.05,
    )

    # Enable gradient checkpointing for 6B model on MPS
    enable_gradient_checkpointing(tuner.peft_model)
    logger.info("Gradient checkpointing enabled")

    # ─── Load training data ──────────────────────────────────────────────
    data_dir = PROJECT_ROOT / "data" / "vb_zimage_train"
    res = args.resolution

    # Build datasets from available tasks
    datasets = []

    # Depth task
    if args.task in ("all", "depth"):
        depth_dir = data_dir / "depth"
        if (depth_dir / "images").exists():
            ds = VisionTaskDataset(data_path=str(depth_dir), resolution=res, task="depth")
            if ds.samples:
                datasets.append(ds)
                logger.info(f"Depth: {len(ds)} samples @ {res}px")

    # Segmentation task
    if args.task in ("all", "segmentation"):
        seg_dir = data_dir / "segmentation"
        if (seg_dir / "images").exists():
            ds = VisionTaskDataset(data_path=str(seg_dir), resolution=res, task="semantic")
            if ds.samples:
                datasets.append(ds)
                logger.info(f"Segmentation: {len(ds)} samples @ {res}px")

    # Edge task
    if args.task in ("all", "edge"):
        edge_dir = data_dir / "edge"
        if (edge_dir / "images").exists():
            ds = VisionTaskDataset(data_path=str(edge_dir), resolution=res, task="edge")
            if ds.samples:
                datasets.append(ds)
                logger.info(f"Edge: {len(ds)} samples @ {res}px")

    # Generation task (text-to-image, no target_rgb)
    if args.task in ("all", "generation"):
        gen_dir = data_dir / "generation"
        if (gen_dir / "images").exists():
            # For generation, we create a dataset with prompts but no targets
            import json
            from PIL import Image
            import torch.nn.functional as F

            class GenerationDataset(VisionTaskDataset):
                """Text-to-image dataset using COCO captions."""
                def __init__(self, data_path, resolution=256):
                    self.samples = []
                    self.resolution = resolution
                    self.task = None  # No target
                    prompts_file = Path(data_path) / "prompts.jsonl"
                    prompts_map = {}
                    if prompts_file.exists():
                        with open(prompts_file) as f:
                            for line in f:
                                item = json.loads(line)
                                prompts_map[item["image"]] = item["prompt"]

                    img_dir = Path(data_path) / "images"
                    for img_path in sorted(img_dir.glob("*.jpg")):
                        prompt = prompts_map.get(img_path.name, "a photograph")
                        self.samples.append({
                            "prompt": prompt,
                            "image_path": str(img_path),
                        })

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    from torchvision import transforms
                    s = self.samples[idx]
                    img = Image.open(s["image_path"]).convert("RGB").resize(
                        (self.resolution, self.resolution), Image.LANCZOS
                    )
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
                    ])
                    img_tensor = transform(img)
                    # For generation: target_rgb = image (self-supervised)
                    return {"prompt": s["prompt"], "image": img_tensor, "target_rgb": img_tensor}

            ds = GenerationDataset(str(gen_dir), resolution=res)
            if ds.samples:
                datasets.append(ds)
                logger.info(f"Generation: {len(ds)} samples @ {res}px")

    if not datasets:
        logger.error("No training data found! Run scripts/prepare_zimage_data.py first.")
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
            for ds, sz in zip(datasets, dataset_sizes):
                # Each sample in this task gets weight inversely proportional to task size
                w = max_size / sz  # minority tasks get higher weight
                weights.extend([w] * sz)
            combined = ConcatDataset(datasets)
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(combined), replacement=True
            )
            logger.info(f"Data balanced: " + ", ".join(
                f"{name}={max_size / sz:.1f}x" for name, sz in
                zip(["depth", "seg", "edge", "gen"], dataset_sizes) if sz > 0
            ))
            # With sampler, steps_per_epoch = total_samples (not affected by sampling)
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
    output_dir = PROJECT_ROOT / "output" / "zimage_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(combined)
    steps_per_epoch = max(1, total_samples // (args.batch_size * args.grad_accum))
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(200, max(1, total_steps // 10))  # 200 warmup for larger config

    # Limit dataset to max_steps * grad_accum samples for early stopping (debug mode)
    if args.max_steps is not None:
        max_samples = args.max_steps * args.grad_accum
        if total_samples > max_samples:
            from torch.utils.data import Subset
            indices = list(range(max_samples))
            combined = Subset(combined, indices)
            total_samples = len(combined)
            sampler = None  # Disable sampler in debug mode to avoid index mismatch
            logger.info(f"Limited dataset to {total_samples} samples for {args.max_steps} steps")

    logger.info(f"Starting training: {args.epochs} epochs, {total_samples} samples, "
                f"batch_size={args.batch_size}, grad_accum={args.grad_accum}, "
                f"resolution={args.resolution}, total_steps={total_steps}")

    # ─── Resume from checkpoint ──────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    if args.resume or args.resume_from:
        import glob
        if args.resume_from:
            resume_dir = Path(args.resume_from)
        else:
            # Auto-find latest checkpoint
            ckpt_dirs = sorted(glob.glob(str(output_dir / "checkpoint-*")),
                               key=lambda x: int(x.split("-")[-1]))
            if ckpt_dirs:
                resume_dir = Path(ckpt_dirs[-1])
            else:
                resume_dir = None

        if resume_dir and resume_dir.exists():
            logger.info(f"Resuming from {resume_dir}")
            # Load LoRA weights
            from peft import set_peft_model_state_dict
            adapter_file = resume_dir / "adapter_model.safetensors"
            if adapter_file.exists():
                from safetensors.torch import load_file
                adapter_state = load_file(str(adapter_file))
                set_peft_model_state_dict(tuner.peft_model, adapter_state)
                logger.info(f"Loaded LoRA weights from {adapter_file}")

            # Load metadata for step/epoch
            meta_file = resume_dir / "metadata.json"
            if meta_file.exists():
                import json as _json
                meta = _json.loads(meta_file.read_text())
                start_step = meta.get("step", 0)
                start_epoch = meta.get("epoch", 0)
                logger.info(f"Resumed at step={start_step}, epoch={start_epoch}")
        else:
            logger.warning("Resume requested but no checkpoint found — starting fresh")

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
        eval_every=10000,  # No val dataset
        use_amp=True,
        num_workers=0,  # MPS must be 0
        log_every=args.log_every,
        start_step=start_step,
        start_epoch=start_epoch,
        train_sampler=sampler,
    )

    logger.info(f"Training complete! Result: {result}")
    logger.info(f"Checkpoints saved to: {output_dir}")

    # Save final LoRA weights as .pt for evaluation
    final_lora_path = output_dir / "lora_final.pt"
    lora_state = tuner.peft_model.state_dict()
    # Filter to only LoRA params (lora_A, lora_B)
    lora_only = {k: v for k, v in lora_state.items() if "lora_" in k}
    torch.save(lora_only, final_lora_path)
    logger.info(f"Final LoRA weights saved to: {final_lora_path} ({len(lora_only)} keys)")


if __name__ == "__main__":
    main()
