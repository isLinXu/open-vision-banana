#!/usr/bin/env python3
"""Training script for Open-Vision-Banana.

Implements the instruction-tuning pipeline from the paper:
- Mix vision task data at 9:1 ratio with base model's training data
- Use rsLoRA (rank=64, alpha=128) for parameter-efficient fine-tuning
- Train only 0.5%-1% of total parameters
- Preserve 90%+ generation capability

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --data-path /path/to/data --epochs 5
    python scripts/train.py --help
"""

import argparse
import logging
from pathlib import Path

import torch

from open_vision_banana import VisionBananaConfig
from open_vision_banana.data import build_dataset, MixedVisionDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Vision Banana (instruction-tuning FLUX.1)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to training config file",
    )
    # Override flags (take precedence over config file)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output/checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype_str]


def _auto_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = VisionBananaConfig.from_yaml(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        cfg = VisionBananaConfig()

    # Apply CLI overrides
    if args.model_name:
        cfg.model.name = args.model_name
    if args.lora_rank:
        cfg.lora.rank = args.lora_rank
    if args.lora_alpha:
        cfg.lora.alpha = args.lora_alpha
    if args.lr:
        cfg.optimizer.lr = args.lr
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.data.batch_size = args.batch_size

    output_dir = args.output_dir
    seed = args.seed
    device = args.device or _auto_device()

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Print config summary
    logger.info("=" * 60)
    logger.info("Vision Banana Training")
    logger.info("=" * 60)
    logger.info(f"Base model: {cfg.model.name}")
    logger.info(f"LoRA rank: {cfg.lora.rank}, alpha: {cfg.lora.alpha} (rsLoRA)")
    logger.info(f"Learning rate: {cfg.optimizer.lr}")
    logger.info(f"Epochs: {cfg.training.epochs}, Batch size: {cfg.data.batch_size}")
    logger.info(f"Device: {device}, dtype: {cfg.model.dtype}")
    logger.info(f"Data mix: {cfg.data.vision_ratio:.0%} vision data")
    logger.info(f"Output: {output_dir}")

    # Load base FLUX.1 model
    logger.info("Loading base FLUX.1 model...")
    from open_vision_banana.inference.generator import BaseGenerator
    from open_vision_banana.training.tuner import InstructionTuner

    generator = BaseGenerator(
        model_name=cfg.model.name,
        device=device,
        dtype=get_dtype(cfg.model.dtype),
    )

    # Set up instruction tuner with rsLoRA
    logger.info("Setting up LoRA instruction alignment...")
    tuner = InstructionTuner(
        model=generator.pipe,
        lora_rank=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        use_rslora=cfg.lora.rs_lora,
        lr=cfg.optimizer.lr,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        from peft import PeftModel
        tuner.peft_model = PeftModel.from_pretrained(
            tuner.peft_model, args.resume
        )

    # Build datasets
    vision_datasets = []

    if args.data_path:
        # Auto-detect task from directory structure
        for task in ["depth", "normal", "semantic", "instance", "referring"]:
            try:
                ds = build_dataset(task, args.data_path, resolution=cfg.data.resolution)
                if len(ds) > 0:
                    vision_datasets.append(ds)
                    logger.info(f"  Loaded {task}: {len(ds)} samples")
            except Exception as e:
                logger.warning(f"  Failed to load {task} from {args.data_path}: {e}")

    if not vision_datasets:
        logger.warning("No vision datasets loaded. Training requires data.")
        logger.info("Provide --data-path pointing to a dataset directory.")
        return

    # Create mixed dataset
    mixed_ds = MixedVisionDataset(
        vision_datasets=vision_datasets,
        vision_ratio=cfg.data.vision_ratio,
        seed=seed,
    )

    # Train
    logger.info("Starting instruction-tuning...")
    tuner.train(
        train_dataset=mixed_ds,
        epochs=cfg.training.epochs,
        batch_size=cfg.data.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.optimizer.max_grad_norm,
        scheduler_type=cfg.optimizer.scheduler,
        warmup_steps=cfg.optimizer.warmup_steps,
        output_dir=output_dir,
        save_every=cfg.training.save_every,
        eval_every=cfg.training.eval_every,
        use_amp=cfg.training.use_amp,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
