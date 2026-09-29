#!/usr/bin/env python3
"""Unified training script for Vision Banana.

Supports all registered models (SANA, FLUX, SD3, etc.) and
multi-task training (depth, normal, segmentation, etc.).

This replaces model-specific scripts (train_sana_512.py, train_flux.py)
with a single entry point that delegates to the modular training framework.

Usage:
    # SANA training
    python scripts/train.py --config configs/sana_1600m_3task_full_clean.yaml --wandb

    # FLUX training
    python scripts/train.py --config configs/flux_3task_full_clean.yaml --wandb

    # Override any config via CLI
    python scripts/train.py --config configs/sana_1600m_3task_full_clean.yaml \
        --steps 10000 --lr 5e-5 --output-dir output/custom_run
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    from open_vision_banana.training.config import (
        apply_cli_overrides,
        build_arg_parser,
        load_config,
    )
    from open_vision_banana.training.trainer import Trainer

    # Parse args and load config
    parser = build_arg_parser()
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Load config from YAML, then apply CLI overrides
    config = load_config(config_path=args.config)
    config = apply_cli_overrides(config, args)

    # Setup file logging (rank 0 only under torchrun, so ranks don't interleave train.log;
    # same rule as DistContext.is_main, which does not exist yet at this point)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(os.environ.get("RANK", "0")) == 0:
        fh = logging.FileHandler(output_dir / "train.log")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)

    # Create and run trainer
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
