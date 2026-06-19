"""DeepSpeed ZeRO-3 utilities for large-model LoRA training.

Usage::

    deepspeed --num_gpus=8 scripts/train.py --deepspeed --data-path ...

Or with torchrun + JSON config override via ``--deepspeed-config``.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_DEEPSPEED_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "deepspeed_zero3.json"
)


def load_deepspeed_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load DeepSpeed JSON config."""
    config_path = Path(path) if path else DEFAULT_DEEPSPEED_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def merge_training_into_config(
    config: dict[str, Any],
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    lr: float,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """Inject runtime training hyperparameters into a DeepSpeed config."""
    merged = deepcopy(config)
    merged["train_micro_batch_size_per_gpu"] = micro_batch_size
    merged["gradient_accumulation_steps"] = gradient_accumulation_steps
    if max_grad_norm is not None:
        merged["gradient_clipping"] = max_grad_norm
    merged.setdefault("optimizer", {})
    if "type" not in merged["optimizer"]:
        merged["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        }
    else:
        merged["optimizer"].setdefault("params", {})["lr"] = lr
    return merged


def initialize_deepspeed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    config_path: str | Path | None = None,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    lr: float = 5e-5,
    max_grad_norm: float = 1.0,
) -> tuple[Any, Any]:
    """Wrap *model* with DeepSpeed ZeRO-3.

    Returns:
        ``(engine, optimizer)`` — *optimizer* may be replaced by DeepSpeed.
    """
    try:
        import deepspeed
    except ImportError as exc:
        raise ImportError(
            "DeepSpeed is required for --deepspeed training. "
            "Install with: pip install deepspeed"
        ) from exc

    base_config = load_deepspeed_config(config_path)
    config = merge_training_into_config(
        base_config,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr=lr,
        max_grad_norm=max_grad_norm,
    )

    logger.info(
        "Initializing DeepSpeed ZeRO stage %s",
        config.get("zero_optimization", {}).get("stage", "?"),
    )

    engine, opt, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config,
    )
    return engine, opt


def is_deepspeed_engine(module: Any) -> bool:
    """Return True when *module* is a DeepSpeed engine."""
    try:
        import deepspeed
    except ImportError:
        return False
    return isinstance(module, deepspeed.DeepSpeedEngine)


def deepspeed_is_main(engine: Any) -> bool:
    """Return True on global rank 0."""
    return getattr(engine, "global_rank", 0) == 0


def deepspeed_world_size(engine: Any) -> int:
    return getattr(engine, "world_size", 1)


def get_train_sampler(dataset, engine: Any):
    """Return DistributedSampler when running multi-process DeepSpeed."""
    world = deepspeed_world_size(engine)
    if world <= 1:
        return None
    from torch.utils.data import DistributedSampler

    return DistributedSampler(
        dataset,
        num_replicas=world,
        rank=getattr(engine, "global_rank", 0),
        shuffle=True,
    )


__all__ = [
    "DEFAULT_DEEPSPEED_CONFIG",
    "deepspeed_is_main",
    "deepspeed_world_size",
    "get_train_sampler",
    "initialize_deepspeed",
    "is_deepspeed_engine",
    "load_deepspeed_config",
    "merge_training_into_config",
]
