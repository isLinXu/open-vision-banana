"""Unified training configuration for multi-model, multi-task training.

Merges the configuration loading from both train_sana_512.py and train_flux.py
into a single dataclass-based system with YAML + CLI override support.

Usage:
    args = build_arg_parser().parse_args()
    config = load_config(args)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Unified training configuration."""

    # ── Model ──────────────────────────────────────────────────────────
    model_name: str = "sana-1600m"
    dtype: str = "bfloat16"
    resolution: int = 512
    quantize: str | None = None  # "4bit", "8bit", None

    # ── LoRA ───────────────────────────────────────────────────────────
    lora_rank: int = 32
    lora_alpha: int = 64
    rslora: bool = True
    lora_targets: list[str] | None = None

    # ── Data ───────────────────────────────────────────────────────────
    data_dir: str | None = None
    pair_manifest: str | None = None
    tar_index: str | None = None
    tasks: list[str] = field(default_factory=lambda: ["depth", "normal", "semantic"])
    num_workers: int = 0
    vision_ratio: float = 1.0
    max_samples_per_task: int | None = None

    # ── Training ───────────────────────────────────────────────────────
    steps: int = 5000
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 3e-5
    max_grad_norm: float = 1.0
    seed: int = 42
    gradient_checkpointing: bool = True

    # ── Scheduler ──────────────────────────────────────────────────────
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    warmup_steps: int | None = None

    # ── Flow matching ──────────────────────────────────────────────────
    img2img_flow: bool = True
    flow_sigma: float = 0.0
    logit_normal: bool = True
    flow_shift: float = 1.0  # SANA flow shift
    min_snr: bool = True
    min_snr_gamma: float = 5.0
    cfg_dropout_rate: float = 0.1

    # ── Loss ───────────────────────────────────────────────────────────
    task_loss_weights: dict[str, float] = field(
        default_factory=lambda: {"depth": 2.5, "normal": 2.0, "semantic": 0.5}
    )
    depth_edge_aux_weight: float = 0.0
    depth_edge_aux_gradient_weight: float = 0.25
    depth_edge_aux_edge_weight: float = 2.0
    depth_edge_aux_edge_threshold: float = 0.025
    depth_edge_aux_sigma_floor: float = 0.10
    normal_angular_loss_weight: float = 0.0

    # ── Checkpoint / Logging ───────────────────────────────────────────
    output_dir: str = "output/training"
    save_every: int = 500
    eval_every: int = 250
    log_every: int = 25

    # ── EMA ────────────────────────────────────────────────────────────
    ema_decay: float = 0.9999
    ema_warmup: int = 500

    # ── W&B ────────────────────────────────────────────────────────────
    wandb: bool = False
    wandb_project: str = "open-vision-banana"
    wandb_run_name: str | None = None

    # ── Device ─────────────────────────────────────────────────────────
    device: str | None = None  # auto-detect if None
    resume: str | None = None

    # ── Task weights string (for YAML parsing) ─────────────────────────
    _task_weights_str: str | None = None

    def get_dtype(self) -> torch.dtype:
        """Convert dtype string to torch.dtype."""
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
            self.dtype
        ]

    def get_device(self) -> torch.device:
        """Get torch device, auto-detecting if needed."""
        if self.device:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def parse_task_weights(self) -> dict[str, float]:
        """Parse task weights from string format 'depth=1.0,normal=2.0' or return dict."""
        if isinstance(self.task_loss_weights, dict) and self.task_loss_weights:
            return self.task_loss_weights
        if self._task_weights_str:
            result = {}
            for pair in self._task_weights_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    result[k.strip()] = float(v.strip())
            return result
        return {"depth": 1.0, "normal": 1.0, "semantic": 1.0}


def _load_yaml_config(config_path: Path | None) -> dict:
    """Load YAML config and flatten to a single dict."""
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
        "model_name": model.get("name", "sana-1600m"),
        "dtype": model.get("dtype", "bfloat16"),
        "resolution": data.get("resolution", model.get("resolution", 512)),
        "quantize": model.get("quantize"),
        "lora_rank": lora.get("rank", 32),
        "lora_alpha": lora.get("alpha", 64),
        "rslora": lora.get("rs_lora", True),
        "lora_targets": lora.get("target_modules"),
        "data_dir": data.get("dir"),
        "pair_manifest": data.get("pair_manifest"),
        "tar_index": data.get("tar_index"),
        "num_workers": data.get("num_workers", 0),
        "tasks": data.get("tasks"),
        "vision_ratio": data.get("vision_ratio", 1.0),
        "steps": training.get("steps", 5000),
        "batch_size": training.get("batch_size", 1),
        "grad_accum": training.get("gradient_accumulation", 4),
        "lr": training.get("lr", 3e-5),
        "max_grad_norm": training.get("max_grad_norm", 1.0),
        "seed": training.get("seed", 42),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "scheduler": scheduler.get("type", "cosine"),
        "warmup_ratio": scheduler.get("warmup_ratio", 0.05),
        "warmup_steps": scheduler.get("warmup_steps"),
        "img2img_flow": flow.get("img2img_flow", True),
        "flow_sigma": flow.get("sigma", 0.0),
        "logit_normal": flow.get("logit_normal_timesteps", True),
        "flow_shift": flow.get("shift", 1.0),
        "min_snr": flow.get("min_snr", True),
        "min_snr_gamma": flow.get("min_snr_gamma", 5.0),
        "cfg_dropout_rate": flow.get("cfg_dropout_rate", 0.1),
        "depth_edge_aux_weight": loss.get("depth_edge_aux_weight", 0.0),
        "depth_edge_aux_gradient_weight": loss.get("depth_edge_aux_gradient_weight", 0.25),
        "depth_edge_aux_edge_weight": loss.get("depth_edge_aux_edge_weight", 2.0),
        "depth_edge_aux_edge_threshold": loss.get("depth_edge_aux_edge_threshold", 0.025),
        "depth_edge_aux_sigma_floor": loss.get("depth_edge_aux_sigma_floor", 0.10),
        "normal_angular_loss_weight": loss.get("normal_angular_loss_weight", 0.0),
        "output_dir": ckpt.get("output_dir", "output/training"),
        "save_every": ckpt.get("save_every", 500),
        "eval_every": ckpt.get("eval_every", 250),
        "log_every": ckpt.get("log_every", 25),
        "ema_decay": ema.get("decay", 0.9999),
        "ema_warmup": ema.get("warmup_steps", 500),
        "wandb": wandb_section.get("enabled", False),
        "wandb_project": wandb_section.get("project", "open-vision-banana"),
        "wandb_run_name": wandb_section.get("run_name"),
        "device": raw.get("device"),
        "resume": raw.get("resume"),
    }

    # Handle task_loss_weights (can be string or dict)
    task_weights = loss.get("task_loss_weights")
    if isinstance(task_weights, str):
        d["_task_weights_str"] = task_weights
    elif isinstance(task_weights, dict):
        d["task_loss_weights"] = task_weights

    return {k: v for k, v in d.items() if v is not None}


def load_config(
    args: argparse.Namespace | None = None, config_path: Path | None = None
) -> TrainingConfig:
    """Load config from YAML file with optional CLI overrides."""
    # Determine config path
    if args is not None and hasattr(args, "config") and args.config is not None:
        config_path = args.config

    # Load YAML defaults
    yaml_dict = _load_yaml_config(config_path)

    # Apply CLI overrides
    if args is not None:
        args_dict = vars(args)
        for key, value in args_dict.items():
            if key == "config":
                continue
            if value is not None:
                # Convert CLI arg names to config names
                config_key = key.replace("-", "_")
                yaml_dict[config_key] = value

    # Build TrainingConfig
    config = TrainingConfig()
    for f in fields(config):
        if f.name.startswith("_"):
            continue
        if f.name in yaml_dict:
            setattr(config, f.name, yaml_dict[f.name])

    # Handle private fields
    if "_task_weights_str" in yaml_dict:
        config._task_weights_str = yaml_dict["_task_weights_str"]
        config.task_loss_weights = config.parse_task_weights()

    return config


def build_arg_parser() -> argparse.ArgumentParser:
    """Build unified argparse parser from TrainingConfig."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()

    cfg = _load_yaml_config(config_args.config)

    p = argparse.ArgumentParser(
        description="Unified Vision Banana training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=config_args.config)

    # Model
    p.add_argument("--model-name", default=cfg.get("model_name", "sana-1600m"))
    p.add_argument("--dtype", default=cfg.get("dtype", "bfloat16"))
    p.add_argument("--resolution", type=int, default=cfg.get("resolution", 512))
    p.add_argument("--quantize", default=cfg.get("quantize"), choices=[None, "4bit", "8bit"])

    # LoRA
    p.add_argument("--lora-rank", type=int, default=cfg.get("lora_rank", 32))
    p.add_argument("--lora-alpha", type=int, default=cfg.get("lora_alpha", 64))
    p.add_argument("--no-rslora", action="store_true", default=not cfg.get("rslora", True))

    # Data
    p.add_argument("--data-dir", default=cfg.get("data_dir"))
    p.add_argument("--pair-manifest", type=Path, default=cfg.get("pair_manifest"))
    p.add_argument("--tar-index", type=Path, default=cfg.get("tar_index"))
    p.add_argument("--tasks", nargs="+", default=cfg.get("tasks"))
    p.add_argument("--num-workers", type=int, default=cfg.get("num_workers", 0))
    p.add_argument("--vision-ratio", type=float, default=cfg.get("vision_ratio", 1.0))
    p.add_argument("--max-samples-per-task", type=int, default=None)

    # Training
    p.add_argument("--steps", type=int, default=cfg.get("steps", 5000))
    p.add_argument("--batch-size", type=int, default=cfg.get("batch_size", 1))
    p.add_argument("--grad-accum", type=int, default=cfg.get("grad_accum", 4))
    p.add_argument("--lr", type=float, default=cfg.get("lr", 3e-5))
    p.add_argument("--max-grad-norm", type=float, default=cfg.get("max_grad_norm", 1.0))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 42))

    # Scheduler
    p.add_argument("--scheduler", default=cfg.get("scheduler", "cosine"))
    p.add_argument("--warmup-ratio", type=float, default=cfg.get("warmup_ratio", 0.05))

    # Flow
    p.add_argument(
        "--no-img2img-flow", action="store_true", default=not cfg.get("img2img_flow", True)
    )
    p.add_argument("--flow-sigma", type=float, default=cfg.get("flow_sigma", 0.0))
    p.add_argument(
        "--no-logit-normal", action="store_true", default=not cfg.get("logit_normal", True)
    )
    p.add_argument("--flow-shift", type=float, default=cfg.get("flow_shift", 1.0))
    p.add_argument("--no-min-snr", action="store_true", default=not cfg.get("min_snr", True))
    p.add_argument("--min-snr-gamma", type=float, default=cfg.get("min_snr_gamma", 5.0))
    p.add_argument("--cfg-dropout-rate", type=float, default=cfg.get("cfg_dropout_rate", 0.1))

    # Loss
    p.add_argument(
        "--depth-edge-aux-weight", type=float, default=cfg.get("depth_edge_aux_weight", 0.0)
    )
    p.add_argument(
        "--depth-edge-aux-gradient-weight",
        type=float,
        default=cfg.get("depth_edge_aux_gradient_weight", 0.25),
    )
    p.add_argument(
        "--depth-edge-aux-edge-weight",
        type=float,
        default=cfg.get("depth_edge_aux_edge_weight", 2.0),
    )
    p.add_argument(
        "--depth-edge-aux-edge-threshold",
        type=float,
        default=cfg.get("depth_edge_aux_edge_threshold", 0.025),
    )
    p.add_argument(
        "--depth-edge-aux-sigma-floor",
        type=float,
        default=cfg.get("depth_edge_aux_sigma_floor", 0.10),
    )
    p.add_argument(
        "--normal-angular-loss-weight",
        type=float,
        default=cfg.get("normal_angular_loss_weight", 0.0),
    )

    # Output
    p.add_argument("--output-dir", default=cfg.get("output_dir", "output/training"))
    p.add_argument("--save-every", type=int, default=cfg.get("save_every", 500))
    p.add_argument("--eval-every", type=int, default=cfg.get("eval_every", 250))
    p.add_argument("--log-every", type=int, default=cfg.get("log_every", 25))

    # EMA
    p.add_argument("--ema-decay", type=float, default=cfg.get("ema_decay", 0.9999))
    p.add_argument("--ema-warmup", type=int, default=cfg.get("ema_warmup", 500))

    # W&B
    p.add_argument("--wandb", action="store_true", default=cfg.get("wandb", False))
    p.add_argument("--wandb-project", default=cfg.get("wandb_project", "open-vision-banana"))
    p.add_argument("--wandb-run-name", default=cfg.get("wandb_run_name"))

    # Device
    p.add_argument("--device", default=cfg.get("device"))
    p.add_argument("--resume", default=cfg.get("resume"))

    return p


def apply_cli_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply CLI argument overrides to config (for non-None values)."""
    # Handle boolean flags that use action="store_true" with negation
    if getattr(args, "no_img2img_flow", False):
        config.img2img_flow = False
    if getattr(args, "no_logit_normal", False):
        config.logit_normal = False
    if getattr(args, "no_min_snr", False):
        config.min_snr = False
    if getattr(args, "no_rslora", False):
        config.rslora = False

    # Apply all other non-None overrides
    override_map = {
        "model_name": "model_name",
        "dtype": "dtype",
        "resolution": "resolution",
        "quantize": "quantize",
        "lora_rank": "lora_rank",
        "lora_alpha": "lora_alpha",
        "data_dir": "data_dir",
        "pair_manifest": "pair_manifest",
        "tar_index": "tar_index",
        "tasks": "tasks",
        "num_workers": "num_workers",
        "vision_ratio": "vision_ratio",
        "max_samples_per_task": "max_samples_per_task",
        "steps": "steps",
        "batch_size": "batch_size",
        "grad_accum": "grad_accum",
        "lr": "lr",
        "max_grad_norm": "max_grad_norm",
        "seed": "seed",
        "scheduler": "scheduler",
        "warmup_ratio": "warmup_ratio",
        "flow_sigma": "flow_sigma",
        "flow_shift": "flow_shift",
        "min_snr_gamma": "min_snr_gamma",
        "cfg_dropout_rate": "cfg_dropout_rate",
        "depth_edge_aux_weight": "depth_edge_aux_weight",
        "depth_edge_aux_gradient_weight": "depth_edge_aux_gradient_weight",
        "depth_edge_aux_edge_weight": "depth_edge_aux_edge_weight",
        "depth_edge_aux_edge_threshold": "depth_edge_aux_edge_threshold",
        "depth_edge_aux_sigma_floor": "depth_edge_aux_sigma_floor",
        "normal_angular_loss_weight": "normal_angular_loss_weight",
        "output_dir": "output_dir",
        "save_every": "save_every",
        "eval_every": "eval_every",
        "log_every": "log_every",
        "ema_decay": "ema_decay",
        "ema_warmup": "ema_warmup",
        "wandb": "wandb",
        "wandb_project": "wandb_project",
        "wandb_run_name": "wandb_run_name",
        "device": "device",
        "resume": "resume",
    }

    for arg_name, config_name in override_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    return config
