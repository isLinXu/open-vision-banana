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
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import get_type_hints

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
    val_max_samples_per_task: int = 40
    task_sampling: str = "proportional"
    task_sampling_weights: dict[str, float] = field(default_factory=dict)

    # ── Training ───────────────────────────────────────────────────────
    steps: int = 5000
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 3e-5
    # AdamW settings; the paper does not state its optimizer, these are repo choices
    # recorded into effective_config.json for reproducibility.
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    drop_grad_norm: float = 0.0
    seed: int = 42
    gradient_checkpointing: bool = True
    depth_curve: str = "metric_power"
    depth_log_alpha: float = 0.4
    depth_colormaps: list[str] | None = None
    semantic_prompt_styles: list[str] | None = None
    semantic_color_remap_prob: float = 0.0
    aspect_preserving_crop: bool = False

    # ── Scheduler ──────────────────────────────────────────────────────
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    warmup_steps: int | None = None

    # ── Flow matching ──────────────────────────────────────────────────
    img2img_flow: bool = True
    flow_sigma: float = 0.0
    logit_normal: bool = True
    flow_shift: float = 1.0  # SANA flow shift
    # Paper (Section 2): "without specialized architectures or custom training losses".
    # Defaults therefore mean "plain flow-matching loss, equal task weights". Older
    # configs that relied on the previous defaults (min_snr on, depth 2.5 / normal 2.0 /
    # semantic 0.5) have those values pinned explicitly in their YAML (2026-10-22).
    min_snr: bool = False
    min_snr_gamma: float = 5.0
    cfg_dropout_rate: float = 0.1

    # ── Loss ───────────────────────────────────────────────────────────
    task_loss_weights: dict[str, float] = field(default_factory=dict)  # empty = 1.0 for every task
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
    ema_enabled: bool = True

    # ── W&B ────────────────────────────────────────────────────────────
    wandb: bool = False
    wandb_project: str = "open-vision-banana"
    wandb_run_name: str | None = None

    # ── Device ─────────────────────────────────────────────────────────
    device: str | None = None  # auto-detect if None
    resume: str | None = None  # checkpoint dir or its lora.pt
    resume_weights_only: bool = False  # LoRA only; fresh optimizer, schedule, step
    # Manifest preflight (data/validation.validate_manifest_contract) runs before the
    # backbone is loaded and refuses to train on contract errors (relative-depth
    # prompts on metric targets, legends longer than the text encoder, FFN-only LoRA).
    # Skipping is possible but recorded in effective_config.json.
    skip_preflight: bool = False

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
        """Return normalized loss weights, independent of sampling weights."""
        # Empty mapping = the paper setting (every task weighted 1.0; losses.py falls back to 1.0).
        return _parse_weights(self.task_loss_weights, "task_loss_weights", allow_empty=True)

    def validate(self) -> None:
        """Normalize and reject invalid effective values before loading a model."""
        hints = get_type_hints(type(self))
        for f in fields(self):
            value = getattr(self, f.name)
            hint = hints[f.name]
            if hint is bool and not isinstance(value, bool):
                raise ValueError(f"{f.name} must be a boolean")
            if (hint is int or (hint == int | None and value is not None)) and type(value) is not int:
                raise ValueError(f"{f.name} must be an integer")
            if hint is float:
                if isinstance(value, bool):
                    raise ValueError(f"{f.name} must be a finite number")
                try:
                    value = float(value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{f.name} must be a finite number") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{f.name} must be a finite number")
                setattr(self, f.name, value)
            if hint in (str, str | None):
                if value is None and hint == str | None:
                    continue
                if isinstance(value, Path):
                    value = str(value)
                    setattr(self, f.name, value)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{f.name} must be a non-empty string")

        for name in ("tasks", "lora_targets", "depth_colormaps", "semantic_prompt_styles"):
            value = getattr(self, name)
            if name in {"lora_targets", "depth_colormaps", "semantic_prompt_styles"} and value is None:
                continue
            if not isinstance(value, list) or not value or any(
                not isinstance(v, str) or not v.strip() for v in value
            ) or len(set(value)) != len(value):
                raise ValueError(f"{name} must be a non-empty list of unique strings")
        if self.depth_colormaps is not None:
            from ..core.depth_mapper import get_available_colormaps

            allowed_cmaps = set(get_available_colormaps())
            unknown_cmaps = sorted(set(self.depth_colormaps) - allowed_cmaps)
            if unknown_cmaps:
                raise ValueError(f"depth_colormaps contains unknown colormaps: {unknown_cmaps}")
        if self.semantic_prompt_styles is not None:
            from ..data.openvocab_semantic import STYLES

            unknown_styles = sorted(set(self.semantic_prompt_styles) - set(STYLES))
            if unknown_styles:
                raise ValueError(
                    f"semantic_prompt_styles contains unknown styles: {unknown_styles}; "
                    f"choose from {list(STYLES)}"
                )
        if not 0.0 <= self.semantic_color_remap_prob < 1.0:
            raise ValueError("semantic_color_remap_prob must be in [0, 1)")
        self.task_loss_weights = self.parse_task_weights()
        self.task_sampling_weights = _parse_weights(
            self.task_sampling_weights, "task_sampling_weights", allow_empty=True
        )
        for name in ("resolution", "lora_rank", "lora_alpha", "steps", "batch_size", "grad_accum",
                     "save_every", "eval_every", "log_every", "val_max_samples_per_task"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("num_workers", "ema_warmup", "seed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("warmup_steps", "max_samples_per_task"):
            value = getattr(self, name)
            if value is not None and value < (1 if name == "max_samples_per_task" else 0):
                raise ValueError(f"{name} is out of range")
        for name in ("lr", "max_grad_norm", "flow_shift", "min_snr_gamma", "depth_edge_aux_sigma_floor"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.drop_grad_norm < 0:
            raise ValueError("drop_grad_norm must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not (0.0 <= self.adam_beta1 < 1.0 and 0.0 <= self.adam_beta2 < 1.0):
            raise ValueError("adam_beta1/adam_beta2 must be in [0, 1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.drop_grad_norm > 0 and self.drop_grad_norm <= self.max_grad_norm:
            raise ValueError("drop_grad_norm must exceed max_grad_norm when enabled")
        if self.depth_curve not in {"metric_power", "log_relative"}:
            raise ValueError("depth_curve must be metric_power or log_relative")
        if self.depth_log_alpha <= 0:
            raise ValueError("depth_log_alpha must be positive")
        for name in ("flow_sigma", "depth_edge_aux_weight", "depth_edge_aux_gradient_weight",
                     "depth_edge_aux_edge_weight", "depth_edge_aux_edge_threshold",
                     "normal_angular_loss_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("warmup_ratio", "cfg_dropout_rate"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 < self.ema_decay < 1:
            raise ValueError("ema_decay must be in (0, 1)")
        if not 0.0 < self.vision_ratio <= 1.0:
            raise ValueError("vision_ratio must be in (0, 1]")
        if self.vision_ratio != 1.0 and "generation" not in self.tasks:
            raise ValueError(
                "vision_ratio must be 1.0 unless 'generation' is in tasks: "
                "generation mixing is not implemented without a generation task"
            )
        if self.tar_index is not None:
            raise ValueError("tar_index is not implemented by the unified Trainer; use pair_manifest")
        if self.task_sampling not in {"proportional", "uniform", "sqrt"}:
            raise ValueError("task_sampling must be proportional, uniform, or sqrt")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16")
        if self.quantize not in {None, "4bit", "8bit"}:
            raise ValueError("quantize must be null, 4bit, or 8bit")
        if self.scheduler not in {"cosine", "cosine_restarts", "polynomial", "constant", "one_cycle"}:
            raise ValueError(
                "scheduler must be cosine, cosine_restarts, polynomial, constant, or one_cycle"
            )

    def to_dict(self) -> dict:
        """Return the validated, JSON-compatible effective configuration."""
        self.validate()
        return asdict(self)


def _parse_weights(value, name: str, *, allow_empty: bool = False) -> dict[str, float]:
    if isinstance(value, str):
        parsed = {}
        for pair in value.split(","):
            if pair.count("=") != 1:
                raise ValueError(f"{name}: expected task=weight pairs")
            key, weight = (part.strip() for part in pair.split("="))
            if key in parsed:
                raise ValueError(f"{name}: duplicate task {key}")
            parsed[key] = weight
        value = parsed
    if not isinstance(value, dict) or (not value and not allow_empty):
        raise ValueError(f"{name} must be a non-empty weight mapping")
    result = {}
    for key, weight in value.items():
        if not isinstance(key, str) or not key.strip() or isinstance(weight, bool):
            raise ValueError(f"{name}: invalid task or weight")
        try:
            weight = float(weight)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name}: invalid weight for {key}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{name}: weights must be finite and positive")
        result[key] = weight
    return result


# Only these aliases are accepted; unsupported historical schemas fail explicitly.
_YAML_SECTIONS = {
    "model": {"name": "model_name", "dtype": "dtype", "resolution": "resolution", "quantize": "quantize"},
    "lora": {"rank": "lora_rank", "alpha": "lora_alpha", "rs_lora": "rslora", "target_modules": "lora_targets"},
    "data": {"dir": "data_dir", "pair_manifest": "pair_manifest", "tar_index": "tar_index",
             "tasks": "tasks", "num_workers": "num_workers", "vision_ratio": "vision_ratio",
             "max_samples_per_task": "max_samples_per_task", "resolution": "resolution",
             "task_sampling": "task_sampling", "task_sampling_weights": "task_sampling_weights",
             "task_weights": "task_sampling_weights",
             "depth_curve": "depth_curve", "depth_log_alpha": "depth_log_alpha",
             "depth_colormaps": "depth_colormaps",
             "semantic_prompt_styles": "semantic_prompt_styles",
             "semantic_color_remap_prob": "semantic_color_remap_prob",
             "aspect_preserving_crop": "aspect_preserving_crop"},
    "training": {"steps": "steps", "batch_size": "batch_size", "gradient_accumulation": "grad_accum",
                 "lr": "lr", "max_grad_norm": "max_grad_norm", "drop_grad_norm": "drop_grad_norm",
                 "seed": "seed",
                 "gradient_checkpointing": "gradient_checkpointing", "skip_preflight": "skip_preflight"},
    "optimizer": {"weight_decay": "weight_decay", "beta1": "adam_beta1", "beta2": "adam_beta2",
                  "eps": "adam_eps"},
    "scheduler": {"type": "scheduler", "warmup_ratio": "warmup_ratio", "warmup_steps": "warmup_steps"},
    "validation": {"max_samples_per_task": "val_max_samples_per_task"},
    "flow": {"img2img_flow": "img2img_flow", "sigma": "flow_sigma",
             "logit_normal_timesteps": "logit_normal", "shift": "flow_shift", "min_snr": "min_snr",
             "min_snr_gamma": "min_snr_gamma", "cfg_dropout_rate": "cfg_dropout_rate"},
    "loss": {name: name for name in ("task_loss_weights", "depth_edge_aux_weight",
             "depth_edge_aux_gradient_weight", "depth_edge_aux_edge_weight", "depth_edge_aux_edge_threshold",
             "depth_edge_aux_sigma_floor", "normal_angular_loss_weight")},
    "checkpoint": {name: name for name in ("output_dir", "save_every", "eval_every", "log_every",
                   "resume", "resume_weights_only")},
    "ema": {"enabled": "ema_enabled", "decay": "ema_decay", "warmup_steps": "ema_warmup"},
    "wandb": {"enabled": "wandb", "project": "wandb_project", "run_name": "wandb_run_name"},
}


def _load_yaml_config(config_path: str | Path | None) -> dict:
    if config_path is None:
        return {}
    import yaml

    with Path(config_path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    top_fields = {"device", "resume", "resume_weights_only"}
    unknown = set(raw) - set(_YAML_SECTIONS) - top_fields - {"experiment"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown, key=str)}")
    result = {}
    for section, aliases in _YAML_SECTIONS.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"{section} must be a mapping")
        # model_id is accepted only as a redundant registry assertion, never ignored.
        if section == "model" and "model_id" in values:
            from ..model_registry import resolve_model_config

            expected = resolve_model_config(values.get("name", "sana-1600m")).model_id
            if values["model_id"] != expected:
                raise ValueError("model.model_id overrides are not implemented; use model.name")
        # Zero dropout is a redundant historical declaration; other values are unsupported.
        if section == "lora" and "dropout" in values and values["dropout"] != 0.0:
            raise ValueError("lora.dropout is not implemented")
        extras = {"model_id"} if section == "model" else {"dropout"} if section == "lora" else set()
        unknown = set(values) - set(aliases) - extras
        if unknown:
            raise ValueError("Unknown configuration fields: " + ", ".join(
                f"{section}.{key}" for key in sorted(unknown, key=str)
            ))
        for key, value in values.items():
            if key in extras:
                continue
            target = aliases[key]
            if target in {"resume", "resume_weights_only"} and (
                value is None or (target == "resume" and isinstance(value, str) and not value.strip())
            ):
                continue
            if section == "data" and key == "task_weights" and isinstance(value, str):
                target = "task_sampling"
            if target in result and result[target] != value:
                raise ValueError(f"Configuration conflict for {section}.{key} ({target})")
            result[target] = value
    for name in top_fields:
        if name not in raw:
            continue
        value = raw[name]
        if name in {"resume", "resume_weights_only"}:
            if value is None or (name == "resume" and isinstance(value, str) and not value.strip()):
                continue
            if name in result and (
                result[name] != value or type(result[name]) is not type(value)
            ):
                raise ValueError(f"Configuration conflict between {name} and checkpoint.{name}")
        result[name] = value
    return result


def load_config(args: argparse.Namespace | None = None, config_path: str | Path | None = None) -> TrainingConfig:
    """Merge defaults < YAML < explicit CLI, then validate the effective values."""
    if args is not None and getattr(args, "config", None) is not None:
        config_path = args.config
    config = TrainingConfig(**_load_yaml_config(config_path))
    if args is not None:
        return apply_cli_overrides(config, args)
    config.validate()
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    """Build a parser without reading process argv or injecting YAML defaults."""
    p = argparse.ArgumentParser(description="Unified Vision Banana training script")
    p.add_argument("--config", type=Path)
    p.add_argument("--print-effective-config", action="store_true")
    hints = get_type_hints(TrainingConfig)
    for f in fields(TrainingConfig):
        name = f.name
        option = "--" + name.replace("_", "-")
        if hints[name] is bool:
            if name == "ema_enabled":
                option = "--ema"
            p.add_argument(option, dest=name, action=argparse.BooleanOptionalAction, default=None)
        elif name in {"tasks", "lora_targets", "depth_colormaps", "semantic_prompt_styles"}:
            p.add_argument(option, nargs="+", default=None)
        elif name in {"task_loss_weights", "task_sampling_weights"}:
            p.add_argument(option, default=None, help="task=weight,task=weight")
        else:
            arg_type = int if hints[name] in (int, int | None) else float if hints[name] is float else str
            p.add_argument(option, type=arg_type, default=None)
    p.add_argument("--ddp", action="store_true", help="accepted for launcher compatibility")
    p.add_argument("--fsdp", "--deepspeed", nargs=0, action=_UnsupportedBackend)
    return p


class _UnsupportedBackend(argparse.Action):
    """Reject unsupported distributed backends explicitly."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            f"{option_string} is not supported by the unified Trainer; LoRA training is "
            "data-parallel only. Launch with torchrun (e.g. "
            "scripts/launch_training_a100_80g_ddp.sh)."
        )


def apply_cli_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Apply only explicit values; shared by both loading interfaces."""
    valid = {f.name for f in fields(config)}
    ignored = {"config", "print_effective_config", "ddp", "fsdp", "deepspeed"}
    for name, value in vars(args).items():
        if value is None or name in ignored:
            continue
        if name.startswith("no_") and name[3:] in valid:
            if value:
                setattr(config, name[3:], False)
        elif name in valid:
            setattr(config, name, value)
        else:
            raise ValueError(f"Unknown CLI configuration field: {name}")
    config.validate()
    return config
