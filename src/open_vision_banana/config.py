"""Configuration system for Vision Banana.

Hierarchical configuration with YAML serialization support.
All sub-configs have sensible defaults matching the paper.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import yaml


def _filtered_from_dict(cls, d: dict):
    """Create a config instance from dict, ignoring unknown keys."""
    valid = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in d.items() if k in valid and k != "self"}
    return cls(**filtered)


class DepthMappingConfig:
    """Depth → RGB mapping parameters."""

    def __init__(self, lam: float = -3.0, c: float = 10.0 / 3.0):
        self.lam = lam
        self.c = c

    def to_dict(self) -> dict:
        return {"lam": self.lam, "c": self.c}

    @classmethod
    def from_dict(cls, d: dict) -> DepthMappingConfig:
        return _filtered_from_dict(cls, d)


class NormalMappingConfig:
    """Normal → RGB mapping parameters (identity — no tunable params)."""

    def to_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, d: dict) -> NormalMappingConfig:
        return cls()


class SegmentationConfig:
    """Instance segmentation decoding parameters."""

    def __init__(
        self,
        color_tolerance: float = 14.0,
        min_area_fraction: float = 2e-4,
        erosion_threshold: float = 0.1,
        bbox_expansion_factor: float = 5.0,
    ):
        self.color_tolerance = color_tolerance
        self.min_area_fraction = min_area_fraction
        self.erosion_threshold = erosion_threshold
        self.bbox_expansion_factor = bbox_expansion_factor

    def to_dict(self) -> dict:
        return {
            "color_tolerance": self.color_tolerance,
            "min_area_fraction": self.min_area_fraction,
            "erosion_threshold": self.erosion_threshold,
            "bbox_expansion_factor": self.bbox_expansion_factor,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SegmentationConfig:
        return _filtered_from_dict(cls, d)


class ModelConfig:
    """Base model configuration."""

    def __init__(
        self,
        name: str = "black-forest-labs/FLUX.1-schnell",
        dtype: str = "bfloat16",
        lora_path: str | None = None,
        lora_revision: str | None = None,
    ):
        self.name = name
        self.dtype = dtype
        self.lora_path = lora_path
        self.lora_revision = lora_revision

    def to_dict(self) -> dict:
        d = {"name": self.name, "dtype": self.dtype}
        if self.lora_path is not None:
            d["lora_path"] = self.lora_path
        if self.lora_revision is not None:
            d["lora_revision"] = self.lora_revision
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return _filtered_from_dict(cls, d)


class LoRAConfig:
    """LoRA instruction tuning parameters."""

    def __init__(
        self,
        rank: int = 64,
        alpha: int = 128,
        rs_lora: bool = True,
        target_modules: list[str] | None = None,
        dropout: float = 0.0,
    ):
        self.rank = rank
        self.alpha = alpha
        self.rs_lora = rs_lora
        self.target_modules = target_modules or [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2",
        ]
        self.dropout = dropout

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "rs_lora": self.rs_lora,
            "target_modules": self.target_modules,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LoRAConfig:
        return _filtered_from_dict(cls, d)


class OptimizerConfig:
    """Optimizer and learning rate schedule configuration."""

    def __init__(
        self,
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        scheduler: str = "cosine",
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

    def to_dict(self) -> dict:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
        }

    @classmethod
    def from_dict(cls, d: dict) -> OptimizerConfig:
        return _filtered_from_dict(cls, d)


class DataConfig:
    """Data loading and mixing configuration."""

    def __init__(
        self,
        vision_ratio: float = 0.1,
        generation_ratio: float = 0.9,
        resolution: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
        augmentation: bool = True,
        generation_path: str | None = None,
        val_split: float = 0.0,
    ):
        self.vision_ratio = vision_ratio
        self.generation_ratio = generation_ratio
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.generation_path = generation_path
        self.val_split = val_split

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "vision_ratio": self.vision_ratio,
            "generation_ratio": self.generation_ratio,
            "resolution": self.resolution,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "augmentation": self.augmentation,
            "val_split": self.val_split,
        }
        if self.generation_path is not None:
            d["generation_path"] = self.generation_path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DataConfig:
        return _filtered_from_dict(cls, d)


class TrainingConfig:
    """Training loop configuration."""

    def __init__(
        self,
        epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_warmup_steps: int = 2000,
        save_every: int = 1000,
        eval_every: int = 500,
        max_keep: int = 5,
        seed: int = 42,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        log_every: int = 10,
        flow_sigma: float = 0.0,
    ):
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_warmup_steps = ema_warmup_steps
        self.save_every = save_every
        self.eval_every = eval_every
        self.max_keep = max_keep
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.log_every = log_every
        self.flow_sigma = flow_sigma

    def to_dict(self) -> dict:
        d = {
            "epochs": self.epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "use_amp": self.use_amp,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "ema_warmup_steps": self.ema_warmup_steps,
            "save_every": self.save_every,
            "eval_every": self.eval_every,
            "max_keep": self.max_keep,
            "seed": self.seed,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "log_every": self.log_every,
            "flow_sigma": self.flow_sigma,
        }
        if self.early_stopping_patience is not None:
            d["early_stopping_patience"] = self.early_stopping_patience
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainingConfig:
        return _filtered_from_dict(cls, d)


class LoggingConfig:
    """Training observability configuration (W&B + TensorBoard)."""

    def __init__(
        self,
        wandb_enabled: bool = False,
        wandb_project: str = "open-vision-banana",
        wandb_run_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_log_checkpoints: bool = False,
        tensorboard_enabled: bool = False,
        tensorboard_dir: str = "output/runs",
        run_name: str | None = None,
    ):
        self.wandb_enabled = wandb_enabled
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_entity = wandb_entity
        self.wandb_log_checkpoints = wandb_log_checkpoints
        self.tensorboard_enabled = tensorboard_enabled
        self.tensorboard_dir = tensorboard_dir
        self.run_name = run_name

    def to_dict(self) -> dict:
        d = {
            "wandb_enabled": self.wandb_enabled,
            "wandb_project": self.wandb_project,
            "wandb_log_checkpoints": self.wandb_log_checkpoints,
            "tensorboard_enabled": self.tensorboard_enabled,
            "tensorboard_dir": self.tensorboard_dir,
        }
        if self.wandb_run_name is not None:
            d["wandb_run_name"] = self.wandb_run_name
        if self.wandb_entity is not None:
            d["wandb_entity"] = self.wandb_entity
        if self.run_name is not None:
            d["run_name"] = self.run_name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LoggingConfig:
        return _filtered_from_dict(cls, d)


class VisionBananaConfig:
    """Top-level configuration for Vision Banana.

    Aggregates all sub-configurations with sensible defaults
    matching the paper's hyperparameters.
    """

    def __init__(
        self,
        model: ModelConfig | None = None,
        lora: LoRAConfig | None = None,
        optimizer: OptimizerConfig | None = None,
        data: DataConfig | None = None,
        training: TrainingConfig | None = None,
        logging: LoggingConfig | None = None,
        depth_mapping: DepthMappingConfig | None = None,
        normal_mapping: NormalMappingConfig | None = None,
        segmentation: SegmentationConfig | None = None,
        output_dir: str = "output/checkpoints",
    ):
        self.model = model or ModelConfig()
        self.lora = lora or LoRAConfig()
        self.optimizer = optimizer or OptimizerConfig()
        self.data = data or DataConfig()
        self.training = training or TrainingConfig()
        self.logging = logging or LoggingConfig()
        self.depth_mapping = depth_mapping or DepthMappingConfig()
        self.normal_mapping = normal_mapping or NormalMappingConfig()
        self.segmentation = segmentation or SegmentationConfig()
        self.output_dir = output_dir

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "lora": self.lora.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "logging": self.logging.to_dict(),
            "depth_mapping": self.depth_mapping.to_dict(),
            "normal_mapping": self.normal_mapping.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VisionBananaConfig:
        return cls(
            model=ModelConfig.from_dict(d.get("model", {})),
            lora=LoRAConfig.from_dict(d.get("lora", {})),
            optimizer=OptimizerConfig.from_dict(d.get("optimizer", {})),
            data=DataConfig.from_dict(d.get("data", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
            logging=LoggingConfig.from_dict(d.get("logging", {})),
            depth_mapping=DepthMappingConfig.from_dict(d.get("depth_mapping", {})),
            normal_mapping=NormalMappingConfig.from_dict(d.get("normal_mapping", {})),
            segmentation=SegmentationConfig.from_dict(d.get("segmentation", {})),
            output_dir=d.get("output_dir", "output/checkpoints"),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VisionBananaConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d or {})


__all__ = [
    "VisionBananaConfig",
    "ModelConfig",
    "LoRAConfig",
    "OptimizerConfig",
    "DataConfig",
    "TrainingConfig",
    "LoggingConfig",
    "DepthMappingConfig",
    "NormalMappingConfig",
    "SegmentationConfig",
]
