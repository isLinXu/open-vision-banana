"""Configuration system for Vision Banana.

The defaults are reproducible project baselines. They are not presented as
hyperparameters disclosed by the Vision Banana paper unless explicitly noted
in the paper-alignment documentation.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import yaml

KNOWN_DEPTH_COLORMAPS = {
    "cube_edge", "legacy_cube_edge", "plasma", "inferno", "viridis", "grayscale", "rainbow",
}


def _filtered_from_dict(cls, d: dict, *, strict: bool = False, path: str = "config"):
    """Create a config instance from a mapping, optionally rejecting unknown keys."""
    if not isinstance(d, dict):
        raise TypeError(f"{path} must be a mapping, got {type(d).__name__}")
    valid = inspect.signature(cls.__init__).parameters
    unknown = sorted(set(d) - (set(valid) - {"self"}))
    if strict and unknown:
        names = ", ".join(f"{path}.{key}" for key in unknown)
        raise ValueError(f"Unknown configuration field(s): {names}")
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
    def from_dict(cls, d: dict, *, strict: bool = False) -> DepthMappingConfig:
        return _filtered_from_dict(cls, d, strict=strict, path="depth_mapping")


class NormalMappingConfig:
    """Normal → RGB mapping parameters (identity — no tunable params)."""

    def to_dict(self) -> dict:
        return {}

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = False) -> NormalMappingConfig:
        if strict and d:
            names = ", ".join(f"normal_mapping.{key}" for key in sorted(d))
            raise ValueError(f"Unknown configuration field(s): {names}")
        return cls()


class SegmentationConfig:
    """Instance segmentation decoding parameters.

    Defaults follow the paper-aligned protocol (Appendix A): seed grouping
    with a fixed color tolerance τ=14. Adjacent grouping and adaptive
    tolerance remain available as explicit opt-ins.
    """

    def __init__(
        self,
        color_tolerance: float = 14.0,
        min_area_fraction: float = 2e-4,
        erosion_threshold: float = 0.1,
        bbox_expansion_factor: float = 5.0,
        grouping_mode: str = "seed",
        adaptive_color_tolerance: bool = False,
    ):
        self.color_tolerance = color_tolerance
        self.min_area_fraction = min_area_fraction
        self.erosion_threshold = erosion_threshold
        self.bbox_expansion_factor = bbox_expansion_factor
        self.grouping_mode = grouping_mode
        self.adaptive_color_tolerance = adaptive_color_tolerance

    def to_dict(self) -> dict:
        return {
            "color_tolerance": self.color_tolerance,
            "min_area_fraction": self.min_area_fraction,
            "erosion_threshold": self.erosion_threshold,
            "bbox_expansion_factor": self.bbox_expansion_factor,
            "grouping_mode": self.grouping_mode,
            "adaptive_color_tolerance": self.adaptive_color_tolerance,
        }

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = False) -> SegmentationConfig:
        return _filtered_from_dict(cls, d, strict=strict, path="segmentation")


class ModelConfig:
    """Base model configuration."""

    def __init__(
        self,
        name: str = "black-forest-labs/FLUX.1-schnell",
        dtype: str = "bfloat16",
        local_path: str | None = None,
        lora_path: str | None = None,
        lora_revision: str | None = None,
    ):
        self.name = name
        self.dtype = dtype
        self.local_path = local_path
        self.lora_path = lora_path
        self.lora_revision = lora_revision

    def to_dict(self) -> dict:
        d = {"name": self.name, "dtype": self.dtype}
        if self.local_path is not None:
            d["local_path"] = self.local_path
        if self.lora_path is not None:
            d["lora_path"] = self.lora_path
        if self.lora_revision is not None:
            d["lora_revision"] = self.lora_revision
        return d

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = False) -> ModelConfig:
        return _filtered_from_dict(cls, d, strict=strict, path="model")


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
        self.target_modules = target_modules
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
    def from_dict(cls, d: dict, *, strict: bool = False) -> LoRAConfig:
        return _filtered_from_dict(cls, d, strict=strict, path="lora")


class OptimizerConfig:
    """Optimizer and learning rate schedule configuration."""

    def __init__(
        self,
        name: str = "adamw",
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        betas: list[float] | tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        scheduler: str = "cosine",
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
    ):
        self.name = name
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = tuple(betas)
        self.eps = eps
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "betas": list(self.betas),
            "eps": self.eps,
            "scheduler": self.scheduler,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
        }

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = False) -> OptimizerConfig:
        return _filtered_from_dict(cls, d, strict=strict, path="optimizer")


class DataConfig:
    """Data loading and mixing configuration."""

    def __init__(
        self,
        root: str | None = None,
        tasks: list[str] | dict[str, Any] | None = None,
        task_dirs: dict[str, str] | None = None,
        task_sampling: str = "uniform",
        task_weights: dict[str, float] | None = None,
        vision_ratio: float = 0.1,
        generation_ratio: float = 0.9,
        resolution: int = 512,
        batch_size: int = 8,
        num_workers: int = 4,
        augmentation: bool = True,
        generation_path: str | None = None,
        val_split: float = 0.0,
        colormaps: list[str] | None = None,
    ):
        self.root = root
        self.tasks = tasks
        self.task_dirs = task_dirs or {}
        self.task_sampling = task_sampling
        self.task_weights = task_weights or {}
        self.vision_ratio = vision_ratio
        self.generation_ratio = generation_ratio
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.generation_path = generation_path
        self.val_split = val_split
        self.colormaps = ["cube_edge"] if colormaps is None else colormaps

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_sampling": self.task_sampling,
            "vision_ratio": self.vision_ratio,
            "generation_ratio": self.generation_ratio,
            "resolution": self.resolution,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "augmentation": self.augmentation,
            "val_split": self.val_split,
            "colormaps": self.colormaps,
        }
        if self.root is not None:
            d["root"] = self.root
        if self.tasks is not None:
            d["tasks"] = self.tasks
        if self.task_dirs:
            d["task_dirs"] = self.task_dirs
        if self.task_weights:
            d["task_weights"] = self.task_weights
        if self.generation_path is not None:
            d["generation_path"] = self.generation_path
        return d

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = False) -> DataConfig:
        return _filtered_from_dict(cls, d, strict=strict, path="data")


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
        gradient_checkpointing: bool = True,
        use_logit_normal_timesteps: bool = True,
        use_min_snr: bool = True,
        min_snr_gamma: float = 5.0,
        cfg_dropout_rate: float = 0.1,
        use_img2img_flow: bool = True,
        flow_sigma: float = 0.0,
        warmup_ratio: float | None = None,
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
        self.gradient_checkpointing = gradient_checkpointing
        self.use_logit_normal_timesteps = use_logit_normal_timesteps
        self.use_min_snr = use_min_snr
        self.min_snr_gamma = min_snr_gamma
        self.cfg_dropout_rate = cfg_dropout_rate
        self.use_img2img_flow = use_img2img_flow
        self.flow_sigma = flow_sigma
        self.warmup_ratio = warmup_ratio

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
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_logit_normal_timesteps": self.use_logit_normal_timesteps,
            "use_min_snr": self.use_min_snr,
            "min_snr_gamma": self.min_snr_gamma,
            "cfg_dropout_rate": self.cfg_dropout_rate,
            "use_img2img_flow": self.use_img2img_flow,
            "flow_sigma": self.flow_sigma,
        }
        if self.warmup_ratio is not None:
            d["warmup_ratio"] = self.warmup_ratio
        if self.early_stopping_patience is not None:
            d["early_stopping_patience"] = self.early_stopping_patience
        return d

    @classmethod
    def from_dict(cls, d: dict, *, strict: bool = False) -> TrainingConfig:
        return _filtered_from_dict(cls, d, strict=strict, path="training")


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
    def from_dict(cls, d: dict, *, strict: bool = False) -> LoggingConfig:
        return _filtered_from_dict(cls, d, strict=strict, path="logging")


class VisionBananaConfig:
    """Top-level configuration for Vision Banana.

    Aggregates all sub-configurations with conservative project defaults.
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
        schema_version: int = 1,
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
        self.schema_version = schema_version

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
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
    def from_dict(cls, d: dict, *, strict: bool = False) -> VisionBananaConfig:
        if not isinstance(d, dict):
            raise TypeError(f"config must be a mapping, got {type(d).__name__}")
        valid_top_level = {
            "schema_version", "model", "lora", "optimizer", "data", "training",
            "logging", "depth_mapping", "normal_mapping", "segmentation", "output_dir",
        }
        unknown = sorted(set(d) - valid_top_level)
        if strict and unknown:
            raise ValueError(
                "Unknown configuration field(s): "
                + ", ".join(f"config.{key}" for key in unknown)
            )
        config = cls(
            model=ModelConfig.from_dict(d.get("model", {}), strict=strict),
            lora=LoRAConfig.from_dict(d.get("lora", {}), strict=strict),
            optimizer=OptimizerConfig.from_dict(d.get("optimizer", {}), strict=strict),
            data=DataConfig.from_dict(d.get("data", {}), strict=strict),
            training=TrainingConfig.from_dict(d.get("training", {}), strict=strict),
            logging=LoggingConfig.from_dict(d.get("logging", {}), strict=strict),
            depth_mapping=DepthMappingConfig.from_dict(
                d.get("depth_mapping", {}), strict=strict
            ),
            normal_mapping=NormalMappingConfig.from_dict(
                d.get("normal_mapping", {}), strict=strict
            ),
            segmentation=SegmentationConfig.from_dict(
                d.get("segmentation", {}), strict=strict
            ),
            output_dir=d.get("output_dir", "output/checkpoints"),
            schema_version=d.get("schema_version", 1),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate cross-field constraints before model or data loading."""
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append(f"schema_version must be 1, got {self.schema_version}")
        if self.model.dtype not in {"float32", "float16", "bfloat16"}:
            errors.append(f"model.dtype is unsupported: {self.model.dtype}")
        if self.lora.rank <= 0 or self.lora.alpha <= 0:
            errors.append("lora.rank and lora.alpha must be positive")
        if not 0.0 <= self.lora.dropout < 1.0:
            errors.append("lora.dropout must be in [0, 1)")
        if self.lora.target_modules is not None and not self.lora.target_modules:
            errors.append("lora.target_modules must be null or a non-empty list")
        if self.optimizer.name not in {"adamw", "adamw_8bit"}:
            errors.append("optimizer.name must be 'adamw' or 'adamw_8bit'")
        if self.optimizer.lr <= 0 or self.optimizer.weight_decay < 0:
            errors.append("optimizer.lr must be positive and weight_decay non-negative")
        if len(self.optimizer.betas) != 2 or any(not 0.0 <= x < 1.0 for x in self.optimizer.betas):
            errors.append("optimizer.betas must contain two values in [0, 1)")
        if self.optimizer.eps <= 0:
            errors.append("optimizer.eps must be positive")
        if not 0.0 < self.data.vision_ratio <= 1.0:
            errors.append("data.vision_ratio must be in (0, 1]")
        if not 0.0 <= self.data.generation_ratio < 1.0:
            errors.append("data.generation_ratio must be in [0, 1)")
        if abs(self.data.vision_ratio + self.data.generation_ratio - 1.0) > 1e-6:
            errors.append("data.vision_ratio + data.generation_ratio must equal 1")
        if self.data.task_sampling not in {"uniform", "dataset_uniform", "proportional"}:
            errors.append(
                "data.task_sampling must be 'uniform', 'dataset_uniform', or 'proportional'"
            )
        if self.data.resolution <= 0 or self.data.batch_size <= 0 or self.data.num_workers < 0:
            errors.append("data resolution/batch_size must be positive; num_workers non-negative")
        if not 0.0 <= self.data.val_split < 1.0:
            errors.append("data.val_split must be in [0, 1)")
        if not self.data.colormaps or any(not str(name).strip() for name in self.data.colormaps):
            errors.append("data.colormaps must contain at least one non-empty name")
        unknown_colormaps = sorted(set(self.data.colormaps) - KNOWN_DEPTH_COLORMAPS)
        if unknown_colormaps:
            errors.append("unknown data.colormaps: " + ", ".join(unknown_colormaps))
        if any(weight <= 0 for weight in self.data.task_weights.values()):
            errors.append("all data.task_weights values must be positive")
        if self.training.epochs <= 0 or self.training.gradient_accumulation_steps <= 0:
            errors.append("training.epochs and gradient_accumulation_steps must be positive")
        if any(
            value <= 0
            for value in (
                self.training.save_every,
                self.training.eval_every,
                self.training.max_keep,
                self.training.log_every,
            )
        ):
            errors.append("training save/eval/max_keep/log intervals must be positive")
        if not 0.0 < self.training.ema_decay < 1.0 or self.training.ema_warmup_steps < 0:
            errors.append("training.ema_decay must be in (0, 1) and ema_warmup_steps non-negative")
        if (
            self.training.early_stopping_patience is not None
            and self.training.early_stopping_patience <= 0
        ):
            errors.append("training.early_stopping_patience must be positive or null")
        if not 0.0 <= self.training.cfg_dropout_rate < 1.0:
            errors.append("training.cfg_dropout_rate must be in [0, 1)")
        if self.training.min_snr_gamma <= 0:
            errors.append("training.min_snr_gamma must be positive")
        if self.training.flow_sigma < 0:
            errors.append("training.flow_sigma must be non-negative")
        if self.training.warmup_ratio is not None and not 0.0 <= self.training.warmup_ratio < 1.0:
            errors.append("training.warmup_ratio must be in [0, 1)")
        if self.depth_mapping.lam >= -1.0 or self.depth_mapping.c <= 0.0:
            errors.append("depth_mapping.lam must be < -1 and depth_mapping.c must be positive")
        if self.segmentation.color_tolerance <= 0:
            errors.append("segmentation.color_tolerance must be positive")
        if self.segmentation.min_area_fraction < 0:
            errors.append("segmentation.min_area_fraction must be non-negative")
        if not 0.0 <= self.segmentation.erosion_threshold <= 1.0:
            errors.append("segmentation.erosion_threshold must be in [0, 1]")
        if self.segmentation.bbox_expansion_factor <= 0:
            errors.append("segmentation.bbox_expansion_factor must be positive")
        if (
            not isinstance(self.segmentation.grouping_mode, str)
            or self.segmentation.grouping_mode not in {"adjacent", "seed"}
        ):
            errors.append("segmentation.grouping_mode must be 'adjacent' or 'seed'")
        if not isinstance(self.segmentation.adaptive_color_tolerance, bool):
            errors.append("segmentation.adaptive_color_tolerance must be boolean")
        if errors:
            raise ValueError("Invalid Vision Banana configuration:\n- " + "\n- ".join(errors))

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path, *, strict: bool = False) -> VisionBananaConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d or {}, strict=strict)


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
