#!/usr/bin/env python3
"""Training script for Open-Vision-Banana.

Implements the paper's generator instruction-tuning pattern with explicit,
project-controlled data mixing and LoRA hyperparameters.

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --data-path /path/to/data --epochs 5
    python scripts/train.py --generation-path /path/to/gen --wandb
    python scripts/train.py --val-split 0.1 --early-stopping-patience 5
    python scripts/train.py --help
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from open_vision_banana import VisionBananaConfig
from open_vision_banana.data import MixedVisionDataset, build_dataset, build_generation_dataset
from open_vision_banana.training.tuner import VisionTaskDataset
from open_vision_banana.data.split import split_vision_datasets
from open_vision_banana.training.logging import create_training_logger_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_TASKS = [
    "preencoded_depth", "depth", "normal", "semantic", "segmentation",
    "instance", "referring", "edge", "flow", "keypoint", "object_detection",
    "depth_edge", "structure", "multiview", "specular_removal",
]
TASK_ALIASES = {
    "depth_estimation": "depth",
    "normal_estimation": "normal",
    "semantic_segmentation": "semantic",
    "instance_segmentation": "instance",
    "referring_segmentation": "referring",
    "edge_detection": "edge",
    "keypoint_detection": "keypoint",
}


def _configured_tasks(value) -> list[str]:
    if value is None:
        return list(DEFAULT_TASKS)
    names = list(value) if isinstance(value, (list, dict)) else []
    if not names:
        raise ValueError("data.tasks must be a non-empty list or mapping")
    tasks = []
    for name in names:
        task = TASK_ALIASES.get(str(name), str(name))
        if task not in DEFAULT_TASKS:
            raise ValueError(f"Unsupported task in data.tasks: {name}")
        if task not in tasks:
            tasks.append(task)
    return tasks


def _resolve_task_root(data_root: Path, task: str, task_dirs: dict[str, str]) -> Path:
    configured = task_dirs.get(task)
    if configured is None:
        for alias, canonical in TASK_ALIASES.items():
            if canonical == task and alias in task_dirs:
                configured = task_dirs[alias]
                break
    if configured is not None:
        path = Path(configured)
        return path if path.is_absolute() else data_root / path
    candidate = data_root / task
    return candidate if candidate.exists() else data_root


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
    parser.add_argument("--generation-path", type=str, default=None,
                        help="Path to generation preservation dataset (9:1 mix)")
    parser.add_argument("--vision-ratio", type=float, default=None,
                        help="Vision sampling ratio (default 0.1 = 10%% vision)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    parser.add_argument("--val-split", type=float, default=None,
                        help="Validation split ratio on vision data (e.g. 0.1)")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Stop after N evals without val loss improvement")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--wandb-log-checkpoints", action="store_true",
                        help="Upload LoRA checkpoints to W&B as artifacts")
    parser.add_argument("--fsdp", action="store_true",
                        help="Enable FSDP multi-GPU training (use with torchrun)")
    parser.add_argument("--ddp", action="store_true",
                        help="Enable DDP multi-GPU training (use with torchrun)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing on transformer blocks")
    parser.add_argument("--deepspeed", action="store_true",
                        help="Enable DeepSpeed ZeRO-3 (use with deepspeed launcher)")
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="configs/deepspeed_zero3.json",
        help="Path to DeepSpeed JSON config",
    )
    # v7 / paper-aligned training options (InstructionTuner)
    parser.add_argument(
        "--colormaps", type=str, default=None,
        help="Comma-separated depth colormaps (e.g. cube_edge,viridis,plasma)",
    )
    parser.add_argument("--flow-sigma", type=float, default=None,
                        help="Optional flow-matching noise sigma (0 = pure flow)")
    parser.add_argument("--no-min-snr", action="store_true",
                        help="Disable Min-SNR loss weighting")
    parser.add_argument("--no-logit-normal-timesteps", action="store_true",
                        help="Use uniform timesteps instead of logit-normal")
    parser.add_argument("--no-img2img-flow", action="store_true",
                        help="Disable img2img-style flow matching (input→target)")
    parser.add_argument("--cfg-dropout-rate", type=float, default=None,
                        help="CFG dropout rate for text conditioning (default 0.1)")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA decay override (default from config)")
    parser.add_argument("--scheduler", type=str, default=None,
                        help="LR scheduler: cosine, cosine_restarts, polynomial, constant, one_cycle")
    parser.add_argument("--warmup-ratio", type=float, default=None,
                        help="Warmup ratio for SchedulerFactory (overrides warmup_steps when set)")
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
        cfg = VisionBananaConfig.from_yaml(config_path, strict=True)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        cfg = VisionBananaConfig()

    # Apply CLI overrides
    if args.model_name:
        cfg.model.name = args.model_name
        cfg.model.local_path = None
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
    if args.generation_path:
        cfg.data.generation_path = args.generation_path
    if args.vision_ratio is not None:
        cfg.data.vision_ratio = args.vision_ratio
        cfg.data.generation_ratio = 1.0 - args.vision_ratio
    if args.val_split is not None:
        cfg.data.val_split = args.val_split
    if args.early_stopping_patience is not None:
        cfg.training.early_stopping_patience = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        cfg.training.early_stopping_min_delta = args.early_stopping_min_delta
    if args.log_every is not None:
        cfg.training.log_every = args.log_every
    if args.wandb:
        cfg.logging.wandb_enabled = True
    if args.wandb_project:
        cfg.logging.wandb_project = args.wandb_project
    if args.wandb_run_name:
        cfg.logging.wandb_run_name = args.wandb_run_name
    if args.tensorboard:
        cfg.logging.tensorboard_enabled = True
    if args.tensorboard_dir:
        cfg.logging.tensorboard_dir = args.tensorboard_dir
    if args.wandb_log_checkpoints:
        cfg.logging.wandb_log_checkpoints = True
        cfg.logging.wandb_enabled = True

    if args.ema_decay is not None:
        cfg.training.ema_decay = args.ema_decay
    if args.no_min_snr:
        cfg.training.use_min_snr = False
    if args.no_logit_normal_timesteps:
        cfg.training.use_logit_normal_timesteps = False
    if args.no_img2img_flow:
        cfg.training.use_img2img_flow = False
    if args.flow_sigma is not None:
        cfg.training.flow_sigma = args.flow_sigma
    if args.cfg_dropout_rate is not None:
        cfg.training.cfg_dropout_rate = args.cfg_dropout_rate
    if args.warmup_ratio is not None:
        cfg.training.warmup_ratio = args.warmup_ratio
    if args.gradient_checkpointing:
        cfg.training.gradient_checkpointing = True
    if args.scheduler:
        cfg.optimizer.scheduler = args.scheduler
    if args.colormaps:
        cfg.data.colormaps = [
            name.strip() for name in args.colormaps.split(",") if name.strip()
        ]

    output_dir = args.output_dir or cfg.output_dir
    seed = args.seed if args.seed is not None else cfg.training.seed
    cfg.output_dir = output_dir
    cfg.training.seed = seed
    configured_data_path = args.data_path or cfg.data.root
    if configured_data_path is None:
        raise ValueError("Training requires --data-path or data.root in the config")
    data_path = Path(configured_data_path).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"Training data root does not exist: {data_path}")
    cfg.data.root = str(data_path.resolve())
    if cfg.data.tasks is None:
        cfg.data.tasks = list(DEFAULT_TASKS)
    cfg.validate()

    device = args.device or _auto_device()

    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Print config summary
    logger.info("=" * 60)
    logger.info("Vision Banana Training")
    logger.info("=" * 60)
    effective_model_name = cfg.model.local_path or cfg.model.name
    logger.info(f"Base model: {effective_model_name}")
    logger.info(f"LoRA rank: {cfg.lora.rank}, alpha: {cfg.lora.alpha} (rsLoRA)")
    logger.info(f"Learning rate: {cfg.optimizer.lr}")
    logger.info(f"Epochs: {cfg.training.epochs}, Batch size: {cfg.data.batch_size}")
    logger.info(f"Device: {device}, dtype: {cfg.model.dtype}")
    logger.info(f"Data mix: {cfg.data.vision_ratio:.0%} vision data")
    if cfg.data.generation_path:
        logger.info(f"Generation data: {cfg.data.generation_path}")
    if cfg.data.val_split > 0:
        logger.info(f"Validation split: {cfg.data.val_split:.0%}")
    if cfg.training.early_stopping_patience:
        logger.info(f"Early stopping patience: {cfg.training.early_stopping_patience}")
    logger.info(f"Output: {output_dir}")

    # Device-specific adjustments
    use_amp = cfg.training.use_amp
    num_workers = cfg.data.num_workers
    if device == "cpu":
        if use_amp:
            logger.warning(
                "AMP not supported on cpu — disabling use_amp automatically."
            )
            use_amp = False
    elif device == "mps" and use_amp:
        logger.info(
            "MPS: enabling bfloat16 autocast (GradScaler disabled — CUDA only)."
        )
    if device in ("mps", "cpu") and num_workers > 0:
        logger.warning(
            f"num_workers={num_workers} may cause hangs on {device} — forcing 0."
        )
        num_workers = 0
    cfg.training.use_amp = use_amp
    cfg.data.num_workers = num_workers

    # Load base FLUX.1 model
    logger.info("Loading base FLUX.1 model…")
    from open_vision_banana.inference.generator import BaseGenerator
    from open_vision_banana.training.tuner import InstructionTuner
    from open_vision_banana.training.utils import VisionAugmentation, enable_gradient_checkpointing

    if sum([args.fsdp, args.ddp, args.deepspeed]) > 1:
        logger.error("Specify only one of --fsdp, --ddp, or --deepspeed")
        return

    train_sampler = None
    is_main_process = True
    fsdp_wrapper = None
    ddp_wrapper = None

    if args.fsdp:
        from open_vision_banana.training.fsdp_utils import FSDPWrapper
        fsdp_wrapper = FSDPWrapper()
        fsdp_wrapper.initialize()
        is_main_process = fsdp_wrapper.is_main()
        device = f"cuda:{fsdp_wrapper.local_rank}" if fsdp_wrapper.enabled else device
        if is_main_process:
            logger.info("FSDP enabled (world_size=%d)", fsdp_wrapper.world_size)
    elif args.ddp:
        from torch.utils.data import DistributedSampler

        from open_vision_banana.training.utils import DDPWrapper
        ddp_wrapper = DDPWrapper()
        ddp_wrapper.initialize()
        is_main_process = ddp_wrapper.is_main()
        device = f"cuda:{ddp_wrapper.local_rank}" if ddp_wrapper.world_size > 1 else device
        if is_main_process:
            logger.info("DDP enabled (world_size=%d)", ddp_wrapper.world_size)

    if isinstance(cfg.data.tasks, dict):
        logger.warning(
            "Legacy mapping-style data.tasks detected: mapping keys select tasks, but nested "
            "dataset metadata is recorded only. Use data.task_dirs for loader paths."
        )
    if is_main_process:
        run_dir = Path(output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(run_dir / "training_config.yaml")
        from open_vision_banana.reproducibility import build_experiment_manifest

        manifest = build_experiment_manifest(
            cfg,
            repo_root=Path(__file__).resolve().parents[1],
            fingerprint_mode="metadata",
            require_data=True,
        )
        manifest["launch"] = {
            "device": device,
            "ddp": args.ddp,
            "fsdp": args.fsdp,
            "deepspeed": args.deepspeed,
            "deepspeed_config": args.deepspeed_config if args.deepspeed else None,
        }
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    generator = BaseGenerator(
        model_name=effective_model_name,
        lora_path=cfg.model.lora_path,
        lora_revision=cfg.model.lora_revision,
        device=device,
        dtype=get_dtype(cfg.model.dtype),
    )

    augmentation = None
    if cfg.data.augmentation:
        augmentation = VisionAugmentation(
            resolution=cfg.data.resolution,
            hflip_prob=0.5,
            crop_scale=(0.8, 1.0),
            color_jitter={"brightness": 0.1, "contrast": 0.1},
            task="depth",
        )

    logger.info("Setting up LoRA instruction alignment…")
    tuner = InstructionTuner(
        model=generator,
        lora_rank=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        use_rslora=cfg.lora.rs_lora,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        optimizer_name=cfg.optimizer.name,
        adam_betas=cfg.optimizer.betas,
        adam_eps=cfg.optimizer.eps,
        augmentation=augmentation,
        use_ema=cfg.training.use_ema,
        ema_decay=cfg.training.ema_decay,
        ema_warmup_steps=cfg.training.ema_warmup_steps,
        use_gradient_checkpointing=cfg.training.gradient_checkpointing,
        use_min_snr=cfg.training.use_min_snr,
        min_snr_gamma=cfg.training.min_snr_gamma,
        use_logit_normal_timesteps=cfg.training.use_logit_normal_timesteps,
        use_img2img_flow=cfg.training.use_img2img_flow,
        flow_sigma=cfg.training.flow_sigma,
        cfg_dropout_rate=cfg.training.cfg_dropout_rate,
        warmup_ratio=cfg.training.warmup_ratio,
    )

    if args.fsdp or args.deepspeed:
        enable_gradient_checkpointing(tuner.peft_model)

    if args.deepspeed:
        tuner.enable_deepspeed(
            config_path=args.deepspeed_config,
            micro_batch_size=cfg.data.batch_size,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            max_grad_norm=cfg.optimizer.max_grad_norm,
        )
        from open_vision_banana.training.deepspeed_utils import deepspeed_is_main
        is_main_process = deepspeed_is_main(tuner.deepspeed_engine)
        if is_main_process:
            logger.info(
                "DeepSpeed enabled (world_size=%d)",
                getattr(tuner.deepspeed_engine, "world_size", 1),
            )
    elif fsdp_wrapper is not None and fsdp_wrapper.enabled:
        tuner.peft_model = fsdp_wrapper.wrap(tuner.peft_model)
    elif ddp_wrapper is not None and ddp_wrapper.world_size > 1:
        tuner.peft_model = ddp_wrapper.wrap(tuner.peft_model, device=device)

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        from open_vision_banana.training.checkpoint import CheckpointManager
        resume_path = Path(args.resume)
        if resume_path.is_dir() and resume_path.name.startswith("checkpoint-"):
            resume_checkpoint = resume_path
        elif resume_path.is_dir():
            resume_checkpoint = CheckpointManager(str(resume_path))._latest_checkpoint()
            if resume_checkpoint is None:
                raise FileNotFoundError(f"No numbered checkpoint found in {resume_path}")
        else:
            resume_checkpoint = resume_path.parent
        meta_path = resume_checkpoint / "metadata.json" if resume_checkpoint.is_dir() else resume_checkpoint.parent / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
        meta = json.loads(meta_path.read_text())
        resume_step = meta.get("step", 0)
        resume_epoch = meta.get("epoch", 0)
        logger.info(f"Resumed from step={resume_step}, epoch={resume_epoch}")
    else:
        resume_step = 0
        resume_epoch = 0
        resume_checkpoint = None

    # Build datasets
    vision_datasets = []

    if data_path is not None:
        for task in _configured_tasks(cfg.data.tasks):
            task_root = _resolve_task_root(data_path, task, cfg.data.task_dirs)
            loaded = False
            # Try specialized build_dataset first
            try:
                ds = build_dataset(task, str(task_root), resolution=cfg.data.resolution)
                if len(ds) > 0:
                    if hasattr(ds, "depth_mapper"):
                        from open_vision_banana.core.depth_mapper import DepthRGBMapper

                        ds.depth_mapper = DepthRGBMapper(
                            lam=cfg.depth_mapping.lam, c=cfg.depth_mapping.c
                        )
                    vision_datasets.append(ds)
                    logger.info(f"  Loaded {task} (specialized): {len(ds)} samples")
                    loaded = True
            except Exception as exc:
                logger.debug("Specialized loader failed for %s: %s", task, exc)

            if loaded:
                continue

            # Fallback: generic images/targets layout via VisionTaskDataset
            task_dir = task_root
            if (task_dir / "images").exists():
                try:
                    ds = VisionTaskDataset(
                        data_path=str(task_dir),
                        resolution=cfg.data.resolution,
                        task=task,
                    )
                    if ds.samples:
                        vision_datasets.append(ds)
                        logger.info(f"  Loaded {task} (generic): {len(ds)} samples")
                except Exception as e:
                    logger.warning(f"  Failed to load {task} from {task_root}: {e}")

    if not vision_datasets:
        logger.warning("No vision datasets loaded. Training requires data.")
        logger.info("Provide --data-path pointing to a dataset directory.")
        return

    val_dataset = None
    train_vision_datasets = vision_datasets
    if cfg.data.val_split > 0:
        train_vision_datasets, val_dataset = split_vision_datasets(
            vision_datasets, cfg.data.val_split, seed=seed, source_level=True,
        )
        if val_dataset is not None:
            logger.info(f"  Validation set: {len(val_dataset)} samples")

    generation_dataset = None
    generation_path = cfg.data.generation_path
    if generation_path is not None and data_path is not None:
        configured_generation_path = Path(generation_path)
        if not configured_generation_path.is_absolute() and not configured_generation_path.exists():
            generation_path = str(data_path / configured_generation_path)
    if generation_path is None and data_path is not None:
        implicit_generation_path = data_path / "generation"
        if (implicit_generation_path / "images").exists():
            generation_path = str(implicit_generation_path)
            logger.info("Using generation data discovered at %s", implicit_generation_path)
    if generation_path:
        generation_dataset = build_generation_dataset(
            generation_path,
            resolution=cfg.data.resolution,
        )
        if len(generation_dataset) == 0:
            logger.warning("Generation path provided but no samples loaded — skipping mix")
            generation_dataset = None
        else:
            logger.info(f"  Generation data: {len(generation_dataset)} samples")

    vision_ratio = cfg.data.vision_ratio
    if generation_dataset is None and vision_ratio < 1.0:
        logger.info(
            "No generation dataset — forcing vision_ratio=1.0 (100%% vision sampling)"
        )
        vision_ratio = 1.0

    mixed_ds = MixedVisionDataset(
        vision_datasets=train_vision_datasets,
        generation_dataset=generation_dataset,
        vision_ratio=vision_ratio,
        seed=seed,
        colormaps=cfg.data.colormaps,
        task_sampling=cfg.data.task_sampling,
        task_weights={
            TASK_ALIASES.get(task, task): weight
            for task, weight in cfg.data.task_weights.items()
        },
        depth_lam=cfg.depth_mapping.lam,
        depth_c=cfg.depth_mapping.c,
    )
    logger.info(f"  Mixed dataset: {mixed_ds}")

    if fsdp_wrapper is not None:
        train_sampler = fsdp_wrapper.get_train_sampler(mixed_ds)
    elif ddp_wrapper is not None and ddp_wrapper.world_size > 1:
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(mixed_ds, shuffle=True)
    elif args.deepspeed:
        from open_vision_banana.training.deepspeed_utils import get_train_sampler
        train_sampler = get_train_sampler(mixed_ds, tuner.deepspeed_engine)

    training_logger = create_training_logger_from_config(cfg)
    training_logger.log_hyperparams({
        "model": effective_model_name,
        "lora_rank": cfg.lora.rank,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "lr": cfg.optimizer.lr,
        "weight_decay": cfg.optimizer.weight_decay,
        "batch_size": cfg.data.batch_size,
        "vision_ratio": vision_ratio,
        "task_sampling": cfg.data.task_sampling,
        "epochs": cfg.training.epochs,
    })

    scheduler_type = args.scheduler or cfg.optimizer.scheduler

    logger.info("Starting instruction-tuning...")
    summary = tuner.train(
        train_dataset=mixed_ds,
        val_dataset=val_dataset,
        epochs=cfg.training.epochs,
        batch_size=cfg.data.batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.optimizer.max_grad_norm,
        scheduler_type=scheduler_type,
        warmup_steps=cfg.optimizer.warmup_steps,
        output_dir=output_dir,
        save_every=cfg.training.save_every,
        eval_every=cfg.training.eval_every,
        max_keep=cfg.training.max_keep,
        use_amp=use_amp,
        num_workers=num_workers,
        training_logger=training_logger,
        log_every=cfg.training.log_every,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        train_sampler=train_sampler,
        is_main_process=is_main_process,
        start_step=resume_step,
        start_epoch=resume_epoch,
        resume_checkpoint=resume_checkpoint,
        restore_rng_on_resume=resume_checkpoint is not None,
    )

    if fsdp_wrapper is not None:
        fsdp_wrapper.cleanup()
    elif ddp_wrapper is not None:
        ddp_wrapper.cleanup()

    logger.info("Training complete!")
    logger.info(
        "Summary: epochs=%s, steps=%s, best_val_loss=%s, stopped_early=%s",
        summary.get("epochs_run"),
        summary.get("global_step"),
        summary.get("best_val_loss"),
        summary.get("stopped_early"),
    )


if __name__ == "__main__":
    main()
