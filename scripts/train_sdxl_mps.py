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
  - native epsilon prediction for SSD/SDXL schedulers
  - optional 8-channel source-latent conditioning for perception tasks

Usage:
  python scripts/train_sdxl_mps.py                              # full training, defaults
  python scripts/train_sdxl_mps.py --model sdxl                 # full SDXL (2.6B)
  python scripts/train_sdxl_mps.py --max-steps 50 --dry-run    # smoke test
  python scripts/train_sdxl_mps.py --resume                     # auto-resume
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.conditioning import (  # noqa: E402
    set_scheduler_prediction_type,
    validate_scheduler_prediction_type,
)
from open_vision_banana.data.depth_encoding import (  # noqa: E402
    audit_depth_target_encoding,
    relative_depth_prompt,
)
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.training.tuner import (  # noqa: E402
    SDXL_CONV_LORA_TARGET_MODULES,
    InstructionTuner,
    VisionTaskDataset,
)
from open_vision_banana.training.utils import enable_gradient_checkpointing  # noqa: E402


class TaskScheduleSampler(torch.utils.data.Sampler[int]):
    """Deterministic sampler that guarantees task coverage in each cycle."""

    def __init__(self, lengths: list[int], labels: list[str], num_samples: int, seed: int = 42):
        if len(lengths) != len(labels):
            raise ValueError("task sampler lengths and labels must have the same size")
        self.lengths = lengths
        self.labels = labels
        self.num_samples = num_samples
        self.seed = seed

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed)
        offsets = []
        cursor = 0
        for length in self.lengths:
            offsets.append(cursor)
            cursor += length
        groups: dict[str, list[int]] = {}
        for index, label in enumerate(self.labels):
            groups.setdefault(label, []).extend(
                range(offsets[index], offsets[index] + self.lengths[index])
            )
        task_names = sorted(groups)
        if not task_names:
            return iter(())
        shuffled = {
            task: torch.randperm(len(indices), generator=rng).tolist()
            for task, indices in groups.items()
        }
        cursors = Counter()
        result: list[int] = []
        while len(result) < self.num_samples:
            for task in task_names:
                if len(result) >= self.num_samples:
                    break
                choices = groups[task]
                position = int(cursors[task])
                if position >= len(choices):
                    position = 0
                    shuffled[task] = torch.randperm(len(choices), generator=rng).tolist()
                order = shuffled[task]
                result.append(choices[order[position]])
                cursors[task] = position + 1
        return iter(result)

    def __len__(self):
        return self.num_samples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_sdxl")


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
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


def resolve_data_dirs(data_dir: str | None) -> list[Path]:
    """Resolve one explicit dataset root or preserve the legacy sources.

    Defaults to vb_snvc_train_v2 (v15_seg_expanded) which has:
    - Unified ADE20K 151-class segmentation palette (no multi-source conflict)
    - PNG lossless depth/normal targets (v1 used lossy JPG)
    - Binary depth_edge (v1 used RGB)
    The older vb_snvc_train (v11_fixed) had ADE20K+lllab+COCO segmentation
    mixing, causing supervision conflicts.
    """
    if data_dir is None:
        return [
            PROJECT_ROOT / "data" / "vb_zimage_train",
            PROJECT_ROOT / "data" / "vb_snvc_train_v2",
        ]

    resolved = Path(data_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"Training data directory not found: {resolved}")
    return [resolved]


def filter_dataset_to_manifest_split(
    dataset,
    data_root: Path,
    task_name: str,
    split: str = "train",
) -> int:
    """Apply v3's content-level split to an in-memory paired dataset."""
    allowed_names = manifest_split_names(data_root, task_name, split)
    if not allowed_names:
        return 0
    before = len(dataset.samples)
    dataset.samples = [
        sample
        for sample in dataset.samples
        if Path(sample["image_path"]).name in allowed_names
    ]
    return before - len(dataset.samples)


def manifest_split_names(
    data_root: Path,
    task_name: str,
    split: str = "train",
) -> set[str]:
    """Return filenames declared for one task/split, if a manifest exists."""
    split_path = data_root / "train_val_split.json"
    if not split_path.is_file():
        return set()
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    return set(payload.get(split, {}).get(task_name, []))


def resolve_max_per_task(
    requested: int | None,
    *,
    source_conditioned: bool,
    explicit_data_dir: bool,
) -> int | None:
    """Avoid silently prefix-capping a dedicated source-conditioned dataset."""
    if requested is not None:
        if requested <= 0:
            raise ValueError("max_per_task must be positive")
        return requested
    if explicit_data_dir:
        return None
    return 300


def resolve_source_training_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply task-aware defaults without breaking generation-only training."""
    if args.source_conditioned is None:
        args.source_conditioned = args.task != "generation"
    if args.x0_loss_weight is None:
        args.x0_loss_weight = 1.0 if args.source_conditioned else 0.0
    if args.pixel_loss_weight is None:
        args.pixel_loss_weight = 0.5 if args.source_conditioned else 0.0
    return args


def validation_task_names(task: str) -> tuple[str, ...]:
    """Return validation dataset names for one training task selection."""
    task_map = {
        "depth": ("depth",),
        "normal": ("normal",),
        "segmentation": ("semantic",),
        "edge": ("edge",),
        "generation": ("generation",),
        "all": ("depth", "normal", "semantic", "edge", "generation"),
    }
    try:
        return task_map[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported training task: {task}") from exc


def main():
    parser = argparse.ArgumentParser(description="SDXL/SSD-1B LoRA training on MPS")
    parser.add_argument("--model", type=str, default="ssd-1b",
                        choices=["ssd-1b", "sdxl"],
                        help="Base model (default: ssd-1b, use sdxl for full 2.6B)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps")
    parser.add_argument("--save-every", type=int, default=200, help="Save every N steps")
    parser.add_argument("--eval-every", type=int, default=10000, help="Evaluate held-out task sets every N steps")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Checkpoint directory (default: output/<model>_checkpoints)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=("Use only this dataset root. If omitted, preserve the legacy "
              "vb_zimage_train + vb_snvc_train behavior."),
    )
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument(
        "--lora-conv-targets",
        action="store_true",
        help=(
            "Extend LoRA beyond attention/FF to the UNet ResNet/resampling "
            "convolutions so source-channel structure can reach the output."
        ),
    )
    parser.add_argument("--resolution", type=int, default=1024, help="Training resolution")
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=None,
        help=("Max samples per vision task. Dedicated source-conditioned data "
              "uses all declared train samples by default; legacy multi-task "
              "training defaults to 300."),
    )
    parser.add_argument("--max-val-per-task", type=int, default=64,
                        help="Cap each held-out task for validation (0 disables cap)")
    parser.add_argument("--max-generation", type=int, default=None,
                        help="Max generation samples (default: all declared train samples)")
    parser.add_argument("--balance-data", action="store_true", default=True,
                        help="Balance task sampling (oversample minority tasks)")
    parser.add_argument(
        "--no-balance-data",
        action="store_false",
        dest="balance_data",
        help="Traverse the concatenated task datasets without replacement.",
    )
    parser.add_argument(
        "--task-ratios",
        type=str,
        default=None,
        help='JSON task weights, e.g. \'{"depth":0.2,"normal":0.2,"semantic":0.2,"edge":0.2,"generation":0.2}\'',
    )
    parser.add_argument(
        "--task-schedule",
        action="store_true",
        help="Use a deterministic round-robin task sampler (coverage smoke gate).",
    )
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint dir")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"], help="Training device")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (enabled by default for SSD/SDXL).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load model and exit")
    parser.add_argument(
        "--audit-data-only",
        action="store_true",
        help="Audit depth encoding and split filtering without loading SSD.",
    )
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "depth", "normal", "segmentation", "edge", "generation"],
                        help="Which task to train on")
    parser.add_argument(
        "--source-conditioned",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("Use the validated 8-channel [noisy target, clean source] SSD path. "
              "Defaults on for perception/mixed training and off for generation-only."),
    )
    parser.add_argument(
        "--prediction-type",
        choices=["epsilon"],
        default="epsilon",
        help="Training and inference scheduler output contract (SSD-1B: epsilon).",
    )
    parser.add_argument(
        "--x0-loss-weight",
        type=float,
        default=None,
        help=("Auxiliary implied-x0 loss weight. Defaults to 1.0 for "
              "source-conditioned training and 0 otherwise."),
    )
    parser.add_argument(
        "--pixel-loss-weight",
        type=float,
        default=None,
        help=("Frozen-VAE decoded pixel L1 weight. Defaults to 0.5 for "
              "source-conditioned training and 0 otherwise."),
    )
    parser.add_argument(
        "--structure-loss-weight",
        type=float,
        default=0.0,
        help="Multi-scale decoded-x0 value/gradient loss weight.",
    )
    parser.add_argument(
        "--structure-gradient-weight",
        type=float,
        default=1.0,
        help="Gradient term inside the decoded-x0 structure loss.",
    )
    parser.add_argument(
        "--scalar-geometry-loss-weight",
        type=float,
        default=0.0,
        help="Decoded-x0 scalar depth-position geometry loss weight.",
    )
    parser.add_argument(
        "--scalar-geometry-gradient-weight",
        type=float,
        default=1.0,
        help="Scalar spatial-gradient term inside the geometry loss.",
    )
    parser.add_argument(
        "--scalar-geometry-ordinal-weight",
        type=float,
        default=1.0,
        help="Long-range ordinal/ranking term inside the geometry loss.",
    )
    parser.add_argument(
        "--spatial-adapter",
        action="store_true",
        help="Add a zero-initialized T2I-Adapter-XL source residual path.",
    )
    parser.add_argument(
        "--spatial-adapter-conditioning-scale",
        type=float,
        default=1.0,
        help="Scale applied to spatial-adapter residuals.",
    )
    parser.add_argument(
        "--spatial-adapter-downscale-factor",
        type=int,
        default=16,
        help="Initial T2I-Adapter-XL pixel unshuffle factor.",
    )
    parser.add_argument(
        "--source-timestep-sampling",
        choices=["uniform", "inference"],
        default="uniform",
        help="Use the full 1000-step schedule or deployed SSD inference timesteps.",
    )
    parser.add_argument(
        "--source-training-inference-steps",
        type=int,
        default=4,
        help="Number of deployed timesteps used by inference-schedule training.",
    )
    parser.add_argument(
        "--source-output-calibration",
        choices=["none", "moments", "moments_and_project"],
        default="none",
        help="Optional inference-only RGB calibration stored with the checkpoint.",
    )
    args = resolve_source_training_defaults(parser.parse_args())
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive")

    try:
        task_ratios = json.loads(args.task_ratios) if args.task_ratios else None
    except json.JSONDecodeError as error:
        parser.error(f"--task-ratios must be valid JSON: {error}")
    if task_ratios is not None:
        if not isinstance(task_ratios, dict) or not task_ratios:
            parser.error("--task-ratios must be a non-empty JSON object")
        if any(float(value) <= 0 for value in task_ratios.values()):
            parser.error("task ratios must be positive")
        ratio_total = sum(float(value) for value in task_ratios.values())
        task_ratios = {str(key): float(value) / ratio_total for key, value in task_ratios.items()}

    try:
        args.max_per_task = resolve_max_per_task(
            args.max_per_task,
            source_conditioned=args.source_conditioned,
            explicit_data_dir=args.data_dir is not None,
        )
    except ValueError as error:
        parser.error(str(error))

    if (
        args.x0_loss_weight < 0
        or args.pixel_loss_weight < 0
        or args.structure_loss_weight < 0
        or args.structure_gradient_weight < 0
        or args.scalar_geometry_loss_weight < 0
        or args.scalar_geometry_gradient_weight < 0
        or args.scalar_geometry_ordinal_weight < 0
        or args.spatial_adapter_conditioning_scale < 0
    ):
        parser.error("loss weights must be non-negative")
    if args.source_training_inference_steps <= 0:
        parser.error("--source-training-inference-steps must be positive")
    if not args.source_conditioned and args.task != "generation":
        parser.error(
            "paired SDXL perception training requires --source-conditioned; "
            "--no-source-conditioned is only valid with --task generation"
        )
    if args.source_conditioned and args.prediction_type != "epsilon":
        parser.error("--source-conditioned requires --prediction-type epsilon")
    if args.x0_loss_weight > 0 and not args.source_conditioned:
        parser.error("--x0-loss-weight requires --source-conditioned")
    if args.pixel_loss_weight > 0 and not args.source_conditioned:
        parser.error("--pixel-loss-weight requires --source-conditioned")
    if args.structure_loss_weight > 0 and not args.source_conditioned:
        parser.error("--structure-loss-weight requires --source-conditioned")
    if args.scalar_geometry_loss_weight > 0 and not args.source_conditioned:
        parser.error("--scalar-geometry-loss-weight requires --source-conditioned")
    if args.spatial_adapter and not args.source_conditioned:
        parser.error("--spatial-adapter requires --source-conditioned")
    if args.spatial_adapter and not args.no_ema:
        parser.error("--spatial-adapter currently requires --no-ema")
    if args.spatial_adapter and args.task in {"all", "generation"}:
        parser.error("--spatial-adapter requires source images for every sample")
    if args.spatial_adapter_downscale_factor <= 0:
        parser.error("--spatial-adapter-downscale-factor must be positive")
    if args.spatial_adapter and args.resolution % args.spatial_adapter_downscale_factor:
        parser.error(
            "--resolution must be divisible by --spatial-adapter-downscale-factor"
        )

    set_seed(args.seed)

    data_dirs = resolve_data_dirs(args.data_dir)
    depth_prompt_override = None
    source_target_colormap = "unknown"
    source_target_channel_mean = None
    source_target_channel_std = None
    source_output_calibration = "none"
    if args.audit_data_only or (args.source_conditioned and args.task == "depth"):
        depth_targets = data_dirs[0] / "depth" / "targets"
        audit = audit_depth_target_encoding(
            depth_targets,
            allowed_names=manifest_split_names(data_dirs[0], "depth"),
        )
        source_target_colormap = audit.detected_colormap
        source_target_channel_mean = audit.channel_mean
        source_target_channel_std = audit.channel_std
        source_output_calibration = args.source_output_calibration
        depth_prompt_override = relative_depth_prompt(source_target_colormap)
        logger.info(
            "Depth target encoding audit: colormap=%s, curve_mae=%.5f, samples=%d",
            audit.detected_colormap,
            audit.mean_curve_mae,
            audit.sample_count,
        )
        logger.info(
            "Depth target channel calibration: mean=%s std=%s samples=%d mode=%s",
            tuple(round(value, 5) for value in audit.channel_mean),
            tuple(round(value, 5) for value in audit.channel_std),
            audit.statistics_sample_count,
            source_output_calibration,
        )
        if audit.detected_colormap != "cube_edge":
            logger.warning(
                "Stored depth prompts/manifest claim paper-style power RGB encoding, "
                "but targets fit %s; overriding the training prompt with the detected "
                "relative-depth encoding.",
                audit.detected_colormap,
            )

    if args.audit_data_only:
        if args.task != "depth":
            parser.error("--audit-data-only currently requires --task depth")
        audit_dataset = VisionTaskDataset(
            data_path=str(data_dirs[0] / "depth"),
            resolution=args.resolution,
            task="depth",
        )
        before_split = len(audit_dataset)
        removed = filter_dataset_to_manifest_split(
            audit_dataset,
            data_dirs[0],
            "depth",
        )
        if depth_prompt_override is not None:
            for sample in audit_dataset.samples:
                sample["prompt"] = depth_prompt_override
        logger.info(
            "Data audit complete: total=%d train=%d excluded_val=%d prompt=%r",
            before_split,
            len(audit_dataset),
            removed,
            audit_dataset.samples[0]["prompt"] if audit_dataset.samples else "",
        )
        return

    device = get_device(args.device)
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
    set_scheduler_prediction_type(generator.pipe.scheduler, args.prediction_type)
    validate_scheduler_prediction_type(generator.pipe.scheduler, args.prediction_type)
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
    lora_target_modules = None
    if args.lora_conv_targets:
        lora_target_modules = (
            list(mc.lora_target_modules) + SDXL_CONV_LORA_TARGET_MODULES
        )
        logger.info(f"Extended LoRA targets with convolutions: {lora_target_modules}")

    tuner = InstructionTuner(
        model=generator,
        task=args.task,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        target_modules=lora_target_modules,
        use_rslora=True,  # Enable rsLoRA for better high-rank training
        lr=args.lr,
        use_ema=not args.no_ema,
        ema_decay=0.9999,
        use_min_snr=not args.source_conditioned,
        use_logit_normal_timesteps=False,
        use_img2img_flow=False,
        cfg_dropout_rate=0.1,
        warmup_ratio=0.05,
        use_source_conditioning=args.source_conditioned,
        prediction_type=args.prediction_type,
        x0_loss_weight=args.x0_loss_weight,
        pixel_loss_weight=args.pixel_loss_weight,
        structure_loss_weight=args.structure_loss_weight,
        structure_gradient_weight=args.structure_gradient_weight,
        scalar_geometry_loss_weight=args.scalar_geometry_loss_weight,
        scalar_geometry_gradient_weight=args.scalar_geometry_gradient_weight,
        scalar_geometry_ordinal_weight=args.scalar_geometry_ordinal_weight,
        use_spatial_adapter=args.spatial_adapter,
        spatial_adapter_conditioning_scale=args.spatial_adapter_conditioning_scale,
        spatial_adapter_downscale_factor=args.spatial_adapter_downscale_factor,
        source_timestep_sampling=args.source_timestep_sampling,
        source_training_inference_steps=args.source_training_inference_steps,
        source_target_colormap=source_target_colormap,
        source_target_channel_mean=source_target_channel_mean,
        source_target_channel_std=source_target_channel_std,
        source_output_calibration=source_output_calibration,
        source_recommended_prompt=(
            depth_prompt_override or VisionTaskDataset.TASK_PROMPTS["depth"]
            if args.task == "depth"
            else ""
        ),
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
    if args.data_dir is not None:
        logger.info(f"Using explicit training data directory only: {data_dirs[0]}")
    else:
        logger.info(
            "Using legacy merged training data directories: %s",
            ", ".join(str(path) for path in data_dirs),
        )

    # Generation dataset class (defined at function scope for reuse)
    import json as _json_mod

    from PIL import Image as _PIL_Image  # noqa: I001

    class GenerationDataset(VisionTaskDataset):
        """Text-to-image dataset using COCO captions."""
        def __init__(self, data_path, resolution=512, max_samples=None):
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
    dataset_task_names: list[str] = []

    def _load_task_dirs(task_name, subdir, vision_task, max_per_task=None, split="train"):
        """Load and merge a task from all data directories."""
        merged = []
        for d in data_dirs:
            task_dir = d / subdir
            if (task_dir / "images").exists():
                ds = VisionTaskDataset(data_path=str(task_dir), resolution=res, task=vision_task)
                if ds.samples:
                    removed = filter_dataset_to_manifest_split(ds, d, task_name, split=split)
                    if removed:
                        logger.info(
                            "  %s: excluded %d declared validation samples",
                            task_name,
                            removed,
                        )
                    if task_name == "depth" and depth_prompt_override is not None:
                        for sample in ds.samples:
                            sample["prompt"] = depth_prompt_override
                    if max_per_task is not None and len(ds.samples) > max_per_task:
                        # Cap samples per task for balance
                        ds.samples = ds.samples[:max_per_task]
                    ds.task = task_name
                    merged.append(ds)
                    logger.info(f"  {task_name} from {d.name}: {len(ds)} samples")
        return merged

    def _load_generation_dirs(split="train"):
        """Load and merge generation data from all directories."""
        merged = []
        for d in data_dirs:
            gen_dir = d / "generation"
            if (gen_dir / "images").exists():
                # Apply the cap after manifest filtering; otherwise a sorted
                # train prefix can hide every held-out validation filename.
                ds = GenerationDataset(
                    str(gen_dir),
                    resolution=res,
                    max_samples=None if split == "val" else args.max_generation,
                )
                if ds.samples:
                    removed = filter_dataset_to_manifest_split(ds, d, "generation", split=split)
                    if removed:
                        logger.info(
                            "  generation: excluded %d declared validation samples",
                            removed,
                        )
                    if split == "val" and args.max_val_per_task:
                        ds.samples = ds.samples[:args.max_val_per_task]
                    ds.task = "generation"
                    merged.append(ds)
                    logger.info(f"  generation from {d.name}: {len(ds)} samples")
        return merged

    # Depth task
    if args.task in ("all", "depth"):
        loaded = _load_task_dirs("depth", "depth", "depth", max_per_task=args.max_per_task)
        datasets.extend(loaded); dataset_task_names.extend(["depth"] * len(loaded))

    # Normal task
    if args.task in ("all", "normal"):
        loaded = _load_task_dirs("normal", "normal", "normal", max_per_task=args.max_per_task)
        datasets.extend(loaded); dataset_task_names.extend(["normal"] * len(loaded))

    # Segmentation task
    # Uses vb_snvc_train_v2 which has unified ADE20K 151-class palette.
    # The older vb_snvc_train (v1) mixed ADE20K+lllab+COCO class spaces,
    # causing supervision conflicts (same color = different classes).
    if args.task in ("all", "segmentation"):
        segmentation_roots = (
            data_dirs
            if args.data_dir is not None
            else [PROJECT_ROOT / "data" / "vb_snvc_train_v2"]
        )
        for data_root in segmentation_roots:
            segmentation_dir = data_root / "segmentation"
            if (segmentation_dir / "images").exists():
                ds = VisionTaskDataset(
                    data_path=str(segmentation_dir), resolution=res, task="semantic"
                )
                if ds.samples:
                    filter_dataset_to_manifest_split(
                        ds,
                        data_root,
                        "segmentation",
                    )
                    if args.max_per_task is not None and len(ds.samples) > args.max_per_task:
                        ds.samples = ds.samples[:args.max_per_task]
                    ds.task = "semantic"
                    datasets.append(ds)
                    dataset_task_names.append("semantic")
                    logger.info(
                        "  segmentation from %s: %d samples", data_root.name, len(ds)
                    )

    # Edge task (edge from vb_zimage_train, structure from vb_snvc_train)
    if args.task in ("all", "edge"):
        loaded = _load_task_dirs("edge", "edge", "edge", max_per_task=args.max_per_task)
        datasets.extend(loaded); dataset_task_names.extend(["edge"] * len(loaded))
        # Also load 'structure' dir from vb_snvc_train as edge-like data
        for d in data_dirs:
            struct_dir = d / "structure"
            if (struct_dir / "images").exists():
                ds = VisionTaskDataset(data_path=str(struct_dir), resolution=res, task="edge")
                if ds.samples:
                    removed = filter_dataset_to_manifest_split(ds, d, "structure")
                    if removed:
                        logger.info("  edge/structure: excluded %d declared validation samples", removed)
                    if args.max_per_task is not None and len(ds.samples) > args.max_per_task:
                        ds.samples = ds.samples[:args.max_per_task]
                    ds.task = "edge"
                    datasets.append(ds)
                    dataset_task_names.append("edge")
                    logger.info(f"  structure(edge) from {d.name}: {len(ds)} samples")

    # Generation task
    if args.task in ("all", "generation"):
        loaded = _load_generation_dirs()
        datasets.extend(loaded); dataset_task_names.extend(["generation"] * len(loaded))

    if not datasets:
        logger.error("No training data found! Run data preparation scripts first.")
        sys.exit(1)

    # Build held-out datasets from the same manifest without contaminating the
    # training set. The tuner records one loss per task at each eval interval.
    val_datasets = {}
    if args.data_dir is not None:
        requested_val_tasks = set(validation_task_names(args.task))
        val_specs: dict[str, list] = {}
        if "depth" in requested_val_tasks:
            val_specs["depth"] = _load_task_dirs(
                "depth", "depth", "depth", split="val"
            )
        if "normal" in requested_val_tasks:
            val_specs["normal"] = _load_task_dirs(
                "normal", "normal", "normal", split="val"
            )
        if "semantic" in requested_val_tasks:
            val_specs["semantic"] = []
        if "edge" in requested_val_tasks:
            val_specs["edge"] = _load_task_dirs(
                "edge", "edge", "edge", split="val"
            )
        if "generation" in requested_val_tasks:
            val_specs["generation"] = _load_generation_dirs(split="val")

        if requested_val_tasks.intersection({"edge", "semantic"}):
            for data_root in data_dirs:
                if "edge" in requested_val_tasks:
                    structure_dir = data_root / "structure"
                    if (structure_dir / "images").exists():
                        ds = VisionTaskDataset(
                            str(structure_dir), resolution=res, task="edge"
                        )
                        filter_dataset_to_manifest_split(
                            ds, data_root, "structure", split="val"
                        )
                        if ds.samples:
                            val_specs["edge"].append(ds)
                if "semantic" in requested_val_tasks:
                    segmentation_dir = data_root / "segmentation"
                    if (segmentation_dir / "images").exists():
                        ds = VisionTaskDataset(
                            str(segmentation_dir), resolution=res, task="semantic"
                        )
                        filter_dataset_to_manifest_split(
                            ds, data_root, "segmentation", split="val"
                        )
                        if ds.samples:
                            val_specs["semantic"].append(ds)
        from torch.utils.data import ConcatDataset
        for task_name, task_sets in val_specs.items():
            if task_sets:
                if args.max_val_per_task:
                    for dataset in task_sets:
                        dataset.samples = dataset.samples[:args.max_val_per_task]
                val_datasets[task_name] = task_sets[0] if len(task_sets) == 1 else ConcatDataset(task_sets)
        logger.info("Validation task sizes: %s", {name: len(ds) for name, ds in val_datasets.items()})

    # Combine all datasets with balanced sampling
    from torch.utils.data import ConcatDataset
    if len(datasets) > 1:
        if args.task_schedule or task_ratios is not None or args.balance_data:
            # WeightedRandomSampler: oversample minority tasks to balance
            from torch.utils.data import WeightedRandomSampler
            dataset_sizes = [len(d) for d in datasets]
            max_size = max(dataset_sizes)
            weights = []
            task_sizes: dict[str, int] = {}
            for label, sz in zip(dataset_task_names, dataset_sizes, strict=True):
                task_sizes[label] = task_sizes.get(label, 0) + sz
            normalized_ratios = task_ratios or {
                label: 1.0 / len(task_sizes) for label in task_sizes
            }
            missing_ratios = sorted(set(task_sizes) - set(normalized_ratios))
            if missing_ratios:
                raise ValueError(f"task ratios missing loaded tasks: {missing_ratios}")
            ratio_sum = sum(normalized_ratios.values())
            normalized_ratios = {
                label: float(value) / ratio_sum for label, value in normalized_ratios.items()
            }
            for label, sz in zip(dataset_task_names, dataset_sizes, strict=True):
                if task_ratios is not None:
                    weights.extend([normalized_ratios[label] / task_sizes[label]] * sz)
                else:
                    weights.extend([max_size / sz] * sz)
            combined = ConcatDataset(datasets)
            if args.task_schedule:
                sampler = TaskScheduleSampler(
                    dataset_sizes,
                    dataset_task_names,
                    num_samples=len(combined),
                    seed=args.seed,
                )
            else:
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
            logger.info("Task ratios: %s", normalized_ratios)
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
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else PROJECT_ROOT / "output" / f"{args.model}_checkpoints"
    )
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
        val_datasets=val_datasets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        scheduler_type="cosine",
        warmup_steps=warmup_steps,
        output_dir=str(output_dir),
        save_every=args.save_every,
        eval_every=args.eval_every,
        use_amp=True,
        num_workers=0,  # MPS must be 0
        log_every=args.log_every,
        start_step=start_step,
        start_epoch=start_epoch,
        train_sampler=sampler,
        resume_checkpoint=resume_checkpoint,
        restore_rng_on_resume=False,  # Old checkpoints may lack rng_state.pt
        max_steps=args.max_steps,
        checkpoint_metadata={
            "task_ratios": task_ratios,
            "task_schedule": bool(args.task_schedule),
            "train_task_sizes": {
                task: sum(len(ds) for ds, label in zip(datasets, dataset_task_names, strict=True) if label == task)
                for task in sorted(set(dataset_task_names))
            },
            "val_task_sizes": {task: len(ds) for task, ds in val_datasets.items()},
        },
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
