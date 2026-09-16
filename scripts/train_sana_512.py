#!/usr/bin/env python3
"""Train Open-Vision-Banana with Sana_600M_512px_diffusers.

Uses the diffusers-format Sana-600M 512px model as the backbone for
LoRA-based instruction tuning on the vb_snvc_train_v3 multi-task dataset.

Key characteristics of Sana_600M_512px_diffusers:
  - Backbone: SanaTransformer (linear DiT, ~590M params)
  - VAE: AutoencoderDC (DC-AE, 32x downsample, 32 latent channels)
  - Text encoder: Gemma-2-2B-IT (max_length=300, dim=2048)
  - Resolution: 512x512
  - Flow-matching scheduler (linear_flow, flow_shift=3.0)

The script reuses the project's InstructionTuner and MixedVisionDataset
infrastructure, with Sana-specific adjustments for VAE encoding and
attention masks.

Usage:
    # Full multi-task training (all 12 tasks)
    python scripts/train_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --steps 10000 --lr 5e-5 \
        --output-dir output/sana_512_vb

    # Single-task debug (depth only)
    python scripts/train_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --tasks depth --steps 2000 \
        --output-dir output/sana_512_depth_debug

    # With generation preservation (9:1 mix)
    python scripts/train_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --vision-ratio 0.9 \
        --output-dir output/sana_512_mixed
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Tolerate partially-downloaded / truncated PNGs in the SNVC manifest instead of
# crashing the whole training run; unreadable images are resampled in __getitem__.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Use HF mirror for models
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from open_vision_banana.data import (
    HomogeneousMixedBatchSampler,
    MixedVisionDataset,
    build_v3_datasets,
    load_v3_train_val_split,
)
from open_vision_banana.data.datasets import CityscapesSegDataset
from open_vision_banana.data.manifest import resolve_manifest_path
from open_vision_banana.model_registry import (
    PipelineComponents,
    apply_lora,
    get_model_config,
    load_pipeline,
)
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.training.optimization import (
    SchedulerFactory,
)
from open_vision_banana.training.sana_flow import (
    flow_min_snr_weight,
    native_sana_timestep,
    normalize_vae_latents,
    prepare_sana_flow_batch,
    sample_sana_flow_timesteps,
    denormalize_vae_latents,
)
from open_vision_banana.training.sana_depth_aux import (
    depth_edge_endpoint_loss,
    depth_task_mask,
)
from open_vision_banana.training.utils import EMAModel

logger = logging.getLogger(__name__)


def portable_project_path(value: Path | None) -> str | None:
    """Serialize project-local paths relative to the repository root."""
    if value is None:
        return None
    path = Path(value)
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()

# Default LoRA targets for Sana transformer blocks
SANA_LORA_TARGETS = [
    "linear",  # attention linear proj
    "proj_out",  # output projection
    "conv_inverted",  # GLUMBConv inverted conv
    "conv_point",  # GLUMBConv pointwise conv
]

# Default task weights (sqrt balancing for imbalanced tasks)
DEFAULT_TASK_WEIGHTS: dict[str, float] = {
    "depth": 2.5,
    "normal": 2.0,
    "edge": 1.5,
    "semantic": 0.5,
    "generation": 0.5,
    "instance": 1.0,
    "keypoint": 1.0,
    "object_detection": 1.0,
    "depth_edge": 1.0,
    "referring": 1.5,
    "multiview": 2.0,
    "specular_removal": 2.0,
}


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    config_defaults = load_config_defaults(config_args.config)

    p = argparse.ArgumentParser(
        description="Train Open-Vision-Banana with Sana_600M_512px_diffusers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="YAML configuration; explicit CLI arguments take precedence.",
    )

    # ── Data ──────────────────────────────────────────────────────────
    p.add_argument(
        "--data-dir",
        required=False,
        default=config_defaults.get("data_dir"),
        help="Path to a v3 dataset or the v5_full dataset root (not required with --tar-index)",
    )
    p.add_argument(
        "--tasks", nargs="+", default=config_defaults.get("tasks"), help="Subset of tasks to train (default: all)"
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=config_defaults.get("resolution", 512),
        help="Training resolution (default 512, matching model native)",
    )
    p.add_argument(
        "--vision-ratio",
        type=float,
        default=config_defaults.get("vision_ratio", 1.0),
        help="Ratio of vision vs generation data (default 1.0 = all vision)",
    )
    p.add_argument(
        "--pair-manifest",
        type=Path,
        default=config_defaults.get("pair_manifest"),
        help=(
            "Explicit JSONL of input/target pairs. Required for the v5_full "
            "flat dataset to preserve annotation-backed pair semantics."
        ),
    )
    p.add_argument(
        "--tar-index",
        type=Path,
        default=config_defaults.get("tar_index"),
        help=(
            "JSON index of tar.gz split volumes (built by build_tar_index_snvc50m.py). "
            "When provided, images are read directly from the compressed tarballs "
            "without full extraction."
        ),
    )
    p.add_argument(
        "--tar-root",
        type=Path,
        default=config_defaults.get("tar_root", "data/snvc50m"),
        help="Directory containing the split tar.gz volumes (used with --tar-index).",
    )
    p.add_argument(
        "--tar-disk-cache",
        type=Path,
        default=config_defaults.get("tar_disk_cache"),
        help="Optional disk cache directory for tar-extracted images.",
    )
    p.add_argument(
        "--tar-target-mode",
        action="store_true",
        default=config_defaults.get("tar_target_mode", False),
        help=(
            "Target-only training from tar volumes (no input RGB images). "
            "Uses noise→target flow matching. Requires per-task tar indexes "
            "built by scripts/build_task_tar_index.py."
        ),
    )
    p.add_argument(
        "--task-tar-index-dir",
        type=Path,
        default=config_defaults.get("task_tar_index_dir"),
        help="Directory containing per-task tar indexes (tar_index_{depth,normal,seg}.json).",
    )
    p.add_argument(
        "--generation-dir",
        type=Path,
        default=config_defaults.get("generation_dir"),
        help=(
            "Text-to-image preservation dataset. Required when --vision-ratio is below 1.0 "
            "for v5_full training."
        ),
    )
    p.add_argument(
        "--require-captioned-generation",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("require_captioned_generation", False),
        help="Require explicit per-image caption metadata for generation replay.",
    )
    p.add_argument(
        "--homogeneous-batches",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("homogeneous_batches", False),
        help="Use deterministic all-vision or all-generation micro-batches when replay is enabled.",
    )

    # ── Model ────────────────────────────────────────────────────────
    p.add_argument(
        "--model-name",
        default=config_defaults.get("model_name", "sana-600m-512px"),
        help="Model registry name (default: sana-600m-512px)",
    )
    p.add_argument("--lora-rank", type=int, default=config_defaults.get("lora_rank", 16))
    p.add_argument("--lora-alpha", type=int, default=config_defaults.get("lora_alpha", 32))
    p.add_argument("--no-rslora", action="store_true", default=not config_defaults.get("rslora", True), help="Disable rsLoRA scaling")
    p.add_argument("--dtype", default=config_defaults.get("dtype", "float16"), choices=["float16", "bfloat16", "float32"])

    # ── Training ─────────────────────────────────────────────────────
    p.add_argument("--steps", type=int, default=config_defaults.get("steps", 10000), help="Total training steps")
    p.add_argument(
        "--stop-after-step",
        type=int,
        default=None,
        help="Stop cleanly after this global step while preserving the regular checkpoint.",
    )
    p.add_argument("--lr", type=float, default=config_defaults.get("lr", 5e-5))
    p.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 4))
    p.add_argument(
        "--grad-accum",
        type=int,
        default=config_defaults.get("grad_accum", 4),
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    p.add_argument("--max-grad-norm", type=float, default=config_defaults.get("max_grad_norm", 1.0))
    p.add_argument("--seed", type=int, default=config_defaults.get("seed", 42))
    p.add_argument("--num-workers", type=int, default=config_defaults.get("num_workers", 4))

    # ── Scheduler ────────────────────────────────────────────────────
    p.add_argument(
        "--scheduler",
        default=config_defaults.get("scheduler", "cosine"),
        choices=["cosine", "constant", "cosine_restarts", "one_cycle"],
    )
    p.add_argument("--warmup-ratio", type=float, default=config_defaults.get("warmup_ratio", 0.05))
    p.add_argument("--warmup-steps", type=int, default=config_defaults.get("warmup_steps"))

    # ── Loss ─────────────────────────────────────────────────────────
    p.add_argument(
        "--no-min-snr",
        action="store_true",
        default=not config_defaults.get("min_snr", True),
        help="Disable Min-SNR loss weighting",
    )
    p.add_argument("--min-snr-gamma", type=float, default=config_defaults.get("min_snr_gamma", 5.0))
    p.add_argument(
        "--no-logit-normal-timesteps",
        action="store_true",
        default=not config_defaults.get("logit_normal_timesteps", True),
        help="Use uniform timesteps instead of logit-normal",
    )
    p.add_argument(
        "--cfg-dropout-rate",
        type=float,
        default=config_defaults.get("cfg_dropout_rate", 0.1),
        help="CFG dropout rate for text conditioning",
    )
    p.add_argument(
        "--flow-sigma",
        type=float,
        default=config_defaults.get("flow_sigma", 0.0),
        help="Flow-matching noise sigma (0 = pure flow)",
    )
    p.add_argument(
        "--no-img2img-flow",
        action="store_true",
        default=not config_defaults.get("img2img_flow", True),
        help="Disable img2img-style flow matching (use noise→target instead)",
    )
    p.add_argument(
        "--depth-edge-aux-weight",
        type=float,
        default=config_defaults.get("depth_edge_aux_weight", 0.0),
        help="Add depth-edge-aware endpoint loss with this multiplier (0 disables it).",
    )
    p.add_argument(
        "--depth-edge-aux-gradient-weight",
        type=float,
        default=config_defaults.get("depth_edge_aux_gradient_weight", 0.25),
        help="Relative spatial-gradient term within the depth-edge auxiliary loss.",
    )
    p.add_argument(
        "--depth-edge-aux-edge-weight",
        type=float,
        default=config_defaults.get("depth_edge_aux_edge_weight", 2.0),
        help="Additional loss weight for latent cells covered by target depth edges.",
    )
    p.add_argument(
        "--depth-edge-aux-edge-threshold",
        type=float,
        default=config_defaults.get("depth_edge_aux_edge_threshold", 0.025),
        help="Decoded normalized-depth gradient needed to mark an edge.",
    )
    p.add_argument(
        "--depth-edge-aux-sigma-floor",
        type=float,
        default=config_defaults.get("depth_edge_aux_sigma_floor", 0.10),
        help="Minimum flow sigma used to normalize endpoint reconstruction error.",
    )

    # ── Evaluation & Saving ──────────────────────────────────────────
    p.add_argument("--output-dir", default=config_defaults.get("output_dir", "output/sana_512_vb"))
    p.add_argument("--save-every", type=int, default=config_defaults.get("save_every", 1000))
    p.add_argument("--eval-every", type=int, default=config_defaults.get("eval_every", 500))
    p.add_argument("--log-every", type=int, default=config_defaults.get("log_every", 50))
    p.add_argument("--ema-decay", type=float, default=config_defaults.get("ema_decay", 0.9999))
    p.add_argument("--ema-warmup", type=int, default=config_defaults.get("ema_warmup", 500))
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)

    # ── Resume ───────────────────────────────────────────────────────
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument(
        "--resume-weights-only",
        action="store_true",
        help="Load only LoRA weights from --resume and reset optimizer, scheduler, and step count.",
    )

    # ── Device ───────────────────────────────────────────────────────
    p.add_argument("--device", default=config_defaults.get("device"), help="Device (auto-detect if not set)")

    # ── Task weights ─────────────────────────────────────────────────
    p.add_argument(
        "--task-weights",
        default=config_defaults.get("task_weights", "sqrt"),
        help="Task weight mode: sqrt, uniform, or comma-separated floats",
    )
    p.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help="Debug gate: cap manifest samples independently for each task.",
    )
    p.add_argument(
        "--task-loss-weights",
        default=config_defaults.get("task_loss_weights"),
        help="Optional per-task loss multipliers, e.g. semantic=2,structure=2.",
    )

    return p.parse_args()


def load_config_defaults(config_path: Path | None) -> dict[str, object]:
    """Map the documented SANA YAML format to CLI defaults.

    The training script intentionally remains CLI-first.  Reading the YAML
    before parsing lets command-line flags override its values naturally.
    """
    if config_path is None:
        return {}
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    import yaml

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("SANA training config must be a mapping")

    model = raw.get("model", {}) or {}
    lora = raw.get("lora", {}) or {}
    data = raw.get("data", {}) or {}
    training = raw.get("training", {}) or {}
    scheduler = raw.get("scheduler", {}) or {}
    flow = raw.get("flow", {}) or {}
    loss = raw.get("loss", {}) or {}
    checkpoint = raw.get("checkpoint", {}) or {}
    ema = raw.get("ema", {}) or {}
    defaults: dict[str, object] = {
        "model_name": model.get("name"),
        "dtype": model.get("dtype"),
        "lora_rank": lora.get("rank"),
        "lora_alpha": lora.get("alpha"),
        "rslora": lora.get("rs_lora"),
        "data_dir": data.get("dir"),
        "pair_manifest": data.get("pair_manifest"),
        "tar_index": data.get("tar_index"),
        "tar_root": data.get("tar_root"),
        "tar_disk_cache": data.get("tar_disk_cache"),
        "tar_target_mode": data.get("tar_target_mode", False),
        "task_tar_index_dir": data.get("task_tar_index_dir"),
        "generation_dir": data.get("generation_dir"),
        "require_captioned_generation": data.get("require_captioned_generation", False),
        "homogeneous_batches": data.get("homogeneous_batches", False),
        "resolution": data.get("resolution", model.get("resolution")),
        "vision_ratio": data.get("vision_ratio"),
        "tasks": data.get("tasks"),
        "num_workers": data.get("num_workers"),
        "task_weights": data.get("task_weights"),
        "steps": training.get("steps"),
        "batch_size": training.get("batch_size"),
        "grad_accum": training.get("gradient_accumulation"),
        "lr": training.get("lr"),
        "max_grad_norm": training.get("max_grad_norm"),
        "seed": training.get("seed"),
        "scheduler": scheduler.get("type"),
        "warmup_ratio": scheduler.get("warmup_ratio"),
        "warmup_steps": scheduler.get("warmup_steps"),
        "img2img_flow": flow.get("img2img_flow"),
        "flow_sigma": flow.get("sigma"),
        "logit_normal_timesteps": flow.get("logit_normal_timesteps"),
        "min_snr": flow.get("min_snr"),
        "min_snr_gamma": flow.get("min_snr_gamma"),
        "cfg_dropout_rate": flow.get("cfg_dropout_rate"),
        "depth_edge_aux_weight": loss.get("depth_edge_aux_weight"),
        "depth_edge_aux_gradient_weight": loss.get("depth_edge_aux_gradient_weight"),
        "depth_edge_aux_edge_weight": loss.get("depth_edge_aux_edge_weight"),
        "depth_edge_aux_edge_threshold": loss.get("depth_edge_aux_edge_threshold"),
        "depth_edge_aux_sigma_floor": loss.get("depth_edge_aux_sigma_floor"),
        "task_loss_weights": loss.get("task_loss_weights", training.get("task_loss_weights")),
        "output_dir": checkpoint.get("output_dir"),
        "save_every": checkpoint.get("save_every"),
        "eval_every": checkpoint.get("eval_every"),
        "log_every": checkpoint.get("log_every"),
        "ema_decay": ema.get("decay"),
        "ema_warmup": ema.get("warmup_steps"),
        "device": raw.get("device"),
    }
    return {name: value for name, value in defaults.items() if value is not None}


def get_dtype(dtype_str: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype_str
    ]


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_text_sana(
    components: PipelineComponents,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    max_length: int = 300,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text prompts for Sana using Gemma-2-2B-IT.

    Returns (encoder_hidden_states, attention_mask).
    """
    tokenizer = components.tokenizer
    text_encoder = components.text_encoder

    # Tokenize with padding to max_length
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    # Handle text encoder on CPU or GPU
    text_device = next(text_encoder.parameters()).device
    input_ids = tokenized.input_ids.to(text_device)
    attention_mask = tokenized.attention_mask.to(text_device)

    with torch.no_grad():
        outputs = text_encoder(input_ids, attention_mask=attention_mask)
        # Gemma returns last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            encoder_hidden_states = outputs.last_hidden_state
        else:
            encoder_hidden_states = outputs[0]

    return encoder_hidden_states.to(device, dtype), attention_mask.to(device, dtype)


def vae_encode_sana(
    vae: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode images to latents using DC-AE.

    DC-AE: 32x downsample, 32 latent channels.
    Input: [B, 3, H, W] in [-1, 1]
    Output: [B, 32, H/32, W/32]

    DC-AE's encode() returns EncoderOutput(latent=...) — NOT latent_dist.
    """
    with torch.no_grad():
        enc_out = vae.encode(images.to(dtype=vae.dtype))
        # DC-AE returns EncoderOutput with .latent attribute
        if hasattr(enc_out, "latent_dist"):
            latents = enc_out.latent_dist.mode()
        elif hasattr(enc_out, "latent"):
            latents = enc_out.latent
        elif hasattr(enc_out, "sample"):
            latents = enc_out.sample
        elif isinstance(enc_out, torch.Tensor):
            latents = enc_out
        else:
            # Fallback: try common attribute names
            for attr in ("latents", "encoded", "z"):
                if hasattr(enc_out, attr):
                    latents = getattr(enc_out, attr)
                    break
            else:
                raise TypeError(
                    f"VAE encode returned {type(enc_out).__name__} with no known "
                    f"latent attribute. Available: {dir(enc_out)}"
                )

    return normalize_vae_latents(latents.to(dtype), vae)


def vae_decode_sana(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode latents to images using DC-AE.

    Input: [B, 32, H/32, W/32]
    Output: [B, 3, H, W] in [-1, 1]
    """
    with torch.no_grad():
        dec_out = vae.decode(denormalize_vae_latents(latents, vae).to(dtype=vae.dtype))
        # DC-AE decode may return DecoderOutput or tensor
        if hasattr(dec_out, "sample"):
            images = dec_out.sample
        elif isinstance(dec_out, torch.Tensor):
            images = dec_out
        else:
            # Try common attribute names
            for attr in ("image", "images", "decoded", "x"):
                if hasattr(dec_out, attr):
                    images = getattr(dec_out, attr)
                    break
            else:
                raise TypeError(
                    f"VAE decode returned {type(dec_out).__name__} with no known "
                    f"sample attribute. Available: {dir(dec_out)}"
                )

    return images


def flexible_collate(batch: list[dict]) -> dict:
    """Custom collate that handles inconsistent keys across samples.

    Different vision tasks may return different optional keys (e.g. only
    depth tasks include 'colormap').  ``default_collate`` requires every
    sample to share the exact same key set, so we:
      1. Compute the intersection of keys across the batch.
      2. Collate only those common keys with ``default_collate``.
      3. Any non-common key is dropped (the training loop does not use
         task-specific extras like 'colormap').
    """
    normalized = []
    for sample in batch:
        sample = dict(sample)
        # Vision tasks begin flow matching at their source image; generation
        # preservation samples explicitly override this with ``noise``.
        sample.setdefault("flow_source", "input")
        normalized.append(sample)

    all_keys = set(normalized[0].keys())
    for sample in normalized[1:]:
        all_keys &= set(sample.keys())
    common = [{k: s[k] for k in all_keys} for s in normalized]
    from torch.utils.data._utils.collate import default_collate

    return default_collate(common)


def select_sana_flow_sources(
    input_latents: torch.Tensor,
    target_latents: torch.Tensor,
    flow_source: object,
) -> torch.Tensor:
    """Select per-item flow origins for mixed vision/generation batches.

    Vision items use their encoded source image. Generation-preservation items
    use fresh Gaussian noise even though an image tensor is present for batch
    collation. This preserves the paper's text-to-image retention objective.
    """
    batch_size = target_latents.shape[0]
    if input_latents.shape != target_latents.shape:
        raise ValueError("input and target latent shapes must match")
    if isinstance(flow_source, str):
        flow_source = [flow_source]
    if flow_source is None:
        flow_source = ["input"] * batch_size
    flow_source = list(flow_source)
    if len(flow_source) != batch_size:
        raise ValueError("flow_source must contain one value per batch item")
    use_input_source = torch.tensor(
        [str(source) == "input" for source in flow_source],
        device=target_latents.device,
        dtype=torch.bool,
    )
    return torch.where(
        use_input_source.view(batch_size, 1, 1, 1),
        input_latents,
        torch.randn_like(target_latents),
    )


def apply_task_loss_weights(
    per_sample_loss: torch.Tensor,
    task_values: object,
    task_loss_weights: dict[str, float] | None,
) -> torch.Tensor:
    """Scale each sample loss with the configured task multiplier."""
    if not task_loss_weights:
        return per_sample_loss

    if isinstance(task_values, str):
        tasks = [task_values]
    elif task_values is None:
        tasks = [""] * per_sample_loss.shape[0]
    else:
        tasks = [str(task) for task in task_values]

    if len(tasks) != per_sample_loss.shape[0]:
        raise ValueError(
            "Task labels must have one value per sample: "
            f"got {len(tasks)} labels for batch size {per_sample_loss.shape[0]}"
        )

    task_scale = torch.tensor(
        [task_loss_weights.get(task, 1.0) for task in tasks],
        device=per_sample_loss.device,
        dtype=per_sample_loss.dtype,
    )
    return per_sample_loss * task_scale


class ManifestVisionDataset(Dataset):
    """Load explicit RGB input/target pairs emitted by the v5 manifest builder.

    The v5 pack stores both source and supervision images in flat task
    directories.  Pairing by filename can therefore use a target image as its
    own input.  This dataset accepts only the annotation-backed pair manifest
    produced by ``build_mar_v5_clean_manifest.py``.
    """

    TASKS: dict[str, tuple[str, str]] = {
        "depth": (
            "depth",
            "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map.",
        ),
        "metric_depth": (
            "depth",
            "Generate a metric depth visualization of this image.",
        ),
        "normal": (
            "normal",
            "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
        ),
        "seg": (
            "semantic",
            "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
        ),
        "cityscapes": (
            "semantic",
            "Generate a 19-class Cityscapes semantic segmentation mask. Use only the official Cityscapes trainId palette, with one solid class color per pixel and sharp object boundaries.",
        ),
        "structure": (
            "structure",
            "Generate a binary structural edge map. Use a black background and thin white edges aligned to visible scene boundaries.",
        ),
    }

    def __init__(
        self,
        manifest_path: Path,
        resolution: int,
        split: str,
        tasks: set[str],
        max_samples: int | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.task = "unknown"
        self.dir_name = "manifest"
        self.resolution = resolution
        self.samples: list[dict[str, str]] = []
        # When data_root is provided, resolve relative paths against it
        # instead of the manifest file's parent directory.  This is needed
        # for the SN-VC-50M manifest where paths are relative to the
        # extracted data root, not the manifest location.
        path_anchor = data_root.resolve() if data_root is not None else manifest_path

        with manifest_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_task = row.get("task")
                task_info = self.TASKS.get(raw_task)
                if task_info is None or task_info[0] not in tasks:
                    continue
                if row.get("split") != split:
                    continue
                image_path = resolve_manifest_path(row.get("input") or "", path_anchor)
                target_path = resolve_manifest_path(row.get("target") or "", path_anchor)
                if not image_path.is_file() or not target_path.is_file():
                    logger.warning(
                        "Skipping unavailable manifest pair at %s:%d", manifest_path, line_no
                    )
                    continue
                prompt = row.get("prompt")
                if not prompt and raw_task == "cityscapes":
                    # The Cityscapes RGB palette is part of the task contract;
                    # keep the training prompt identical to inference.
                    prompt = PromptBuilder.build_semantic_prompt(
                        CityscapesSegDataset.CITYSCAPES_COLORS
                    )
                prompt = prompt or task_info[1]
                self.samples.append(
                    {
                        "image_path": str(image_path),
                        "target_path": str(target_path),
                        "task": task_info[0],
                        "dir_name": raw_task,
                        "prompt": prompt,
                    }
                )

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        if self.samples:
            names = {sample["dir_name"] for sample in self.samples}
            self.dir_name = next(iter(names)) if len(names) == 1 else "manifest"
            labels = {sample["task"] for sample in self.samples}
            self.task = next(iter(labels)) if len(labels) == 1 else "mixed"

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sample(self, idx: int) -> dict:
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as image:
            image.load()
            image = image.convert("RGB").resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            input_tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)
        with Image.open(sample["target_path"]) as image:
            image.load()
            target_resampling = (
                Image.Resampling.NEAREST
                if sample["task"] == "semantic"
                else Image.Resampling.LANCZOS
            )
            image = image.convert("RGB").resize(
                (self.resolution, self.resolution), target_resampling
            )
            target_tensor = torch.from_numpy(
                np.asarray(image, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)

        return {
            "prompt": sample["prompt"],
            "image": input_tensor * 2.0 - 1.0,
            "image_range": "-1_1",
            "target_rgb": target_tensor,
            "target_rgb_range": "0_1",
            "task": sample["task"],
            "flow_source": "input",
        }

    def __getitem__(self, idx: int) -> dict:
        # Some SNVC pairs reference truncated / unreadable images. LOAD_TRUNCATED_IMAGES
        # keeps most usable; if a sample still fails, resample a bounded number of times
        # so a single bad file never aborts the run.
        last_err = None
        for attempt, cur in enumerate(self._candidate_indices(idx)):
            try:
                return self._load_sample(cur)
            except Exception as err:  # noqa: BLE001 - any decode/IO error triggers retry
                last_err = err
                if attempt == 0:
                    logger.warning("Skipping unreadable sample idx=%s (%s)", cur, err)
        raise RuntimeError(f"All {4} candidate samples failed to load (last: {last_err})")

    def _candidate_indices(self, idx: int):
        yield idx
        rng = random.Random(idx)
        n = len(self.samples)
        for _ in range(3):
            yield rng.randrange(n) if n else idx



def main():
    args = parse_args()
    if args.depth_edge_aux_weight < 0:
        raise ValueError("--depth-edge-aux-weight must be non-negative")
    if args.depth_edge_aux_gradient_weight < 0:
        raise ValueError("--depth-edge-aux-gradient-weight must be non-negative")
    if args.depth_edge_aux_edge_weight < 0:
        raise ValueError("--depth-edge-aux-edge-weight must be non-negative")
    if args.depth_edge_aux_edge_threshold <= 0:
        raise ValueError("--depth-edge-aux-edge-threshold must be positive")
    if args.depth_edge_aux_sigma_floor <= 0:
        raise ValueError("--depth-edge-aux-sigma-floor must be positive")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    device = torch.device(args.device or auto_device())
    dtype = get_dtype(args.dtype)

    logger.info("=" * 70)
    logger.info("Open-Vision-Banana — Sana_600M_512px Training")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {device} | dtype: {dtype}")
    logger.info(
        f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, rsLoRA={not args.no_rslora}"
    )
    logger.info(
        f"Training: steps={args.steps}, lr={args.lr}, effective_batch={args.batch_size * args.grad_accum}"
    )
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Resolution: {args.resolution}")
    logger.info(f"Output: {args.output_dir}")
    if args.depth_edge_aux_weight:
        logger.info(
            "Depth edge auxiliary loss: weight=%.3f edge_weight=%.3f gradient_weight=%.3f",
            args.depth_edge_aux_weight,
            args.depth_edge_aux_edge_weight,
            args.depth_edge_aux_gradient_weight,
        )

    # ── Set seed ─────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load model ───────────────────────────────────────────────────
    logger.info("Loading Sana_600M_512px_diffusers...")
    model_cfg = get_model_config(args.model_name)
    logger.info(f"  model_id: {model_cfg.model_id}")
    logger.info(f"  family: {model_cfg.family}")
    logger.info(f"  VAE downsample: {model_cfg.vae_downsample_factor}x")
    logger.info(f"  text_encoder_max_length: {model_cfg.text_encoder_max_length}")

    components = load_pipeline(
        model_cfg,
        device=device,
        dtype=dtype,
        enable_gradient_checkpointing=args.gradient_checkpointing,
    )

    # Sana-specific dtype adjustments. Keep the frozen components at a stable
    # dtype for the full run; recasting a 2B text encoder every batch is both
    # slow and prone to MPS memory fragmentation.
    if dtype == torch.float16:
        logger.info("Applying Sana dtype adjustments: text_encoder→fp32, vae→bf16")
        components.text_encoder.to(torch.float32)
        components.vae.to(torch.bfloat16)

    # Clear GPU memory before applying LoRA to avoid OOM
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    backbone = components.unet  # transformer
    vae = components.vae

    backbone_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"  Backbone params: {backbone_params / 1e6:.1f}M")

    # ── Apply LoRA ───────────────────────────────────────────────────
    logger.info("Applying LoRA adapters...")
    peft_model = apply_lora(
        backbone,
        config=model_cfg,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        use_rslora=not args.no_rslora,
    )

    # ── EMA ──────────────────────────────────────────────────────────
    ema = EMAModel(peft_model, decay=args.ema_decay, warmup_steps=args.ema_warmup)

    # ── Build datasets ───────────────────────────────────────────────
    logger.info("Building datasets...")
    data_root = Path(args.data_dir) if args.data_dir else None

    # Determine which tasks to load
    all_tasks = [
        "depth",
        "normal",
        "segmentation",
        "structure",
        "generation",
        "instance",
        "keypoint",
        "object_detection",
        "depth_edge",
        "referring",
        "multiview",
        "specular_removal",
    ]
    tasks_to_load = args.tasks if args.tasks else all_tasks

    generation_dataset = None
    if args.tar_target_mode:
        # ── Target-only mode: read ONLY targets from tar (no input RGB) ──
        from open_vision_banana.data.tar_dataset import TarTargetDataset

        tar_root = args.tar_root.expanduser().resolve()
        task_index_dir = args.task_tar_index_dir
        if task_index_dir is None:
            # Default: same directory as tar_index
            task_index_dir = args.tar_index.parent / "task_indexes" if args.tar_index else Path("output/snvc50m_manifest/task_indexes")
        task_index_dir = task_index_dir.expanduser().resolve()

        if not task_index_dir.is_dir():
            logger.error("Task tar index directory not found: %s", task_index_dir)
            logger.error("Run: python scripts/build_task_tar_index.py")
            return

        logger.info("Using tar-target-only mode: %s", task_index_dir)

        # Map task names to index files
        task_index_map = {
            "depth": task_index_dir / "tar_index_depth.json",
            "normal": task_index_dir / "tar_index_normal.json",
            "seg": task_index_dir / "tar_index_seg.json",
        }

        # Filter to requested tasks
        aliases = {"segmentation": "seg", "semantic": "seg"}
        requested_tasks = set()
        for t in tasks_to_load:
            mapped = aliases.get(t, t)
            if mapped in task_index_map:
                requested_tasks.add(mapped)

        train_vision_datasets = []
        for task in sorted(requested_tasks):
            idx_path = task_index_map[task]
            if not idx_path.is_file():
                logger.warning("Task index not found: %s (skipping %s)", idx_path, task)
                continue
            ds = TarTargetDataset(
                tar_index_path=idx_path,
                tar_root=tar_root,
                task=task,
                resolution=args.resolution,
                disk_cache_dir=args.tar_disk_cache,
                max_samples=args.max_samples_per_task,
            )
            if len(ds) > 0:
                train_vision_datasets.append(ds)
                logger.info("  %s: %d samples", task, len(ds))

        # No val dataset in target-only mode (no split info in tar index)
        val_dataset = None
        logger.info("Tar-target-only: %d total train samples",
                    sum(len(ds) for ds in train_vision_datasets))

    elif args.tar_index is not None:
        # ── Tar-direct-access mode: read from compressed tarballs ──────
        from open_vision_banana.data.tar_dataset import TarArchiveDataset

        tar_index_path = args.tar_index.expanduser().resolve()
        tar_root = args.tar_root.expanduser().resolve()
        if not tar_index_path.is_file():
            logger.error("Tar index not found: %s", tar_index_path)
            return
        logger.info("Using tar-direct-access mode: %s", tar_index_path)

        aliases = {"segmentation": "semantic", "seg": "semantic"}
        manifest_tasks = {aliases.get(task, task) for task in tasks_to_load}
        manifest_path = args.pair_manifest.expanduser().resolve() if args.pair_manifest else None

        train_vision_datasets = [
            TarArchiveDataset(
                manifest_path=manifest_path,
                tar_index_path=tar_index_path,
                tar_root=tar_root,
                resolution=args.resolution,
                split="train",
                tasks={task},
                disk_cache_dir=args.tar_disk_cache,
                max_samples=args.max_samples_per_task,
            )
            for task in sorted(manifest_tasks)
        ]
        train_vision_datasets = [ds for ds in train_vision_datasets if len(ds) > 0]

        val_datasets = [
            TarArchiveDataset(
                manifest_path=manifest_path,
                tar_index_path=tar_index_path,
                tar_root=tar_root,
                resolution=args.resolution,
                split="val",
                tasks={task},
                disk_cache_dir=args.tar_disk_cache,
            )
            for task in sorted(manifest_tasks)
        ]
        val_datasets = [ds for ds in val_datasets if len(ds) > 0]
        from torch.utils.data import ConcatDataset

        val_dataset = ConcatDataset(val_datasets) if val_datasets else None
        logger.info("Tar-direct-access: %d train samples, %d val samples",
                    sum(len(ds) for ds in train_vision_datasets),
                    sum(len(ds) for ds in val_datasets) if val_datasets else 0)
    elif args.pair_manifest is not None:
        manifest_path = args.pair_manifest.expanduser().resolve()
        if not manifest_path.is_file():
            logger.error("Pair manifest not found: %s", manifest_path)
            return
        aliases = {"segmentation": "semantic", "seg": "semantic"}
        manifest_tasks = {aliases.get(task, task) for task in tasks_to_load}
        train_vision_datasets = [
            ManifestVisionDataset(
                manifest_path,
                args.resolution,
                "train",
                {task},
                args.max_samples_per_task,
                data_root=data_root,
            )
            for task in sorted(manifest_tasks)
        ]
        train_vision_datasets = [ds for ds in train_vision_datasets if len(ds) > 0]
        val_datasets = [
            ManifestVisionDataset(manifest_path, args.resolution, "val", {task}, data_root=data_root)
            for task in sorted(manifest_tasks)
        ]
        val_datasets = [ds for ds in val_datasets if len(ds) > 0]
        from torch.utils.data import ConcatDataset

        val_dataset = ConcatDataset(val_datasets) if val_datasets else None
        logger.info("Detected explicit v5 pair manifest: %s", manifest_path)
    else:
        if data_root is None:
            logger.error(
                "--data-dir is required when not using --tar-index. "
                "Provide either --data-dir or --tar-index."
            )
            return
        v3_split = load_v3_train_val_split(data_root)
        is_v3 = v3_split is not None or (data_root / "manifest.json").exists()

        if not is_v3:
            logger.error(
                "Dataset must be v3 format or supply --pair-manifest for v5_full data"
            )
            return

        logger.info(f"Detected v3-format dataset at {data_root}")
        train_vision_datasets = build_v3_datasets(
            data_root,
            resolution=args.resolution,
            split="train",
            use_split_file=v3_split is not None,
        )
        train_vision_datasets = [
            ds
            for ds in train_vision_datasets
            if ds.task in tasks_to_load or ds.dir_name in tasks_to_load
        ]

        val_dataset = None
        if v3_split is not None:
            val_datasets = build_v3_datasets(
                data_root,
                resolution=args.resolution,
                split="val",
                use_split_file=True,
            )
            val_datasets = [
                ds
                for ds in val_datasets
                if ds.task in tasks_to_load or ds.dir_name in tasks_to_load
            ]
            if val_datasets:
                from torch.utils.data import ConcatDataset

                val_dataset = ConcatDataset(val_datasets)

    if args.vision_ratio < 1.0:
        from open_vision_banana.data import build_generation_dataset

        generation_dir = args.generation_dir
        if generation_dir is None and args.pair_manifest is None:
            candidate = data_root / "generation"
            generation_dir = candidate if candidate.is_dir() else None
        if generation_dir is None:
            raise ValueError(
                "--generation-dir is required when --vision-ratio is below 1.0 "
                "for an explicit v5 pair manifest."
            )
        generation_dir = generation_dir.expanduser().resolve()
        if not generation_dir.is_dir():
            raise NotADirectoryError(f"Generation preservation data not found: {generation_dir}")

        gen_split_filter = None
        if args.pair_manifest is None and v3_split is not None:
            gen_filenames = v3_split.get("train", {}).get("generation")
            if gen_filenames is not None:
                gen_split_filter = (set(gen_filenames), "train")
        generation_dataset = build_generation_dataset(
            generation_dir,
            resolution=args.resolution,
            include_input_image=True,
            split_filter=gen_split_filter,
            require_captions=args.require_captioned_generation,
        )
        if len(generation_dataset) == 0:
            raise ValueError(f"No usable samples found in generation dataset: {generation_dir}")
        logger.info(
            "  Generation preservation data: %d samples from %s",
            len(generation_dataset),
            generation_dir,
        )

    for ds in train_vision_datasets:
        logger.info(f"  [train] {ds.dir_name}/{ds.task}: {len(ds)} samples")
    if val_dataset is not None:
        logger.info(f"  [val] Validation set: {len(val_dataset)} samples")

    if not train_vision_datasets:
        logger.error("No training datasets loaded. Check --data-dir, --pair-manifest, and --tasks")
        return

    # Task weights — resolve to a dict[str, float] for MixedVisionDataset.
    # Supported values: "sqrt" (1/sqrt(count)), "uniform" (equal), or
    # comma-separated floats matching the number of vision datasets.
    task_weights_raw = args.task_weights
    task_weights: dict[str, float] | None = None

    if task_weights_raw == "uniform":
        task_weights = None  # uniform sampling
    elif task_weights_raw in ("sqrt", None):
        # 1/sqrt(n_samples) weighting per task
        import math
        from open_vision_banana.data.mixed import _dataset_task
        task_weights = {}
        for ds in train_vision_datasets:
            task = _dataset_task(ds)
            task_weights[task] = task_weights.get(task, 0.0) + len(ds)
        task_weights = {t: 1.0 / math.sqrt(n) for t, n in task_weights.items()}
    else:
        # Parse comma-separated floats → assign to tasks in sorted order
        try:
            weights_list = [float(w.strip()) for w in task_weights_raw.split(",")]
            from open_vision_banana.data.mixed import _dataset_task
            tasks_sorted = sorted({_dataset_task(ds) for ds in train_vision_datasets})
            if len(weights_list) != len(tasks_sorted):
                logger.warning(
                    f"task_weights count ({len(weights_list)}) != "
                    f"num tasks ({len(tasks_sorted)}); using sqrt"
                )
                import math
                task_weights = {}
                for ds in train_vision_datasets:
                    task = _dataset_task(ds)
                    task_weights[task] = task_weights.get(task, 0.0) + len(ds)
                task_weights = {t: 1.0 / math.sqrt(n) for t, n in task_weights.items()}
            else:
                task_weights = dict(zip(tasks_sorted, weights_list))
        except ValueError:
            logger.warning(f"Invalid task_weights value: {task_weights_raw}; using sqrt")
            import math
            from open_vision_banana.data.mixed import _dataset_task
            task_weights = {}
            for ds in train_vision_datasets:
                task = _dataset_task(ds)
                task_weights[task] = task_weights.get(task, 0.0) + len(ds)
            task_weights = {t: 1.0 / math.sqrt(n) for t, n in task_weights.items()}

    if task_weights:
        # Normalize for display
        w_sum = sum(task_weights.values())
        w_display = {t: round(w / w_sum, 4) for t, w in task_weights.items()}
        logger.info(f"  Task weights (normalized): {w_display}")

    task_loss_weights: dict[str, float] = {}
    if args.task_loss_weights:
        for item in args.task_loss_weights.split(","):
            name, separator, value = item.partition("=")
            if not separator:
                raise ValueError(f"Invalid task loss weight {item!r}; expected task=value")
            task_loss_weights[name.strip()] = float(value)
        logger.info("  Task loss weights: %s", task_loss_weights)

    mixed_ds = MixedVisionDataset(
        vision_datasets=train_vision_datasets,
        generation_dataset=generation_dataset,
        vision_ratio=args.vision_ratio if generation_dataset else 1.0,
        seed=args.seed,
        task_weights=task_weights,
    )
    logger.info(f"  Mixed dataset total: {len(mixed_ds)} samples")

    # ── Override for tar-target-mode: check gzip index availability ──────
    _use_sequential = False
    if args.tar_target_mode:
        gzip_index_path = Path("output/snvc50m_manifest/gzip_index/stream.gzidx")
        if gzip_index_path.is_file():
            logger.info("  Pre-built gzip index found → using random access mode")
            _use_sequential = False  # Can use MixedVisionDataset with shuffle
        else:
            logger.info("  No gzip index → using sequential access mode")
            _use_sequential = True
            from open_vision_banana.data.tar_dataset import SequentialMultiTaskDataset
            task_ds_map = {}
            task_order = []
            for ds in train_vision_datasets:
                t = ds.task
                task_ds_map[t] = ds
                task_order.append(t)
            mixed_ds = SequentialMultiTaskDataset(
                task_datasets=task_ds_map,
                task_order=task_order,
            )
            logger.info("  Using SequentialMultiTaskDataset (no shuffle, sequential gzip read)")

    # ── DataLoader ───────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    # Sequential mode: no workers (shared gzip stream), no shuffle
    if _use_sequential:
        train_loader = DataLoader(
            mixed_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            collate_fn=flexible_collate,
        )
        logger.info("  Sequential DataLoader: num_workers=0, shuffle=False")
    else:
        persistent = args.num_workers > 0

        homogeneous_sampler = None
        if args.homogeneous_batches and generation_dataset is not None:
            num_batches = max(1, len(mixed_ds) // args.batch_size)
            homogeneous_sampler = HomogeneousMixedBatchSampler(
                num_batches=num_batches,
                batch_size=args.batch_size,
                vision_ratio=args.vision_ratio,
                seed=args.seed,
            )
            train_loader = DataLoader(
                mixed_ds,
                batch_sampler=homogeneous_sampler,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=persistent,
                collate_fn=flexible_collate,
            )
            logger.info(
                "  Homogeneous branch schedule: %d batches (%d vision / %d generation)",
                num_batches,
                round(num_batches * args.vision_ratio),
                num_batches - round(num_batches * args.vision_ratio),
            )
        else:
            train_loader = DataLoader(
                mixed_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=True,
                persistent_workers=persistent,
                collate_fn=flexible_collate,
            )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, 4),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=persistent,
            collate_fn=flexible_collate,
        )

    # ── Optimizer & Scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    warmup_steps = args.warmup_steps or int(args.steps * args.warmup_ratio)
    lr_scheduler = SchedulerFactory.create(
        optimizer=optimizer,
        total_steps=args.steps,
        warmup_steps=warmup_steps,
        scheduler_type=args.scheduler,
    )

    # ── Training state ───────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_fingerprint = None
    if args.pair_manifest is not None:
        manifest_path = args.pair_manifest.parent / "manifest.json"
        if manifest_path.is_file():
            try:
                dataset_fingerprint = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("dataset_fingerprint")
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read dataset fingerprint from %s", manifest_path)

    # Save config
    config_dump = {
        "model": args.model_name,
        "model_id": model_cfg.model_id,
        "data_dir": portable_project_path(data_root),
        "pair_manifest": portable_project_path(args.pair_manifest),
        "tar_index": portable_project_path(args.tar_index) if hasattr(args, 'tar_index') else None,
        "tar_root": portable_project_path(args.tar_root) if hasattr(args, 'tar_root') else None,
        "generation_dir": portable_project_path(args.generation_dir),
        "require_captioned_generation": bool(args.require_captioned_generation),
        "dataset_fingerprint": dataset_fingerprint,
        "vision_ratio": args.vision_ratio if generation_dataset else 1.0,
        "homogeneous_batches": bool(homogeneous_sampler is not None),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "rslora": not args.no_rslora,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "warmup_steps": warmup_steps,
        "warmup_ratio": args.warmup_ratio,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "resolution": args.resolution,
        "dtype": args.dtype,
        "flow_sigma": args.flow_sigma,
        "cfg_dropout_rate": args.cfg_dropout_rate,
        "depth_edge_aux_weight": args.depth_edge_aux_weight,
        "depth_edge_aux_gradient_weight": args.depth_edge_aux_gradient_weight,
        "depth_edge_aux_edge_weight": args.depth_edge_aux_edge_weight,
        "depth_edge_aux_edge_threshold": args.depth_edge_aux_edge_threshold,
        "depth_edge_aux_sigma_floor": args.depth_edge_aux_sigma_floor,
        "use_img2img_flow": not args.no_img2img_flow,
        "use_min_snr": not args.no_min_snr,
        "use_logit_normal": not args.no_logit_normal_timesteps,
        "flow_scheduler": "native_dpmsolver_flow_prediction",
        "flow_shift": float(getattr(getattr(components.scheduler, "config", None), "flow_shift", 3.0)),
        "vae_scaling_factor": float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0)),
        "ema_decay": args.ema_decay,
        "tasks": [d.task for d in train_vision_datasets],
        "task_weights": args.task_weights,
        "task_loss_weights": task_loss_weights,
        "resume": args.resume,
        "resume_weights_only": args.resume_weights_only,
        "seed": args.seed,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config_dump, indent=2, ensure_ascii=False)
    )

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    # Resume
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            logger.info(f"Resuming from {resume_path}")
            # Load checkpoint to CPU first to avoid GPU OOM, then move to device
            ckpt = torch.load(str(resume_path), map_location="cpu")
            peft_model.load_state_dict(ckpt.get("lora_state_dict", ckpt), strict=False)
            if args.resume_weights_only:
                logger.info("  Loaded LoRA weights only; optimizer, scheduler, and step count reset")
            else:
                global_step = ckpt.get("step", 0)
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "scheduler_state_dict" in ckpt:
                    lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                # Restore EMA shadow; move to the shadow tensors' device.
                ema_ckpt = ckpt.get("ema_state_dict")
                if ema_ckpt:
                    for name, tensor in ema_ckpt.items():
                        if name in ema.shadow_params:
                            ema.shadow_params[name] = tensor.to(
                                ema.shadow_params[name].device
                            )
                    ema.step = ckpt.get("step", 0)
                    logger.info("  Restored EMA shadow weights")
                logger.info(f"  Resumed at step {global_step}, best_val_loss={best_val_loss:.4f}")

    # ── Training loop ────────────────────────────────────────────────
    logger.info("Starting training...")
    model_dtype = next(peft_model.parameters()).dtype
    flow_scheduler = components.scheduler
    if flow_scheduler is None:
        raise RuntimeError("SANA pipeline did not provide its native flow scheduler")
    vae_downsample = model_cfg.vae_downsample_factor  # 32 for DC-AE
    max_length = model_cfg.text_encoder_max_length  # 300

    # Adjust for device limitations
    use_amp = device.type == "cuda" and dtype != torch.float32
    amp_dtype = torch.bfloat16 if device.type == "mps" else dtype
    scaler = torch.amp.GradScaler("cuda") if use_amp and dtype == torch.float16 else None

    train_metrics = {"loss": 0.0, "flow_loss": 0.0, "depth_edge_aux_loss": 0.0, "steps": 0}
    last_avg_loss: float | None = None
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)

    target_step = args.steps
    if args.stop_after_step is not None:
        if args.stop_after_step <= 0:
            raise ValueError("--stop-after-step must be positive")
        target_step = min(target_step, args.stop_after_step)

    while global_step < target_step:
        updates_per_epoch = max(1, (len(train_loader) + args.grad_accum - 1) // args.grad_accum)
        epoch = start_epoch + (global_step // updates_per_epoch)
        mixed_ds.set_epoch(epoch)
        if homogeneous_sampler is not None:
            homogeneous_sampler.set_epoch(epoch)
        peft_model.train()

        for batch in train_loader:
            if global_step >= target_step:
                break

            prompts = batch.get("prompt", [""] * args.batch_size)
            if isinstance(prompts, str):
                prompts = [prompts]

            # Load images and targets
            # image: [B, 3, H, W] in [-1, 1] (input image) — may be absent in target-only mode
            # target_rgb: [B, 3, H, W] in [0, 1] (target visualization)
            target_rgb = batch["target_rgb"].to(device=device, dtype=model_dtype)

            # Sana expects images in [-1, 1] for VAE encode
            target_images = target_rgb * 2.0 - 1.0  # [0,1] → [-1,1]

            # ── Encode target to latents ──────────────────────────────
            with torch.no_grad():
                target_latents = vae_encode_sana(vae, target_images, device, model_dtype)

                # Input images may be absent (target-only mode) or unused (no-img2img-flow)
                input_latents = None
                if "image" in batch and not args.no_img2img_flow:
                    input_images = batch["image"].to(device=device, dtype=model_dtype)
                    input_latents = vae_encode_sana(vae, input_images, device, model_dtype)
                    # Ensure same spatial dims
                    if input_latents.shape != target_latents.shape:
                        input_latents = F.adaptive_avg_pool2d(
                            input_latents, target_latents.shape[-2:]
                        )

            batch_size = target_latents.shape[0]

            # ── Native SANA flow matching ─────────────────────────────
            sigma, timesteps = sample_sana_flow_timesteps(
                batch_size,
                device,
                flow_scheduler,
                logit_normal=not args.no_logit_normal_timesteps,
            )
            if not args.no_img2img_flow and input_latents is not None:
                source_latents = select_sana_flow_sources(
                    input_latents,
                    target_latents,
                    batch.get("flow_source", ["input"] * batch_size),
                )
            else:
                source_latents = torch.randn_like(target_latents)
            noisy_latents, velocity_target = prepare_sana_flow_batch(
                source_latents,
                target_latents,
                sigma,
                noise_scale=args.flow_sigma,
            )

            # ── Text encoding ─────────────────────────────────────────
            with torch.no_grad():
                prompt_embeds, attention_mask = encode_text_sana(
                    components, prompts, device, model_dtype, max_length
                )

                # CFG dropout
                if args.cfg_dropout_rate > 0:
                    drop_mask = torch.rand(batch_size, device=device) < args.cfg_dropout_rate
                    if drop_mask.any():
                        uncond_embeds, uncond_mask = encode_text_sana(
                            components, [""] * batch_size, device, model_dtype, max_length
                        )
                        drop = drop_mask.view(-1, *([1] * (prompt_embeds.dim() - 1)))
                        prompt_embeds = torch.where(drop, uncond_embeds, prompt_embeds)
                        if attention_mask is not None:
                            drop_mask_2d = drop_mask.view(-1, 1)
                            attention_mask = torch.where(drop_mask_2d, uncond_mask, attention_mask)

            # ── Forward pass ──────────────────────────────────────────
            autocast_ctx = torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp or device.type == "mps",
            )

            with autocast_ctx:
                predicted_velocity = peft_model(
                    noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=attention_mask,
                    timestep=native_sana_timestep(timesteps, batch_size, peft_model),
                    return_dict=False,
                )[0]

                sq_error = (predicted_velocity.float() - velocity_target.float()).square()
                per_sample_loss = sq_error.flatten(1).mean(dim=1)
                per_sample_loss = apply_task_loss_weights(
                    per_sample_loss, batch.get("task"), task_loss_weights
                )
                flow_loss = per_sample_loss.mean()

                if not args.no_min_snr:
                    sample_contrib = per_sample_loss * flow_min_snr_weight(sigma, args.min_snr_gamma)
                    flow_loss = sample_contrib.mean()
                else:
                    sample_contrib = per_sample_loss

                # Per-task loss accounting (observability only; does not change optimization).
                batch_tasks = batch.get("task")
                if batch_tasks is not None:
                    task_loss_acc = train_metrics.setdefault("task_loss", {})
                    for _t, _c in zip(batch_tasks, sample_contrib.detach().float().tolist()):
                        _ent = task_loss_acc.setdefault(_t, [0.0, 0])
                        _ent[0] += float(_c)
                        _ent[1] += 1

            depth_aux_loss = flow_loss.new_zeros(())
            if args.depth_edge_aux_weight:
                depth_mask = depth_task_mask(batch.get("task"), batch_size, device)
                if depth_mask.any():
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        depth_aux_loss = depth_edge_endpoint_loss(
                            predicted_velocity[depth_mask],
                            noisy_latents[depth_mask],
                            target_latents[depth_mask],
                            target_rgb[depth_mask],
                            sigma[depth_mask],
                            edge_threshold=args.depth_edge_aux_edge_threshold,
                            edge_weight=args.depth_edge_aux_edge_weight,
                            gradient_weight=args.depth_edge_aux_gradient_weight,
                            sigma_floor=args.depth_edge_aux_sigma_floor,
                        )
            total_loss = flow_loss.float() + args.depth_edge_aux_weight * depth_aux_loss
            if not torch.isfinite(total_loss).all():
                raise FloatingPointError(
                    f"Non-finite training loss at global_step={global_step}, "
                    f"tasks={batch.get('task')}"
                )
            loss = total_loss / args.grad_accum

            # ── Backward ──────────────────────────────────────────────
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_metrics["loss"] += total_loss.item()
            train_metrics["flow_loss"] += flow_loss.item()
            train_metrics["depth_edge_aux_loss"] += depth_aux_loss.item()
            train_metrics["steps"] += 1

            # ``global_step`` is an optimizer update, never a micro-batch.
            micro_step += 1
            if micro_step % args.grad_accum != 0:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(peft_model)
            global_step += 1

            # ── Logging ──────────────────────────────────────────────
            if global_step % args.log_every == 0:
                avg_loss = train_metrics["loss"] / max(1, train_metrics["steps"])
                avg_flow_loss = train_metrics["flow_loss"] / max(1, train_metrics["steps"])
                avg_depth_aux_loss = train_metrics["depth_edge_aux_loss"] / max(1, train_metrics["steps"])
                last_avg_loss = avg_loss
                lr = lr_scheduler.get_last_lr()[0]
                # Per-task mean flow loss over this logging window (diagnostics).
                task_loss_acc = train_metrics.get("task_loss", {})
                task_str = " ".join(
                    f"{t}={v[0] / max(1, v[1]):.4f}" for t, v in sorted(task_loss_acc.items())
                )
                logger.info(
                    f"  [step {global_step}/{args.steps}] "
                    f"loss={avg_loss:.4f} flow={avg_flow_loss:.4f} edge_aux={avg_depth_aux_loss:.4f} lr={lr:.2e} "
                    f"epoch={epoch} "
                    f"latent_shape={list(target_latents.shape[1:])} | task_loss: {task_str}"
                )
                train_metrics = {"loss": 0.0, "flow_loss": 0.0, "depth_edge_aux_loss": 0.0, "steps": 0}

            # ── Save checkpoint ──────────────────────────────────────
            if global_step % args.save_every == 0:
                save_checkpoint(
                    output_dir,
                    peft_model,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    last_avg_loss,
                    best_val_loss,
                    args,
                    ema=ema,
                )
                logger.info(f"  Saved checkpoint at step {global_step}")

            # ── Validation (on EMA shadow weights) ────────────────────
            if val_loader is not None and global_step % args.eval_every == 0:
                ema_backup = ema.apply_shadow(peft_model)
                try:
                    val_loss = validate(
                        peft_model,
                        vae,
                        components,
                        val_loader,
                        device,
                        model_dtype,
                        args,
                        vae_downsample,
                        max_length,
                        task_loss_weights,
                    )
                finally:
                    ema.restore_shadow(peft_model, ema_backup)
                logger.info(f"  [val] step={global_step} val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        output_dir,
                        peft_model,
                        optimizer,
                        lr_scheduler,
                        global_step,
                        val_loss,
                        best_val_loss,
                        args,
                        suffix="_best",
                        ema=ema,
                    )
                    logger.info(f"  New best val_loss={best_val_loss:.4f}")

    if args.stop_after_step is not None and global_step >= target_step and target_step < args.steps:
        logger.info("Segment stop reached at step %d/%d; regular checkpoint retained", global_step, args.steps)
        return

    # ── Final save ──────────────────────────────────────────────────
    save_checkpoint(
        output_dir,
        peft_model,
        optimizer,
        lr_scheduler,
        global_step,
        None,
        best_val_loss,
        args,
        suffix="_final",
        ema=ema,
    )
    logger.info("Training complete!")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Best val_loss: {best_val_loss:.4f}")
    logger.info(f"  Checkpoints saved to: {output_dir}")


def save_checkpoint(
    output_dir: Path,
    peft_model,
    optimizer,
    lr_scheduler,
    step: int,
    val_loss: float | None,
    best_val_loss: float,
    args,
    suffix: str = "",
    ema=None,
):
    """Save LoRA checkpoint with training state.

    ``lora.pt`` keeps online weights for resume plus ``ema_state_dict``.
    ``lora_adapter/`` holds EMA shadow weights (what viz/eval should load);
    online weights go to ``lora_adapter_online/``.
    """
    ckpt_dir = output_dir / f"checkpoint-{step}{suffix}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter weights
    lora_state = {}
    for name, param in peft_model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu().clone()

    ckpt = {
        "lora_state_dict": lora_state,
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
    }

    has_ema = ema is not None and getattr(ema, "shadow_params", None)
    if has_ema:
        # EMA shadow saved for resume/inspection; lora.pt stays online-weighted.
        ckpt["ema_state_dict"] = {
            name: tensor.cpu().clone() for name, tensor in ema.shadow_params.items()
        }

    torch.save(ckpt, str(ckpt_dir / "lora.pt"))

    # lora_adapter = inference-facing weights; prefer EMA when available.
    if has_ema:
        backup = ema.apply_shadow(peft_model)
        try:
            peft_model.save_pretrained(str(ckpt_dir / "lora_adapter"))
            logger.info(f"  LoRA adapter (EMA) saved to {ckpt_dir / 'lora_adapter'}")
        finally:
            ema.restore_shadow(peft_model, backup)
        try:
            peft_model.save_pretrained(str(ckpt_dir / "lora_adapter_online"))
        except Exception as e:
            logger.warning(f"  Could not save online LoRA adapter: {e}")
    else:
        try:
            peft_model.save_pretrained(str(ckpt_dir / "lora_adapter"))
            logger.info(f"  LoRA adapter saved to {ckpt_dir / 'lora_adapter'}")
        except Exception as e:
            logger.warning(f"  Could not save LoRA adapter via save_pretrained: {e}")

    # Save metadata
    meta = {
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "model_name": args.model_name,
        "ema_decay": getattr(args, "ema_decay", None),
        "adapter_dir": "lora_adapter (EMA shadow)" if has_ema else "lora_adapter (online)",
    }
    (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


@torch.no_grad()
def validate(
    peft_model,
    vae,
    components: PipelineComponents,
    val_loader,
    device: torch.device,
    model_dtype: torch.dtype,
    args,
    vae_downsample: int,
    max_length: int,
    task_loss_weights: dict[str, float] | None = None,
) -> float:
    """Run validation and return average loss."""
    peft_model.eval()
    total_loss = 0.0
    num_batches = 0
    max_val_batches = 20  # Limit val batches for speed

    # Match training autocast settings to avoid fp16 overflow on MPS
    amp_dtype = torch.bfloat16 if device.type == "mps" else model_dtype
    use_autocast = device.type == "mps" or (device.type == "cuda" and model_dtype != torch.float32)
    flow_scheduler = components.scheduler
    if flow_scheduler is None:
        raise RuntimeError("SANA pipeline did not provide its native flow scheduler")

    for i, batch in enumerate(val_loader):
        if i >= max_val_batches:
            break

        prompts = batch.get("prompt", [""])
        if isinstance(prompts, str):
            prompts = [prompts]

        target_rgb = batch["target_rgb"].to(device=device, dtype=model_dtype)
        target_images = target_rgb * 2.0 - 1.0

        target_latents = vae_encode_sana(vae, target_images, device, model_dtype)

        input_latents = None
        if not args.no_img2img_flow:
            input_images = batch["image"].to(device=device, dtype=model_dtype)
            input_latents = vae_encode_sana(vae, input_images, device, model_dtype)
            if input_latents.shape != target_latents.shape:
                input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])

        batch_size = target_latents.shape[0]

        sigma, timesteps = sample_sana_flow_timesteps(
            batch_size,
            device,
            flow_scheduler,
            logit_normal=not args.no_logit_normal_timesteps,
        )
        if not args.no_img2img_flow and input_latents is not None:
            source_latents = select_sana_flow_sources(
                input_latents,
                target_latents,
                batch.get("flow_source", ["input"] * batch_size),
            )
        else:
            source_latents = torch.randn_like(target_latents)
        noisy_latents, velocity_target = prepare_sana_flow_batch(
            source_latents,
            target_latents,
            sigma,
            noise_scale=args.flow_sigma,
        )

        prompt_embeds, attention_mask = encode_text_sana(
            components, prompts, device, model_dtype, max_length
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_autocast,
        ):
            predicted_velocity = peft_model(
                noisy_latents,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=attention_mask,
                timestep=native_sana_timestep(timesteps, batch_size, peft_model),
                return_dict=False,
            )[0]

        sq_error = (predicted_velocity.float() - velocity_target.float()).square()
        per_sample_loss = sq_error.flatten(1).mean(dim=1)
        per_sample_loss = apply_task_loss_weights(
            per_sample_loss, batch.get("task"), task_loss_weights
        )
        flow_loss = per_sample_loss.mean()
        if not args.no_min_snr:
            flow_loss = (per_sample_loss * flow_min_snr_weight(sigma, args.min_snr_gamma)).mean()

        depth_aux_loss = flow_loss.new_zeros(())
        if args.depth_edge_aux_weight:
            depth_mask = depth_task_mask(batch.get("task"), batch_size, device)
            if depth_mask.any():
                depth_aux_loss = depth_edge_endpoint_loss(
                    predicted_velocity[depth_mask],
                    noisy_latents[depth_mask],
                    target_latents[depth_mask],
                    target_rgb[depth_mask],
                    sigma[depth_mask],
                    edge_threshold=args.depth_edge_aux_edge_threshold,
                    edge_weight=args.depth_edge_aux_edge_weight,
                    gradient_weight=args.depth_edge_aux_gradient_weight,
                    sigma_floor=args.depth_edge_aux_sigma_floor,
                )
        loss = flow_loss.float() + args.depth_edge_aux_weight * depth_aux_loss

        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite validation loss at batch {i}")

        total_loss += loss.item()
        num_batches += 1

    peft_model.train()
    return total_loss / max(1, num_batches)


if __name__ == "__main__":
    main()
