#!/usr/bin/env python3
"""Overfit SSD-1B on tiny v3 depth data with an explicit source condition.

This diagnostic uses an InstructPix2Pix-style UNet input:

    [noisy target latent, source image latent] -> target noise

The target is noised with the native SSD/SDXL epsilon scheduler, while the
source latent remains available at every denoising step. This makes the gate a
real source-to-target conditional task without requiring an external ControlNet.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.conditioning import (  # noqa: E402
    MULTI_STEP_EPSILON,
    SINGLE_STEP_X0,
    SourceConditioningConfig,
    calibrate_source_conditioned_output,
    read_source_conditioning_config,
    write_source_conditioning_config,
)
from open_vision_banana.data.depth_encoding import (  # noqa: E402
    DepthEncodingAudit,
    audit_depth_target_encoding,
    relative_depth_prompt,
)
from open_vision_banana.evaluation.relative_depth import (  # noqa: E402
    evaluate_relative_depth_rgb,
)
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.model_registry import vae_decode  # noqa: E402
from open_vision_banana.training.tuner import (  # noqa: E402
    SDXL_CONV_LORA_TARGET_MODULES,
    InstructionTuner,
)

logger = logging.getLogger("quick_overfit_ssd_source_condition")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def expand_unet_input(unet: nn.Module, condition_channels: int = 4) -> nn.Conv2d:
    """Expand UNet conv_in and zero-initialize the new condition channels."""
    old = unet.conv_in
    if not isinstance(old, nn.Conv2d):
        raise TypeError(f"Expected Conv2d conv_in, got {type(old).__name__}")
    if condition_channels <= 0:
        raise ValueError("condition_channels must be positive")

    expanded = nn.Conv2d(
        in_channels=old.in_channels + condition_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, : old.in_channels].copy_(old.weight)
        if old.bias is not None:
            expanded.bias.copy_(old.bias)
    unet.conv_in = expanded
    if hasattr(unet, "register_to_config"):
        unet.register_to_config(in_channels=expanded.in_channels)
    return expanded


def build_conditioned_model_input(
    scheduler,
    target_latent: torch.Tensor,
    source_latent: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Noise the target, scale it for Euler, then append the clean source."""
    if target_latent.shape != source_latent.shape or noise.shape != target_latent.shape:
        raise ValueError("target, source, and noise latents must have identical shapes")
    noisy_target = scheduler.add_noise(target_latent, noise, timesteps)
    sigma = sigma_for_timesteps(scheduler, timesteps, noisy_target)
    scaled_target = noisy_target / torch.sqrt(sigma.square() + 1.0)
    return torch.cat((scaled_target, source_latent), dim=1)


def build_x0_model_input(
    scheduler,
    source_latent: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Build the deterministic x0-regression input: noised source + clean source.

    Unlike the epsilon regime, the target latent never enters the model input,
    so training and single-step inference see the identical distribution.
    """
    if noise.shape != source_latent.shape:
        raise ValueError("source and noise latents must have identical shapes")
    noisy_source = scheduler.add_noise(source_latent, noise, timesteps)
    sigma = sigma_for_timesteps(scheduler, timesteps, noisy_source)
    scaled_source = noisy_source / torch.sqrt(sigma.square() + 1.0)
    return torch.cat((scaled_source, source_latent), dim=1)


def sigma_for_timesteps(
    scheduler, timesteps: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """Return broadcast Euler sigmas corresponding to a timestep batch."""
    indices = [
        scheduler.index_for_timestep(timestep, scheduler.timesteps)
        for timestep in timesteps
    ]
    sigma = scheduler.sigmas.to(device=reference.device, dtype=reference.dtype)[indices]
    return sigma.view(reference.shape[0], *([1] * (reference.ndim - 1)))


def repeat_final_timestep(timesteps: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat the lowest-noise inference timestep for final-color refinement."""
    if timesteps.numel() == 0:
        raise ValueError("timesteps must be non-empty")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if repeats == 1:
        return timesteps
    return torch.cat((timesteps, timesteps[-1:].repeat(repeats - 1)))


def training_schedule_indices(
    global_step: int,
    *,
    sample_count: int,
    timestep_count: int,
) -> tuple[int, int]:
    """Map a cumulative step to its sample and denoising-timestep slots."""
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if sample_count <= 0 or timestep_count <= 0:
        raise ValueError("sample_count and timestep_count must be positive")
    return (
        global_step % sample_count,
        (global_step // sample_count) % timestep_count,
    )


def infer_completed_steps(metrics: dict | None) -> int:
    """Read cumulative steps from current or legacy gate metrics."""
    if not metrics:
        return 0
    training_steps = metrics.get("training_steps")
    if isinstance(training_steps, dict):
        cumulative = training_steps.get("cumulative")
        if isinstance(cumulative, int) and cumulative >= 0:
            return cumulative
    config = metrics.get("config")
    if isinstance(config, dict):
        legacy_steps = config.get("max_steps")
        if isinstance(legacy_steps, int) and legacy_steps >= 0:
            return legacy_steps
    loss = metrics.get("loss")
    if isinstance(loss, dict) and isinstance(loss.get("history"), list):
        return len(loss["history"])
    return 0


def find_depth_pairs(
    data_dir: Path,
    count: int,
    *,
    allowed_stems: set[str] | None = None,
) -> list[tuple[Path, Path]]:
    task_dir = data_dir / "depth"
    image_dir = task_dir / "images"
    target_dir = task_dir / "targets"
    if not image_dir.is_dir() or not target_dir.is_dir():
        raise NotADirectoryError(f"Expected depth/images and depth/targets under {data_dir}")
    targets = {
        path.stem: path
        for path in sorted(target_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    }
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if allowed_stems is not None and image_path.stem not in allowed_stems:
            continue
        target_path = targets.get(image_path.stem)
        if target_path is not None:
            pairs.append((image_path, target_path))
        if len(pairs) == count:
            break
    if len(pairs) < count:
        raise ValueError(f"Requested {count} depth pairs, found {len(pairs)}")
    return pairs


def manifest_depth_stems(data_dir: Path, split: Literal["train", "val"]) -> set[str]:
    """Load declared depth sample stems from the dataset split manifest."""
    manifest_path = data_dir / "train_val_split.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split-clean gate requires train_val_split.json: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = payload.get(split, {}).get("depth") if isinstance(payload, dict) else None
    if not isinstance(names, list) or not names:
        raise ValueError(f"Manifest has no non-empty {split}.depth list: {manifest_path}")
    return {Path(str(name)).stem for name in names}


def find_split_clean_depth_pairs(
    data_dir: Path,
    *,
    train_count: int,
    eval_count: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Select disjoint train/val pairs exactly as declared by the manifest."""
    train_stems = manifest_depth_stems(data_dir, "train")
    eval_stems = manifest_depth_stems(data_dir, "val")
    overlap = train_stems & eval_stems
    if overlap:
        examples = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Depth train/val manifest overlap detected: {examples}")
    train_pairs = find_depth_pairs(
        data_dir,
        train_count,
        allowed_stems=train_stems,
    )
    eval_pairs = find_depth_pairs(
        data_dir,
        eval_count,
        allowed_stems=eval_stems,
    )
    selected_overlap = {
        source_path.stem for source_path, _ in train_pairs
    } & {source_path.stem for source_path, _ in eval_pairs}
    if selected_overlap:
        raise RuntimeError("Selected train/eval depth pairs are not disjoint")
    return train_pairs, eval_pairs


def audit_depth_pairs(
    data_dir: Path,
    pairs: list[tuple[Path, Path]],
) -> tuple[DepthEncodingAudit, str]:
    """Audit selected targets and return the only prompt valid for that encoding."""
    if not pairs:
        raise ValueError("pairs must be non-empty")
    audit = audit_depth_target_encoding(
        data_dir / "depth" / "targets",
        max_images=min(16, len(pairs)),
        allowed_names={target_path.name for _, target_path in pairs},
    )
    if not audit.is_curve_consistent:
        raise ValueError(
            "Selected depth targets do not consistently match a registered "
            f"colormap (best={audit.detected_colormap!r}, "
            f"curve_mae={audit.mean_curve_mae:.5f})"
        )
    return audit, relative_depth_prompt(audit.detected_colormap)


def load_tensor(path: Path, resolution: int, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = (tensor / 2.0 + 0.5).clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a CPU [1, 3, H, W] tensor in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


def unit_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] or [3, H, W] tensor in [0, 1] to PIL."""
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = (
        tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def calibrate_depth_prediction(
    prediction: Image.Image,
    audit: DepthEncodingAudit,
    *,
    mode: Literal["moments", "moments_and_project"],
) -> Image.Image:
    """Apply the production source-conditioned depth calibration contract."""
    config = SourceConditioningConfig(
        target_colormap=audit.detected_colormap,
        target_channel_mean=audit.channel_mean,
        target_channel_std=audit.channel_std,
        output_calibration=mode,
    )
    calibrated = calibrate_source_conditioned_output(
        pil_to_unit_tensor(prediction),
        config,
    )
    return unit_tensor_to_pil(calibrated)


def image_metrics(
    prediction: Image.Image,
    target: Image.Image,
    *,
    colormap: str,
) -> dict[str, float]:
    """Measure raw RGB fidelity and relative position on the audited curve."""
    pred = np.asarray(prediction, dtype=np.float32) / 255.0
    tgt = np.asarray(target, dtype=np.float32) / 255.0
    metrics = evaluate_relative_depth_rgb(
        pil_to_unit_tensor(prediction),
        pil_to_unit_tensor(target),
        colormap=colormap,
    )
    metrics["rgb_mse"] = float(np.mean((pred - tgt) ** 2))
    return metrics


def summarize_sample_metrics(
    samples: list[dict],
    metric_group: str,
) -> dict[str, float]:
    """Average one metric group across a non-empty evaluation split."""
    if not samples:
        raise ValueError("samples must be non-empty")
    metric_names = tuple(samples[0][metric_group])
    return {
        key: float(np.mean([sample[metric_group][key] for sample in samples]))
        for key in metric_names
    }


def gate_passed(
    metrics: dict[str, float],
    *,
    max_mae: float,
    min_correlation: float,
) -> bool:
    """Apply the RGB/relative-depth gate to one aggregate metric group."""
    return (
        metrics["rgb_mae"] < max_mae
        and metrics["relative_position_pearson_r"] > min_correlation
    )


def make_comparison_grid(
    rows: list[tuple[Image.Image, ...]],
    output_path: Path,
    *,
    labels: tuple[str, ...],
) -> None:
    if not rows or not labels or any(len(row) != len(labels) for row in rows):
        raise ValueError("rows and labels must define a non-empty rectangular grid")
    cell = rows[0][0].width
    header = 30
    canvas = Image.new(
        "RGB",
        (cell * len(labels), header + cell * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * cell + 8, 8), label, fill="black")
    for row_index, row in enumerate(rows):
        y = header + row_index * cell
        for column, image in enumerate(row):
            canvas.paste(image, (column * cell, y))
    canvas.save(output_path)


def get_underlying_unet(tuner: InstructionTuner) -> nn.Module:
    model = tuner.peft_model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


@torch.no_grad()
def encode_pairs(
    tuner: InstructionTuner,
    pairs: list[tuple[Path, Path]],
    resolution: int,
    device: torch.device,
) -> list[dict]:
    encoded = []
    for source_path, target_path in pairs:
        source_pixels = load_tensor(source_path, resolution, device)
        target_pixels = load_tensor(target_path, resolution, device)
        encoded.append(
            {
                "source_path": source_path,
                "target_path": target_path,
                "source_pixels": source_pixels.cpu(),
                "target_pixels": target_pixels.cpu(),
                "source_latent": tuner._encode_latents(source_pixels).detach(),
                "target_latent": tuner._encode_latents(target_pixels).detach(),
            }
        )
    return encoded


@torch.no_grad()
def predict(
    tuner: InstructionTuner,
    source_latent: torch.Tensor,
    prompt_condition: dict,
    inference_steps: int,
    seed: int,
    *,
    prediction_target: Literal["epsilon", "x0"] = "epsilon",
    eval_timestep_index: int = -1,
) -> Image.Image:
    pipe = tuner.base_model._t2i_pipe
    device = source_latent.device
    model_dtype = next(tuner.peft_model.parameters()).dtype
    scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    scheduler.register_to_config(
        prediction_type="sample" if prediction_target == "x0" else "epsilon"
    )
    scheduler.set_timesteps(inference_steps, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    tuner.peft_model.eval()

    if prediction_target == "x0":
        # Single-step deterministic regression: one forward at the deployed
        # timestep, take the UNet output as x0, no scheduler iteration.
        timestep = scheduler.timesteps[eval_timestep_index]
        timestep_batch = timestep.reshape(1).to(device)
        noise = torch.randn(
            source_latent.shape,
            generator=generator,
            device=device,
            dtype=model_dtype,
        )
        model_input = build_x0_model_input(
            scheduler,
            source_latent,
            noise,
            timestep_batch,
        )
        latents = tuner._forward_registry(
            model_input,
            timestep_batch,
            prompt_condition["prompt_embeds"],
            prompt_condition["pooled_prompt_embeds"],
            prompt_condition.get("attention_mask"),
            model_dtype,
            device,
        )
    else:
        latents = torch.randn(
            source_latent.shape,
            generator=generator,
            device=device,
            dtype=model_dtype,
        ) * scheduler.init_noise_sigma
        for timestep in scheduler.timesteps:
            scaled_target = scheduler.scale_model_input(latents, timestep)
            model_input = torch.cat((scaled_target, source_latent), dim=1)
            timestep_batch = timestep.reshape(1).to(device)
            predicted_noise = tuner._forward_registry(
                model_input,
                timestep_batch,
                prompt_condition["prompt_embeds"],
                prompt_condition["pooled_prompt_embeds"],
                prompt_condition.get("attention_mask"),
                model_dtype,
                device,
            )
            latents = scheduler.step(predicted_noise, timestep, latents).prev_sample

    vae = pipe.vae
    decoded = vae_decode(
        vae,
        latents.to(device=next(vae.parameters()).device, dtype=next(vae.parameters()).dtype),
        tuner.model_config,
    )
    return tensor_to_pil(decoded)


@torch.no_grad()
def probe_per_timestep(
    tuner: InstructionTuner,
    encoded_pairs: list[dict],
    prompt_conditions: list[dict],
    *,
    prediction_target: Literal["epsilon", "x0"],
    inference_steps: int,
    seed: int,
    colormap: str,
) -> list[dict]:
    """Measure single-forward implied-x0 fidelity at every deployed timestep.

    For epsilon the target channels hold pure noise (the inference-time
    distribution); for x0 they hold the noised source latent. The resulting
    per-timestep Pearson r curve shows at which noise level source-to-target
    structure survives a single UNet forward.
    """
    if len(encoded_pairs) != len(prompt_conditions):
        raise ValueError("encoded pairs and prompt conditions must have equal length")
    pipe = tuner.base_model._t2i_pipe
    vae = pipe.vae
    device = encoded_pairs[0]["source_latent"].device
    model_dtype = next(tuner.peft_model.parameters()).dtype
    scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
    scheduler.set_timesteps(inference_steps, device=device)
    tuner.peft_model.eval()

    results = []
    for timestep_index, timestep in enumerate(scheduler.timesteps):
        timestep_batch = timestep.reshape(1).to(device)
        per_sample = []
        for index, sample in enumerate(encoded_pairs):
            source_latent = sample["source_latent"].to(device=device, dtype=model_dtype)
            generator = torch.Generator(device=device).manual_seed(seed + index)
            noise = torch.randn(
                source_latent.shape,
                generator=generator,
                device=device,
                dtype=model_dtype,
            )
            base_latent = (
                source_latent
                if prediction_target == "x0"
                else torch.zeros_like(source_latent)
            )
            noisy = scheduler.add_noise(base_latent, noise, timestep_batch)
            sigma = sigma_for_timesteps(scheduler, timestep_batch, noisy)
            scaled = noisy / torch.sqrt(sigma.square() + 1.0)
            model_input = torch.cat((scaled, source_latent), dim=1)
            prediction = tuner._forward_registry(
                model_input,
                timestep_batch,
                prompt_conditions[index]["prompt_embeds"],
                prompt_conditions[index]["pooled_prompt_embeds"],
                prompt_conditions[index].get("attention_mask"),
                model_dtype,
                device,
            )
            implied_x0 = (
                prediction if prediction_target == "x0" else noisy - sigma * prediction
            )
            decoded = vae_decode(
                vae,
                implied_x0.to(
                    device=next(vae.parameters()).device,
                    dtype=next(vae.parameters()).dtype,
                ),
                tuner.model_config,
            )
            per_sample.append(
                image_metrics(
                    tensor_to_pil(decoded),
                    tensor_to_pil(sample["target_pixels"]),
                    colormap=colormap,
                )
            )
        aggregated = {
            key: float(np.mean([metrics[key] for metrics in per_sample]))
            for key in per_sample[0]
        }
        results.append(
            {
                "timestep_index": timestep_index,
                "timestep": int(timestep),
                "sigma": float(scheduler.sigmas[timestep_index]),
                **aggregated,
            }
        )
    return results


@torch.no_grad()
def evaluate_encoded_pairs(
    tuner: InstructionTuner,
    encoded_pairs: list[dict],
    prompt_conditions: list[dict],
    audit: DepthEncodingAudit,
    *,
    inference_steps: int,
    seed: int,
    output_calibration: Literal["moments", "moments_and_project"],
    prediction_target: Literal["epsilon", "x0"] = "epsilon",
    eval_timestep_index: int = -1,
    before_predictions: list[Image.Image] | None = None,
) -> tuple[list[dict], list[tuple[Image.Image, ...]], dict[str, dict[str, float]]]:
    """Evaluate a split and retain images for its comparison grid."""
    if len(encoded_pairs) != len(prompt_conditions):
        raise ValueError("encoded pairs and prompt conditions must have equal length")
    if before_predictions is not None and len(before_predictions) != len(encoded_pairs):
        raise ValueError("before predictions must match the evaluated split")

    rows = []
    samples = []
    for index, sample in enumerate(encoded_pairs):
        prediction = predict(
            tuner,
            sample["source_latent"],
            prompt_conditions[index],
            inference_steps,
            seed + index,
            prediction_target=prediction_target,
            eval_timestep_index=eval_timestep_index,
        )
        calibrated_prediction = calibrate_depth_prediction(
            prediction,
            audit,
            mode=output_calibration,
        )
        source_image = tensor_to_pil(sample["source_pixels"])
        target_image = tensor_to_pil(sample["target_pixels"])
        sample_metrics = {
            "sample": sample["source_path"].name,
            "after": image_metrics(
                prediction,
                target_image,
                colormap=audit.detected_colormap,
            ),
            "after_calibrated": image_metrics(
                calibrated_prediction,
                target_image,
                colormap=audit.detected_colormap,
            ),
        }
        row = (source_image, prediction, calibrated_prediction, target_image)
        if before_predictions is not None:
            sample_metrics["before"] = image_metrics(
                before_predictions[index],
                target_image,
                colormap=audit.detected_colormap,
            )
            row = (
                source_image,
                before_predictions[index],
                prediction,
                calibrated_prediction,
                target_image,
            )
        samples.append(sample_metrics)
        rows.append(row)

    summaries = {
        "after": summarize_sample_metrics(samples, "after"),
        "after_calibrated": summarize_sample_metrics(samples, "after_calibrated"),
    }
    if before_predictions is not None:
        summaries["before"] = summarize_sample_metrics(samples, "before")
    return samples, rows, summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Previous gate output containing adapter/ and conv_in.pt.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Load --resume-dir and evaluate train/val without updating weights.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of manifest train.depth samples.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=4,
        help="Number of independent manifest val.depth samples.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument(
        "--prediction-target",
        choices=["epsilon", "x0"],
        default="epsilon",
        help=(
            "epsilon: legacy noisy-target/iterative-sampler regime. "
            "x0: deterministic regression on the noised source with "
            "single-step x0 inference (train/infer distributions match)."
        ),
    )
    parser.add_argument(
        "--eval-timestep-index",
        type=int,
        default=-1,
        help="Deployed-schedule index used by single-step x0 inference.",
    )
    parser.add_argument(
        "--lora-conv-targets",
        action="store_true",
        help="Extend LoRA to the UNet ResNet/resampling convolutions.",
    )
    parser.add_argument(
        "--final-timestep-repeats",
        type=int,
        default=1,
        help="Oversample the lowest-noise timestep to refine target colors.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--x0-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary implied-x0 loss weight; emphasizes high-noise accuracy.",
    )
    parser.add_argument(
        "--pixel-loss-weight",
        type=float,
        default=0.0,
        help="Frozen-VAE decoded pixel L1 loss weight for colour calibration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--gate-mae", type=float, default=0.10)
    parser.add_argument("--gate-correlation", type=float, default=0.70)
    parser.add_argument(
        "--output-calibration",
        choices=["moments", "moments_and_project"],
        default="moments_and_project",
        help="Calibration used for the production-equivalent gate result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.final_timestep_repeats <= 0:
        raise ValueError(
            "--num-samples and --final-timestep-repeats must be positive"
        )
    if not args.eval_only and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive when training")
    if args.eval_only and args.resume_dir is None:
        raise ValueError("--eval-only requires --resume-dir")
    if args.x0_loss_weight < 0 or args.pixel_loss_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if args.prediction_target == "x0" and args.x0_loss_weight > 0:
        raise ValueError(
            "--x0-loss-weight is an epsilon-regime auxiliary; the x0 target "
            "already regresses the clean latent directly"
        )
    if not -args.inference_steps <= args.eval_timestep_index < args.inference_steps:
        raise ValueError(
            "--eval-timestep-index must index the deployed schedule "
            f"[-{args.inference_steps}, {args.inference_steps})"
        )
    eval_timestep_index = args.eval_timestep_index % args.inference_steps
    if args.eval_samples <= 0:
        raise ValueError("--eval-samples must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_pairs, eval_pairs = find_split_clean_depth_pairs(
        data_dir,
        train_count=args.num_samples,
        eval_count=args.eval_samples,
    )
    logger.info(
        "Using split-clean v3 depth pairs from %s: train=%d val=%d",
        data_dir,
        len(train_pairs),
        len(eval_pairs),
    )
    depth_audit, depth_prompt = audit_depth_pairs(data_dir, train_pairs)
    eval_depth_audit, _ = audit_depth_pairs(data_dir, eval_pairs)
    if eval_depth_audit.detected_colormap != depth_audit.detected_colormap:
        raise ValueError(
            "Train/val depth colormap mismatch: "
            f"train={depth_audit.detected_colormap}, "
            f"val={eval_depth_audit.detected_colormap}"
        )
    logger.info(
        "Depth target encoding audit: colormap=%s curve_mae=%.5f samples=%d",
        depth_audit.detected_colormap,
        depth_audit.mean_curve_mae,
        depth_audit.sample_count,
    )
    logger.info("Using audited depth prompt: %s", depth_prompt)

    started = time.time()
    # MPS 上 bf16 不稳定；SSD-1B 足够小，fp32 可行
    _dtype = torch.float32 if device.type == "mps" else torch.bfloat16
    logger.info("Using dtype=%s on device=%s", _dtype, device)
    generator = BaseGenerator(
        model_name="segmind/SSD-1B",
        device=str(device),
        dtype=_dtype,
    )
    unet = generator._t2i_pipe.unet
    original_channels = unet.conv_in.in_channels
    expand_unet_input(unet, condition_channels=generator.model_config.vae_latent_channels)
    logger.info("Expanded UNet conv_in: %d -> %d channels", original_channels, unet.conv_in.in_channels)

    tuner = InstructionTuner(
        model=generator,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        lr=args.lr,
        target_modules=(
            (
                list(generator.model_config.lora_target_modules)
                + SDXL_CONV_LORA_TARGET_MODULES
            )
            if args.lora_conv_targets
            else None
        ),
        use_rslora=True,
        use_ema=False,
        use_gradient_checkpointing=False,
        use_min_snr=False,
        cfg_dropout_rate=0.0,
    )
    train_unet = get_underlying_unet(tuner)
    vae = generator._t2i_pipe.vae
    if args.pixel_loss_weight > 0:
        vae.requires_grad_(False)
    if hasattr(train_unet, "enable_gradient_checkpointing"):
        train_unet.enable_gradient_checkpointing()
    resume_metrics = None
    initial_step = 0
    resolved_resume_dir = None
    if args.resume_dir is not None:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        resolved_resume_dir = args.resume_dir.expanduser().resolve()
        resume_conditioning = read_source_conditioning_config(resolved_resume_dir)
        expected_prediction_type = (
            "sample" if args.prediction_target == "x0" else "epsilon"
        )
        if (
            resume_conditioning is not None
            and resume_conditioning.prediction_type != expected_prediction_type
        ):
            raise ValueError(
                "Resume checkpoint was trained with prediction_type="
                f"{resume_conditioning.prediction_type!r}; current run expects "
                f"{expected_prediction_type!r}"
            )
        conv_path = resolved_resume_dir / "conv_in.pt"
        adapter_dir = resolved_resume_dir / "adapter"
        if not conv_path.is_file() or not adapter_dir.is_dir():
            raise FileNotFoundError(
                "Resume directory must contain conv_in.pt and adapter/: "
                f"{resolved_resume_dir}"
            )
        train_unet.conv_in.load_state_dict(
            torch.load(conv_path, map_location=device, weights_only=True)
        )
        adapter_state = load_peft_weights(str(adapter_dir), device=str(device))
        set_peft_model_state_dict(tuner.peft_model, adapter_state, adapter_name="default")
        metrics_path = resolved_resume_dir / "metrics.json"
        if metrics_path.is_file():
            resume_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            initial_step = infer_completed_steps(resume_metrics)
        logger.info("Loaded source-conditioned weights from %s", resolved_resume_dir)
    train_unet.conv_in.requires_grad_(True)
    trainable = [parameter for parameter in tuner.peft_model.parameters() if parameter.requires_grad]
    optimizer = None
    if not args.eval_only:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
        if resolved_resume_dir is not None:
            optimizer_path = resolved_resume_dir / "optimizer.pt"
            if optimizer_path.is_file():
                optimizer.load_state_dict(
                    torch.load(
                        optimizer_path,
                        map_location=device,
                        weights_only=False,
                    )
                )
                logger.info("Restored optimizer state from %s", optimizer_path)
            else:
                logger.warning(
                    "No optimizer state in legacy checkpoint; continuation starts "
                    "with fresh AdamW moments"
                )
    logger.info(
        "Trainable parameters: %d (LoRA + expanded conv_in)",
        sum(parameter.numel() for parameter in trainable),
    )

    encoded_train = encode_pairs(tuner, train_pairs, args.resolution, device)
    encoded_eval = encode_pairs(tuner, eval_pairs, args.resolution, device)
    prompts = [depth_prompt] * len(train_pairs)
    prompt_conditions = [tuner._encode_text([prompt], device) for prompt in prompts]
    eval_prompt_conditions = [
        tuner._encode_text([depth_prompt], device) for _ in eval_pairs
    ]
    scheduler = generator._t2i_pipe.scheduler
    scheduler.register_to_config(
        prediction_type="sample" if args.prediction_target == "x0" else "epsilon"
    )
    scheduler.set_timesteps(args.inference_steps, device=device)
    training_timesteps = repeat_final_timestep(
        scheduler.timesteps.detach().clone(), args.final_timestep_repeats
    )
    model_dtype = next(tuner.peft_model.parameters()).dtype
    rng = torch.Generator(device=device).manual_seed(args.seed)
    if initial_step > 0 and not args.eval_only:
        noise_shape = encoded_train[0]["target_latent"].shape
        for _ in range(initial_step):
            torch.randn(
                noise_shape,
                generator=rng,
                device=device,
                dtype=model_dtype,
            )
        logger.info("Advanced training RNG by %d completed steps", initial_step)

    before_predictions = None
    if not args.eval_only:
        before_predictions = []
        for index, sample in enumerate(encoded_eval):
            before_predictions.append(
                predict(
                    tuner,
                    sample["source_latent"],
                    eval_prompt_conditions[index],
                    args.inference_steps,
                    args.seed + index,
                    prediction_target=args.prediction_target,
                    eval_timestep_index=eval_timestep_index,
                )
            )

    loss_history = []
    if not args.eval_only:
        tuner.peft_model.train()
        assert optimizer is not None
        for step in range(args.max_steps):
            global_step = initial_step + step
            sample_index, timestep_index = training_schedule_indices(
                global_step,
                sample_count=len(encoded_train),
                timestep_count=len(training_timesteps),
            )
            sample = encoded_train[sample_index]
            source_latent = sample["source_latent"].to(device=device, dtype=model_dtype)
            target_latent = sample["target_latent"].to(device=device, dtype=model_dtype)
            timestep = training_timesteps[timestep_index]
            timesteps = timestep.reshape(1).to(device)
            noise = torch.randn(
                target_latent.shape,
                generator=rng,
                device=device,
                dtype=model_dtype,
            )
            model_input = (
                build_x0_model_input(
                    scheduler,
                    source_latent,
                    noise,
                    timesteps,
                )
                if args.prediction_target == "x0"
                else build_conditioned_model_input(
                    scheduler,
                    target_latent,
                    source_latent,
                    noise,
                    timesteps,
                )
            )
            prediction = tuner._forward_registry(
                model_input,
                timesteps,
                prompt_conditions[sample_index]["prompt_embeds"],
                prompt_conditions[sample_index]["pooled_prompt_embeds"],
                prompt_conditions[sample_index].get("attention_mask"),
                model_dtype,
                device,
            )
            if args.prediction_target == "x0":
                # Deterministic regression: the UNet output IS the clean
                # target latent, so the primary loss lives in x0 space.
                loss = (prediction.float() - target_latent.float()).square().mean()
                if args.pixel_loss_weight > 0:
                    vae_device = next(vae.parameters()).device
                    vae_dtype = next(vae.parameters()).dtype
                    decoded_x0 = vae_decode(
                        vae,
                        prediction.to(device=vae_device, dtype=vae_dtype),
                        tuner.model_config,
                    )
                    target_pixels = sample["target_pixels"].to(
                        device=decoded_x0.device,
                        dtype=torch.float32,
                    )
                    pixel_l1 = (decoded_x0.float() - target_pixels).abs().mean()
                    loss = loss + args.pixel_loss_weight * pixel_l1
            else:
                epsilon_error = prediction.float() - noise.float()
                loss = epsilon_error.square().mean()
                if args.x0_loss_weight > 0:
                    sigma = sigma_for_timesteps(scheduler, timesteps, target_latent).float()
                    implied_x0_loss = (sigma * epsilon_error).square().mean()
                    loss = loss + args.x0_loss_weight * implied_x0_loss
                if args.pixel_loss_weight > 0:
                    sigma = sigma_for_timesteps(scheduler, timesteps, target_latent).float()
                    implied_x0 = target_latent.float() - sigma * epsilon_error
                    vae_device = next(vae.parameters()).device
                    vae_dtype = next(vae.parameters()).dtype
                    decoded_x0 = vae_decode(
                        vae,
                        implied_x0.to(device=vae_device, dtype=vae_dtype),
                        tuner.model_config,
                    )
                    target_pixels = sample["target_pixels"].to(
                        device=decoded_x0.device,
                        dtype=torch.float32,
                    )
                    pixel_l1 = (decoded_x0.float() - target_pixels).abs().mean()
                    loss = loss + args.pixel_loss_weight * pixel_l1
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            loss_history.append(float(loss.detach()))
            if (step + 1) % args.log_every == 0 or step == 0:
                window = loss_history[-args.log_every :]
                logger.info(
                    "Step %d/%d | loss=%.5f | window_mean=%.5f",
                    global_step + 1,
                    initial_step + args.max_steps,
                    loss_history[-1],
                    float(np.mean(window)),
                )

    train_samples, train_rows, train_summaries = evaluate_encoded_pairs(
        tuner,
        encoded_train,
        prompt_conditions,
        depth_audit,
        inference_steps=args.inference_steps,
        seed=args.seed,
        output_calibration=args.output_calibration,
        prediction_target=args.prediction_target,
        eval_timestep_index=eval_timestep_index,
    )
    samples, rows, eval_summaries = evaluate_encoded_pairs(
        tuner,
        encoded_eval,
        eval_prompt_conditions,
        depth_audit,
        inference_steps=args.inference_steps,
        seed=args.seed,
        output_calibration=args.output_calibration,
        prediction_target=args.prediction_target,
        eval_timestep_index=eval_timestep_index,
        before_predictions=before_predictions,
    )
    per_timestep_train = probe_per_timestep(
        tuner,
        encoded_train,
        prompt_conditions,
        prediction_target=args.prediction_target,
        inference_steps=args.inference_steps,
        seed=args.seed,
        colormap=depth_audit.detected_colormap,
    )
    per_timestep_val = probe_per_timestep(
        tuner,
        encoded_eval,
        eval_prompt_conditions,
        prediction_target=args.prediction_target,
        inference_steps=args.inference_steps,
        seed=args.seed,
        colormap=depth_audit.detected_colormap,
    )
    before_summary = eval_summaries.get("before")
    after_raw_summary = eval_summaries["after"]
    after_calibrated_summary = eval_summaries["after_calibrated"]
    train_raw_summary = train_summaries["after"]
    train_calibrated_summary = train_summaries["after_calibrated"]
    raw_passed = gate_passed(
        after_raw_summary,
        max_mae=args.gate_mae,
        min_correlation=args.gate_correlation,
    )
    calibrated_passed = gate_passed(
        after_calibrated_summary,
        max_mae=args.gate_mae,
        min_correlation=args.gate_correlation,
    )
    train_raw_passed = gate_passed(
        train_raw_summary,
        max_mae=args.gate_mae,
        min_correlation=args.gate_correlation,
    )
    train_calibrated_passed = gate_passed(
        train_calibrated_summary,
        max_mae=args.gate_mae,
        min_correlation=args.gate_correlation,
    )
    if loss_history:
        loss_report = {
            "initial_step": initial_step,
            "additional_steps": len(loss_history),
            "cumulative_steps": initial_step + len(loss_history),
            "first": loss_history[0],
            "last": loss_history[-1],
            "minimum": min(loss_history),
            "history": loss_history,
        }
    elif resume_metrics is not None and isinstance(resume_metrics.get("loss"), dict):
        source_loss = resume_metrics["loss"]
        source_history = source_loss.get("history", [])
        loss_report = {
            "source_checkpoint": str(resolved_resume_dir),
            "first": source_loss.get("first"),
            "last": source_loss.get("last"),
            "minimum": source_loss.get("minimum"),
            "steps": len(source_history),
        }
    else:
        loss_report = None
    serialized_config = {
        key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    report = {
        "model": "segmind/SSD-1B",
        "architecture": "8-channel source-conditioned UNet",
        "prediction_target": args.prediction_target,
        "inference_protocol": (
            SINGLE_STEP_X0 if args.prediction_target == "x0" else MULTI_STEP_EPSILON
        ),
        "eval_timestep_index": (
            eval_timestep_index if args.prediction_target == "x0" else None
        ),
        "lora_target_modules": list(tuner.target_modules),
        "data_dir": str(data_dir),
        "split": {
            "manifest": str(data_dir / "train_val_split.json"),
            "train_samples": [source.name for source, _ in train_pairs],
            "eval_samples": [source.name for source, _ in eval_pairs],
            "overlap": [],
        },
        "depth_encoding": asdict(depth_audit),
        "eval_depth_encoding": asdict(eval_depth_audit),
        "prompts": prompts,
        "config": serialized_config,
        "evaluation_only": args.eval_only,
        "checkpoint": str(resolved_resume_dir) if resolved_resume_dir else None,
        "training_steps": {
            "initial": initial_step,
            "additional": 0 if args.eval_only else len(loss_history),
            "cumulative": initial_step + (0 if args.eval_only else len(loss_history)),
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "elapsed_seconds": time.time() - started,
        "loss": loss_report,
        "train_after": train_raw_summary,
        "train_after_calibrated": train_calibrated_summary,
        "before": before_summary,
        "after": after_raw_summary,
        "after_calibrated": after_calibrated_summary,
        "gate": {
            "rgb_mae_below": args.gate_mae,
            "relative_position_pearson_r_above": args.gate_correlation,
            "raw_passed": raw_passed,
            "calibrated_passed": calibrated_passed,
            # Honest gate: only raw predictions count; calibration is display-only.
            "passed": raw_passed,
        },
        "train_gate": {
            "rgb_mae_below": args.gate_mae,
            "relative_position_pearson_r_above": args.gate_correlation,
            "raw_passed": train_raw_passed,
            "calibrated_passed": train_calibrated_passed,
            "passed": train_raw_passed,
        },
        "per_timestep_train": per_timestep_train,
        "per_timestep_val": per_timestep_val,
        "train_sample_metrics": train_samples,
        "samples": samples,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    make_comparison_grid(
        rows,
        output_dir / "source_conditioned_grid.png",
        labels=(
            ("Source", "Before", "After raw", "After calibrated", "Target")
            if before_predictions is not None
            else ("Source", "After raw", "After calibrated", "Target")
        ),
    )
    make_comparison_grid(
        train_rows,
        output_dir / "train_source_conditioned_grid.png",
        labels=("Source", "After raw", "After calibrated", "Target"),
    )
    if not args.eval_only:
        tuner.peft_model.save_pretrained(output_dir / "adapter")
        torch.save(train_unet.conv_in.state_dict(), output_dir / "conv_in.pt")
        conditioning_config = SourceConditioningConfig(
            in_channels=train_unet.conv_in.in_channels,
            condition_channels=generator.model_config.vae_latent_channels,
            prediction_type=(
                "sample" if args.prediction_target == "x0" else "epsilon"
            ),
            x0_loss_weight=args.x0_loss_weight,
            pixel_loss_weight=args.pixel_loss_weight,
            timestep_sampling="inference",
            training_num_inference_steps=args.inference_steps,
            inference_protocol=(
                SINGLE_STEP_X0
                if args.prediction_target == "x0"
                else MULTI_STEP_EPSILON
            ),
            single_step_timestep_index=(
                eval_timestep_index if args.prediction_target == "x0" else None
            ),
            target_colormap=depth_audit.detected_colormap,
            target_channel_mean=tuple(depth_audit.channel_mean),
            target_channel_std=tuple(depth_audit.channel_std),
            output_calibration=args.output_calibration,
            recommended_prompt=depth_prompt,
            default_num_inference_steps=args.inference_steps,
        )
        write_source_conditioning_config(output_dir, conditioning_config)
        assert optimizer is not None
        torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    logger.info("Train after raw: %s", json.dumps(train_raw_summary))
    logger.info("Train after calibrated: %s", json.dumps(train_calibrated_summary))
    logger.info("Before: %s", json.dumps(before_summary))
    logger.info("After raw: %s", json.dumps(after_raw_summary))
    logger.info("After calibrated: %s", json.dumps(after_calibrated_summary))
    logger.info(
        "Per-timestep val Pearson r: %s",
        json.dumps(
            {
                str(entry["timestep"]): round(
                    entry["relative_position_pearson_r"], 4
                )
                for entry in per_timestep_val
            }
        ),
    )
    logger.info(
        "Gate passed: %s (raw=%s, calibrated=%s)",
        raw_passed,
        raw_passed,
        calibrated_passed,
    )
    logger.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
