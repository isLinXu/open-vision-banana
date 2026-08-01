#!/usr/bin/env python3
"""Run a reproducible SSD-1B depth overfit evaluation on a paired dataset.

The script evaluates either the untouched SSD-1B base model or one LoRA
checkpoint on the same deterministic samples, writes per-sample predictions,
a triptych grid, and lightweight RGB reconstruction metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.conditioning import (  # noqa: E402
    scheduler_prediction_type,
    set_scheduler_prediction_type,
    validate_scheduler_prediction_type,
)
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.training.tuner import VisionTaskDataset  # noqa: E402

logger = logging.getLogger("quick_validate_ssd_v3")


def validate_checkpoint_path(checkpoint: Path) -> Path:
    """Validate root or nested PEFT layout used by source-conditioned gates."""
    if (
        (checkpoint / "adapter_model.safetensors").is_file()
        or (checkpoint / "adapter" / "adapter_model.safetensors").is_file()
    ):
        return checkpoint
    raise FileNotFoundError(
        "Invalid LoRA checkpoint (expected adapter_model.safetensors "
        f"at root or adapter/: {checkpoint}"
    )


def find_pairs(task_dir: Path, num_samples: int) -> list[tuple[Path, Path]]:
    """Return the first deterministic image/target pairs from a task directory."""
    image_dir = task_dir / "images"
    target_dir = task_dir / "targets"
    if not image_dir.is_dir() or not target_dir.is_dir():
        raise NotADirectoryError(f"Expected images/ and targets/ under {task_dir}")

    targets = {
        path.stem: path
        for path in sorted(target_dir.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        target_path = targets.get(image_path.stem)
        if target_path is not None:
            pairs.append((image_path, target_path))
        if len(pairs) == num_samples:
            break
    if len(pairs) < num_samples:
        raise ValueError(
            f"Requested {num_samples} pairs from {task_dir}, found only {len(pairs)}"
        )
    return pairs


def load_rgb(path: Path, resolution: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.min().item() < 0:
        tensor = (tensor + 1.0) / 2.0
    array = (
        tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def reconstruction_metrics(prediction: Image.Image, target: Image.Image) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float32) / 255.0
    tgt = np.asarray(target, dtype=np.float32) / 255.0
    pred_flat = pred.reshape(-1)
    tgt_flat = tgt.reshape(-1)
    pred_std = float(pred_flat.std())
    tgt_std = float(tgt_flat.std())
    correlation = (
        float(np.corrcoef(pred_flat, tgt_flat)[0, 1])
        if pred_std > 1e-8 and tgt_std > 1e-8
        else 0.0
    )
    return {
        "mse": float(np.mean((pred - tgt) ** 2)),
        "mae": float(np.mean(np.abs(pred - tgt))),
        "pearson_r": correlation,
    }


def make_grid(rows: list[tuple[Image.Image, Image.Image, Image.Image]], path: Path) -> None:
    cell = rows[0][0].width
    header = 28
    canvas = Image.new("RGB", (cell * 3, header + cell * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Input", "Prediction", "Target")):
        draw.text((column * cell + 8, 7), label, fill="black")
    for row_index, images in enumerate(rows):
        y = header + row_index * cell
        for column, image in enumerate(images):
            canvas.paste(image, (column * cell, y))
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prediction-type",
        choices=("auto", "epsilon", "sample"),
        default="epsilon",
        help=("Inference scheduler semantics. Defaults to epsilon to match SSD-1B "
              "training; auto is retained only for explicit checkpoint diagnosis."),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    task_dir = args.data_dir.expanduser().resolve() / "depth"
    pairs = find_pairs(task_dir, args.num_samples)
    if args.checkpoint is not None:
        checkpoint = validate_checkpoint_path(args.checkpoint.expanduser().resolve())
    else:
        checkpoint = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = args.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading SSD-1B on %s (checkpoint=%s)", device, checkpoint or "base")
    started = time.time()
    generator = BaseGenerator(
        model_name="segmind/SSD-1B",
        lora_path=str(checkpoint) if checkpoint is not None else None,
        device=device,
        dtype=torch.bfloat16,
    )
    logger.info("Model loaded in %.1fs", time.time() - started)
    scheduler = generator._t2i_pipe.scheduler
    if args.prediction_type == "auto":
        expected_prediction_type = (
            generator.source_conditioning_config.prediction_type
            if generator.source_conditioning_config is not None
            else "epsilon"
        )
    else:
        expected_prediction_type = args.prediction_type
    if args.prediction_type != "auto":
        set_scheduler_prediction_type(scheduler, expected_prediction_type)
    effective_prediction_type = validate_scheduler_prediction_type(
        scheduler,
        expected_prediction_type,
    )
    logger.info("Using inference prediction_type=%s", effective_prediction_type)

    prompt = VisionTaskDataset.TASK_PROMPTS["depth"]
    sample_metrics = []
    grid_rows = []
    for index, (image_path, target_path) in enumerate(pairs):
        input_image = load_rgb(image_path, args.resolution)
        target_image = load_rgb(target_path, args.resolution)
        infer_started = time.time()
        with torch.inference_mode():
            result = generator.generate(
                prompt=prompt,
                image=input_image,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                strength=args.strength,
                height=args.resolution,
                width=args.resolution,
                seed=args.seed + index,
            )
        prediction = tensor_to_pil(result[0])
        prediction_path = prediction_dir / f"{image_path.stem}.png"
        prediction.save(prediction_path)
        metrics = reconstruction_metrics(prediction, target_image)
        metrics.update(
            {
                "sample": image_path.name,
                "inference_seconds": time.time() - infer_started,
            }
        )
        sample_metrics.append(metrics)
        grid_rows.append((input_image, prediction, target_image))
        logger.info("%s: mse=%.5f r=%.4f", image_path.name, metrics["mse"], metrics["pearson_r"])

    summary = {
        key: float(np.mean([sample[key] for sample in sample_metrics]))
        for key in ("mse", "mae", "pearson_r", "inference_seconds")
    }
    report = {
        "model": "segmind/SSD-1B",
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "prompt": prompt,
        "config": {
            "num_samples": args.num_samples,
            "resolution": args.resolution,
            "steps": args.steps,
            "strength": args.strength,
            "seed": args.seed,
            "requested_prediction_type": args.prediction_type,
            "prediction_type": scheduler_prediction_type(scheduler),
        },
        "summary": summary,
        "samples": sample_metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    make_grid(grid_rows, args.output_dir / "depth_overfit_grid.png")
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    logger.info("Artifacts written to %s", args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
