#!/usr/bin/env python3
"""Evaluate a SANA LoRA checkpoint with the v5 task-specific metrics.

The evaluator deliberately uses the same annotation-backed validation rows for
every model being compared.  It keeps SANA's generated RGB image as the source
of truth, then decodes that RGB representation into the metric appropriate to
the task (relative depth, normal angle, palette mIoU, or edge F1).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from infer_sana_512 import (
    flow_matching_inference,
    get_dtype,
    load_model_and_lora,
)

LOGGER = logging.getLogger(__name__)
TASKS = ("depth", "normal", "semantic", "structure")
TASK_PROMPTS = {
    "depth": "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map.",
    "normal": "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
    "semantic": "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
    "structure": "Generate a binary structural edge map. Use a black background and thin white edges aligned to visible scene boundaries.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Manifest split to evaluate (default: val).",
    )
    parser.add_argument("--n-per-task", type=int, default=128)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Offset into each task's selected manifest rows; enables stable chunked evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--model-name", default="sana-600m-512px")
    parser.add_argument("--tasks", default="depth,normal,semantic,structure", help="Comma-separated tasks to evaluate")
    parser.add_argument(
        "--preserve-manifest-order",
        action="store_true",
        help="Keep manifest order instead of sorting paths; use this to score a capped training subset exactly.",
    )
    parser.add_argument("--save-panels", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_rows(
    manifest: Path,
    n_per_task: int,
    requested_tasks: tuple[str, ...],
    *,
    split: str,
    preserve_manifest_order: bool,
    start_index: int,
) -> dict[str, list[dict]]:
    rows_by_task = {task: [] for task in requested_tasks}
    aliases = {"seg": "semantic", "segmentation": "semantic"}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            task = aliases.get(row.get("task"), row.get("task"))
            if task in rows_by_task:
                rows_by_task[task].append(row)
    for task in requested_tasks:
        if not preserve_manifest_order:
            rows_by_task[task].sort(key=lambda row: str(row.get("input", "")))
        rows_by_task[task] = rows_by_task[task][start_index : start_index + n_per_task]
    return {task: rows for task, rows in rows_by_task.items() if rows}


def load_rgb(path: str | Path, resolution: int, *, nearest: bool = False) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            (resolution, resolution),
            Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS,
        )
        values = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(values).permute(2, 0, 1)


def normal_angle_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_vec = torch.nn.functional.normalize(pred * 2.0 - 1.0, dim=0, eps=1e-6)
    target_vec = torch.nn.functional.normalize(target * 2.0 - 1.0, dim=0, eps=1e-6)
    dot = (pred_vec * target_vec).sum(dim=0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(dot)).mean().item())


def edge_f1_ods(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_gray = pred.mean(dim=0)
    target_edge = target.mean(dim=0) >= 0.5
    best = 0.0
    for threshold in np.linspace(0.1, 0.9, 17):
        pred_edge = pred_gray >= float(threshold)
        tp = int((pred_edge & target_edge).sum())
        fp = int((pred_edge & ~target_edge).sum())
        fn = int((~pred_edge & target_edge).sum())
        best = max(best, 2.0 * tp / max(1, 2 * tp + fp + fn))
    return float(best)


def load_palette(rows: list[dict]) -> dict | None:
    for row in rows:
        value = row.get("segmentation_palette")
        if value:
            path = Path(value)
            data = json.loads(path.read_text(encoding="utf-8"))
            palette = np.asarray(data["class_palette"], dtype=np.float32) / 255.0
            ignore = np.asarray(data.get("ignore_color", [0, 0, 0]), dtype=np.float32) / 255.0
            return {"path": str(path.resolve()), "palette": palette, "ignore": ignore}
    return None


def quantize_palette(image: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    pixels = image.permute(1, 2, 0).numpy().reshape(-1, 3)
    labels = np.empty(len(pixels), dtype=np.int32)
    for start in range(0, len(pixels), 8192):
        chunk = pixels[start : start + 8192]
        labels[start : start + len(chunk)] = ((chunk[:, None, :] - palette[None, :, :]) ** 2).sum(2).argmin(1)
    return labels


def semantic_metrics(pred: torch.Tensor, target: torch.Tensor, palette_info: dict) -> dict[str, float | int]:
    palette, ignore = palette_info["palette"], palette_info["ignore"]
    pred_labels = quantize_palette(pred, palette)
    target_pixels = target.permute(1, 2, 0).numpy().reshape(-1, 3)
    valid = np.max(np.abs(target_pixels - ignore), axis=1) > (0.5 / 255.0)
    target_labels = quantize_palette(target, palette)
    confusion = np.bincount(
        len(palette) * target_labels[valid] + pred_labels[valid],
        minlength=len(palette) * len(palette),
    ).reshape(len(palette), len(palette))
    tp = np.diag(confusion).astype(np.float64)
    totals = confusion.sum(1).astype(np.float64)
    pred_totals = confusion.sum(0).astype(np.float64)
    union = totals + pred_totals - tp
    present = union > 0
    return {
        "pixel_accuracy": float(tp.sum() / max(1.0, totals.sum())),
        "miou": float((tp[present] / union[present]).mean()) if present.any() else 0.0,
        "present_classes": int(present.sum()),
        "valid_pixels": int(totals.sum()),
    }


def make_panel(path: Path, source: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, title: str) -> None:
    size, gap, label_h = 256, 8, 30
    images = [source, pred, target]
    canvas = Image.new("RGB", (size * 3 + gap * 4, size + label_h + gap * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), title, fill="black")
    for index, image in enumerate(images):
        values = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        preview = Image.fromarray(values).resize((size, size), Image.Resampling.NEAREST)
        canvas.paste(preview, (gap + index * (size + gap), label_h + gap))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if args.n_per_task < 1:
        raise SystemExit("--n-per-task must be positive")
    if args.start_index < 0:
        raise SystemExit("--start-index must be non-negative")
    if not args.pair_manifest.is_file():
        raise SystemExit(f"manifest not found: {args.pair_manifest}")
    if not args.checkpoint.is_dir():
        raise SystemExit(f"SANA adapter directory not found: {args.checkpoint}")

    requested_tasks = tuple(dict.fromkeys(task.strip() for task in args.tasks.split(",") if task.strip()))
    aliases = {"seg": "semantic", "segmentation": "semantic"}
    requested_tasks = tuple(aliases.get(task, task) for task in requested_tasks)
    unknown = sorted(set(requested_tasks) - set(TASKS))
    if unknown:
        raise SystemExit(f"unknown task(s): {', '.join(unknown)}")
    rows_by_task = load_rows(
        args.pair_manifest,
        args.n_per_task,
        requested_tasks,
        split=args.split,
        preserve_manifest_order=args.preserve_manifest_order,
        start_index=args.start_index,
    )
    palette_info = load_palette(rows_by_task.get("semantic", []))
    if "semantic" in rows_by_task and palette_info is None:
        raise SystemExit("semantic rows have no segmentation_palette")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.pair_manifest.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "resolution": args.resolution,
        "split": args.split,
        "n_per_task": args.n_per_task,
        "start_index": args.start_index,
        "batch_size": args.batch_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "img2img_flow": True,
        "seed": args.seed,
        "tasks": {task: len(rows) for task, rows in rows_by_task.items()},
        "selection": "manifest order" if args.preserve_manifest_order else "sorted by input path",
    }, indent=2) + "\n", encoding="utf-8")

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    dtype = get_dtype(args.dtype)
    model_args = argparse.Namespace(
        model_name=args.model_name,
        checkpoint=str(args.checkpoint),
        lora_rank=16,
        lora_alpha=32,
        no_rslora=False,
        dtype=args.dtype,
    )
    LOGGER.info("Loading SANA checkpoint %s on %s", args.checkpoint, device)
    peft_model, vae, components, model_cfg = load_model_and_lora(model_args, device, dtype)
    model_dtype = next(peft_model.parameters()).dtype
    metrics: dict[str, list[dict]] = {task: [] for task in rows_by_task}
    start_all = time.monotonic()

    for task, rows in rows_by_task.items():
        LOGGER.info("Evaluating %s: %d %s samples", task, len(rows), args.split)
        for batch_start in range(0, len(rows), args.batch_size):
            batch_rows = rows[batch_start : batch_start + args.batch_size]
            sources = [load_rgb(row["input"], args.resolution) for row in batch_rows]
            targets = [load_rgb(row["target"], args.resolution, nearest=task == "semantic") for row in batch_rows]
            input_batch = torch.stack([(source * 2.0 - 1.0) for source in sources]).to(device=device, dtype=model_dtype)
            predictions = flow_matching_inference(
                peft_model, vae, components, input_batch,
                [row.get("prompt", TASK_PROMPTS[task]) for row in batch_rows], device, model_dtype,
                num_steps=args.num_inference_steps, use_img2img=True,
                max_length=model_cfg.text_encoder_max_length, guidance_scale=args.guidance_scale,
            ).cpu()
            for offset, (row, source, target, prediction) in enumerate(zip(batch_rows, sources, targets, predictions)):
                index = batch_start + offset
                sample_result: dict[str, float | int | str] = {
                    "index": index, "input": row["input"], "target": row["target"],
                    "rgb_mae": float((prediction - target).abs().mean()),
                    "psnr": float(10.0 * math.log10(1.0 / max(1e-12, float((prediction - target).square().mean())))),
                }
                if task == "depth":
                    sample_result.update(evaluate_relative_depth_rgb(prediction[None], target[None], colormap="turbo"))
                elif task == "normal":
                    sample_result["normal_mean_angular_error_deg"] = normal_angle_deg(prediction, target)
                elif task == "semantic":
                    sample_result.update({f"semantic_{key}": value for key, value in semantic_metrics(prediction, target, palette_info).items()})
                elif task == "structure":
                    sample_result["structure_edge_f1_ods"] = edge_f1_ods(prediction, target)
                    sample_result["structure_positive_fraction"] = float((prediction.mean(0) >= 0.5).float().mean())
                metrics[task].append(sample_result)
                if args.save_panels:
                    stem = Path(row["input"]).stem
                    make_panel(args.output_dir / f"{task}_{index:04d}_{stem}.png", source, prediction, target, f"{task} #{index}")
            done = batch_start + len(batch_rows)
            if done % 16 == 0 or done == len(rows):
                LOGGER.info("  %s %d/%d (%.1fs)", task, done, len(rows), time.monotonic() - start_all)

    per_task: dict[str, dict] = {}
    for task, values in metrics.items():
        if not values:
            continue
        numeric_keys = sorted({key for value in values for key in value if isinstance(value.get(key), (int, float))})
        per_task[task] = {key: float(np.mean([value[key] for value in values])) for key in numeric_keys}
        per_task[task]["count"] = len(values)
    report = {
        "model": "SANA checkpoint",
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.pair_manifest.resolve()),
        "split": args.split,
        "selection": (
            "manifest order, first n_per_task rows per task"
            if args.preserve_manifest_order
            else "sorted by input path, first n_per_task rows per task"
        ),
        "config": json.loads((args.output_dir / "config.json").read_text()),
        "per_task": per_task,
        "samples": metrics,
        "notes": {
            "normal": "v5 normal targets are known to be collapsed; normal angle is reported for audit only, not as a valid quality claim.",
            "depth": "turbo RGB decoding reports relative ordering metrics; per-image RGB targets do not provide global metric scale.",
            "specialist_comparison": "Compare semantic mIoU against the DINOv2 specialist gate and normal angle against the DIODE DINOv2 gate with dataset caveat.",
        },
        "elapsed_sec": time.monotonic() - start_all,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Saved %s", args.output_dir / "metrics.json")
    print(json.dumps({"per_task": per_task, "metrics": str((args.output_dir / "metrics.json").resolve())}, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
