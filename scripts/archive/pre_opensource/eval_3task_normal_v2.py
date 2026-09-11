#!/usr/bin/env python3
"""Evaluate checkpoint-5500_best on 3 tasks: depth, normal, seg.

Generates:
  - Input / GT / Prediction visualization panels
  - Summary grids per task
  - Quantitative metrics: RMSE (depth), angular error (normal), mIoU (seg),
    plus RGB MAE/PSNR for all tasks.

Usage:
  python scripts/eval_3task_normal_v2.py \
      --checkpoint output/sana_1600m_v5full_3task_normal_v2/checkpoint-5500_best/lora_adapter \
      --pair-manifest output/snvc_v5full_pairs/pairs_5task_v3.jsonl \
      --output-dir output/sana_1600m_v5full_3task_normal_v2/eval_5500 \
      --n-per-task 30
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.inference.sana_img2img import flow_matching_inference
from open_vision_banana.model_registry import (
    apply_lora,
    get_model_config,
    load_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_3task")

# ── Task prompts (matching training) ──────────────────────────────────────
TASK_PROMPTS = {
    "depth": "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map.",
    "normal": "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
    "seg": "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
}

# ── ADE20K 150-class palette (from decoder.py) ────────────────────────────
ADE20K_151_PALETTE = {
    0: (0, 0, 0), 1: (0, 255, 235), 2: (4, 250, 7), 3: (0, 255, 133),
    4: (120, 120, 120), 5: (255, 0, 235), 6: (235, 255, 7), 7: (255, 41, 0),
    8: (10, 255, 71), 9: (255, 0, 255), 10: (255, 8, 41), 11: (255, 0, 214),
    12: (7, 255, 255), 13: (255, 224, 0), 14: (6, 51, 255), 15: (0, 255, 184),
    16: (4, 200, 3), 17: (255, 0, 152), 18: (8, 255, 51), 19: (245, 0, 255),
    20: (140, 140, 140), 21: (255, 0, 5), 22: (143, 255, 140), 23: (255, 51, 0),
    24: (255, 194, 7), 25: (255, 245, 0), 26: (140, 140, 140), 27: (255, 0, 31),
    28: (255, 122, 8), 29: (255, 214, 0), 30: (20, 255, 0), 31: (255, 163, 0),
    32: (0, 0, 0), 33: (133, 255, 0), 34: (224, 5, 255), 35: (255, 0, 0),
    36: (180, 120, 120), 37: (245, 0, 255), 38: (150, 5, 61), 39: (255, 163, 0),
    40: (255, 41, 10), 41: (255, 0, 163), 42: (255, 5, 153), 43: (255, 0, 41),
    44: (224, 255, 8), 45: (0, 0, 0), 46: (235, 12, 255), 47: (255, 0, 245),
    48: (120, 120, 80), 49: (0, 0, 0), 50: (255, 6, 82), 51: (255, 0, 112),
    52: (0, 0, 0), 53: (255, 245, 0), 54: (204, 255, 4), 55: (0, 255, 163),
    56: (0, 0, 0), 57: (0, 0, 0), 58: (250, 10, 15), 59: (0, 0, 0),
    60: (0, 255, 20), 61: (0, 255, 122), 62: (31, 255, 0), 63: (255, 0, 133),
    64: (204, 70, 3), 65: (255, 235, 0), 66: (220, 220, 220), 67: (0, 255, 214),
    68: (61, 230, 250), 69: (92, 0, 255), 70: (112, 9, 255), 71: (255, 0, 184),
    72: (255, 31, 0), 73: (255, 235, 0), 74: (255, 71, 0), 75: (255, 0, 82),
    76: (255, 224, 0), 77: (0, 0, 0), 78: (0, 235, 255), 79: (0, 0, 0),
    80: (11, 102, 255), 81: (255, 0, 92), 82: (8, 255, 214), 83: (255, 0, 71),
    84: (255, 9, 224), 85: (255, 255, 0), 86: (255, 184, 6), 87: (0, 255, 245),
    88: (153, 255, 0), 89: (255, 0, 173), 90: (0, 173, 255), 91: (255, 0, 204),
    92: (0, 0, 0), 93: (255, 0, 61), 94: (11, 200, 200), 95: (0, 0, 0),
    96: (0, 102, 200), 97: (0, 0, 0), 98: (255, 9, 92), 99: (255, 102, 0),
    100: (255, 6, 51), 101: (255, 0, 235), 102: (0, 0, 0), 103: (255, 214, 0),
    104: (0, 0, 0), 105: (0, 0, 0), 106: (0, 0, 0), 107: (0, 0, 0),
    108: (0, 0, 0), 109: (255, 0, 122), 110: (0, 0, 0), 111: (0, 255, 102),
    112: (255, 7, 71), 113: (255, 0, 20), 114: (7, 255, 224), 115: (0, 0, 0),
    116: (9, 7, 230), 117: (51, 0, 255), 118: (0, 0, 0), 119: (255, 184, 0),
    120: (0, 0, 255), 121: (255, 0, 10), 122: (31, 0, 255), 123: (0, 0, 0),
    124: (0, 0, 0), 125: (0, 255, 10), 126: (255, 82, 0), 127: (0, 0, 0),
    128: (6, 230, 230), 129: (0, 0, 0), 130: (120, 120, 70), 131: (0, 0, 0),
    132: (80, 50, 50), 133: (255, 224, 0), 134: (0, 0, 0), 135: (255, 204, 0),
    136: (102, 8, 255), 137: (255, 0, 102), 138: (160, 150, 20), 139: (255, 0, 51),
    140: (255, 61, 6), 141: (0, 0, 0), 142: (0, 163, 255), 143: (255, 82, 0),
    144: (204, 5, 255), 145: (255, 184, 0), 146: (0, 0, 0), 147: (0, 0, 0),
    148: (230, 230, 230), 149: (0, 255, 61), 150: (255, 51, 7),
}


# ── Helpers ───────────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [3,H,W] or [1,3,H,W] tensor in [0,1] to PIL Image."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().float()
    if t.min() < -0.01:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    if t.dim() == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)
    elif t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_rgb(path: str, resolution: int, *, nearest: bool = False) -> torch.Tensor:
    """Load image as [3,H,W] tensor in [0,1]."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize(
            (resolution, resolution),
            Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS,
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def make_panel(source: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, title: str) -> Image.Image:
    """Create Input | Prediction | GT comparison panel."""
    size = 256
    gap, label_h = 8, 30
    images = [source, pred, target]
    labels = ["Input", "Prediction", "Ground Truth"]
    canvas = Image.new("RGB", (size * 3 + gap * 4, size + label_h + gap * 2), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((gap, 6), title, fill=(255, 255, 255), font=font)
    for idx, (image, label) in enumerate(zip(images, labels)):
        values = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        preview = Image.fromarray(values).resize((size, size), Image.Resampling.LANCZOS)
        x = gap + idx * (size + gap)
        y = label_h + gap
        canvas.paste(preview, (x, y))
        draw.text((x + 5, size + label_h - 20), label, fill=(200, 200, 200), font=font)
    return canvas


def make_summary_grid(panels: list[Image.Image], title: str) -> Image.Image:
    """Stack multiple panels vertically into a summary image."""
    if not panels:
        return None
    w, h = panels[0].size
    canvas = Image.new("RGB", (w, h * len(panels)), (30, 30, 30))
    for i, panel in enumerate(panels):
        canvas.paste(panel, (0, i * h))
    return canvas


# ── Metric functions ──────────────────────────────────────────────────────

def normal_angle_deg(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean angular error in degrees between predicted and target normals."""
    pred_vec = F.normalize(pred * 2.0 - 1.0, dim=0, eps=1e-6)
    target_vec = F.normalize(target * 2.0 - 1.0, dim=0, eps=1e-6)
    dot = (pred_vec * target_vec).sum(dim=0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(dot)).mean().item())


def quantize_palette(image: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    """Map RGB pixels to nearest palette class indices."""
    pixels = image.permute(1, 2, 0).numpy().reshape(-1, 3)
    labels = np.empty(len(pixels), dtype=np.int32)
    for start in range(0, len(pixels), 8192):
        chunk = pixels[start : start + 8192]
        labels[start : start + len(chunk)] = (
            (chunk[:, None, :] - palette[None, :, :]) ** 2
        ).sum(2).argmin(1)
    return labels


def compute_miou(pred: torch.Tensor, target: torch.Tensor, palette: np.ndarray) -> dict:
    """Compute mIoU using palette quantization."""
    pred_labels = quantize_palette(pred, palette)
    target_labels = quantize_palette(target, palette)
    n_classes = len(palette)
    confusion = np.bincount(
        n_classes * target_labels + pred_labels,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    tp = np.diag(confusion).astype(np.float64)
    totals = confusion.sum(1).astype(np.float64)
    pred_totals = confusion.sum(0).astype(np.float64)
    union = totals + pred_totals - tp
    present = union > 0
    miou = float((tp[present] / union[present]).mean()) if present.any() else 0.0
    pixel_acc = float(tp.sum() / max(1.0, totals.sum()))
    return {"miou": miou, "pixel_accuracy": pixel_acc, "present_classes": int(present.sum())}


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="LoRA adapter directory")
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-task", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("Loading Sana-1600M pipeline...")
    model_cfg = get_model_config("sana-1600m")
    components = load_pipeline(model_cfg, device=device, dtype=dtype)
    # Keep text_encoder in native dtype to avoid OOM on 24GB GPU
    if dtype in (torch.float16, torch.bfloat16):
        components.vae.to(dtype)

    backbone = components.unet
    vae = components.vae

    logger.info("Applying LoRA from %s", args.checkpoint)
    peft_model = apply_lora(
        backbone, config=model_cfg, rank=32, alpha=64, use_rslora=True,
    )
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(backbone, str(args.checkpoint), is_trainable=False)
    peft_model.eval()
    model_dtype = next(peft_model.parameters()).dtype
    logger.info("Model dtype: %s", model_dtype)

    # Read metadata
    ckpt_root = args.checkpoint.parent
    metadata_path = ckpt_root / "metadata.json"
    if metadata_path.is_file():
        meta = json.loads(metadata_path.read_text())
        logger.info("Checkpoint: step=%d, val_loss=%.4f", meta.get("step", 0), meta.get("val_loss", 0))

    # ── Load val rows ─────────────────────────────────────────────────────
    logger.info("Loading val rows from %s", args.pair_manifest)
    rows_by_task: dict[str, list[dict]] = {"depth": [], "normal": [], "seg": []}
    with args.pair_manifest.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            task = row["task"]
            if task in rows_by_task and len(rows_by_task[task]) < args.n_per_task:
                rows_by_task[task].append(row)

    for task, rows in rows_by_task.items():
        logger.info("  %s: %d val samples", task, len(rows))

    # ADE20K palette for seg metrics
    ade_palette = np.array(
        [ADE20K_151_PALETTE[i] for i in range(151)], dtype=np.float32
    ) / 255.0

    # ── Evaluate each task ────────────────────────────────────────────────
    all_metrics: dict[str, list[dict]] = {t: [] for t in rows_by_task}
    panels_by_task: dict[str, list[Image.Image]] = {t: [] for t in rows_by_task}
    t_start = time.monotonic()

    for task, rows in rows_by_task.items():
        if not rows:
            continue
        logger.info("\n=== Evaluating %s: %d samples ===", task, len(rows))

        for idx, row in enumerate(rows):
            input_path = row["input"]
            target_path = row["target"]

            if not Path(input_path).is_file():
                logger.warning("  Skip %d: input not found: %s", idx, input_path)
                continue
            if not Path(target_path).is_file():
                logger.warning("  Skip %d: target not found: %s", idx, target_path)
                continue

            # Load images
            source = load_rgb(input_path, args.resolution)
            target = load_rgb(target_path, args.resolution, nearest=(task == "seg"))

            input_batch = ((source * 2.0 - 1.0).unsqueeze(0)).to(device=device, dtype=model_dtype)
            prompt = TASK_PROMPTS.get(task, f"Generate {task} map")

            # Inference
            with torch.no_grad():
                prediction = flow_matching_inference(
                    peft_model, vae, components, input_batch,
                    [prompt], device, model_dtype,
                    num_steps=args.num_inference_steps,
                    use_img2img=True,
                    max_length=model_cfg.text_encoder_max_length,
                    guidance_scale=args.guidance_scale,
                ).cpu()

            pred = prediction[0]

            # ── Compute metrics ───────────────────────────────────────────
            sample: dict = {
                "index": idx,
                "input": input_path,
                "rgb_mae": float((pred - target).abs().mean()),
                "psnr": float(10.0 * math.log10(1.0 / max(1e-12, float((pred - target).square().mean())))),
            }

            if task == "depth":
                sample.update(evaluate_relative_depth_rgb(
                    pred.unsqueeze(0), target.unsqueeze(0), colormap="turbo",
                ))
            elif task == "normal":
                sample["normal_mean_angular_error_deg"] = normal_angle_deg(pred, target)
            elif task == "seg":
                sample.update({
                    f"seg_{k}": v for k, v in compute_miou(pred, target, ade_palette).items()
                })

            all_metrics[task].append(sample)

            # ── Save panel ────────────────────────────────────────────────
            panel = make_panel(source, pred, target, f"{task} #{idx}")
            panels_by_task[task].append(panel)

            panel_path = args.output_dir / f"{task}_{idx:04d}_{Path(input_path).stem}.png"
            panel.save(str(panel_path))

            # Log progress
            metric_str = f"MAE={sample['rgb_mae']:.4f}"
            if task == "depth":
                metric_str += f", RMSE={sample.get('relative_position_rmse', 0):.4f}"
                metric_str += f", Spearman={sample.get('relative_position_spearman_r', 0):.4f}"
            elif task == "normal":
                metric_str += f", Angle={sample.get('normal_mean_angular_error_deg', 0):.1f}°"
            elif task == "seg":
                metric_str += f", mIoU={sample.get('seg_miou', 0):.4f}"

            if (idx + 1) % 5 == 0 or idx == 0:
                logger.info("  [%d/%d] %s", idx + 1, len(rows), metric_str)

    # ── Save summary grids ────────────────────────────────────────────────
    for task, panels in panels_by_task.items():
        if panels:
            summary = make_summary_grid(panels, task)
            if summary:
                summary.save(str(args.output_dir / f"{task}_summary.png"))
                logger.info("Saved summary: %s_summary.png", task)

    # ── Aggregate metrics ─────────────────────────────────────────────────
    per_task: dict[str, dict] = {}
    for task, values in all_metrics.items():
        if not values:
            continue
        numeric_keys = sorted({
            k for v in values for k in v if isinstance(v.get(k), (int, float))
        })
        per_task[task] = {
            k: float(np.mean([v[k] for v in values])) for k in numeric_keys
        }
        per_task[task]["count"] = len(values)

    elapsed = time.monotonic() - t_start

    report = {
        "model": "sana-1600m",
        "checkpoint": str(args.checkpoint.resolve()),
        "step": meta.get("step", 0) if metadata_path.is_file() else 0,
        "val_loss": meta.get("val_loss", 0) if metadata_path.is_file() else 0,
        "manifest": str(args.pair_manifest.resolve()),
        "split": "val",
        "n_per_task": args.n_per_task,
        "num_inference_steps": args.num_inference_steps,
        "per_task": per_task,
        "elapsed_sec": round(elapsed, 1),
    }

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    logger.info("\nSaved metrics: %s", metrics_path)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  EVALUATION RESULTS — Checkpoint step={report.get('step', '?')}, "
          f"val_loss={report.get('val_loss', 0):.4f}")
    print("=" * 70)

    for task, m in per_task.items():
        print(f"\n  [{task.upper()}] ({m['count']} samples)")
        print(f"    RGB MAE:     {m.get('rgb_mae', 0):.4f}")
        print(f"    PSNR:        {m.get('psnr', 0):.2f} dB")
        if task == "depth":
            print(f"    RMSE:        {m.get('relative_position_rmse', 0):.4f}")
            print(f"    Spearman R:  {m.get('relative_position_spearman_r', 0):.4f}")
            print(f"    Pearson R:   {m.get('relative_position_pearson_r', 0):.4f}")
        elif task == "normal":
            print(f"    Mean Angle:  {m.get('normal_mean_angular_error_deg', 0):.2f}°")
        elif task == "seg":
            print(f"    mIoU:        {m.get('seg_miou', 0):.4f}")
            print(f"    Pixel Acc:   {m.get('seg_pixel_accuracy', 0):.4f}")
            print(f"    Classes:     {m.get('seg_present_classes', 0)}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Output: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
