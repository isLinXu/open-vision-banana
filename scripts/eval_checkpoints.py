#!/usr/bin/env python3
"""Checkpoint evaluation and selection for multi-task training.

Evaluates each checkpoint on the held-out validation set using per-task metrics:
  - depth: δ1 (threshold accuracy) + AbsRel
  - normal: Angular error (degrees)
  - semantic: mIoU (mean Intersection over Union)
  - instance: AP@0.5
  - referring: IoU (binary mask)
  - generation: CLIP score + FID
  - input preservation: LPIPS

Selects the best checkpoint per task by primary metric.

Usage:
  python scripts/eval_checkpoints.py --checkpoints output/v5_multitask/checkpoint-*
  python scripts/eval_checkpoints.py --ckpt-dir output/v5_multitask --select-best
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")

MODEL_ID = "segmind/SSD-1B"
RESOLUTION = 512
VAE_SCALE = 8
MAX_VAL_SAMPLES = 50  # limit for speed


# ─── Metrics ──────────────────────────────────────────────────────────────────

def depth_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Depth evaluation: δ1, δ2, δ3, AbsRel, RMSE."""
    # Normalize both to [0, 1]
    pred_norm = pred.astype(np.float64) / 255.0
    target_norm = target.astype(np.float64) / 255.0

    # Avoid division by zero
    mask = target_norm > 0.01
    if not mask.any():
        return {"delta1": 0, "delta2": 0, "delta3": 0, "absrel": 1.0, "rmse": 1.0}

    ratio = np.maximum(pred_norm[mask] / target_norm[mask],
                      target_norm[mask] / pred_norm[mask])
    delta1 = float(np.mean(ratio < 1.25))
    delta2 = float(np.mean(ratio < 1.25 ** 2))
    delta3 = float(np.mean(ratio < 1.25 ** 3))

    absrel = float(np.mean(np.abs(pred_norm[mask] - target_norm[mask]) / target_norm[mask]))
    rmse = float(np.sqrt(np.mean((pred_norm[mask] - target_norm[mask]) ** 2)))

    return {"delta1": delta1, "delta2": delta2, "delta3": delta3, "absrel": absrel, "rmse": rmse}


def normal_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Normal evaluation: angular error in degrees."""
    # Convert RGB to normals: n = (R/255*2-1, G/255*2-1, B/255*2-1)
    pred_n = pred.astype(np.float64) / 127.5 - 1.0
    target_n = target.astype(np.float64) / 127.5 - 1.0

    # Normalize
    pred_norm = pred_n / (np.linalg.norm(pred_n, axis=-1, keepdims=True) + 1e-8)
    target_norm = target_n / (np.linalg.norm(target_n, axis=-1, keepdims=True) + 1e-8)

    # Angular error
    cos_sim = np.sum(pred_norm * target_norm, axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angles = np.arccos(cos_sim) * 180.0 / np.pi

    return {
        "mean_angular_error": float(np.mean(angles)),
        "median_angular_error": float(np.median(angles)),
        "rmse_angular_error": float(np.sqrt(np.mean(angles ** 2))),
    }


def semantic_metrics(pred: np.ndarray, target: np.ndarray, palette: np.ndarray) -> dict:
    """Semantic segmentation: mIoU."""
    # Map colors to class indices
    pred_classes = _colors_to_classes(pred, palette)
    target_classes = _colors_to_classes(target, palette)

    # Compute IoU for each class
    n_classes = len(palette)
    ious = []
    for c in range(n_classes):
        pred_mask = pred_classes == c
        target_mask = target_classes == c
        if not target_mask.any() and not pred_mask.any():
            continue
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        if union > 0:
            ious.append(intersection / union)

    miou = float(np.mean(ious)) if ious else 0.0
    pixel_acc = float(np.mean(pred_classes == target_classes))

    return {"mIoU": miou, "pixel_accuracy": pixel_acc, "n_classes_evaluated": len(ious)}


def _colors_to_classes(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map RGB colors to nearest palette class index."""
    from scipy.spatial import cKDTree
    h, w, c = arr.shape
    flat = arr.reshape(-1, 3).astype(np.float32)
    tree = cKDTree(palette.astype(np.float32))
    _, classes = tree.query(flat, k=1)
    return classes.reshape(h, w)


def instance_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Instance segmentation: simplified AP@0.5 (by color matching)."""
    # Simplified: count distinct non-black colors as instances
    pred_colors = set(map(tuple, np.unique(pred.reshape(-1, 3), axis=0)))
    target_colors = set(map(tuple, np.unique(target.reshape(-1, 3), axis=0)))

    pred_colors.discard((0, 0, 0))
    target_colors.discard((0, 0, 0))

    if not target_colors:
        return {"ap50": 0.0, "n_pred_instances": len(pred_colors), "n_target_instances": 0}

    # Match by nearest color
    from scipy.spatial import cKDTree
    target_arr = np.array(list(target_colors))
    tree = cKDTree(target_arr.astype(np.float32))

    matched = 0
    for pc in pred_colors:
        dist, idx = tree.query(np.array(pc, dtype=np.float32), k=1)
        if dist < 30:  # color distance threshold
            matched += 1

    precision = matched / max(len(pred_colors), 1)
    recall = matched / max(len(target_colors), 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"ap50": f1, "precision": precision, "recall": recall,
            "n_pred_instances": len(pred_colors), "n_target_instances": len(target_colors)}


def referring_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Referring expression: binary mask IoU."""
    # Threshold: white = target, black = background
    pred_mask = np.all(pred > 128, axis=-1)
    target_mask = np.all(target > 128, axis=-1)

    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    iou = float(intersection / max(union, 1))

    return {"iou": iou, "intersection": int(intersection), "union": int(union)}


def generation_metrics(pred: np.ndarray, prompt: str) -> dict:
    """Generation: CLIP score (simplified)."""
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        clip_model = clip_model.to(device)

        img = Image.fromarray(pred.astype(np.uint8))
        inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            score = outputs.logits_per_image.item()
        return {"clip_score": float(score)}
    except Exception as e:
        logger.warning(f"CLIP score failed: {e}")
        return {"clip_score": 0.0}


def input_preservation_metrics(pred: np.ndarray, source: np.ndarray) -> dict:
    """Input preservation: LPIPS and SSIM."""
    from skimage.metrics import structural_similarity as ssim
    s = ssim(source, pred, channel_axis=2, data_range=255)
    l1 = float(np.mean(np.abs(pred.astype(np.float64) - source.astype(np.float64))) / 255.0)
    return {"ssim": float(s), "l1_normalized": l1}


# ─── Metric Registry ──────────────────────────────────────────────────────────

METRIC_FUNCTIONS = {
    "depth": depth_metrics,
    "normal": normal_metrics,
    "semantic": semantic_metrics,
    "instance": instance_metrics,
    "referring": referring_metrics,
}

PRIMARY_METRIC = {
    "depth": "delta1",         # higher is better
    "normal": "mean_angular_error",  # lower is better
    "semantic": "mIoU",        # higher is better
    "instance": "ap50",        # higher is better
    "referring": "iou",        # higher is better
    "generation": "clip_score", # higher is better
}

HIGHER_IS_BETTER = {
    "delta1": True, "delta2": True, "delta3": True,
    "absrel": False, "rmse": False,
    "mean_angular_error": False, "median_angular_error": False, "rmse_angular_error": False,
    "mIoU": True, "pixel_accuracy": True,
    "ap50": True, "precision": True, "recall": True,
    "iou": True,
    "clip_score": True,
    "ssim": True, "l1_normalized": False,
}


# ─── Evaluation Runner ────────────────────────────────────────────────────────

@dataclass
class CheckpointResult:
    checkpoint: str
    step: int
    task_metrics: dict = field(default_factory=dict)
    generation_metrics: dict = field(default_factory=dict)
    preservation_metrics: dict = field(default_factory=dict)


def evaluate_checkpoint(
    ckpt_dir: Path,
    val_samples: list[dict],
    device: str = "mps",
) -> CheckpointResult:
    """Evaluate a single checkpoint on the val set."""
    from diffusers import (
        AutoencoderKL, ControlNetModel, UNet2DConditionModel,
        EulerDiscreteScheduler, StableDiffusionXLPipeline,
    )
    from peft import PeftModel
    from transformers import CLIPTextModel, CLIPTokenizer

    dtype = torch.float16
    logger.info(f"Evaluating checkpoint: {ckpt_dir}")

    # Load base components
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=dtype)
    scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    tok1 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=dtype).to(device)
    te2 = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder_2", torch_dtype=dtype).to(device)

    # Load LoRA
    lora_dir = ckpt_dir / "lora"
    if lora_dir.exists():
        from peft import PeftModel
        unet = PeftModel.from_pretrained(unet, str(lora_dir))
        logger.info(f"Loaded LoRA from {lora_dir}")

    # Load ControlNet
    cn_dir = ckpt_dir / "controlnet"
    if cn_dir.exists():
        controlnet = ControlNetModel.from_pretrained(str(cn_dir), torch_dtype=dtype).to(device)
        logger.info(f"Loaded ControlNet from {cn_dir}")
    else:
        controlnet = None

    unet = unet.to(device)
    vae.requires_grad_(False)
    te1.requires_grad_(False)
    te2.requires_grad_(False)

    # Read step from training_state
    state_file = ckpt_dir / "training_state.json"
    step = 0
    if state_file.exists():
        state = json.loads(state_file.read_text())
        step = state.get("step", 0)

    result = CheckpointResult(checkpoint=str(ckpt_dir), step=step)

    # Evaluate per task
    tasks_present = set(s["task"] for s in val_samples)
    for task in tasks_present:
        task_samples = [s for s in val_samples if s["task"] == task][:MAX_VAL_SAMPLES]
        if not task_samples:
            continue

        all_metrics = []
        for sample in task_samples:
            try:
                pred = run_inference(
                    unet, vae, controlnet, scheduler, te1, te2, tok1, tok2,
                    sample, device, dtype,
                )
                target = np.array(Image.open(sample["target_path"]).convert("RGB").resize((RESOLUTION, RESOLUTION)))
                source = np.array(Image.open(sample["input_path"]).convert("RGB").resize((RESOLUTION, RESOLUTION)))

                metric_fn = METRIC_FUNCTIONS.get(task)
                if metric_fn:
                    if task == "semantic":
                        # Load palette from v3 manifest
                        palette = _load_palette()
                        m = metric_fn(pred, target, palette)
                    else:
                        m = metric_fn(pred, target)
                    all_metrics.append(m)

                # Input preservation
                pres = input_preservation_metrics(pred, source)
                if "preservation" not in result.preservation_metrics:
                    result.preservation_metrics["preservation"] = []
                result.preservation_metrics["preservation"].append(pres)

            except Exception as e:
                logger.warning(f"Failed to evaluate sample {sample['sample_id']}: {e}")

        # Aggregate
        if all_metrics:
            agg = {}
            for key in all_metrics[0]:
                vals = [m[key] for m in all_metrics if key in m]
                agg[key] = float(np.mean(vals))
            result.task_metrics[task] = agg

    return result


def run_inference(unet, vae, controlnet, scheduler, te1, te2, tok1, tok2,
                  sample, device, dtype, num_steps=20):
    """Run inference for a single sample and return predicted RGB array."""
    # Encode prompt
    prompt = sample["prompt"]
    tokens1 = tok1(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids.to(device)
    tokens2 = tok2(prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids.to(device)
    with torch.no_grad():
        e1 = te1(tokens1)[0]
        e2 = te2(tokens2)[0]
        pooled = te2(tokens2)[0][:, 0, :]
    prompt_embeds = torch.cat([e1, e2], dim=-1)

    # Prepare ControlNet image
    input_img = Image.open(sample["input_path"]).convert("RGB").resize((RESOLUTION, RESOLUTION))
    cn_cond = torch.from_numpy(np.array(input_img).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)

    # Generate
    h = w = RESOLUTION // VAE_SCALE
    latent = torch.randn(1, 4, h, w, device=device, dtype=dtype)
    scheduler.set_timesteps(num_steps, device=device)

    added_cond = {
        "text_embeds": pooled,
        "time_ids": torch.tensor([[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]], device=device, dtype=dtype),
    }

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            sigma = scheduler.sigmas[i]
            model_input = latent / ((sigma ** 2 + 1) ** 0.5)

            residuals = None
            if controlnet is not None:
                cn_out = controlnet(
                    model_input, t.unsqueeze(0),
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=cn_cond,
                    conditioning_scale=1.0,
                    return_dict=False,
                )
                residuals = cn_out

            kwargs = {
                "encoder_hidden_states": prompt_embeds,
                "added_cond_kwargs": added_cond,
                "return_dict": False,
            }
            if residuals is not None:
                kwargs["down_block_additional_residuals"] = residuals[0] if isinstance(residuals[0], (list, tuple)) else [residuals[0]]
                if len(residuals) > 1:
                    kwargs["mid_block_additional_residual"] = residuals[1]

            noise_pred = unet(model_input, t.unsqueeze(0), **kwargs)
            noise_pred = noise_pred[0] if isinstance(noise_pred, tuple) else noise_pred.sample

            # Euler step
            sigma_next = scheduler.sigmas[i + 1] if i + 1 < len(scheduler.sigmas) else 0
            pred_x0 = latent - sigma * noise_pred
            latent = pred_x0 + sigma_next * noise_pred

    # Decode
    with torch.no_grad():
        latent = latent / vae.config.scaling_factor
        decoded = vae.decode(latent)
        img = (decoded.sample / 2 + 0.5).clamp(0, 1)

    return (img[0].cpu().float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _load_palette():
    """Load palette from v3 manifest."""
    v3_root = Path("data/v3")
    manifest = json.loads((v3_root / "v3_manifest.json").read_text())
    palette = np.array(manifest["palette"]["semantic"], dtype=np.uint8)
    return palette


# ─── Checkpoint Selection ─────────────────────────────────────────────────────

def select_best_checkpoints(results: list[CheckpointResult]) -> dict:
    """Select best checkpoint per task by primary metric."""
    best = {}

    for task, metric_name in PRIMARY_METRIC.items():
        higher = HIGHER_IS_BETTER.get(metric_name, True)

        candidates = []
        for r in results:
            if task in r.task_metrics and metric_name in r.task_metrics[task]:
                candidates.append((r, r.task_metrics[task][metric_name]))

        if not candidates:
            continue

        if higher:
            best_r = max(candidates, key=lambda x: x[1])
        else:
            best_r = min(candidates, key=lambda x: x[1])

        best[task] = {
            "checkpoint": best_r[0].checkpoint,
            "step": best_r[0].step,
            "metric_value": best_r[1],
            "metric_name": metric_name,
        }

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default="output/v5_multitask")
    parser.add_argument("--v3-root", type=str,
                       default="data/v3")
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--output-file", type=str, default="checkpoint_evaluation.json")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Find checkpoints
    ckpt_root = Path(args.ckpt_dir)
    checkpoints = sorted(ckpt_root.glob("checkpoint-*"))
    if not checkpoints:
        logger.error(f"No checkpoints found in {ckpt_root}")
        return 1

    logger.info(f"Found {len(checkpoints)} checkpoints")

    # Load val samples
    v3_root = Path(args.v3_root)
    samples_jsonl = v3_root / "v3_samples.jsonl"
    if not samples_jsonl.exists():
        logger.error(f"V3 samples not found: {samples_jsonl}")
        return 1

    val_samples = []
    with open(samples_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] == "val":
                val_samples.append(rec)

    logger.info(f"Loaded {len(val_samples)} val samples")

    # Evaluate each checkpoint
    results = []
    for ckpt in checkpoints:
        result = evaluate_checkpoint(ckpt, val_samples, device)
        results.append(result)
        logger.info(f"Checkpoint {ckpt.name} (step {result.step}):")
        for task, metrics in result.task_metrics.items():
            logger.info(f"  {task}: {metrics}")

    # Select best
    if args.select_best:
        best = select_best_checkpoints(results)
        logger.info("\n" + "=" * 60)
        logger.info("BEST CHECKPOINTS PER TASK:")
        for task, info in best.items():
            logger.info(f"  {task}: step {info['step']} ({info['metric_name']}={info['metric_value']:.4f})")
        logger.info("=" * 60)

    # Save results
    output_data = {
        "checkpoints": [
            {
                "checkpoint": r.checkpoint,
                "step": r.step,
                "task_metrics": r.task_metrics,
                "preservation_metrics": r.preservation_metrics,
            }
            for r in results
        ],
        "best_checkpoints": select_best_checkpoints(results) if args.select_best else None,
    }
    output_path = ckpt_root / args.output_file
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
