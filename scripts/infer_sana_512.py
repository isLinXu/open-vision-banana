#!/usr/bin/env python3
"""Inference and visualization for Sana_600M_512px LoRA checkpoint.

Loads a trained LoRA checkpoint and runs img2img flow-matching inference
on validation set samples (or custom prompts), producing side-by-side
comparison images:  Input | Predicted | Target.

Usage:
    # Infer on validation set (one sample per task)
    python scripts/infer_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --checkpoint output/sana_512_vb/checkpoint-6000_best/lora.pt \
        --output-dir output/sana_512_vb/inference

    # Custom prompt with a specific input image
    python scripts/infer_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --checkpoint output/sana_512_vb/checkpoint-6000_best/lora.pt \
        --output-dir output/sana_512_vb/inference \
        --custom-prompt "Depth estimation: convert this image to a depth map" \
        --num-samples 4

    # Full validation set evaluation with PSNR metrics
    python scripts/infer_sana_512.py \
        --data-dir /path/to/vb_snvc_train_v3 \
        --checkpoint output/sana_512_vb/checkpoint-6000_best/lora.pt \
        --output-dir output/sana_512_vb/inference \
        --eval-mode
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

import torch
import torch.nn.functional as F
from PIL import Image

# Use HF mirror for models
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data import build_v3_datasets, load_v3_train_val_split
from open_vision_banana.inference.sana_img2img import (
    encode_text_sana,
    flow_matching_inference,
    vae_decode_sana,
    vae_encode_sana,
)
from open_vision_banana.model_registry import (
    PipelineComponents,
    apply_lora,
    get_model_config,
    load_pipeline,
)

logger = logging.getLogger(__name__)

# LoRA targets matching training script
SANA_LORA_TARGETS = [
    "linear",
    "proj_out",
    "conv_inverted",
    "conv_point",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference and visualization for Sana_600M_512px LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument("--data-dir", required=True, help="Path to vb_snvc_train_v3 dataset")
    p.add_argument(
        "--pair-manifest",
        type=Path,
        default=None,
        help=(
            "Explicit v5 JSONL of annotation-backed input/target pairs. "
            "Required when validating the flat v5_full dataset."
        ),
    )
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--tasks", nargs="+", default=None, help="Subset of tasks (default: all)")
    p.add_argument("--split", choices=["train", "val"], default="val", help="Manifest split to visualize")

    # Model
    p.add_argument("--model-name", default="sana-600m-512px")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a lora.pt checkpoint or PEFT lora_adapter directory",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-rslora", action="store_true")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])

    # Inference
    p.add_argument("--num-samples", type=int, default=3, help="Samples per task for visualization")
    p.add_argument("--num-inference-steps", type=int, default=20, help="Denoising steps")
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale; 1.0 disables guidance.",
    )
    p.add_argument(
        "--no-img2img-flow",
        action="store_true",
        help="Disable img2img flow (use noise->target instead)",
    )
    p.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        help="Custom prompt for single-image inference (bypasses dataset)",
    )
    p.add_argument(
        "--eval-mode",
        action="store_true",
        help="Run full validation evaluation with PSNR metrics",
    )
    p.add_argument("--max-eval-samples", type=int, default=100, help="Max samples for eval mode")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default="output/sana_512_vb/inference")

    return p.parse_args()


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


# ── Visualization ───────────────────────────────────────────────────


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a single [3, H, W] tensor in [0, 1] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.clamp(0.0, 1.0).cpu()
    tensor = (tensor * 255).byte()
    tensor = tensor.permute(1, 2, 0)
    return Image.fromarray(tensor.numpy())


def create_comparison_grid(
    input_img: torch.Tensor,
    predicted_img: torch.Tensor,
    target_img: torch.Tensor,
    task_name: str,
    prompt: str,
    sample_idx: int,
) -> Image.Image:
    """Create a side-by-side comparison: Input | Predicted | Target.

    Each image is [3, H, W] in [0, 1].
    """
    pil_input = tensor_to_pil(input_img)
    pil_pred = tensor_to_pil(predicted_img)
    pil_target = tensor_to_pil(target_img)

    w, h = pil_input.size
    pad = 10
    label_h = 30

    total_w = w * 3 + pad * 4
    total_h = h + label_h * 3 + pad * 2

    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(grid)

    # Title
    title = f"Task: {task_name} | Sample {sample_idx}"
    draw.text((pad, 5), title, fill=(0, 0, 0))

    # Prompt (truncated)
    prompt_short = prompt[:120] + "..." if len(prompt) > 120 else prompt
    draw.text((pad, label_h + 5), f"Prompt: {prompt_short}", fill=(50, 50, 50))

    # Labels for each column
    y_offset = label_h * 2 + pad
    draw.text((pad, y_offset - label_h + 5), "Input", fill=(200, 0, 0))
    draw.text((w + pad * 2, y_offset - label_h + 5), "Predicted", fill=(0, 150, 0))
    draw.text((w * 2 + pad * 3, y_offset - label_h + 5), "Target", fill=(0, 0, 200))

    # Paste images
    grid.paste(pil_input, (pad, y_offset))
    grid.paste(pil_pred, (w + pad * 2, y_offset))
    grid.paste(pil_target, (w * 2 + pad * 3, y_offset))

    return grid


# ── Metrics ─────────────────────────────────────────────────────────


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between predicted and target images ([B, 3, H, W] in [0, 1])."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def compute_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute L1 distance."""
    return F.l1_loss(pred, target).item()


# ── Main inference flow ─────────────────────────────────────────────


def load_model_and_lora(args, device, dtype):
    """Load the Sana pipeline and apply LoRA checkpoint."""
    logger.info("Loading Sana_600M_512px_diffusers...")
    model_cfg = get_model_config(args.model_name)
    logger.info(f"  model_id: {model_cfg.model_id}")
    logger.info(f"  family: {model_cfg.family}")
    logger.info(f"  VAE downsample: {model_cfg.vae_downsample_factor}x")

    components = load_pipeline(
        model_cfg,
        device=device,
        dtype=dtype,
        enable_gradient_checkpointing=False,
    )

    # Sana dtype adjustments
    if dtype == torch.float16:
        logger.info("Applying Sana dtype adjustments: text_encoder->fp32, vae->bf16")
        components.text_encoder.to(torch.float32)
        components.vae.to(torch.bfloat16)

    backbone = components.unet
    vae = components.vae

    backbone_params = sum(p.numel() for p in backbone.parameters())
    logger.info(f"  Backbone params: {backbone_params / 1e6:.1f}M")

    ckpt_path = Path(args.checkpoint)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    if ckpt_path.is_dir():
        adapter_file = ckpt_path / "adapter_model.safetensors"
        if not adapter_file.is_file():
            raise FileNotFoundError(
                f"PEFT adapter directory is missing {adapter_file.name}: {ckpt_path}"
            )
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(backbone, str(ckpt_path), is_trainable=False)
        logger.info("  Restored PEFT adapter weights")
    else:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info("Applying LoRA adapters...")
        peft_model = apply_lora(
            backbone,
            config=model_cfg,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            use_rslora=not args.no_rslora,
        )
        ckpt = torch.load(str(ckpt_path), map_location=device)
        lora_state = ckpt.get("lora_state_dict", ckpt)
        peft_model.load_state_dict(lora_state, strict=False)
        step = ckpt.get("step", 0)
        val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", "N/A"))
        logger.info(f"  Restored step={step}, val_loss={val_loss}")

    peft_model.eval()

    return peft_model, vae, components, model_cfg


def build_val_datasets(args, data_root):
    """Build validation datasets from v3 data or an explicit v5 pair manifest."""
    if args.pair_manifest is not None:
        manifest_path = args.pair_manifest.expanduser().resolve()
        if not manifest_path.is_file():
            logger.error("Pair manifest not found: %s", manifest_path)
            return []

        # Reuse the training dataset implementation so validation preserves
        # the annotation-backed v5 source/target semantics.
        from train_sana_512 import ManifestVisionDataset

        task_aliases = {"segmentation": "semantic", "seg": "semantic"}
        requested = args.tasks or ["depth", "normal", "semantic", "structure"]
        manifest_tasks = {task_aliases.get(task, task) for task in requested}
        val_datasets = [
            ManifestVisionDataset(manifest_path, args.resolution, args.split, {task})
            for task in sorted(manifest_tasks)
        ]
        val_datasets = [ds for ds in val_datasets if len(ds) > 0]
        logger.info("Detected explicit v5 pair manifest: %s", manifest_path)
        for ds in val_datasets:
            logger.info("  [val] %s/%s: %d samples", ds.dir_name, ds.task, len(ds))
        return val_datasets

    v3_split = load_v3_train_val_split(data_root)
    if v3_split is None and not (data_root / "manifest.json").exists():
        logger.error("Dataset must be in v3 format")
        sys.exit(1)

    all_tasks = [
        "depth", "normal", "segmentation", "structure",
        "instance", "keypoint", "object_detection",
        "depth_edge", "referring", "multiview", "specular_removal",
    ]
    tasks_to_load = args.tasks if args.tasks else all_tasks

    val_datasets = build_v3_datasets(
        data_root,
        resolution=args.resolution,
        split=args.split,
        use_split_file=v3_split is not None,
    )
    val_datasets = [
        ds for ds in val_datasets if ds.task in tasks_to_load or ds.dir_name in tasks_to_load
    ]

    for ds in val_datasets:
        logger.info(f"  [val] {ds.dir_name}/{ds.task}: {len(ds)} samples")

    return val_datasets


def run_visualization(args, peft_model, vae, components, model_cfg, val_datasets, device, dtype):
    """Run inference on a few samples per task and save comparison grids."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = next(peft_model.parameters()).dtype
    max_length = model_cfg.text_encoder_max_length
    use_img2img = not args.no_img2img_flow

    logger.info(f"Inference config: steps={args.num_inference_steps}, "
                f"guidance={args.guidance_scale}, img2img={use_img2img}, "
                f"samples_per_task={args.num_samples}")

    all_results = []

    for ds in val_datasets:
        task_name = ds.task
        dir_name = ds.dir_name
        n_samples = min(args.num_samples, len(ds))
        logger.info(f"\nProcessing task: {task_name}/{dir_name} ({n_samples} samples)")

        for i in range(n_samples):
            try:
                sample = ds[i]
            except Exception as e:
                logger.warning(f"  Skipping sample {i}: {e}")
                continue

            input_img = sample["image"]  # [3, H, W] in [-1, 1]
            target_rgb = sample["target_rgb"]  # [3, H, W] in [0, 1]
            prompt = sample.get("prompt", "")

            # Prepare batch (B=1)
            input_batch = input_img.unsqueeze(0).to(device=device, dtype=model_dtype)

            t0 = time.time()
            predicted = flow_matching_inference(
                peft_model,
                vae,
                components,
                input_batch,
                [prompt],
                device,
                model_dtype,
                num_steps=args.num_inference_steps,
                use_img2img=use_img2img,
                max_length=max_length,
                guidance_scale=args.guidance_scale,
            )
            elapsed = time.time() - t0

            pred_img = predicted[0].cpu()
            target_img = target_rgb.cpu()
            input_for_display = ((input_img.cpu() + 1.0) / 2.0).clamp(0, 1)

            # Metrics
            psnr = compute_psnr(pred_img.unsqueeze(0), target_img.unsqueeze(0))
            l1 = compute_l1(pred_img.unsqueeze(0), target_img.unsqueeze(0))

            logger.info(f"  [{task_name} #{i}] PSNR={psnr:.2f} dB, L1={l1:.4f}, time={elapsed:.1f}s")

            all_results.append({
                "task": task_name,
                "dir": dir_name,
                "sample_idx": i,
                "prompt": prompt,
                "psnr": psnr,
                "l1": l1,
                "time_sec": elapsed,
            })

            # Save comparison grid
            grid = create_comparison_grid(
                input_for_display,
                pred_img,
                target_img,
                task_name,
                prompt,
                i,
            )
            grid_path = output_dir / f"{task_name}_{dir_name}_sample{i}.png"
            grid.save(str(grid_path))
            logger.info(f"    Saved: {grid_path}")

    # Save metrics JSON
    metrics_path = output_dir / "inference_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nMetrics saved to {metrics_path}")

    # Print summary
    if all_results:
        avg_psnr = sum(r["psnr"] for r in all_results) / len(all_results)
        avg_l1 = sum(r["l1"] for r in all_results) / len(all_results)
        avg_time = sum(r["time_sec"] for r in all_results) / len(all_results)
        logger.info(f"\n{'='*60}")
        logger.info(f"Inference Summary ({len(all_results)} samples)")
        logger.info(f"{'='*60}")
        logger.info(f"  Avg PSNR: {avg_psnr:.2f} dB")
        logger.info(f"  Avg L1:   {avg_l1:.4f}")
        logger.info(f"  Avg time: {avg_time:.1f}s per sample")
        logger.info(f"  Steps:    {args.num_inference_steps}")

        # Per-task breakdown
        tasks_seen = {}
        for r in all_results:
            t = r["task"]
            if t not in tasks_seen:
                tasks_seen[t] = []
            tasks_seen[t].append(r["psnr"])

        logger.info(f"\nPer-task PSNR:")
        for task, psnrs in sorted(tasks_seen.items()):
            avg = sum(psnrs) / len(psnrs)
            logger.info(f"  {task:25s}: {avg:.2f} dB ({len(psnrs)} samples)")


def run_eval_mode(args, peft_model, vae, components, model_cfg, val_datasets, device, dtype):
    """Run full validation evaluation with metrics."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dtype = next(peft_model.parameters()).dtype
    max_length = model_cfg.text_encoder_max_length
    use_img2img = not args.no_img2img_flow

    all_psnr = []
    all_l1 = []
    total_samples = 0
    per_task = {}

    for ds in val_datasets:
        task_name = ds.task
        n = min(args.max_eval_samples, len(ds))
        logger.info(f"Evaluating {task_name}: {n} samples")

        for i in range(n):
            try:
                sample = ds[i]
            except Exception as e:
                logger.warning(f"  Skip {i}: {e}")
                continue

            input_img = sample["image"]
            target_rgb = sample["target_rgb"]
            prompt = sample.get("prompt", "")

            input_batch = input_img.unsqueeze(0).to(device=device, dtype=model_dtype)

            with torch.no_grad():
                predicted = flow_matching_inference(
                    peft_model,
                    vae,
                    components,
                    input_batch,
                    [prompt],
                    device,
                    model_dtype,
                    num_steps=args.num_inference_steps,
                    use_img2img=use_img2img,
                    max_length=max_length,
                    guidance_scale=args.guidance_scale,
                )

            pred_img = predicted[0].cpu()
            target_img = target_rgb.cpu()

            psnr = compute_psnr(pred_img.unsqueeze(0), target_img.unsqueeze(0))
            l1 = compute_l1(pred_img.unsqueeze(0), target_img.unsqueeze(0))

            all_psnr.append(psnr)
            all_l1.append(l1)
            total_samples += 1

            if task_name not in per_task:
                per_task[task_name] = {"psnr": [], "l1": []}
            per_task[task_name]["psnr"].append(psnr)
            per_task[task_name]["l1"].append(l1)

            if (i + 1) % 20 == 0:
                logger.info(f"  [{i+1}/{n}] PSNR={psnr:.2f}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results ({total_samples} samples)")
    logger.info(f"{'='*60}")
    logger.info(f"  Overall PSNR: {sum(all_psnr)/len(all_psnr):.2f} dB")
    logger.info(f"  Overall L1:   {sum(all_l1)/len(all_l1):.4f}")

    logger.info(f"\nPer-task:")
    for task in sorted(per_task.keys()):
        d = per_task[task]
        avg_psnr = sum(d["psnr"]) / len(d["psnr"])
        avg_l1 = sum(d["l1"]) / len(d["l1"])
        logger.info(f"  {task:25s}: PSNR={avg_psnr:.2f} dB, L1={avg_l1:.4f} ({len(d['psnr'])} samples)")

    # Save results
    results = {
        "total_samples": total_samples,
        "overall_psnr": sum(all_psnr) / len(all_psnr) if all_psnr else 0,
        "overall_l1": sum(all_l1) / len(all_l1) if all_l1 else 0,
        "per_task": {
            t: {
                "psnr": sum(d["psnr"]) / len(d["psnr"]),
                "l1": sum(d["l1"]) / len(d["l1"]),
                "count": len(d["psnr"]),
            }
            for t, d in per_task.items()
        },
    }
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {results_path}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    device = torch.device(args.device or auto_device())
    dtype = get_dtype(args.dtype)

    logger.info("=" * 70)
    logger.info("Open-Vision-Banana — Sana_600M_512px Inference")
    logger.info("=" * 70)
    logger.info(f"Device: {device} | dtype: {dtype}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output: {args.output_dir}")

    if args.guidance_scale < 0:
        raise ValueError("--guidance-scale must be non-negative")

    torch.manual_seed(args.seed)

    # Load model + LoRA
    peft_model, vae, components, model_cfg = load_model_and_lora(args, device, dtype)

    if args.custom_prompt:
        # Custom prompt mode: pick random input images from val set
        logger.info(f"\nCustom prompt mode: {args.custom_prompt}")
        data_root = Path(args.data_dir)
        val_datasets = build_val_datasets(args, data_root)

        if not val_datasets:
            logger.error("No validation datasets found")
            return

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_dtype = next(peft_model.parameters()).dtype

        for ds in val_datasets[:3]:  # Use first 3 datasets
            for i in range(min(args.num_samples, len(ds))):
                sample = ds[i]
                input_img = sample["image"]
                input_batch = input_img.unsqueeze(0).to(device=device, dtype=model_dtype)

                predicted = flow_matching_inference(
                    peft_model,
                    vae,
                    components,
                    input_batch,
                    [args.custom_prompt],
                    device,
                    model_dtype,
                    num_steps=args.num_inference_steps,
                    use_img2img=not args.no_img2img_flow,
                    max_length=model_cfg.text_encoder_max_length,
                    guidance_scale=args.guidance_scale,
                )

                pred_img = predicted[0].cpu()
                input_display = ((input_img.cpu() + 1.0) / 2.0).clamp(0, 1)

                # Save side-by-side
                pil_input = tensor_to_pil(input_display)
                pil_pred = tensor_to_pil(pred_img)

                w, h = pil_input.size
                combo = Image.new("RGB", (w * 2 + 20, h + 40), (255, 255, 255))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(combo)
                draw.text((5, 5), "Input", fill=(200, 0, 0))
                draw.text((w + 15, 5), "Predicted", fill=(0, 150, 0))
                combo.paste(pil_input, (5, 30))
                combo.paste(pil_pred, (w + 15, 30))

                path = output_dir / f"custom_{ds.task}_{i}.png"
                combo.save(str(path))
                logger.info(f"  Saved: {path}")

        return

    # Build validation datasets
    data_root = Path(args.data_dir)
    val_datasets = build_val_datasets(args, data_root)

    if not val_datasets:
        logger.error("No validation datasets loaded")
        return

    if args.eval_mode:
        run_eval_mode(args, peft_model, vae, components, model_cfg, val_datasets, device, dtype)
    else:
        run_visualization(args, peft_model, vae, components, model_cfg, val_datasets, device, dtype)


if __name__ == "__main__":
    main()
