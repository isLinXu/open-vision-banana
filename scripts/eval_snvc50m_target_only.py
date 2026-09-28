#!/usr/bin/env python3
"""Evaluate SNVC-50M target-only trained model.

Since training uses noise→target flow matching (no input RGB), evaluation:
1. Generates samples from random noise for each task
2. Reads GT targets from compressed tar.gz for distribution comparison
3. Computes quantitative metrics on distribution-level statistics
4. Produces visualization grids (Generated samples + GT comparisons)
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from open_vision_banana.model_registry import (
    PipelineComponents,
    apply_lora,
    get_model_config,
    load_pipeline,
)
from open_vision_banana.inference.sana_img2img import flow_matching_inference

# ─── Task prompts (from training) ───────────────────────────────────────────
_TASK_PROMPTS = {
    "depth": "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map.",
    "normal": "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
    "seg": "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
}


# ─── Tar reader for loading GT targets ──────────────────────────────────────
def create_tar_reader(tar_root: Path, gzip_index: Path):
    """Create an IndexedTarReader for reading GT targets from compressed volumes."""
    from open_vision_banana.data.tar_dataset import IndexedTarReader
    all_volumes = sorted(tar_root.glob("*.tar.gz*"))
    return IndexedTarReader(all_volumes, gzip_index)


def load_task_index(task: str, index_dir: Path) -> dict:
    """Load per-task tar index and return the files dict."""
    idx_path = index_dir / f"tar_index_{task}.json"
    with open(idx_path) as f:
        idx = json.load(f)
    # Index format: {task, volumes, files, count}
    files = idx.get("files", idx)
    print(f"  Loaded {task} index: {len(files)} files from {len(idx.get('volumes', []))} volumes")
    return files


def read_target_from_tar(reader, tar_path: str, offset: int, size: int, target_size: int = 512) -> np.ndarray:
    """Read a target image from the compressed tar stream."""
    data = reader.read_at(offset, size)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(img)


# ─── Visualization helpers ──────────────────────────────────────────────────
def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().float()
    if t.min() < -0.1:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    if t.dim() == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)
    elif t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_comparison_row(images: list, labels: list[str]) -> Image.Image:
    """Create horizontal comparison row."""
    w, h = images[0].size
    label_h = 30
    n = len(images)
    canvas = Image.new("RGB", (w * n, h + label_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * w, label_h))
        draw.text((i * w + 8, 5), label, fill=(255, 255, 255), font=font)
    return canvas


def make_grid(images: list, cols: int = 4) -> Image.Image:
    """Arrange images in a grid."""
    if not images:
        return None
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (w * cols, h * rows), (30, 30, 30))
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        canvas.paste(img, (c * w, r * h))
    return canvas


# ─── Metrics ────────────────────────────────────────────────────────────────
def compute_depth_metrics(pred: np.ndarray, gt_samples: list[np.ndarray]) -> dict:
    """Compute distribution-level depth metrics."""
    pred_gray = np.mean(pred, axis=-1)
    metrics = {}
    gt_means = [np.mean(np.mean(g, axis=-1)) for g in gt_samples]
    gt_stds = [np.std(np.mean(g, axis=-1)) for g in gt_samples]
    metrics["pred_mean"] = float(np.mean(pred_gray))
    metrics["pred_std"] = float(np.std(pred_gray))
    metrics["gt_mean"] = float(np.mean(gt_means))
    metrics["gt_std"] = float(np.mean(gt_stds))
    metrics["mean_diff"] = abs(metrics["pred_mean"] - metrics["gt_mean"])
    # Relative RMSE (normalized)
    pred_norm = (pred_gray - pred_gray.mean()) / (pred_gray.std() + 1e-8)
    gt_norms = [(np.mean(g, axis=-1) - np.mean(np.mean(g, axis=-1))) / (np.std(np.mean(g, axis=-1)) + 1e-8) for g in gt_samples]
    rmses = []
    for gt_n in gt_norms:
        rmse = np.sqrt(np.mean((pred_norm - gt_n) ** 2))
        rmses.append(rmse)
    metrics["normalized_rmse_mean"] = float(np.mean(rmses))
    return metrics


def compute_normal_metrics(pred: np.ndarray, gt_samples: list[np.ndarray]) -> dict:
    """Compute normal map angular error metrics."""
    # Convert RGB to normal vectors [0,1] -> [-1,1]
    pred_n = pred.astype(np.float32) / 255.0 * 2.0 - 1.0
    pred_n = pred_n / (np.linalg.norm(pred_n, axis=-1, keepdims=True) + 1e-8)
    
    angular_errors = []
    for gt in gt_samples:
        gt_n = gt.astype(np.float32) / 255.0 * 2.0 - 1.0
        gt_n = gt_n / (np.linalg.norm(gt_n, axis=-1, keepdims=True) + 1e-8)
        cos_sim = np.sum(pred_n * gt_n, axis=-1)
        cos_sim = np.clip(cos_sim, -1, 1)
        angle = np.degrees(np.arccos(cos_sim))
        angular_errors.append(float(np.mean(angle)))
    
    metrics = {
        "mean_angular_error": float(np.mean(angular_errors)),
        "median_angular_error": float(np.median(angular_errors)),
        "pred_normal_mean": float(np.mean(pred_n)),
        "pred_normal_std": float(np.std(pred_n)),
    }
    return metrics


def compute_seg_metrics(pred: np.ndarray, gt_samples: list[np.ndarray]) -> dict:
    """Compute segmentation palette consistency metrics."""
    # Count unique colors in prediction
    pred_flat = pred.reshape(-1, 3)
    unique_colors = np.unique(pred_flat, axis=0)
    
    # Check how much of the image uses a small palette (mode coverage)
    from collections import Counter
    color_tuples = [tuple(c) for c in pred_flat]
    color_counts = Counter(color_tuples)
    total_pixels = len(color_tuples)
    top_k = color_counts.most_common(20)
    top_k_coverage = sum(c for _, c in top_k) / total_pixels
    
    # Check boundary sharpness (how many pixels change color vs smooth)
    gray = np.mean(pred, axis=-1)
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    boundary_ratio = (np.mean(grad_x > 5) + np.mean(grad_y > 5)) / 2
    
    metrics = {
        "unique_colors": len(unique_colors),
        "top20_color_coverage": float(top_k_coverage),
        "boundary_sharpness_ratio": float(boundary_ratio),
        "pred_mean_rgb": [float(x) for x in np.mean(pred, axis=(0, 1))],
    }
    return metrics


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate SNVC-50M target-only model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to lora.pt")
    parser.add_argument("--tar-root", type=Path, default=Path("/media/linxu/thinkHub/datasets/SenseNova-Vision-Corpus-50M"))
    parser.add_argument("--gzip-index", type=Path, default=Path("output/snvc50m_manifest/gzip_index/stream.gzidx"))
    parser.add_argument("--task-index-dir", type=Path, default=Path("output/snvc50m_manifest/task_indexes"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/eval_snvc50m_target_only"))
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--gt-samples", type=int, default=20, help="GT samples for distribution comparison")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(args.dtype, torch.bfloat16)

    # ── 1. Load model ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading Sana-600M-512px pipeline...")
    model_cfg = get_model_config("sana-600m-512px")
    components = load_pipeline(model_cfg, device=device, dtype=dtype)
    components.text_encoder.to(torch.float32)
    if dtype in (torch.float16, torch.bfloat16):
        components.vae.to(dtype)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = components.unet
    peft_model = apply_lora(
        backbone,
        config=model_cfg,
        rank=ckpt.get("lora_rank", 32),
        alpha=ckpt.get("lora_alpha", 64),
        use_rslora=True,
    )
    peft_model.load_state_dict(ckpt.get("lora_state_dict", ckpt.get("model", {})), strict=False)
    peft_model.to(device=device, dtype=dtype)
    peft_model.eval()
    print("Model loaded successfully!")

    # ── 2. Load tar reader for GT targets ──────────────────────────────────
    print("\nLoading gzip index for GT target access...")
    tar_reader = create_tar_reader(args.tar_root, args.gzip_index)
    
    # Load task indexes and sample GT entries
    task_gt_entries = {}
    for task in ["depth", "normal", "seg"]:
        idx = load_task_index(task, args.task_index_dir)
        all_keys = list(idx.keys())
        # Random sample for GT comparison
        gt_indices = np.random.choice(len(all_keys), min(args.gt_samples, len(all_keys)), replace=False)
        task_gt_entries[task] = [(all_keys[i], idx[all_keys[i]]) for i in gt_indices]
        print(f"  {task}: {len(idx)} total entries, sampled {len(gt_indices)} for GT comparison")

    # ── 3. Generate samples and compute metrics ────────────────────────────
    all_metrics = {}
    
    for task_name in ["depth", "normal", "seg"]:
        print(f"\n{'=' * 60}")
        print(f"Processing task: {task_name}")
        print(f"{'=' * 60}")
        
        # Generate samples from noise
        generated_images = []
        gen_tensors = []
        prompt = _TASK_PROMPTS[task_name]
        
        for i in range(args.samples_per_task):
            # Create dummy input (just for shape, will be replaced with noise)
            dummy_input = torch.zeros(1, 3, args.resolution, args.resolution, device=device, dtype=dtype)
            
            with torch.no_grad():
                output = flow_matching_inference(
                    peft_model,
                    components.vae,
                    components,
                    dummy_input,
                    prompts=[prompt],
                    device=device,
                    model_dtype=dtype,
                    num_steps=args.num_inference_steps,
                    guidance_scale=1.0,
                    use_img2img=False,  # noise→target mode
                )
            
            gen_pil = tensor_to_pil(output[0].cpu())
            generated_images.append(gen_pil)
            gen_tensors.append(np.array(gen_pil))
            print(f"  Generated [{i+1}/{args.samples_per_task}]: "
                  f"min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        # Load GT targets from tar
        print(f"\n  Loading {len(task_gt_entries[task_name])} GT targets from compressed volumes...")
        gt_images = []
        for tar_path, info in task_gt_entries[task_name]:
            try:
                gt_arr = read_target_from_tar(tar_reader, tar_path, info["offset"], info["size"], args.resolution)
                gt_images.append(gt_arr)
            except Exception as e:
                print(f"  Warning: Failed to read {tar_path}: {e}")
        
        print(f"  Loaded {len(gt_images)} GT targets")
        
        # Compute metrics
        gen_arr = np.stack(gen_tensors)  # [N, H, W, 3]
        
        if task_name == "depth":
            metrics = compute_depth_metrics(np.mean(gen_arr, axis=0), gt_images)
        elif task_name == "normal":
            metrics = compute_normal_metrics(np.mean(gen_arr, axis=0), gt_images)
        elif task_name == "seg":
            metrics = compute_seg_metrics(np.mean(gen_arr, axis=0), gt_images)
        else:
            metrics = {}
        
        # Add generation quality metrics
        metrics["gen_mean"] = float(np.mean(gen_arr))
        metrics["gen_std"] = float(np.std(gen_arr))
        metrics["gen_range"] = [float(np.min(gen_arr)), float(np.max(gen_arr))]
        if gt_images:
            gt_stack = np.stack(gt_images)
            metrics["gt_mean"] = float(np.mean(gt_stack))
            metrics["gt_std"] = float(np.std(gt_stack))
        
        all_metrics[task_name] = metrics
        
        print(f"\n  Metrics for {task_name}:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
        
        # ── Save visualizations ────────────────────────────────────────────
        task_dir = args.output_dir / task_name
        task_dir.mkdir(exist_ok=True)
        
        # Save individual generated samples
        for i, img in enumerate(generated_images):
            img.save(task_dir / f"generated_{i:03d}.png")
        
        # Save GT samples for comparison
        for i, gt_arr in enumerate(gt_images[:8]):
            Image.fromarray(gt_arr).save(task_dir / f"gt_{i:03d}.png")
        
        # Create comparison grid: Generated samples
        grid_gen = make_grid(generated_images, cols=4)
        if grid_gen:
            grid_gen.save(task_dir / "generated_grid.png")
        
        # Create GT grid
        gt_pils = [Image.fromarray(g) for g in gt_images[:8]]
        grid_gt = make_grid(gt_pils, cols=4)
        if grid_gt:
            grid_gt.save(task_dir / "gt_grid.png")
        
        # Create side-by-side: Generated | GT
        if gt_pils:
            comparison_rows = []
            for i in range(min(4, len(generated_images), len(gt_pils))):
                row = make_comparison_row(
                    [generated_images[i], gt_pils[i]],
                    ["Generated (noise→target)", "GT (from tar)"]
                )
                comparison_rows.append(row)
            combined = make_grid(comparison_rows, cols=1)
            if combined:
                combined.save(task_dir / "comparison.png")
        
        print(f"  Visualizations saved to: {task_dir}")
    
    # ── 4. Save metrics summary ────────────────────────────────────────────
    metrics_path = args.output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n{'=' * 60}")
    print(f"Metrics saved to: {metrics_path}")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: sana-600m-512px (LoRA rank={ckpt.get('lora_rank', 32)})")
    print(f"Tasks: {list(all_metrics.keys())}")
    print(f"Samples per task: {args.samples_per_task}")
    print(f"GT samples for comparison: {args.gt_samples}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"W&B: https://wandb.ai/linxu/open-vision-banana")


if __name__ == "__main__":
    main()
