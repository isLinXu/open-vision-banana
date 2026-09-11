#!/usr/bin/env python3
"""Robust qualitative evaluation with retry mechanism and stability checks."""
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
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


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image. Handles both [0, 1] and [-1, 1] ranges."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().float()
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    if t.dim() == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)
    elif t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def is_valid_output(tensor: torch.Tensor, threshold: float = 0.01) -> bool:
    """Check if output tensor is valid (not NaN/Inf, not uniform)."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    
    # Check if output is not uniform (variance check)
    if tensor.std() < threshold:
        return False
    
    return True


def load_image(path: str, size: int = 512) -> Image.Image:
    """Load and resize image."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return img


def make_comparison_grid(images: list[Image.Image], labels: list[str]) -> Image.Image:
    """Create a horizontal comparison grid with labels."""
    w, h = images[0].size
    label_h = 35
    n = len(images)
    
    canvas = Image.new("RGB", (w * n, h + label_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * w, label_h))
        draw.text((i * w + 10, 8), label, fill=(255, 255, 255), font=font)
    
    return canvas


def make_summary_grid(task_images: dict[str, list[Image.Image]], task_name: str) -> Image.Image:
    """Create a summary grid showing multiple samples for one task."""
    sample_ids = sorted(task_images.keys())
    n_samples = len(sample_ids)
    
    if n_samples == 0:
        return None
    
    w, h = task_images[sample_ids[0]][0].size
    label_h = 35
    col_labels = ["Input", "Ground Truth", "Model Output"]
    n_cols = 3
    
    canvas_w = w * n_cols
    canvas_h = (h + label_h) * n_samples
    
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
        small_font = font
    
    for col, label in enumerate(col_labels):
        draw.text((col * w + 10, 8), label, fill=(255, 255, 255), font=font)
    
    for row, sample_id in enumerate(sample_ids):
        y_offset = label_h + row * (h + label_h)
        draw.text((5, y_offset + 5), f"#{sample_id}", fill=(200, 200, 200), font=small_font)
        
        for col, img in enumerate(task_images[sample_id]):
            canvas.paste(img, (col * w, y_offset + label_h))
    
    return canvas


def run_inference_with_retry(
    peft_model,
    vae,
    components,
    input_tensor,
    prompt,
    device,
    dtype,
    num_steps,
    guidance_scale,
    max_retries=3,
):
    """Run inference with retry mechanism for stability."""
    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                output = flow_matching_inference(
                    peft_model,
                    vae,
                    components,
                    input_tensor,
                    prompts=[prompt],
                    device=device,
                    model_dtype=dtype,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
            
            # Check for NaN/Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"    Attempt {attempt+1}: NaN/Inf detected, retrying...")
                continue
            
            # Check for uniform output (variance too low)
            if output.std() < 0.01:
                print(f"    Attempt {attempt+1}: Uniform output detected (std={output.std():.6f}), retrying...")
                continue
            
            return output, True
            
        except Exception as e:
            print(f"    Attempt {attempt+1}: Exception {e}, retrying...")
            continue
    
    return None, False


def main():
    parser = argparse.ArgumentParser(description="Robust qualitative evaluation with retry")
    parser.add_argument("--checkpoint", required=True, help="Path to lora.pt")
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading Sana pipeline...")
    model_cfg = get_model_config("sana-1600m")
    device = torch.device(args.device)
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    
    components = load_pipeline(model_cfg, device=device, dtype=dtype)
    components.text_encoder.to(torch.float32)
    if dtype in (torch.float16, torch.bfloat16):
        components.vae.to(dtype)
    
    # Apply LoRA
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
    
    # Load pairs
    print(f"Loading pairs from: {args.pair_manifest}")
    pairs = []
    with open(args.pair_manifest) as f:
        for line in f:
            pairs.append(json.loads(line))
    
    task_pairs = {}
    for p in pairs:
        task = p["task"]
        if task not in task_pairs:
            task_pairs[task] = []
        task_pairs[task].append(p)
    
    print(f"Available tasks: {list(task_pairs.keys())}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Statistics
    stats = {}
    
    for task_name in ["depth", "normal", "seg"]:
        if task_name not in task_pairs:
            print(f"Skipping {task_name} (no pairs)")
            continue
        
        task_p = task_pairs[task_name]
        indices = np.random.choice(len(task_p), min(args.num_samples, len(task_p)), replace=False)
        sampled_pairs = [task_p[i] for i in indices]
        
        print(f"\nProcessing {task_name} ({len(sampled_pairs)} samples)...")
        
        task_images = {}
        success_count = 0
        fail_count = 0
        
        for idx, pair in enumerate(sampled_pairs):
            input_path = pair["input"]
            target_path = pair["target"]
            
            if not os.path.exists(input_path):
                print(f"  [{idx+1}/{len(sampled_pairs)}] Input not found, skipping")
                fail_count += 1
                continue
            
            input_img = load_image(input_path, args.resolution)
            input_tensor = torch.from_numpy(np.array(input_img)).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(device=device, dtype=dtype)
            
            if os.path.exists(target_path):
                target_img = load_image(target_path, args.resolution)
                target_tensor = torch.from_numpy(np.array(target_img)).permute(2, 0, 1).float() / 255.0
                target_tensor = target_tensor.unsqueeze(0).to(device=device, dtype=dtype)
            else:
                target_tensor = None
            
            # Inference with retry
            output_tensor, success = run_inference_with_retry(
                peft_model,
                components.vae,
                components,
                input_tensor,
                f"Generate {task_name} map",
                device,
                dtype,
                args.num_inference_steps,
                args.guidance_scale,
                max_retries=args.max_retries,
            )
            
            if success:
                print(f"  [{idx+1}/{len(sampled_pairs)}] Success: min={output_tensor.min():.4f}, max={output_tensor.max():.4f}, mean={output_tensor.mean():.4f}, std={output_tensor.std():.4f}")
                success_count += 1
            else:
                print(f"  [{idx+1}/{len(sampled_pairs)}] Failed after {args.max_retries} retries")
                fail_count += 1
                # Create a placeholder for failed samples
                output_tensor = torch.zeros(1, 3, args.resolution, args.resolution)
            
            input_pil = input_img
            target_pil = tensor_to_pil(target_tensor) if target_tensor is not None else None
            output_pil = tensor_to_pil(output_tensor[0].cpu())
            
            task_images[idx] = [
                input_pil,
                target_pil if target_pil else Image.new("RGB", (args.resolution, args.resolution), (128, 128, 128)),
                output_pil
            ]
        
        stats[task_name] = {"success": success_count, "fail": fail_count}
        
        # Save results
        task_dir = args.output_dir / task_name
        task_dir.mkdir(exist_ok=True)
        
        for idx, images in task_images.items():
            grid = make_comparison_grid(images, ["Input", "Ground Truth", "Model Output"])
            grid.save(task_dir / f"sample_{idx:03d}.png")
        
        summary = make_summary_grid(task_images, task_name)
        if summary:
            summary.save(args.output_dir / f"{task_name}_summary.png")
            print(f"  Saved summary: {args.output_dir / f'{task_name}_summary.png'}")
    
    # Print statistics
    print("\n" + "="*60)
    print("INFERENCE STATISTICS")
    print("="*60)
    for task, s in stats.items():
        total = s["success"] + s["fail"]
        success_rate = s["success"] / total * 100 if total > 0 else 0
        print(f"{task:10s}: {s['success']:3d}/{total:3d} successful ({success_rate:.1f}%)")
    print("="*60)
    
    print(f"\nRobust qualitative evaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
