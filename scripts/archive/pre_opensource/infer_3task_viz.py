#!/usr/bin/env python3
"""Generate side-by-side comparison visualizations for 3-task training results.

Creates comparison grids: [Input | GT Target | Model Output] for depth, normal, semantic tasks.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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


def load_image(path: str, resolution: int = 512) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    return img


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image. Handles both [0, 1] and [-1, 1] ranges."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().float()
    # If range is [-1, 1], convert to [0, 1]
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    if t.dim() == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)
    elif t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to lora.pt")
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading Sana pipeline...")
    model_id = os.environ.get("OPEN_VISION_BANANA_MODEL_PATH", "Efficient-Large-Model/Sana_600M_512px_diffusers")
    model_cfg = get_model_config("sana-1600m")
    device = torch.device(args.device)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    components = load_pipeline(model_cfg, device=device, dtype=dtype)
    
    # Keep text_encoder in its native dtype (bfloat16) to avoid memory pressure
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
    lora_state = ckpt.get("lora_state_dict", ckpt)
    peft_model.load_state_dict(lora_state, strict=False)
    peft_model.eval()
    step = ckpt.get("step", 0)
    val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", "N/A"))
    print(f"  Restored step={step}, val_loss={val_loss}")
    
    # Get model dtype from parameters
    model_dtype = next(peft_model.parameters()).dtype
    print(f"  Model dtype: {model_dtype}")
    
    # Load pairs
    print(f"Loading pairs from {args.pair_manifest}")
    pairs = {"depth": [], "normal": [], "seg": []}
    with args.pair_manifest.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") != "val":
                continue
            task = row["task"]
            if task in pairs and len(pairs[task]) < args.num_samples:
                pairs[task].append(row)
    
    # Task prompts (seg uses semantic prompt)
    task_prompts = {
        "depth": "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map.",
        "normal": "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
        "seg": "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
    }
    
    # Run inference for each task
    for task, task_pairs in pairs.items():
        if not task_pairs:
            continue
        
        print(f"\nProcessing {task} ({len(task_pairs)} samples)...")
        
        for i, pair in enumerate(task_pairs):
            input_path = pair["input"]
            target_path = pair["target"]
            
            # Load input
            input_img = load_image(input_path, args.resolution)
            target_img = load_image(target_path, args.resolution)
            
            # Prepare tensors
            input_tensor = torch.from_numpy(np.array(input_img)).float() / 255.0
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
            input_tensor = input_tensor.to(device=device, dtype=model_dtype)
            
            # Encode text
            prompt = task_prompts[task]
            
            # Run flow matching inference
            output_tensor = flow_matching_inference(
                peft_model,
                components.vae,
                components,
                input_tensor,
                [prompt],
                device,
                model_dtype,
                num_steps=args.num_inference_steps,
                use_img2img=True,
                max_length=300,
                guidance_scale=1.0,
            )
            
            # Convert to PIL
            print(f"  DEBUG: output_tensor shape={output_tensor.shape}, min={output_tensor.min():.4f}, max={output_tensor.max():.4f}, mean={output_tensor.mean():.4f}")
            output_pil = tensor_to_pil(output_tensor[0].cpu())
            
            # Create comparison
            comparison = make_comparison_grid(
                [input_img, target_img, output_pil],
                ["Input", "Ground Truth", "Model Output"]
            )
            
            # Save
            out_path = args.output_dir / f"{task}_sample_{i:03d}.png"
            comparison.save(out_path)
            print(f"  Saved: {out_path}")
    
    print(f"\nDone! Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
