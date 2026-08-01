#!/usr/bin/env python3
"""Inference + side-by-side visualization for a source-conditioned LoRA checkpoint.

Loads checkpoint-50 (or any source-conditioned SSD-1B LoRA checkpoint) and runs
depth / normal / edge / semantic / generation tasks on validation samples,
saving a side-by-side comparison: [input | GT target | model RGB output | decoded].

Usage:
    python scripts/infer_checkpoint_viz.py \
        --checkpoint output/vision_banana_lora/checkpoint-50 \
        --data-dir data/vb_snvc_train_v3 \
        --output-dir output/viz_checkpoint_50 \
        --num-samples 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.inference.optimization import InferenceOptions
from open_vision_banana.inference.pipeline import VisionBanana

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_image(path: str | Path, resolution: int = 1024) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().clamp(0, 1)
    if t.dim() == 2:
        t = t.unsqueeze(0).repeat(3, 1, 1)
    elif t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def normalize_to_display(t: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to [0, 1] for display."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().float()
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    return t


def make_side_by_side(
    images: list[Image.Image], labels: list[str], label_h: int = 30
) -> Image.Image:
    w, h = images[0].size
    total_w = w * len(images)
    total_h = h + label_h
    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img, (i * w, label_h))
        draw.text((i * w + 5, 5), label, fill=(255, 255, 255))
    return canvas


def run_task(
    model: VisionBanana,
    task: str,
    image_tensor: torch.Tensor,
    device: str,
    seed: int = 42,
    steps: int = 4,
) -> dict:
    kwargs = dict(
        task=task,
        num_inference_steps=steps,
        strength=0.85,
        seed=seed,
    )
    if task == "semantic":
        from open_vision_banana.inference.decoder import ADE20K_151_PALETTE

        palette_255 = {k: v for k, v in ADE20K_151_PALETTE.items()}
        kwargs["class_color_map"] = palette_255
    return model.predict(image=image_tensor.to(device), **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="output/viz_checkpoint")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model with checkpoint: %s", args.checkpoint)
    dtype = torch.float32 if device == "mps" else torch.bfloat16
    inference_options = InferenceOptions(dtype="float32" if device == "mps" else "bfloat16")
    model = VisionBanana.from_pretrained(
        model_name="segmind/SSD-1B",
        lora_path=args.checkpoint,
        device=device,
        dtype=dtype,
        inference_options=inference_options,
    )
    logger.info("Model loaded.")

    data_dir = Path(args.data_dir)
    tasks_config = {
        "depth": {"dir": "depth", "has_target": True},
        "normal": {"dir": "normal", "has_target": True},
        "edge": {"dir": "structure", "has_target": True},
        "semantic": {"dir": "segmentation", "has_target": True},
    }

    for task_name, cfg in tasks_config.items():
        task_dir = data_dir / cfg["dir"]
        if not task_dir.exists():
            logger.warning("Task dir not found: %s, skipping %s", task_dir, task_name)
            continue

        img_dir = task_dir / "images"
        tgt_dir = task_dir / "targets"
        images_sorted = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

        if not images_sorted:
            logger.warning("No images found for task %s", task_name)
            continue

        samples = images_sorted[: args.num_samples]
        task_out = output_dir / task_name
        task_out.mkdir(parents=True, exist_ok=True)

        for img_path in samples:
            stem = img_path.stem
            logger.info("Processing [%s] %s", task_name, stem)

            input_tensor = load_image(str(img_path), args.resolution)
            try:
                result = run_task(
                    model, task_name, input_tensor, device, seed=args.seed, steps=args.steps
                )
            except Exception as e:
                logger.error("Inference failed for %s/%s: %s", task_name, stem, e)
                continue

            panels = []
            labels = []

            input_pil = tensor_to_pil((input_tensor[0] + 1) / 2)
            panels.append(input_pil)
            labels.append("Input")

            if cfg["has_target"]:
                tgt_path = tgt_dir / f"{stem}.png"
                if not tgt_path.exists():
                    tgt_path = tgt_dir / f"{stem}.jpg"
                if tgt_path.exists():
                    tgt_img = (
                        Image.open(tgt_path)
                        .convert("RGB")
                        .resize((args.resolution, args.resolution), Image.LANCZOS)
                    )
                    panels.append(tgt_img)
                    labels.append("GT Target")

            rgb_out = result.get("rgb_output")
            if rgb_out is not None:
                rgb_pil = tensor_to_pil(rgb_out)
                panels.append(rgb_pil)
                labels.append("Model Output (RGB)")

            if task_name == "depth" and "depth" in result:
                depth_vis = normalize_to_display(result["depth"])
                panels.append(tensor_to_pil(depth_vis))
                labels.append("Decoded Depth")

            comparison = make_side_by_side(panels, labels)
            save_path = task_out / f"{stem}_comparison.png"
            comparison.save(save_path)
            logger.info("Saved: %s", save_path)

            if rgb_out is not None:
                tensor_to_pil(rgb_out).save(task_out / f"{stem}_rgb_output.png")

    logger.info("All visualizations saved to %s", output_dir)


if __name__ == "__main__":
    main()
