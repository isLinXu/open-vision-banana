#!/usr/bin/env python3
"""Inference script for Open-Vision-Banana.

Supports single-image and batch inference for all 10 vision tasks:
depth, normal, semantic, instance, referring, edge, flow, keypoint,
object_detection, depth_edge.

Usage:
    python scripts/inference.py --image input.jpg --task depth --output-dir output/
    python scripts/inference.py --image-dir /path/to/images --task edge
    python scripts/inference.py --image input.jpg --task referring --expression "the dog"
    python scripts/inference.py --image input.jpg --task semantic --class-map colors.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALL_TASKS = [
    "depth", "normal", "semantic", "instance", "referring",
    "edge", "flow", "keypoint", "object_detection", "depth_edge",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Vision Banana Inference")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--image-dir", type=str, help="Directory of input images")
    parser.add_argument(
        "--task", type=str, default="depth", choices=ALL_TASKS,
        help="Vision task to perform",
    )
    parser.add_argument("--model-name", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="LoRA weights: local path, hf:repo/id, or registry:alias")
    parser.add_argument("--lora-revision", type=str, default=None)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.85,
                        help="Img2img noise strength (0, 1]")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--expression", type=str, default=None,
                        help="Referring expression (required for --task referring)")
    parser.add_argument("--object-name", type=str, default="objects",
                        help="Object category for instance segmentation")
    parser.add_argument("--class-map", type=str, default=None,
                        help="JSON class-color map for semantic segmentation")
    return parser.parse_args()


def load_image(path: str, resolution: int = 512) -> torch.Tensor:
    """Load image as tensor [1, 3, H, W] in [-1, 1]."""
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    tensor = pil_to_tensor(img).float() / 255.0
    return (tensor * 2.0 - 1.0).unsqueeze(0)


def save_rgb_tensor(tensor: torch.Tensor, path: Path) -> None:
    """Save [B,3,H,W] or [3,H,W] tensor in [0,1] as PNG."""
    import numpy as np
    from PIL import Image

    if tensor.dim() == 4:
        tensor = tensor[0]
    vis = tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    Image.fromarray((vis * 255).astype(np.uint8)).save(path)


def save_task_output(result: dict, task: str, stem: str, output_dir: Path) -> None:
    """Save task-specific artifacts alongside RGB visualization."""
    import numpy as np
    from PIL import Image

    if "rgb_output" in result:
        save_rgb_tensor(result["rgb_output"], output_dir / f"{stem}_{task}_vis.png")

    if task == "depth" and "depth" in result:
        depth = result["depth"].squeeze().cpu().numpy()
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        Image.fromarray((depth_norm * 255).astype(np.uint8)).save(
            output_dir / f"{stem}_depth.png"
        )

    elif task == "normal" and "normal" in result:
        save_rgb_tensor(result["normal"] * 0.5 + 0.5, output_dir / f"{stem}_normal.png")

    elif task == "semantic" and "mask" in result:
        mask = result["mask"].squeeze().cpu().numpy()
        Image.fromarray(mask.astype(np.uint8)).save(output_dir / f"{stem}_semantic.png")

    elif task == "instance" and "instances" in result:
        inst_vis = np.zeros(result["rgb_output"].shape[-2:], dtype=np.uint8)
        for inst in result["instances"][0]:
            inst_vis[inst["mask"].numpy()] = inst.get("label", 1)
        Image.fromarray(inst_vis).save(output_dir / f"{stem}_instance.png")

    elif task == "referring" and "mask" in result:
        mask = (result["mask"].squeeze().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask).save(output_dir / f"{stem}_referring.png")

    elif task in ("edge", "depth_edge") and "edge_mask" in result:
        mask = (result["edge_mask"].squeeze().cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask).save(output_dir / f"{stem}_{task}.png")

    elif task == "flow" and "flow" in result:
        save_rgb_tensor(result.get("visualization", result["rgb_output"]),
                        output_dir / f"{stem}_flow_vis.png")

    elif task == "keypoint" and "keypoints" in result:
        save_rgb_tensor(result.get("visualization", result["rgb_output"]),
                        output_dir / f"{stem}_keypoint_vis.png")

    elif task == "object_detection" and "bboxes" in result:
        save_rgb_tensor(result.get("visualization", result["rgb_output"]),
                        output_dir / f"{stem}_detection_vis.png")


def main():
    args = parse_args()

    if args.task == "referring" and not args.expression:
        raise ValueError("--expression is required for --task referring")

    if torch.cuda.is_available():
        device = args.device or "cuda"
    elif torch.backends.mps.is_available():
        device = args.device or "mps"
    else:
        device = args.device or "cpu"

    torch.manual_seed(args.seed)

    from open_vision_banana.inference.optimization import InferenceOptions
    from open_vision_banana.inference.pipeline import VisionBanana

    dtype_str = "bfloat16" if device == "cuda" else "float32"
    inference_options = InferenceOptions(
        enable_sequential_cpu_offload=args.cpu_offload,
        enable_vae_slicing=args.vae_slicing,
        enable_vae_tiling=args.vae_tiling,
        enable_attention_slicing=args.attention_slicing,
        enable_torch_compile=args.torch_compile,
        dtype=dtype_str,
    )

    model = VisionBanana.from_pretrained(
        model_name=args.model_name,
        lora_path=args.lora_path,
        lora_revision=args.lora_revision,
        device=device,
        inference_options=inference_options,
    )

    class_color_map = None
    if args.class_map:
        with open(args.class_map) as f:
            raw = json.load(f)
        class_color_map = {k: tuple(v) for k, v in raw.items()}
    elif args.task == "semantic":
        from open_vision_banana.data.datasets import CityscapesSegDataset
        class_color_map = CityscapesSegDataset.CITYSCAPES_COLORS

    image_paths: list[Path] = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(sorted(img_dir.glob(ext)))
    else:
        raise ValueError("Must specify --image or --image-dir")

    if not image_paths:
        raise FileNotFoundError("No images found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predict_kwargs = dict(
        task=args.task,
        num_inference_steps=args.steps,
        strength=args.strength,
        seed=args.seed,
    )
    if class_color_map is not None:
        predict_kwargs["class_color_map"] = class_color_map
    if args.expression:
        predict_kwargs["expression"] = args.expression
    if args.object_name:
        predict_kwargs["object_name"] = args.object_name

    for img_path in image_paths:
        logger.info("Processing %s (%s)", img_path.name, args.task)
        image_tensor = load_image(str(img_path), resolution=args.resolution).to(device)
        result = model.predict(image=image_tensor, **predict_kwargs)
        save_task_output(result, args.task, img_path.stem, output_dir)

    logger.info("All results saved to %s", output_dir)


if __name__ == "__main__":
    main()
