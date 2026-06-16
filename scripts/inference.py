#!/usr/bin/env python3
"""Inference script for Open-Vision-Banana.

Supports single-image and batch inference for all vision tasks:
- Depth estimation
- Surface normal estimation
- Semantic segmentation
- Instance segmentation
- Referring expression segmentation

Usage:
    # Single image — depth
    python scripts/inference.py --image input.jpg --task depth --output output/

    # Batch inference
    python scripts/inference.py --image-dir /path/to/images --task semantic

    # With LoRA checkpoint
    python scripts/inference.py --image input.jpg --task normal --lora-path checkpoints/best/

    # Referring expression
    python scripts/inference.py --image input.jpg --task referring --expression "the dog on the left"
"""

import argparse
import logging
from pathlib import Path

import torch

from open_vision_banana import (
    DepthRGBMapper,
    NormalRGBMapper,
    SegmentationDecoder,
    VisionBananaConfig,
)
from open_vision_banana.inference.prompt import PromptBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Vision Banana Inference")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--image-dir", type=str, help="Directory of input images")
    parser.add_argument("--task", type=str, default="depth",
                        choices=["depth", "normal", "semantic", "instance", "referring"],
                        help="Vision task to perform")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                        help="Path to config file")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA weights for inference")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (auto-detected if omitted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of denoising steps (1 for schnell)")
    parser.add_argument("--expression", type=str, default=None,
                        help="Referring expression (required for --task referring)")
    parser.add_argument("--class-map", type=str, default=None,
                        help="Path to JSON class-color map for semantic segmentation")
    return parser.parse_args()


def load_image(path: str, resolution: int = 512) -> torch.Tensor:
    """Load image as tensor [1, 3, H, W] in [-1, 1]."""
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    img = Image.open(path).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.LANCZOS)
    tensor = pil_to_tensor(img).float() / 255.0  # [3, H, W] in [0, 1]
    tensor = tensor * 2.0 - 1.0  # [-1, 1]
    return tensor.unsqueeze(0)  # [1, 3, H, W]


def save_visualization(tensor: torch.Tensor, path: Path, task: str):
    """Save a tensor as an image file with task-appropriate processing."""
    import numpy as np
    from PIL import Image

    # Clamp to [0, 1]
    vis = tensor.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    vis = (vis * 255).astype(np.uint8)
    img = Image.fromarray(vis)
    img.save(path)
    logger.info(f"Saved {task} visualization to {path}")


def run_inference(
    image_tensor: torch.Tensor,
    task: str,
    generator,
    config: VisionBananaConfig,
    expression: str = None,
    class_color_map: dict = None,
    steps: int = 1,
    seed: int = 42,
) -> dict:
    """Run inference for a single image.

    Returns:
        Dictionary with task-specific outputs and the raw RGB visualization.
    """
    # Build prompt
    if task == "depth":
        prompt = PromptBuilder.build_depth_prompt()
    elif task == "normal":
        prompt = PromptBuilder.build_normal_prompt()
    elif task == "semantic":
        cmap = class_color_map or config.segmentation_decoding.CITYSCAPES_COLORS
        prompt = PromptBuilder.build_semantic_prompt(cmap)
    elif task == "instance":
        prompt = PromptBuilder.build_instance_prompt("objects")
    elif task == "referring":
        if not expression:
            raise ValueError("--expression is required for referring task")
        prompt = PromptBuilder.build_referring_prompt(expression)
    else:
        raise ValueError(f"Unknown task: {task}")

    # Generate RGB visualization
    result_rgb = generator.generate(
        prompt=prompt,
        image=image_tensor,
        num_inference_steps=steps,
        seed=seed,
    )

    # Decode RGB visualization back to task-specific output
    output = {"rgb_visualization": result_rgb}

    if task == "depth":
        mapper = DepthRGBMapper(
            lam=config.depth_mapping.lam,
            c=config.depth_mapping.c,
        )
        output["depth"] = mapper.rgb_to_depth(result_rgb)

    elif task == "normal":
        mapper = NormalRGBMapper()
        output["normal"] = mapper.rgb_to_normal(result_rgb)

    elif task == "semantic":
        cmap = class_color_map or config.segmentation_decoding.CITYSCAPES_COLORS
        decoder = SegmentationDecoder()
        output["mask"] = decoder.decode_semantic(result_rgb, cmap)

    elif task == "instance":
        decoder = SegmentationDecoder()
        output["instances"] = decoder.decode_instance(result_rgb)

    elif task == "referring":
        # Simple threshold-based mask extraction
        gray = result_rgb.mean(dim=1, keepdim=True)
        output["mask"] = (gray > 0.5).float()

    return output


def main():
    args = parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        cfg = VisionBananaConfig.from_yaml(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        cfg = VisionBananaConfig()

    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    torch.manual_seed(args.seed)

    # Load model
    from open_vision_banana.inference.generator import BaseGenerator
    generator = BaseGenerator(
        model_name=cfg.model.name,
        lora_path=args.lora_path,
        device=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    # Load class-color map if provided
    class_color_map = None
    if args.class_map:
        import json
        with open(args.class_map) as f:
            raw = json.load(f)
        class_color_map = {k: tuple(v) for k, v in raw.items()}

    # Collect input images
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(sorted(img_dir.glob(ext)))
    else:
        raise ValueError("Must specify --image or --image-dir")

    if not image_paths:
        raise FileNotFoundError("No images found")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    for img_path in image_paths:
        logger.info(f"Processing {img_path.name} ({args.task})")

        image_tensor = load_image(str(img_path), resolution=cfg.data.resolution)
        image_tensor = image_tensor.to(device)

        result = run_inference(
            image_tensor=image_tensor,
            task=args.task,
            generator=generator,
            config=cfg,
            expression=args.expression,
            class_color_map=class_color_map,
            steps=args.steps,
            seed=args.seed,
        )

        # Save raw RGB visualization
        stem = img_path.stem
        save_visualization(
            result["rgb_visualization"],
            output_dir / f"{stem}_{args.task}_vis.png",
            args.task,
        )

        # Save task-specific output
        if args.task == "depth" and "depth" in result:
            import numpy as np
            from PIL import Image as PILImage
            depth = result["depth"].squeeze().cpu().numpy()
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_vis = (depth_norm * 255).astype(np.uint8)
            PILImage.fromarray(depth_vis).save(output_dir / f"{stem}_depth.png")

        elif args.task == "normal" and "normal" in result:
            save_visualization(
                result["normal"] * 0.5 + 0.5,  # [-1,1] → [0,1]
                output_dir / f"{stem}_normal.png",
                "normal",
            )

        elif args.task == "semantic" and "mask" in result:
            import numpy as np
            from PIL import Image as PILImage
            mask = result["mask"].squeeze().cpu().numpy()
            PILImage.fromarray(mask.astype(np.uint8)).save(
                output_dir / f"{stem}_semantic.png"
            )

        elif args.task == "instance" and "instances" in result:
            import numpy as np
            from PIL import Image as PILImage
            inst_vis = np.zeros(result["rgb_visualization"].shape[-2:], dtype=np.uint8)
            for inst in result["instances"][0]:
                inst_vis[inst["mask"].numpy()] = inst.get("label", 1)
            PILImage.fromarray(inst_vis).save(output_dir / f"{stem}_instance.png")

        elif args.task == "referring" and "mask" in result:
            import numpy as np
            from PIL import Image as PILImage
            mask = (result["mask"].squeeze().cpu().numpy() * 255).astype(np.uint8)
            PILImage.fromarray(mask).save(output_dir / f"{stem}_referring.png")

    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
