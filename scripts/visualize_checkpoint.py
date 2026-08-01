#!/usr/bin/env python3
"""Visualize a LoRA checkpoint on fixed depth samples.

Loads a LoRA checkpoint (PEFT adapter_dir or .pt file) and runs img2img
inference on a few fixed samples, saving side-by-side comparisons:
  input | output | decoded | target

Usage:
  python scripts/visualize_checkpoint.py --checkpoint output/.../checkpoint-100 --step 100
  python scripts/visualize_checkpoint.py --checkpoint output/.../lora_final.pt --step 300
  python scripts/visualize_checkpoint.py --checkpoint output/.../checkpoint-100 --step 100 --baseline-only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.inference.decoder import VisualizationDecoder  # noqa: E402
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.inference.prompt import PromptBuilder  # noqa: E402
from open_vision_banana.training.tuner import VisionTaskDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("viz")

MODEL_ID = "segmind/SSD-1B"
RESOLUTION = 512
VIZ_SAMPLE_INDICES = [0, 10, 20, 30, 40]
TASK_PROMPTS = {
    "depth": PromptBuilder.build_depth_prompt(),
    "normal": PromptBuilder.build_normal_prompt(),
    "edge": PromptBuilder.build_edge_prompt(),
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tensor_to_pil(t):
    """Convert [C,H,W] or [B,C,H,W] tensor in [0,1] or [-1,1] to PIL."""
    t = t.float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_viz_samples(data_dir, task="depth", n=5):
    """Load fixed visualization samples."""
    task_dir = data_dir / task
    if not (task_dir / "images").exists():
        logger.error(f"No data at {task_dir}")
        return []
    ds = VisionTaskDataset(data_path=str(task_dir), resolution=RESOLUTION, task=task)
    samples = []
    for idx in VIZ_SAMPLE_INDICES[:n]:
        if idx < len(ds):
            samples.append(ds[idx])
    return samples


def run_visualization(
    generator,
    decoder,
    viz_samples,
    task,
    output_dir,
    step,
    inference_steps,
):
    """Run inference and save comparison images."""
    viz_dir = output_dir / f"viz_step_{step}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    prompt = TASK_PROMPTS.get(task, "Generate a visualization of this image.")

    logger.info(f"Visualizing step {step}: {len(viz_samples)} samples")

    results_meta = []

    for i, sample in enumerate(viz_samples):
        try:
            input_img = sample.get("image")
            target = sample.get("target_rgb")

            if input_img is None:
                continue

            input_pil = tensor_to_pil(input_img) if isinstance(input_img, torch.Tensor) else input_img

            with torch.no_grad():
                output_tensor = generator.generate(
                    prompt=prompt,
                    image=input_pil,
                    num_inference_steps=inference_steps,
                    guidance_scale=0.0,
                    strength=0.85,
                    height=RESOLUTION,
                    width=RESOLUTION,
                    seed=42,
                )

            output_pil = tensor_to_pil(output_tensor)

            # Decode
            decoded_viz = None
            if task == "depth":
                result = decoder.decode_depth(output_tensor)
                if "visualization" in result:
                    decoded_viz = tensor_to_pil(result["visualization"])
            elif task == "normal":
                result = decoder.decode_normal(output_tensor)
                if "visualization" in result:
                    decoded_viz = tensor_to_pil(result["visualization"])
            elif task == "edge":
                result = decoder.decode_edge(output_tensor)
                if "visualization" in result:
                    decoded_viz = tensor_to_pil(result["visualization"])

            # Build comparison
            panels = [input_pil.resize((256, 256)), output_pil.resize((256, 256))]
            labels = ["input", "output"]

            if decoded_viz is not None:
                panels.append(decoded_viz.resize((256, 256)))
                labels.append("decoded")

            if target is not None:
                target_pil = tensor_to_pil(target) if isinstance(target, torch.Tensor) else target
                panels.append(target_pil.resize((256, 256)))
                labels.append("target")

            total_w = sum(p.width for p in panels)
            max_h = max(p.height for p in panels)
            combined = Image.new("RGB", (total_w + (len(panels) - 1) * 4, max_h), (255, 255, 255))
            x = 0
            for p in panels:
                combined.paste(p, (x, 0))
                x += p.width + 4

            fname = viz_dir / f"sample_{i:02d}.png"
            combined.save(str(fname))
            logger.info(f"  Sample {i}: saved {fname}")

            results_meta.append({"sample": i, "labels": labels, "file": str(fname)})

        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")

    # Save metadata
    import json
    with open(viz_dir / "meta.json", "w") as f:
        json.dump({"step": step, "task": task, "samples": results_meta}, f, indent=2)

    return viz_dir


def main():
    global RESOLUTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint (adapter dir or .pt file)")
    parser.add_argument("--step", type=int, required=True, help="Step number for naming")
    parser.add_argument("--task", default="depth", choices=["depth", "normal", "edge"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline-only", action="store_true",
                        help="Skip LoRA loading, visualize baseline model only")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--inference-steps", type=int, default=4)
    args = parser.parse_args()
    RESOLUTION = args.resolution

    device = get_device()
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output" / "train_viz_depth"
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_root = output_dir / "visualizations"
    viz_root.mkdir(exist_ok=True)

    ckpt_path = None if args.baseline_only else Path(args.checkpoint)
    lora_path = str(ckpt_path) if ckpt_path is not None and ckpt_path.is_dir() else None

    # Let BaseGenerator load directory checkpoints so it restores the LoRA,
    # expanded source-conditioning conv, and epsilon scheduler as one contract.
    logger.info("Loading SSD-1B...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=MODEL_ID,
        lora_path=lora_path,
        device=str(device),
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Load LoRA
    if not args.baseline_only:
        logger.info(f"Loading LoRA from {ckpt_path}")
        if ckpt_path is not None and not ckpt_path.is_dir():
            generator.pipe.load_lora_weights(str(ckpt_path))
        logger.info("LoRA loaded")

    # Load samples
    data_dir = PROJECT_ROOT / "data" / "vb_snvc_train_v2"
    viz_samples = load_viz_samples(data_dir, args.task, args.num_samples)
    logger.info(f"Loaded {len(viz_samples)} samples")

    # Visualize
    decoder = VisualizationDecoder()
    run_visualization(
        generator,
        decoder,
        viz_samples,
        args.task,
        viz_root,
        args.step,
        args.inference_steps,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
