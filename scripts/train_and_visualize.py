#!/usr/bin/env python3
"""Train SSD-1B with periodic checkpoint visualization.

Runs full-data training, and after each checkpoint save, automatically
loads the LoRA weights and runs inference on a few fixed validation samples
to visualize depth / normal / segmentation / edge results side-by-side
with the ground truth.

Usage:
  python scripts/train_and_visualize.py --max-steps 500 --save-every 100
  python scripts/train_and_visualize.py --task depth --max-steps 300
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
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset  # noqa: E402
from open_vision_banana.training.utils import enable_gradient_checkpointing  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_viz")

MODEL_ID = "segmind/SSD-1B"
RESOLUTION = 256

# Fixed visualization samples (deterministic selection from training data)
VIZ_SAMPLE_INDICES = [0, 10, 20, 30, 40]

# Task prompts
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
        t = t[0]  # take first sample from batch
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def load_viz_samples(data_dir, task, n=5):
    """Load fixed visualization samples from the dataset."""
    task_dir = data_dir / task
    if not (task_dir / "images").exists():
        return []

    ds = VisionTaskDataset(data_path=str(task_dir), resolution=RESOLUTION, task=task)
    samples = []
    for idx in VIZ_SAMPLE_INDICES[:n]:
        if idx < len(ds):
            samples.append(ds[idx])
    return samples


def visualize_checkpoint(
    generator,
    decoder,
    viz_samples,
    task,
    output_dir,
    step,
):
    """Run inference on viz samples and save comparison images."""
    viz_dir = output_dir / f"viz_step_{step}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    prompt = TASK_PROMPTS.get(task, "Generate a visualization of this image.")
    logger.info(f"  Visualizing step {step}: {len(viz_samples)} samples, task={task}")

    for i, sample in enumerate(viz_samples):
        try:
            input_img = sample.get("image")
            target = sample.get("target_rgb")

            if input_img is None:
                logger.warning(f"  Sample {i}: no input image, skipping")
                continue

            # Prepare input as PIL for generator
            if isinstance(input_img, torch.Tensor):
                input_pil = tensor_to_pil(input_img)
            else:
                input_pil = input_img

            # Run img2img inference
            with torch.no_grad():
                output_tensor = generator.generate(
                    prompt=prompt,
                    image=input_pil,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    strength=0.85,
                    height=RESOLUTION,
                    width=RESOLUTION,
                    seed=42,
                )

            output_pil = tensor_to_pil(output_tensor)

            # Decode output based on task (decoder expects [B,C,H,W])
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

            # Build comparison image: input | output | decoded | target
            panels = [input_pil.resize((256, 256)), output_pil.resize((256, 256))]
            if decoded_viz is not None:
                panels.append(decoded_viz.resize((256, 256)))
            if target is not None:
                target_pil = (
                    tensor_to_pil(target)
                    if isinstance(target, torch.Tensor)
                    else target
                )
                panels.append(target_pil.resize((256, 256)))

            # Pad to same size and concatenate horizontally
            total_w = sum(p.width for p in panels)
            max_h = max(p.height for p in panels)
            combined = Image.new("RGB", (total_w, max_h), (255, 255, 255))
            x = 0
            for p in panels:
                combined.paste(p, (x, 0))
                x += p.width

            combined.save(viz_dir / f"sample_{i:02d}.png")

        except Exception as e:
            logger.warning(f"  Sample {i} visualization failed: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"  Visualizations saved to {viz_dir}")
    return viz_dir


def main():
    global RESOLUTION
    parser = argparse.ArgumentParser(description="Train with periodic visualization")
    parser.add_argument("--model", default="ssd-1b")
    parser.add_argument("--task", default="depth",
                        choices=["depth", "normal", "edge", "all"])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-per-task", type=int, default=300)
    parser.add_argument("--max-generation", type=int, default=3000)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    RESOLUTION = args.resolution

    # Setup
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    logger.info(f"Device: {device}")

    # Output dir
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else PROJECT_ROOT / "output" / f"train_viz_{args.task}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_root = output_dir / "visualizations"
    viz_root.mkdir(exist_ok=True)

    # Load model
    logger.info(f"Loading {args.model}...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=MODEL_ID,
        device=str(device),
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Load viz samples BEFORE training (from v2 data)
    data_dir = PROJECT_ROOT / "data" / "vb_snvc_train_v2"
    primary_task = args.task if args.task != "all" else "depth"
    viz_samples = load_viz_samples(data_dir, primary_task, n=5)
    logger.info(f"Loaded {len(viz_samples)} visualization samples for task={primary_task}")

    # Decoder for post-processing
    decoder = VisualizationDecoder()

    # ── Baseline visualization (no LoRA) ──
    logger.info("=" * 60)
    logger.info("Baseline visualization (step 0, no LoRA)")
    logger.info("=" * 60)
    visualize_checkpoint(generator, decoder, viz_samples, primary_task, viz_root, 0)

    # ── Setup LoRA training ──
    logger.info(f"Setting up LoRA: rank={args.rank}, alpha={args.alpha}, lr={args.lr}")
    tuner = InstructionTuner(
        model=generator,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        use_rslora=True,
        lr=args.lr,
        use_ema=not False,
        ema_decay=0.9999,
        use_min_snr=False,
        use_logit_normal_timesteps=False,
        use_img2img_flow=False,
        cfg_dropout_rate=0.1,
        warmup_ratio=0.05,
        use_source_conditioning=True,
        prediction_type="epsilon",
        x0_loss_weight=0.0,
    )
    generator.source_conditioning_config = tuner._source_conditioning_config()

    # Gradient checkpointing
    unet = getattr(generator.pipe, "unet", None)
    if unet is not None and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on UNet")
    else:
        enable_gradient_checkpointing(tuner.peft_model)

    # ── Load training data ──
    res = args.resolution
    data_dirs = [data_dir]

    from torch.utils.data import ConcatDataset

    datasets = []

    def _load_task_dirs(task_name, subdir, vision_task, max_per_task=None):
        merged = []
        for d in data_dirs:
            task_dir = d / subdir
            if (task_dir / "images").exists():
                ds = VisionTaskDataset(data_path=str(task_dir), resolution=res, task=vision_task)
                if ds.samples:
                    if max_per_task is not None and len(ds.samples) > max_per_task:
                        ds.samples = ds.samples[:max_per_task]
                    merged.append(ds)
                    logger.info(f"  {task_name}: {len(ds)} samples")
        return merged

    if args.task in ("all", "depth"):
        datasets.extend(_load_task_dirs("depth", "depth", "depth", args.max_per_task))
    if args.task in ("all", "normal"):
        datasets.extend(_load_task_dirs("normal", "normal", "normal", args.max_per_task))
    if args.task in ("all", "edge"):
        datasets.extend(_load_task_dirs("edge", "edge", "edge", args.max_per_task))
        for d in data_dirs:
            struct_dir = d / "structure"
            if (struct_dir / "images").exists():
                ds = VisionTaskDataset(data_path=str(struct_dir), resolution=res, task="edge")
                if ds.samples:
                    if len(ds.samples) > args.max_per_task:
                        ds.samples = ds.samples[:args.max_per_task]
                    datasets.append(ds)
                    logger.info(f"  structure(edge): {len(ds)} samples")

    if not datasets:
        logger.error("No training data found!")
        sys.exit(1)

    # Balance
    from torch.utils.data import WeightedRandomSampler
    if len(datasets) > 1:
        dataset_sizes = [len(d) for d in datasets]
        max_size = max(dataset_sizes)
        weights = []
        for _ds, sz in zip(datasets, dataset_sizes, strict=True):
            w = max_size / sz
            weights.extend([w] * sz)
        combined = ConcatDataset(datasets)
        sampler_generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(combined),
            replacement=True,
            generator=sampler_generator,
        )
        logger.info("Data balanced: " + ", ".join(
            f"ds{i}={max_size / sz:.1f}x" for i, sz in enumerate(dataset_sizes) if sz > 0
        ))
    else:
        combined = datasets[0]
        sampler = None

    total_samples = len(combined)
    steps_per_epoch = max(1, (total_samples + args.batch_size * args.grad_accum - 1) // (args.batch_size * args.grad_accum))
    total_steps = min(args.epochs * steps_per_epoch if hasattr(args, 'epochs') else 9999, args.max_steps)
    warmup_steps = min(200, max(1, total_steps // 10))

    # Limit dataset for max_steps
    max_samples = args.max_steps * args.grad_accum
    if total_samples > max_samples:
        from torch.utils.data import Subset
        subset_generator = torch.Generator().manual_seed(args.seed + 1)
        indices = torch.randperm(total_samples, generator=subset_generator)[:max_samples].tolist()
        combined = Subset(combined, indices)
        total_samples = len(combined)
        sampler = None
        logger.info(f"Limited dataset to {total_samples} samples for {args.max_steps} steps")

    logger.info(f"Starting training: {total_samples} samples, total_steps={total_steps}")

    # ── Train with periodic visualization ──
    # We train in chunks of save_every steps, visualizing after each chunk.
    steps_done = 0
    epochs = 3

    while steps_done < args.max_steps:
        chunk_steps = min(args.save_every, args.max_steps - steps_done)

        result = tuner.train(
            train_dataset=combined,
            val_dataset=None,
            epochs=epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=1.0,
            scheduler_type="cosine",
            warmup_steps=warmup_steps,
            output_dir=str(output_dir),
            save_every=chunk_steps,
            eval_every=10000,
            use_amp=True,
            num_workers=0,
            log_every=args.log_every,
            start_step=steps_done,
            start_epoch=0,
            train_sampler=sampler,
            resume_checkpoint=None,
            restore_rng_on_resume=False,
            max_steps=steps_done + chunk_steps,
        )

        steps_done = result.get("global_step", steps_done + chunk_steps)
        logger.info(f"Training chunk done: step={steps_done}")

        # ── Visualize the live checkpoint ──
        ckpt_dir = output_dir / f"checkpoint-{steps_done}"
        logger.info(f"Visualizing live LoRA state saved at {ckpt_dir}")

        logger.info("-" * 60)
        logger.info(f"Visualizing checkpoint at step {steps_done}")
        logger.info("-" * 60)
        tuner.peft_model.eval()
        visualize_checkpoint(generator, decoder, viz_samples, primary_task, viz_root, steps_done)
        tuner.peft_model.train()

    logger.info("=" * 60)
    logger.info(f"Training + visualization complete! Total steps: {steps_done}")
    logger.info(f"Visualizations: {viz_root}")
    logger.info(f"Checkpoints: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
