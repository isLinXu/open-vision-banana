#!/usr/bin/env python3
"""Vision Banana depth LoRA inference — image→depth RGB on MPS.

Loads trained rsLoRA + channel adapter weights, takes input images,
and generates depth RGB visualisations using the tiny-sd pipeline.

Pipeline:
  1. Input image → VAE encode → input_latents
  2. Start from random noise latents
  3. For each step: concat(input_latents, noisy_latents) → adapter → UNet → denoise
  4. Final latents → VAE decode → depth RGB image

Output:
  - input_*.png: original input images
  - depth_*.png: generated depth RGB visualisations
  - gt_*.png: ground-truth depth RGB (from dataset)
  - loss_curve.png: training loss curve
  - results_summary.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# HuggingFace mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_infer")

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "segmind/tiny-sd"
LORA_PATH = Path(
    "/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_depth_lora/vb_lora_final.pt"
)
TRAINING_LOG = LORA_PATH.parent / "training_log.json"
OUTPUT_DIR = LORA_PATH.parent / "validation"
DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vision_banana_depth")
RESOLUTION = 256
SEED = 42
NUM_INFERENCE_STEPS = 30
NUM_SAMPLES = 6


class ChannelAdapter(nn.Module):
    """Learnable 8→4 channel adapter (matches training)."""

    def __init__(self, in_channels: int = 8, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def load_lora_weights(unet, adapter, lora_state: dict):
    """Load LoRA + adapter weights from checkpoint."""
    # Load LoRA weights into UNet
    missing = []
    for name, param in unet.named_parameters():
        if param.requires_grad:
            key = f"unet.{name}"
            if key in lora_state:
                param.data.copy_(lora_state[key])
            else:
                stripped = name.replace("base_model.model.", "")
                key2 = f"unet.{stripped}"
                if key2 in lora_state:
                    param.data.copy_(lora_state[key2])
                else:
                    missing.append(name)

    if missing:
        logger.warning(f"Missing {len(missing)} LoRA keys (first 5: {missing[:5]})")

    # Load adapter weights
    for name, param in adapter.named_parameters():
        key = f"adapter.{name}"
        if key in lora_state:
            param.data.copy_(lora_state[key])
        else:
            logger.warning(f"Missing adapter key: {key}")

    logger.info(f"Loaded {len(lora_state)} weights ({len(missing)} missing LoRA keys)")


def generate_depth(
    unet,
    vae,
    scheduler,
    adapter,
    input_image_tensor: torch.Tensor,
    device: torch.device,
    num_steps: int = 30,
    seed: int = 42,
) -> Image.Image:
    """Generate depth RGB from input image using trained model.

    Args:
        input_image_tensor: [1, 3, 256, 256] in [-1, 1]

    Returns:
        PIL Image (256×256) depth RGB visualisation
    """
    torch.manual_seed(seed)
    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)

    # Encode input image to latents
    with torch.no_grad():
        input_latents = vae.encode(input_image_tensor).latent_dist.sample()
        input_latents = input_latents * vae_scaling  # [1, 4, 32, 32]

    # Start from random noise
    latents = torch.randn(1, 4, 32, 32, device=device)

    # Set timesteps
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps

    # Text encoding (placeholder — depth prompt doesn't matter much for image-conditioned gen)
    # Use empty prompt encoding
    encoder_hidden_states = torch.zeros(1, 77, 768, device=device)

    with torch.no_grad():
        for _i, t in enumerate(timesteps):
            # Concatenate input latents with current noisy latents
            concat = torch.cat([input_latents, latents], dim=1)  # [1, 8, 32, 32]
            adapted = adapter(concat)  # [1, 4, 32, 32]

            # Predict noise
            noise_pred = unet(adapted, t.unsqueeze(0), encoder_hidden_states).sample

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    # Decode latents to image
    latents = latents / vae_scaling
    with torch.no_grad():
        image = vae.decode(latents).sample  # [1, 3, 256, 256] in [-1, 1]

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def plot_loss_curve(log_path: Path, output_path: Path):
    """Generate a loss curve plot from training log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(log_path) as f:
        log = json.load(f)

    metrics = log["metrics"]
    if not metrics:
        logger.warning("No metrics in training log, skipping loss curve")
        return

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(steps, losses, "b-o", linewidth=2, markersize=5, label="Training Loss")
    ax.fill_between(steps, losses, alpha=0.15, color="blue")

    ax.set_xlabel("Optimizer Step", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title(
        "Vision Banana rsLoRA Training — Depth Task (MPS)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if len(losses) > 1:
        ax.annotate(f"{losses[0]:.4f}", (steps[0], losses[0]),
                    textcoords="offset points", xytext=(10, 10), fontsize=9, color="blue")
        ax.annotate(f"{losses[-1]:.4f}", (steps[-1], losses[-1]),
                    textcoords="offset points", xytext=(-30, -15), fontsize=9, color="blue")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss curve saved to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # ── Load base pipeline ───────────────────────────────────────────
    from diffusers import StableDiffusionPipeline

    logger.info("Loading base model (segmind/tiny-sd)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler

    # ── Apply LoRA architecture ──────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
        use_rslora=True,
    )
    unet = get_peft_model(unet, lora_config)

    # ── Create channel adapter ───────────────────────────────────────
    adapter = ChannelAdapter(8, 4).to(device)

    # ── Load trained weights ─────────────────────────────────────────
    logger.info(f"Loading trained weights from {LORA_PATH}...")
    lora_state = torch.load(LORA_PATH, map_location="cpu")
    load_lora_weights(unet, adapter, lora_state)
    unet.eval()
    adapter.eval()

    # ── Select sample images ─────────────────────────────────────────
    image_paths = sorted((DATA_DIR / "images").glob("*.jpg"))
    target_paths = sorted((DATA_DIR / "targets").glob("*.jpg"))

    # Pick evenly spaced samples
    indices = np.linspace(0, len(image_paths) - 1, NUM_SAMPLES, dtype=int)
    logger.info(f"Selected {NUM_SAMPLES} samples for validation")

    # ── Generate depth visualisations ────────────────────────────────
    results = []
    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        tgt_path = target_paths[idx]
        stem = img_path.stem

        # Load input image
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(
            np.array(img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS), dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        # Save input image
        img.save(OUTPUT_DIR / f"input_{i:02d}.png")

        # Save ground truth
        gt = Image.open(tgt_path).convert("RGB")
        gt.save(OUTPUT_DIR / f"gt_{i:02d}.png")

        # Generate depth
        t0 = time.time()
        depth_img = generate_depth(
            unet, vae, scheduler, adapter,
            img_tensor, device,
            num_steps=NUM_INFERENCE_STEPS,
            seed=SEED + i,
        )
        elapsed = time.time() - t0

        depth_img.save(OUTPUT_DIR / f"depth_{i:02d}.png")
        logger.info(
            f"[{i+1}/{NUM_SAMPLES}] {stem} | "
            f"generated depth RGB in {elapsed:.1f}s"
        )
        results.append({
            "index": i,
            "stem": stem,
            "input": f"input_{i:02d}.png",
            "ground_truth": f"gt_{i:02d}.png",
            "predicted": f"depth_{i:02d}.png",
            "time_s": round(elapsed, 1),
        })

    # ── Generate loss curve ──────────────────────────────────────────
    loss_path = OUTPUT_DIR / "loss_curve.png"
    plot_loss_curve(TRAINING_LOG, loss_path)

    # ── Save results summary ─────────────────────────────────────────
    summary = {
        "model": MODEL_ID,
        "task": "depth",
        "lora_path": str(LORA_PATH),
        "config": {
            "lora_rank": 64,
            "lora_alpha": 128,
            "use_rslora": True,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
            "num_samples": NUM_SAMPLES,
            "resolution": RESOLUTION,
        },
        "generations": results,
    }

    # Add training stats if available
    if TRAINING_LOG.exists():
        with open(TRAINING_LOG) as f:
            train_log = json.load(f)
        if train_log.get("metrics"):
            summary["training"] = {
                "initial_loss": train_log["metrics"][0]["loss"],
                "final_loss": train_log["final_loss"],
                "total_steps": train_log["total_steps"],
                "epochs": train_log["config"]["epochs"],
                "loss_reduction_pct": round(
                    (1 - train_log["final_loss"] / train_log["metrics"][0]["loss"]) * 100, 1
                ),
            }

    summary_path = OUTPUT_DIR / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Validation complete!")
    logger.info(f"  Input images:  {OUTPUT_DIR}/input_*.png")
    logger.info(f"  Ground truth:  {OUTPUT_DIR}/gt_*.png")
    logger.info(f"  Predictions:   {OUTPUT_DIR}/depth_*.png")
    logger.info(f"  Loss curve:    {loss_path}")
    logger.info(f"  Summary:       {summary_path}")
    if "training" in summary:
        t = summary["training"]
        logger.info(f"  Loss: {t['initial_loss']:.4f} → {t['final_loss']:.4f} ({t['loss_reduction_pct']}% reduction)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
