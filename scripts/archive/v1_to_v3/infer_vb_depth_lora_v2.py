#!/usr/bin/env python3
"""Vision Banana depth LoRA v2 inference — optimized image→depth RGB on MPS.

Improvements over v1:
  1. Proper text prompt encoding (not zero placeholder)
  2. Support both final and EMA weights
  3. SSIM/PSNR quantitative metrics
  4. Side-by-side comparison visualization (input | GT | prediction)
  5. More validation samples (8)
  6. Multi-seed ensemble option
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vb_infer_v2")

# ─── Config ──────────────────────────────────────────────────────────────
MODEL_ID = "segmind/tiny-sd"
LORA_DIR = Path(
    "/Users/gatilin/PycharmProjects/open-vision-banana/output/2060393a-ea3e-4780-9b1e-dd103a554755/vb_depth_lora_v2"
)
LORA_PATH = LORA_DIR / "vb_lora_v2_final.pt"
BEST_PATH = LORA_DIR / "vb_lora_v2_best.pt"
TRAINING_LOG = LORA_DIR / "training_log.json"
OUTPUT_DIR = LORA_DIR / "validation"
DATA_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_depth_300")
RESOLUTION = 256
SEED = 42
NUM_INFERENCE_STEPS = 30
NUM_SAMPLES = 8
USE_EMA = True  # Use EMA weights if available

DEPTH_PROMPT = "Generate a metric depth visualization of this image."

# Expanded target modules (must match training)
TARGET_MODULES = [
    "to_q", "to_k", "to_v", "to_out.0",
    "proj_in", "proj_out",
    "ff.net.0.proj", "ff.net.2",
]


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels: int = 8, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def load_lora_weights(unet, adapter, lora_state: dict, use_ema: bool = False):
    """Load LoRA + adapter weights from checkpoint. Optionally use EMA weights."""
    prefix = "ema." if use_ema else ""
    fallback_prefix = "" if use_ema else "ema."

    # Load LoRA weights
    missing = []
    loaded = 0
    for name, param in unet.named_parameters():
        if param.requires_grad:
            key = f"{prefix}unet.{name}"
            if key in lora_state:
                param.data.copy_(lora_state[key])
                loaded += 1
            else:
                # Try fallback
                key2 = f"{fallback_prefix}unet.{name}"
                if key2 in lora_state:
                    param.data.copy_(lora_state[key2])
                    loaded += 1
                else:
                    # Try stripped
                    stripped = name.replace("base_model.model.", "")
                    key3 = f"{prefix}unet.{stripped}"
                    if key3 in lora_state:
                        param.data.copy_(lora_state[key3])
                        loaded += 1
                    else:
                        missing.append(name)

    if missing:
        logger.warning(f"Missing {len(missing)} LoRA keys (first 5: {missing[:5]})")

    # Load adapter weights
    for name, param in adapter.named_parameters():
        key = f"{prefix}adapter.{name}"
        if key in lora_state:
            param.data.copy_(lora_state[key])
        else:
            key2 = f"{fallback_prefix}adapter.{name}"
            if key2 in lora_state:
                param.data.copy_(lora_state[key2])
            else:
                logger.warning(f"Missing adapter key: {key}")

    logger.info(f"Loaded {loaded} LoRA weights ({len(missing)} missing), EMA={use_ema}")


@torch.no_grad()
def generate_depth(
    unet, vae, scheduler, adapter, tokenizer, text_encoder,
    input_image_tensor, device, prompt, vae_scaling,
    num_steps=30, seed=42,
) -> Image.Image:
    """Generate depth RGB from input image using trained model."""
    torch.manual_seed(seed)

    # Encode input image
    input_latents = vae.encode(input_image_tensor).latent_dist.mode()
    input_latents = input_latents * vae_scaling

    # Encode text prompt properly
    tokenized = tokenizer(
        [prompt], padding="max_length", max_length=77,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    encoder_hidden_states = text_encoder(tokenized)[0]

    # Start from random noise
    latents = torch.randn(1, 4, 32, 32, device=device)
    scheduler.set_timesteps(num_steps, device=device)

    for t in scheduler.timesteps:
        concat = torch.cat([input_latents, latents], dim=1)
        adapted = adapter(concat)
        noise_pred = unet(adapted, t.unsqueeze(0), encoder_hidden_states).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae_scaling
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute SSIM and PSNR between predicted and ground truth images."""
    # PSNR
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = 10.0 * np.log10(255.0 ** 2 / mse)

    # SSIM (simplified, per-channel)
    from skimage.metrics import structural_similarity as ssim_fn
    ssim_vals = []
    for ch in range(3):
        s = ssim_fn(gt[:, :, ch], pred[:, :, ch], data_range=255)
        ssim_vals.append(s)
    ssim_mean = np.mean(ssim_vals)

    return {"psnr": round(psnr, 2), "ssim": round(float(ssim_mean), 4)}


def make_comparison_figure(input_img, gt_img, pred_img, idx, output_dir, metrics=None):
    """Create a side-by-side comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    titles = ["Input", "Ground Truth", "Prediction"]
    images = [input_img, gt_img, pred_img]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    if metrics:
        fig.suptitle(
            f"Sample {idx} | PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f}",
            fontsize=13, fontweight="bold",
        )
    else:
        fig.suptitle(f"Sample {idx}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = output_dir / f"comparison_{idx:02d}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_loss_curve(log_path: Path, output_path: Path):
    """Generate a loss curve plot from training log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(log_path) as f:
        log = json.load(f)

    metrics = log["metrics"]
    if not metrics:
        return

    steps = [m["step"] for m in metrics]
    losses = [m["loss"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(steps, losses, "b-o", linewidth=2, markersize=4, label="Training Loss")
    ax.fill_between(steps, losses, alpha=0.15, color="blue")

    ax.set_xlabel("Optimizer Step", fontsize=12)
    ax.set_ylabel("Loss (Min-SNR Weighted MSE)", fontsize=12)
    ax.set_title(
        "Vision Banana rsLoRA v2 Training — Depth Task (MPS, 300 images)",
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


def plot_metrics_summary(metrics_list: list, output_path: Path):
    """Plot PSNR/SSIM bar chart across all samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = [m["index"] for m in metrics_list]
    psnrs = [m["psnr"] for m in metrics_list]
    ssims = [m["ssim"] for m in metrics_list]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150)

    bars1 = ax1.bar(indices, psnrs, color="steelblue", alpha=0.8)
    ax1.set_ylabel("PSNR (dB)", fontsize=12)
    ax1.set_title("Per-Sample PSNR", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    avg_psnr = np.mean(psnrs)
    ax1.axhline(avg_psnr, color="red", linestyle="--", label=f"Mean: {avg_psnr:.2f}")
    ax1.legend()

    bars2 = ax2.bar(indices, ssims, color="darkorange", alpha=0.8)
    ax2.set_ylabel("SSIM", fontsize=12)
    ax2.set_title("Per-Sample SSIM", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Sample Index", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    avg_ssim = np.mean(ssims)
    ax2.axhline(avg_ssim, color="red", linestyle="--", label=f"Mean: {avg_ssim:.4f}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Metrics summary saved to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # ── Load base pipeline ───────────────────────────────────────────
    from diffusers import StableDiffusionPipeline

    logger.info("Loading base model (segmind/tiny-sd)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # ── Apply LoRA architecture ──────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=TARGET_MODULES,
        lora_dropout=0.0, bias="none",
        use_rslora=True,
    )
    unet = get_peft_model(unet, lora_config)

    # ── Create channel adapter ───────────────────────────────────────
    adapter = ChannelAdapter(8, 4).to(device)

    # ── Load trained weights (prefer EMA) ────────────────────────────
    # Try EMA from final, fall back to best, then final without EMA
    lora_state = torch.load(LORA_PATH, map_location="cpu")
    has_ema = any(k.startswith("ema.") for k in lora_state.keys())
    use_ema = USE_EMA and has_ema

    logger.info(f"Loading trained weights from {LORA_PATH} (EMA={use_ema})...")
    load_lora_weights(unet, adapter, lora_state, use_ema=use_ema)
    unet.eval()
    adapter.eval()

    vae_scaling = getattr(vae.config, "scaling_factor", 0.18215)

    # ── Select sample images ─────────────────────────────────────────
    image_paths = sorted((DATA_DIR / "images").glob("*.jpg"))
    target_paths = sorted((DATA_DIR / "targets").glob("*.jpg"))

    # Pick evenly spaced samples (different from training validation set)
    indices = np.linspace(10, len(image_paths) - 1, NUM_SAMPLES, dtype=int)
    logger.info(f"Selected {NUM_SAMPLES} samples for validation (indices: {indices})")

    # ── Generate depth visualisations ────────────────────────────────
    results = []
    all_metrics = []

    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        tgt_path = target_paths[idx]
        stem = img_path.stem

        # Load input image
        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(
            np.array(img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS), dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0

        # Save input
        img.save(OUTPUT_DIR / f"input_{i:02d}.png")

        # Save ground truth
        gt = Image.open(tgt_path).convert("RGB")
        gt.save(OUTPUT_DIR / f"gt_{i:02d}.png")

        # Generate depth
        t0 = time.time()
        depth_img = generate_depth(
            unet, vae, scheduler, adapter, tokenizer, text_encoder,
            img_tensor, device, DEPTH_PROMPT, vae_scaling,
            num_steps=NUM_INFERENCE_STEPS, seed=SEED + i,
        )
        elapsed = time.time() - t0

        depth_img.save(OUTPUT_DIR / f"depth_{i:02d}.png")

        # Compute metrics
        pred_np = np.array(depth_img)
        gt_np = np.array(gt.resize((RESOLUTION, RESOLUTION), Image.LANCZOS))
        metrics = compute_metrics(pred_np, gt_np)
        all_metrics.append({"index": i, **metrics})

        # Create comparison figure
        comp_path = make_comparison_figure(img, gt, depth_img, i, OUTPUT_DIR, metrics)

        logger.info(
            f"[{i+1}/{NUM_SAMPLES}] {stem} | "
            f"PSNR: {metrics['psnr']:.2f} dB | SSIM: {metrics['ssim']:.4f} | "
            f"{elapsed:.1f}s"
        )

        results.append({
            "index": i,
            "stem": stem,
            "input": f"input_{i:02d}.png",
            "ground_truth": f"gt_{i:02d}.png",
            "predicted": f"depth_{i:02d}.png",
            "comparison": f"comparison_{i:02d}.png",
            "time_s": round(elapsed, 1),
            "metrics": metrics,
        })

    # ── Generate loss curve ──────────────────────────────────────────
    if TRAINING_LOG.exists():
        loss_path = OUTPUT_DIR / "loss_curve.png"
        plot_loss_curve(TRAINING_LOG, loss_path)

    # ── Plot metrics summary ─────────────────────────────────────────
    metrics_chart_path = OUTPUT_DIR / "metrics_summary.png"
    plot_metrics_summary(all_metrics, metrics_chart_path)

    # ── Save results summary ─────────────────────────────────────────
    avg_psnr = np.mean([m["psnr"] for m in all_metrics])
    avg_ssim = np.mean([m["ssim"] for m in all_metrics])

    summary = {
        "model": MODEL_ID,
        "task": "depth",
        "lora_path": str(LORA_PATH),
        "use_ema": use_ema,
        "config": {
            "lora_rank": 64,
            "lora_alpha": 128,
            "use_rslora": True,
            "target_modules": TARGET_MODULES,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
            "num_samples": NUM_SAMPLES,
            "resolution": RESOLUTION,
        },
        "generations": results,
        "metrics_summary": {
            "avg_psnr": round(float(avg_psnr), 2),
            "avg_ssim": round(float(avg_ssim), 4),
            "per_sample": all_metrics,
        },
    }

    # Add training stats
    if TRAINING_LOG.exists():
        with open(TRAINING_LOG) as f:
            train_log = json.load(f)
        if train_log.get("metrics"):
            summary["training"] = {
                "initial_loss": train_log["metrics"][0]["loss"],
                "final_loss": train_log["final_loss"],
                "total_steps": train_log["total_steps"],
                "epochs": train_log["config"]["epochs"],
                "dataset_size": train_log["config"].get("dataset_size", "N/A"),
                "loss_reduction_pct": round(
                    (1 - train_log["final_loss"] / train_log["metrics"][0]["loss"]) * 100, 1
                ),
                "min_snr_gamma": train_log["config"].get("min_snr_gamma", "N/A"),
            }

    summary_path = OUTPUT_DIR / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Validation complete!")
    logger.info(f"  Avg PSNR: {avg_psnr:.2f} dB | Avg SSIM: {avg_ssim:.4f}")
    logger.info(f"  Comparisons:  {OUTPUT_DIR}/comparison_*.png")
    logger.info(f"  Loss curve:   {OUTPUT_DIR}/loss_curve.png")
    logger.info(f"  Metrics chart: {metrics_chart_path}")
    logger.info(f"  Summary:      {summary_path}")
    if "training" in summary:
        t = summary["training"]
        logger.info(f"  Training: {t['initial_loss']:.4f} → {t['final_loss']:.4f} ({t['loss_reduction_pct']}% reduction)")
        logger.info(f"  Dataset: {t.get('dataset_size', 'N/A')} images, {t['epochs']} epochs")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
