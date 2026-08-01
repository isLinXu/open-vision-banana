#!/usr/bin/env python3
"""Depth task overfit gate: InstructPix2Pix-style conditioning + native epsilon.

Uses channel concatenation (4+4=8 channels) for source image conditioning
instead of ControlNet. This avoids ControlNet architecture mismatch issues
with SSD-1B and is simpler to implement.

The UNet conv_in is modified to accept 8 input channels (4 noisy latent + 4 source latent).
Only the new conv_in weights and LoRA adapters are trained.

Gate criteria: SSIM > 0.80 AND L1 < 0.10 on training samples after 300-500 steps.

Usage:
  python scripts/overfit_gate_depth.py --use-synthetic --num-samples 16 --max-steps 300
  python scripts/overfit_gate_depth.py --num-samples 8 --max-steps 500 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("overfit_gate")

MODEL_ID = "segmind/SSD-1B"
RESOLUTION = 512
SEED = 42
LATENT_CHANNELS = 4
VAE_SCALE = 8


# ─── Synthetic Depth Pairs ────────────────────────────────────────────────────


def generate_synthetic_depth_pairs(n: int = 16, size: int = 512) -> list[dict]:
    """Generate simple synthetic input→depth pairs for overfit gate."""
    rng = np.random.RandomState(SEED)
    pairs = []

    for i in range(n):
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 128
        depth_map = np.zeros((size, size), dtype=np.float32)
        depth_map[:] = 0.5

        n_shapes = rng.randint(2, 4)
        for s in range(n_shapes):
            shape_type = rng.choice(["rect", "circle"])
            depth_val = rng.uniform(0.1, 0.9)

            if shape_type == "rect":
                x0, y0 = rng.randint(50, size - 200, size=2)
                w, h = rng.randint(80, 200, size=2)
                x1, y1 = min(x0 + w, size), min(y0 + h, size)
                color = rng.randint(0, 255, size=3)
                canvas[y0:y1, x0:x1] = color
                depth_map[y0:y1, x0:x1] = depth_val
            else:
                cx, cy = rng.randint(100, size - 100, size=2)
                r = rng.randint(40, 100)
                yy, xx = np.ogrid[:size, :size]
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
                color = rng.randint(0, 255, size=3)
                canvas[mask] = color
                depth_map[mask] = depth_val

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.cm as cm

            norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            depth_rgb = (cm.turbo(norm)[:, :, :3] * 255).astype(np.uint8)
        except ImportError:
            depth_rgb = np.stack([depth_map * 255] * 3, axis=-1).astype(np.uint8)

        pairs.append(
            {
                "input": Image.fromarray(canvas),
                "target": Image.fromarray(depth_rgb),
                "prompt": "Estimate the depth map. Closer objects in warm colors, farther in cool colors.",
            }
        )

    return pairs


def load_real_depth_pairs(v3_root: Path, n: int = 16) -> list[dict]:
    """Load depth pairs from v3 dataset."""
    depth_dir = v3_root / "depth" / "train"
    if not depth_dir.exists():
        return []

    pairs = []
    input_files = sorted((depth_dir / "input").glob("*.jpg"))[:n]
    for inp_path in input_files:
        tgt_path = depth_dir / "target" / f"{inp_path.stem}.png"
        if not tgt_path.exists():
            continue
        pairs.append(
            {
                "input": Image.open(inp_path).convert("RGB"),
                "target": Image.open(tgt_path).convert("RGB"),
                "prompt": "Estimate the depth map. Closer objects in warm colors, farther in cool colors.",
            }
        )

    return pairs


# ─── Model Loading ────────────────────────────────────────────────────────────


def load_models(device: str = "mps"):
    """Load SSD-1B pipeline components."""
    from diffusers import (
        AutoencoderKL,
        UNet2DConditionModel,
        EulerDiscreteScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    dtype = torch.float32 if device == "cpu" else torch.float16

    logger.info(f"Loading VAE from {MODEL_ID}...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype)

    logger.info(f"Loading UNet from {MODEL_ID}...")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=dtype)

    logger.info(f"Loading text encoders from {MODEL_ID}...")
    tokenizer_one = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=dtype
    )
    text_encoder_two = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder_2", torch_dtype=dtype
    )

    logger.info(f"Loading scheduler from {MODEL_ID}...")
    scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    logger.info(f"Scheduler prediction_type: {scheduler.config.prediction_type}")

    for m in [vae, unet, text_encoder_one, text_encoder_two]:
        m.to(device)
        m.requires_grad_(False)

    return {
        "vae": vae,
        "unet": unet,
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "scheduler": scheduler,
    }


def modify_unet_for_ip2p(unet, device: str):
    """Modify UNet conv_in to accept 8 channels (4 noisy + 4 source latent).

    InstructPix2Pix approach: concatenate source latent with noisy latent.
    """
    old_conv = unet.conv_in
    old_weight = old_conv.weight.data.clone()
    old_bias = old_conv.bias.data.clone() if old_conv.bias is not None else None

    in_channels = old_conv.in_channels  # 4
    out_channels = old_conv.out_channels  # 320
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride
    padding = old_conv.padding

    new_conv = nn.Conv2d(
        in_channels * 2,  # 4 → 8
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    ).to(device)

    # Initialize: first 4 channels copy old weights, last 4 channels zero-init
    with torch.no_grad():
        new_conv.weight.data[:, :in_channels] = old_weight
        new_conv.weight.data[:, in_channels:] = 0  # zero-init for source channels
        if old_bias is not None:
            new_conv.bias.data = old_bias

    unet.conv_in = new_conv
    unet.config.in_channels = in_channels * 2

    # Make conv_in trainable
    for p in unet.conv_in.parameters():
        p.requires_grad_(True)

    logger.info(f"Modified UNet conv_in: {in_channels}→{in_channels * 2} channels")
    logger.info(
        f"  conv_in trainable params: {sum(p.numel() for p in unet.conv_in.parameters()):,}"
    )

    return unet


def setup_lora(unet, rank: int = 64, alpha: int = 128):
    """Attach rsLoRA to UNet attention + FFN layers."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
        ],
        lora_dropout=0.0,
        bias="none",
        use_rslora=True,
    )

    unet = get_peft_model(unet, config)
    base_dtype = next(unet.base_model.parameters()).dtype
    unet = unet.to(dtype=base_dtype)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(f"LoRA: {trainable:,} trainable / {total:,} total ({100 * trainable / total:.2f}%)")
    return unet


# ─── Encoding ─────────────────────────────────────────────────────────────────


def encode_text(models, prompt: str, device: str):
    """Encode a text prompt for SSD-1B's dual-encoder architecture.

    SSD-1B (and SDXL) uses two CLIP text encoders:
    - text_encoder_one: CLIPTextModel -> last_hidden_state (77, 768)
    - text_encoder_two: CLIPTextModelWithProjection -> last_hidden_state + text_embeds (pooled)

    The pooled prompt embeds must be the projected output (.text_embeds),
    NOT the [CLS] token of the last_hidden_state.  Using the wrong tensor
    as pooled embeds causes added_cond_kwargs to receive semantically
    meaningless vectors, which corrupts the micro-conditioning signal.
    """
    tok1 = models["tokenizer_one"](
        prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True
    ).input_ids.to(device)
    tok2 = models["tokenizer_two"](
        prompt, return_tensors="pt", padding="max_length", max_length=77, truncation=True
    ).input_ids.to(device)
    with torch.no_grad():
        e1 = models["text_encoder_one"](tok1)[0]  # last_hidden_state [1, 77, 768]
        enc2_out = models["text_encoder_two"](tok2)  # call only once
        e2 = enc2_out[0]  # last_hidden_state [1, 77, 1280]
        # CLIPTextModelWithProjection exposes .text_embeds (projected pooler output)
        pooled = enc2_out.text_embeds  # [1, 1280]
    return torch.cat([e1, e2], dim=-1), pooled


def vae_encode(models, image: Image.Image, device: str):
    dtype = models["vae"].dtype
    arr = (
        np.array(image.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)).astype(np.float32) / 127.5
        - 1.0
    )
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        lat = models["vae"].encode(t).latent_dist.sample() * models["vae"].config.scaling_factor
    return lat


def vae_decode(models, latent, device: str):
    with torch.no_grad():
        latent = latent / models["vae"].config.scaling_factor
        decoded = models["vae"].decode(latent.to(models["vae"].dtype))
        img = (decoded.sample / 2 + 0.5).clamp(0, 1)
    return img


# ─── Training ────────────────────────────────────────────────────────────────


def train_overfit_gate(
    models: dict,
    pairs: list[dict],
    num_steps: int = 300,
    lr: float = 1e-4,
    device: str = "mps",
):
    """Train IP2P-style (channel concat) + LoRA on tiny fixed set."""
    unet = models["unet"]

    # Modify UNet for 8-channel input
    unet = modify_unet_for_ip2p(unet, device)

    # Setup LoRA
    unet = setup_lora(unet)
    models["unet"] = unet

    dtype = torch.float16

    # Trainable params: conv_in + LoRA
    conv_in_params = [p for p in unet.conv_in.parameters() if p.requires_grad]
    lora_params = [p for p in unet.parameters() if p.requires_grad and p not in conv_in_params]
    all_params = conv_in_params + lora_params

    # Cast trainable to fp32 for optimizer, keep base in fp16
    for p in all_params:
        p.data = p.data.to(torch.float32)

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    # Pre-encode
    logger.info(f"Pre-encoding {len(pairs)} pairs...")
    encoded = []
    for pair in pairs:
        target_latent = vae_encode(models, pair["target"], device)
        source_latent = vae_encode(models, pair["input"], device)
        prompt_embeds, pooled = encode_text(models, pair["prompt"], device)
        encoded.append(
            {
                "target_latent": target_latent,
                "source_latent": source_latent,
                "prompt_embeds": prompt_embeds,
                "pooled": pooled,
            }
        )

    scheduler = models["scheduler"]
    rng = torch.Generator(device="cpu").manual_seed(SEED)

    # Initialize scheduler timesteps for training (needed for add_noise)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

    losses = []
    eval_interval = 100  # less frequent eval to save time
    eval_num_steps = 10  # fewer inference steps for eval

    logger.info(f"Starting overfit gate: {num_steps} steps, {len(pairs)} samples, lr={lr}")
    logger.info(f"  Architecture: IP2P (8ch conv_in) + LoRA, native epsilon prediction")

    for step in range(num_steps):
        idx = torch.randint(0, len(encoded), (1,), generator=rng).item()
        data = encoded[idx]

        z0 = data["target_latent"]
        source_lat = data["source_latent"]
        prompt_embeds = data["prompt_embeds"]
        pooled = data["pooled"]

        noise = torch.randn_like(z0)
        b = z0.shape[0]

        # Logit-normal timestep
        u = torch.randn(b, device="cpu", generator=rng).to(device)
        timesteps = torch.sigmoid(u) * (scheduler.config.num_train_timesteps - 1)
        timesteps = timesteps.long()

        # Add noise (epsilon prediction)
        noisy = scheduler.add_noise(z0, noise, timesteps)

        # Scale ONLY the noisy target, NOT the clean source latent.
        # EulerDiscreteScheduler.scale_model_input divides by sqrt(sigma^2+1).
        # The clean source latent is conditioning, not part of the denoised
        # trajectory. Scaling the entire 8-channel concat (old code) attenuates
        # the source signal at high-noise timesteps, making the model blind to
        # the input image -- the primary cause of mean-prediction outputs.
        step_idx = scheduler.index_for_timestep(int(timesteps[0]), scheduler.timesteps.to(device))
        sigma = scheduler.sigmas.to(device=device, dtype=torch.float16)[step_idx]
        scaled_noisy = noisy / ((sigma**2 + 1) ** 0.5)
        model_input = torch.cat([scaled_noisy, source_lat], dim=1)

        added_cond_kwargs = {
            "text_embeds": pooled,
            "time_ids": torch.tensor(
                [[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]], device=device, dtype=dtype
            ),
        }

        with torch.autocast(device_type=device, dtype=torch.float16):
            pred = unet(
                model_input.to(dtype),
                timesteps,
                encoder_hidden_states=prompt_embeds.to(dtype),
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )
        pred = pred[0] if isinstance(pred, tuple) else pred.sample

        # Native epsilon prediction loss
        target = noise
        loss = F.mse_loss(pred.float(), target.float())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 10 == 0:
            avg = np.mean(losses[-10:])
            logger.info(f"Step {step + 1}/{num_steps}: loss={loss.item():.4f}, avg10={avg:.4f}")

        if (step + 1) % eval_interval == 0 or step == 0:
            metrics = evaluate_reconstruction(
                models, encoded, device, num_eval=min(2, len(encoded)), num_steps=eval_num_steps
            )
            logger.info(
                f"Step {step + 1} eval: SSIM={metrics['ssim']:.4f}, L1={metrics['l1']:.4f}, "
                f"PSNR={metrics['psnr']:.2f}"
            )

    return losses


def evaluate_reconstruction(models, encoded, device, num_eval=4, num_steps=10):
    """Evaluate reconstruction quality on training samples."""
    from skimage.metrics import structural_similarity as ssim
    import copy

    # Deep copy scheduler to avoid mutating training state
    scheduler = copy.deepcopy(models["scheduler"])
    dtype = torch.float16

    ssim_scores = []
    l1_scores = []
    psnr_scores = []

    for i in range(min(num_eval, len(encoded))):
        data = encoded[i]
        z0 = data["target_latent"]
        source_lat = data["source_latent"]
        prompt_embeds = data["prompt_embeds"]
        pooled = data["pooled"]

        num_inference_steps = num_steps
        scheduler.set_timesteps(num_inference_steps, device=device)

        latent = torch.randn_like(z0)

        with torch.no_grad():
            for j, t in enumerate(scheduler.timesteps):
                # Scale ONLY the noisy target, NOT the clean source latent.
                sigma = scheduler.sigmas[j]
                scaled_noisy = latent / ((sigma**2 + 1) ** 0.5)
                model_input = torch.cat([scaled_noisy, source_lat], dim=1)

                added_cond = {
                    "text_embeds": pooled,
                    "time_ids": torch.tensor(
                        [[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]],
                        device=device,
                        dtype=dtype,
                    ),
                }

                with torch.autocast(device_type=device, dtype=torch.float16):
                    noise_pred = models["unet"](
                        model_input.to(dtype),
                        t.unsqueeze(0),
                        encoder_hidden_states=prompt_embeds.to(dtype),
                        added_cond_kwargs=added_cond,
                        return_dict=False,
                    )
                noise_pred = noise_pred[0] if isinstance(noise_pred, tuple) else noise_pred.sample

                # Use scheduler.step for correct Euler ancestral sampling.
                # The old manual Euler was mathematically close but did not
                # match EulerDiscreteScheduler.step() exactly (it omits the
                # sigma-derived step size and uses a simplified x0 formula).
                latent = scheduler.step(
                    noise_pred.float(), t, latent,
                ).prev_sample

        pred_img = vae_decode(models, latent, device)
        target_img = vae_decode(models, z0, device)

        pred_np = pred_img[0].cpu().float().permute(1, 2, 0).numpy()
        target_np = target_img[0].cpu().float().permute(1, 2, 0).numpy()

        s = ssim(target_np, pred_np, channel_axis=2, data_range=1.0)
        l1 = float(np.mean(np.abs(target_np - pred_np)))
        mse = float(np.mean((target_np - pred_np) ** 2))
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))

        ssim_scores.append(s)
        l1_scores.append(l1)
        psnr_scores.append(psnr)

    return {
        "ssim": float(np.mean(ssim_scores)),
        "l1": float(np.mean(l1_scores)),
        "psnr": float(np.mean(psnr_scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Depth overfit gate (IP2P-style)")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument(
        "--v3-root",
        type=str,
        default="/Users/gatilin/MyWork/datasets/lllab-vision-banana-dataset/v3",
    )
    parser.add_argument("--output-dir", type=str, default="output/overfit_gate_depth")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if args.use_synthetic:
        pairs = generate_synthetic_depth_pairs(args.num_samples)
    else:
        pairs = load_real_depth_pairs(Path(args.v3_root), args.num_samples)
        if len(pairs) < args.num_samples:
            synthetic = generate_synthetic_depth_pairs(args.num_samples - len(pairs))
            pairs.extend(synthetic)

    logger.info(f"Using {len(pairs)} pairs for overfit gate")

    models = load_models(device)
    losses = train_overfit_gate(models, pairs, args.max_steps, args.lr, device)

    # Final eval
    encoded = []
    for pair in pairs:
        target_latent = vae_encode(models, pair["target"], device)
        source_latent = vae_encode(models, pair["input"], device)
        prompt_embeds, pooled = encode_text(models, pair["prompt"], device)
        encoded.append(
            {
                "target_latent": target_latent,
                "source_latent": source_latent,
                "prompt_embeds": prompt_embeds,
                "pooled": pooled,
            }
        )

    metrics = evaluate_reconstruction(
        models, encoded, device, num_eval=min(8, len(pairs)), num_steps=20
    )

    logger.info("=" * 60)
    logger.info(f"OVERFIT GATE RESULTS ({len(pairs)} samples, {args.max_steps} steps)")
    logger.info(f"  Architecture: IP2P (8ch) + LoRA, epsilon prediction")
    logger.info(f"  SSIM:  {metrics['ssim']:.4f}  (gate: > 0.80)")
    logger.info(f"  L1:    {metrics['l1']:.4f}  (gate: < 0.10)")
    logger.info(f"  PSNR:  {metrics['psnr']:.2f} dB")
    passed = metrics["ssim"] > 0.80 and metrics["l1"] < 0.10
    logger.info(f"  GATE:  {'PASSED' if passed else 'FAILED'}")
    logger.info("=" * 60)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "num_samples": len(pairs),
        "max_steps": args.max_steps,
        "lr": args.lr,
        "use_synthetic": args.use_synthetic,
        "architecture": "ip2p_8ch_lora_epsilon",
        "losses": losses,
        "final_metrics": metrics,
        "gate_passed": passed,
        "gate_thresholds": {"ssim": 0.80, "l1": 0.10},
    }
    with open(out_dir / "overfit_gate_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save visualization
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(min(4, len(pairs)), 3, figsize=(12, 4 * min(4, len(pairs))))
        if min(4, len(pairs)) == 1:
            axes = axes[None, :]

        for i in range(min(4, len(pairs))):
            axes[i, 0].imshow(np.array(pairs[i]["input"].resize((256, 256))))
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(np.array(pairs[i]["target"].resize((256, 256))))
            axes[i, 1].set_title("Target")
            axes[i, 1].axis("off")

            data = encoded[i]
            pred_img = vae_decode(models, data["target_latent"], device)
            pred_np = pred_img[0].cpu().float().permute(1, 2, 0).numpy()
            axes[i, 2].imshow(pred_np)
            axes[i, 2].set_title("VAE decode (target)")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / "overfit_gate_visualization.png", dpi=150)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
