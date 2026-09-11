#!/usr/bin/env python3
"""Multi-task training framework: ControlNet + LoRA with native epsilon prediction.

Supports the paper's 5 tasks (depth, normal, semantic, instance, referring)
plus a generation protection branch. Uses per-step conditioning (ControlNet)
and native epsilon/v prediction semantics.

Training mix:
  - 90% vision task steps (configurable per-task ratio)
  - 10% generation protection steps (text-to-image, no ControlNet)
  - NOTE: 90/10 is a project choice, not a paper-disclosed value.

Usage:
  python scripts/train_v5_multitask.py --tasks depth,semantic --max-steps 5000
  python scripts/train_v5_multitask.py --tasks all --generation-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from open_vision_banana.dynamic_prompts import build_prompt, PROMPT_BUILDERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_v5")

# ─── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "segmind/SSD-1B"
CN_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"
RESOLUTION = 512
SEED = 20260726
LATENT_CHANNELS = 4
VAE_SCALE = 8
DEFAULT_GENERATION_RATIO = 0.10  # 10% generation protection — PROJECT CHOICE


# ─── Data Loading ─────────────────────────────────────────────────────────────

@dataclass
class TrainingSample:
    sample_id: str
    task: str
    input_path: str
    target_path: str
    prompt: str
    meta: dict = field(default_factory=dict)


class V5Dataset:
    """Multi-task dataset that loads from v3 manifest."""

    def __init__(self, v3_root: Path, tasks: list[str], split: str = "train"):
        self.v3_root = v3_root
        self.tasks = tasks
        self.split = split
        self.samples: list[TrainingSample] = []
        self.task_indices: dict[str, list[int]] = {}
        self._load()

    def _load(self):
        samples_jsonl = self.v3_root / "v3_samples.jsonl"
        if not samples_jsonl.exists():
            raise FileNotFoundError(f"V3 samples not found: {samples_jsonl}")

        with open(samples_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                if rec["split"] != self.split:
                    continue
                if rec["task"] not in self.tasks:
                    continue
                idx = len(self.samples)
                self.samples.append(TrainingSample(
                    sample_id=rec["sample_id"],
                    task=rec["task"],
                    input_path=rec["input_path"],
                    target_path=rec["target_path"],
                    prompt=rec["prompt"],
                    meta=rec.get("meta", {}),
                ))
                self.task_indices.setdefault(rec["task"], []).append(idx)

        logger.info(f"V5Dataset ({self.split}): {len(self.samples)} samples")
        for task, indices in self.task_indices.items():
            logger.info(f"  {task}: {len(indices)} samples")

    def get_sample(self, task: str | None = None) -> TrainingSample:
        """Get a random sample, optionally from a specific task."""
        if task and task in self.task_indices:
            idx = random.choice(self.task_indices[task])
        else:
            idx = random.randint(0, len(self.samples) - 1)
        return self.samples[idx]


# ─── Model Setup ──────────────────────────────────────────────────────────────

def load_models(device: str = "mps"):
    """Load SSD-1B + ControlNet with native epsilon prediction."""
    from diffusers import (
        AutoencoderKL,
        ControlNetModel,
        UNet2DConditionModel,
        EulerDiscreteScheduler,
    )
    from transformers import CLIPTextModel, CLIPTokenizer

    dtype = torch.float32 if device == "cpu" else torch.float16

    logger.info("Loading model components...")
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=dtype)
    tokenizer_one = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    tokenizer_two = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
    text_encoder_one = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=dtype)
    text_encoder_two = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder_2", torch_dtype=dtype)
    scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    # Verify prediction_type is native (epsilon)
    pt = scheduler.config.prediction_type
    logger.info(f"Scheduler prediction_type: {pt}")
    assert pt in ("epsilon", "v_prediction"), f"Expected epsilon/v, got {pt}"

    # ControlNet
    try:
        controlnet = ControlNetModel.from_pretrained(CN_MODEL_ID, torch_dtype=dtype)
        logger.info("Pre-trained ControlNet loaded.")
    except Exception:
        controlnet = ControlNetModel.from_unet(unet)
        logger.info("ControlNet initialized from UNet.")

    # Move to device
    for m in [vae, unet, text_encoder_one, text_encoder_two, controlnet]:
        m.to(device)
        if m is not controlnet:
            m.requires_grad_(False)

    return {
        "vae": vae, "unet": unet, "controlnet": controlnet,
        "tokenizer_one": tokenizer_one, "tokenizer_two": tokenizer_two,
        "text_encoder_one": text_encoder_one, "text_encoder_two": text_encoder_two,
        "scheduler": scheduler,
    }


def setup_lora(unet, rank=64, alpha=128):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                       "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2"],
        lora_dropout=0.0, bias="none", use_rslora=True,
    )
    unet = get_peft_model(unet, config)
    n_train = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in unet.parameters())
    logger.info(f"LoRA: {n_train:,}/{n_total:,} ({100*n_train/n_total:.2f}%)")
    return unet


# ─── Encoding Utilities ───────────────────────────────────────────────────────

def encode_text(models, prompt: str, device: str):
    dtype = models["text_encoder_one"].dtype
    tok1 = models["tokenizer_one"](prompt, return_tensors="pt", padding="max_length",
                                    max_length=77, truncation=True).input_ids.to(device)
    tok2 = models["tokenizer_two"](prompt, return_tensors="pt", padding="max_length",
                                    max_length=77, truncation=True).input_ids.to(device)
    with torch.no_grad():
        e1 = models["text_encoder_one"](tok1)[0]
        e2 = models["text_encoder_two"](tok2)[0]
        pooled = models["text_encoder_two"](tok2)[0][:, 0, :]
    return torch.cat([e1, e2], dim=-1), pooled


def vae_encode(models, image: Image.Image, device: str):
    dtype = models["vae"].dtype
    arr = np.array(image.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        lat = models["vae"].encode(t).latent_dist.sample() * models["vae"].config.scaling_factor
    return lat


def prepare_cn_image(image: Image.Image, device: str, dtype):
    arr = np.array(image.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)


# ─── Training Step ────────────────────────────────────────────────────────────

def training_step_vision(
    models, sample: TrainingSample, device: str, rng: torch.Generator,
    use_min_snr: bool = True, min_snr_gamma: float = 5.0,
):
    """Single training step for a vision task (with ControlNet conditioning)."""
    dtype = next(models["unet"].parameters()).dtype
    scheduler = models["scheduler"]

    # Load images
    input_img = Image.open(sample.input_path).convert("RGB")
    target_img = Image.open(sample.target_path).convert("RGB")

    # Encode
    z0 = vae_encode(models, target_img, device)
    cn_cond = prepare_cn_image(input_img, device, dtype)
    prompt_embeds, pooled = encode_text(models, sample.prompt, device)

    # Sample noise + timestep (logit-normal)
    noise = torch.randn_like(z0, generator=rng)
    b = z0.shape[0]
    u = torch.randn(b, device=device, generator=rng)
    timesteps = torch.sigmoid(u) * (scheduler.config.num_train_timesteps - 1)
    timesteps = timesteps.long()

    # Add noise — epsilon prediction
    noisy = scheduler.add_noise(z0, noise, timesteps)

    # Scale model input
    step_idx = [scheduler.index_for_timestep(int(t), scheduler.timesteps.to(device)) for t in timesteps]
    sigma = scheduler.sigmas.to(device=device, dtype=noisy.dtype)[step_idx]
    sigma = sigma.view(b, *([1] * (noisy.dim() - 1)))
    model_input = noisy / ((sigma ** 2 + 1) ** 0.5)

    # ControlNet forward
    cn_residuals = models["controlnet"](
        model_input, timesteps,
        encoder_hidden_states=prompt_embeds.to(dtype),
        controlnet_cond=cn_cond,
        conditioning_scale=1.0,
        return_dict=False,
    )
    cn_down, cn_mid = cn_residuals[0], cn_residuals[1] if len(cn_residuals) > 1 else cn_residuals[0]

    # UNet forward
    added_cond = {
        "text_embeds": pooled.to(dtype),
        "time_ids": torch.tensor([[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]],
                                device=device, dtype=dtype),
    }
    pred = models["unet"](
        model_input.to(dtype), timesteps,
        encoder_hidden_states=prompt_embeds.to(dtype),
        added_cond_kwargs=added_cond,
        down_block_additional_residuals=cn_down if isinstance(cn_down, (list, tuple)) else [cn_down],
        mid_block_additional_residual=cn_mid if not isinstance(cn_mid, (list, tuple)) else cn_mid[0],
        return_dict=False,
    )
    pred = pred[0] if isinstance(pred, tuple) else pred.sample

    # Native epsilon prediction loss
    target = noise
    per_sample = (pred.float() - target.float()).square().mean(dim=(1, 2, 3))

    # Min-SNR weighting
    if use_min_snr:
        snr = 1.0 / (sigma.view(b).float() ** 2 + 1e-12)
        cap = torch.minimum(snr, torch.full_like(snr, min_snr_gamma))
        weight = cap / snr.clamp_min(1e-12)
        per_sample = per_sample * weight.view(per_sample.shape)

    return per_sample.mean()


def training_step_generation(
    models, device: str, rng: torch.Generator,
):
    """Generation protection step (text-to-image, no ControlNet)."""
    dtype = next(models["unet"].parameters()).dtype
    scheduler = models["scheduler"]

    # Random generation prompt
    prompt = build_prompt("generation")

    # Random latent (pure noise → target is also noise, but we sample a "fake" z0)
    # For generation protection, we use a simple regularization:
    # generate from noise, compute self-consistency loss
    h = w = RESOLUTION // VAE_SCALE
    z0 = torch.randn(1, LATENT_CHANNELS, h, w, device=device, generator=rng,
                     dtype=dtype)
    noise = torch.randn_like(z0, generator=rng)

    b = z0.shape[0]
    u = torch.randn(b, device=device, generator=rng)
    timesteps = torch.sigmoid(u) * (scheduler.config.num_train_timesteps - 1)
    timesteps = timesteps.long()

    noisy = scheduler.add_noise(z0, noise, timesteps)
    step_idx = [scheduler.index_for_timestep(int(t), scheduler.timesteps.to(device)) for t in timesteps]
    sigma = scheduler.sigmas.to(device=device, dtype=noisy.dtype)[step_idx]
    sigma = sigma.view(b, *([1] * (noisy.dim() - 1)))
    model_input = noisy / ((sigma ** 2 + 1) ** 0.5)

    prompt_embeds, pooled = encode_text(models, prompt, device)
    added_cond = {
        "text_embeds": pooled.to(dtype),
        "time_ids": torch.tensor([[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]],
                                device=device, dtype=dtype),
    }

    # No ControlNet for generation
    pred = models["unet"](
        model_input.to(dtype), timesteps,
        encoder_hidden_states=prompt_embeds.to(dtype),
        added_cond_kwargs=added_cond,
        return_dict=False,
    )
    pred = pred[0] if isinstance(pred, tuple) else pred.sample

    # Epsilon prediction
    target = noise
    loss = F.mse_loss(pred.float(), target.float())
    return loss


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(
    models, dataset: V5Dataset, val_dataset: V5Dataset | None,
    max_steps: int, lr: float, generation_ratio: float,
    task_ratios: dict[str, float] | None,
    save_every: int, output_dir: Path, device: str,
):
    """Main training loop."""
    # Setup LoRA
    models["unet"] = setup_lora(models["unet"])

    # ControlNet trainable
    models["controlnet"].requires_grad_(True)

    # Optimizer
    cn_params = list(models["controlnet"].parameters())
    lora_params = [p for p in models["unet"].parameters() if p.requires_grad]
    all_params = cn_params + lora_params
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.01)

    # Task ratios
    if task_ratios is None:
        n_tasks = len(dataset.tasks)
        task_ratios = {t: 1.0 / n_tasks for t in dataset.tasks}

    # Normalize ratios
    total_vision_ratio = sum(task_ratios.values())
    task_ratios = {t: r / total_vision_ratio for t, r in task_ratios.items()}

    logger.info(f"Training config:")
    logger.info(f"  Tasks: {dataset.tasks}")
    logger.info(f"  Task ratios: {task_ratios}")
    logger.info(f"  Generation ratio: {generation_ratio} (PROJECT CHOICE, not paper value)")
    logger.info(f"  Max steps: {max_steps}, LR: {lr}")
    logger.info(f"  Save every: {save_every}")

    rng = torch.Generator(device=device).manual_seed(SEED)
    losses = []
    task_losses = {t: [] for t in dataset.tasks + ["generation"]}

    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(max_steps):
        # Decide step type
        is_generation = random.random() < generation_ratio

        if is_generation:
            loss = training_step_generation(models, device, rng)
            task_key = "generation"
        else:
            # Pick task by ratio
            r = random.random()
            cumulative = 0.0
            selected_task = dataset.tasks[0]
            for task, ratio in task_ratios.items():
                cumulative += ratio
                if r < cumulative:
                    selected_task = task
                    break

            sample = dataset.get_sample(selected_task)
            loss = training_step_vision(models, sample, device, rng)
            task_key = selected_task

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        losses.append(loss.item())
        task_losses[task_key].append(loss.item())

        if (step + 1) % 10 == 0:
            avg = np.mean(losses[-10:])
            logger.info(f"Step {step+1}/{max_steps} [{task_key}]: loss={loss.item():.4f}, avg10={avg:.4f}")

        # Save checkpoint
        if (step + 1) % save_every == 0:
            ckpt_dir = output_dir / f"checkpoint-{step+1}"
            ckpt_dir.mkdir(exist_ok=True)
            # Save LoRA
            models["unet"].save_pretrained(ckpt_dir / "lora")
            # Save ControlNet
            models["controlnet"].save_pretrained(ckpt_dir / "controlnet")
            # Save training state
            state = {
                "step": step + 1,
                "losses": losses[-100:],
                "task_losses": {k: v[-100:] for k, v in task_losses.items()},
                "config": {
                    "tasks": dataset.tasks,
                    "task_ratios": task_ratios,
                    "generation_ratio": generation_ratio,
                    "lr": lr,
                    "seed": SEED,
                },
            }
            with open(ckpt_dir / "training_state.json", "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved checkpoint: {ckpt_dir}")

    # Final save
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    models["unet"].save_pretrained(final_dir / "lora")
    models["controlnet"].save_pretrained(final_dir / "controlnet")

    # Save full log
    with open(output_dir / "training_log.json", "w") as f:
        json.dump({
            "all_losses": losses,
            "task_losses": task_losses,
            "config": {
                "tasks": dataset.tasks,
                "task_ratios": task_ratios,
                "generation_ratio": generation_ratio,
                "max_steps": max_steps,
                "lr": lr,
                "seed": SEED,
            },
        }, f, indent=2)

    logger.info(f"Training complete. Final model: {final_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-root", type=str,
                       default="/Users/gatilin/MyWork/datasets/lllab-vision-banana-dataset/v3")
    parser.add_argument("--tasks", type=str, default="depth,normal,semantic,instance,referring",
                       help="Comma-separated task names, or 'all'")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--generation-ratio", type=float, default=DEFAULT_GENERATION_RATIO)
    parser.add_argument("--task-ratios", type=str, default=None,
                       help="JSON dict of task:ratio, e.g. '{\"depth\":0.3,\"semantic\":0.3,\"specular\":0.4}'")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="output/v5_multitask")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tasks = args.tasks.split(",") if args.tasks != "all" else \
            ["depth", "normal", "semantic", "instance", "referring"]

    task_ratios = json.loads(args.task_ratios) if args.task_ratios else None

    v3_root = Path(args.v3_root)
    dataset = V5Dataset(v3_root, tasks, split="train")
    val_dataset = V5Dataset(v3_root, tasks, split="val")

    models = load_models(device)
    train(models, dataset, val_dataset, args.max_steps, args.lr,
          args.generation_ratio, task_ratios, args.save_every,
          Path(args.output_dir), device)


if __name__ == "__main__":
    main()
