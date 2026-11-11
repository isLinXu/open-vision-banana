#!/usr/bin/env python3
"""Janus-1.3B continuous RGB output: generate RGB directly, without VQ tokens.

Improvements:
- Replace gen_head with an RGB head (predicts RGB values directly)
- MSE loss instead of CrossEntropy
- No VQ encode/decode, RGB is output directly
- Closer to the paper's continuous NBP generation

Usage:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_continuous_rgb.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, "/tmp/Janus")

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.training.tuner import VisionTaskDataset, _collate_fn

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "deepseek-ai/Janus-1.3B")
DATA_DIR = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
STEPS = int(os.environ.get("VB_STEPS", "500"))
NSAMPLES = int(os.environ.get("VB_NSAMPLES", "8"))
RES = int(os.environ.get("VB_RES", "384"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_continuous_rgb")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32
VISION_PROMPT = "Generate a metric depth visualization of this image."
IMAGE_TOKEN_NUM = 576  # 24x24


class RGBHead(nn.Module):
    """RGB prediction head: LLM hidden states → RGB values."""

    def __init__(self, hidden_dim: int, image_token_num: int, image_size: int):
        super().__init__()
        self.image_token_num = image_token_num
        self.image_size = image_size
        self.patch_size = image_size // int(math.sqrt(image_token_num))

        # hidden_dim → RGB (3 channels)
        self.proj = nn.Linear(hidden_dim, self.patch_size * self.patch_size * 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [b, image_token_num, hidden_dim]
        Returns:
            rgb: [b, 3, image_size, image_size]
        """
        b = hidden_states.shape[0]
        # [b, image_token_num, hidden_dim] → [b, image_token_num, patch_size*patch_size*3]
        x = self.proj(hidden_states)
        # [b, image_token_num, patch_size*patch_size*3] → [b, 3, image_size, image_size]
        x = x.view(b, self.image_token_num, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 4, 1, 2, 3)  # [b, 3, image_token_num, patch_size, patch_size]
        # reshape to [b, 3, image_size, image_size]
        h = w = int(math.sqrt(self.image_token_num))
        x = x.reshape(b, 3, h * self.patch_size, w * self.patch_size)
        return x


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={DTYPE}")
    print(f"[cfg] steps={STEPS}  nsamples={NSAMPLES}  res={RES}  batch={BATCH}")
    print(f"[cfg] lr={LR}  rank={RANK}  alpha={ALPHA}  out={OUT}")

    # 1) load Janus-1.3B ───────────────────────────────────────────────
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from janus.models.modeling_vlm import MultiModalityConfig, MultiModalityCausalLM
    AutoConfig.register("multi_modality", MultiModalityConfig)
    AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    print(f"[model] loaded in {time.time()-t0:.1f}s")
    print(f"[model] type={type(model).__name__}")
    print(f"[model] params={sum(p.numel() for p in model.parameters()):,}")

    # 2) dataset ───────────────────────────────────────────────────────
    ds = VisionTaskDataset(data_path=DATA_DIR, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data] loaded {n_total} depth samples @ {RES}px")
    if n_total < NSAMPLES:
        raise RuntimeError(f"Need {NSAMPLES} samples, have {n_total}")

    subset = torch.utils.data.Subset(ds, list(range(NSAMPLES)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )

    # 3) LoRA on language_model ─────────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=ALPHA,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model.language_model, lora_config)
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in peft_model.parameters())
    print(f"[lora] trainable = {n_train:,}  ({100*n_train/n_total_p:.2f}%)")

    # freeze the vision and generation parts
    model.vision_model.requires_grad_(False)
    model.gen_vision_model.requires_grad_(False)
    model.aligner.requires_grad_(False)
    model.gen_aligner.requires_grad_(False)
    model.gen_head.requires_grad_(False)
    model.gen_embed.requires_grad_(False)

    # 4) RGB Head (new) ────────────────────────────────────────────────
    hidden_dim = model.config.language_config.hidden_size
    rgb_head = RGBHead(hidden_dim, IMAGE_TOKEN_NUM, RES).to(device, DTYPE)
    rgb_head_params = sum(p.numel() for p in rgb_head.parameters())
    print(f"[rgb_head] params={rgb_head_params:,}  ({rgb_head_params/n_train*100:.2f}% of LoRA)")

    # 5) Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad] +
        [p for p in rgb_head.parameters()],
        lr=LR, weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 30)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 6) training loop ─────────────────────────────────────────────────
    print(f"\n[train] Janus continuous RGB: {STEPS} steps")
    print(f"[train] prompt: {VISION_PROMPT[:80]}...")

    # pre-encode the prompt
    prompt_ids = tokenizer(VISION_PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    losses: list[float] = []
    it = iter(loader)
    t_train = time.time()

    for step in range(1, STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        opt.zero_grad(set_to_none=True)

        # take a batch (batch_size=1 for simplicity)
        image = batch["image"][0:1].to(device, DTYPE)  # [-1, 1]
        target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)  # [0, 1]

        # 1. image understanding path
        pixel_values = image.unsqueeze(1)

        # 2. build the input embeds
        seq_len = prompt_len + IMAGE_TOKEN_NUM
        full_input_ids = torch.cat([
            prompt_ids,
            torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
        ], dim=1)
        images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        images_seq_mask[0, prompt_len:] = True
        images_emb_mask = torch.ones(1, 1, IMAGE_TOKEN_NUM, dtype=torch.bool, device=device)

        input_embeds = model.prepare_inputs_embeds(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )

        # 3. forward pass
        outputs = model.language_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state

        # 4. RGB prediction (new)
        gen_hidden = hidden_states[:, prompt_len:, :]  # [1, 576, D]
        pred_rgb = rgb_head(gen_hidden)  # [1, 3, RES, RES]

        # 5. MSE loss (continuous RGB)
        loss = F.mse_loss(pred_rgb, target_rgb)

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(peft_model.parameters()) + list(rgb_head.parameters()), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)
        if step % 500 == 0:
            mid_ckpt = OUT / f"ckpt_step{step}"
            mid_ckpt.mkdir(parents=True, exist_ok=True)
            peft_model.save_pretrained(str(mid_ckpt / "lora"))
            torch.save(rgb_head.state_dict(), mid_ckpt / "rgb_head.pt")
            print(f"[ckpt] mid-training save → {mid_ckpt}", flush=True)

    if not losses:
        raise RuntimeError("No finite training steps")

    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING' if avg_last < avg_first else 'NOT decreasing'})")

    # save the checkpoint
    ckpt_dir = OUT / "ckpt_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ckpt_dir / "lora"))
    torch.save(rgb_head.state_dict(), ckpt_dir / "rgb_head.pt")
    print(f"[ckpt] saved → {ckpt_dir}")

    # 7) inference evaluation ──────────────────────────────────────────
    print("\n[eval] inference on training samples...")
    mapper = DepthRGBMapper(colormap="viridis")
    results = []

    @torch.no_grad()
    def infer(sample_idx: int) -> dict:
        s = ds[sample_idx]
        inp_tensor = s["image"]  # [-1, 1]
        tgt_rgb = s["target_rgb"]  # [0, 1]
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy()
        )

        # inference
        pixel_values = inp_tensor.unsqueeze(0).unsqueeze(0).to(device, DTYPE)
        seq_len = prompt_len + IMAGE_TOKEN_NUM
        full_input_ids = torch.cat([
            prompt_ids,
            torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
        ], dim=1)
        images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        images_seq_mask[0, prompt_len:] = True
        images_emb_mask = torch.ones(1, 1, IMAGE_TOKEN_NUM, dtype=torch.bool, device=device)
        input_embeds = model.prepare_inputs_embeds(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        outputs = model.language_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        gen_hidden = hidden_states[:, prompt_len:, :]
        pred_rgb = rgb_head(gen_hidden).squeeze(0).clamp(0, 1).cpu()

        # triptych
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"panel_{sample_idx:02d}.png")

        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        return {"idx": sample_idx, "rgb_mae": rgb_mae}

    for idx in range(min(NSAMPLES, 2)):
        r = infer(idx)
        results.append(r)
        print(f"[eval] idx={idx:>3}  rgb_mae={r['rgb_mae']:.4f}")

    avg_mae = sum(r["rgb_mae"] for r in results) / len(results)
    print(f"\n[gate] avg rgb_mae={avg_mae:.4f}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] Janus continuous RGB complete ({total_min:.1f} min)")


if __name__ == "__main__":
    main()
