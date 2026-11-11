#!/usr/bin/env python3
"""Janus-1.3B continuous RGB v5 — Conv2D Decoder + Vision Fine-tuning.

v3 problem: LLM hidden states lose spatial information → blurry predictions
v4 problem: vision features + MLP lack spatial modelling capacity → even blurrier
v5 approach:
  1. Reshape the SigLIP features to 2D (24×24×1024) to keep the spatial structure
  2. Upsample 16× to 384×384×3 with Conv2D + PixelShuffle
  3. Unfreeze the last 4 SigLIP layers so the vision features adapt to the task
  4. Use BatchNorm (MPS compatible) + sensible initialization

Usage:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_continuous_rgb_v5.py
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

from open_vision_banana.training.tuner import VisionTaskDataset, _collate_fn

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "deepseek-ai/Janus-1.3B")
DATA_ROOT = os.environ.get(
    "VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3",
)

TASKS = os.environ.get("VB_TASKS", "depth").split(",")
N_PER_TASK = int(os.environ.get("VB_N_PER_TASK", "500"))
VAL_PER_TASK = int(os.environ.get("VB_VAL_PER_TASK", "50"))

STEPS = int(os.environ.get("VB_STEPS", "2000"))
RES = int(os.environ.get("VB_RES", "384"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "5e-5"))
VISION_LR = float(os.environ.get("VB_VISION_LR", "5e-6"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))

DROPOUT = float(os.environ.get("VB_DROPOUT", "0.1"))
LORA_DROPOUT = float(os.environ.get("VB_LORA_DROPOUT", "0.05"))
WEIGHT_DECAY = float(os.environ.get("VB_WEIGHT_DECAY", "0.01"))
PATIENCE = int(os.environ.get("VB_PATIENCE", "20"))
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "50"))
GRAD_CLIP = float(os.environ.get("VB_GRAD_CLIP", "1.0"))
AUGMENT_FLIP = os.environ.get("VB_AUGMENT_FLIP", "1") == "1"

# v5: unfreeze the last N layers of the vision model
UNFREEZE_VISION_BLOCKS = int(os.environ.get("VB_UNFREEZE_BLOCKS", "4"))

RESUME_CKPT = os.environ.get("VB_RESUME", "")

OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_continuous_rgb_v5")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32
IMAGE_TOKEN_NUM = 576

TASK_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate a surface normal estimation visualization of this image.",
    "instance": "Generate an instance segmentation visualization of this image.",
    "referring": "Generate a referring expression segmentation visualization of this image.",
    "keypoint": "Generate a keypoint detection visualization of this image.",
}


class ConvRGBDecoder(nn.Module):
    """Conv2D decoder: SigLIP features (24×24×1024) → RGB (384×384×3).

    Upsamples with PixelShuffle to keep spatial continuity.
    """

    def __init__(self, feat_dim: int = 1024, grid_size: int = 24, target_size: int = 384, dropout: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.target_size = target_size

        # 24×24 → 24×24 (channel reduction)
        self.stem = nn.Sequential(
            nn.Conv2d(feat_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        # 24×24 → 48×48 (PixelShuffle 2x)
        self.up1 = nn.Sequential(
            nn.Conv2d(512, 256 * 4, 3, padding=1),
            nn.BatchNorm2d(256 * 4),
            nn.GELU(),
            nn.PixelShuffle(2),
        )  # → 256 × 48 × 48
        # 48×48 → 96×96 (PixelShuffle 2x)
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.BatchNorm2d(128 * 4),
            nn.GELU(),
            nn.PixelShuffle(2),
        )  # → 128 × 96 × 96
        # 96×96 → 192×192 (PixelShuffle 2x)
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1),
            nn.BatchNorm2d(64 * 4),
            nn.GELU(),
            nn.PixelShuffle(2),
        )  # → 64 × 192 × 192
        # 192×192 → 384×384 (PixelShuffle 2x)
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.BatchNorm2d(32 * 4),
            nn.GELU(),
            nn.PixelShuffle(2),
        )  # → 32 × 384 × 384
        # Final 1×1 conv → 3 channels
        self.head = nn.Conv2d(32, 3, 1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, 576, D] → [B, 3, 384, 384]."""
        b = features.shape[0]
        x = features.transpose(1, 2)  # [B, D, 576]
        x = x.view(b, -1, self.grid_size, self.grid_size)  # [B, D, 24, 24]
        x = self.stem(x)    # [B, 512, 24, 24]
        x = self.up1(x)     # [B, 256, 48, 48]
        x = self.up2(x)     # [B, 128, 96, 96]
        x = self.up3(x)     # [B, 64, 192, 192]
        x = self.up4(x)     # [B, 32, 384, 384]
        x = self.head(x)    # [B, 3, 384, 384]
        x = torch.sigmoid(x)
        return x


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _augment_flip(image: torch.Tensor, target_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() < 0.5:
        return image.flip(-1), target_rgb.flip(-1)
    return image, target_rgb


def _unfreeze_vision_blocks(vision_model, n_blocks: int):
    """Unfreeze the last N transformer blocks of the vision model."""
    # freeze everything first
    vision_model.requires_grad_(False)

    # find the transformer blocks
    if hasattr(vision_model, "vision_tower"):
        tower = vision_model.vision_tower
    else:
        tower = vision_model

    # SigLIP ViT: the blocks live in tower.blocks
    if hasattr(tower, "blocks"):
        blocks = tower.blocks
        total = len(blocks)
        start = max(0, total - n_blocks)
        for i in range(start, total):
            for p in blocks[i].parameters():
                p.requires_grad_(True)
        print(f"[vision] unfrozen blocks {start}-{total-1} (last {n_blocks} of {total})")
    else:
        print(f"[vision] WARN: could not find blocks in {type(tower).__name__}, keeping all frozen")


def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={DTYPE}")
    print(f"[cfg] tasks={TASKS}  n_per_task={N_PER_TASK}  val_per_task={VAL_PER_TASK}")
    print(f"[cfg] steps={STEPS}  res={RES}  batch={BATCH}  lr={LR}  vision_lr={VISION_LR}")
    print(f"[cfg] rank={RANK}  alpha={ALPHA}")
    print(f"[cfg] dropout={DROPOUT}  lora_dropout={LORA_DROPOUT}  wd={WEIGHT_DECAY}")
    print(f"[cfg] patience={PATIENCE}  val_every={VAL_EVERY}  grad_clip={GRAD_CLIP}")
    print(f"[cfg] augment_flip={AUGMENT_FLIP}  unfreeze_blocks={UNFREEZE_VISION_BLOCKS}")
    print(f"[cfg] out={OUT}")

    # 1) load Janus-1.3B ───────────────────────────────────────────────
    import janus.models.modeling_vlm as janus_vlm
    for cfg_cls in [janus_vlm.VisionConfig, janus_vlm.AlignerConfig,
                    janus_vlm.GenVisionConfig, janus_vlm.GenAlignerConfig,
                    janus_vlm.GenHeadConfig]:
        if hasattr(cfg_cls, '__dataclass_fields__'):
            if 'params' in cfg_cls.__dataclass_fields__:
                cfg_cls.__dataclass_fields__['params'].default = None

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

    # 2) dataset ───────────────────────────────────────────────────────
    all_train_datasets = []
    all_val_datasets = []

    for task in TASKS:
        data_path = Path(DATA_ROOT) / task
        if not data_path.exists():
            print(f"[warn] task data not found: {data_path}, skipping")
            continue
        ds = VisionTaskDataset(data_path=str(data_path), resolution=RES, task=task)
        n_total = len(ds)
        need = N_PER_TASK + VAL_PER_TASK
        actual_n = min(need, n_total)
        actual_train = int(actual_n * N_PER_TASK / need)
        actual_val = actual_n - actual_train
        print(f"[data] {task}: {n_total} total → train={actual_train} val={actual_val}")

        train_idx = list(range(actual_train))
        val_idx = list(range(actual_train, actual_train + actual_val))
        all_train_datasets.append(torch.utils.data.Subset(ds, train_idx))
        all_val_datasets.append(torch.utils.data.Subset(ds, val_idx))

    if not all_train_datasets:
        raise RuntimeError(f"No task data found under {DATA_ROOT}")

    train_combined = torch.utils.data.ConcatDataset(all_train_datasets)
    val_combined = torch.utils.data.ConcatDataset(all_val_datasets)

    train_loader = torch.utils.data.DataLoader(
        train_combined, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_combined, batch_size=BATCH, shuffle=False, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    print(f"[data] combined train={len(train_combined)}  val={len(val_combined)}")

    # 3) LoRA on language_model ──────────────────────────────────────
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=RANK, lora_alpha=ALPHA, lora_dropout=LORA_DROPOUT,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model.language_model, lora_config)
    n_lora = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[lora] trainable = {n_lora:,}")

    # freeze the LLM (v5 does not use the LLM forward)
    peft_model.requires_grad_(False)

    # 4) unfreeze the end of the vision model ──────────────────────────
    _unfreeze_vision_blocks(model.vision_model, UNFREEZE_VISION_BLOCKS)

    # freeze the other parts
    model.aligner.requires_grad_(False)
    model.gen_vision_model.requires_grad_(False)
    model.gen_aligner.requires_grad_(False)
    model.gen_head.requires_grad_(False)
    model.gen_embed.requires_grad_(False)

    # 5) Conv RGB Decoder ────────────────────────────────────────────
    vision_dim = 1024
    decoder = ConvRGBDecoder(vision_dim, grid_size=24, target_size=RES, dropout=DROPOUT).to(device, DTYPE)
    n_decoder = sum(p.numel() for p in decoder.parameters())
    print(f"[decoder] ConvRGBDecoder params={n_decoder:,}")

    # 6) trainable params summary ──────────────────────────────────────
    trainable_params = [p for p in model.vision_model.parameters() if p.requires_grad] + \
                       list(decoder.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[total] trainable params = {n_trainable:,}")

    # 7) optimizer (grouped learning rates) ────────────────────────────
    vision_params = [p for p in model.vision_model.parameters() if p.requires_grad]
    decoder_params = list(decoder.parameters())

    opt = torch.optim.AdamW([
        {"params": decoder_params, "lr": LR},
        {"params": vision_params, "lr": VISION_LR},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 20)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 8) forward pass ──────────────────────────────────────────────────
    def _forward(image: torch.Tensor, task: str) -> torch.Tensor:
        """image: [1, 3, 384, 384] → pred_rgb: [1, 3, 384, 384]."""
        vis_features = model.vision_model(image)  # [1, 576, 1024]
        pred_rgb = decoder(vis_features)          # [1, 3, 384, 384]
        return pred_rgb

    # 9) validation ────────────────────────────────────────────────────
    @torch.no_grad()
    def run_validation() -> float:
        decoder.eval()
        model.vision_model.eval()
        val_losses = []
        for batch in val_loader:
            image = batch["image"][0:1].to(device, DTYPE)
            target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)
            pred_rgb = _forward(image, "depth")
            loss = F.mse_loss(pred_rgb, target_rgb)
            val_losses.append(loss.item())
        decoder.train()
        model.vision_model.train()
        return sum(val_losses) / len(val_losses) if val_losses else float("inf")

    # 10) training loop ────────────────────────────────────────────────
    print(f"\n[train] Janus continuous RGB v5: {STEPS} steps")
    print(f"[train] Conv2D Decoder + Vision fine-tuning (last {UNFREEZE_VISION_BLOCKS} blocks)")

    losses: list[float] = []
    val_history: list[dict] = []
    best_val_loss = float("inf")
    patience_counter = 0
    it = iter(train_loader)
    t_train = time.time()

    for step in range(1, STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        opt.zero_grad(set_to_none=True)

        image = batch["image"][0:1].to(device, DTYPE)
        target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)

        if AUGMENT_FLIP:
            image, target_rgb = _augment_flip(image, target_rgb)

        pred_rgb = _forward(image, "depth")
        loss = F.mse_loss(pred_rgb, target_rgb)

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)

        if step % VAL_EVERY == 0:
            val_loss = run_validation()
            record = {
                "step": step, "train_loss": float(loss.item()),
                "avg30": float(sum(losses[-30:]) / len(losses[-30:])),
                "val_loss": float(val_loss), "lr": float(sched.get_last_lr()[0]),
            }
            val_history.append(record)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_ckpt = OUT / "ckpt_best"
                best_ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(decoder.state_dict(), best_ckpt / "decoder.pt")
                # save the weights of the vision model's last layers
                vision_state = {}
                for name, p in model.vision_model.named_parameters():
                    if p.requires_grad:
                        vision_state[name] = p.data.cpu().clone()
                torch.save(vision_state, best_ckpt / "vision_tail.pt")
                config_snapshot = {
                    "model_id": MODEL_ID, "tasks": TASKS,
                    "n_per_task": N_PER_TASK, "steps": STEPS,
                    "lr": LR, "vision_lr": VISION_LR,
                    "rank": RANK, "alpha": ALPHA,
                    "dropout": DROPOUT, "lora_dropout": LORA_DROPOUT,
                    "weight_decay": WEIGHT_DECAY,
                    "unfreeze_blocks": UNFREEZE_VISION_BLOCKS,
                    "best_val_loss": best_val_loss, "best_step": step,
                }
                with open(best_ckpt / "train_config.json", "w") as f:
                    json.dump(config_snapshot, f, indent=2)
                marker = "  *BEST*"
            else:
                patience_counter += 1
                marker = ""
            print(f"[val] step {step:>4}  val_loss={val_loss:.4f}  "
                  f"best={best_val_loss:.4f}  patience={patience_counter}/{PATIENCE}{marker}",
                  flush=True)

            if patience_counter >= PATIENCE:
                print(f"[early] early stop at step {step} (best={best_val_loss:.4f})")
                break

    if not losses:
        raise RuntimeError("No finite training steps")

    # 11) training summary ─────────────────────────────────────────────
    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING' if avg_last < avg_first else 'NOT decreasing'})")

    final_ckpt = OUT / "ckpt_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), final_ckpt / "decoder.pt")
    print(f"[ckpt] saved → {final_ckpt}")

    history_out = OUT / "training_history.json"
    with open(history_out, "w") as f:
        json.dump({
            "config": {
                "model_id": MODEL_ID, "tasks": TASKS,
                "n_per_task": N_PER_TASK, "steps": STEPS,
                "res": RES, "lr": LR, "vision_lr": VISION_LR,
                "rank": RANK, "alpha": ALPHA,
                "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
                "unfreeze_blocks": UNFREEZE_VISION_BLOCKS,
            },
            "summary": {
                "total_steps": len(losses),
                "best_val_loss": best_val_loss,
                "avg_first_third": avg_first,
                "avg_last_third": avg_last,
                "stopped_early": patience_counter >= PATIENCE,
                "total_minutes": (time.time() - t_train) / 60,
            },
            "val_history": val_history,
        }, f, indent=2)
    print(f"[history] saved → {history_out}")

    # 12) inference evaluation ─────────────────────────────────────────
    print("\n[eval] inference on held-out samples...")
    decoder.eval()
    model.vision_model.eval()

    results = []
    for task_i, task in enumerate(TASKS):
        if task_i < len(all_val_datasets):
            for i in range(min(2, len(all_val_datasets[task_i]))):
                s = all_val_datasets[task_i][i]
                inp_tensor = s["image"]
                tgt_rgb = s["target_rgb"]
                pil_in = Image.fromarray(
                    (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
                    .mul(255).byte().numpy()
                )

                with torch.no_grad():
                    img = inp_tensor.unsqueeze(0).to(device, DTYPE)
                    pred_rgb = _forward(img, task).squeeze(0).clamp(0, 1).cpu()

                panel = Image.new("RGB", (RES * 3, RES))
                panel.paste(pil_in.resize((RES, RES)), (0, 0))
                panel.paste(Image.fromarray(
                    (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
                ).resize((RES, RES)), (RES, 0))
                panel.paste(Image.fromarray(
                    (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
                ).resize((RES, RES)), (RES * 2, 0))
                panel.save(OUT / f"eval_{task}_{i:02d}.png")

                rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
                results.append({"idx": i, "task": task, "rgb_mae": rgb_mae})
                print(f"[eval] task={task}  idx={i:>3}  rgb_mae={rgb_mae:.4f}")

    if results:
        avg_mae = sum(r["rgb_mae"] for r in results) / len(results)
        print(f"\n[gate] avg rgb_mae={avg_mae:.4f} across {len(results)} eval samples")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] Janus continuous RGB v5 complete ({total_min:.1f} min)")
    print(f"[done] best val_loss={best_val_loss:.4f}")
    print(f"[done] checkpoint → {OUT / 'ckpt_best'}")


if __name__ == "__main__":
    main()
