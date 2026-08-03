#!/usr/bin/env python3
"""Janus-1.3B continuous RGB v6 — U-Net Skip + Full Vision Unfreeze.

v5 结论：Conv2D Decoder 比 MLP 好（val_loss 0.0282 vs 0.0305），但 MAE 仍不如 v3
v6 改进：
  1. U-Net 风格 skip connection：输入图像像素直接注入 decoder 每个分辨率层
  2. 全解冻 SigLIP（24 blocks）+ 低学习率
  3. 输入图像经 learnable stem 在 bottleneck 与 SigLIP features 融合

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_continuous_rgb_v6.py
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
DATA_ROOT = os.environ.get("VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3")

TASKS = os.environ.get("VB_TASKS", "depth").split(",")
N_PER_TASK = int(os.environ.get("VB_N_PER_TASK", "500"))
VAL_PER_TASK = int(os.environ.get("VB_VAL_PER_TASK", "50"))

STEPS = int(os.environ.get("VB_STEPS", "2000"))
RES = int(os.environ.get("VB_RES", "384"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "5e-5"))
VISION_LR = float(os.environ.get("VB_VISION_LR", "1e-5"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))

DROPOUT = float(os.environ.get("VB_DROPOUT", "0.1"))
WEIGHT_DECAY = float(os.environ.get("VB_WEIGHT_DECAY", "0.01"))
PATIENCE = int(os.environ.get("VB_PATIENCE", "20"))
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "50"))
GRAD_CLIP = float(os.environ.get("VB_GRAD_CLIP", "1.0"))
AUGMENT_FLIP = os.environ.get("VB_AUGMENT_FLIP", "1") == "1"

OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_continuous_rgb_v6")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32
IMAGE_TOKEN_NUM = 576

TASK_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
}


class UNetRGBDecoder(nn.Module):
    """U-Net 风格 decoder：SigLIP features + 输入图像 skip connections → RGB.

    架构：
      bottleneck: [B, 1024+64, 24, 24]  (SigLIP + image stem)
      up1 + skip: [B, 256+3, 48, 48]    → conv → 256
      up2 + skip: [B, 128+3, 96, 96]    → conv → 128
      up3 + skip: [B, 64+3, 192, 192]   → conv → 64
      up4 + skip: [B, 32+3, 384, 384]   → conv → 32
      head:       [B, 3, 384, 384]
    """

    def __init__(self, feat_dim: int = 1024, grid_size: int = 24,
                 target_size: int = 384, dropout: float = 0.1):
        super().__init__()
        self.grid_size = grid_size
        self.target_size = target_size

        # 输入图像 stem：384×384 RGB → 24×24×64（与 SigLIP 对齐）
        self.img_stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),   # → 192×192
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),   # → 96×96
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 5, stride=2, padding=2),   # → 48×48
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),   # → 24×24
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Dropout2d(dropout),
        )

        # Bottleneck: SigLIP(1024) + img_stem(64) = 1088 → 512
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feat_dim + 64, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.GELU(),
            nn.Dropout2d(dropout),
        )

        # Decoder blocks: up + skip fusion
        # up1: 512 → 256×4=1024, PixelShuffle(2) → 256 at 48×48, +3 skip → 259 → 256
        self.up1 = nn.Sequential(
            nn.Conv2d(512, 256 * 4, 3, padding=1), nn.BatchNorm2d(256 * 4),
            nn.GELU(), nn.PixelShuffle(2),
        )
        self.fuse1 = self._make_fuse(256 + 3, 256)

        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, 3, padding=1), nn.BatchNorm2d(128 * 4),
            nn.GELU(), nn.PixelShuffle(2),
        )
        self.fuse2 = self._make_fuse(128 + 3, 128)

        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1), nn.BatchNorm2d(64 * 4),
            nn.GELU(), nn.PixelShuffle(2),
        )
        self.fuse3 = self._make_fuse(64 + 3, 64)

        self.up4 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, 3, padding=1), nn.BatchNorm2d(32 * 4),
            nn.GELU(), nn.PixelShuffle(2),
        )
        self.fuse4 = self._make_fuse(32 + 3, 32)

        # Final head
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 3, 1),
        )

        self._init_weights()

    @staticmethod
    def _make_fuse(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.GELU(),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
        """
        features:    [B, 576, D]  SigLIP output
        input_image: [B, 3, 384, 384]  归一化后的输入图像
        → [B, 3, 384, 384]
        """
        b = features.shape[0]

        # Reshape SigLIP features → 2D
        vis = features.transpose(1, 2).view(b, -1, self.grid_size, self.grid_size)

        # Image stem → 24×24
        img_feat = self.img_stem(input_image)  # [B, 64, 24, 24]

        # Bottleneck
        x = torch.cat([vis, img_feat], dim=1)  # [B, 1088, 24, 24]
        x = self.bottleneck(x)                  # [B, 512, 24, 24]

        # Prepare skip connections (downsampled input image)
        sk1 = F.avg_pool2d(input_image, 8)   # [B, 3, 48, 48]
        sk2 = F.avg_pool2d(input_image, 4)   # [B, 3, 96, 96]
        sk3 = F.avg_pool2d(input_image, 2)   # [B, 3, 192, 192]
        sk4 = input_image                     # [B, 3, 384, 384]

        # Decoder with skip connections
        x = self.up1(x)                     # [B, 256, 48, 48]
        x = self.fuse1(torch.cat([x, sk1], dim=1))  # [B, 256, 48, 48]

        x = self.up2(x)                     # [B, 128, 96, 96]
        x = self.fuse2(torch.cat([x, sk2], dim=1))  # [B, 128, 96, 96]

        x = self.up3(x)                     # [B, 64, 192, 192]
        x = self.fuse3(torch.cat([x, sk3], dim=1))  # [B, 64, 192, 192]

        x = self.up4(x)                     # [B, 32, 384, 384]
        x = self.fuse4(torch.cat([x, sk4], dim=1))  # [B, 32, 384, 384]

        x = self.head(x)                      # [B, 3, 384, 384]
        x = torch.sigmoid(x)
        return x


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _augment_flip(image: torch.Tensor, target_rgb: torch.Tensor):
    if torch.rand(1).item() < 0.5:
        return image.flip(-1), target_rgb.flip(-1)
    return image, target_rgb


def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={DTYPE}")
    print(f"[cfg] tasks={TASKS}  n_per_task={N_PER_TASK}  val_per_task={VAL_PER_TASK}")
    print(f"[cfg] steps={STEPS}  res={RES}  batch={BATCH}  lr={LR}  vision_lr={VISION_LR}")
    print(f"[cfg] dropout={DROPOUT}  wd={WEIGHT_DECAY}  patience={PATIENCE}")
    print(f"[cfg] augment_flip={AUGMENT_FLIP}")
    print(f"[cfg] out={OUT}")

    # 1) 加载 Janus ──────────────────────────────────────────────────
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

    # 2) 数据集 ──────────────────────────────────────────────────────
    all_train_datasets, all_val_datasets = [], []
    for task in TASKS:
        data_path = Path(DATA_ROOT) / task
        if not data_path.exists():
            continue
        ds = VisionTaskDataset(data_path=str(data_path), resolution=RES, task=task)
        n_total = len(ds)
        need = N_PER_TASK + VAL_PER_TASK
        actual_n = min(need, n_total)
        actual_train = int(actual_n * N_PER_TASK / need)
        actual_val = actual_n - actual_train
        print(f"[data] {task}: {n_total} total → train={actual_train} val={actual_val}")
        all_train_datasets.append(torch.utils.data.Subset(ds, list(range(actual_train))))
        all_val_datasets.append(torch.utils.data.Subset(ds, list(range(actual_train, actual_train + actual_val))))

    train_combined = torch.utils.data.ConcatDataset(all_train_datasets)
    val_combined = torch.utils.data.ConcatDataset(all_val_datasets)
    train_loader = torch.utils.data.DataLoader(
        train_combined, batch_size=BATCH, shuffle=True, num_workers=0,
        collate_fn=_collate_fn, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_combined, batch_size=BATCH, shuffle=False, num_workers=0,
        collate_fn=_collate_fn, drop_last=True)
    print(f"[data] combined train={len(train_combined)}  val={len(val_combined)}")

    # 3) 全解冻 vision model ─────────────────────────────────────────
    model.vision_model.requires_grad_(True)
    n_vision = sum(p.numel() for p in model.vision_model.parameters() if p.requires_grad)
    print(f"[vision] ALL unfrozen, params={n_vision:,}")

    # 冻结其他部分
    model.language_model.requires_grad_(False)
    model.aligner.requires_grad_(False)
    model.gen_vision_model.requires_grad_(False)
    model.gen_aligner.requires_grad_(False)
    model.gen_head.requires_grad_(False)
    model.gen_embed.requires_grad_(False)

    # 4) U-Net RGB Decoder ───────────────────────────────────────────
    decoder = UNetRGBDecoder(1024, grid_size=24, target_size=RES, dropout=DROPOUT).to(device, DTYPE)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"[decoder] UNetRGBDecoder params={n_dec:,}")

    # 5) Optimizer ───────────────────────────────────────────────────
    vision_params = [p for p in model.vision_model.parameters() if p.requires_grad]
    decoder_params = list(decoder.parameters())
    n_trainable = sum(p.numel() for p in decoder_params) + sum(p.numel() for p in vision_params)
    print(f"[total] trainable params = {n_trainable:,}")

    opt = torch.optim.AdamW([
        {"params": decoder_params, "lr": LR},
        {"params": vision_params, "lr": VISION_LR},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        warmup = max(1, STEPS // 20)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 6) 前向传播 ────────────────────────────────────────────────────
    def _forward(image_normalized: torch.Tensor) -> torch.Tensor:
        """image_normalized: [1, 3, 384, 384] (归一化到 [-1,1]).
        返回 pred_rgb: [1, 3, 384, 384].
        """
        # 反归一化到 [0,1] 给 decoder 做 skip
        image_01 = image_normalized * 0.5 + 0.5
        vis_features = model.vision_model(image_normalized)  # [1, 576, 1024]
        pred_rgb = decoder(vis_features, image_01)           # [1, 3, 384, 384]
        return pred_rgb

    # 7) 验证 ────────────────────────────────────────────────────────
    @torch.no_grad()
    def run_validation() -> float:
        decoder.eval()
        model.vision_model.eval()
        val_losses = []
        for batch in val_loader:
            image = batch["image"][0:1].to(device, DTYPE)
            target_rgb = batch["target_rgb"][0:1].to(device, DTYPE)
            pred_rgb = _forward(image)
            loss = F.mse_loss(pred_rgb, target_rgb)
            val_losses.append(loss.item())
        decoder.train()
        model.vision_model.train()
        return sum(val_losses) / len(val_losses) if val_losses else float("inf")

    # 8) 训练循环 ────────────────────────────────────────────────────
    print(f"\n[train] Janus continuous RGB v6: {STEPS} steps")
    print(f"[train] U-Net skip + full vision unfreeze")

    losses, val_history = [], []
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

        pred_rgb = _forward(image)
        loss = F.mse_loss(pred_rgb, target_rgb)

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.vision_model.parameters() if p.requires_grad] + decoder_params,
            GRAD_CLIP)
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
            val_history.append({
                "step": step, "train_loss": float(loss.item()),
                "avg30": float(sum(losses[-30:]) / len(losses[-30:])),
                "val_loss": float(val_loss), "lr": float(sched.get_last_lr()[0]),
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_ckpt = OUT / "ckpt_best"
                best_ckpt.mkdir(parents=True, exist_ok=True)
                torch.save(decoder.state_dict(), best_ckpt / "decoder.pt")
                vision_state = {n: p.data.cpu().clone()
                                for n, p in model.vision_model.named_parameters()
                                if p.requires_grad}
                torch.save(vision_state, best_ckpt / "vision_tail.pt")
                with open(best_ckpt / "train_config.json", "w") as f:
                    json.dump({
                        "model_id": MODEL_ID, "tasks": TASKS,
                        "n_per_task": N_PER_TASK, "steps": STEPS,
                        "lr": LR, "vision_lr": VISION_LR,
                        "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY,
                        "best_val_loss": best_val_loss, "best_step": step,
                    }, f, indent=2)
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

    # 9) 保存 ────────────────────────────────────────────────────────
    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}")

    final_ckpt = OUT / "ckpt_final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), final_ckpt / "decoder.pt")

    with open(OUT / "training_history.json", "w") as f:
        json.dump({
            "config": {"model_id": MODEL_ID, "lr": LR, "vision_lr": VISION_LR,
                        "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY},
            "summary": {
                "total_steps": len(losses), "best_val_loss": best_val_loss,
                "avg_first_third": avg_first, "avg_last_third": avg_last,
                "stopped_early": patience_counter >= PATIENCE,
                "total_minutes": (time.time() - t_train) / 60,
            },
            "val_history": val_history,
        }, f, indent=2)

    # 10) 快速 eval ──────────────────────────────────────────────────
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
                    .mul(255).byte().numpy())

                with torch.no_grad():
                    img = inp_tensor.unsqueeze(0).to(device, DTYPE)
                    pred_rgb = _forward(img).squeeze(0).clamp(0, 1).cpu()

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
        print(f"\n[gate] avg rgb_mae={avg_mae:.4f}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] Janus continuous RGB v6 complete ({total_min:.1f} min)")
    print(f"[done] best val_loss={best_val_loss:.4f}")
    print(f"[done] checkpoint → {OUT / 'ckpt_best'}")


if __name__ == "__main__":
    main()
