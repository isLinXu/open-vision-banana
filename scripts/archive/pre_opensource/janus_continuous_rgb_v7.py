#!/usr/bin/env python3
"""Janus-1.3B continuous RGB v7 — Two-stage: Depth Prediction + Colormap.

v3~v6 结论：直接预测 RGB 效果差，因为 SigLIP 提取语义特征而非几何特征
v7 方案：
  1. 不直接预测 RGB，而是预测 relative depth（单通道 [0,1]）
  2. 训练时：target_rgb → rgb_to_depth → normalized_depth → L1 loss
  3. 推理时：predicted_depth → depth_to_rgb (cube-edge colormap) → RGB
  4. 优势：深度估计比 RGB 回归简单得多，colormap 是确定性可逆的

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_continuous_rgb_v7.py
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
from open_vision_banana.core.depth_mapper import DepthRGBMapper

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
VISION_LR = float(os.environ.get("VB_VISION_LR", "2e-5"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))

DROPOUT = float(os.environ.get("VB_DROPOUT", "0.05"))
WEIGHT_DECAY = float(os.environ.get("VB_WEIGHT_DECAY", "0.01"))
PATIENCE = int(os.environ.get("VB_PATIENCE", "20"))
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "50"))
GRAD_CLIP = float(os.environ.get("VB_GRAD_CLIP", "1.0"))
AUGMENT_FLIP = os.environ.get("VB_AUGMENT_FLIP", "1") == "1"

# v7: 解冻 vision model 最后 N 层（比 v5 的 4 层多，比 v6 的 24 层少）
UNFREEZE_VISION_BLOCKS = int(os.environ.get("VB_UNFREEZE_BLOCKS", "8"))

OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_continuous_rgb_v7")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32
IMAGE_TOKEN_NUM = 576

TASK_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
}


class DepthDecoder(nn.Module):
    """轻量级 depth decoder：SigLIP features → 单通道 normalized depth.

    架构：
      stem:     [B, 1024, 24, 24] → [B, 256, 24, 24]
      up1+PS:   → [B, 128, 48, 48]
      up2+PS:   → [B, 64, 96, 96]
      up3+PS:   → [B, 32, 192, 192]
      up4+PS:   → [B, 16, 384, 384]
      head:     → [B, 1, 384, 384]  (sigmoid → [0, 1])
    """

    def __init__(self, feat_dim: int = 1024, grid_size: int = 24,
                 target_size: int = 384, dropout: float = 0.05):
        super().__init__()
        self.grid_size = grid_size
        self.target_size = target_size

        self.stem = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128 * 4, 3, padding=1),
            nn.BatchNorm2d(128 * 4), nn.GELU(),
            nn.PixelShuffle(2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, 3, padding=1),
            nn.BatchNorm2d(64 * 4), nn.GELU(),
            nn.PixelShuffle(2),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32 * 4, 3, padding=1),
            nn.BatchNorm2d(32 * 4), nn.GELU(),
            nn.PixelShuffle(2),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(32, 16 * 4, 3, padding=1),
            nn.BatchNorm2d(16 * 4), nn.GELU(),
            nn.PixelShuffle(2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8), nn.GELU(),
            nn.Conv2d(8, 1, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, 576, D] → normalized depth [B, 1, 384, 384] in [0, 1]."""
        b = features.shape[0]
        x = features.transpose(1, 2).view(b, -1, self.grid_size, self.grid_size)
        x = self.stem(x)     # [B, 256, 24, 24]
        x = self.up1(x)      # [B, 128, 48, 48]
        x = self.up2(x)      # [B, 64, 96, 96]
        x = self.up3(x)      # [B, 32, 192, 192]
        x = self.up4(x)      # [B, 16, 384, 384]
        x = self.head(x)     # [B, 1, 384, 384]
        return torch.sigmoid(x)


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


def _unfreeze_vision_blocks(vision_model, n_blocks: int):
    """解冻 vision model 最后 N 个 transformer block."""
    vision_model.requires_grad_(False)
    if hasattr(vision_model, "vision_tower"):
        tower = vision_model.vision_tower
    else:
        tower = vision_model
    if hasattr(tower, "blocks"):
        blocks = tower.blocks
        total = len(blocks)
        start = max(0, total - n_blocks)
        for i in range(start, total):
            for p in blocks[i].parameters():
                p.requires_grad_(True)
        print(f"[vision] unfrozen blocks {start}-{total-1} (last {n_blocks} of {total})")
    else:
        print(f"[vision] WARN: could not find blocks in {type(tower).__name__}")


def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={DTYPE}")
    print(f"[cfg] tasks={TASKS}  n_per_task={N_PER_TASK}  val_per_task={VAL_PER_TASK}")
    print(f"[cfg] steps={STEPS}  res={RES}  batch={BATCH}  lr={LR}  vision_lr={VISION_LR}")
    print(f"[cfg] dropout={DROPOUT}  wd={WEIGHT_DECAY}  patience={PATIENCE}")
    print(f"[cfg] augment_flip={AUGMENT_FLIP}  unfreeze_blocks={UNFREEZE_VISION_BLOCKS}")
    print(f"[cfg] out={OUT}")
    print(f"[cfg] *** v7: predict depth + colormap (two-stage) ***")

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

    # 2) Depth decoder ──────────────────────────────────────────────
    decoder = DepthDecoder(1024, grid_size=24, target_size=RES, dropout=DROPOUT).to(device, DTYPE)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"[decoder] DepthDecoder, params={n_params:,}")

    # 3) 解冻 vision blocks ─────────────────────────────────────────
    _unfreeze_vision_blocks(model.vision_model, UNFREEZE_VISION_BLOCKS)

    # 4) LoRA on language_model（保持与 v3~v6 一致）──────────────────
    from peft import LoraConfig, get_peft_model
    lora_cfg = LoraConfig(
        r=RANK, lora_alpha=ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type=None,
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)
    model.language_model.requires_grad_(False)
    for name, param in model.language_model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    n_lora = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
    print(f"[lora] rank={RANK}  alpha={ALPHA}  trainable={n_lora:,}")

    # 5) 数据集 ─────────────────────────────────────────────────────
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

    # 6) Depth mapper（用于 RGB ↔ depth 转换）──────────────────────
    depth_mapper = DepthRGBMapper()  # cube_edge colormap
    # 将 mapper 的 keypoints 移到 GPU 用于批量转换
    _keypoints = depth_mapper.colormap_keypoints.to(device)

    def target_rgb_to_normalized_depth(rgb_tensor: torch.Tensor) -> torch.Tensor:
        """target_rgb [B, 3, H, W] in [0,1] → normalized depth [B, 1, H, W] in [0,1]."""
        return depth_mapper.rgb_to_normalized(rgb_tensor)

    def predicted_depth_to_rgb(depth_tensor: torch.Tensor) -> torch.Tensor:
        """predicted normalized depth [B, 1, H, W] in [0,1] → RGB [B, 3, H, W] in [0,1]."""
        return depth_mapper.normalized_to_rgb(depth_tensor)

    # 7) 优化器 ─────────────────────────────────────────────────────
    trainable_params = [
        {"params": [p for p in decoder.parameters() if p.requires_grad], "lr": LR},
        {"params": [p for p in model.vision_model.parameters() if p.requires_grad], "lr": VISION_LR},
        {"params": [p for p in model.language_model.parameters() if p.requires_grad], "lr": LR},
    ]
    n_trainable = sum(sum(p.numel() for p in g["params"]) for g in trainable_params)
    print(f"[optim] total trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=LR * 0.01)

    # 8) 训练循环 ───────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    running_loss = []
    t_start = time.time()

    print(f"\n[train] starting training for {STEPS} steps...")
    while global_step < STEPS:
        for batch in train_loader:
            if global_step >= STEPS:
                break

            image = batch["image"].to(device, DTYPE)
            target_rgb = batch["target_rgb"].to(device, DTYPE)

            if AUGMENT_FLIP:
                image, target_rgb = _augment_flip(image, target_rgb)

            # ★ v7 核心：将 target RGB 转换为 normalized depth
            with torch.no_grad():
                target_depth = target_rgb_to_normalized_depth(target_rgb)  # [B, 1, H, W]

            # Forward: SigLIP → depth decoder
            vis_features = model.vision_model(image)  # [1, 576, 1024]
            pred_depth = decoder(vis_features)         # [1, 1, 384, 384]

            # Loss: L1 in depth space
            loss = F.l1_loss(pred_depth, target_depth)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(decoder.parameters()) +
                [p for p in model.vision_model.parameters() if p.requires_grad] +
                [p for p in model.language_model.parameters() if p.requires_grad],
                GRAD_CLIP,
            )
            optimizer.step()
            scheduler.step()

            running_loss.append(loss.item())
            if len(running_loss) > 30:
                running_loss.pop(0)
            avg_loss = sum(running_loss) / len(running_loss)

            global_step += 1
            elapsed = time.time() - t_start
            speed = elapsed / global_step
            eta = speed * (STEPS - global_step)

            if global_step % 30 == 0 or global_step == 1:
                print(f"[train] step {global_step:>5}/{STEPS}  loss={loss.item():.4f}  "
                      f"avg30={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"({speed:.2f}s/step  eta={eta/60:.0f}min)")

            # Validation
            if global_step % VAL_EVERY == 0:
                decoder.eval()
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_img = val_batch["image"].to(device, DTYPE)
                        val_tgt_rgb = val_batch["target_rgb"].to(device, DTYPE)
                        val_tgt_depth = target_rgb_to_normalized_depth(val_tgt_rgb)

                        val_feat = model.vision_model(val_img)
                        val_pred = decoder(val_feat)
                        val_loss = F.l1_loss(val_pred, val_tgt_depth)
                        val_losses.append(val_loss.item())

                val_loss = sum(val_losses) / len(val_losses)

                # 也计算 RGB-space loss 用于对比
                val_pred_rgb = predicted_depth_to_rgb(pred_depth)
                val_rgb_mae = F.l1_loss(val_pred_rgb, target_rgb).item()

                status = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    status = " *BEST*"
                    # 保存 checkpoint
                    ckpt_dir = OUT / "ckpt_best"
                    ckpt_dir.mkdir(exist_ok=True)
                    torch.save(decoder.state_dict(), ckpt_dir / "decoder.pt")
                    # 保存 vision tail
                    vision_state = {}
                    for name, param in model.vision_model.named_parameters():
                        if param.requires_grad:
                            vision_state[name] = param.data.cpu().clone()
                    torch.save(vision_state, ckpt_dir / "vision_tail.pt")
                    with open(ckpt_dir / "train_config.json", "w") as f:
                        json.dump({"val_loss": val_loss, "step": global_step,
                                   "version": "v7", "approach": "depth+colormap"}, f, indent=2)
                else:
                    patience_counter += 1

                print(f"[val]   step {global_step}  val_loss={val_loss:.4f}  "
                      f"best={best_val_loss:.4f}  patience={patience_counter}/{PATIENCE}  "
                      f"rgb_mae={val_rgb_mae:.4f}{status}")

                decoder.train()
                model.train()

                if patience_counter >= PATIENCE:
                    print(f"[early stop] patience={PATIENCE} reached at step {global_step}")
                    break

    # 9) 最终评估 ───────────────────────────────────────────────────
    print(f"\n[eval] running final evaluation on val set...")
    decoder.eval()
    model.eval()

    # 加载 best checkpoint
    ckpt_path = OUT / "ckpt_best" / "decoder.pt"
    if ckpt_path.exists():
        decoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"[eval] loaded best decoder from step with val_loss={best_val_loss:.4f}")

    # 加载 vision tail
    vision_tail_path = OUT / "ckpt_best" / "vision_tail.pt"
    if vision_tail_path.exists():
        vision_state = torch.load(vision_tail_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.vision_model.load_state_dict(vision_state, strict=False)
        print(f"[eval] loaded vision tail, missing={len(missing)}, unexpected={len(unexpected)}")

    all_metrics = []
    eval_dir = OUT / "eval"
    eval_dir.mkdir(exist_ok=True)

    ds = VisionTaskDataset(data_path=str(Path(DATA_ROOT) / "depth"), resolution=RES, task="depth")
    n_train = 500
    n_eval = min(10, len(ds) - n_train)

    for idx_i in range(n_eval):
        idx = n_train + idx_i
        sample = ds[idx]
        inp_tensor = sample["image"]
        tgt_rgb = sample["target_rgb"]

        with torch.no_grad():
            pixel_values = inp_tensor.unsqueeze(0).to(device, DTYPE)
            vis_features = model.vision_model(pixel_values)
            pred_depth = decoder(vis_features)  # [1, 1, 384, 384]
            pred_rgb = predicted_depth_to_rgb(pred_depth).squeeze(0).clamp(0, 1).cpu()

        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        rgb_mse = ((pred_rgb.float() - tgt_rgb.float()) ** 2).mean().item()
        psnr = 10 * math.log10(1.0 / max(rgb_mse, 1e-10))

        pred_mean = pred_rgb.mean().item()
        pred_std = pred_rgb.std().item()
        tgt_mean = tgt_rgb.mean().item()
        tgt_std = tgt_rgb.std().item()

        # Depth-space metrics
        pred_d = pred_depth.squeeze().cpu()
        tgt_d = target_rgb_to_normalized_depth(tgt_rgb.unsqueeze(0)).squeeze().cpu()
        depth_mae = (pred_d.float() - tgt_d.float()).abs().mean().item()

        metric = {
            "idx": idx, "rgb_mae": rgb_mae, "psnr": psnr,
            "depth_mae": depth_mae,
            "pred_mean": pred_mean, "pred_std": pred_std,
            "tgt_mean": tgt_mean, "tgt_std": tgt_std,
        }
        all_metrics.append(metric)
        print(f"  [{idx_i+1}/{n_eval}] idx={idx}  rgb_mae={rgb_mae:.4f}  "
              f"depth_mae={depth_mae:.4f}  psnr={psnr:.1f}dB  "
              f"pred_std={pred_std:.4f}  tgt_std={tgt_std:.4f}")

        # 保存面板图: Input | Pred | Target
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy()
        )
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(eval_dir / f"eval_depth_{idx_i:02d}.png")

    # 汇总
    avg_mae = sum(m["rgb_mae"] for m in all_metrics) / len(all_metrics)
    avg_depth_mae = sum(m["depth_mae"] for m in all_metrics) / len(all_metrics)
    avg_psnr = sum(m["psnr"] for m in all_metrics) / len(all_metrics)
    avg_pred_std = sum(m["pred_std"] for m in all_metrics) / len(all_metrics)
    avg_tgt_std = sum(m["tgt_std"] for m in all_metrics) / len(all_metrics)

    print(f"\n{'='*60}")
    print(f"V7 DIAGNOSTIC SUMMARY (depth + colormap)")
    print(f"{'='*60}")
    print(f"avg rgb_mae   = {avg_mae:.4f}")
    print(f"avg depth_mae = {avg_depth_mae:.4f}")
    print(f"avg psnr      = {avg_psnr:.1f} dB")
    print(f"avg pred_std  = {avg_pred_std:.4f}")
    print(f"avg tgt_std   = {avg_tgt_std:.4f}")

    with open(eval_dir / "diag_summary.json", "w") as f:
        json.dump({
            "metrics": all_metrics,
            "summary": {
                "avg_mae": avg_mae, "avg_depth_mae": avg_depth_mae,
                "avg_psnr": avg_psnr, "avg_pred_std": avg_pred_std,
                "avg_tgt_std": avg_tgt_std,
            }
        }, f, indent=2)

    total_time = (time.time() - t_start) / 60
    print(f"\n[done] Janus continuous RGB v7 complete ({total_time:.1f} min)")
    print(f"[done] best val_loss={best_val_loss:.4f}")
    print(f"[done] checkpoint → {OUT / 'ckpt_best'}")


if __name__ == "__main__":
    main()
