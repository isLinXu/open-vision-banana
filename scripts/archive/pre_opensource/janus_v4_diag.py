#!/usr/bin/env python3
"""Janus v4 checkpoint 诊断评估 — 多样本 + 统计分析."""
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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, "/tmp/Janus")

from open_vision_banana.training.tuner import VisionTaskDataset

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = "deepseek-ai/Janus-1.3B"
DATA_ROOT = "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3"
CKPT = str(PROJECT_ROOT / "output" / "janus_continuous_rgb_v4" / "ckpt_best")
RES = 384
IMAGE_TOKEN_NUM = 576
DTYPE = torch.float32
N_EVAL = 10


class RGBHead(nn.Module):
    """与 v4 训练一致：手动归一化 + sigmoid."""

    def __init__(self, input_dim, image_token_num, image_size, dropout=0.0):
        super().__init__()
        self.image_token_num = image_token_num
        self.image_size = image_size
        self.patch_size = image_size // int(math.sqrt(image_token_num))
        out_dim = self.patch_size * self.patch_size * 3
        mid_dim = min(input_dim, 512)

        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))

        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, hidden_states):
        b = hidden_states.shape[0]
        mean = self.running_mean
        var = self.running_var
        x = (hidden_states - mean) / (var.sqrt() + 1e-6)
        x = self.proj(x)
        x = torch.sigmoid(x)
        x = x.view(b, self.image_token_num, self.patch_size, self.patch_size, 3)
        x = x.permute(0, 4, 1, 2, 3)
        h = w = int(math.sqrt(self.image_token_num))
        x = x.reshape(b, 3, h * self.patch_size, w * self.patch_size)
        return x


def _resolve_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    torch.manual_seed(42)
    device = _resolve_device()
    print(f"[cfg] device={device}  ckpt={CKPT}  n_eval={N_EVAL}")

    # 1) 加载模型
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

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    model.eval()

    # 2) 加载 LoRA
    from peft import PeftModel
    adapter_dir = Path(CKPT) / "lora"
    if adapter_dir.exists():
        model.language_model = PeftModel.from_pretrained(model.language_model, str(adapter_dir))
        model.language_model.eval()
        print(f"[lora] loaded from {adapter_dir}")

    # 3) 加载 RGB Head（v4: input_dim=1024, vision features）
    vision_dim = 1024
    rgb_head = RGBHead(vision_dim, IMAGE_TOKEN_NUM, RES, dropout=0.0).to(device, DTYPE)
    rgb_head_path = Path(CKPT) / "rgb_head.pt"
    rgb_head.load_state_dict(torch.load(rgb_head_path, map_location=device, weights_only=True))
    rgb_head.eval()
    print(f"[rgb_head] loaded, input_dim={vision_dim}, params={sum(p.numel() for p in rgb_head.parameters()):,}")

    # 4) 加载验证集
    ds = VisionTaskDataset(data_path=str(Path(DATA_ROOT) / "depth"), resolution=RES, task="depth")
    n_total = len(ds)
    n_train = 500
    val_indices = list(range(n_train, min(n_train + N_EVAL, n_total)))
    print(f"[data] total={n_total}, eval indices={val_indices[0]}-{val_indices[-1]}")

    prompt = "Generate a metric depth visualization of this image."
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    p_len = p_ids.shape[1]

    # 5) 推理 + 诊断
    all_metrics = []
    OUT = PROJECT_ROOT / "output" / "janus_continuous_rgb_v4" / "diag"
    OUT.mkdir(parents=True, exist_ok=True)

    for idx_i, idx in enumerate(val_indices):
        sample = ds[idx]
        inp_tensor = sample["image"]
        tgt_rgb = sample["target_rgb"]

        with torch.no_grad():
            # v4: 使用 vision features
            pixel_values = inp_tensor.unsqueeze(0).to(device, DTYPE)  # [1, 3, 384, 384]
            vis_features = model.vision_model(pixel_values)  # [1, 576, 1024]
            pred_rgb = rgb_head(vis_features).squeeze(0).clamp(0, 1).cpu()

        # 指标
        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        rgb_mse = ((pred_rgb.float() - tgt_rgb.float()) ** 2).mean().item()
        psnr = 10 * math.log10(1.0 / max(rgb_mse, 1e-10))

        per_ch_mae = []
        for c in range(3):
            ch_mae = (pred_rgb[c].float() - tgt_rgb[c].float()).abs().mean().item()
            per_ch_mae.append(ch_mae)

        pred_mean = pred_rgb.mean().item()
        pred_std = pred_rgb.std().item()
        pred_min = pred_rgb.min().item()
        pred_max = pred_rgb.max().item()
        tgt_mean = tgt_rgb.mean().item()
        tgt_std = tgt_rgb.std().item()

        h_half, w_half = pred_rgb.shape[1] // 2, pred_rgb.shape[2] // 2
        quadrants_pred = [
            pred_rgb[:, :h_half, :w_half].mean().item(),
            pred_rgb[:, :h_half, w_half:].mean().item(),
            pred_rgb[:, h_half:, :w_half].mean().item(),
            pred_rgb[:, h_half:, w_half:].mean().item(),
        ]
        quadrants_tgt = [
            tgt_rgb[:, :h_half, :w_half].mean().item(),
            tgt_rgb[:, :h_half, w_half:].mean().item(),
            tgt_rgb[:, h_half:, :w_half].mean().item(),
            tgt_rgb[:, h_half:, w_half:].mean().item(),
        ]
        quadrant_diff = [abs(p - t) for p, t in zip(quadrants_pred, quadrants_tgt)]

        metric = {
            "idx": idx, "rgb_mae": rgb_mae, "rgb_mse": rgb_mse, "psnr": psnr,
            "per_ch_mae": per_ch_mae,
            "pred_mean": pred_mean, "pred_std": pred_std,
            "pred_range": [pred_min, pred_max],
            "tgt_mean": tgt_mean, "tgt_std": tgt_std,
            "quadrant_pred": quadrants_pred, "quadrant_tgt": quadrants_tgt,
            "quadrant_err": quadrant_diff,
        }
        all_metrics.append(metric)

        print(f"\n[eval] sample {idx_i+1}/{len(val_indices)} (idx={idx})")
        print(f"  rgb_mae={rgb_mae:.4f}  mse={rgb_mse:.4f}  psnr={psnr:.1f}dB")
        print(f"  per_ch_mae=[R={per_ch_mae[0]:.4f} G={per_ch_mae[1]:.4f} B={per_ch_mae[2]:.4f}]")
        print(f"  pred: mean={pred_mean:.4f} std={pred_std:.4f} range=[{pred_min:.4f}, {pred_max:.4f}]")
        print(f"  tgt:  mean={tgt_mean:.4f} std={tgt_std:.4f}")
        print(f"  quadrant_err={[f'{e:.4f}' for e in quadrant_diff]}")

        # 保存面板图
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
        panel.save(OUT / f"diag_depth_{idx:04d}.png")

        diff = (pred_rgb.float() - tgt_rgb.float()).abs()
        diff_vis = (diff * 5.0).clamp(0, 1)
        Image.fromarray(
            (diff_vis.permute(1, 2, 0).mul(255).byte().numpy())
        ).resize((RES, RES)).save(OUT / f"diag_depth_{idx:04d}_diff.png")

    # 6) 汇总
    print("\n" + "=" * 60)
    print("V4 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    avg_mae = sum(m["rgb_mae"] for m in all_metrics) / len(all_metrics)
    avg_psnr = sum(m["psnr"] for m in all_metrics) / len(all_metrics)
    avg_pred_mean = sum(m["pred_mean"] for m in all_metrics) / len(all_metrics)
    avg_pred_std = sum(m["pred_std"] for m in all_metrics) / len(all_metrics)
    avg_tgt_mean = sum(m["tgt_mean"] for m in all_metrics) / len(all_metrics)
    avg_tgt_std = sum(m["tgt_std"] for m in all_metrics) / len(all_metrics)

    print(f"avg rgb_mae  = {avg_mae:.4f}")
    print(f"avg psnr     = {avg_psnr:.1f} dB")
    print(f"avg pred: mean={avg_pred_mean:.4f}  std={avg_pred_std:.4f}")
    print(f"avg tgt:  mean={avg_tgt_mean:.4f}  std={avg_tgt_std:.4f}")

    pred_means = [m["pred_mean"] for m in all_metrics]
    pred_stds = [m["pred_std"] for m in all_metrics]
    print(f"\npred_mean across samples: min={min(pred_means):.4f} max={max(pred_means):.4f}")
    print(f"pred_std across samples:  min={min(pred_stds):.4f} max={max(pred_stds):.4f}")

    if avg_pred_std < 0.05:
        print("\n[WARN] 预测几乎退化为常量！")
    if max(pred_means) - min(pred_means) < 0.05:
        print("\n[WARN] 不同样本的预测均值几乎相同！")

    # v3 vs v4 对比
    print("\n" + "=" * 60)
    print("V3 vs V4 COMPARISON")
    print("=" * 60)
    v3_diag = PROJECT_ROOT / "output" / "janus_continuous_rgb_v3" / "diag" / "diag_summary.json"
    if v3_diag.exists():
        with open(v3_diag) as f:
            v3 = json.load(f)
        v3s = v3["summary"]
        print(f"{'metric':<20} {'v3':>10} {'v4':>10} {'delta':>10}")
        print(f"{'avg_mae':<20} {v3s['avg_mae']:>10.4f} {avg_mae:>10.4f} {avg_mae-v3s['avg_mae']:>+10.4f}")
        print(f"{'avg_psnr':<20} {v3s['avg_psnr']:>9.1f}dB {avg_psnr:>9.1f}dB {avg_psnr-v3s['avg_psnr']:>+9.1f}dB")
        print(f"{'avg_pred_std':<20} {v3s['avg_pred_std']:>10.4f} {avg_pred_std:>10.4f} {avg_pred_std-v3s['avg_pred_std']:>+10.4f}")
        print(f"{'avg_tgt_std':<20} {v3s['avg_tgt_std']:>10.4f} {avg_tgt_std:>10.4f}")
    else:
        print("(v3 diag not found for comparison)")

    with open(OUT / "diag_summary.json", "w") as f:
        json.dump({"metrics": all_metrics, "summary": {
            "avg_mae": avg_mae, "avg_psnr": avg_psnr,
            "avg_pred_mean": avg_pred_mean, "avg_pred_std": avg_pred_std,
            "avg_tgt_mean": avg_tgt_mean, "avg_tgt_std": avg_tgt_std,
        }}, f, indent=2)
    print(f"\n[diag] saved → {OUT}")


if __name__ == "__main__":
    main()
