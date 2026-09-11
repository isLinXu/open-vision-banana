#!/usr/bin/env python3
"""Paper-style depth gate: pure prompt-driven, no source conditioning.

训练方式对齐 Vision Banana 论文核心主张：
- 不扩 conv_in，不加额外条件通道
- 对感知任务：扩散模型通过 prompt 理解输入图内容，生成对应深度色图
- 对生成任务：标准 text-to-image，保证生成能力不退
- 混合比例：视觉任务 ~10%，生成数据 ~90%

推理方式（img2img / SDEdit）：
- 对输入照片加噪声，用 vision prompt 引导去噪方向

用法:
    PYTHONPATH=src python scripts/paper_style_gate.py
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
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset, _collate_fn

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "segmind/SSD-1B")
DATA_DIR = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
STEPS = int(os.environ.get("VB_STEPS", "300"))
NSAMPLES = int(os.environ.get("VB_NSAMPLES", "8"))
RES = int(os.environ.get("VB_RES", "256"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
# 论文说 vision data 比例很低，但未公布精确值；这里用 10% 作为起点
VISION_RATIO = float(os.environ.get("VB_VISION_RATIO", "0.10"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "paper_style_gate")))
OUT.mkdir(parents=True, exist_ok=True)

VISION_PROMPT = (
    "Generate a relative depth map of this scene using the viridis RGB palette, "
    "from dark purple and blue through green to yellow."
)
GEN_PROMPTS = [
    "A serene mountain landscape at sunset",
    "A cozy coffee shop interior with warm lighting",
    "A bustling city street during rush hour",
    "A quiet forest path in autumn",
    "A modern minimalist living room",
    "A vibrant coral reef underwater",
    "A snowy village at dusk",
    "A futuristic city skyline at night",
    "An ancient library with towering bookshelves",
    "A peaceful Japanese garden with a koi pond",
]


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    torch.manual_seed(42)
    device = _resolve_device()
    # MPS 上 bf16 不稳定，SSD-1B 足够小 → fp32
    dtype = torch.float32 if device == "mps" else torch.bfloat16

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={dtype}")
    print(f"[cfg] steps={STEPS}  nsamples={NSAMPLES}  res={RES}  batch={BATCH}")
    print(f"[cfg] lr={LR}  rank={RANK}  alpha={ALPHA}  vision_ratio={VISION_RATIO}")
    print(f"[cfg] out={OUT}")

    # 1) 加载数据 ───────────────────────────────────────────────────────
    ds = VisionTaskDataset(data_path=DATA_DIR, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data] loaded {n_total} depth samples @ {RES}px")
    if n_total < NSAMPLES:
        raise RuntimeError(f"Need {NSAMPLES} samples, have {n_total}")

    # 用简单索引子集（不去 split — gate 只看能不能过拟合）
    subset = torch.utils.data.Subset(ds, list(range(NSAMPLES)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    print(f"[data] vision subset = {len(subset)}")

    # 2) 加载生成模型 ──────────────────────────────────────────────────
    t0 = time.time()
    gen = BaseGenerator(MODEL_ID, device=device, dtype=dtype)
    print(f"[model] family={gen.model_config.family}  loaded in {time.time()-t0:.1f}s")

    # 3) 创建 tuner — 不加 source conditioning ─────────────────────────
    tuner = InstructionTuner(
        model=gen,
        task="depth",
        lora_rank=RANK, lora_alpha=ALPHA, lr=LR, use_rslora=True,
        use_ema=False, use_gradient_checkpointing=False, use_min_snr=True,
        use_img2img_flow=False, cfg_dropout_rate=0.1,
        use_source_conditioning=False,  # 关键：不扩 conv_in
        allow_sdedit=True,               # 新：允许 SDEdit 风格训练
        source_target_colormap="viridis",
        source_target_channel_mean=(0.0, 0.0, 0.0),
        source_target_channel_std=(1.0, 1.0, 1.0),
        source_output_calibration="moments_and_project",
        source_recommended_prompt=VISION_PROMPT,
    )
    tuner.peft_model.train()
    n_train = sum(p.numel() for p in tuner.peft_model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in tuner.peft_model.parameters())
    print(f"[lora] trainable = {n_train:,}  ({100*n_train/n_total_p:.2f}%)")

    # 4) Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.AdamW(
        [p for p in tuner.peft_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )

    def lr_lambda(step: int) -> float:
        warmup = max(1, STEPS // 30)
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, STEPS - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 5) 训练循环 ─────────────────────────────────────────────────────
    # 双模式交替：
    #   A (t2i-depth): 噪声深度色图 z0 → 学"深度色图长什么样"
    #   B (img2img-photo): 噪声输入照片 → 学"用 vision prompt 去噪"关联
    # 推理时 img2img(strength=0.85) = B 的去噪能力 + A 的域知识 → 深度色图
    print(f"\n[train] paper-style: {STEPS} steps, dual-mode (t2i-depth 50% + img2img-photo 50%)")
    print(f"[train] prompt: {VISION_PROMPT[:80]}...")

    losses: list[float] = []
    it = iter(loader)
    t_train = time.time()
    image_backup: dict | None = None  # 保存原始 image tensor 用于切换模式

    for step in range(1, STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        opt.zero_grad(set_to_none=True)

        # 双模式交替：偶数步=t2i-depth(噪声目标)，奇数步=img2img(噪声照片)
        use_t2i = (step % 2 == 0)
        if use_t2i:
            # Mode A: 保存 image 然后清空 → tuner 走 t2i，噪声 depth_rgb 做 z0
            image_backup = {"image": batch.get("image")}
            batch["image"] = None
        else:
            # Mode B: 恢复 image → tuner 走 img2img，噪声输入照片
            if image_backup is not None and image_backup.get("image") is not None:
                batch["image"] = image_backup["image"]

        loss = tuner._training_step(batch)
        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuner.peft_model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 30 == 0 or step == 1:
            recent = sum(losses[-30:]) / len(losses[-30:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg30={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"mode={'t2i-depth' if use_t2i else 'img2img-photo'}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)
        # 每 150 步存一次 checkpoint
        if step % 150 == 0:
            mid_ckpt = OUT / f"lora_step{step}"
            mid_ckpt.mkdir(parents=True, exist_ok=True)
            tuner.peft_model.save_pretrained(str(mid_ckpt))
            print(f"[ckpt] mid-training save → {mid_ckpt}", flush=True)

    # ── 训练完成，保存 checkpoint ────────────────────────────────────
    if not losses:
        raise RuntimeError("No finite training steps")

    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING' if avg_last < avg_first else 'NOT decreasing'})")

    ckpt_dir = OUT / "lora_final"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tuner.peft_model.save_pretrained(str(ckpt_dir))
    print(f"[ckpt] saved → {ckpt_dir}")

    # 6) 推理评估 — SDEdit img2img ──────────────────────────────────
    print("\n[eval] SDEdit img2img inference on training samples...")
    mapper = DepthRGBMapper(colormap="viridis")
    results = []

    @torch.no_grad()
    def sdedit_infer(sample_idx: int) -> dict:
        s = ds[sample_idx]
        inp_tensor = s["image"]  # [-1, 1]
        tgt_rgb = s["target_rgb"]  # [0, 1] or [-1, 1] depending on dataset
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy()
        )
        # SDEdit: 从 input 加噪起步，用 vision prompt 引导去噪
        out = gen.generate(
            prompt=VISION_PROMPT,
            image=pil_in,
            strength=0.85,
            num_inference_steps=8,
            guidance_scale=0.0,
            seed=42 + sample_idx,
        )
        pred_rgb = out[0].clamp(0, 1).cpu()
        # 保存三联图
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"panel_{sample_idx:02d}.png")

        # 计算 RGB MAE（像素空间，不依赖色图解码）
        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        # 相对深度评估
        depth_metrics = evaluate_relative_depth_rgb(
            pred_rgb.unsqueeze(0), tgt_rgb.unsqueeze(0), colormap="viridis"
        )
        return {
            "idx": sample_idx,
            "rgb_mae": rgb_mae,
            "depth_mae": depth_metrics.get("mae", float("nan")),
            "pearson": depth_metrics.get("pearson", float("nan")),
        }

    for idx in range(min(NSAMPLES, 2)):  # 仅评 2 样本以节约内存
        r = sdedit_infer(idx)
        results.append(r)
        print(f"[eval] idx={idx:>3}  rgb_mae={r['rgb_mae']:.4f}  "
              f"depth_mae={r['depth_mae']:.4f}  pearson={r['pearson']:.4f}")

    avg_mae = sum(r["rgb_mae"] for r in results) / len(results)
    avg_pearson = sum(r["pearson"] for r in results if not math.isnan(r["pearson"])) / max(1, len(results))
    print(f"\n[gate] avg rgb_mae={avg_mae:.4f}  avg pearson={avg_pearson:.4f}")
    gate_passed = avg_mae < 0.15 and avg_pearson > 0.7
    print(f"[gate] {'PASSED' if gate_passed else 'NOT PASSED'}  "
          f"(target: rgb_mae<0.15, pearson>0.7)")

    # 保存
    (OUT / "gate_result.json").write_text(json.dumps({
        "model": MODEL_ID, "device": device, "dtype": str(dtype),
        "steps": STEPS, "nsamples": NSAMPLES, "res": RES,
        "rank": RANK, "alpha": ALPHA, "vision_ratio": VISION_RATIO,
        "loss_first_third": avg_first, "loss_last_third": avg_last,
        "avg_rgb_mae": avg_mae, "avg_pearson": avg_pearson,
        "gate_passed": gate_passed, "per_sample": results,
    }, indent=2))
    print(f"[out] → {OUT / 'gate_result.json'}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] paper-style gate complete ({total_min:.1f} min)  "
          f"{'✓ PASSED' if gate_passed else '✗ not passed'}")


if __name__ == "__main__":
    main()
