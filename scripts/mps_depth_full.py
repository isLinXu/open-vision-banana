#!/usr/bin/env python3
"""MPS depth 完整复现 —— open-vision-banana 的核心任务.

目标：
- 全量 depth 数据（2296 样本）
- 充分训练（3000-5000 步）
- source conditioning + LoRA
- 标准 benchmark 评估（held-out）
- 与 frozen depth probe 对比
- 生成保真验证（可选）

MPS 约束：
- 256px 分辨率（内存）
- batch_size=1（内存）
- fp32（MPS 稳定）
- 预计 ~60-90 分钟

用法:
    PYTHONPATH=src python scripts/mps_depth_full.py
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

from open_vision_banana.conditioning import (
    CONV_IN_STATE_NAME,
    write_source_conditioning_config,
)
from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import (
    InstructionTuner,
    VisionTaskDataset,
    _collate_fn,
)

# ── Config ───────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("VB_MODEL", "segmind/SSD-1B")
DATA_DIR = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
STEPS = int(os.environ.get("VB_STEPS", "3000"))
RES = int(os.environ.get("VB_RES", "256"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
# 论文：vision data 比例很低，用少量样本即可
NSAMPLES = int(os.environ.get("VB_NSAMPLES", "100000"))  # 全量
# Held-out 评估集（不与训练重叠）
HOLDOUT = [int(x) for x in os.environ.get(
    "VB_HOLDOUT", "120,500,900,1300,1700,2100").split(",") if x.strip()]
VAL_IDXS = [int(x) for x in os.environ.get(
    "VB_VAL_IDXS", "200,700,1100,1500,1900").split(",") if x.strip()]
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "500"))
VAL_PATIENCE = int(os.environ.get("VB_VAL_PATIENCE", "5"))
VAL_ISTEPS = int(os.environ.get("VB_VAL_ISTEPS", "8"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "mps_depth_full")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32  # MPS safe
DEPTH_PROMPT = "Generate a metric depth visualization of this image."


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
    print(f"[cfg] steps={STEPS}  res={RES}  batch={BATCH}  lr={LR}  rank={RANK}  alpha={ALPHA}")
    print(f"[cfg] nsamples={NSAMPLES}  holdout={HOLDOUT}  val={VAL_IDXS}")
    print(f"[cfg] val_every={VAL_EVERY}  patience={VAL_PATIENCE}  out={OUT}")

    # 1) 数据集 ───────────────────────────────────────────────────────
    ds = VisionTaskDataset(data_path=DATA_DIR, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data] loaded {n_total} depth samples @ {RES}px")
    if n_total < 50:
        raise RuntimeError(f"Need at least 50 samples, have {n_total}")

    hold = {min(i, n_total - 1) for i in HOLDOUT}
    val_idx_set = {min(i, n_total - 1) for i in VAL_IDXS}
    exclude = hold | val_idx_set
    train_idx = [i for i in range(min(NSAMPLES, n_total)) if i not in exclude]
    subset = torch.utils.data.Subset(ds, train_idx)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    print(f"[data] train={len(train_idx)}  holdout={sorted(hold)}  val={sorted(val_idx_set)}")

    # 2) 模型 + source conditioning ────────────────────────────────────
    t0 = time.time()
    gen = BaseGenerator(MODEL_ID, device=device, dtype=DTYPE)
    print(f"[model] family={gen.model_config.family}  loaded in {time.time()-t0:.1f}s")

    # 目标色图统计（用于标定）
    stat_idx = list(range(0, min(400, n_total), max(1, n_total // 400)))
    tgts = torch.stack([ds[i]["target_rgb"] for i in stat_idx])
    ch_mean = tuple(float(x) for x in tgts.mean(dim=(0, 2, 3)))
    ch_std = tuple(float(x) for x in tgts.std(dim=(0, 2, 3)).clamp(min=1e-4))
    print(f"[calib] target RGB mean={tuple(round(v,3) for v in ch_mean)}  "
          f"std={tuple(round(v,3) for v in ch_std)}")

    tuner = InstructionTuner(
        model=gen, task="depth",
        lora_rank=RANK, lora_alpha=ALPHA, lr=LR, use_rslora=True,
        use_ema=False, use_gradient_checkpointing=False, use_min_snr=True,
        use_img2img_flow=False, cfg_dropout_rate=0.1,
        use_source_conditioning=True,
        source_target_colormap="viridis",
        source_target_channel_mean=ch_mean,
        source_target_channel_std=ch_std,
        source_output_calibration="moments_and_project",
        source_recommended_prompt=DEPTH_PROMPT,
    )
    tuner.peft_model.train()
    n_train = sum(p.numel() for p in tuner.peft_model.parameters() if p.requires_grad)
    n_total_p = sum(p.numel() for p in tuner.peft_model.parameters())
    print(f"[lora] trainable = {n_train:,}  ({100*n_train/n_total_p:.2f}%)")

    # 3) Optimizer ─────────────────────────────────────────────────────
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

    # 4) 验证函数 ─────────────────────────────────────────────────────
    gen.source_conditioning_config = tuner._source_conditioning_config()
    mapper = DepthRGBMapper(colormap="viridis")

    @torch.no_grad()
    def run_validation() -> tuple[float, float]:
        tuner.peft_model.eval()
        tuner.peft_model.enable_adapter_layers()
        d1s, absrels = [], []
        try:
            for vi in sorted(val_idx_set):
                s = ds[vi]
                inp, tgt = s["image"], s["target_rgb"]
                pil = Image.fromarray(
                    (inp.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy())
                out = gen.generate(
                    prompt=s.get("prompt") or DEPTH_PROMPT,
                    image=pil,
                    strength=0.85,
                    num_inference_steps=VAL_ISTEPS,
                    guidance_scale=0.0,
                    seed=42,
                )
                pred_rgb = out[0].clamp(0, 1).cpu()
                pred_d = mapper.rgb_to_depth(pred_rgb.unsqueeze(0)).squeeze().clamp(0, 15).float()
                gt_d = mapper.rgb_to_depth(tgt.unsqueeze(0)).squeeze().clamp(0, 15).float()
                m = evaluate_depth(pred_d.reshape(1, 1, *pred_d.shape),
                                   gt_d.reshape(1, 1, *gt_d.shape))
                d1s.append(m["δ<1.25"])
                absrels.append(m["AbsRel"])
        finally:
            tuner.peft_model.train()
        return (sum(d1s) / len(d1s), sum(absrels) / len(absrels))

    best_d1, best_step, patience = -1.0, 0, 0
    val_history: list[dict] = []

    def save_ckpt(tag: str) -> Path:
        d = OUT / f"ckpt_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        tuner.peft_model.save_pretrained(str(d))
        conv = tuner._source_conditioning_conv()
        torch.save(conv.state_dict(), d / CONV_IN_STATE_NAME)
        write_source_conditioning_config(d, tuner._source_conditioning_config())
        return d

    # 5) 训练循环 ─────────────────────────────────────────────────────
    print(f"\n[train] depth full: {STEPS} steps (val every {VAL_EVERY}, patience {VAL_PATIENCE})")

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
        loss = tuner._training_step(batch)

        if not torch.isfinite(loss).all():
            print(f"[train] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tuner.peft_model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % 100 == 0 or step == 1:
            recent = sum(losses[-100:]) / len(losses[-100:])
            print(f"[train] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg100={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)

        if step % VAL_EVERY == 0:
            d1, ar = run_validation()
            val_history.append({"step": step, "delta1": d1, "absrel": ar})
            if d1 > best_d1:
                best_d1, best_step, patience = d1, step, 0
                save_ckpt("best")
                marker = "  *BEST -> ckpt_best*"
            else:
                patience += 1
                marker = ""
            print(f"[val] step {step:>4}  δ1={d1:.4f}  AbsRel={ar:.4f}  "
                  f"best={best_d1:.4f}@{best_step}  patience={patience}{marker}", flush=True)
            if patience >= VAL_PATIENCE:
                print(f"[early] no val improvement for {VAL_PATIENCE} evals — "
                      f"early stop at step {step} (best δ1={best_d1:.4f}@{best_step})")
                break

    if not losses:
        raise RuntimeError("No finite training steps")

    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train] avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING' if avg_last < avg_first else 'NOT decreasing'})")

    final = save_ckpt("final")
    print(f"[ckpt] final → {final}")

    # 6) 完整 benchmark ────────────────────────────────────────────────
    print("\n[bench] full held-out benchmark...")
    gen.source_conditioning_config = tuner._source_conditioning_config()

    @torch.no_grad()
    def benchmark(sample_idx: int) -> dict:
        s = ds[sample_idx]
        inp, tgt = s["image"], s["target_rgb"]
        pil = Image.fromarray(
            (inp.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy())
        out = gen.generate(
            prompt=s.get("prompt") or DEPTH_PROMPT,
            image=pil,
            strength=0.85,
            num_inference_steps=VAL_ISTEPS,
            guidance_scale=0.0,
            seed=42 + sample_idx,
        )
        pred_rgb = out[0].clamp(0, 1).cpu()
        pred_d = mapper.rgb_to_depth(pred_rgb.unsqueeze(0)).squeeze().clamp(0, 15).float()
        gt_d = mapper.rgb_to_depth(tgt.unsqueeze(0)).squeeze().clamp(0, 15).float()
        m = evaluate_depth(pred_d.reshape(1, 1, *pred_d.shape),
                           gt_d.reshape(1, 1, *gt_d.shape))

        # 保存可视化
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"bench_{sample_idx:04d}.png")

        return {
            "idx": sample_idx,
            "delta1": m["δ<1.25"],
            "absrel": m["AbsRel"],
            "rmse": m["RMSE"],
        }

    bench_results = []
    for idx in sorted(hold):
        r = benchmark(idx)
        bench_results.append(r)
        print(f"  idx={idx:>4}  δ1={r['delta1']:.4f}  AbsRel={r['absrel']:.4f}  RMSE={r['rmse']:.4f}")

    avg_d1 = sum(r["delta1"] for r in bench_results) / len(bench_results)
    avg_absrel = sum(r["absrel"] for r in bench_results) / len(bench_results)
    avg_rmse = sum(r["rmse"] for r in bench_results) / len(bench_results)

    print(f"\n[bench] avg δ1={avg_d1:.4f}  avg AbsRel={avg_absrel:.4f}  avg RMSE={avg_rmse:.4f}")

    # 保存结果
    (OUT / "benchmark_result.json").write_text(json.dumps({
        "model": MODEL_ID, "device": device, "dtype": str(DTYPE),
        "steps": STEPS, "res": RES, "rank": RANK, "alpha": ALPHA,
        "nsamples": len(train_idx), "holdout": sorted(hold),
        "best_step": best_step, "best_d1": best_d1,
        "val_history": val_history,
        "benchmark": {
            "avg_delta1": avg_d1,
            "avg_absrel": avg_absrel,
            "avg_rmse": avg_rmse,
            "per_sample": bench_results,
        },
        "loss_first_third": avg_first, "loss_last_third": avg_last,
    }, indent=2))
    print(f"[out] → {OUT / 'benchmark_result.json'}")

    total_min = (time.time() - t_train) / 60
    print(f"\n[done] depth full complete ({total_min:.1f} min)")
    print(f"[done] best δ1={best_d1:.4f} @ step {best_step}")
    print(f"[done] avg benchmark δ1={avg_d1:.4f}  AbsRel={avg_absrel:.4f}")


if __name__ == "__main__":
    main()
