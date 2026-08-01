#!/usr/bin/env python3
"""SSD-1B (small SDXL) depth-task smoke validation on Apple Silicon (MPS).

Quick validation gate that sidesteps the FLUX fp16/MPS numerical instability:
we swap the 12B FLUX transformer for the 1.3B-param SSD-1B UNet (a distilled
SDXL, ~10x smaller) and train a real LoRA on a SMALL depth subset.  The SDXL
path uses native epsilon-prediction + Euler scheduler + scale_model_input +
min-SNR — a far more robust objective than the FLUX flow-matching branch that
NaN'd after step 1.

Validates:
  1) SSD-1B loads on MPS straight from the local HF cache (fully offline).
  2) ``InstructionTuner._training_step_sdxl`` yields finite, decreasing loss.
  3) LoRA img2img inference decodes to a plausible depth map (3-panel viz + AbsRel).

Usage:
    PYTHONPATH=src python scripts/mps_smoke_sdxl_depth.py
Env overrides:
    VB_MODEL     HF id / local path          (default segmind/SSD-1B)
    VB_DATA      depth dir (images/targets/prompts)
    VB_STEPS     training steps              (default 32)
    VB_NSAMPLES  training subset size        (default 96)
    VB_RES       resolution                  (default 256)
    VB_LR        LoRA learning rate          (default 1e-4)
    VB_DTYPE     float32 | float16           (default float32 — stable on MPS)
    VB_EVAL_IDX  held-out sample index       (default 120)
    VB_ISTEPS    img2img denoise steps       (default 20)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# ── Offline-first: force HF cache-only BEFORE importing torch/diffusers ────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import (
    InstructionTuner,
    VisionTaskDataset,
    _collate_fn,
)

# ── Config ─────────────────────────────────────────────────────────────────
MODEL = os.environ.get("VB_MODEL", "segmind/SSD-1B")
DEPTH = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
STEPS = int(os.environ.get("VB_STEPS", "200"))
NSAMPLES = int(os.environ.get("VB_NSAMPLES", "256"))
RES = int(os.environ.get("VB_RES", "256"))
LR = float(os.environ.get("VB_LR", "1e-4"))
DTYPE = torch.float32 if os.environ.get("VB_DTYPE", "float32") == "float32" else torch.float16
# Comma-separated held-out sample indices for evaluation.
EVAL_IDXS = [int(x) for x in os.environ.get("VB_EVAL_IDXS", "120,500,900").split(",") if x.strip()]
ISTEPS = int(os.environ.get("VB_ISTEPS", "20"))
STRENGTH = float(os.environ.get("VB_STRENGTH", "0.85"))
GUIDANCE = float(os.environ.get("VB_GUIDANCE", "2.0"))
# Clamp decoded depth to this max (m) for metric reporting — the inverse power
# transform is ill-conditioned near normalized→1, so undertrained predictions
# explode to 1000s of meters and swamp AbsRel. 0 disables clamping.
MAX_DEPTH = float(os.environ.get("VB_MAX_DEPTH", "15.0"))
OUT = PROJECT_ROOT / "output" / "mps_smoke_sdxl"
OUT.mkdir(parents=True, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"[device] {DEVICE}  (mps={torch.backends.mps.is_available()})")
print(f"[model ] {MODEL}  dtype={DTYPE}")
print(f"[data  ] {DEPTH}")
print(f"[cfg   ] steps={STEPS} nsamples={NSAMPLES} res={RES} lr={LR} "
      f"eval_idxs={EVAL_IDXS} isteps={ISTEPS} strength={STRENGTH} guidance={GUIDANCE}")


def main() -> None:
    torch.manual_seed(42)

    # 1) Dataset (depth only) + small training subset ─────────────────────
    ds = VisionTaskDataset(data_path=DEPTH, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data  ] loaded {n_total} depth samples @ {RES}px")
    if n_total == 0:
        raise RuntimeError("Empty dataset — check images/targets/prompts layout")
    eval_idxs = [min(i, n_total - 1) for i in EVAL_IDXS]
    eval_set = set(eval_idxs)
    train_idx = [i for i in range(min(NSAMPLES, n_total)) if i not in eval_set]
    subset = torch.utils.data.Subset(ds, train_idx)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=1, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn,
    )
    print(f"[data  ] train subset={len(train_idx)}  held-out eval idxs={eval_idxs}")

    # 2) Base generator (SSD-1B / SDXL family) on MPS ─────────────────────
    t0 = time.time()
    gen = BaseGenerator(MODEL, device=DEVICE, dtype=DTYPE)
    cfg = gen.model_config
    print(f"[model ] family={cfg.family} pipe={cfg.pipeline_class} "
          f"loaded in {time.time()-t0:.1f}s")

    # 3) LoRA tuner (dispatches to _training_step_sdxl for family=='sdxl') ─
    tuner = InstructionTuner(
        model=gen,
        task="depth",
        lora_rank=16,
        lora_alpha=32,
        lr=LR,
        use_rslora=True,
        use_ema=False,
        use_gradient_checkpointing=False,  # small model — no need, keeps it fast
        use_min_snr=False,
        use_img2img_flow=False,            # FLUX-only flag; ignored for sdxl
        cfg_dropout_rate=0.1,
        use_source_conditioning=True,
        prediction_type="epsilon",
    )
    gen.source_conditioning_config = tuner._source_conditioning_config()
    tuner.peft_model.train()
    n_train = sum(p.numel() for p in tuner.peft_model.parameters() if p.requires_grad)
    print(f"[lora  ] trainable LoRA params = {n_train:,} "
          f"({100*n_train/sum(p.numel() for p in tuner.peft_model.parameters()):.2f}%)")

    # 4) Manual training loop (mirrors train()'s core math, no autocast → fp32)
    opt = torch.optim.AdamW(
        (p for p in tuner.peft_model.parameters() if p.requires_grad),
        lr=LR, weight_decay=0.01,
    )
    print(f"\n[train ] starting {STEPS} real MPS steps ...")
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
        loss = tuner._training_step(batch)  # family==sdxl → _training_step_sdxl
        if not torch.isfinite(loss).all():
            print(f"[train ] step {step:>3}: NON-FINITE loss={loss.item()} — skipping update")
            continue
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(
                tuner.peft_model.parameters(), 1.0, error_if_nonfinite=True
            )
        except Exception:
            print(f"[train ] step {step:>3}: non-finite grad — skipping update "
                  f"(loss {loss.item():.4f} recorded)")
            losses.append(loss.item())
            continue
        opt.step()
        losses.append(loss.item())
        print(f"[train ] step {step:>3}/{STEPS}  loss={loss.item():.6f}  "
              f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)

    if not losses:
        raise RuntimeError("No finite training steps — training failed")

    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    trend = "DECREASING" if avg_last < avg_first else "NOT decreasing"
    print(f"\n[train ] finite steps={len(losses)}/{STEPS}  "
          f"avg(first 1/3)={avg_first:.6f}  avg(last 1/3)={avg_last:.6f}  -> {trend}")

    (OUT / "losses.json").write_text(json.dumps({
        "model": MODEL, "device": DEVICE, "dtype": str(DTYPE), "steps": STEPS,
        "res": RES, "lr": LR, "losses": losses,
        "avg_first_third": avg_first, "avg_last_third": avg_last,
    }, indent=2))

    lora_dir = OUT / "lora_adapter"
    tuner.peft_model.save_pretrained(str(lora_dir))
    print(f"[out   ] LoRA -> {lora_dir}")

    # 5) Held-out img2img inference → decode depth → viz + metric ─────────
    do_eval(tuner, gen, ds, eval_idxs, OUT)
    print("\n[done  ] SSD-1B depth smoke validation complete.")


@torch.no_grad()
def do_eval(tuner, gen, ds, eval_idxs: list[int], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mapper = DepthRGBMapper()  # cube_edge — matches training encoding

    def decode(rgb_3hw: torch.Tensor) -> torch.Tensor:
        # rgb_to_depth returns [B,1,H,W]; .squeeze() collapses to [H,W]
        d = mapper.rgb_to_depth(rgb_3hw.unsqueeze(0)).squeeze().cpu().float()
        if MAX_DEPTH > 0:
            d = d.clamp(0.0, MAX_DEPTH)
        return d

    def metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
        # evaluate_depth expects [B,1,H,W]
        return evaluate_depth(
            pred.reshape(1, 1, *pred.shape).float(),
            gt.reshape(1, 1, *gt.shape).float(),
        )

    rows = []
    all_lora, all_base = [], []
    print(f"\n[eval  ] {len(eval_idxs)} held-out samples (depth clamped to [0,{MAX_DEPTH}]m)")
    for idx in eval_idxs:
        sample = ds[idx]
        inp = sample["image"]            # [-1,1]  [3,H,W]
        tgt_rgb = sample["target_rgb"]   # [0,1]   [3,H,W]
        prompt = sample.get("prompt") or "Generate a metric depth visualization of this image."
        inp_pil = Image.fromarray(
            (inp.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy()
        )

        def predict(enable_lora: bool) -> torch.Tensor:
            if enable_lora:
                tuner.peft_model.enable_adapter_layers()
            else:
                tuner.peft_model.disable_adapter_layers()
            out = gen.generate(
                prompt=prompt, image=inp_pil, strength=STRENGTH,
                num_inference_steps=ISTEPS, guidance_scale=GUIDANCE, seed=42,
            )
            tuner.peft_model.enable_adapter_layers()
            return out[0].clamp(0, 1).cpu()  # [3,H,W] in [0,1]

        pred_lora_rgb = predict(True)
        pred_base_rgb = predict(False)

        gt_depth = decode(tgt_rgb)
        pred_lora = decode(pred_lora_rgb)
        pred_base = decode(pred_base_rgb)
        m_lora = metrics(pred_lora, gt_depth)
        m_base = metrics(pred_base, gt_depth)
        all_lora.append(m_lora)
        all_base.append(m_base)
        rows.append((idx, inp, gt_depth, pred_lora, pred_base, m_lora, m_base))
        print(f"  idx={idx:>4}  GT[{gt_depth.min():.2f},{gt_depth.max():.2f}]m "
              f"mean={gt_depth.mean():.2f}m | "
              f"LoRA AbsRel={m_lora['AbsRel']:.3f} RMSE={m_lora['RMSE']:.3f} "
              f"δ1={m_lora['δ<1.25']:.3f} | "
              f"base AbsRel={m_base['AbsRel']:.3f} RMSE={m_base['RMSE']:.3f}", flush=True)

    def agg(ms):
        return {k: float(np.mean([m[k] for m in ms]))
                for k in ["AbsRel", "SqRel", "RMSE", "δ<1.25", "δ<1.25²"]}

    agg_lora, agg_base = agg(all_lora), agg(all_base)
    print(f"\n[eval  ] MEAN over {len(eval_idxs)} samples:")
    print(f"        LoRA : AbsRel={agg_lora['AbsRel']:.4f}  RMSE={agg_lora['RMSE']:.4f}  "
          f"δ1={agg_lora['δ<1.25']:.4f}")
    print(f"        base : AbsRel={agg_base['AbsRel']:.4f}  RMSE={agg_base['RMSE']:.4f}  "
          f"δ1={agg_base['δ<1.25']:.4f}")

    (out_dir / "eval.json").write_text(json.dumps({
        "eval_idxs": eval_idxs, "max_depth_clamp": MAX_DEPTH,
        "per_sample": [
            {"idx": idx, "lora": ml, "base": mb}
            for (idx, *_rest, ml, mb) in rows
        ],
        "mean": {"lora": agg_lora, "base": agg_base},
    }, indent=2, default=str))

    # Grid viz: one row per sample → input / GT / LoRA pred / base pred
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n), squeeze=False)
    for r, (idx, inp, gt_depth, pred_lora, pred_base, m_lora, m_base) in enumerate(rows):
        inp_img = (inp.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
        axes[r][0].imshow(inp_img); axes[r][0].set_title(f"Input idx={idx}")
        axes[r][1].imshow(gt_depth.numpy(), cmap="viridis"); axes[r][1].set_title("GT depth")
        axes[r][2].imshow(pred_lora.numpy(), cmap="viridis")
        axes[r][2].set_title(f"LoRA (AbsRel={m_lora['AbsRel']:.2f})")
        axes[r][3].imshow(pred_base.numpy(), cmap="viridis")
        axes[r][3].set_title(f"base (AbsRel={m_base['AbsRel']:.2f})")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"SSD-1B depth LoRA — {STEPS} steps @ {RES}px (device={DEVICE})")
    fig_path = out_dir / "smoke_sdxl_depth.png"
    fig.savefig(fig_path, dpi=100, bbox_inches="tight")
    print(f"[viz   ] grid -> {fig_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
