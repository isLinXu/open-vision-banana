#!/usr/bin/env python3
"""Formal SSD-1B depth LoRA training on Apple Silicon (MPS).

Post-smoke training run toward validation gate 3.  Builds on the smoke finding
(SSD-1B trains stably on MPS in fp32) and scales up: paper-grade rsLoRA
(rank 64 / alpha 128), a larger data subset, batched updates, and a
warmup+cosine LR schedule, with periodic checkpointing so the convergence
trajectory can be evaluated by ``scripts/mps_eval_sdxl_depth.py``.

Fully offline (local HF cache).  Runs the SDXL epsilon-prediction objective
via ``InstructionTuner._training_step_sdxl``.

Usage:
    PYTHONPATH=src python scripts/mps_train_sdxl_depth.py
Env overrides:
    VB_MODEL     HF id / local path            (default segmind/SSD-1B)
    VB_DATA      depth dir (images/targets/prompts)
    VB_STEPS     total optimizer steps         (default 3000)
    VB_NSAMPLES  training subset size          (default 1800)
    VB_RES       resolution                    (default 256)
    VB_BATCH     batch size                    (default 2)
    VB_LR        peak learning rate            (default 1e-4)
    VB_RANK      LoRA rank                     (default 64)
    VB_ALPHA     LoRA alpha                    (default 128)
    VB_WARMUP    warmup steps                  (default 100)
    VB_SAVE_EVERY  checkpoint interval         (default 500)
    VB_HOLDOUT   comma-separated TEST idxs     (excluded from training)
    VB_VAL_IDXS  comma-separated VAL idxs for early-stopping (also excluded)
    VB_VAL_EVERY   run validation every N steps (default 250)
    VB_VAL_PATIENCE early-stop after N non-improving vals (default 5)
    VB_VAL_ISTEPS  denoise steps for validation inference (default 8)
    VB_OUT       output dir                    (default output/mps_train_sdxl)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    "data/vb_snvc_train_v3/depth",
)
STEPS = int(os.environ.get("VB_STEPS", "3000"))
NSAMPLES = int(os.environ.get("VB_NSAMPLES", "100000"))  # large -> full dataset
RES = int(os.environ.get("VB_RES", "512"))
BATCH = int(os.environ.get("VB_BATCH", "1"))
LR = float(os.environ.get("VB_LR", "1e-4"))
RANK = int(os.environ.get("VB_RANK", "64"))
ALPHA = int(os.environ.get("VB_ALPHA", "128"))
WARMUP = int(os.environ.get("VB_WARMUP", "100"))
SAVE_EVERY = int(os.environ.get("VB_SAVE_EVERY", "500"))
# Test set (final reporting) AND val set (early-stopping) are BOTH excluded
# from training so selection/eval stay honest.
HOLDOUT = [int(x) for x in os.environ.get(
    "VB_HOLDOUT", "120,500,900,1300,1700,2100").split(",") if x.strip()]
VAL_IDXS = [int(x) for x in os.environ.get(
    "VB_VAL_IDXS", "200,700,1100,1500,1900").split(",") if x.strip()]
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "250"))
VAL_PATIENCE = int(os.environ.get("VB_VAL_PATIENCE", "5"))
VAL_ISTEPS = int(os.environ.get("VB_VAL_ISTEPS", "8"))
DTYPE = torch.float32  # stable on MPS; feasible because SSD-1B is small
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "mps_train_sdxl")))
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
print(f"[cfg   ] steps={STEPS} nsamples={NSAMPLES} res={RES} batch={BATCH} "
      f"lr={LR} rank={RANK} alpha={ALPHA} warmup={WARMUP} save_every={SAVE_EVERY}")
print(f"[cfg   ] test_holdout={HOLDOUT}  val={VAL_IDXS} "
      f"val_every={VAL_EVERY} patience={VAL_PATIENCE} val_isteps={VAL_ISTEPS}")
print(f"[cfg   ] out={OUT}")


def lr_lambda(step: int) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < WARMUP:
        return (step + 1) / max(1, WARMUP)
    progress = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * min(1.0, progress)))


def main() -> None:
    torch.manual_seed(42)

    # 1) Dataset + held-out exclusion ─────────────────────────────────────
    ds = VisionTaskDataset(data_path=DEPTH, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data  ] loaded {n_total} depth samples @ {RES}px")
    if n_total == 0:
        raise RuntimeError("Empty dataset — check images/targets/prompts layout")
    hold = {min(i, n_total - 1) for i in HOLDOUT}
    val_idx_set = {min(i, n_total - 1) for i in VAL_IDXS}
    exclude = hold | val_idx_set
    train_idx = [i for i in range(min(NSAMPLES, n_total)) if i not in exclude]
    subset = torch.utils.data.Subset(ds, train_idx)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    print(f"[data  ] train subset={len(train_idx)}  batch={BATCH}  "
          f"test_holdout={sorted(hold)}  val={sorted(val_idx_set)}")

    # 2) Base generator + tuner ───────────────────────────────────────────
    t0 = time.time()
    gen = BaseGenerator(MODEL, device=DEVICE, dtype=DTYPE)
    print(f"[model ] family={gen.model_config.family} loaded in {time.time()-t0:.1f}s")

    # Target channel statistics for inference-time output calibration.  The
    # UNet's raw RGB output drifts in colour balance (VAE + finite training),
    # which the ill-conditioned cube_edge inverse blows up to "all far";
    # matching these moments at inference restores the depth scale.  This only
    # affects inference (baked into conditioning_config.json), NOT the weights.
    stat_idx = list(range(0, min(400, n_total), max(1, n_total // 400)))
    tgts = torch.stack([ds[i]["target_rgb"] for i in stat_idx])
    ch_mean = tuple(float(x) for x in tgts.mean(dim=(0, 2, 3)))
    ch_std = tuple(float(x) for x in tgts.std(dim=(0, 2, 3)).clamp(min=1e-4))
    print(f"[calib] target RGB mean={tuple(round(v,3) for v in ch_mean)} "
          f"std={tuple(round(v,3) for v in ch_std)} (n={len(stat_idx)})")

    # Source-conditioning (InstructPix2Pix): noise the TARGET latent z0 and
    # concatenate the clean input-photo latent as extra UNet input channels.
    # This is the CORRECT input->target objective for perception tasks — the
    # default img2img epsilon path instead noises the INPUT and reconstructs it.
    tuner = InstructionTuner(
        model=gen, task="depth",
        lora_rank=RANK, lora_alpha=ALPHA, lr=LR, use_rslora=True,
        use_ema=False, use_gradient_checkpointing=False, use_min_snr=True,
        use_img2img_flow=False, cfg_dropout_rate=0.1,
        use_source_conditioning=True,
        source_target_colormap="cube_edge",
        source_target_channel_mean=ch_mean,
        source_target_channel_std=ch_std,
        source_output_calibration="moments_and_project",
        source_recommended_prompt="Generate a metric depth visualization of this image.",
    )
    tuner.peft_model.train()
    n_train = sum(p.numel() for p in tuner.peft_model.parameters() if p.requires_grad)
    print(f"[lora  ] trainable params = {n_train:,} "
          f"({100*n_train/sum(p.numel() for p in tuner.peft_model.parameters()):.2f}%)")

    # 3) Optimizer + warmup/cosine schedule ───────────────────────────────
    opt = torch.optim.AdamW(
        (p for p in tuner.peft_model.parameters() if p.requires_grad),
        lr=LR, weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def save_ckpt(tag: str) -> Path:
        d = OUT / f"ckpt_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        tuner.peft_model.save_pretrained(str(d))
        # Source-conditioning artifacts required for inference: the trained
        # expanded conv_in (source channels are base params, not LoRA) + the
        # conditioning contract the generator reads to route the forward.
        conv = tuner._source_conditioning_conv()
        torch.save(conv.state_dict(), d / CONV_IN_STATE_NAME)
        write_source_conditioning_config(d, tuner._source_conditioning_config())
        return d

    # ── Validation / early-stopping setup ────────────────────────────────
    # Route generate() through the calibrated source-conditioned path so val
    # inference matches deployment (moments_and_project already applied inside).
    gen.source_conditioning_config = tuner._source_conditioning_config()
    mapper = DepthRGBMapper(colormap="cube_edge")
    _VAL_PROMPT = "Generate a metric depth visualization of this image."

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
                out = gen.generate(prompt=s.get("prompt") or _VAL_PROMPT, image=pil,
                                   strength=0.85, num_inference_steps=VAL_ISTEPS,
                                   guidance_scale=0.0, seed=42)
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

    # 4) Training loop ────────────────────────────────────────────────────
    print(f"\n[train ] starting {STEPS} steps "
          f"(val every {VAL_EVERY}, patience {VAL_PATIENCE}) ...")
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
            print(f"[train ] step {step:>4}: NON-FINITE loss — skip", flush=True)
            continue
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(
                tuner.peft_model.parameters(), 1.0, error_if_nonfinite=True
            )
        except Exception:
            print(f"[train ] step {step:>4}: non-finite grad — skip "
                  f"(loss {loss.item():.4f})", flush=True)
            losses.append(loss.item())
            sched.step()
            continue
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 20 == 0 or step == 1:
            recent = sum(losses[-20:]) / len(losses[-20:])
            print(f"[train ] step {step:>4}/{STEPS}  loss={loss.item():.4f}  "
                  f"avg20={recent:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"({(time.time()-t_train)/step:.2f}s/step)", flush=True)
        if step % SAVE_EVERY == 0:
            d = save_ckpt(f"step{step}")
            print(f"[ckpt  ] saved -> {d}", flush=True)
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
            print(f"[val   ] step {step:>4}  δ1={d1:.4f}  AbsRel={ar:.4f}  "
                  f"best={best_d1:.4f}@{best_step}  patience={patience}{marker}", flush=True)
            if patience >= VAL_PATIENCE:
                print(f"[early ] no val improvement for {VAL_PATIENCE} evals — "
                      f"early stop at step {step} (best δ1={best_d1:.4f}@{best_step})")
                break

    if not losses:
        raise RuntimeError("No finite training steps — training failed")

    k = max(1, len(losses) // 3)
    avg_first = sum(losses[:k]) / k
    avg_last = sum(losses[-k:]) / k
    print(f"\n[train ] finite steps={len(losses)}/{STEPS}  "
          f"avg(first 1/3)={avg_first:.4f}  avg(last 1/3)={avg_last:.4f}  "
          f"({'DECREASING' if avg_last < avg_first else 'NOT decreasing'})")
    print(f"[train ] total time {(time.time()-t_train)/60:.1f} min")

    final = save_ckpt("final")
    (OUT / "losses.json").write_text(json.dumps({
        "model": MODEL, "device": DEVICE, "steps": STEPS, "nsamples": len(train_idx),
        "res": RES, "batch": BATCH, "lr": LR, "rank": RANK, "alpha": ALPHA,
        "losses": losses, "avg_first_third": avg_first, "avg_last_third": avg_last,
        "train_minutes": (time.time()-t_train)/60,
        "val_history": val_history,
        "best": {"step": best_step, "delta1": best_d1, "ckpt": "ckpt_best"},
    }, indent=2))
    print(f"[out   ] final LoRA -> {final}")
    print(f"[best  ] best val δ1={best_d1:.4f} @ step {best_step} -> {OUT / 'ckpt_best'}")
    print(f"[out   ] losses+val -> {OUT / 'losses.json'}")
    print("\n[done  ] formal SSD-1B depth training complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
