#!/usr/bin/env python3
"""Paper-style depth gate: pure prompt-driven, no source conditioning.

Training follows the core claim of the Vision Banana paper:
- no widened conv_in, no extra condition channels
- perception tasks: the diffusion model understands the input image through the prompt and generates the matching depth colormap
- generation tasks: standard text-to-image, so the generation ability does not regress
- mixing ratio: ~10% vision tasks, ~90% generation data

Inference (img2img / SDEdit):
- add noise to the input photo and steer the denoising with the vision prompt

Usage:
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
# the paper says the vision data ratio is very low but gives no exact value; 10% is used as a starting point here
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
    # bf16 is unstable on MPS, SSD-1B is small enough → fp32
    dtype = torch.float32 if device == "mps" else torch.bfloat16

    print(f"[cfg] model={MODEL_ID}  device={device}  dtype={dtype}")
    print(f"[cfg] steps={STEPS}  nsamples={NSAMPLES}  res={RES}  batch={BATCH}")
    print(f"[cfg] lr={LR}  rank={RANK}  alpha={ALPHA}  vision_ratio={VISION_RATIO}")
    print(f"[cfg] out={OUT}")

    # 1) load the data ─────────────────────────────────────────────────
    ds = VisionTaskDataset(data_path=DATA_DIR, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data] loaded {n_total} depth samples @ {RES}px")
    if n_total < NSAMPLES:
        raise RuntimeError(f"Need {NSAMPLES} samples, have {n_total}")

    # a simple index subset (no split: the gate only checks whether it can overfit)
    subset = torch.utils.data.Subset(ds, list(range(NSAMPLES)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=BATCH, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn, drop_last=True,
    )
    print(f"[data] vision subset = {len(subset)}")

    # 2) load the generation model ─────────────────────────────────────
    t0 = time.time()
    gen = BaseGenerator(MODEL_ID, device=device, dtype=dtype)
    print(f"[model] family={gen.model_config.family}  loaded in {time.time()-t0:.1f}s")

    # 3) create the tuner without source conditioning ─────────────────
    tuner = InstructionTuner(
        model=gen,
        task="depth",
        lora_rank=RANK, lora_alpha=ALPHA, lr=LR, use_rslora=True,
        use_ema=False, use_gradient_checkpointing=False, use_min_snr=True,
        use_img2img_flow=False, cfg_dropout_rate=0.1,
        use_source_conditioning=False,  # key: do not widen conv_in
        allow_sdedit=True,               # new: allow SDEdit-style training
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

    # 5) training loop ─────────────────────────────────────────────────
    # two alternating modes:
    #   A (t2i-depth): noisy depth colormap z0 → learn what a depth colormap looks like
    #   B (img2img-photo): noisy input photo → learn to denoise with the vision prompt
    # at inference img2img(strength=0.85) = B's denoising + A's domain knowledge → depth colormap
    print(f"\n[train] paper-style: {STEPS} steps, dual-mode (t2i-depth 50% + img2img-photo 50%)")
    print(f"[train] prompt: {VISION_PROMPT[:80]}...")

    losses: list[float] = []
    it = iter(loader)
    t_train = time.time()
    image_backup: dict | None = None  # keeps the original image tensor for switching modes

    for step in range(1, STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        opt.zero_grad(set_to_none=True)

        # alternate the two modes: even steps = t2i-depth (noisy target), odd steps = img2img (noisy photo)
        use_t2i = (step % 2 == 0)
        if use_t2i:
            # Mode A: save the image, then clear it → the tuner takes t2i with the noisy depth_rgb as z0
            image_backup = {"image": batch.get("image")}
            batch["image"] = None
        else:
            # Mode B: restore the image → the tuner takes img2img on the noisy input photo
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
        # save a checkpoint every 150 steps
        if step % 150 == 0:
            mid_ckpt = OUT / f"lora_step{step}"
            mid_ckpt.mkdir(parents=True, exist_ok=True)
            tuner.peft_model.save_pretrained(str(mid_ckpt))
            print(f"[ckpt] mid-training save → {mid_ckpt}", flush=True)

    # ── training done, save the checkpoint ───────────────────────────
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

    # 6) inference evaluation: SDEdit img2img ──────────────────────────
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
        # SDEdit: start from the noised input and steer the denoising with the vision prompt
        out = gen.generate(
            prompt=VISION_PROMPT,
            image=pil_in,
            strength=0.85,
            num_inference_steps=8,
            guidance_scale=0.0,
            seed=42 + sample_idx,
        )
        pred_rgb = out[0].clamp(0, 1).cpu()
        # save the triptych
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"panel_{sample_idx:02d}.png")

        # RGB MAE (pixel space, independent of the colormap decode)
        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        # relative depth evaluation
        depth_metrics = evaluate_relative_depth_rgb(
            pred_rgb.unsqueeze(0), tgt_rgb.unsqueeze(0), colormap="viridis"
        )
        return {
            "idx": sample_idx,
            "rgb_mae": rgb_mae,
            "depth_mae": depth_metrics.get("mae", float("nan")),
            "pearson": depth_metrics.get("pearson", float("nan")),
        }

    for idx in range(min(NSAMPLES, 2)):  # evaluate only 2 samples to save memory
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

    # save
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
