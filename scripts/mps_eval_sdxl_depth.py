#!/usr/bin/env python3
"""Validation gate 3 — three-way depth benchmark on Apple Silicon (MPS).

Evaluates a *trained* SSD-1B depth LoRA against two references on held-out
samples from the SNvC depth set:

  1. SSD-1B + trained LoRA   (our paradigm: RGB-encoded cube_edge img2img)
  2. SSD-1B base (LoRA off)  (within-pipeline ablation)
  3. Depth-Anything-V2-Small (local zero-shot SOTA baseline, median-scaled)

Metrics: AbsRel / SqRel / RMSE / RMSE_log / δ<1.25{,²,³} (standard protocol).
FLUX/SSD predictions are ABSOLUTE metres via cube_edge decode (no scaling);
Depth-Anything outputs RELATIVE depth, median-scaled to GT per image.
Decoded depth is clamped to ``--max-depth`` for the ill-conditioned inverse
power-transform tail.

Fully offline (local HF cache).  LoRA checkpoint produced by
``scripts/mps_train_sdxl_depth.py``.

Example:
    PYTHONPATH=src python scripts/mps_eval_sdxl_depth.py \
        --lora output/mps_train_sdxl/ckpt_final --eval-idxs 120,500,900,1300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.conditioning import (
    CONV_IN_STATE_NAME,
    read_source_conditioning_config,
)
from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset

METRIC_KEYS = ["AbsRel", "SqRel", "RMSE", "RMSE_log", "δ<1.25", "δ<1.25²", "δ<1.25³"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="segmind/SSD-1B")
    ap.add_argument("--data",
                    default="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth")
    ap.add_argument("--lora", default=str(PROJECT_ROOT / "output/mps_train_sdxl/ckpt_final"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output/mps_eval_sdxl"))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--eval-idxs", default="120,500,900,1300,1700,2100")
    ap.add_argument("--infer-steps", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--max-depth", type=float, default=15.0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--da-repo", default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--no-da", action="store_true", help="skip Depth-Anything baseline")
    ap.add_argument("--output-calibration", default="moments_and_project",
                    choices=["none", "moments", "moments_and_project"],
                    help="inference-time colour-balance calibration (matches target "
                         "channel statistics + optional colormap projection)")
    ap.add_argument("--stats-n", type=int, default=400,
                    help="#targets for channel mean/std estimation")
    return ap.parse_args()


def load_trained_lora(gen, lora_dir: Path, rank: int, alpha: int) -> InstructionTuner:
    """Build the source-conditioned LoRA structure, then restore trained weights.

    Restores THREE things the training script saved: LoRA A/B factors, the
    trained expanded conv_in (source channels are base params, not LoRA), and
    the conditioning contract so ``gen.generate`` routes through the
    source-conditioned (InstructPix2Pix) denoising loop.
    """
    tuner = InstructionTuner(
        model=gen, task="depth", lora_rank=rank, lora_alpha=alpha,
        use_rslora=True, use_ema=False, use_gradient_checkpointing=False,
        use_min_snr=True, use_img2img_flow=False,
        use_source_conditioning=True, source_target_colormap="cube_edge",
    )
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    sd = load_file(str(lora_dir / "adapter_model.safetensors"))
    set_peft_model_state_dict(tuner.peft_model, sd)
    # Restore the trained expanded conv_in (8-channel) weights.
    conv = tuner._source_conditioning_conv()
    conv.load_state_dict(torch.load(lora_dir / CONV_IN_STATE_NAME, map_location="cpu"))
    # Route gen.generate(image=...) through the source-conditioned path.
    gen.source_conditioning_config = read_source_conditioning_config(lora_dir)
    tuner.peft_model.eval()
    return tuner


@torch.no_grad()
def predict(tuner, gen, image: Image.Image, prompt: str, enable_lora: bool,
            steps: int, strength: float, guidance: float) -> torch.Tensor:
    if enable_lora:
        tuner.peft_model.enable_adapter_layers()
    else:
        tuner.peft_model.disable_adapter_layers()
    out = gen.generate(prompt=prompt, image=image, strength=strength,
                       num_inference_steps=steps, guidance_scale=guidance, seed=42)
    tuner.peft_model.enable_adapter_layers()
    return out[0].clamp(0, 1).cpu()  # [3,H,W] in [0,1]


def main() -> None:
    args = parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    eval_idxs = [int(x) for x in args.eval_idxs.split(",") if x.strip()]
    mapper = DepthRGBMapper(colormap="cube_edge")

    def decode(rgb_3hw: torch.Tensor) -> torch.Tensor:
        d = mapper.rgb_to_depth(rgb_3hw.unsqueeze(0)).squeeze().cpu().float()
        if args.max_depth > 0:
            d = d.clamp(0.0, args.max_depth)
        return d

    def metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
        return evaluate_depth(pred.reshape(1, 1, *pred.shape).float(),
                              gt.reshape(1, 1, *gt.shape).float())

    # ── Load SSD-1B + trained LoRA ───────────────────────────────────────
    t0 = time.time()
    print(f"[model ] loading {args.model} on {args.device} ...", flush=True)
    gen = BaseGenerator(args.model, device=args.device, dtype=torch.float32)
    tuner = load_trained_lora(gen, Path(args.lora), args.rank, args.alpha)
    print(f"[model ] loaded + LoRA({args.lora}) in {time.time()-t0:.1f}s", flush=True)

    ds = VisionTaskDataset(data_path=args.data, resolution=args.res, task="depth")
    print(f"[data  ] {len(ds)} samples; evaluating idxs={eval_idxs}", flush=True)

    # ── Inference-time output calibration (colour-balance fix) ───────────
    # The trained UNet's raw RGB output carries the right structure but drifts
    # in colour balance (VAE + finite training), which the ill-conditioned
    # cube_edge inverse blows up to "all far".  Matching the target channel
    # mean/std (+ projecting onto the colormap curve) restores the scale.
    if args.output_calibration != "none" and gen.source_conditioning_config is not None:
        from dataclasses import replace
        idxs = list(range(0, min(args.stats_n, len(ds)), max(1, len(ds) // args.stats_n)))
        tgts = torch.stack([ds[i]["target_rgb"] for i in idxs])
        ch_mean = tuple(float(x) for x in tgts.mean(dim=(0, 2, 3)))
        ch_std = tuple(float(x) for x in tgts.std(dim=(0, 2, 3)).clamp(min=1e-4))
        gen.source_conditioning_config = replace(
            gen.source_conditioning_config,
            output_calibration=args.output_calibration,
            target_colormap="cube_edge",
            target_channel_mean=ch_mean,
            target_channel_std=ch_std,
        )
        print(f"[calib] {args.output_calibration}  mean={tuple(round(v,3) for v in ch_mean)} "
              f"std={tuple(round(v,3) for v in ch_std)} (n={len(idxs)})", flush=True)

    per = {"lora": [], "base": []}
    rows = []
    for idx in eval_idxs:
        idx = min(idx, len(ds) - 1)
        s = ds[idx]
        inp, tgt_rgb = s["image"], s["target_rgb"]
        prompt = s.get("prompt") or "Generate a metric depth visualization of this image."
        inp_pil = Image.fromarray(
            (inp.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy())
        gt = decode(tgt_rgb)
        d_lora = decode(predict(tuner, gen, inp_pil, prompt, True,
                                args.infer_steps, args.strength, args.guidance))
        d_base = decode(predict(tuner, gen, inp_pil, prompt, False,
                                args.infer_steps, args.strength, args.guidance))
        m_l, m_b = metrics(d_lora, gt), metrics(d_base, gt)
        per["lora"].append(m_l)
        per["base"].append(m_b)
        rows.append({"idx": idx, "inp": inp, "gt": gt,
                     "lora": d_lora, "base": d_base, "m_lora": m_l, "m_base": m_b})
        print(f"  idx={idx:>4}  GT mean={gt.mean():.2f}m | "
              f"LoRA AbsRel={m_l['AbsRel']:.3f} δ1={m_l['δ<1.25']:.3f} | "
              f"base AbsRel={m_b['AbsRel']:.3f}", flush=True)

    # ── Depth-Anything zero-shot baseline (median-scaled) ────────────────
    da_metrics_list = []
    if not args.no_da:
        try:
            from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation
            print(f"[da    ] loading {args.da_repo} (local) ...", flush=True)
            da = DepthAnythingForDepthEstimation.from_pretrained(
                args.da_repo, local_files_only=True).to(args.device)
            proc = AutoImageProcessor.from_pretrained(args.da_repo, local_files_only=True)
            da.eval()
            for r in rows:
                idx = r["idx"]
                s = ds[min(idx, len(ds) - 1)]
                gt = r["gt"]
                img_pil = Image.fromarray(
                    (s["image"].permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy())
                inp = proc(images=img_pil, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    pd = da(**inp).predicted_depth
                pd = torch.nn.functional.interpolate(
                    pd.unsqueeze(1), size=gt.shape[-2:], mode="bilinear",
                    align_corners=False).squeeze()
                m = gt > 0
                scale = gt[m].median() / pd[m].median().clamp(min=1e-6)
                pd = (pd * scale).clamp(0, args.max_depth if args.max_depth > 0 else 1e9)
                m_d = metrics(pd.cpu().float(), gt)
                da_metrics_list.append(m_d)
                r["da"] = pd.cpu()
                r["m_da"] = m_d
            print(f"[da    ] evaluated on {len(da_metrics_list)} samples", flush=True)
        except Exception as e:
            traceback.print_exc()
            print(f"[da    ] baseline skipped: {e}", flush=True)

    # ── Aggregate ────────────────────────────────────────────────────────
    def agg(ms):
        if not ms:
            return {}
        return {k: {"mean": float(np.mean([m[k] for m in ms])),
                    "std": float(np.std([m[k] for m in ms], ddof=1) if len(ms) > 1 else 0.0)}
                for k in METRIC_KEYS if k in ms[0]}

    result = {
        "lora_checkpoint": args.lora, "eval_idxs": eval_idxs,
        "infer": {"steps": args.infer_steps, "strength": args.strength,
                  "guidance": args.guidance, "max_depth": args.max_depth},
        "lora": {"per_sample": per["lora"], "aggregate": agg(per["lora"])},
        "base": {"per_sample": per["base"], "aggregate": agg(per["base"])},
    }
    if da_metrics_list:
        result["depth_anything_v2_small"] = {
            "per_sample": da_metrics_list, "aggregate": agg(da_metrics_list)}

    (OUT / "depth_eval.json").write_text(json.dumps(result, indent=2, default=str))

    # ── Console table ────────────────────────────────────────────────────
    def line(name, ag):
        a = {k: v.get("mean", float("nan")) for k, v in ag.items()}
        print(f"  {name:<26} AbsRel={a.get('AbsRel', float('nan')):.4f}  "
              f"RMSE={a.get('RMSE', float('nan')):.4f}  δ1={a.get('δ<1.25', float('nan')):.4f}")
    print("\n" + "=" * 60)
    print("GATE-3 DEPTH BENCHMARK — mean over held-out samples")
    print("=" * 60)
    line("SSD-1B + LoRA (ours)", result["lora"]["aggregate"])
    line("SSD-1B base (no LoRA)", result["base"]["aggregate"])
    if da_metrics_list:
        line("Depth-Anything-V2-Small", result["depth_anything_v2_small"]["aggregate"])
    print(f"\nFull results -> {OUT / 'depth_eval.json'}")

    # ── Viz grid ─────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ncols = 5 if da_metrics_list else 4
        n = len(rows)
        fig, axes = plt.subplots(n, ncols, figsize=(5 * ncols, 5 * n), squeeze=False)
        for r, row in enumerate(rows):
            inp_img = (row["inp"].clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
            cols = [(inp_img, f"Input idx={row['idx']}", None),
                    (row["gt"].numpy(), "GT depth", "viridis"),
                    (row["lora"].numpy(), f"LoRA {row['m_lora']['AbsRel']:.2f}", "viridis"),
                    (row["base"].numpy(), f"base {row['m_base']['AbsRel']:.2f}", "viridis")]
            if da_metrics_list and "da" in row:
                cols.append((row["da"].numpy(), f"DA {row['m_da']['AbsRel']:.2f}", "viridis"))
            for c, (img, title, cmap) in enumerate(cols):
                axes[r][c].imshow(img, cmap=cmap) if cmap else axes[r][c].imshow(img)
                axes[r][c].set_title(title, fontsize=10)
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"Gate-3 depth benchmark — {Path(args.lora).name} @ {args.res}px")
        fig_path = OUT / "depth_eval.png"
        fig.savefig(fig_path, dpi=100, bbox_inches="tight")
        print(f"Viz -> {fig_path}")
    except Exception as e:
        print(f"[viz   ] skipped: {e}")


if __name__ == "__main__":
    main()
