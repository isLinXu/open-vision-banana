#!/usr/bin/env python3
"""Diagnose source-conditioned SSD-1B depth output: raw RGB vs calibrated.

The decoded-depth "all far" collapse may be (a) a genuine structure collapse or
(b) a colour-balance/scale issue that the codebase's output calibration
(``calibrate_source_conditioned_output``, moments matching + optional colormap
projection) can fix WITHOUT retraining.  This script captures the model's RAW
RGB output and tests both, on one held-out sample.

    PYTHONPATH=src python scripts/diag_sc_calib.py \
        --lora output/mps_train_sdxl/ckpt_step500 --idx 120
"""
from __future__ import annotations

import argparse
import os
import sys
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
    SourceConditioningConfig,
    calibrate_source_conditioned_output,
    read_source_conditioning_config,
)
from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="segmind/SSD-1B")
    ap.add_argument("--data",
                    default="/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth")
    ap.add_argument("--lora", default=str(PROJECT_ROOT / "output/mps_train_sdxl/ckpt_step500"))
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--idx", type=int, default=120)
    ap.add_argument("--infer-steps", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--stats-n", type=int, default=200)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output/mps_diag_sc"))
    args = ap.parse_args()
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    mapper = DepthRGBMapper(colormap="cube_edge")

    # ── Load model + trained source-conditioned LoRA ─────────────────────
    gen = BaseGenerator(args.model, device=device, dtype=torch.float32)
    tuner = InstructionTuner(
        model=gen, task="depth", lora_rank=args.rank, lora_alpha=args.alpha,
        use_rslora=True, use_ema=False, use_gradient_checkpointing=False,
        use_min_snr=True, use_img2img_flow=False,
        use_source_conditioning=True, source_target_colormap="cube_edge",
    )
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(tuner.peft_model, load_file(str(Path(args.lora) / "adapter_model.safetensors")))
    tuner._source_conditioning_conv().load_state_dict(
        torch.load(Path(args.lora) / CONV_IN_STATE_NAME, map_location="cpu"))
    base_cfg = read_source_conditioning_config(args.lora)
    tuner.peft_model.eval()
    print(f"[load ] LoRA+conv_in from {args.lora}")

    # ── Dataset sample + target channel statistics ───────────────────────
    ds = VisionTaskDataset(data_path=args.data, resolution=args.res, task="depth")
    s = ds[min(args.idx, len(ds) - 1)]
    inp, tgt_rgb = s["image"], s["target_rgb"]
    prompt = s.get("prompt") or "Generate a metric depth visualization of this image."
    inp_pil = Image.fromarray(
        (inp.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy())

    # Target channel mean/std over a sample of cube_edge targets
    tgts = torch.stack([ds[i]["target_rgb"] for i in range(0, min(args.stats_n, len(ds)), 7)])
    ch_mean = tuple(float(x) for x in tgts.mean(dim=(0, 2, 3)))
    ch_std = tuple(float(x) for x in tgts.std(dim=(0, 2, 3)).clamp(min=1e-4))
    print(f"[stats] target RGB mean={tuple(round(v,3) for v in ch_mean)} "
          f"std={tuple(round(v,3) for v in ch_std)} (n={tgts.shape[0]})")

    # ── Inference: raw RGB output ────────────────────────────────────────
    def run(cfg: SourceConditioningConfig) -> torch.Tensor:
        gen.source_conditioning_config = cfg
        tuner.peft_model.enable_adapter_layers()
        out = gen.generate(prompt=prompt, image=inp_pil, strength=args.strength,
                           num_inference_steps=args.infer_steps, guidance_scale=0.0, seed=42)
        return out[0].clamp(0, 1).cpu()  # [3,H,W]

    raw_rgb = run(base_cfg)  # no calibration
    print(f"[infer] raw RGB mean={tuple(round(float(raw_rgb[c].mean()),3) for c in range(3))} "
          f"min={float(raw_rgb.min()):.2f} max={float(raw_rgb.max()):.2f}")

    # Calibrated output (moments_and_project onto the cube_edge curve)
    calib_cfg = SourceConditioningConfig(
        in_channels=base_cfg.in_channels, condition_channels=base_cfg.condition_channels,
        prediction_type="epsilon", target_colormap="cube_edge",
        target_channel_mean=ch_mean, target_channel_std=ch_std,
        output_calibration="moments_and_project",
        default_num_inference_steps=args.infer_steps,
    )
    # Re-run with calibration applied inside generate(); but calibration runs on
    # the SAME denoised output, so instead apply to raw_rgb for a fair compare.
    calib_rgb = calibrate_source_conditioned_output(raw_rgb.unsqueeze(0), calib_cfg)[0]

    # ── Decode depths & metrics ──────────────────────────────────────────
    def depth(rgb):
        return mapper.rgb_to_depth(rgb.unsqueeze(0)).squeeze().cpu().float().clamp(0, 15)
    def metric(pred, gt):
        return evaluate_depth(pred.reshape(1, 1, *pred.shape), gt.reshape(1, 1, *gt.shape))

    gt_d = depth(tgt_rgb)
    raw_d = depth(raw_rgb)
    cal_d = depth(calib_rgb)
    m_raw = metric(raw_d, gt_d)
    m_cal = metric(cal_d, gt_d)
    print(f"\n[depth] GT   mean={gt_d.mean():.2f}m range[{gt_d.min():.2f},{gt_d.max():.2f}]")
    print(f"[depth] raw  mean={raw_d.mean():.2f}m  AbsRel={m_raw['AbsRel']:.3f} δ1={m_raw['δ<1.25']:.3f}")
    print(f"[depth] cali mean={cal_d.mean():.2f}m  AbsRel={m_cal['AbsRel']:.3f} δ1={m_cal['δ<1.25']:.3f}")

    # ── Viz ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0][0].imshow((inp.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()); axes[0][0].set_title("Input")
    axes[0][1].imshow(tgt_rgb.permute(1, 2, 0).numpy()); axes[0][1].set_title("Target RGB (cube_edge GT)")
    axes[0][2].imshow(gt_d.numpy(), cmap="viridis"); axes[0][2].set_title("GT depth")
    axes[1][0].imshow(raw_rgb.permute(1, 2, 0).numpy()); axes[1][0].set_title("Model RAW RGB out")
    axes[1][1].imshow(raw_d.numpy(), cmap="viridis"); axes[1][1].set_title(f"raw depth AbsRel={m_raw['AbsRel']:.2f}")
    axes[1][2].imshow(cal_d.numpy(), cmap="viridis"); axes[1][2].set_title(f"calib depth AbsRel={m_cal['AbsRel']:.2f}")
    for a in axes.flat:
        a.axis("off")
    fig.suptitle(f"SC diag idx={args.idx} — {Path(args.lora).name}")
    fp = OUT / f"diag_idx{args.idx}.png"
    fig.savefig(fp, dpi=100, bbox_inches="tight")
    print(f"\n[viz  ] -> {fp}")


if __name__ == "__main__":
    main()
