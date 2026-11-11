#!/usr/bin/env python3
"""Standalone depth benchmark: load a trained checkpoint and evaluate it."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
from open_vision_banana.training.tuner import VisionTaskDataset

MODEL_ID = os.environ.get("VB_MODEL", "segmind/SSD-1B")
DATA_DIR = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
CKPT_DIR = os.environ.get("VB_CKPT", "")
RES = int(os.environ.get("VB_RES", "256"))
HOLDOUT = [int(x) for x in os.environ.get(
    "VB_HOLDOUT", "120,500,900,1300,1700,2100").split(",") if x.strip()]
ISTEPS = int(os.environ.get("VB_ISTEPS", "8"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "mps_depth_benchmark")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32
DEPTH_PROMPT = "Generate a metric depth visualization of this image."


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    device = _resolve_device()
    print(f"[cfg] model={MODEL_ID}  device={device}")
    print(f"[cfg] ckpt={CKPT_DIR}  res={RES}  holdout={HOLDOUT}")

    if not CKPT_DIR or not Path(CKPT_DIR).exists():
        raise RuntimeError(f"Checkpoint not found: {CKPT_DIR}")

    # load the model
    gen = BaseGenerator(MODEL_ID, device=device, dtype=DTYPE)
    print(f"[model] loaded")

    # load the checkpoint
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    from open_vision_banana.conditioning import expand_unet_conv_in
    ckpt = Path(CKPT_DIR)
    adapter_dir = ckpt / "adapter"
    conv_path = ckpt / CONV_IN_STATE_NAME
    if conv_path.exists():
        conv = torch.load(conv_path, map_location=device, weights_only=True)
        # widen conv_in to 8 channels first, then load
        expand_unet_conv_in(gen._t2i_pipe.unet, condition_channels=4)
        gen._t2i_pipe.unet.conv_in.load_state_dict(conv)
        print(f"[ckpt] loaded conv_in from {conv_path}")
    if adapter_dir.exists():
        adapter_state = load_peft_weights(str(adapter_dir), device=device)
        set_peft_model_state_dict(gen._t2i_pipe.unet, adapter_state)
        print(f"[ckpt] loaded adapter from {adapter_dir}")
    cond_config = read_source_conditioning_config(ckpt)
    if cond_config:
        gen.source_conditioning_config = cond_config
        print(f"[ckpt] loaded source conditioning config")

    # dataset
    ds = VisionTaskDataset(data_path=DATA_DIR, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data] loaded {n_total} samples")

    mapper = DepthRGBMapper(colormap="viridis")

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
            num_inference_steps=ISTEPS,
            guidance_scale=0.0,
            seed=42 + sample_idx,
        )
        pred_rgb = out[0].clamp(0, 1).cpu()
        pred_d = mapper.rgb_to_depth(pred_rgb.unsqueeze(0)).squeeze().clamp(0, 15).float()
        gt_d = mapper.rgb_to_depth(tgt.unsqueeze(0)).squeeze().clamp(0, 15).float()
        m = evaluate_depth(pred_d.reshape(1, 1, *pred_d.shape),
                           gt_d.reshape(1, 1, *gt_d.shape))

        # save the visualization
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

    results = []
    for idx in sorted(HOLDOUT):
        r = benchmark(idx)
        results.append(r)
        print(f"  idx={idx:>4}  δ1={r['delta1']:.4f}  AbsRel={r['absrel']:.4f}  RMSE={r['rmse']:.4f}")

    avg_d1 = sum(r["delta1"] for r in results) / len(results)
    avg_absrel = sum(r["absrel"] for r in results) / len(results)
    avg_rmse = sum(r["rmse"] for r in results) / len(results)

    print(f"\n[bench] avg δ1={avg_d1:.4f}  avg AbsRel={avg_absrel:.4f}  avg RMSE={avg_rmse:.4f}")

    (OUT / "benchmark_result.json").write_text(json.dumps({
        "model": MODEL_ID, "ckpt": CKPT_DIR, "device": device,
        "res": RES, "holdout": sorted(HOLDOUT),
        "avg_delta1": avg_d1, "avg_absrel": avg_absrel, "avg_rmse": avg_rmse,
        "per_sample": results,
    }, indent=2))
    print(f"[out] → {OUT / 'benchmark_result.json'}")


if __name__ == "__main__":
    main()
