#!/usr/bin/env python3
"""End-to-end depth benchmark eval: run a FLUX+LoRA checkpoint over a real
depth dataset (NYU/ETH3D/iBims) and report real metric numbers.

Mirrors the img2img-flow inference used in training/eval:
  start z = vae_encode(input_image); 20-step Euler with v = transformer(z, t)
  x0 = z_final; decode -> RGB -> depth; compare to GT over valid_mask.

Usage:
  PYTHONPATH=src python scripts/eval_depth_any_ckpt.py \
      --ckpt output/micro_overfit/lora_overfit.safetensors \
      --dataset nyu_depth --root data/benchmarks/nyu/extracted \
      --resolution 256 --max-samples 16
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.model_registry import resolve_model_config, vae_encode, vae_decode
from open_vision_banana.data.datasets import NYUDepthDataset, ETH3DDataset, IBimsDepthDataset

DATASETS = {"nyu_depth": NYUDepthDataset, "eth3d": ETH3DDataset, "ibims": IBimsDepthDataset}

MODEL = "/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell"


def _pack(latents, b, c, h, w):
    return latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(b, (h // 2) * (w // 2), c * 4)


def _unpack(latents, h, w):
    b, n, c = latents.shape
    return latents.view(b, h // 2, w // 2, c // 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(b, c // 4, h, w)


def _img_ids(b, h, w, device, dtype):
    ids = torch.zeros(h, w, 3)
    ids[..., 1] += torch.arange(h)[:, None]
    ids[..., 2] += torch.arange(w)[None, :]
    return ids.reshape(h * w, 3).to(device=device, dtype=dtype)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--single-step", action="store_true", help="also report 1-step x0")
    args = ap.parse_args()

    DEVICE = "mps"
    mapper = DepthRGBMapper(colormap="cube_edge")
    gen = BaseGenerator(model_name=MODEL, device=DEVICE, dtype=torch.bfloat16)
    pipe = gen._t2i_pipe
    cfg = resolve_model_config(MODEL)
    vae = pipe.vae
    mdtype = next(pipe.transformer.parameters()).dtype

    # load LoRA onto transformer
    from peft import PeftModel
    tr = pipe.transformer
    tr = PeftModel.from_pretrained(tr, args.ckpt)
    tr = tr.merge_and_unload() if hasattr(tr, "merge_and_unload") else tr
    tr.eval()

    ds = DATASETS[args.dataset](args.root, split="val", resolution=args.resolution)
    print(f"dataset {args.dataset}: {len(ds)} samples", flush=True)

    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    orig = [e.dtype if e is not None else None for e in encoders]
    for e in encoders:
        if e is not None:
            e.to(torch.float32)
    try:
        txt = pipe.encode_prompt(prompt=["x"], prompt_2=["x"], device=DEVICE,
                                 num_images_per_prompt=1, max_sequence_length=256)
    finally:
        for e, d in zip(encoders, orig):
            if e is not None and d is not None:
                e.to(d)
    pe, pooled, text_ids = txt[0], txt[1], (txt[2] if len(txt) > 2 else None)

    h = w = args.resolution // 8
    agg = {"δ<1.25": [], "δ<1.25²": [], "δ<1.25³": [], "AbsRel": [], "RMSE": [], "pearson": []}
    t0 = time.time()
    for idx in range(min(args.max_samples, len(ds))):
        sample = ds[idx]
        img = sample["image"]
        gt = sample["gt_depth"].float()
        vmask = sample.get("valid_mask")
        if vmask is not None:
            vmask = vmask.float()
        else:
            vmask = (gt > 0).float()
        x = torch.from_numpy(np.array(img.convert("RGB").resize((args.resolution, args.resolution), Image.LANCZOS))
                             .astype(np.float32)).permute(2, 0, 1)[None] / 255.0
        input_lat = vae_encode(vae, (x * 2 - 1).to(DEVICE, next(vae.parameters()).dtype), cfg)

        z = input_lat.clone()
        dt = 1.0 / args.steps
        for s in range(args.steps):
            tt = torch.tensor(s * dt, device=DEVICE, dtype=mdtype)
            zp = _pack(z.detach(), 1, z.shape[1], h, w)
            ids = _img_ids(1, h // 2, w // 2, DEVICE, zp.dtype)
            out = tr(hidden_states=zp.to(mdtype), timestep=tt.reshape(1),
                     encoder_hidden_states=pe.to(mdtype),
                     pooled_projections=pooled.to(mdtype) if pooled is not None else None,
                     txt_ids=text_ids.to(mdtype) if text_ids is not None else None,
                     img_ids=ids.to(mdtype), return_dict=False)
            v = _unpack(out[0] if isinstance(out, (tuple, list)) else out.sample, h, w)
            z = z + dt * v
        dec = vae_decode(vae, z.to(next(vae.parameters()).dtype), cfg)
        pred_rgb = ((dec + 1) / 2).clamp(0, 1).cpu()
        pred_depth = mapper.rgb_to_depth(pred_rgb).squeeze(0).float()

        m = evaluate_depth(pred_depth[None, None], gt[None, None], mask=vmask[None, None] if vmask is not None else None)
        pr = float(np.corrcoef(pred_depth.flatten().numpy(), gt.flatten().numpy())[0, 1]) if not np.isnan(
            np.corrcoef(pred_depth.flatten().numpy(), gt.flatten().numpy())[0, 1]) else 0.0
        for k in ("δ<1.25", "δ<1.25²", "δ<1.25³", "AbsRel", "RMSE"):
            agg[k].append(float(m[k]))
        agg["pearson"].append(pr)
        if (idx + 1) % 4 == 0:
            print(f"  {idx+1}/{min(args.max_samples, len(ds))} "
                  f"δ1={np.mean(agg['δ<1.25'][-4:]):.3f} "
                  f"AbsRel={np.mean(agg['AbsRel'][-4:]):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    summary = {k: round(float(np.mean(v)), 4) for k, v in agg.items() if v}
    outp = PROJECT_ROOT / "output" / f"eval_{args.dataset}_{Path(args.ckpt).parent.name}.json"
    outp.write_text(json.dumps({"ckpt": args.ckpt, "dataset": args.dataset, "n": len(agg["δ<1.25"]),
                                "steps": args.steps, "resolution": args.resolution, "metrics": summary}, indent=2))
    print("SUMMARY:", json.dumps(summary, indent=2))
    print("saved ->", outp)


if __name__ == "__main__":
    main()
