#!/usr/bin/env python3
"""Decisive diagnostic: can a trained flow-matching LoRA reconstruct what it memorized?

Runs 20-step Euler integration from input latents with checkpoint-300 on:
  A) TRAIN samples (the model was trained on these 2181 depth samples)
  B) VAL samples (held out)

For each sample we report, per Euler step:
  - cos_sim(v_pred, v_true) where v_true = target_latents - input_latents
  - running latent error vs target
and final: latent cos_sim, decoded depth Pearson r, AbsRel, delta1.

Interpretation:
  - Train OK + Val bad  => machinery consistent, generalization failure (data/steps).
  - Train also bad      => train/eval inconsistency bug; per-step table shows where.
"""
from __future__ import annotations

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
from open_vision_banana.model_registry import resolve_model_config, vae_decode, vae_encode

DEVICE = "mps"
RES = 512
N_STEPS = 20
MODEL = "/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell"
LORA = "output/flux_depth_gate/checkpoint-300"
DATA = PROJECT_ROOT / "data/vb_snvc_train_v3/depth"
OUT = PROJECT_ROOT / "output/diag_flow_reconstruction"

TRAIN_IDS = ["depth_hyp_real_00000", "depth_hyp_real_00001",
             "depth_hyp_real_00002", "depth_hyp_real_00003"]
VAL_IDS = ["depth_hyp_real_00006", "depth_hyp_real_00031",
           "depth_hyp_real_00037"]
DEFAULT_PROMPT = ("Generate a depth map of this image. Use a power-transform "
                  "color encoding where R,G,B encode metric depth values.")


def _pack(latents, b, c, h, w):
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def _unpack(latents, h, w):
    b, n, c = latents.shape
    latents = latents.view(b, h // 2, w // 2, c // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, c // 4, h, w)


def _img_ids(b, h, w, device, dtype):
    ids = torch.zeros(h, w, 3)
    ids[..., 1] += torch.arange(h)[:, None]
    ids[..., 2] += torch.arange(w)[None, :]
    return ids.reshape(h * w, 3).to(device=device, dtype=dtype)


def cos(a, b):
    return (torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0)).item()


@torch.no_grad()
def run_sample(pipe, model_config, image, tgt_rgb, prompt, mapper):
    device = DEVICE
    model_dtype = next(pipe.transformer.parameters()).dtype
    vae = pipe.vae
    vae_dtype = next(vae.parameters()).dtype

    img = image.resize((RES, RES), Image.LANCZOS)
    x = torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1)[None] / 255.0
    input_lat = vae_encode(vae, (x * 2 - 1).to(device, vae_dtype), model_config)

    tgt_img = Image.open(tgt_rgb).convert("RGB").resize((RES, RES), Image.LANCZOS)
    tgt = torch.from_numpy(
        np.array(tgt_img).astype(np.float32).transpose(2, 0, 1))[None] / 255.0
    target_lat = vae_encode(vae, (tgt * 2 - 1).to(device, vae_dtype), model_config)
    gt_depth = mapper.rgb_to_depth(tgt.cpu()).squeeze(0)

    v_true = (target_lat - input_lat).float()

    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    orig = [e.dtype if e is not None else None for e in encoders]
    for e in encoders:
        if e is not None:
            e.to(torch.float32)
    try:
        txt = pipe.encode_prompt(prompt=[prompt], prompt_2=[prompt], device=device,
                                 num_images_per_prompt=1, max_sequence_length=512)
    finally:
        for e, d in zip(encoders, orig):
            if e is not None and d is not None:
                e.to(d)
    prompt_embeds, pooled = txt[0], txt[1]
    text_ids = txt[2] if len(txt) > 2 else None

    z = input_lat.clone()
    dt = 1.0 / N_STEPS
    h, w = z.shape[-2], z.shape[-1]
    step_log = []
    for s in range(N_STEPS):
        t_val = s * dt
        t_tensor = torch.tensor([t_val], device=device, dtype=model_dtype)
        zp = _pack(z.detach(), 1, z.shape[1], h, w)
        ids = _img_ids(1, h // 2, w // 2, device, z.dtype)
        out = pipe.transformer(
            hidden_states=zp.to(model_dtype), timestep=t_tensor,
            encoder_hidden_states=prompt_embeds.to(model_dtype),
            pooled_projections=pooled.to(model_dtype) if pooled is not None else None,
            txt_ids=text_ids.to(model_dtype) if text_ids is not None else None,
            img_ids=ids.to(model_dtype), return_dict=False,
        )
        v = out[0] if isinstance(out, (tuple, list)) else out.sample
        v = _unpack(v, h, w)
        step_log.append({
            "t": round(t_val, 3),
            "v_cos": round(cos(v, v_true), 4),
            "v_norm_ratio": round((v.float().norm() / v_true.norm()).item(), 3),
        })
        z = z + dt * v

    lat_cos = cos(z, target_lat)
    dec = vae_decode(vae, z.to(vae_dtype), model_config)
    pred_rgb = ((dec + 1) / 2).clamp(0, 1).cpu()
    pred_depth = mapper.rgb_to_depth(pred_rgb).squeeze(0).float()
    gt_depth = gt_depth.float()
    m = evaluate_depth(pred_depth[None, None], gt_depth[None, None])
    pr = np.corrcoef(pred_depth.flatten().numpy(), gt_depth.flatten().numpy())[0, 1]
    return {"latent_cos": round(lat_cos, 4), "pearson": round(float(pr), 4),
            "AbsRel": round(float(m["AbsRel"]), 4), "delta1": round(float(m["δ<1.25"]), 4),
            "steps": step_log}, pred_rgb


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mapper = DepthRGBMapper(colormap="cube_edge")
    print(f"Loading FLUX + LoRA {LORA} ...", flush=True)
    t0 = time.time()
    gen = BaseGenerator(model_name=MODEL, lora_path=LORA, device=DEVICE, dtype=torch.bfloat16)
    pipe = gen._t2i_pipe
    cfg = resolve_model_config(MODEL)
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    report = {}
    for split, ids in [("train", TRAIN_IDS), ("val", VAL_IDS)]:
        for sid in ids:
            img_p = DATA / "images" / f"{sid}.jpg"
            tgt_p = DATA / "targets" / f"{sid}.png"
            prm_p = DATA / "prompts" / f"{sid}.txt"
            if not img_p.exists() or not tgt_p.exists():
                print(f"skip {sid}", flush=True)
                continue
            prompt = prm_p.read_text().strip() if prm_p.exists() else DEFAULT_PROMPT
            t1 = time.time()
            res, pred_rgb = run_sample(pipe, cfg, Image.open(img_p).convert("RGB"),
                                       tgt_p, prompt, mapper)
            res["prompt"] = prompt[:80]
            res["elapsed_s"] = round(time.time() - t1, 1)
            report[f"{split}/{sid}"] = res
            Image.fromarray((pred_rgb[0].permute(1, 2, 0) * 255).byte().numpy()).save(
                OUT / f"{split}_{sid}_pred.png")
            print(f"[{split}] {sid}: lat_cos={res['latent_cos']} pearson={res['pearson']} "
                  f"AbsRel={res['AbsRel']} d1={res['delta1']} ({res['elapsed_s']}s)", flush=True)
    (OUT / "diag_report.json").write_text(json.dumps(report, indent=2))
    print(f"saved -> {OUT/'diag_report.json'}", flush=True)


if __name__ == "__main__":
    main()
