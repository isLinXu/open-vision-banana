#!/usr/bin/env python3
"""Decisive overfit micro-test for the depth img2img-flow mapping.

Trains a fresh LoRA on a TINY set (N pairs) for many steps (many epochs) so the
model *must* memorize. If latent_cos -> ~1.0 and delta1 rises sharply, the
training/inference machinery is correct and the production failure is purely a
scale (steps/data) problem. If it cannot overfit even here, there is a
structural bug in the velocity definition / x0 reconstruction.

Mirrors scripts/train.py img2img_flow path exactly:
  noisy = (1-t)*input + t*target (+ no noise_term)
  velocity_target = target - input
  loss = MSE(v_pred, velocity_target)

Processes ONE sample at a time (batch=1) to exactly match the working
inference call in diag_flow_reconstruction.py (avoids the rotary-emb shape bug).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.model_registry import resolve_model_config, vae_encode, vae_decode

DEVICE = "mps"
RES = 128
N_PAIRS = 4
STEPS = 300
LR = 1e-4
MODEL = "/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell"
DATA = PROJECT_ROOT / "data/vb_snvc_train_v3/depth"
OUT = PROJECT_ROOT / "output/micro_overfit"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = ["depth_hyp_real_00000", "depth_hyp_real_00001",
         "depth_hyp_real_00002", "depth_hyp_real_00003"]
PROMPT = ("Generate a depth map of this image. Use a power-transform "
          "color encoding where R,G,B encode metric depth values.")


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


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


@torch.no_grad()
def encode_prompt(pipe, encoders, orig, device):
    for e in encoders:
        if e is not None:
            e.to(torch.float32)
    try:
        txt = pipe.encode_prompt(prompt=[PROMPT], prompt_2=[PROMPT], device=device,
                                 num_images_per_prompt=1, max_sequence_length=256)
    finally:
        for e, d in zip(encoders, orig):
            if e is not None and d is not None:
                e.to(d)
    return txt


def transformer_step(tr, z, t_val, pe, pooled, text_ids, mdtype, h):
    zp = _pack(z.detach(), 1, z.shape[1], h, h)
    ids = _img_ids(1, h // 2, h // 2, DEVICE, zp.dtype)
    out = tr(hidden_states=zp.to(mdtype), timestep=torch.tensor(t_val, device=DEVICE, dtype=mdtype).reshape(1),
             encoder_hidden_states=pe.to(mdtype),
             pooled_projections=pooled.to(mdtype) if pooled is not None else None,
             txt_ids=text_ids.to(mdtype) if text_ids is not None else None,
             img_ids=ids.to(mdtype), return_dict=False)
    v = out[0] if isinstance(out, (tuple, list)) else out.sample
    return _unpack(v, h, h)


def main():
    mapper = DepthRGBMapper(colormap="cube_edge")
    t0 = time.time()
    gen = BaseGenerator(model_name=MODEL, device=DEVICE, dtype=torch.bfloat16)
    pipe = gen._t2i_pipe
    cfg = resolve_model_config(MODEL)
    vae = pipe.vae
    mdtype = next(pipe.transformer.parameters()).dtype
    encoders = [pipe.text_encoder, pipe.text_encoder_2]
    orig = [e.dtype if e is not None else None for e in encoders]
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    # fresh LoRA
    lora = LoraConfig(r=16, lora_alpha=32, use_rslora=True,
                      target_modules=["to_k", "to_v"], lora_dropout=0.0)
    tr = get_peft_model(pipe.transformer, lora)
    tr.train()
    opt = torch.optim.AdamW(tr.parameters(), lr=LR)

    # pre-encode image latents + gt depth + prompt (batch=1)
    txt = encode_prompt(pipe, encoders, orig, DEVICE)
    pe, pooled = txt[0], txt[1]
    text_ids = txt[2] if len(txt) > 2 else None

    enc = []
    with torch.no_grad():
        for sid in PAIRS:
            img = Image.open(DATA / "images" / f"{sid}.jpg").convert("RGB").resize((RES, RES), Image.LANCZOS)
            tgt = Image.open(DATA / "targets" / f"{sid}.png").convert("RGB").resize((RES, RES), Image.LANCZOS)
            x = torch.from_numpy(np.array(img).astype(np.float32)).permute(2, 0, 1)[None] / 255.0
            t = torch.from_numpy(np.array(tgt).astype(np.float32)).permute(2, 0, 1)[None] / 255.0
            il = vae_encode(vae, (x * 2 - 1).to(DEVICE, next(vae.parameters()).dtype), cfg).detach()
            tl = vae_encode(vae, (t * 2 - 1).to(DEVICE, next(vae.parameters()).dtype), cfg).detach()
            gt = mapper.rgb_to_depth(t.cpu()).squeeze(0).float()
            enc.append((il, tl, gt))

    h = RES // 8
    losses = []
    for step in range(STEPS):
        i = step % N_PAIRS
        il, tl, _ = enc[i]
        t_val = float(np.random.lognormal(mean=0.0, sigma=1.0))
        t_val = min(max(t_val, 1e-3), 1.0)
        tb = torch.tensor(t_val, device=DEVICE, dtype=mdtype)
        noisy = (1 - tb) * il + tb * tl
        vtar = (tl - il).float()
        v = transformer_step(tr, noisy, t_val, pe, pooled, text_ids, mdtype, h)
        loss = F.mse_loss(v.float(), vtar)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tr.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"step {step+1} loss={loss.item():.4f}", flush=True)

    torch.save(tr.state_dict(), OUT / "lora_overfit.safetensors")
    tr.save_pretrained(OUT / "lora_overfit")  # peft/diffusers-compatible dir
    print("trained.", flush=True)

    tr.eval()
    report = {}
    for i, (il, tl, gt) in enumerate(enc):
        # single-step x0
        v1 = transformer_step(tr, il, 0.0, pe, pooled, text_ids, mdtype, h)
        x0_single = (il + 1.0 * v1).float()
        dec1 = vae_decode(vae, x0_single.to(next(vae.parameters()).dtype), cfg)
        pred1 = ((dec1 + 1) / 2).clamp(0, 1).cpu()
        d1 = mapper.rgb_to_depth(pred1).squeeze(0).float()
        m1 = evaluate_depth(d1[None, None], gt[None, None])
        # 20-step
        z = il.clone()
        dt = 1.0 / 20
        for s in range(20):
            v = transformer_step(tr, z, s * dt, pe, pooled, text_ids, mdtype, h)
            z = z + dt * v
        lat_cos = cos(z, tl)
        dec = vae_decode(vae, z.to(next(vae.parameters()).dtype), cfg)
        pred = ((dec + 1) / 2).clamp(0, 1).cpu()
        d = mapper.rgb_to_depth(pred).squeeze(0).float()
        m = evaluate_depth(d[None, None], gt[None, None])
        pr = float(np.corrcoef(d.flatten().numpy(), gt.flatten().numpy())[0, 1])
        report[PAIRS[i]] = {"latent_cos": round(lat_cos, 4), "pearson": round(pr, 4),
                            "delta1_single": round(float(m1["δ<1.25"]), 4),
                            "AbsRel_single": round(float(m1["AbsRel"]), 4),
                            "delta1": round(float(m["δ<1.25"]), 4),
                            "AbsRel": round(float(m["AbsRel"]), 4)}
        Image.fromarray((pred[0].permute(1, 2, 0) * 255).byte().numpy()).save(OUT / f"{PAIRS[i]}_pred.png")
        print(f"{PAIRS[i]}: lat_cos={report[PAIRS[i]]['latent_cos']} "
              f"d1_single={report[PAIRS[i]]['delta1_single']} "
              f"d1_20step={report[PAIRS[i]]['delta1']}", flush=True)

    (OUT / "overfit_report.json").write_text(json.dumps(
        {"final_loss": round(losses[-1], 4), "mean_loss_last50": round(float(np.mean(losses[-50:])), 4),
         "samples": report}, indent=2))
    print("saved ->", OUT / "overfit_report.json", flush=True)


if __name__ == "__main__":
    main()
