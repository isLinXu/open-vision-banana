#!/usr/bin/env python3
"""MPS depth reproduction + 3-way evaluation (validation gate 3).

Self-contained: trains a real FLUX.1-schnell depth LoRA on Apple Silicon (MPS),
then evaluates THREE predictors on held-out samples:

  1. FLUX + trained LoRA   (our paradigm)
  2. FLUX base (LoRA disabled)  (within-pipeline ablation)
  3. Depth-Anything-V2-Small    (local zero-shot baseline, median-scaled)

Metrics: AbsRel / SqRel / RMSE / δ<1.25 (standard depth-estimation protocol).
FLUX predicts ABSOLUTE meters via cube_edge RGB encoding (no scaling needed);
Depth-Anything predicts RELATIVE depth, so it is median-scaled to GT per image.

Outputs:
  output/mps_eval/lora_adapter/    trained LoRA (adapter_config.json + .safetensors)
  output/mps_eval/depth_eval.json  per-sample + aggregate metrics for all 3 models
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
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset, _collate_fn

PROMPT = "depth estimation, depth map visualization"


def build_heldout(data_dir: Path, n: int, start: int = 2000):
    imgs = sorted((data_dir / "images").glob("*.jpg")) + sorted(
        (data_dir / "images").glob("*.png")
    )
    items = []
    for p in imgs[start : start + n]:
        tgt = data_dir / "targets" / (p.stem + ".png")
        if tgt.exists():
            items.append((p, tgt))
    return items


@torch.no_grad()
def flux_predict(generator, image: Image.Image, steps: int, strength: float):
    out = generator.generate(
        prompt=PROMPT, image=image,
        strength=strength, num_inference_steps=steps, guidance_scale=0.0,
    )
    rgb = torch.from_numpy(np.array(out[0]).astype(np.float32).transpose(2, 0, 1)) / 255.0
    return rgb.unsqueeze(0)  # [1,3,H,W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell")
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data/vb_snvc_train_v3/depth"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "output/mps_eval"))
    ap.add_argument("--train-steps", type=int, default=80)
    ap.add_argument("--eval-n", type=int, default=24)
    ap.add_argument("--infer-steps", type=int, default=4)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--da-repo", default="depth-anything/Depth-Anything-V2-Small-hf")
    args = ap.parse_args()

    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    mapper = DepthRGBMapper(colormap="cube_edge")

    # ── 1) Train a real FLUX depth LoRA on MPS (fp16, no autocast) ────────
    print(f"[1/4] Loading FLUX on {args.device} (fp16) ...", flush=True)
    t0 = time.time()
    gen = BaseGenerator(model_name=args.model, device=args.device, dtype=torch.float16)
    # Keep small auxiliary models in float32 to avoid fp16 NaN (esp. T5 text encoder).
    for _m in ("text_encoder", "text_encoder_2", "vae"):
        _mod = getattr(gen.pipe, _m, None)
        if _mod is not None:
            _mod.to(torch.float32)
    tuner = InstructionTuner(
        model=gen, task="depth", flow_sigma=1.0, use_img2img_flow=True,
    )
    print(f"      loaded in {time.time()-t0:.1f}s; "
          f"trainable LoRA params = "
          f"{sum(p.numel() for p in tuner.peft_model.parameters() if p.requires_grad)/1e6:.1f}M",
          flush=True)

    ds = VisionTaskDataset(data_path=args.data, resolution=args.res, task="depth")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=_collate_fn,
    )
    opt = torch.optim.AdamW(
        (p for p in tuner.peft_model.parameters() if p.requires_grad), lr=5e-5
    )
    print(f"[2/4] Training {args.train_steps} steps ...", flush=True)
    losses = []
    step = 0
    while step < args.train_steps:
        for batch in loader:
            if step >= args.train_steps:
                break
            opt.zero_grad(set_to_none=True)
            loss = tuner._training_step(batch)
            if not torch.isfinite(loss).all():
                # Skip this batch (don't poison LoRA with a NaN gradient).
                print(f"      step {step:>3}: non-finite loss, skip", flush=True)
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tuner.peft_model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 10 == 0:
                print(f"      step {step:>3}/{args.train_steps}  loss={loss.item():.4f}", flush=True)
    (OUT / "losses.json").write_text(json.dumps(losses))
    lora_dir = OUT / "lora_adapter"
    tuner.peft_model.save_pretrained(str(lora_dir))  # adapter_config.json + .safetensors
    print(f"      LoRA saved -> {lora_dir}", flush=True)

    # ── 2) Held-out samples ──────────────────────────────────────────────
    print(f"[3/4] Evaluating {args.eval_n} held-out samples (LoRA vs base FLUX) ...", flush=True)
    items = build_heldout(Path(args.data), args.eval_n, start=2000)
    print(f"      got {len(items)} held-out samples", flush=True)

    per = {"lora": [], "base": []}
    gts, preds_lora, preds_base = [], [], []

    for i, (img_p, tgt_p) in enumerate(items):
        image = Image.open(img_p).convert("RGB")
        tgt_rgb = torch.from_numpy(
            np.array(Image.open(tgt_p).convert("RGB")).astype(np.float32).transpose(2, 0, 1)
        ) / 255.0
        gt = mapper.rgb_to_depth(tgt_rgb.unsqueeze(0)).squeeze(0)  # [H,W] meters
        gts.append(gt.flatten())

        tuner.peft_model.set_adapter("default")
        tuner.peft_model.enable_adapter_layers()
        d_l = mapper.rgb_to_depth(flux_predict(gen, image, args.infer_steps, args.strength)).squeeze(0)

        tuner.peft_model.disable_adapter_layers()
        d_b = mapper.rgb_to_depth(flux_predict(gen, image, args.infer_steps, args.strength)).squeeze(0)
        tuner.peft_model.enable_adapter_layers()

        preds_lora.append(d_l.flatten())
        preds_base.append(d_b.flatten())
        m_l = evaluate_depth(d_l.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))
        m_b = evaluate_depth(d_b.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))
        per["lora"].append({"image": str(img_p), **m_l})
        per["base"].append({"image": str(img_p), **m_b})
        print(f"      [{i+1}/{len(items)}] LoRA AbsRel={m_l['AbsRel']:.3f} "
              f"base AbsRel={m_b['AbsRel']:.3f} δ1={m_l['δ<1.25']:.3f}", flush=True)

    # ── 3) Depth-Anything zero-shot baseline (local) ─────────────────────
    print(f"[4/4] Depth-Anything zero-shot baseline (local) ...", flush=True)
    per_da, preds_da = [], []
    try:
        from transformers import DepthAnythingForDepthEstimation, AutoImageProcessor
        da = DepthAnythingForDepthEstimation.from_pretrained(
            args.da_repo, local_files_only=True
        ).to(args.device)
        proc = AutoImageProcessor.from_pretrained(args.da_repo, local_files_only=True)
        da.eval()
        for i, (img_p, tgt_p) in enumerate(items):
            image = Image.open(img_p).convert("RGB")
            tgt_rgb = torch.from_numpy(
                np.array(Image.open(tgt_p).convert("RGB")).astype(np.float32).transpose(2, 0, 1)
            ) / 255.0
            gt = mapper.rgb_to_depth(tgt_rgb.unsqueeze(0)).squeeze(0)
            inp = proc(images=image, return_tensors="pt").to(args.device)
            with torch.no_grad():
                pd = da(**inp).predicted_depth
            pd = torch.nn.functional.interpolate(
                pd.unsqueeze(1), size=gt.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1).squeeze(0).cpu()
            m = gt > 0
            scale = gt[m].median() / pd[m].median().clamp(min=1e-6)
            pd = pd * scale
            preds_da.append(pd.flatten())
            m_d = evaluate_depth(pd.unsqueeze(0).unsqueeze(0), gt.unsqueeze(0).unsqueeze(0))
            per_da.append({"image": str(img_p), **m_d})
        print(f"      Depth-Anything evaluated on {len(per_da)} samples", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"      Depth-Anything baseline skipped: {e}", flush=True)

    # ── Aggregate ────────────────────────────────────────────────────────
    def agg(samples):
        out = {}
        for k in ["AbsRel", "SqRel", "RMSE", "RMSE_log", "δ<1.25", "δ<1.25²", "δ<1.25³"]:
            vals = [s[k] for s in samples if k in s]
            if vals:
                out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
        return out

    def pooled(preds, gts):
        P = torch.cat(preds); G = torch.cat(gts)
        m = G > 0
        return evaluate_depth(P[m].unsqueeze(0).unsqueeze(0), G[m].unsqueeze(0).unsqueeze(0))

    result = {
        "n_samples": len(items),
        "losses_first_third_mean": float(np.mean(losses[: len(losses)//3])),
        "losses_last_third_mean": float(np.mean(losses[-len(losses)//3:])),
        "lora": {"per_sample": per["lora"], "aggregate": agg(per["lora"]),
                 "pooled": pooled(preds_lora, gts)},
        "base_flux": {"per_sample": per["base"], "aggregate": agg(per["base"]),
                      "pooled": pooled(preds_base, gts)},
    }
    if per_da:
        result["depth_anything_v2_small"] = {
            "per_sample": per_da, "aggregate": agg(per_da),
            "pooled": pooled(preds_da, gts),
        }

    (OUT / "depth_eval.json").write_text(json.dumps(result, indent=2, default=str))

    def line(name, d):
        a = d.get("pooled", {})
        print(f"  {name:<26} AbsRel={a.get('AbsRel',float('nan')):.4f}  "
              f"RMSE={a.get('RMSE',float('nan')):.4f}  δ1={a.get('δ<1.25',float('nan')):.4f}")
    print("\n" + "=" * 64)
    print("DEPTH EVALUATION — pooled (all pixels)")
    print("=" * 64)
    line("FLUX + LoRA (ours)", result["lora"])
    line("FLUX base (no LoRA)", result["base_flux"])
    if per_da:
        line("Depth-Anything-V2-Small", result["depth_anything_v2_small"])
    print(f"\nFull results -> {OUT / 'depth_eval.json'}")


if __name__ == "__main__":
    main()
