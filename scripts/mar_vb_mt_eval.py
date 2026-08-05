#!/usr/bin/env python3
"""MAR-VB multitask eval: rgb_mae per task + paper metrics (depth)."""
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

if not torch.cuda.is_available() and torch.backends.mps.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self.to("mps")

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
sys.path.insert(0, "/Users/gatilin/MyWork/models/ckpts")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mar_vb_train import MARVB, TaskPairDataset, VAE_SCALE, DATA_ROOT  # noqa: E402
from mar.mar import mar_base  # noqa: E402
from mar.vae import AutoencoderKL  # noqa: E402
from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.evaluation.depth import evaluate_depth  # noqa: E402
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb  # noqa: E402

CKPT_DIR = "/Users/gatilin/MyWork/models/ckpts/mar"
OUT_DIR = os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "mar_vb_mt_v1"))
CKPT = os.environ.get("VB_CKPT", os.path.join(OUT_DIR, "ckpt_best.pt"))
TASKS = os.environ.get("VB_TASKS", "depth,normal,segmentation").split(",")
N_EVAL = int(os.environ.get("VB_N_EVAL", "8"))
NUM_ITER = int(os.environ.get("VB_SAMPLE_ITER", "32"))
CFG = float(os.environ.get("VB_CFG", "2.0"))

device = "mps"
base = mar_base(buffer_size=64, diffloss_d=6, diffloss_w=1024, num_sampling_steps="100")
model = MARVB(base)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(sd["model"])
model.to(device).eval()
print(f"[load] {os.path.basename(CKPT)} step={sd['step']} val_loss={sd['val_loss']:.4f}")

vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                    ckpt_path=f"{CKPT_DIR}/kl16.safetensors").to(device).eval()
mapper = DepthRGBMapper(colormap="cube_edge")

t0 = time.time()
summary, all_panels = {}, []
for tid, task in enumerate(TASKS):
    root = PROJECT_ROOT / "data" / "vb_snvc_train_v3" / task
    imgs = {p.stem for p in (root / "images").glob("*.jpg")}
    tgts = {p.stem for p in (root / "targets").glob("*.png")}
    stems = sorted(imgs & tgts)
    rng = np.random.RandomState(42)
    rng.shuffle(stems)
    val = stems[:50][:N_EVAL]
    dl = DataLoader(TaskPairDataset(str(root), val, tid), batch_size=4, shuffle=False)

    maes, metrics_list, panels = [], [], []
    for im, tg, _ in dl:
        im, tg = im.to(device), tg.to(device)
        with torch.no_grad():
            ctx = vae.encode(im).sample().mul_(VAE_SCALE)
            lat = model.sample_vb(ctx, torch.full((im.size(0),), tid, device=device, dtype=torch.long),
                                  num_iter=NUM_ITER, cfg=CFG)
            pred = vae.decode(lat / VAE_SCALE).clamp(-1, 1)
        pred01 = ((pred + 1) / 2).cpu()
        gt01 = ((tg + 1) / 2).cpu()
        for b in range(pred01.size(0)):
            mae = (pred01[b] - gt01[b]).abs().mean().item()
            maes.append(mae)
            if task == "depth":  # paper metrics only for depth
                pd = mapper.rgb_to_depth(pred01[b:b+1])
                gd = mapper.rgb_to_depth(gt01[b:b+1])
                valid = (gd < 9.9).squeeze(1)
                m = evaluate_depth(pd, gd, valid_mask=valid)
                rel = evaluate_relative_depth_rgb(pred01[b:b+1], gt01[b:b+1], colormap="cube_edge")
                metrics_list.append({**m, **{f"rel_{k}": v for k, v in rel.items()}})
        panels.append(torch.cat([im.cpu(), gt01 * 2 - 1, pred.cpu()], dim=0))
    summary[task] = {
        "rgb_mae_mean": float(np.mean(maes)),
        "rgb_mae_std": float(np.std(maes)),
        "n": len(maes),
    }
    if metrics_list:
        for k in metrics_list[0]:
            summary[task][k] = float(np.mean([m[k] for m in metrics_list]))
    all_panels.append((task, torch.cat(panels, dim=0)))
    print(f"[{task}] rgb_mae={summary[task]['rgb_mae_mean']:.4f}±{summary[task]['rgb_mae_std']:.4f} ({(time.time()-t0)/60:.1f}min)")

for task, p in all_panels:
    save_image(p, os.path.join(OUT_DIR, f"panel_final_{task}.png"), nrow=4,
               normalize=True, value_range=(-1, 1))
import json
json.dump({"ckpt": os.path.basename(CKPT), "step": sd["step"], "num_iter": NUM_ITER,
           "cfg": CFG, "summary": summary},
          open(os.path.join(OUT_DIR, "multitask_metrics.json"), "w"), indent=2)
print("\n=== SUMMARY ===")
for t, s in summary.items():
    extra = f" δ1={s.get('δ<1.25',0):.3f} spearman={s.get('rel_relative_position_spearman_r',0):.3f}" if 'δ<1.25' in s else ""
    print(f"  {t:14s} rgb_mae={s['rgb_mae_mean']:.4f}{extra}")
