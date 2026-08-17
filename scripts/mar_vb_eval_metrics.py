#!/usr/bin/env python3
"""MAR-VB paper metrics: decode RGB via DepthRGBMapper, compute δ1/AbsRel/RMSE."""
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

from mar_vb_train import MARVB, DepthPairDataset, VAE_SCALE, DATA_DIR  # noqa: E402
from mar.mar import mar_base  # noqa: E402
from mar.vae import AutoencoderKL  # noqa: E402
from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.evaluation.depth import evaluate_depth  # noqa: E402
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb  # noqa: E402

CKPT_DIR = "/Users/gatilin/MyWork/models/ckpts/mar"
OUT_DIR = os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "mar_vb_depth_v1"))
CKPT = os.environ.get("VB_CKPT", os.path.join(OUT_DIR, "ckpt_best.pt"))
N_EVAL = int(os.environ.get("VB_N_EVAL", "12"))
NUM_ITER = int(os.environ.get("VB_SAMPLE_ITER", "32"))
CFG = float(os.environ.get("VB_CFG", "2.0"))
IMG_SIZE = int(os.environ.get("VB_IMG_SIZE", "256"))  # must match ckpt

device = "mps"
base = mar_base(buffer_size=64, diffloss_d=6, diffloss_w=1024,
                num_sampling_steps="100", img_size=IMG_SIZE)
model = MARVB(base)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(sd["model"])
model.to(device).eval()
print(f"[load] {os.path.basename(CKPT)} step={sd['step']} val_loss={sd['val_loss']:.4f}")

vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                    ckpt_path=f"{CKPT_DIR}/kl16.safetensors").to(device).eval()
mapper = DepthRGBMapper(colormap="turbo")  # v3/v5 depth targets use turbo-like rainbow encoding

stems = sorted({p.stem for p in __import__("pathlib").Path(DATA_DIR, "images").glob("*.jpg")}
               & {p.stem for p in __import__("pathlib").Path(DATA_DIR, "targets").glob("*.png")})
rng = np.random.RandomState(42)
rng.shuffle(stems)
val = stems[:50][:N_EVAL]
dl = DataLoader(DepthPairDataset(DATA_DIR, val), batch_size=4, shuffle=False)

all_metrics, panels = [], []
t0 = time.time()
for i, (im, tg) in enumerate(dl):
    im, tg = im.to(device), tg.to(device)
    with torch.no_grad():
        ctx = vae.encode(im).sample().mul_(VAE_SCALE)
        lat = model.sample_vb(ctx, torch.zeros(im.size(0), dtype=torch.long, device=device),
                              num_iter=NUM_ITER, cfg=CFG)
        pred = vae.decode(lat / VAE_SCALE).clamp(-1, 1)
    pred01 = ((pred + 1) / 2).cpu()
    gt01 = ((tg + 1) / 2).cpu()
    pred_depth = mapper.rgb_to_depth(pred01)
    gt_depth = mapper.rgb_to_depth(gt01)
    panels.append(torch.cat([im.cpu(), gt01 * 2 - 1, pred.cpu()], dim=0))
    for b in range(pred01.size(0)):  # per-sample metrics
        pd, gd = pred_depth[b:b+1], gt_depth[b:b+1]
        valid = (gd < 9.9).squeeze(1)  # exclude colormap saturation zone (λc=10m)
        raw = evaluate_depth(pd, gd, valid_mask=valid)
        scale = (gd.squeeze(1)[valid].median() /
                 pd.squeeze(1)[valid].median().clamp(min=1e-6))  # median align
        ali = evaluate_depth(pd * scale, gd, valid_mask=valid)
        rel = evaluate_relative_depth_rgb(pred01[b:b+1], gt01[b:b+1], colormap="turbo")
        all_metrics.append({
            "AbsRel_raw": raw["AbsRel"], "δ1_raw": raw["δ<1.25"],
            "AbsRel_aligned": ali["AbsRel"], "δ1_aligned": ali["δ<1.25"],
            "RMSE_aligned": ali["RMSE"],
            "pearson": rel["relative_position_pearson_r"],
            "spearman": rel["relative_position_spearman_r"],
            "rgb_mae": rel["rgb_mae"],
        })
    m = all_metrics[-1]
    print(f"[batch {i+1}] last-sample AbsRel_al={m['AbsRel_aligned']:.3f} "
          f"δ1_al={m['δ1_aligned']:.3f} pearson={m['pearson']:.3f} ({(time.time()-t0)/60:.1f}min)")

mean = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
save_image(torch.cat(panels, dim=0), os.path.join(OUT_DIR, "panel_metrics_eval.png"),
           nrow=4, normalize=True, value_range=(-1, 1))
print("\n=== MEAN over", len(all_metrics), "samples ===")
for k, v in mean.items():
    print(f"  {k:16s} = {v:.4f}")
import json
json.dump({"ckpt": os.path.basename(CKPT), "step": sd["step"], "n": len(all_metrics),
           "num_iter": NUM_ITER, "cfg": CFG, "mean": mean},
          open(os.path.join(OUT_DIR, "paper_metrics.json"), "w"), indent=2)
