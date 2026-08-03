#!/usr/bin/env python3
"""Standalone MAR-VB sampling from a trained checkpoint (debug/eval)."""
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchvision.utils import save_image

if not torch.cuda.is_available() and torch.backends.mps.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self.to("mps")

sys.path.insert(0, "/Users/gatilin/MyWork/models/ckpts")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from mar_vb_train import MARVB, DepthPairDataset, VAE_SCALE, DATA_DIR  # noqa: E402
from mar.mar import mar_base  # noqa: E402
from mar.vae import AutoencoderKL  # noqa: E402

CKPT_DIR = "/Users/gatilin/MyWork/models/ckpts/mar"
CKPT = os.environ.get("VB_CKPT",
    "/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722/output/mar_vb_smoke/ckpt_best.pt")
OUT = os.environ.get("VB_OUT",
    "/Users/gatilin/PycharmProjects/open-vision-banana-git-v260722/output/mar_vb_smoke")
NUM_ITER = int(os.environ.get("VB_SAMPLE_ITER", "32"))
CFG = float(os.environ.get("VB_CFG", "2.0"))

device = "mps"
base = mar_base(buffer_size=64, diffloss_d=6, diffloss_w=1024, num_sampling_steps="100")
model = MARVB(base)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(sd["model"])
model.to(device).eval()
print(f"[load] ckpt step={sd['step']} val_loss={sd['val_loss']:.4f}")

vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                    ckpt_path=f"{CKPT_DIR}/kl16.safetensors").to(device).eval()

stems = sorted({p.stem for p in (__import__("pathlib").Path(DATA_DIR) / "images").glob("*.jpg")}
               & {p.stem for p in (__import__("pathlib").Path(DATA_DIR) / "targets").glob("*.png")})
rng = np.random.RandomState(42)
rng.shuffle(stems)
val = stems[:50]
dl = DataLoader(DepthPairDataset(DATA_DIR, val), batch_size=4, shuffle=False)
im, tg = next(iter(dl))
im, tg = im.to(device), tg.to(device)

t0 = time.time()
with torch.no_grad():
    ctx = vae.encode(im).sample().mul_(VAE_SCALE)
    lat = model.sample_vb(ctx, torch.zeros(im.size(0), dtype=torch.long, device=device),
                          num_iter=NUM_ITER, cfg=CFG)
    pred = vae.decode(lat / VAE_SCALE).clamp(-1, 1)
mae = ((pred + 1) / 2 - (tg + 1) / 2).abs().mean().item()
panel = torch.cat([im, tg, pred], dim=0)
out = os.path.join(OUT, "panel_standalone.png")
save_image(panel, out, nrow=4, normalize=True, value_range=(-1, 1))
print(f"[done] {time.time()-t0:.1f}s rgb_mae={mae:.4f} -> {out}")
