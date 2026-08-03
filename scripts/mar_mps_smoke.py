#!/usr/bin/env python3
"""MAR (jadechoghari/mar) MPS smoke test — class-conditional generation.

Bypasses the repo's CUDA-hardcoded pipeline.py: loads mar_base + kl16 VAE
directly on MPS in fp32, generates 4 ImageNet classes at 256px.

Usage:
  /Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python scripts/mar_mps_smoke.py
"""
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from torchvision.utils import save_image

# smoke hack: redirect .cuda() -> mps (mar.py/diffloss.py hardcode .cuda())
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self.to("mps")

CKPT_DIR = "/Users/gatilin/MyWork/models/ckpts/mar"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "output", "mar_mps_smoke")
sys.path.insert(0, "/Users/gatilin/MyWork/models/ckpts")

from mar.mar import mar_base  # noqa: E402
from mar.vae import AutoencoderKL  # noqa: E402

SEED = 42
LABELS = [207, 360, 388, 113]  # golden retriever, ... (ImageNet ids)
NUM_AR_STEPS = 64
CFG = 4.0


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[env] device={device} torch={torch.__version__}")
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    model = mar_base(buffer_size=64, diffloss_d=6, diffloss_w=1024,
                     num_sampling_steps="100")
    model.load_state_dict(load_file(f"{CKPT_DIR}/mar-base.safetensors"))
    model.to(device).eval()
    print(f"[load] mar_base ok ({time.time()-t0:.1f}s)")

    t0 = time.time()
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                        ckpt_path=f"{CKPT_DIR}/kl16.safetensors").to(device).eval()
    print(f"[load] kl16 vae ok ({time.time()-t0:.1f}s)")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()
    with torch.no_grad():
        tokens = model.sample_tokens(
            bsz=len(LABELS), num_iter=NUM_AR_STEPS,
            cfg=CFG, cfg_schedule="constant",
            labels=torch.tensor(LABELS).long().to(device),
            temperature=1.0, progress=True,
        )
        t_ar = time.time() - t0
        images = vae.decode(tokens / 0.2325)
    print(f"[gen] AR sampling {t_ar:.1f}s, vae decode {time.time()-t0-t_ar:.1f}s")

    out = os.path.join(OUT_DIR, f"mar_base_mps_seed{SEED}.png")
    save_image(images, out, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"[done] saved -> {out}")


if __name__ == "__main__":
    main()
