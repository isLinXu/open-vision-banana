#!/usr/bin/env python3
"""GPU smoke test for the real FLUX img2img-LoRA depth training (verification gate 2 + 3-image viz).

This is the ACTUAL end-to-end smoke that exercises InstructionTuner._flow_training_step
on a tiny depth subset, then produces the 3-panel reconstruction viz
[input | target depth RGB | predicted depth RGB -> decoded depth].

It reuses the real APIs from scripts/train.py:
  - BaseGenerator(model_name, device, dtype)
  - InstructionTuner(model=gen, task="depth", flow_sigma=..., use_img2img_flow=True)
  - tuner.train(dataset, max_steps=..., batch_size=..., ...)
  - generator.generate(prompt, image=..., strength=0.85)

REQUIRES: CUDA GPU + diffusers + a prepared depth dataset from scripts/prepare_hypersim.py.
Run:
  python scripts/smoke_train_depth.py --data-path data/prepared/hypersim --max-steps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.config import VisionBananaConfig  # noqa: E402
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.training.tuner import InstructionTuner  # noqa: E402
from open_vision_banana.data.datasets import build_dataset  # noqa: E402
from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402


def get_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        name, torch.bfloat16
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--data-path", required=True, help="Prepared depth dir (prepare_hypersim.py output)")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[FATAL] CUDA not available — this smoke requires a GPU. Use scripts/smoke_depth_pipeline.py (GPU-free) instead.")

    cfg = VisionBananaConfig.from_yaml(args.config)
    device = args.device
    dtype = get_dtype(cfg.model.dtype)

    # 1) generator (loads FLUX.1-dev)
    generator = BaseGenerator(model_name=cfg.model.name, device=device, dtype=dtype)

    # 2) tuner (img2img flow, flow_sigma from config)
    tuner = InstructionTuner(
        model=generator,
        lora_rank=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        use_rslora=cfg.lora.rs_lora,
        target_modules=cfg.lora.target_modules,
        use_img2img_flow=True,
        flow_sigma=cfg.training.flow_sigma,
        task="depth",
        use_ema=cfg.training.use_ema,
        ema_decay=cfg.training.ema_decay,
    )

    # 3) tiny depth subset
    full = build_dataset("depth", args.data_path, resolution=args.resolution)
    if len(full) == 0:
        sys.exit(f"[FATAL] no depth samples at {args.data_path}")
    n = min(args.max_steps * args.batch_size * 2, len(full))
    subset = Subset(full, list(range(n)))
    print(f"[data] using {len(subset)} samples for smoke")

    # 4) a few real training steps through the img2img flow branch
    print(f"[train] running {args.max_steps} optimizer steps (img2img flow)...")
    tuner.train(subset, epochs=1, batch_size=args.batch_size,
                max_steps=args.max_steps, save_every=10**9, eval_every=10**9,
                use_amp=True, num_workers=0)

    # 5) 3-image reconstruction viz on one sample
    mapper = DepthRGBMapper()
    sample = full[0]
    inp = sample["image"]  # tensor [3,H,W] in [-1,1] or PIL
    prompt = sample["prompt"]
    target_rgb = sample["target_rgb"]  # [3,H,W] in [0,1]

    if isinstance(inp, torch.Tensor):
        gen_in = inp
    else:
        import numpy as np
        from PIL import Image
        arr = np.asarray(inp.convert("RGB")).astype(np.float32) / 255.0
        gen_in = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1

    with torch.no_grad():
        pred_rgb = generator.generate(prompt=prompt, image=gen_in, strength=0.85, seed=0)  # [1,3,H,W] [0,1]
    pred_depth = mapper.rgb_to_depth(pred_rgb).squeeze(0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    disp_in = (gen_in.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
    axes[0].imshow(disp_in)
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[1].imshow(target_rgb.permute(1, 2, 0).numpy() if torch.is_tensor(target_rgb) else target_rgb)
    axes[1].set_title("Target depth RGB\n(GT encoded)")
    axes[1].axis("off")
    im = axes[2].imshow(pred_depth, cmap="magma")
    axes[2].set_title("Predicted depth\n(LoRA after smoke steps)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="meters")
    out = ROOT / "output" / "smoke_train_depth.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[saved] {out}")
    print("[OK] real img2img-LoRA depth smoke complete. Inspect output/smoke_train_depth.png.")


if __name__ == "__main__":
    main()
