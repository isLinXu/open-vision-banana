#!/usr/bin/env python3
"""GPU-free smoke test for the depth data pipeline (verification gate 1).

Purpose
-------
Before spending GPU hours, confirm the **data + colormap roundtrip** that the
whole reproduction stands on is sane:
  input image  ->  (GT depth in meters)  ->  DepthRGBMapper.depth_to_rgb
               ->  model training target (RGB)  ->  rgb_to_depth  ->  recovered depth

This uses the REAL ``DepthRGBMapper`` (default colormap = cube_edge, lam=-3, c=10/3)
and confirms train-time encoding == eval-time decoding (gate 1).

Outputs
-------
  output/smoke_depth_pipeline.png   : 3-panel [Input | Target RGB depth | Decoded depth]
  stdout                            : mean relative roundtrip error

Run (no GPU, no diffusers required):
  python scripts/smoke_depth_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

# make the package importable when run directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402


def find_sample_image() -> Path | None:
    """Pick any PNG from the repo to act as a stand-in input image."""
    for pat in ["output/inference_specular/*.png", "output/**/*.png", "data/**/*.png"]:
        hits = sorted(ROOT.glob(pat))
        if hits:
            return hits[0]
    return None


def synth_metric_depth(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A plausible smooth metric-depth map in meters (0..~12 m)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # radial ramp (near -> far) + a couple of gaussian "objects"
    cy, cx = h * 0.55, w * 0.45
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / max(h, w)
    depth = 1.0 + 11.0 * r
    for _ in range(3):
        ay, ax = rng.integers(0, h), rng.integers(0, w)
        sy, sx = rng.integers(h // 8, h // 3), rng.integers(w // 8, w // 3)
        g = np.exp(-(((yy - ay) ** 2) / (2 * sy**2) + ((xx - ax) ** 2) / (2 * sx**2)))
        depth = depth - 3.0 * g  # closer objects
    depth = np.clip(depth, 0.2, 25.0)
    return depth.astype(np.float32)


def main() -> None:
    mapper = DepthRGBMapper()  # default cube_edge, lam=-3, c=10/3

    # --- input image (stand-in; any repo PNG) ---
    img_path = find_sample_image()
    if img_path is not None:
        img = Image.open(img_path).convert("RGB").resize((256, 256))
        np_img = np.asarray(img)
        print(f"[input] using stand-in image: {img_path.name}")
    else:
        rng = np.random.default_rng(1)
        np_img = (rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
        print("[input] no sample image found; using random RGB")

    # --- ground-truth metric depth (meters) ---
    H = W = 256
    gt_depth = synth_metric_depth(H, W)

    # --- encode: depth(meters) -> RGB (this is the model's training target) ---
    depth_t = torch.from_numpy(gt_depth)[None, None, ...]  # [1,1,H,W]
    target_rgb = mapper.depth_to_rgb(depth_t).squeeze(0).permute(1, 2, 0).numpy()  # [H,W,3]

    # --- decode: RGB -> depth (eval-time inverse) ---
    target_rgb_t = torch.from_numpy(target_rgb).permute(2, 0, 1)[None, ...]  # [1,3,H,W]
    rec_depth = mapper.rgb_to_depth(target_rgb_t).squeeze().numpy()  # [H,W]

    # --- roundtrip error (relative, over valid GT>0) ---
    mask = gt_depth > 0
    rel = np.abs(rec_depth[mask] - gt_depth[mask]) / np.clip(gt_depth[mask], 1e-3, None)
    mean_rel = float(rel.mean())
    print(f"[roundtrip] mean relative error = {mean_rel:.4%}  (cube_edge, lam=-3, c=10/3)")

    # --- 3-panel visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(np_img)
    axes[0].set_title("Input image\n(stand-in)")
    axes[0].axis("off")

    axes[1].imshow(target_rgb)
    axes[1].set_title("Target depth RGB\n(model training target)")
    axes[1].axis("off")

    im = axes[2].imshow(rec_depth, cmap="magma")
    axes[2].set_title(f"Decoded depth\n(mean rel err {mean_rel:.2%})")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="meters")

    fig.suptitle("Depth pipeline smoke (GPU-free) — cube_edge colormap roundtrip", fontsize=12)
    out_path = ROOT / "output" / "smoke_depth_pipeline.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_path}")

    assert mean_rel < 0.01, "colormap roundtrip error too high — check cube_edge impl"
    print("[OK] verification gate 1 (colormap train==decode) consistent.")


if __name__ == "__main__":
    main()
