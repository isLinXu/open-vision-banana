#!/usr/bin/env python3
"""Standalone chart generator for depth mapping experiment.

Run from repo root::

    python experiments/generate_charts.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "src" / "open_vision_banana" / "core"

# Set up fake package hierarchy so relative imports work
sys.modules["open_vision_banana"] = types.ModuleType("open_vision_banana")
sys.modules["open_vision_banana.core"] = types.ModuleType("open_vision_banana.core")

spec = importlib.util.spec_from_file_location(
    "open_vision_banana.core.depth_mapper", CORE_DIR / "depth_mapper.py"
)
depth_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_mod)
sys.modules["open_vision_banana.core.depth_mapper"] = depth_mod

spec2 = importlib.util.spec_from_file_location(
    "open_vision_banana.core.adaptive_depth_mapper", CORE_DIR / "adaptive_depth_mapper.py"
)
adaptive_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(adaptive_mod)
sys.modules["open_vision_banana.core.adaptive_depth_mapper"] = adaptive_mod

import torch  # noqa: E402

plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

results_path = EXPERIMENTS_DIR / "depth_mapping_results.json"
with open(results_path, encoding="utf-8") as f:
    data = json.load(f)

strategies = list(data.keys())
colors = ["#2E86AB", "#A23B72", "#F18F01"]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Depth Mapping Algorithm Benchmark", fontsize=14, fontweight="bold", y=1.02)

# 1. Overall Relative Error
ax = axes[0, 0]
vals = [data[s]["roundtrip"]["overall_mean_rel_error"] for s in strategies]
bars = ax.bar(strategies, vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Mean Relative Error")
ax.set_title("1. Overall Roundtrip Accuracy ↓")
ax.set_ylim(0, max(vals) * 1.5 if max(vals) > 0 else 0.01)
for bar, val in zip(bars, vals, strict=False):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(vals) * 0.05,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# 2. Near vs Far Precision
ax = axes[0, 1]
x = np.arange(len(strategies))
width = 0.35
near_vals = [data[s]["roundtrip"]["near_mean_rel_error"] for s in strategies]
far_vals = [data[s]["roundtrip"]["far_mean_rel_error"] for s in strategies]
ax.bar(x - width / 2, near_vals, width, label="Near (0-5m)", color="#2E86AB", edgecolor="black")
ax.bar(x + width / 2, far_vals, width, label="Far (20-100m)", color="#F18F01", edgecolor="black")
ax.set_ylabel("Mean Relative Error")
ax.set_title("2. Near vs Far Precision ↓")
ax.set_xticks(x)
ax.set_xticklabels([s.replace(" (Baseline)", "") for s in strategies], fontsize=8)
ax.legend()

# 3. Speed Comparison
ax = axes[0, 2]
encode_vals = [data[s]["speed"]["encode_ms"] for s in strategies]
decode_vals = [data[s]["speed"]["decode_ms"] for s in strategies]
ax.bar(x - width / 2, encode_vals, width, label="Encode", color="#2E86AB", edgecolor="black")
ax.bar(x + width / 2, decode_vals, width, label="Decode", color="#F18F01", edgecolor="black")
ax.set_ylabel("Time (ms)")
ax.set_title("3. Runtime Speed ↓")
ax.set_xticks(x)
ax.set_xticklabels([s.replace(" (Baseline)", "") for s in strategies], fontsize=8)
ax.legend()

# 4. Visual Quality
ax = axes[1, 0]
var_vals = [data[s]["visual"]["overall_variance"] for s in strategies]
ax.bar(strategies, var_vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("RGB Variance")
ax.set_title("4. Visual Distinguishability ↑")

# 5. Error curve (from live mapper roundtrip)
ax = axes[1, 1]
DepthRGBMapper = depth_mod.DepthRGBMapper
mapper = DepthRGBMapper()
depths = torch.logspace(-1, 2, 200)
rgb = mapper.depth_to_rgb(depths.view(1, 1, 1, -1))
recovered = mapper.rgb_to_depth(rgb).squeeze()
rel_err = ((recovered - depths).abs() / depths.clamp(min=0.1)).numpy()
ax.semilogx(depths.numpy(), rel_err, color="#2E86AB", linewidth=2)
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Relative Error")
ax.set_title("5. Paper Roundtrip Error Curve")
ax.grid(True, alpha=0.3)

# 6. Summary text
ax = axes[1, 2]
ax.axis("off")
fastest = min(strategies, key=lambda s: data[s]["speed"]["total_ms"])
best_near = min(strategies, key=lambda s: data[s]["roundtrip"]["near_mean_rel_error"])
summary = (
    f"Fastest: {fastest}\n"
    f"  total={data[fastest]['speed']['total_ms']:.1f} ms\n\n"
    f"Best near precision: {best_near}\n"
    f"  rel={data[best_near]['roundtrip']['near_mean_rel_error']:.4f}\n\n"
    f"Decode (Paper): {data['Paper (Fixed)']['speed']['decode_ms']:.1f} ms"
)
ax.text(0.05, 0.95, summary, va="top", fontsize=11, family="monospace")

plt.tight_layout()
out_png = EXPERIMENTS_DIR / "depth_mapping_benchmark.png"
out_curve = EXPERIMENTS_DIR / "depth_error_curve.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")

# Separate error curve figure
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.semilogx(depths.numpy(), rel_err, color="#2E86AB", linewidth=2)
ax2.set_xlabel("Depth (m)")
ax2.set_ylabel("Relative Roundtrip Error")
ax2.set_title("Depth Roundtrip Error vs Depth (Paper Method)")
ax2.grid(True, alpha=0.3)
fig2.savefig(out_curve, dpi=150, bbox_inches="tight")
print(f"Saved {out_png}")
print(f"Saved {out_curve}")
