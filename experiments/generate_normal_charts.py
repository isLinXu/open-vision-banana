#!/usr/bin/env python3
"""Chart generator for normal mapping experiment."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent

plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

results_path = EXPERIMENTS_DIR / "normal_mapping_results.json"
with open(results_path, encoding="utf-8") as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Normal Mapping Roundtrip & Robustness Benchmark", fontsize=14, fontweight="bold", y=1.02)

# 1. Roundtrip Accuracy (cosine similarity)
ax = axes[0, 0]
patterns = ["random", "gradient", "spherical"]
cos_vals = [data["roundtrip"][p]["mean_cosine_sim"] for p in patterns]
colors = ["#2E86AB", "#A23B72", "#F18F01"]
bars = ax.bar(patterns, cos_vals, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Cosine Similarity")
ax.set_title("1. Roundtrip Accuracy ↑")
ax.set_ylim(0.98, 1.001)
for bar, val in zip(bars, cos_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.0005,
        f"{val:.6f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# 2. Noise Robustness (cosine sim)
ax = axes[0, 1]
noise_data = data["noise_robustness"]
sigmas = [d["noise_sigma"] for d in noise_data]
cos_sims = [d["mean_cosine_sim"] for d in noise_data]
ax.plot(sigmas, cos_sims, "o-", color="#2E86AB", linewidth=2, markersize=6, markeredgecolor="black")
ax.set_xlabel("RGB Noise σ")
ax.set_ylabel("Cosine Similarity")
ax.set_title("2. Noise Robustness (Cosine Sim) ↑")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
for s, c in zip(sigmas, cos_sims):
    ax.text(s, c + 0.03, f"{c:.3f}", ha="center", fontsize=8)

# 3. Noise Robustness (angular error)
ax = axes[1, 0]
angles = [d["mean_angular_error_deg"] for d in noise_data]
ax.plot(sigmas, angles, "s-", color="#A23B72", linewidth=2, markersize=6, markeredgecolor="black")
ax.set_xlabel("RGB Noise σ")
ax.set_ylabel("Angular Error (°)")
ax.set_title("3. Noise Robustness (Angular Error) ↓")
ax.grid(True, alpha=0.3)
for s, a in zip(sigmas, angles):
    ax.text(s, a + 2, f"{a:.1f}°", ha="center", fontsize=8)

# 4. Summary & Speed
ax = axes[1, 1]
ax.axis("off")
speed = data["speed"]
edge_passed = sum(1 for e in data["edge_cases"] if e["passed"])
edge_total = len(data["edge_cases"])
summary = (
    f"NormalRGBMapper Summary\n"
    f"{'─' * 30}\n\n"
    f"Roundtrip (random):\n"
    f"  cos_sim = {data['roundtrip']['random']['mean_cosine_sim']:.6f}\n"
    f"  angular_err = {data['roundtrip']['random']['mean_angular_error_deg']:.3f}°\n\n"
    f"Flip consistency:\n"
    f"  x-accuracy = {data['flip_consistency']['x_component_flip_accuracy']:.4f}\n\n"
    f"Edge cases: {edge_passed}/{edge_total} passed\n\n"
    f"Speed (512×512):\n"
    f"  encode: {speed['encode_ms']:.1f} ms\n"
    f"  decode: {speed['decode_ms']:.1f} ms\n"
    f"  total: {speed['total_ms']:.1f} ms"
)
ax.text(0.05, 0.95, summary, va="top", fontsize=11, family="monospace")

plt.tight_layout()
out_png = EXPERIMENTS_DIR / "normal_mapping_benchmark.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved {out_png}")

# Separate noise curve figure
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(sigmas, cos_sims, "o-", color="#2E86AB", linewidth=2, markersize=6, markeredgecolor="black")
ax2.set_xlabel("RGB Noise σ")
ax2.set_ylabel("Cosine Similarity")
ax2.set_title("Normal Mapping Noise Robustness Curve")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)
out_curve = EXPERIMENTS_DIR / "normal_noise_curve.png"
fig2.savefig(out_curve, dpi=150, bbox_inches="tight")
print(f"Saved {out_curve}")
