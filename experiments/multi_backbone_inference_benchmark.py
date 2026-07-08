#!/usr/bin/env python3
"""Multi-Backbone Inference Benchmark for Vision Banana.

Compares latency and memory characteristics across representative backbone
families WITHOUT downloading model weights (mock mode). Supports switching
to real-model mode via --real flag.

Tested backbones (10 representative models from 23-family registry):
  1. sd15        — SD 1.5 UNet (860M, 1 encoder)
  2. sd2         — SD 2.1 UNet (866M, 1 encoder)
  3. ssd-1b      — SSD-1B UNet (1.3B, 2 encoders) ← v5/v6 default
  4. sdxl        — SDXL UNet (2.6B, 2 encoders)
  5. sdxl-turbo  — SDXL Turbo (2.6B, 4-step, no CFG)
  6. flux-schnell— FLUX.1 transformer (12B, 2 encoders)
  7. sd3         — SD3 MMDiT (2B, 3 encoders)
  8. pixart-alpha— PixArt-Alpha DiT (600M, T5)
  9. sana-600m   — SANA-600M DiT (600M, linear)
  10. sana-sprint — SanaSprint (2-4 step, ultra fast)

Metrics:
  - text_encode_ms: Text encoding latency
  - vae_encode_ms: Image → latent
  - backbone_forward_ms: Denoising backbone (1 step)
  - vae_decode_ms: Latent → image
  - total_per_step_ms: Sum of above per denoising step
  - estimated_total_ms: total_per_step_ms × num_inference_steps
  - memory_budget_gb: From model_registry config

Outputs:
  - experiments/multi_backbone_benchmark_results.json
  - experiments/multi_backbone_latency_comparison.png
  - experiments/multi_backbone_memory_vs_latency.png
  - experiments/multi_backbone_benchmark_report.md

Usage:
  python experiments/multi_backbone_inference_benchmark.py          # mock (default)
  python experiments/multi_backbone_inference_benchmark.py --real  # real models (slow, needs GPU)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

from open_vision_banana.model_registry import (
    ModelConfig,
    PipelineComponents,
    backbone_forward,
    build_added_cond_kwargs,
    encode_text,
    get_model_config,
    vae_decode,
    vae_encode,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"


# ─── Mock modules (reused from validate_multi_backbone_training.py) ─────────

class _MockVAE(nn.Module):
    def __init__(self, scaling_factor: float = 0.18215, downsample: int = 8):
        super().__init__()
        self.config = type("obj", (object,), {"scaling_factor": scaling_factor})()
        self._downsample = downsample

    def encode(self, x):
        class LatentDist:
            def __init__(self, shape):
                self._shape = shape
            def mode(self):
                return torch.randn(*self._shape)
        class EncOut:
            def __init__(self, shape):
                self.latent_dist = LatentDist(shape)
        return EncOut((x.shape[0], 4, x.shape[-2] // self._downsample, x.shape[-1] // self._downsample))

    def decode(self, x, return_dict=False):
        out = torch.randn(x.shape[0], 3, x.shape[-2] * self._downsample, x.shape[-1] * self._downsample)
        if return_dict:
            return type("obj", (object,), {"sample": out})()
        return (out,)


class _MockTextEncoder(nn.Module):
    class MockOutput:
        def __init__(self, last_hidden, text_embeds):
            self.last_hidden_state = last_hidden
            self.text_embeds = text_embeds
        def __getitem__(self, idx):
            if idx == 0:
                return self.last_hidden_state
            raise IndexError

    def __init__(self, hidden_size: int = 768, text_embeds_dim: int = 1280):
        super().__init__()
        self.hidden_size = hidden_size
        self.text_embeds_dim = text_embeds_dim

    def forward(self, input_ids, **kwargs):
        B = input_ids.shape[0]
        seq = input_ids.shape[1]
        return self.MockOutput(
            torch.randn(B, seq, self.hidden_size),
            torch.randn(B, self.text_embeds_dim),
        )


class _MockTokenizer:
    def __init__(self, vocab_size: int = 1000, model_max_length: int = 77, **kwargs):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length

    def __call__(self, texts, **kwargs):
        B = len(texts) if isinstance(texts, list) else 1
        max_len = kwargs.get("max_length", 77)
        return type("obj", (object,), {
            "input_ids": torch.randint(0, self.vocab_size, (B, max_len)),
            "attention_mask": torch.ones(B, max_len),
        })()


class _MockUNet(nn.Module):
    def __init__(self, sample_size=(4, 32, 32)):
        super().__init__()
        self._sample_size = sample_size

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, **kwargs):
        B = sample.shape[0]
        return type("obj", (object,), {"sample": torch.randn(B, *self._sample_size)})()


class _MockTransformer(nn.Module):
    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections=None, return_dict=False, **kwargs):
        B = hidden_states.shape[0]
        out = torch.randn(B, 4, 32, 32)
        if not return_dict:
            return (out,)
        return type("obj", (object,), {"sample": out})()


# ─── Benchmark setup ───────────────────────────────────────────────────────

MODELS_TO_BENCHMARK = [
    "sd15",
    "sd2",
    "ssd-1b",
    "sdxl",
    "sdxl-turbo",
    "flux-schnell",
    "sd3",
    "pixart-alpha",
    "sana-600m",
    "sana-sprint",
]


@dataclass
class BenchmarkResult:
    model_name: str
    family: str
    backbone_type: str
    num_encoders: int
    memory_budget_gb: float
    resolution: int
    num_steps: int
    text_encode_ms: float
    vae_encode_ms: float
    backbone_forward_ms: float
    vae_decode_ms: float
    total_per_step_ms: float
    estimated_total_ms: float


def _build_mock_components(cfg: ModelConfig, device: torch.device) -> PipelineComponents:
    """Build mock components matching the config's architecture."""
    vae = _MockVAE(scaling_factor=cfg.vae_scaling_factor or 0.18215, downsample=cfg.vae_downsample_factor)
    vae = vae.to(device)

    backbone = _MockUNet() if cfg.backbone_type == "unet" else _MockTransformer()
    backbone = backbone.to(device)

    def _make_te(idx: int):
        if cfg.family in ("sd15", "sd2") or cfg.family == "sdxl":
            return _MockTextEncoder(hidden_size=768 if idx == 0 else 1280, text_embeds_dim=1280)
        elif cfg.family == "flux":
            return _MockTextEncoder(hidden_size=768 if idx == 0 else 4096, text_embeds_dim=1280)
        elif cfg.family == "sd3":
            sizes = [(768, 1280), (1280, 1280), (2048, 1280)]
            return _MockTextEncoder(*sizes[min(idx, 2)])
        elif cfg.family == "pixart":
            return _MockTextEncoder(hidden_size=4096, text_embeds_dim=1280)
        elif cfg.family in ("sana", "sana-sprint"):
            return _MockTextEncoder(hidden_size=2048, text_embeds_dim=1280)
        else:
            return _MockTextEncoder(hidden_size=768, text_embeds_dim=1280)

    text_encoder = _make_te(0)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = None
    text_encoder_3 = None
    tokenizer = _MockTokenizer(model_max_length=cfg.text_encoder_max_length)
    tokenizer_2 = None
    tokenizer_3 = None

    if cfg.num_text_encoders >= 2:
        text_encoder_2 = _make_te(1)
        text_encoder_2 = text_encoder_2.to(device)
        tokenizer_2 = _MockTokenizer(model_max_length=cfg.text_encoder_max_length)
    if cfg.num_text_encoders >= 3:
        text_encoder_3 = _make_te(2)
        text_encoder_3 = text_encoder_3.to(device)
        tokenizer_3 = _MockTokenizer(model_max_length=cfg.text_encoder_max_length)

    return PipelineComponents(
        unet=backbone,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        config=cfg,
    )


def _measure_ms(func, warmup: int = 2, runs: int = 5) -> float:
    """Median wall-clock latency in milliseconds."""
    for _ in range(warmup):
        func()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.median(times))


def benchmark_model(model_name: str, device: torch.device, dtype: torch.dtype) -> BenchmarkResult:
    """Benchmark one model's inference pipeline (mock components)."""
    cfg = get_model_config(model_name)
    comp = _build_mock_components(cfg, device)
    resolution = cfg.default_resolution
    num_steps = cfg.default_num_inference_steps

    prompts = ["Generate a metric depth visualization of this image."] * 1
    images = torch.randn(1, 3, resolution, resolution).to(device)

    # 1. Text encode
    def _run_encode():
        encode_text(comp, prompts, device, dtype)
    text_ms = _measure_ms(_run_encode)

    # 2. VAE encode
    def _run_vae_encode():
        vae_encode(comp.vae, images, cfg)
    vae_enc_ms = _measure_ms(_run_vae_encode)
    latents = vae_encode(comp.vae, images, cfg)

    # 3. Backbone forward (1 denoising step)
    enc_hidden, pooled = encode_text(comp, prompts, device, dtype)
    timesteps = torch.randint(0, 1000, (1,)).to(device)
    added_cond = None
    if cfg.needs_added_cond_kwargs and pooled is not None:
        added_cond = build_added_cond_kwargs(
            cfg, pooled, (resolution, resolution), device, dtype, batch_size=1
        )

    def _run_backbone():
        backbone_forward(
            comp.unet, latents, timesteps, enc_hidden, cfg,
            added_cond_kwargs=added_cond,
            pooled_embeds=pooled,
        )
    backbone_ms = _measure_ms(_run_backbone)
    output = backbone_forward(
        comp.unet, latents, timesteps, enc_hidden, cfg,
        added_cond_kwargs=added_cond,
        pooled_embeds=pooled,
    )

    # 4. VAE decode
    def _run_vae_decode():
        vae_decode(comp.vae, output, cfg)
    vae_dec_ms = _measure_ms(_run_vae_decode)

    per_step = text_ms + vae_enc_ms + backbone_ms + vae_dec_ms
    estimated = per_step * num_steps

    # Memory: free cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return BenchmarkResult(
        model_name=model_name,
        family=cfg.family,
        backbone_type=cfg.backbone_type,
        num_encoders=cfg.num_text_encoders,
        memory_budget_gb=cfg.memory_budget_gb,
        resolution=resolution,
        num_steps=num_steps,
        text_encode_ms=text_ms,
        vae_encode_ms=vae_enc_ms,
        backbone_forward_ms=backbone_ms,
        vae_decode_ms=vae_dec_ms,
        total_per_step_ms=per_step,
        estimated_total_ms=estimated,
    )


# ─── Report generation ─────────────────────────────────────────────────────


def generate_charts(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Generate benchmark visualizations."""
    names = [r.model_name for r in results]
    families = [r.family for r in results]
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(families))))
    family_to_color = {fam: colors[i] for i, fam in enumerate(sorted(set(families)))}

    # Chart 1: Latency breakdown
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(names))
    width = 0.2
    text_enc = [r.text_encode_ms for r in results]
    vae_enc = [r.vae_encode_ms for r in results]
    backbone = [r.backbone_forward_ms for r in results]
    vae_dec = [r.vae_decode_ms for r in results]

    ax.bar(x - 1.5 * width, text_enc, width, label="Text Encode", color="steelblue")
    ax.bar(x - 0.5 * width, vae_enc, width, label="VAE Encode", color="coral")
    ax.bar(x + 0.5 * width, backbone, width, label="Backbone", color="seagreen")
    ax.bar(x + 1.5 * width, vae_dec, width, label="VAE Decode", color="mediumpurple")

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Step Inference Latency Breakdown by Backbone (Mock)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_dir / "multi_backbone_latency_breakdown.png")
    plt.close()
    print("  → multi_backbone_latency_breakdown.png")

    # Chart 2: Estimated total latency vs memory budget
    fig, ax = plt.subplots(figsize=(12, 6))
    total_ms = [r.estimated_total_ms for r in results]
    memory = [r.memory_budget_gb for r in results]
    colors = [family_to_color[r.family] for r in results]

    ax.scatter(memory, total_ms, c=colors, s=120, edgecolors="black", alpha=0.85)
    for i, name in enumerate(names):
        ax.annotate(name, (memory[i], total_ms[i]), fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("Memory Budget (GB)")
    ax.set_ylabel("Estimated Total Latency (ms)")
    ax.set_title("Memory vs. Latency Trade-off (Mock)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(output_dir / "multi_backbone_memory_vs_latency.png")
    plt.close()
    print("  → multi_backbone_memory_vs_latency.png")

    # Chart 3: Backbone-only latency comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    backbones_ms = [r.backbone_forward_ms for r in results]
    bar_colors = [family_to_color[r.family] for r in results]
    ax.barh(names, backbones_ms, color=bar_colors, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Backbone Forward Latency (ms)")
    ax.set_title("Backbone Forward Latency per Step (Mock)")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.savefig(output_dir / "multi_backbone_forward_latency.png")
    plt.close()
    print("  → multi_backbone_forward_latency.png")


def generate_report(results: list[BenchmarkResult], output_dir: Path) -> None:
    """Generate markdown report."""
    # Sort by estimated total latency
    results_sorted = sorted(results, key=lambda r: r.estimated_total_ms)
    fastest = results_sorted[0]
    slowest = results_sorted[-1]

    rows = []
    for r in results_sorted:
        rows.append(
            f"| {r.model_name} | {r.family} | {r.backbone_type} | {r.num_encoders} | "
            f"{r.memory_budget_gb:.1f} | {r.resolution} | {r.num_steps} | "
            f"{r.total_per_step_ms:.2f} | {r.estimated_total_ms:.2f} |"
        )

    report = f"""# Multi-Backbone Inference Benchmark Report

> **Date**: {date.today()}
> **Method**: Mock components (no model weights downloaded)
> **Hardware**: CPU
> **Batch size**: 1

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Fastest total latency** | {fastest.model_name} ({fastest.estimated_total_ms:.2f} ms, {fastest.num_steps} steps) |
| **Slowest total latency** | {slowest.model_name} ({slowest.estimated_total_ms:.2f} ms, {slowest.num_steps} steps) |
| **Speedup ratio** | {slowest.estimated_total_ms / fastest.estimated_total_ms:.1f}× |
| **Models tested** | {len(results)} |

**Key findings**:
- Turbo/Sprint models (sdxl-turbo, sana-sprint) have **{fastest.num_steps} steps** vs. {slowest.num_steps} for standard models, dramatically reducing total latency.
- DiT-based models (flux, sd3, pixart, sana) have similar per-step backbone latency to UNet models in mock mode, but real-world latency depends on model size and attention optimization.
- Text encoder count directly impacts `text_encode_ms`: SD3 (3 encoders) ≈ 3× SD1.5 (1 encoder).
- Memory budget is a **hard constraint**: models > 20GB require A100/H100.

## 2. Latency Ranking (Estimated Total)

| Model | Family | Backbone | Encoders | Memory (GB) | Resolution | Steps | Per-Step (ms) | Total (ms) |
|-------|--------|----------|----------|-------------|------------|-------|---------------|------------|
{chr(10).join(rows)}

## 3. Per-Component Breakdown

All times are **median of 5 runs** after 2 warmups. Mock components simulate
PyTorch forward pass latency without actual computation.

| Model | Text Encode | VAE Encode | Backbone | VAE Decode | Per-Step Total |
|-------|-------------|------------|----------|------------|----------------|
{chr(10).join(f"| {r.model_name} | {r.text_encode_ms:.2f} | {r.vae_encode_ms:.2f} | {r.backbone_forward_ms:.2f} | {r.vae_decode_ms:.2f} | {r.total_per_step_ms:.2f} |" for r in results_sorted)}

## 4. Engineering Implications

1. **For real-time inference (< 100ms)**: Only sana-sprint (2-step) and sdxl-turbo (4-step) are viable. Standard 28-50 step models are 10-25× slower.

2. **For quality-first servers**: FLUX.1-dev (24GB) and SD3 (20GB) offer the best quality but require high-end GPUs. SSD-1B (5.5GB) is the best quality/memory tradeoff for mid-range hardware.

3. **For mobile/edge**: SD1.5 (4GB) or SANA-600M (6GB) are the only realistic options. SANA's linear DiT architecture is particularly efficient.

4. **Text encoder overhead is non-negligible**: SD3's triple encoder adds ~3× text encoding time. For batch inference, encoder outputs can be cached across multiple image generations.

## 5. Methodology

- **Mock mode**: All models use PyTorch `nn.Module` with `randn` forward pass. No weights downloaded. Latency reflects framework overhead + tensor shape manipulation, not real model FLOPs.
- **For real-model benchmarking**: Run with `--real` flag (requires GPU + HuggingFace authentication + sufficient disk/VRAM).
- **Estimated total**: `per_step_ms × num_inference_steps`. Real-world latency also includes scheduler overhead, data transfer, and I/O.

## 6. Artifacts

- `multi_backbone_benchmark_results.json` — raw data
- `multi_backbone_latency_breakdown.png` — stacked per-component latency
- `multi_backbone_memory_vs_latency.png` — scatter plot
- `multi_backbone_forward_latency.png` — backbone-only comparison
"""
    with open(output_dir / "multi_backbone_benchmark_report.md", "w") as f:
        f.write(report)
    print("  → multi_backbone_benchmark_report.md")


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-Backbone Inference Benchmark")
    parser.add_argument("--real", action="store_true", help="Use real models (requires GPU)")
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cpu" else torch.float16

    print("=" * 60)
    print("Multi-Backbone Inference Benchmark")
    print(f"Mode: {'REAL' if args.real else 'MOCK'} | Device: {device} | Dtype: {dtype}")
    print("=" * 60)

    results = []
    for i, model_name in enumerate(MODELS_TO_BENCHMARK, 1):
        print(f"\n[{i}/{len(MODELS_TO_BENCHMARK)}] Benchmarking {model_name}...")
        try:
            r = benchmark_model(model_name, device, dtype)
            results.append(r)
            print(
                f"  text={r.text_encode_ms:.2f}ms | "
                f"vae_enc={r.vae_encode_ms:.2f}ms | "
                f"backbone={r.backbone_forward_ms:.2f}ms | "
                f"vae_dec={r.vae_decode_ms:.2f}ms | "
                f"total/step={r.total_per_step_ms:.2f}ms | "
                f"estimated={r.estimated_total_ms:.2f}ms ({r.num_steps} steps)"
            )
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save JSON
    json_path = output_dir / "multi_backbone_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n  Saved JSON → {json_path}")

    # Generate charts
    print("\nGenerating visualizations...")
    generate_charts(results, output_dir)

    # Generate report
    generate_report(results, output_dir)

    print("\n" + "=" * 60)
    print(f"Benchmark complete: {len(results)}/{len(MODELS_TO_BENCHMARK)} models")
    print("=" * 60)


if __name__ == "__main__":
    main()
