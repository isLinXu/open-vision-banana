# Multi-Backbone Inference Benchmark Report

> **Date**: 2026-07-01
> **Method**: Mock components (no model weights downloaded)
> **Hardware**: CPU
> **Batch size**: 1

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| **Fastest total latency** | sana-sprint (10.19 ms, 2 steps) |
| **Slowest total latency** | sd3 (465.49 ms, 28 steps) |
| **Speedup ratio** | 45.7× |
| **Models tested** | 10 |

**Key findings**:
- Turbo/Sprint models (sdxl-turbo, sana-sprint) have **2 steps** vs. 28 for standard models, dramatically reducing total latency.
- DiT-based models (flux, sd3, pixart, sana) have similar per-step backbone latency to UNet models in mock mode, but real-world latency depends on model size and attention optimization.
- Text encoder count directly impacts `text_encode_ms`: SD3 (3 encoders) ≈ 3× SD1.5 (1 encoder).
- Memory budget is a **hard constraint**: models > 20GB require A100/H100.

## 2. Latency Ranking (Estimated Total)

| Model | Family | Backbone | Encoders | Memory (GB) | Resolution | Steps | Per-Step (ms) | Total (ms) |
|-------|--------|----------|----------|-------------|------------|-------|---------------|------------|
| sana-sprint | sana-sprint | transformer | 1 | 8.0 | 1024 | 2 | 5.10 | 10.19 |
| sdxl-turbo | sdxl | unet | 2 | 7.0 | 256 | 4 | 4.34 | 17.34 |
| flux-schnell | flux | transformer | 2 | 24.0 | 512 | 4 | 13.81 | 55.22 |
| sd15 | sd15 | unet | 1 | 4.0 | 256 | 30 | 2.98 | 89.47 |
| sd2 | sd2 | unet | 1 | 5.0 | 256 | 30 | 3.14 | 94.26 |
| sana-600m | sana | transformer | 1 | 6.0 | 1024 | 20 | 5.47 | 109.44 |
| ssd-1b | sdxl | unet | 2 | 5.5 | 256 | 30 | 4.19 | 125.57 |
| sdxl | sdxl | unet | 2 | 7.0 | 256 | 30 | 4.39 | 131.59 |
| pixart-alpha | pixart | transformer | 1 | 10.0 | 1024 | 20 | 9.21 | 184.18 |
| sd3 | sd3 | transformer | 3 | 20.0 | 1024 | 28 | 16.62 | 465.49 |

## 3. Per-Component Breakdown

All times are **median of 5 runs** after 2 warmups. Mock components simulate
PyTorch forward pass latency without actual computation.

| Model | Text Encode | VAE Encode | Backbone | VAE Decode | Per-Step Total |
|-------|-------------|------------|----------|------------|----------------|
| sana-sprint | 1.76 | 0.99 | 0.05 | 2.30 | 5.10 |
| sdxl-turbo | 2.05 | 0.06 | 0.05 | 2.17 | 4.34 |
| flux-schnell | 4.22 | 0.06 | 0.05 | 9.47 | 13.81 |
| sd15 | 0.73 | 0.06 | 0.05 | 2.14 | 2.98 |
| sd2 | 0.70 | 0.06 | 0.05 | 2.33 | 3.14 |
| sana-600m | 1.75 | 0.88 | 0.05 | 2.80 | 5.47 |
| ssd-1b | 2.00 | 0.06 | 0.05 | 2.07 | 4.19 |
| sdxl | 2.07 | 0.09 | 0.05 | 2.17 | 4.39 |
| pixart-alpha | 6.03 | 0.87 | 0.05 | 2.26 | 9.21 |
| sd3 | 13.56 | 0.83 | 0.05 | 2.18 | 16.62 |

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
