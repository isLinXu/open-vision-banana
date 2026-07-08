#!/usr/bin/env python3
"""Benchmarking and profiling utilities for Vision Banana.

Measures throughput, latency, and memory usage for core operations:
- Depth/Normal RGB mapping encode/decode
- Segmentation decoding
- End-to-end pipeline (when model available)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --resolution 1024 --iterations 100
"""

import argparse
import json
import time
from pathlib import Path

import torch

from open_vision_banana import DepthRGBMapper, NormalRGBMapper, RGBMapper, SegmentationDecoder


def benchmark_fn(fn, *args, warmup=5, iterations=50, label=""):
    """Benchmark a function, returning timing statistics."""
    # Warmup
    for _ in range(warmup):
        fn(*args)

    # Timed runs
    times = []
    for _ in range(iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = torch.tensor(times)
    stats = {
        "mean_ms": times.mean().item() * 1000,
        "std_ms": times.std().item() * 1000,
        "median_ms": times.median().item() * 1000,
        "p95_ms": times.quantile(0.95).item() * 1000,
        "min_ms": times.min().item() * 1000,
        "max_ms": times.max().item() * 1000,
    }

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        for k, v in stats.items():
            print(f"  {k:>12s}: {v:8.2f}")

    return stats, result


def measure_memory(fn, *args, label=""):
    """Measure peak GPU memory usage."""
    if not torch.cuda.is_available():
        return None

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    fn(*args)

    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    if label:
        print(f"  Peak GPU memory ({label}): {peak_mb:.1f} MB")

    return peak_mb


def main():
    parser = argparse.ArgumentParser(description="Vision Banana Benchmarks")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON output path for structured benchmark results",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    res = args.resolution
    iters = args.iterations

    print("Vision Banana Benchmark Suite")
    print(f"Resolution: {res}×{res}, Device: {device}, Iterations: {iters}")

    # ── Depth Mapping ──
    mapper_d = DepthRGBMapper()
    depth = torch.rand(1, 1, res, res, device=device) * 50.0

    stats_enc, rgb_d = benchmark_fn(
        mapper_d.depth_to_rgb, depth,
        iterations=iters, label="Depth → RGB Encoding",
    )
    mem_enc = measure_memory(mapper_d.depth_to_rgb, depth, label="depth_enc")

    stats_dec, _ = benchmark_fn(
        mapper_d.rgb_to_depth, rgb_d,
        iterations=iters, label="RGB → Depth Decoding",
    )
    measure_memory(mapper_d.rgb_to_depth, rgb_d, label="depth_dec")

    # Roundtrip accuracy
    recovered = mapper_d.rgb_to_depth(rgb_d)
    near_mask = depth < 10.0
    if near_mask.any():
        near_err = (depth[near_mask] - recovered[near_mask]).abs().mean().item()
        print(f"\n  Near-depth roundtrip MAE (<10m): {near_err:.4f} m")

    # ── Normal Mapping ──
    mapper_n = NormalRGBMapper()
    normal = torch.randn(1, 3, res, res, device=device)
    normal = normal / normal.norm(dim=1, keepdim=True).clamp(min=1e-8)

    stats_n_enc, rgb_n = benchmark_fn(
        mapper_n.normal_to_rgb, normal,
        iterations=iters, label="Normal → RGB Encoding",
    )

    stats_n_dec, recovered_n = benchmark_fn(
        mapper_n.rgb_to_normal, rgb_n,
        iterations=iters, label="RGB → Normal Decoding",
    )

    # Roundtrip angular error
    cos_sim = (normal * recovered_n).sum(dim=1).mean().item()
    ang_err = torch.acos(torch.tensor(cos_sim).clamp(-1, 1)) * 180 / 3.14159265
    print(f"\n  Normal roundtrip angular error: {ang_err:.2f}°")

    # ── Segmentation Decoding ──
    decoder = SegmentationDecoder()
    seg_rgb = torch.rand(1, 3, res, res, device=device)

    stats_seg, _ = benchmark_fn(
        decoder.decode_instance, seg_rgb,
        iterations=min(iters, 10),  # Fewer iters for heavier op
        label="Instance Segmentation Decoding",
    )

    # ── Unified RGBMapper ──
    mapper = RGBMapper()

    stats_u_enc, _ = benchmark_fn(
        mapper.encode, depth, "depth",
        iterations=iters, label="Unified Mapper: encode(depth)",
    )

    stats_u_dec, _ = benchmark_fn(
        mapper.decode, rgb_d, "depth",
        iterations=iters, label="Unified Mapper: decode(depth)",
    )

    # ── Summary Table ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY (median latency at {res}×{res})")
    print(f"{'='*60}")
    print(f"  {'Operation':<35s} {'Median (ms)':>12s} {'P95 (ms)':>12s}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")

    results: dict[str, dict[str, float]] = {}
    for label, stats in [
        ("Depth encode", stats_enc),
        ("Depth decode", stats_dec),
        ("Normal encode", stats_n_enc),
        ("Normal decode", stats_n_dec),
        ("Instance decode", stats_seg),
        ("Unified encode(depth)", stats_u_enc),
        ("Unified decode(depth)", stats_u_dec),
    ]:
        print(f"  {label:<35s} {stats['median_ms']:>12.2f} {stats['p95_ms']:>12.2f}")
        results[label] = {
            "median_ms": stats["median_ms"],
            "p95_ms": stats["p95_ms"],
            "mean_ms": stats["mean_ms"],
            "std_ms": stats["std_ms"],
            "min_ms": stats["min_ms"],
            "max_ms": stats["max_ms"],
        }

    if mem_enc is not None:
        print(f"\n  Peak GPU memory: {mem_enc:.1f} MB (depth encode)")
        results["_meta"] = {
            "peak_gpu_memory_mb": mem_enc,
            "resolution": res,
            "device": device,
            "iterations": iters,
        }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
