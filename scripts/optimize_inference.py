#!/usr/bin/env python3
"""Apply and benchmark inference optimizations for Vision Banana.

Reads preset profiles from configs/optimization_presets.yaml and provides
a unified CLI for selecting optimization strategies based on hardware constraints.

Usage:
    # Show available profiles and VRAM impact
    python scripts/optimize_inference.py --list-presets

    # Use a preset profile
    python scripts/optimize_inference.py --profile balanced

    # Custom options (override preset)
    python scripts/optimize_inference.py --profile memory --torch-compile

    # Benchmark mapper throughput with preset dtype
    python scripts/optimize_inference.py --profile speed --benchmark-mappers

    # Apply to an actual inference run
    python scripts/optimize_inference.py --profile balanced --run \
        --image data/sample.jpg --task depth
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from open_vision_banana.inference.optimization import (
    InferenceOptions,
    estimate_vram_savings,
    options_from_dict,
    quantize_mapper_int8,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PRESET_PATH = Path(__file__).parent.parent / "configs" / "optimization_presets.yaml"


def load_presets(path: Path = PRESET_PATH) -> dict[str, dict[str, Any]]:
    """Load optimization presets from YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("profiles", {})


def list_presets() -> None:
    """Print available preset profiles with VRAM impact estimates."""
    presets = load_presets()
    if not presets:
        print("No presets found in", PRESET_PATH)
        return

    print(f"Available profiles ({len(presets)} total):")
    print("-" * 60)

    for name, cfg in presets.items():
        opts = options_from_dict(cfg)
        notes = estimate_vram_savings(opts)
        print(f"\n  {name}:")
        for k, v in cfg.items():
            print(f"    {k}: {v}")
        if notes:
            print("  Expected impact:")
            for key, note in notes.items():
                print(f"    {key}: {note}")

    print("\nUsage: python scripts/optimize_inference.py --profile <name>")


def options_from_preset(name: str) -> InferenceOptions:
    """Build InferenceOptions from a named preset."""
    presets = load_presets()
    if name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return options_from_dict(presets[name])


def _benchmark_mapper(model: torch.nn.Module, x: torch.Tensor, iters: int) -> dict[str, float]:
    """Benchmark a mapper, returning timing statistics."""
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        times = []
        for _ in range(iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    times_t = torch.tensor(times)
    return {
        "mean_ms": times_t.mean().item(),
        "std_ms": times_t.std().item(),
        "p95_ms": times_t.quantile(0.95).item(),
        "min_ms": times_t.min().item(),
    }


def benchmark_mappers(options: InferenceOptions, resolution: int = 512, iters: int = 100) -> None:
    """Benchmark depth/normal mappers with the preset's dtype configuration."""
    from open_vision_banana.core import DepthRGBMapper, NormalRGBMapper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    depth = torch.rand(1, 1, resolution, resolution, device=device) * 100.0
    normal = torch.randn(1, 3, resolution, resolution, device=device)
    normal = normal / normal.norm(dim=1, keepdim=True).clamp(min=1e-8)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(options.dtype, torch.float32)

    depth = depth.to(dtype)
    normal = normal.to(dtype)

    d_mapper = DepthRGBMapper()
    n_mapper = NormalRGBMapper()

    if options.quantize_mappers_int8:
        d_mapper = quantize_mapper_int8(d_mapper)
        n_mapper = quantize_mapper_int8(n_mapper)
        print(f"\nMapper benchmark ({resolution}×{resolution}, INT8, {iters} iters):")
    else:
        print(f"\nMapper benchmark ({resolution}×{resolution}, {options.dtype}, {iters} iters):")

    d_stats = _benchmark_mapper(d_mapper, depth, iters)
    n_stats = _benchmark_mapper(n_mapper, normal, iters)

    print(f"  Depth encode:  {d_stats['mean_ms']:.3f} ± {d_stats['std_ms']:.3f} ms/iter")
    print(f"  Normal encode: {n_stats['mean_ms']:.3f} ± {n_stats['std_ms']:.3f} ms/iter")

    # Roundtrip accuracy
    with torch.no_grad():
        rgb_d = d_mapper.depth_to_rgb(depth)
        rec_d = d_mapper.rgb_to_depth(rgb_d)
        near_err = (depth[depth < 10.0] - rec_d[depth < 10.0]).abs().mean().item()
        print(f"  Near-depth roundtrip MAE (<10m): {near_err:.4f} m")

        rgb_n = n_mapper.normal_to_rgb(normal)
        rec_n = n_mapper.rgb_to_normal(rgb_n)
        cos_sim = (normal * rec_n).sum(dim=1).mean().item()
        ang_err = torch.acos(torch.tensor(cos_sim).clamp(-1, 1)) * 180 / 3.14159265
        print(f"  Normal roundtrip angular error: {ang_err:.2f}°")


def run_inference(options: InferenceOptions, image: str, task: str) -> None:
    """Run inference with applied optimizations."""
    from open_vision_banana import VisionBanana

    logger.info("Loading model with optimizations...")
    model = VisionBanana.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        optimization_options=options,
    )
    result = model.predict(image, task=task)
    logger.info("Inference complete. Result shape: %s", result.shape if hasattr(result, "shape") else "N/A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vision Banana inference optimization tool"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available preset profiles",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Preset profile name (e.g. balanced, speed, memory, quality, cpu)",
    )
    parser.add_argument(
        "--preset-file",
        type=str,
        default=None,
        help="Custom preset YAML file path",
    )
    # Override flags
    parser.add_argument("--cpu-offload", action="store_true", help="Enable sequential CPU offload")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--attention-slicing", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16", "int8"])
    parser.add_argument("--int8-mappers", action="store_true", help="Quantize mappers to INT8")
    # Actions
    parser.add_argument("--benchmark-mappers", action="store_true", help="Benchmark mapper throughput")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--run", action="store_true", help="Run actual inference")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--task", type=str, default="depth")
    return parser.parse_args()


def build_options(args: argparse.Namespace) -> InferenceOptions:
    """Build options from preset + CLI overrides."""
    if args.preset_file:
        presets = load_presets(Path(args.preset_file))
        if args.profile:
            opts = options_from_dict(presets.get(args.profile, {}))
        else:
            opts = InferenceOptions()
    elif args.profile:
        opts = options_from_preset(args.profile)
    else:
        opts = InferenceOptions()

    # CLI overrides
    if args.cpu_offload:
        opts.enable_sequential_cpu_offload = True
    if args.vae_slicing:
        opts.enable_vae_slicing = True
    if args.vae_tiling:
        opts.enable_vae_tiling = True
    if args.attention_slicing:
        opts.enable_attention_slicing = True
    if args.torch_compile:
        opts.enable_torch_compile = True
    if args.dtype:
        opts.dtype = args.dtype
    if args.int8_mappers:
        opts.quantize_mappers_int8 = True

    return opts


def main() -> int:
    args = parse_args()

    if args.list_presets:
        list_presets()
        return 0

    options = build_options(args)

    print("=" * 60)
    print("Vision Banana Inference Optimization")
    print("=" * 60)
    print(f"Profile: {args.profile or 'custom'}")
    print("\nActive configuration:")
    for k, v in options.__dict__.items():
        if v and v is not False:
            print(f"  {k}: {v}")

    notes = estimate_vram_savings(options)
    if notes:
        print("\nExpected VRAM impact:")
        for key, note in notes.items():
            print(f"  {key}: {note}")

    if args.benchmark_mappers:
        benchmark_mappers(options, resolution=args.resolution, iters=args.iterations)

    if args.run:
        if not args.image:
            logger.error("--run requires --image")
            return 1
        run_inference(options, args.image, args.task)

    return 0


if __name__ == "__main__":
    sys.exit(main())
