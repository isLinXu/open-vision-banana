#!/usr/bin/env python3
"""End-to-end pipeline benchmark for Vision Banana.

Measures full inference latency, throughput, and VRAM for the complete pipeline
(input → prompt → FLUX generation → RGB decoding → task output).

Usage:
    # Standard benchmark (single image, all tasks)
    python scripts/benchmark_pipeline.py --model black-forest-labs/FLUX.1-schnell

    # Benchmark with optimization preset
    python scripts/benchmark_pipeline.py --profile balanced --device cuda

    # Batch throughput test
    python scripts/benchmark_pipeline.py --batch-size 4 --num-samples 20

    # Memory profiling
    python scripts/benchmark_pipeline.py --profile memory --measure-vram
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from open_vision_banana.inference.optimization import (
    InferenceOptions,
    estimate_vram_savings,
    options_from_dict,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PRESET_PATH = Path(__file__).parent.parent / "configs" / "optimization_presets.yaml"


@dataclass
class BenchmarkResult:
    """Structured benchmark result for a single configuration."""

    task: str
    preset: str
    device: str
    batch_size: int
    resolution: int
    num_inference_steps: int
    median_latency_ms: float
    mean_latency_ms: float
    std_latency_ms: float
    p95_latency_ms: float
    throughput_img_per_sec: float
    peak_vram_mb: float | None = None
    preset_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["preset_notes"] = dict(self.preset_notes)
        return d


def _load_preset(name: str) -> InferenceOptions:
    """Load a preset from YAML."""
    try:
        import yaml
    except ImportError:
        return InferenceOptions()

    with open(PRESET_PATH) as f:
        data = yaml.safe_load(f)
    profiles = data.get("profiles", {})
    if name not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return options_from_dict(profiles[name])


def measure_vram_peak() -> float | None:
    """Return peak VRAM in MB, or None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_vram_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def benchmark_task(
    model,
    task: str,
    resolution: int,
    batch_size: int,
    num_samples: int,
    warmup: int,
    device: str,
) -> list[float]:
    """Run inference multiple times and collect latencies."""
    times: list[float] = []

    # Create synthetic input
    dummy_input = torch.randn(batch_size, 3, resolution, resolution, device=device)

    # Warmup
    logger.info("Warming up (%d iterations)...", warmup)
    for _ in range(warmup):
        try:
            with torch.no_grad():
                _ = model(dummy_input, task=task)
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    logger.info("Benchmarking %d iterations...", num_samples)
    for _ in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                _ = model(dummy_input, task=task)
        except Exception as exc:
            logger.warning("Inference error during benchmark: %s", exc)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return times


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    """Run benchmarks across tasks and configurations."""
    results: list[BenchmarkResult] = []
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolution = args.resolution
    batch_size = args.batch_size
    num_samples = args.num_samples
    warmup = args.warmup

    # Try to load the model
    try:
        from open_vision_banana import VisionBanana
    except ImportError:
        logger.error("Cannot import VisionBanana. Ensure package is installed: pip install -e .")
        return []

    logger.info("Loading model: %s", args.model)
    try:
        model = VisionBanana.from_pretrained(args.model)
        model = model.to(device)
    except Exception as exc:
        logger.warning("Could not load full model (%s). Running mapper-only benchmark.", exc)
        return _run_mapper_only_benchmark(args, device)

    tasks = args.tasks.split(",") if args.tasks else ["depth", "normal"]
    presets = args.presets.split(",") if args.presets else ["balanced"]

    for preset_name in presets:
        logger.info("=" * 60)
        logger.info("Benchmarking preset: %s", preset_name)
        logger.info("=" * 60)

        try:
            options = _load_preset(preset_name)
        except ValueError as exc:
            logger.warning("%s", exc)
            continue

        notes = estimate_vram_savings(options)

        for task in tasks:
            logger.info("  Task: %s", task)
            reset_vram_peak()

            times = benchmark_task(
                model, task, resolution, batch_size, num_samples, warmup, device
            )
            if not times:
                continue

            times_t = torch.tensor(times)
            peak_vram = measure_vram_peak()

            result = BenchmarkResult(
                task=task,
                preset=preset_name,
                device=device,
                batch_size=batch_size,
                resolution=resolution,
                num_inference_steps=args.num_inference_steps,
                median_latency_ms=times_t.median().item(),
                mean_latency_ms=times_t.mean().item(),
                std_latency_ms=times_t.std().item(),
                p95_latency_ms=times_t.quantile(0.95).item(),
                throughput_img_per_sec=(batch_size * 1000.0) / times_t.median().item(),
                peak_vram_mb=peak_vram,
                preset_notes=notes,
            )
            results.append(result)

            logger.info(
                "    median=%.1f ms, p95=%.1f ms, throughput=%.2f img/s, vram=%s MB",
                result.median_latency_ms,
                result.p95_latency_ms,
                result.throughput_img_per_sec,
                f"{peak_vram:.0f}" if peak_vram else "N/A",
            )

    return results


def _run_mapper_only_benchmark(
    args: argparse.Namespace, device: str
) -> list[BenchmarkResult]:
    """Fallback: benchmark only the mapper/decoding pipeline (no FLUX)."""
    from open_vision_banana.core import DepthRGBMapper, NormalRGBMapper, SegmentationDecoder

    results: list[BenchmarkResult] = []
    resolution = args.resolution
    batch_size = args.batch_size
    num_samples = args.num_samples

    mappers = {
        "depth": DepthRGBMapper(),
        "normal": NormalRGBMapper(),
    }
    decoder = SegmentationDecoder()

    for task, mapper in mappers.items():
        logger.info("  Mapper benchmark: %s", task)
        reset_vram_peak()

        if task == "depth":
            x = torch.rand(batch_size, 1, resolution, resolution, device=device) * 50.0
        else:
            x = torch.randn(batch_size, 3, resolution, resolution, device=device)
            x = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = mapper.encode(x) if hasattr(mapper, "encode") else mapper(x)

        times = []
        for _ in range(num_samples):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                if task == "depth":
                    rgb = mapper.depth_to_rgb(x)
                    _ = mapper.rgb_to_depth(rgb)
                else:
                    rgb = mapper.normal_to_rgb(x)
                    _ = mapper.rgb_to_normal(rgb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        times_t = torch.tensor(times)
        peak_vram = measure_vram_peak()

        result = BenchmarkResult(
            task=task,
            preset="mapper_only",
            device=device,
            batch_size=batch_size,
            resolution=resolution,
            num_inference_steps=0,
            median_latency_ms=times_t.median().item(),
            mean_latency_ms=times_t.mean().item(),
            std_latency_ms=times_t.std().item(),
            p95_latency_ms=times_t.quantile(0.95).item(),
            throughput_img_per_sec=(batch_size * 1000.0) / times_t.median().item(),
            peak_vram_mb=peak_vram,
        )
        results.append(result)

        logger.info(
            "    median=%.2f ms, p95=%.2f ms, throughput=%.2f img/s",
            result.median_latency_ms,
            result.p95_latency_ms,
            result.throughput_img_per_sec,
        )

    # Segmentation decoder benchmark
    logger.info("  Mapper benchmark: segmentation")
    reset_vram_peak()
    seg_rgb = torch.rand(batch_size, 3, 128, 128, device=device)
    times = []
    for _ in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = decoder.decode_instance(seg_rgb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times_t = torch.tensor(times)
    peak_vram = measure_vram_peak()
    result = BenchmarkResult(
        task="segmentation",
        preset="mapper_only",
        device=device,
        batch_size=batch_size,
        resolution=128,
        num_inference_steps=0,
        median_latency_ms=times_t.median().item(),
        mean_latency_ms=times_t.mean().item(),
        std_latency_ms=times_t.std().item(),
        p95_latency_ms=times_t.quantile(0.95).item(),
        throughput_img_per_sec=(batch_size * 1000.0) / times_t.median().item(),
        peak_vram_mb=peak_vram,
    )
    results.append(result)
    logger.info(
        "    median=%.2f ms, p95=%.2f ms, throughput=%.2f img/s",
        result.median_latency_ms,
        result.p95_latency_ms,
        result.throughput_img_per_sec,
    )

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("  PIPELINE BENCHMARK SUMMARY")
    print("=" * 80)
    print(
        f"  {'Task':<15} {'Preset':<12} {'Device':<8} {'Median(ms)':<12} {'P95(ms)':<10} "
        f"{'Throughput':<12} {'VRAM(MB)':<10}"
    )
    print("  " + "-" * 78)
    for r in results:
        vram = f"{r.peak_vram_mb:.0f}" if r.peak_vram_mb else "N/A"
        print(
            f"  {r.task:<15} {r.preset:<12} {r.device:<8} {r.median_latency_ms:<12.1f} "
            f"{r.p95_latency_ms:<10.1f} {r.throughput_img_per_sec:<12.2f} {vram:<10}"
        )
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline benchmark for Vision Banana"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-schnell",
        help="Model identifier for full pipeline benchmark",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default="balanced",
        help="Comma-separated preset names (balanced,speed,memory,quality,cpu)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="depth,normal",
        help="Comma-separated task names",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON output path for structured results",
    )
    parser.add_argument(
        "--mapper-only",
        action="store_true",
        help="Benchmark only mapper/decoding (no FLUX model load)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mapper_only:
        # Force mapper-only mode
        args.model = "none"

    results = run_benchmark(args)
    if not results:
        logger.error("No benchmark results collected.")
        return 1

    print_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("Results saved to: %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
