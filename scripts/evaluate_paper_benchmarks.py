#!/usr/bin/env python3
"""Evaluate decoded-task metrics on the paper-aligned benchmarks.

Examples:
    python scripts/evaluate_paper_benchmarks.py \
        --model-name black-forest-labs/FLUX.1-schnell \
        --lora-path output/checkpoint-best \
        --depth-root /data/benchmarks \
        --normal-root /data/benchmarks \
        --cityscapes-root /data/cityscapes \
        --benchmark all

The command requires local benchmark data and model weights. It never falls
back to synthetic data or RGB reconstruction metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data import build_dataset  # noqa: E402
from open_vision_banana.evaluation.evaluator import VisionEvaluator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paper_benchmarks")


DEPTH_DATASETS = {
    "nyu": "nyu_depth",
    "ibims": "ibims",
    "eth3d": "eth3d",
    "diode": "diode",
    "kitti": "kitti",
}
NORMAL_DATASETS = {
    "nyu": "nyu_normal",
    "diode": "diode_normal",
    "scannet": "normal",
    "vkitti": "vkitti_normal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["depth", "normal", "semantic", "all"], default="all")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--depth-root", type=Path, default=None)
    parser.add_argument("--normal-root", type=Path, default=None)
    parser.add_argument("--cityscapes-root", type=Path, default=None)
    parser.add_argument("--nyu-root", type=Path, default=None)
    parser.add_argument("--ibims-root", type=Path, default=None)
    parser.add_argument("--eth3d-root", type=Path, default=None)
    parser.add_argument("--diode-root", type=Path, default=None)
    parser.add_argument("--kitti-root", type=Path, default=None)
    parser.add_argument("--scannet-root", type=Path, default=None)
    parser.add_argument("--vkitti-root", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def resolve_device(value: str | None) -> str:
    if value:
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(value: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def fingerprint_dataset(dataset: Any) -> str:
    """Hash ordered image/target paths without hashing large image contents."""
    digest = hashlib.sha256()
    for sample in getattr(dataset, "samples", []):
        for key in ("image", "target"):
            value = sample.get(key)
            if value is not None:
                path = Path(value)
                digest.update(str(path.resolve()).encode())
                if path.exists():
                    stat = path.stat()
                    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def select_dataset(dataset: Any, max_samples: int | None) -> Any:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


def build_jobs(args: argparse.Namespace) -> list[tuple[str, str, Path, str]]:
    jobs: list[tuple[str, str, Path, str]] = []
    if args.benchmark in ("depth", "all"):
        depth_roots = {
            "nyu": args.nyu_root or args.depth_root,
            "ibims": args.ibims_root or args.depth_root,
            "eth3d": args.eth3d_root or args.depth_root,
            "diode": args.diode_root or args.depth_root,
            "kitti": args.kitti_root or args.depth_root,
        }
        for name, task in DEPTH_DATASETS.items():
            if depth_roots[name] is not None:
                jobs.append(("depth", name, depth_roots[name], task))
    if args.benchmark in ("normal", "all"):
        normal_roots = {
            "nyu": args.nyu_root or args.normal_root,
            "diode": args.diode_root or args.normal_root,
            "scannet": args.scannet_root or args.normal_root,
            "vkitti": args.vkitti_root or args.normal_root,
        }
        for name, task in NORMAL_DATASETS.items():
            if normal_roots[name] is not None:
                jobs.append(("normal", name, normal_roots[name], task))
    if args.benchmark in ("semantic", "all") and args.cityscapes_root is not None:
        jobs.append(("semantic", "cityscapes", args.cityscapes_root, "semantic"))
    return jobs


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    jobs = build_jobs(args)
    if not jobs:
        raise SystemExit("No benchmark roots supplied")

    from open_vision_banana.inference.pipeline import VisionBanana

    model = VisionBanana.from_pretrained(
        model_name=args.model_name,
        lora_path=args.lora_path,
        device=device,
        dtype=resolve_dtype(args.dtype),
    )
    evaluator = VisionEvaluator(model, device=device)
    results: dict[str, Any] = {
        "metadata": {
            "evaluation_mode": "decoded_task_metrics",
            "protocol": {
                "depth": "decoded RGB to metric depth; valid GT pixels only; NYU/KITTI crop when resolution permits",
                "normal": "decoded RGB to unit camera-space normals; valid unit-vector pixels only",
                "semantic": "decoded RGB to class labels; Cityscapes labelIds mapped to trainIds; global confusion matrix",
                "aggregation": "pixel-weighted across the complete selected dataset, not batch mean",
            },
            "model_name": args.model_name,
            "lora_path": str(args.lora_path) if args.lora_path else None,
            "device": device,
            "dtype": args.dtype,
            "resolution": args.resolution,
            "batch_size": args.batch_size,
            "num_inference_steps": args.num_inference_steps,
            "strength": args.strength,
            "seed": args.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "benchmarks": {},
    }

    for benchmark, name, root, dataset_key in jobs:
        split = "val" if benchmark != "semantic" else args.split
        dataset = build_dataset(dataset_key, str(root), split=split, resolution=args.resolution)
        if len(dataset) == 0:
            raise RuntimeError(f"Empty benchmark dataset: {name} ({root})")
        selected = select_dataset(dataset, args.max_samples)
        loader = DataLoader(
            selected,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers if device == "cuda" else 0,
        )
        class_color_map = None
        if benchmark == "semantic":
            from open_vision_banana.data.datasets import CityscapesSegDataset
            class_color_map = CityscapesSegDataset.CITYSCAPES_COLORS
        metrics = evaluator.evaluate(
            loader,
            task=benchmark,
            num_classes=19 if benchmark == "semantic" else None,
            class_color_map=class_color_map,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            seed=args.seed,
            dataset_name=name,
        )
        key = f"{benchmark}/{name}"
        results["benchmarks"][key] = {
            "dataset_key": dataset_key,
            "root": str(root),
            "split": split,
            "samples": len(selected),
            "source_samples": len(dataset),
            "dataset_fingerprint": fingerprint_dataset(dataset),
            "decoded_metrics": metrics,
        }
        logger.info("%s: %s", key, metrics)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    if args.output_csv is not None:
        import csv
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["benchmark", "dataset", "metric", "value"])
            for key, record in results["benchmarks"].items():
                benchmark, dataset = key.split("/", 1)
                for metric, value in record["decoded_metrics"].items():
                    if isinstance(value, (int, float)):
                        writer.writerow([benchmark, dataset, metric, value])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
