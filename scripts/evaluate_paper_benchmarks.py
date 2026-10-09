#!/usr/bin/env python3
"""Evaluate decoded-task metrics on the paper-aligned benchmarks.

Examples:
    python scripts/evaluate_paper_benchmarks.py \
        --model-name black-forest-labs/FLUX.1-schnell \
        --lora-path output/checkpoint-best \
        --depth-root /data/benchmarks \
        --normal-root /data/benchmarks \
        --cityscapes-root /data/cityscapes \
        --benchmark all \
        --output-json output/eval/paper_metrics.json

    python scripts/evaluate_paper_benchmarks.py \
        --benchmark semantic --cityscapes-root /data/cityscapes \
        --output-json output/eval/preflight.json --preflight-only

Evaluation requires local benchmark data and model weights. Preflight-only
mode requires benchmark data but never loads model weights. Neither mode falls
back to synthetic data or RGB reconstruction metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data import build_dataset  # noqa: E402
from open_vision_banana.data.manifest import fingerprint_records  # noqa: E402
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
# Dataset adapters that can return unresized GT (``native_gt=True``).
NATIVE_GT_DATASETS = {
    "nyu_depth",
    "ibims",
    "diode",
    "kitti",
    "eth3d",
    "nyu_normal",
    "diode_normal",
    "normal",
    "vkitti_normal",
    "referring",
}
# Extra adapter kwargs that make a benchmark match the paper's definition.
# The paper evaluates DIODE-Indoor, not the indoor+outdoor union, and RefCOCOg
# on the UMD split.
BENCHMARK_DATASET_KWARGS: dict[str, dict[str, Any]] = {
    "diode": {"scene_type": "indoors"},
    "diode_normal": {"scene_type": "indoors"},
    "referring": {"protocol": "umd"},
}
NORMAL_DATASETS = {
    "nyu": "nyu_normal",
    "diode": "diode_normal",
    "scannet": "normal",
    "vkitti": "vkitti_normal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["depth", "normal", "semantic", "referring", "all"],
        default="all",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Base model identifier. Required for evaluation, omitted for --preflight-only.",
    )
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--device", default=None)
    # SANA's MPS kernels can produce non-finite img2img samples in fp16/bf16.
    # Keep evaluation reproducible and finite by defaulting to fp32; users may
    # still request reduced precision on CUDA/CPU explicitly.
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--allow-mps-reduced-dtype",
        action="store_true",
        help=(
            "Explicitly allow fp16/bf16 on MPS for an audited compatibility run. "
            "Non-finite generator output still hard-fails."
        ),
    )
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
    parser.add_argument("--refcocog-root", type=Path, default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--native-resolution",
        action="store_true",
        help=(
            "Score at the original GT resolution. Depth: NYU, iBims, DIODE, "
            "KITTI, ETH3D. Normals: NYU, DIODE-Indoor, ScanNet, VKITTI. "
            "Referring: RefCOCOg masks. Semantic jobs ignore the flag. "
            "Use --batch-size 1 for datasets with mixed image sizes."
        ),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--depth-colormap",
        default="cube_edge",
        help="Colormap used to invert predicted depth RGB. Default matches the paper cube-edge.",
    )
    parser.add_argument(
        "--invert-depth",
        action="store_true",
        help="Decode bright pixels as near (1 - normalized) before the power inverse.",
    )
    parser.add_argument(
        "--depth-prompt",
        default=None,
        help="Override the depth prompt. Use the checkpoint's training prompt when metadata.json has none.",
    )
    parser.add_argument(
        "--relative-luminance",
        action="store_true",
        help="Also score a closer=brighter luminance map after median scale (relative_delta_1.25_aligned).",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate benchmark roots and splits, then write the audit without loading a model.",
    )
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


def resolve_runtime_dtype(
    value: str,
    device: str,
    *,
    allow_mps_reduced_dtype: bool = False,
) -> torch.dtype:
    """Resolve an evaluation dtype and reject known-unstable MPS modes."""
    dtype = resolve_dtype(value)
    if device == "mps" and dtype != torch.float32 and not allow_mps_reduced_dtype:
        raise ValueError(
            "SANA evaluation on MPS requires --dtype float32; fp16/bf16 can "
            "produce non-finite scheduler outputs. Pass "
            "--allow-mps-reduced-dtype only for an explicit compatibility run."
        )
    return dtype


def resolve_inference_steps(model_name: str | None, requested: int) -> int:
    """Keep FLUX-schnell's 1-step default, but use SANA's trained 20-step ODE."""
    if requested != 1 or not model_name:
        return requested
    from open_vision_banana.model_registry import resolve_model_config

    try:
        config = resolve_model_config(model_name)
    except Exception:
        return requested
    if config.family == "sana":
        return int(config.default_num_inference_steps or 20)
    return requested


def fingerprint_dataset(dataset: Any) -> str:
    """Hash dataset records with paths normalized to the dataset root."""
    root = getattr(dataset, "root_dir", None)
    records: list[dict[str, Any]] = []
    for sample in getattr(dataset, "samples", []):
        record: dict[str, Any] = {}
        for key in ("image", "target", "expression"):
            value = sample.get(key)
            if value is not None:
                record[key] = str(value)
        records.append(record)
    return fingerprint_records(records, root)


def select_dataset(dataset: Any, max_samples: int | None) -> Any:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, range(max_samples))


def _resolve_collection_root(root: Path | None, name: str) -> Path | None:
    """Resolve either a per-dataset root or a collection with named children."""
    if root is None:
        return None
    nested = root / name
    if nested.is_dir():
        return nested
    # A collection root must not be passed to every adapter: doing so turns a
    # missing dataset into a misleading "empty layout" error for another one.
    if root.name.casefold() == name.casefold():
        return root
    return root / name if root.is_dir() else None


def build_jobs(args: argparse.Namespace) -> list[tuple[str, str, Path, str]]:
    jobs: list[tuple[str, str, Path, str]] = []
    if args.benchmark in ("depth", "all"):
        depth_roots = {
            "nyu": args.nyu_root or _resolve_collection_root(args.depth_root, "nyu"),
            "ibims": args.ibims_root or _resolve_collection_root(args.depth_root, "ibims"),
            "eth3d": args.eth3d_root or _resolve_collection_root(args.depth_root, "eth3d"),
            "diode": args.diode_root or _resolve_collection_root(args.depth_root, "diode"),
            "kitti": args.kitti_root or _resolve_collection_root(args.depth_root, "kitti"),
        }
        for name, task in DEPTH_DATASETS.items():
            if depth_roots[name] is not None:
                jobs.append(("depth", name, depth_roots[name], task))
    if args.benchmark in ("normal", "all"):
        normal_roots = {
            "nyu": args.nyu_root or _resolve_collection_root(args.normal_root, "nyu"),
            "diode": args.diode_root or _resolve_collection_root(args.normal_root, "diode"),
            "scannet": args.scannet_root or _resolve_collection_root(args.normal_root, "scannet"),
            "vkitti": args.vkitti_root or _resolve_collection_root(args.normal_root, "vkitti"),
        }
        for name, task in NORMAL_DATASETS.items():
            if normal_roots[name] is not None:
                jobs.append(("normal", name, normal_roots[name], task))
    if args.benchmark in ("semantic", "all") and args.cityscapes_root is not None:
        jobs.append(("semantic", "cityscapes", args.cityscapes_root, "semantic"))
    if args.benchmark in ("referring", "all") and args.refcocog_root is not None:
        jobs.append(("referring", "refcocog", args.refcocog_root, "referring"))
    return jobs


def preflight_jobs(
    jobs: list[tuple[str, str, Path, str]],
    *,
    split: str,
    resolution: int,
) -> dict[str, Any]:
    """Check requested benchmark adapters before allocating model resources."""
    records: list[dict[str, Any]] = []
    for benchmark, name, root, dataset_key in jobs:
        dataset_split = split
        root = root.expanduser()
        record: dict[str, Any] = {
            "benchmark": benchmark,
            "dataset": name,
            "dataset_key": dataset_key,
            "root": str(root),
            "split": dataset_split,
            "ok": False,
        }
        if not root.is_dir():
            record["error"] = f"Benchmark root does not exist or is not a directory: {root}"
        else:
            try:
                dataset = build_dataset(
                    dataset_key,
                    str(root),
                    split=dataset_split,
                    resolution=resolution,
                    **BENCHMARK_DATASET_KWARGS.get(dataset_key, {}),
                )
                record["samples"] = len(dataset)
                if len(dataset) == 0:
                    record["error"] = (
                        "Adapter found no usable pairs. Verify the dataset layout and requested split."
                    )
                else:
                    record["ok"] = True
                    record["dataset_fingerprint"] = fingerprint_dataset(dataset)
            except Exception as exc:
                record["error"] = f"Adapter initialization failed: {type(exc).__name__}: {exc}"
        records.append(record)
    return {"ok": all(record["ok"] for record in records), "jobs": records}


def _json_safe(value: Any) -> Any:
    """Convert non-finite metric values to JSON null without hiding them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(results), indent=2, allow_nan=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.preflight_only and not args.model_name:
        raise SystemExit("--model-name is required unless --preflight-only is set")
    args.num_inference_steps = resolve_inference_steps(args.model_name, args.num_inference_steps)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    jobs = build_jobs(args)
    if not jobs:
        raise SystemExit("No benchmark roots supplied")

    preflight = preflight_jobs(jobs, split=args.split, resolution=args.resolution)
    results: dict[str, Any] = {
        "metadata": {
            "evaluation_mode": "decoded_task_metrics",
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
        "preflight": preflight,
        "benchmarks": {},
    }
    if args.preflight_only or not preflight["ok"]:
        write_results(args.output_json, results)
        for record in preflight["jobs"]:
            if not record["ok"]:
                logger.error("%s/%s: %s", record["benchmark"], record["dataset"], record["error"])
        if not preflight["ok"]:
            return 2
        return 0

    from open_vision_banana.inference.pipeline import VisionBanana

    runtime_dtype = resolve_runtime_dtype(
        args.dtype,
        device,
        allow_mps_reduced_dtype=args.allow_mps_reduced_dtype,
    )
    results["metadata"]["dtype"] = str(runtime_dtype).replace("torch.", "")
    model = VisionBanana.from_pretrained(
        model_name=args.model_name,
        lora_path=args.lora_path,
        device=device,
        dtype=runtime_dtype,
    )
    if args.depth_prompt:
        model.training_prompts["depth"] = args.depth_prompt
    evaluator = VisionEvaluator(model, device=device)
    results["metadata"]["depth_colormap"] = args.depth_colormap
    results["metadata"]["invert_depth"] = bool(args.invert_depth)
    results["metadata"]["depth_prompt"] = args.depth_prompt
    results["metadata"]["relative_luminance"] = bool(args.relative_luminance)
    results["metadata"]["protocol"] = {
        "depth": "decoded RGB to metric depth; valid GT pixels only; NYU Eigen crop + 10m cap / KITTI Garg crop + 80m cap / ETH3D 80m cap when those datasets are named; AbsRel_aligned/delta_1.25_aligned are per-image median-scale diagnostics, not paper protocol; *_per_image keys are means of per-image metrics (headline keys are pixel-weighted); crop/cap/aggregation are conventions, not paper-disclosed",
        "normal": "decoded RGB to unit camera-space normals; valid pixels are finite, non-zero GT (plus DIODE mask when present); mean_angle_deg_per_image is the mean of per-image means; NYU/ScanNet/VKITTI require community normal maps and are not derived from depth; DIODE is indoors",
        "semantic": "decoded RGB to class labels; Cityscapes labelIds mapped to trainIds; global confusion matrix",
        "referring": "RefCOCOg UMD when refs(umd).p is present, else refs(g).p; every sentence is a sample; cIoU/gIoU from polygon or RLE masks, never from bounding boxes",
        "aggregation": "pixel-weighted across the complete selected dataset, not batch mean",
    }

    for benchmark, name, root, dataset_key in jobs:
        # Preflight and execution must use the same requested split.  The
        # default is ``val`` for the paper gate; callers may explicitly run
        # the official train split when that data is present.
        split = args.split
        native = args.native_resolution and benchmark in {"depth", "normal", "referring"}
        if native and dataset_key not in NATIVE_GT_DATASETS:
            raise SystemExit(
                f"--native-resolution is not implemented for dataset {dataset_key!r}; "
                f"supported: {sorted(NATIVE_GT_DATASETS)}"
            )
        dataset = build_dataset(
            dataset_key,
            str(root),
            split=split,
            resolution=args.resolution,
            **BENCHMARK_DATASET_KWARGS.get(dataset_key, {}),
            **({"native_gt": True} if native else {}),
        )
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
            native_resolution=native,
            depth_colormap=args.depth_colormap,
            invert_depth=args.invert_depth,
            report_relative_luminance=args.relative_luminance,
        )
        key = f"{benchmark}/{name}"
        results["benchmarks"][key] = {
            "dataset_key": dataset_key,
            "root": str(root),
            "split": split,
            "native_resolution": native,
            "dataset_kwargs": BENCHMARK_DATASET_KWARGS.get(dataset_key, {}),
            "samples": len(selected),
            "source_samples": len(dataset),
            "dataset_fingerprint": fingerprint_dataset(dataset),
            "decoded_metrics": metrics,
        }
        logger.info("%s: %s", key, metrics)

    write_results(args.output_json, results)
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
