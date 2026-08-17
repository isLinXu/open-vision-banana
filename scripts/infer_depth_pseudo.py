#!/usr/bin/env python3
"""Infer or audit v5 pseudo-depth RGB maps with Depth-Anything V2 Small.

The v5 depth labels were not human or sensor depth annotations.  They were
created by running ``depth-anything/Depth-Anything-V2-Small-hf`` on each input,
normalising the result independently for that image, converting it to the
project's rainbow RGB encoding, and resizing the RGB result to 256px.  This
script repeats that known target construction without the lossy MAR/SANA
generation path.

Use ``--input-image`` for a deployable high-resolution prediction.  Use
``--pair-manifest`` to audit a held-out manifest subset against the saved v5
targets.  Model loading is deliberately local-only so inference remains
reproducible and never silently substitutes a different model revision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-image", type=Path, help="One RGB image to estimate.")
    source.add_argument("--pair-manifest", type=Path, help="Clean v5 JSONL manifest to audit.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, help="Prediction path for --input-image mode.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=20, help="Audit samples; 0 means all.")
    parser.add_argument("--num-visualizations", type=int, default=3)
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help="Square RGB output resolution. 0 preserves the input resolution in single-image mode "
        "and reuses each target resolution in audit mode.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    return parser.parse_args()


def select_evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Select deterministic, evenly spread indices without duplication."""
    if total < 1:
        return []
    if count < 1:
        raise ValueError("count must be positive when selecting samples")
    return np.linspace(0, total - 1, min(total, count), dtype=np.int64).tolist()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 0 or args.num_visualizations < 0:
        raise SystemExit("--num-samples and --num-visualizations must be non-negative")
    if args.resolution < 0:
        raise SystemExit("--resolution must be non-negative")


def select_device(value: str | None) -> torch.device:
    if value is not None:
        if value == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available")
        if value == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_depth_estimator(model_name: str, device: torch.device):
    """Load exactly the cached Depth-Anything model and image processor."""
    from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    model = DepthAnythingForDepthEstimation.from_pretrained(
        model_name, local_files_only=True
    ).to(device).eval()
    return model, processor


@torch.no_grad()
def estimate_depth(image: Image.Image, model, processor, device: torch.device) -> np.ndarray:
    """Return the model's relative depth resized to the original image shape."""
    rgb = image.convert("RGB")
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    outputs = model(**inputs)
    prediction = processor.post_process_depth_estimation(
        outputs, target_sizes=[(rgb.height, rgb.width)]
    )[0]["predicted_depth"]
    return prediction.detach().float().cpu().numpy()


def normalize_depth_to_uint8(depth: np.ndarray) -> np.ndarray:
    """Match the pipeline's per-image depth visualisation quantisation."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got {depth.shape}")
    minimum, maximum = float(depth.min()), float(depth.max())
    if maximum - minimum < 1e-6:
        return np.full(depth.shape, 128, dtype=np.uint8)
    return ((depth - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)


def depth_to_rgb(depth: np.ndarray) -> Image.Image:
    """Match ``gen_depth_targets_v5_full.py`` rainbow encoding byte-for-byte."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got {depth.shape}")
    minimum, maximum = float(depth.min()), float(depth.max())
    if maximum - minimum < 1e-6:
        return Image.new("RGB", (depth.shape[1], depth.shape[0]), (128, 128, 128))
    normalized = ((depth - minimum) / (maximum - minimum) * 255.0).astype(np.uint8)
    rgb = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    for value in range(256):
        mask = normalized == value
        t = value / 255.0
        if t < 0.25:
            color = (0, int(t * 4.0 * 255), 255)
        elif t < 0.5:
            color = (0, 255, int(255 - (t - 0.25) * 4.0 * 255))
        elif t < 0.75:
            color = (int((t - 0.5) * 4.0 * 255), 255, 0)
        else:
            color = (255, int(255 - (t - 0.75) * 4.0 * 255), 0)
        rgb[mask] = color
    return Image.fromarray(rgb, mode="RGB")


def infer_rgb(image: Image.Image, model, processor, device: torch.device,
              resolution: int) -> tuple[Image.Image, np.ndarray]:
    """Run depth estimation and render the v5 RGB pseudo-depth output."""
    depth = estimate_depth(image, model, processor, device)
    rgb = depth_to_rgb(normalize_depth_to_uint8(depth))
    if resolution:
        rgb = rgb.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return rgb, depth


def rgb_metrics(prediction: Image.Image, target: Image.Image) -> dict[str, float | None]:
    pred = np.asarray(prediction.convert("RGB"), dtype=np.float32)
    ref = np.asarray(target.convert("RGB"), dtype=np.float32)
    if pred.shape != ref.shape:
        raise ValueError(f"Prediction and target shapes differ: {pred.shape} != {ref.shape}")
    delta = pred - ref
    mae = float(np.abs(delta).mean())
    mse = float(np.square(delta).mean())
    return {
        "rgb_mae_0_255": mae,
        "rgb_rmse_0_255": float(math.sqrt(mse)),
        # Exact reproduction has mathematically infinite PSNR.  Keep JSON
        # standards-compliant reports by expressing it as null plus a flag.
        "rgb_psnr_db": None if mse == 0 else float(20 * math.log10(255.0) - 10 * math.log10(mse)),
        "rgb_psnr_is_infinite": mse == 0,
        "exact_rgb_pixel_fraction": float(np.all(delta == 0, axis=-1).mean()),
        "max_channel_abs_error": float(np.abs(delta).max()),
    }


def write_panel(path: Path, source: Image.Image, prediction: Image.Image,
                target: Image.Image | None = None) -> None:
    """Save source | prediction | target | amplified absolute-error images."""
    preview_size = 256
    images = [source.convert("RGB").resize((preview_size, preview_size), Image.Resampling.BICUBIC),
              prediction.convert("RGB").resize((preview_size, preview_size), Image.Resampling.NEAREST)]
    labels = ["Input", "Depth-Anything pseudo depth"]
    if target is not None:
        target_preview = target.convert("RGB").resize((preview_size, preview_size), Image.Resampling.NEAREST)
        error = ImageChops.difference(images[1], target_preview).point(lambda value: min(255, value * 8))
        images.extend((target_preview, error))
        labels.extend(("Saved v5 target", "Absolute error x8"))

    gap, label_height = 8, 28
    canvas = Image.new(
        "RGB",
        (len(images) * preview_size + (len(images) + 1) * gap, preview_size + label_height + 2 * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        x = gap + index * (preview_size + gap)
        draw.text((x, 6), label, fill="black")
        canvas.paste(image, (x, label_height + gap))
    canvas.save(path)


def load_depth_rows(manifest: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    rows = [
        row for row in rows
        if row.get("task") == "depth"
        and row.get("split") == split
        and row.get("target_kind") == "depth_pseudo_rgb"
    ]
    rows.sort(key=lambda row: row["input"])
    return rows


def run_single_image(args: argparse.Namespace, model, processor, device: torch.device) -> None:
    input_path = args.input_image.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input image not found: {input_path}")
    with Image.open(input_path) as source:
        output_resolution = args.resolution or max(source.size)
        prediction, depth = infer_rgb(source, model, processor, device, output_resolution)
        source_copy = source.convert("RGB")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_image = (args.output_image or args.output_dir / f"{input_path.stem}_depth_pseudo.png").expanduser().resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    prediction.save(output_image)
    write_panel(args.output_dir / f"{input_path.stem}_comparison.png", source_copy, prediction)
    report = {
        "mode": "single_image",
        "input": str(input_path),
        "output": str(output_image),
        "model": args.model,
        "device": str(device),
        "output_resolution": output_resolution,
        "input_size": list(source_copy.size),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
    }
    (args.output_dir / "depth_inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def run_manifest_audit(args: argparse.Namespace, model, processor, device: torch.device) -> None:
    manifest = args.pair_manifest.expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"Pair manifest not found: {manifest}")
    rows = load_depth_rows(manifest, args.split)
    if not rows:
        raise SystemExit(f"No depth_pseudo_rgb rows found for split={args.split!r}")
    indices = list(range(len(rows))) if args.num_samples == 0 else select_evenly_spaced_indices(len(rows), args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, float | None]] = []
    samples: list[dict] = []
    for output_index, dataset_index in enumerate(indices):
        row = rows[dataset_index]
        input_path, target_path = Path(row["input"]), Path(row["target"])
        if not input_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"Missing input or target for manifest index {dataset_index}")
        with Image.open(input_path) as source, Image.open(target_path) as target:
            target_rgb = target.convert("RGB")
            output_resolution = args.resolution or target_rgb.width
            prediction, depth = infer_rgb(source, model, processor, device, output_resolution)
            source_copy = source.convert("RGB")
        sample_metrics = rgb_metrics(prediction, target_rgb)
        metrics.append(sample_metrics)
        samples.append({
            "dataset_index": dataset_index,
            "input": str(input_path.resolve()),
            "target": str(target_path.resolve()),
            "target_size": list(target_rgb.size),
            "prediction_size": list(prediction.size),
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            **sample_metrics,
        })
        if output_index < args.num_visualizations:
            prediction.save(args.output_dir / f"sample_{output_index:02d}_index_{dataset_index:04d}_prediction.png")
            write_panel(
                args.output_dir / f"sample_{output_index:02d}_index_{dataset_index:04d}.png",
                source_copy,
                prediction,
                target_rgb,
            )
        print(
            f"[{output_index + 1}/{len(indices)}] index={dataset_index} "
            f"mae={sample_metrics['rgb_mae_0_255']:.4f} "
            f"exact={sample_metrics['exact_rgb_pixel_fraction']:.4f}",
            flush=True,
        )

    aggregate = {
        key: float(np.mean([sample[key] for sample in metrics]))
        for key in ("rgb_mae_0_255", "rgb_rmse_0_255", "exact_rgb_pixel_fraction", "max_channel_abs_error")
    }
    finite_psnr = [sample["rgb_psnr_db"] for sample in metrics if sample["rgb_psnr_db"] is not None]
    aggregate["rgb_psnr_db"] = float(np.mean(finite_psnr)) if finite_psnr else None
    aggregate["rgb_psnr_is_infinite_fraction"] = float(
        np.mean([sample["rgb_psnr_is_infinite"] for sample in metrics])
    )
    report = {
        "mode": "manifest_audit",
        "pair_manifest": str(manifest),
        "split": args.split,
        "model": args.model,
        "local_files_only": True,
        "device": str(device),
        "num_samples": len(indices),
        "sample_indices": indices,
        "requested_resolution": args.resolution,
        "aggregate_rgb_reproduction": aggregate,
        "samples": samples,
    }
    (args.output_dir / "depth_inference.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = select_device(args.device)
    model, processor = load_depth_estimator(args.model, device)
    if args.input_image is not None:
        run_single_image(args, model, processor, device)
    else:
        run_manifest_audit(args, model, processor, device)


if __name__ == "__main__":
    main()
