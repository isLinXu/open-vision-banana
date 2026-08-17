#!/usr/bin/env python3
"""Generate high-resolution binary structure maps from the target rule.

The v5 structure targets in ``sana_structure_binary_targets_v1`` are derived
from their input image using Canny edges and long LSD segments.  A 512px SANA
output must pass through a 16x16 DC-AE latent grid, which cannot faithfully
represent the sparse, thin target.  This utility replays the audited target
rule directly at image resolution instead of decoding a lossy RGB diffusion
sample.

It supports either a standalone image or an annotated manifest audit.  In
manifest mode, metrics are reported only when targets are available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_sana_structure_binary_targets import build_binary_structure_map
from eval_sana_structure_checkpoints import add_counts, edge_counts, edge_scores, select_evenly_spaced_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-image", type=Path, help="One image to convert into a binary structure map.")
    source.add_argument("--pair-manifest", type=Path, help="v5 JSONL manifest to audit or visualize.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-image", type=Path, help="Output filename for --input-image mode.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=20, help="Manifest samples to audit; 0 means all.")
    parser.add_argument("--num-visualizations", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--min-line-fraction", type=float, default=0.08)
    parser.add_argument("--edge-width", type=int, default=3)
    parser.add_argument("--tolerance-px", type=float, default=2.0)
    return parser.parse_args()


def read_target_binary(path: Path, resolution: int) -> np.ndarray:
    with Image.open(path) as image:
        target = np.asarray(
            image.convert("RGB").resize((resolution, resolution), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    return target[..., 0] >= 128


def write_panel(path: Path, input_path: Path, prediction: np.ndarray, target_path: Path | None, resolution: int) -> None:
    """Save a compact input | deterministic prediction | target panel."""
    with Image.open(input_path) as image:
        source = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
    prediction_image = Image.fromarray(prediction, mode="RGB")
    images = [source, prediction_image]
    labels = ["Input", "Predicted"]
    if target_path is not None:
        with Image.open(target_path) as image:
            images.append(image.convert("RGB").resize((resolution, resolution), Image.Resampling.NEAREST))
        labels.append("Target")

    label_height = 28
    gap = 8
    canvas = Image.new("RGB", (len(images) * resolution + (len(images) + 1) * gap, resolution + label_height + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        x = gap + index * (resolution + gap)
        draw.text((x, 6), label, fill=(0, 0, 0))
        canvas.paste(image, (x, label_height + gap))
    canvas.save(path)


def validate_args(args: argparse.Namespace) -> None:
    if args.resolution < 32:
        raise SystemExit("--resolution must be at least 32")
    if not 0.0 < args.min_line_fraction <= 1.0:
        raise SystemExit("--min-line-fraction must be in (0, 1]")
    if args.edge_width < 1 or args.edge_width % 2 == 0:
        raise SystemExit("--edge-width must be a positive odd integer")
    if args.num_samples < 0 or args.num_visualizations < 0:
        raise SystemExit("--num-samples and --num-visualizations must be non-negative")
    if args.tolerance_px < 0:
        raise SystemExit("--tolerance-px must be non-negative")


def run_single_image(args: argparse.Namespace) -> None:
    input_path = args.input_image.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input image not found: {input_path}")
    prediction = build_binary_structure_map(
        input_path, args.resolution, args.min_line_fraction, args.edge_width
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_image = args.output_image or args.output_dir / f"{input_path.stem}_structure.png"
    output_image = output_image.expanduser().resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction, mode="RGB").save(output_image)
    write_panel(args.output_dir / f"{input_path.stem}_comparison.png", input_path, prediction, None, args.resolution)
    report = {
        "mode": "single_image",
        "input": str(input_path),
        "output": str(output_image),
        "resolution": args.resolution,
        "min_line_fraction": args.min_line_fraction,
        "edge_width": args.edge_width,
        "positive_fraction": float((prediction[..., 0] > 0).mean()),
    }
    (args.output_dir / "structure_inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def run_manifest_audit(args: argparse.Namespace) -> None:
    manifest = args.pair_manifest.expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"Pair manifest not found: {manifest}")
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    rows = [row for row in rows if row.get("task") == "structure" and row.get("split") == args.split]
    rows.sort(key=lambda row: row["input"])
    if not rows:
        raise SystemExit(f"No structure rows found for split={args.split!r}")
    indices = list(range(len(rows))) if args.num_samples == 0 else select_evenly_spaced_indices(len(rows), args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = {"matched_prediction": 0, "prediction_total": 0, "matched_target": 0, "target_total": 0}
    exact_matches = 0
    positive_fractions: list[float] = []
    samples: list[dict] = []
    for output_index, dataset_index in enumerate(indices):
        row = rows[dataset_index]
        input_path = Path(row["input"])
        target_path = Path(row["target"])
        if not input_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"Missing input or target for manifest index {dataset_index}")
        prediction = build_binary_structure_map(
            input_path, args.resolution, args.min_line_fraction, args.edge_width
        )
        prediction_binary = prediction[..., 0] >= 128
        target_binary = read_target_binary(target_path, args.resolution)
        counts = add_counts(counts, edge_counts(prediction_binary, target_binary, args.tolerance_px))
        exact = bool(np.array_equal(prediction_binary, target_binary))
        exact_matches += int(exact)
        positive_fractions.append(float(prediction_binary.mean()))
        samples.append({
            "dataset_index": dataset_index,
            "input": str(input_path.resolve()),
            "target": str(target_path.resolve()),
            "exact_binary_match": exact,
            "prediction_positive_fraction": float(prediction_binary.mean()),
        })
        if output_index < args.num_visualizations:
            write_panel(
                args.output_dir / f"sample_{output_index:02d}_index_{dataset_index:04d}.png",
                input_path,
                prediction,
                target_path,
                args.resolution,
            )

    report = {
        "mode": "manifest_audit",
        "pair_manifest": str(manifest),
        "split": args.split,
        "num_samples": len(indices),
        "sample_indices": indices,
        "resolution": args.resolution,
        "min_line_fraction": args.min_line_fraction,
        "edge_width": args.edge_width,
        "tolerance_px": args.tolerance_px,
        "exact_binary_match_fraction": exact_matches / len(indices),
        "mean_prediction_positive_fraction": float(np.mean(positive_fractions)),
        "edge_metrics": {**edge_scores(counts), **counts},
        "samples": samples,
    }
    (args.output_dir / "structure_inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.input_image is not None:
        run_single_image(args)
    else:
        run_manifest_audit(args)


if __name__ == "__main__":
    main()
