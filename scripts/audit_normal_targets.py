#!/usr/bin/env python3
"""Audit whether normal RGB targets contain meaningful directional supervision.

This audit deliberately measures a constant-direction baseline.  A valid dense
normal dataset should not be solved by emitting one global normal direction:
its angular spread and neighbour variation should reveal image-dependent
geometry.  The script works on the clean v5 manifest and does not alter any
source labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.normal_mapper import NormalRGBMapper

ANGLE_HISTOGRAM_BINS = 18_000
RAW_NORM_HISTOGRAM_BINS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--num-samples", type=int, default=0, help="0 audits all rows in the split.")
    parser.add_argument("--num-visualizations", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=256)
    return parser.parse_args()


def select_evenly_spaced_indices(total: int, count: int) -> list[int]:
    if total < 1:
        return []
    if count < 1:
        raise ValueError("count must be positive when selecting samples")
    return np.linspace(0, total - 1, min(total, count), dtype=np.int64).tolist()


def load_rows(manifest: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    rows = [
        row for row in rows
        if row.get("task") == "normal"
        and row.get("split") == split
        and row.get("target_kind") == "source_normal_rgb"
    ]
    rows.sort(key=lambda row: row["input"])
    return rows


def decode_raw_normals(rgb: torch.Tensor) -> torch.Tensor:
    """Decode the exact camera-space normal RGB convention without renormalising."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected [3, H, W] RGB tensor, got {tuple(rgb.shape)}")
    return torch.stack((1.0 - 2.0 * rgb[0], 2.0 * rgb[1] - 1.0, 2.0 * rgb[2] - 1.0))


def load_normal(path: Path, resolution: int, mapper: NormalRGBMapper) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as image:
        rgb_array = np.asarray(
            image.convert("RGB").resize((resolution, resolution), Image.Resampling.NEAREST),
            dtype=np.float32,
        ) / 255.0
    rgb = torch.from_numpy(rgb_array).permute(2, 0, 1)
    raw = decode_raw_normals(rgb)
    normal = mapper.rgb_to_normal(rgb.unsqueeze(0)).squeeze(0)
    return raw, normal


def angular_error_degrees(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-vector angle with left/right shaped [N, 3] or broadcastable [3]."""
    left = torch.nn.functional.normalize(left, dim=-1, eps=1e-8)
    right = torch.nn.functional.normalize(right, dim=-1, eps=1e-8)
    cosine = (left * right).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def neighbour_angular_variation(normal: torch.Tensor) -> torch.Tensor:
    """Angles between horizontally and vertically adjacent normal vectors."""
    horizontal = angular_error_degrees(normal[:, :, :-1].movedim(0, -1), normal[:, :, 1:].movedim(0, -1))
    vertical = angular_error_degrees(normal[:, :-1, :].movedim(0, -1), normal[:, 1:, :].movedim(0, -1))
    return torch.cat((horizontal.reshape(-1), vertical.reshape(-1)))


class BoundedHistogram:
    """Streaming scalar summary with deterministic quantiles and bounded memory."""

    def __init__(self, minimum: float, maximum: float, bins: int) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.bins = bins
        self.counts = np.zeros(bins, dtype=np.int64)
        self.total = 0
        self.sum = 0.0

    def add(self, values: torch.Tensor) -> None:
        array = values.detach().float().cpu().numpy().reshape(-1)
        if array.size == 0:
            return
        array = np.nan_to_num(array, nan=self.minimum, posinf=self.maximum, neginf=self.minimum)
        array = np.clip(array, self.minimum, self.maximum)
        positions = ((array - self.minimum) / (self.maximum - self.minimum) * self.bins).astype(np.int64)
        positions = np.clip(positions, 0, self.bins - 1)
        self.counts += np.bincount(positions, minlength=self.bins)
        self.total += int(array.size)
        self.sum += float(array.sum(dtype=np.float64))

    def mean(self) -> float:
        return self.sum / self.total if self.total else 0.0

    def quantile(self, quantile: float) -> float:
        if not self.total:
            return 0.0
        target = int(np.clip(np.ceil(quantile * self.total), 1, self.total))
        index = int(np.searchsorted(np.cumsum(self.counts), target, side="left"))
        return self.minimum + (index + 0.5) * (self.maximum - self.minimum) / self.bins

    def fraction_below(self, value: float) -> float:
        if not self.total:
            return 0.0
        index = int(np.clip(np.floor((value - self.minimum) / (self.maximum - self.minimum) * self.bins), 0, self.bins))
        return float(self.counts[:index].sum() / self.total)


def write_panel(path: Path, input_path: Path, target_path: Path, resolution: int) -> None:
    with Image.open(input_path) as image:
        source = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
    with Image.open(target_path) as image:
        target = image.convert("RGB").resize((resolution, resolution), Image.Resampling.NEAREST)
    gap, label_height = 8, 28
    canvas = Image.new("RGB", (2 * resolution + 3 * gap, resolution + label_height + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), "Input", fill="black")
    draw.text((2 * gap + resolution, 6), "Normal target", fill="black")
    canvas.paste(source, (gap, label_height + gap))
    canvas.paste(target, (2 * gap + resolution, label_height + gap))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if args.num_samples < 0 or args.num_visualizations < 0:
        raise SystemExit("--num-samples and --num-visualizations must be non-negative")
    if args.resolution < 16:
        raise SystemExit("--resolution must be at least 16")
    manifest = args.pair_manifest.expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"Pair manifest not found: {manifest}")
    rows = load_rows(manifest, args.split)
    if not rows:
        raise SystemExit(f"No normal source_normal_rgb rows found for split={args.split!r}")
    indices = list(range(len(rows))) if args.num_samples == 0 else select_evenly_spaced_indices(len(rows), args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapper = NormalRGBMapper()
    direction_sum = torch.zeros(3)
    vector_count = 0
    raw_norms = BoundedHistogram(0.0, 2.0, RAW_NORM_HISTOGRAM_BINS)
    neighbour_angles = BoundedHistogram(0.0, 180.0, ANGLE_HISTOGRAM_BINS)
    samples: list[dict] = []
    for output_index, dataset_index in enumerate(indices):
        row = rows[dataset_index]
        input_path, target_path = Path(row["input"]), Path(row["target"])
        if not input_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"Missing input or target at manifest index {dataset_index}")
        raw, normal = load_normal(target_path, args.resolution, mapper)
        flat = normal.movedim(0, -1).reshape(-1, 3)
        sample_raw_norms = raw.norm(dim=0).reshape(-1)
        sample_neighbour_angles = neighbour_angular_variation(normal)
        direction_sum += flat.sum(dim=0)
        vector_count += len(flat)
        raw_norms.add(sample_raw_norms)
        neighbour_angles.add(sample_neighbour_angles)
        if output_index < args.num_visualizations:
            write_panel(args.output_dir / f"sample_{output_index:02d}_index_{dataset_index:04d}.png", input_path, target_path, args.resolution)
        samples.append({
            "dataset_index": dataset_index,
            "input": str(input_path.resolve()),
            "target": str(target_path.resolve()),
            "raw_norm_median": float(sample_raw_norms.median()),
            "raw_norm_p95_abs_error": float((sample_raw_norms - 1.0).abs().quantile(0.95)),
            "neighbour_angular_error_mean_deg": float(sample_neighbour_angles.mean()),
            "neighbour_angular_error_p95_deg": float(sample_neighbour_angles.quantile(0.95)),
        })

    global_direction = torch.nn.functional.normalize(direction_sum / vector_count, dim=0, eps=1e-8)
    baseline_angles = BoundedHistogram(0.0, 180.0, ANGLE_HISTOGRAM_BINS)
    # Second pass avoids retaining every image's pixels in RAM simply to score
    # the one global constant-direction baseline.
    for sample, dataset_index in zip(samples, indices, strict=True):
        row = rows[dataset_index]
        _, normal = load_normal(Path(row["target"]), args.resolution, mapper)
        flat = normal.movedim(0, -1).reshape(-1, 3)
        sample_baseline_angles = angular_error_degrees(flat, global_direction)
        baseline_angles.add(sample_baseline_angles)
        sample_direction = torch.nn.functional.normalize(flat.mean(dim=0), dim=0, eps=1e-8)
        per_image_angles = angular_error_degrees(flat, sample_direction)
        sample["per_image_constant_angular_error_mean_deg"] = float(per_image_angles.mean())
        sample["per_image_constant_angular_error_median_deg"] = float(per_image_angles.median())

    report = {
        "mode": "normal_target_audit",
        "pair_manifest": str(manifest),
        "split": args.split,
        "num_samples": len(indices),
        "sample_indices": indices,
        "resolution": args.resolution,
        "global_constant_direction": [float(value) for value in global_direction],
        "raw_normal_length": {
            "median": raw_norms.quantile(0.5),
            "p95_abs_error_from_unit": max(1.0 - raw_norms.quantile(0.05), raw_norms.quantile(0.95) - 1.0),
        },
        "constant_direction_baseline": {
            "mean_angular_error_deg": baseline_angles.mean(),
            "median_angular_error_deg": baseline_angles.quantile(0.5),
            "within_1deg_fraction": baseline_angles.fraction_below(1.0),
            "within_5deg_fraction": baseline_angles.fraction_below(5.0),
            "within_30deg_fraction": baseline_angles.fraction_below(30.0),
        },
        "local_directional_variation": {
            "neighbour_angular_error_mean_deg": neighbour_angles.mean(),
            "neighbour_angular_error_median_deg": neighbour_angles.quantile(0.5),
            "neighbour_angular_error_p95_deg": neighbour_angles.quantile(0.95),
        },
        "diagnosis": (
            "collapsed_constant_normal_labels"
            if baseline_angles.mean() < 5.0 and neighbour_angles.mean() < 2.0
            else "directionally_variable_labels"
        ),
        "samples": samples,
    }
    (args.output_dir / "normal_target_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
