#!/usr/bin/env python3
"""Build a reproducible depth-only manifest with high-edge train oversampling.

The source manifest remains the authority for valid input/target pairs and the
validation split.  Every depth row is retained once.  Rows in the highest
edge-density fraction of the training split are appended ``repeat - 1`` extra
times, which makes the existing uniform sample selection favour geometric
boundaries without changing the validation distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper


@dataclass(frozen=True)
class EdgeScore:
    """Decoded depth-edge statistics used to rank one manifest row."""

    density: float
    energy: float


def decoded_depth_edge_score(
    target_path: Path,
    *,
    mapper: DepthRGBMapper,
    resolution: int,
    edge_threshold: float,
) -> EdgeScore:
    """Measure boundary coverage after decoding the target's turbo RGB depth."""
    with Image.open(target_path) as image:
        rgb_array = np.asarray(
            image.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
    rgb = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0)
    depth = mapper.rgb_to_normalized(rgb).squeeze(0).squeeze(0)

    # Use forward differences on the decoded scalar depth rather than RGB
    # channel differences, so the false-color encoding cannot inflate scores.
    grad_x = torch.zeros_like(depth)
    grad_y = torch.zeros_like(depth)
    grad_x[:, 1:] = (depth[:, 1:] - depth[:, :-1]).abs()
    grad_y[1:, :] = (depth[1:, :] - depth[:-1, :]).abs()
    gradient = torch.maximum(grad_x, grad_y)
    return EdgeScore(
        density=float((gradient >= edge_threshold).float().mean().item()),
        energy=float(gradient.mean().item()),
    )


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        str(percentile): float(np.quantile(array, percentile))
        for percentile in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def build_weighted_manifest(
    source_manifest: Path,
    output_manifest: Path,
    stats_path: Path,
    *,
    edge_fraction: float,
    high_edge_repeat: int,
    resolution: int,
    edge_threshold: float,
) -> dict:
    """Create the depth-only manifest and return its auditable summary."""
    if not 0.0 < edge_fraction <= 1.0:
        raise ValueError("edge_fraction must be in (0, 1]")
    if high_edge_repeat < 1:
        raise ValueError("high_edge_repeat must be at least 1")

    records: list[tuple[int, str, dict]] = []
    train_candidates: list[tuple[EdgeScore, int, dict]] = []
    mapper = DepthRGBMapper(colormap="turbo")

    with source_manifest.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if row.get("task") != "depth":
                continue
            records.append((line_number, raw_line.rstrip("\n"), row))
            if row.get("split") != "train":
                continue
            target_path = Path(row.get("target", ""))
            if not target_path.is_file():
                raise FileNotFoundError(
                    f"depth target is unavailable at manifest line {line_number}: {target_path}"
                )
            train_candidates.append(
                (
                    decoded_depth_edge_score(
                        target_path,
                        mapper=mapper,
                        resolution=resolution,
                        edge_threshold=edge_threshold,
                    ),
                    line_number,
                    row,
                )
            )

    if not train_candidates:
        raise ValueError(f"No depth training rows found in {source_manifest}")
    train_count = len(train_candidates)
    selected_count = max(1, math.ceil(train_count * edge_fraction))
    ranked = sorted(
        train_candidates,
        key=lambda item: (-item[0].density, -item[0].energy, str(item[2].get("input", "")), item[1]),
    )
    selected = ranked[:selected_count]

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for _, raw_line, _ in records:
            handle.write(raw_line + "\n")
        for score, line_number, row in selected:
            for copy_index in range(2, high_edge_repeat + 1):
                duplicate = dict(row)
                duplicate["sampling"] = {
                    "strategy": "decoded_depth_edge_density",
                    "source_manifest_line": line_number,
                    "copy_index": copy_index,
                    "edge_density": score.density,
                    "edge_energy": score.energy,
                }
                handle.write(json.dumps(duplicate, ensure_ascii=False, separators=(",", ":")) + "\n")

    densities = [score.density for score, _, _ in train_candidates]
    energies = [score.energy for score, _, _ in train_candidates]
    split_counts = {"train": 0, "val": 0}
    for _, _, row in records:
        split = str(row.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1
    summary = {
        "source_manifest": str(source_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "task": "depth",
        "algorithm": {
            "decode_colormap": "turbo",
            "resolution": resolution,
            "edge_threshold": edge_threshold,
            "edge_score": "fraction of decoded-depth pixels with max forward gradient >= threshold",
            "tie_breaker": "mean decoded-depth max-forward-gradient energy",
        },
        "reweighting": {
            "edge_fraction": edge_fraction,
            "high_edge_repeat": high_edge_repeat,
            "selected_train_rows": selected_count,
            "all_train_rows": train_count,
            "effective_train_rows": train_count + selected_count * (high_edge_repeat - 1),
        },
        "base_split_counts": split_counts,
        "edge_density_quantiles": _quantiles(densities),
        "edge_energy_quantiles": _quantiles(energies),
        "selection_cutoff": {
            "edge_density": selected[-1][0].density,
            "edge_energy": selected[-1][0].energy,
        },
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--edge-fraction", type=float, default=0.25)
    parser.add_argument("--high-edge-repeat", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--edge-threshold", type=float, default=0.025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_weighted_manifest(
        args.source_manifest,
        args.output_manifest,
        args.stats_json,
        edge_fraction=args.edge_fraction,
        high_edge_repeat=args.high_edge_repeat,
        resolution=args.resolution,
        edge_threshold=args.edge_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
