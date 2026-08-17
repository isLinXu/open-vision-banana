#!/usr/bin/env python3
"""Audit local DIODE and Hypersim RGB-to-camera-normal supervision.

The previous v5 normal targets collapse to nearly one global direction.  This
script audits only local source datasets before any training data is built. It
checks that RGB/normal pairs exist, decodes the standard ``2 * RGB - 1`` normal
encoding, excludes non-unit background pixels, and measures both a
constant-direction baseline and valid-neighbour angular variation.

No source image or label is modified.  A dataset is considered suitable for a
future RGB-to-normal training gate only when it has enough valid pixels and
cannot be represented by a nearly constant normal direction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_DIODE_ROOT = Path("/Users/gatilin/MyWork/datasets/diode_eval/data/diode-normal-256")
DEFAULT_HYPERSIM_ROOT = Path("/Users/gatilin/MyWork/datasets/hypersim")
NORMAL_LENGTH_TOLERANCE = 0.12


@dataclass(frozen=True)
class NormalPair:
    dataset: str
    input_path: Path
    normal_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("diode", "hypersim", "both"), default="both")
    parser.add_argument("--diode-root", type=Path, default=DEFAULT_DIODE_ROOT)
    parser.add_argument("--hypersim-root", type=Path, default=DEFAULT_HYPERSIM_ROOT)
    parser.add_argument("--max-samples", type=int, default=256, help="0 audits every discovered pair.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-visualizations", type=int, default=3)
    parser.add_argument("--normal-length-tolerance", type=float, default=NORMAL_LENGTH_TOLERANCE)
    return parser.parse_args()


def evenly_spaced(items: list[NormalPair], maximum: int) -> list[NormalPair]:
    if maximum == 0 or maximum >= len(items):
        return items
    if maximum < 0:
        raise ValueError("max-samples must be non-negative")
    indices = np.linspace(0, len(items) - 1, maximum, dtype=np.int64)
    return [items[int(index)] for index in indices]


def discover_diode(root: Path) -> list[NormalPair]:
    pairs: list[NormalPair] = []
    for normal in sorted(root.rglob("*_normal.png")):
        base = Path(str(normal)[:-len("_normal.png")])
        source = base.with_suffix(".png")
        if source.is_file():
            pairs.append(NormalPair("diode", source, normal))
    return pairs


def discover_hypersim(root: Path) -> list[NormalPair]:
    pairs: list[NormalPair] = []
    for normal in sorted((root / "scenes").glob("**/scene_cam_*_geometry_preview/*.normal_cam.png")):
        preview = normal.parent
        final_name = preview.name.replace("_geometry_preview", "_final_preview")
        frame = normal.name.removesuffix(".normal_cam.png")
        source = preview.parent / final_name / f"{frame}.color.jpg"
        if source.is_file():
            pairs.append(NormalPair("hypersim", source, normal))
    return pairs


def decode_normal(path: Path, resolution: int, normal_length_tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Return normal [H,W,3] and valid supervision mask from an RGBA/RGB PNG."""
    with Image.open(path) as image:
        rgba = np.asarray(
            image.convert("RGBA").resize((resolution, resolution), Image.Resampling.NEAREST),
            dtype=np.float32,
        )
    normal = rgba[..., :3] / 255.0 * 2.0 - 1.0
    length = np.linalg.norm(normal, axis=-1)
    # Some converted DIODE files encode missing pixels as opaque black, so
    # alpha alone is insufficient. Unit-length is required for valid normals.
    valid = (rgba[..., 3] > 0) & (np.abs(length - 1.0) <= normal_length_tolerance)
    safe_normal = normal / np.maximum(length[..., None], 1e-8)
    return safe_normal.astype(np.float32), valid


def angular_degrees(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cosine = np.sum(left * right, axis=-1)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def neighbour_angles(normal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    horizontal_valid = valid[:, :-1] & valid[:, 1:]
    vertical_valid = valid[:-1, :] & valid[1:, :]
    horizontal = angular_degrees(normal[:, :-1], normal[:, 1:])[horizontal_valid]
    vertical = angular_degrees(normal[:-1, :], normal[1:, :])[vertical_valid]
    return np.concatenate((horizontal, vertical))


def sample_summary(pair: NormalPair, resolution: int, tolerance: float) -> tuple[dict, np.ndarray, np.ndarray]:
    normal, valid = decode_normal(pair.normal_path, resolution, tolerance)
    directions = normal[valid]
    angles = neighbour_angles(normal, valid)
    return {
        "input": str(pair.input_path.resolve()),
        "normal": str(pair.normal_path.resolve()),
        "valid_pixel_fraction": float(valid.mean()),
        "valid_pixel_count": int(valid.sum()),
        "mean_direction": directions.mean(axis=0).tolist() if len(directions) else [0.0, 0.0, 0.0],
        "neighbour_angular_error_mean_deg": float(angles.mean()) if len(angles) else 0.0,
        "neighbour_angular_error_p95_deg": float(np.quantile(angles, 0.95)) if len(angles) else 0.0,
    }, normal, valid


def write_panel(path: Path, pair: NormalPair, resolution: int) -> None:
    with Image.open(pair.input_path) as image:
        source = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
    with Image.open(pair.normal_path) as image:
        target = image.convert("RGB").resize((resolution, resolution), Image.Resampling.NEAREST)
    gap, label_height = 8, 28
    canvas = Image.new("RGB", (2 * resolution + 3 * gap, resolution + label_height + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), "RGB input", fill="black")
    draw.text((2 * gap + resolution, 6), "Camera normal target", fill="black")
    canvas.paste(source, (gap, label_height + gap))
    canvas.paste(target, (2 * gap + resolution, label_height + gap))
    canvas.save(path)


def aggregate_dataset(pairs: list[NormalPair], resolution: int, tolerance: float,
                      num_visualizations: int, output_dir: Path) -> dict:
    if not pairs:
        return {"discovered_pairs": 0, "audited_pairs": 0, "suitable_for_training": False}
    all_directions, all_neighbour_angles, samples = [], [], []
    valid_pixels, total_pixels = 0, 0
    for index, pair in enumerate(pairs):
        summary, normal, valid = sample_summary(pair, resolution, tolerance)
        directions = normal[valid]
        if len(directions):
            all_directions.append(directions)
        angles = neighbour_angles(normal, valid)
        if len(angles):
            all_neighbour_angles.append(angles)
        valid_pixels += int(valid.sum())
        total_pixels += int(valid.size)
        samples.append(summary)
        if index < num_visualizations:
            write_panel(output_dir / f"{pair.dataset}_sample_{index:02d}.png", pair, resolution)
    directions = np.concatenate(all_directions) if all_directions else np.empty((0, 3), dtype=np.float32)
    local_angles = np.concatenate(all_neighbour_angles) if all_neighbour_angles else np.empty(0, dtype=np.float32)
    global_direction = directions.mean(axis=0) if len(directions) else np.zeros(3, dtype=np.float32)
    global_direction /= max(float(np.linalg.norm(global_direction)), 1e-8)
    constant_angles = angular_degrees(directions, global_direction) if len(directions) else np.empty(0, dtype=np.float32)
    valid_fraction = valid_pixels / total_pixels if total_pixels else 0.0
    constant_mean = float(constant_angles.mean()) if len(constant_angles) else 0.0
    local_mean = float(local_angles.mean()) if len(local_angles) else 0.0
    suitable = bool(valid_fraction >= 0.2 and constant_mean >= 5.0 and local_mean >= 2.0)
    return {
        "discovered_pairs": len(pairs),
        "audited_pairs": len(samples),
        "resolution": resolution,
        "normal_encoding": "normal = normalize(2 * rgb / 255 - 1)",
        "normal_length_tolerance": tolerance,
        "valid_pixel_fraction": valid_fraction,
        "global_constant_direction": [float(value) for value in global_direction],
        "constant_direction_baseline": {
            "mean_angular_error_deg": constant_mean,
            "median_angular_error_deg": float(np.median(constant_angles)) if len(constant_angles) else 0.0,
            "within_5deg_fraction": float((constant_angles < 5.0).mean()) if len(constant_angles) else 0.0,
        },
        "local_directional_variation": {
            "neighbour_angular_error_mean_deg": local_mean,
            "neighbour_angular_error_median_deg": float(np.median(local_angles)) if len(local_angles) else 0.0,
            "neighbour_angular_error_p95_deg": float(np.quantile(local_angles, 0.95)) if len(local_angles) else 0.0,
        },
        "suitable_for_training": suitable,
        "diagnosis": "directionally_variable_valid_normals" if suitable else "insufficient_or_collapsed_supervision",
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    if args.resolution < 16:
        raise SystemExit("--resolution must be at least 16")
    if not 0.0 < args.normal_length_tolerance < 1.0:
        raise SystemExit("--normal-length-tolerance must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    discovered: dict[str, list[NormalPair]] = {}
    if args.dataset in {"diode", "both"}:
        discovered["diode"] = discover_diode(args.diode_root.expanduser().resolve())
    if args.dataset in {"hypersim", "both"}:
        discovered["hypersim"] = discover_hypersim(args.hypersim_root.expanduser().resolve())
    report = {
        "mode": "real_normal_dataset_audit",
        "max_samples": args.max_samples,
        "datasets": {},
    }
    for name, pairs in discovered.items():
        selected = evenly_spaced(pairs, args.max_samples)
        dataset_dir = args.output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result = aggregate_dataset(
            selected, args.resolution, args.normal_length_tolerance,
            args.num_visualizations, dataset_dir,
        )
        result["root"] = str((args.diode_root if name == "diode" else args.hypersim_root).expanduser().resolve())
        result["discovered_pairs"] = len(pairs)
        report["datasets"][name] = result
    (args.output_dir / "real_normal_dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
