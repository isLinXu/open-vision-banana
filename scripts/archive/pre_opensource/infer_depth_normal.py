#!/usr/bin/env python3
"""Generate a relative-depth-derived surface normal map at image resolution.

This is a task-specialist fallback for v5 normal estimation.  The historic v5
normal labels use unscaled per-pixel finite differences, which makes almost all
vectors point toward the camera.  Here gradients are converted from pixel to
normalised-image coordinates before constructing unit camera-space normals.
The result is a geometric visualisation derived from Depth-Anything relative
depth, not metric or sensor-calibrated surface normal ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from infer_depth_pseudo import estimate_depth, load_depth_estimator, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-image", type=Path)
    parser.add_argument("--depth-output", type=Path, help="Optional 16-bit relative-depth array (.npy).")
    parser.add_argument("--resolution", type=int, default=0, help="Square output size; 0 preserves input resolution.")
    parser.add_argument(
        "--normal-strength",
        type=float,
        default=1.0,
        help="Multiplier for gradients in normalised image coordinates.",
    )
    parser.add_argument("--model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    return parser.parse_args()


def normal_from_relative_depth(depth: np.ndarray, normal_strength: float = 1.0) -> np.ndarray:
    """Convert [H,W] relative depth to unit camera-space normals in [-1,1].

    Relative depth has arbitrary scale, so each image is normalised before
    differentiation.  ``np.gradient`` returns per-pixel derivatives; multiply
    by H/W to express those slopes over the image's [0,1] coordinate range.
    Image rows grow downward while camera-space y grows upward, hence ``ny``
    uses the positive row derivative.
    """
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"Expected a depth map of at least 2x2, got {values.shape}")
    if normal_strength <= 0:
        raise ValueError("normal_strength must be positive")
    minimum, maximum = float(values.min()), float(values.max())
    if maximum - minimum < 1e-6:
        normal = np.zeros((*values.shape, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        return normal
    relative = (values - minimum) / (maximum - minimum)
    derivative_y, derivative_x = np.gradient(relative)
    derivative_x *= values.shape[1] * normal_strength
    derivative_y *= values.shape[0] * normal_strength
    normal = np.stack((-derivative_x, derivative_y, np.ones_like(relative)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1e-8)
    return normal.astype(np.float32)


def normal_to_rgb(normal: np.ndarray) -> Image.Image:
    """Encode camera-space normals with the project's R=(1-x)/2 convention."""
    values = np.asarray(normal, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Expected [H,W,3] normal map, got {values.shape}")
    encoded = np.stack(
        ((1.0 - values[..., 0]) * 0.5, (1.0 + values[..., 1]) * 0.5, (1.0 + values[..., 2]) * 0.5),
        axis=-1,
    )
    return Image.fromarray(np.rint(encoded.clip(0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def angular_variation_degrees(normal: np.ndarray) -> dict[str, float]:
    """Summarise local normal variation, which detects collapsed constant maps."""
    horizontal = np.sum(normal[:, :-1] * normal[:, 1:], axis=-1)
    vertical = np.sum(normal[:-1, :] * normal[1:, :], axis=-1)
    angles = np.rad2deg(np.arccos(np.clip(np.concatenate((horizontal.ravel(), vertical.ravel())), -1.0, 1.0)))
    return {
        "neighbour_angular_error_mean_deg": float(angles.mean()),
        "neighbour_angular_error_median_deg": float(np.median(angles)),
        "neighbour_angular_error_p95_deg": float(np.quantile(angles, 0.95)),
    }


def write_panel(path: Path, source: Image.Image, normal_rgb: Image.Image) -> None:
    preview_size, gap, label_height = 320, 8, 28
    source = source.convert("RGB").resize((preview_size, preview_size), Image.Resampling.BICUBIC)
    normal_rgb = normal_rgb.resize((preview_size, preview_size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (2 * preview_size + 3 * gap, preview_size + label_height + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), "Input", fill="black")
    draw.text((preview_size + 2 * gap, 6), "Relative-depth-derived normal", fill="black")
    canvas.paste(source, (gap, label_height + gap))
    canvas.paste(normal_rgb, (preview_size + 2 * gap, label_height + gap))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    if args.resolution < 0:
        raise SystemExit("--resolution must be non-negative")
    if args.normal_strength <= 0:
        raise SystemExit("--normal-strength must be positive")
    input_path = args.input_image.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input image not found: {input_path}")
    device = select_device(args.device)
    model, processor = load_depth_estimator(args.model, device)
    with Image.open(input_path) as image:
        source = image.convert("RGB")
    depth = estimate_depth(source, model, processor, device)
    if args.resolution:
        depth_tensor = torch.from_numpy(depth)[None, None]
        depth = torch.nn.functional.interpolate(
            depth_tensor, size=(args.resolution, args.resolution), mode="bilinear", align_corners=False
        ).squeeze().numpy()
    normal = normal_from_relative_depth(depth, args.normal_strength)
    normal_rgb = normal_to_rgb(normal)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_image = (args.output_image or args.output_dir / f"{input_path.stem}_normal_from_depth.png").expanduser().resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    normal_rgb.save(output_image)
    if args.depth_output is not None:
        depth_output = args.depth_output.expanduser().resolve()
        depth_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_output, depth)
    write_panel(args.output_dir / f"{input_path.stem}_comparison.png", source, normal_rgb)
    report = {
        "mode": "single_image",
        "input": str(input_path),
        "output": str(output_image),
        "depth_output": str(args.depth_output.expanduser().resolve()) if args.depth_output else None,
        "model": args.model,
        "local_files_only": True,
        "device": str(device),
        "normal_source": "relative_depth_derived",
        "output_size": list(normal_rgb.size),
        "normal_strength": args.normal_strength,
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        **angular_variation_degrees(normal),
    }
    (args.output_dir / "normal_inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
