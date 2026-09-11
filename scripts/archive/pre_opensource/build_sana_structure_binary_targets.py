#!/usr/bin/env python3
"""Build sparse binary structural-edge targets for SANA specialization.

The original ``structure_aligned`` targets combine dilated Canny edges, LSD
lines, and a wide gray halo.  They occupy roughly half of the image while the
task prompt asks for thin binary edges, which encourages a diffuse response.
This script derives an isolated target set from the input images without
changing the source dataset or its input manifest.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS = PROJECT_ROOT / "output/mar_vb_clean_pairs_v4/pairs.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/sana_structure_binary_targets_v1/targets"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "output/sana_structure_binary_targets_v1/pairs.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--min-line-fraction",
        type=float,
        default=0.08,
        help="Discard LSD segments shorter than this fraction of image size.",
    )
    parser.add_argument(
        "--edge-width",
        type=int,
        default=3,
        help="Final binary edge width in pixels; must be positive and odd.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N structure rows.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_binary_structure_map(
    input_path: str | Path,
    resolution: int,
    min_line_fraction: float,
    edge_width: int,
) -> np.ndarray:
    """Return an RGB, 0/255, input-aligned structural edge target."""
    image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Could not read {input_path}")
    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    median = float(np.median(blurred))
    lower = max(0, int(0.80 * median))
    upper = min(255, max(lower + 1, int(1.60 * median)))
    canny = cv2.Canny(blurred, lower, upper)

    line_canvas = np.zeros_like(canny)
    min_length = resolution * min_line_fraction
    detector = cv2.createLineSegmentDetector(0)
    lines = detector.detect(blurred)[0]
    if lines is not None:
        # OpenCV 4.x has returned both ``(N, 1, 4)`` and ``(N, 4)``
        # layouts across releases.  Flatten only the singleton wrapper so
        # each segment remains an explicit four-coordinate row.
        segments = np.asarray(lines, dtype=np.float32).reshape(-1, 4)
        for x1, y1, x2, y2 in segments:
            if float(np.hypot(x2 - x1, y2 - y1)) < min_length:
                continue
            cv2.line(
                line_canvas,
                (round(float(x1)), round(float(y1))),
                (round(float(x2)), round(float(y2))),
                255,
                thickness=1,
                lineType=cv2.LINE_8,
            )

    edges = np.maximum(canny, line_canvas)
    if edge_width > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
        edges = cv2.dilate(edges, kernel, iterations=1)
    binary = np.where(edges > 0, 255, 0).astype(np.uint8)
    return np.repeat(binary[..., None], 3, axis=2)


def _write_target(payload: tuple[str, str, int, float, int, bool]) -> tuple[str, float, bool]:
    input_text, output_text, resolution, min_line_fraction, edge_width, overwrite = payload
    output_path = Path(output_text)
    if output_path.is_file() and not overwrite:
        existing = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        if existing is not None:
            return input_text, float((existing > 0).mean()), False

    target = build_binary_structure_map(input_text, resolution, min_line_fraction, edge_width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp.png")
    if not cv2.imwrite(str(temp_path), cv2.cvtColor(target, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write {temp_path}")
    temp_path.replace(output_path)
    return input_text, float((target[..., 0] > 0).mean()), True


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.resolution < 32:
        raise SystemExit("--resolution must be at least 32")
    if not 0.0 < args.min_line_fraction <= 1.0:
        raise SystemExit("--min-line-fraction must be in (0, 1]")
    if args.edge_width < 1 or args.edge_width % 2 == 0:
        raise SystemExit("--edge-width must be a positive odd integer")
    if not args.pairs.is_file():
        raise SystemExit(f"Pairs manifest not found: {args.pairs}")

    rows = [json.loads(line) for line in args.pairs.open(encoding="utf-8") if line.strip()]
    structure_rows = [row for row in rows if row.get("task") == "structure"]
    if args.limit is not None:
        structure_rows = structure_rows[: args.limit]
    if not structure_rows:
        raise SystemExit("No structure rows found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, str, int, float, int, bool]] = []
    output_paths: dict[str, Path] = {}
    for row in structure_rows:
        input_path = Path(row["input"])
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing structure input: {input_path}")
        output_path = args.output_dir / f"{input_path.stem}.png"
        jobs.append(
            (
                str(input_path),
                str(output_path),
                args.resolution,
                args.min_line_fraction,
                args.edge_width,
                args.overwrite,
            )
        )
        output_paths[str(input_path)] = output_path.resolve()

    densities: list[float] = []
    written = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (_, density, did_write) in enumerate(pool.map(_write_target, jobs), 1):
            densities.append(density)
            written += int(did_write)
            if index % 500 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)}] binary structure targets ready", flush=True)

    manifest_rows = []
    for row in rows:
        replacement = output_paths.get(str(Path(row.get("input", ""))))
        if replacement is not None:
            row = dict(row)
            row["target"] = str(replacement)
            row["target_kind"] = "sparse_binary_structure_edges_v1"
        manifest_rows.append(row)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input_pairs": str(args.pairs.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "target_dir": str(args.output_dir.resolve()),
        "structure_pairs_processed": len(structure_rows),
        "targets_written": written,
        "resolution": args.resolution,
        "min_line_fraction": args.min_line_fraction,
        "edge_width": args.edge_width,
        "mean_positive_fraction": float(np.mean(densities)),
        "min_positive_fraction": float(np.min(densities)),
        "max_positive_fraction": float(np.max(densities)),
    }
    report_path = args.output_manifest.with_suffix(".binary_structure.audit.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
