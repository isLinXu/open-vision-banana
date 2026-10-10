"""Paper capability groups for arXiv 2604.20329v3.

These are the Vision Banana *paper* numbers, used only as reference rows.
They are not measurements from this repository. Generation win rates are
human pairwise judgments, not CLIP or FID.
"""

from __future__ import annotations

from typing import Any

# Paper Table / section numbers. Do not copy these into "our" result columns.
PAPER_CAPABILITIES: dict[str, dict[str, dict[str, Any]]] = {
    "2d": {
        "cityscapes": {
            "benchmark": "semantic",
            "dataset": "cityscapes",
            "split": "val",
            "metric": "mIoU",
            "paper": 0.699,
            "note": "Cityscapes val, 19-class legend including absent classes",
        },
        "refcocog": {
            "benchmark": "referring",
            "dataset": "refcocog",
            "split": "val",
            "metric": "cIoU",
            "paper": 0.738,
            "note": "RefCOCOg UMD val",
        },
        "reasonseg": {
            "benchmark": "referring",
            "dataset": "reasonseg",
            "split": "val",
            "metric": "gIoU",
            "paper": 0.793,
            "note": "ReasonSeg val; paper rewrites queries with Gemini 2.5 Pro",
        },
        "saco": {
            "benchmark": "instance",
            "dataset": "saco",
            "split": "val",
            "metric": "cgF1",
            "paper": 47.5,
            "note": "SA-Co/Gold; paper filters NPs with Gemini 3.1 Flash-Lite",
        },
    },
    "3d": {
        "nyu": {
            "benchmark": "depth",
            "dataset": "nyu",
            "metric": "delta_1.25",
            "paper": 0.948,
            "paper_absrel": 0.081,
        },
        "ibims": {
            "benchmark": "depth",
            "dataset": "ibims",
            "metric": "delta_1.25",
            "paper": 0.934,
            "paper_absrel": 0.078,
        },
        "eth3d": {
            "benchmark": "depth",
            "dataset": "eth3d",
            "metric": "delta_1.25",
            "paper": 0.935,
            "paper_absrel": 0.103,
        },
        "diode": {
            "benchmark": "depth",
            "dataset": "diode",
            "metric": "delta_1.25",
            "paper": 0.917,
            "paper_absrel": 0.108,
            "note": "DIODE-Indoor",
        },
        "kitti": {
            "benchmark": "depth",
            "dataset": "kitti",
            "metric": "delta_1.25",
            "paper": 0.915,
            "paper_absrel": 0.107,
        },
        "nuscenes": {
            "benchmark": "depth",
            "dataset": "nuscenes",
            "metric": "delta_1.25",
            "paper": 0.643,
            "paper_absrel": 0.219,
            "note": "80 m cap; paper does not disclose a crop",
        },
    },
    "generation": {
        "genai_bench": {
            "benchmark": "generation",
            "dataset": "genai_bench",
            "metric": "human_win_rate",
            "paper": 0.535,
            "note": "GenAI-Bench T2I human pairwise; CLIP/FID are proxies only",
        },
        "imgedit": {
            "benchmark": "generation",
            "dataset": "imgedit",
            "metric": "human_win_rate",
            "paper": 0.478,
            "note": "ImgEdit human pairwise; not a reconstruction score",
        },
    },
}

CAPABILITY_TO_BENCHMARKS = {
    "2d": ("semantic", "referring", "instance"),
    "3d": ("depth", "normal"),
    "generation": ("generation",),
}


def capability_for_benchmark(benchmark: str, dataset: str) -> str:
    """Map a job onto the paper 2D / 3D / generation axis."""
    if benchmark == "generation":
        return "generation"
    if benchmark in {"depth", "normal"}:
        return "3d"
    if benchmark in {"semantic", "referring", "instance"}:
        return "2d"
    raise KeyError(f"Unknown paper benchmark {benchmark}/{dataset}")


def paper_reference(benchmark: str, dataset: str) -> dict[str, Any] | None:
    """Return the paper reference row, or None when this repo has no Table entry."""
    for group in PAPER_CAPABILITIES.values():
        row = group.get(dataset)
        if row is not None and row["benchmark"] == benchmark:
            return dict(row)
    if benchmark == "normal":
        return {
            "benchmark": "normal",
            "dataset": dataset,
            "metric": "mean_angle_deg",
            "note": "Paper Table 7 indoor/outdoor normals; 4-set mean 18.928°",
        }
    return None
