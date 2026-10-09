"""Published baseline metrics from the Vision Banana paper for comparison.

Every value below is transcribed from arXiv:2604.20329v3 (Tables 1-7).
Keys match the metric names returned by ``open_vision_banana.evaluation``
so ``format_sota_comparison`` can look them up directly.

Caveats when comparing a local run against these numbers:

* The paper numbers are *aggregates* (6- or 4-dataset averages for depth and
  normal). A single-dataset local run is only indicative, never a Table 1
  reproduction.
* ``mIoU`` / ``cIoU`` are stored as fractions in [0, 1] to match the
  evaluators; ``cgF1`` follows SA-Co (already scaled by 100).
* SAM 3 and similar models marked non-zero-shot in the paper are omitted
  here unless noted.
"""

from __future__ import annotations

# metric_name -> {method: value}
# Lower-is-better metrics are listed in _LOWER_IS_BETTER.
SOTA_BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "depth": {
        # Table 6, average over 6 datasets (NYU, iBims-1, ETH3D, DIODE-Indoor,
        # KITTI, nuScenes). Depth Anything 3 / DepthLM report only partial
        # averages and are therefore not listed.
        "delta_1.25": {
            "Vision Banana": 0.882,
            "UniK3D": 0.823,
            "MoGe-2": 0.802,
            "Depth Pro": 0.715,
        },
    },
    "normal": {
        # Table 1: mean angle error averaged over NYUv2, DIODE-indoor,
        # ScanNet and VKITTI (Table 7 per-dataset values average to this).
        "Mean Angular Error (°)": {
            "Vision Banana": 18.928,
            "Lotus-2": 19.642,
        },
    },
    "semantic": {
        # Table 2, Cityscapes val, zero-shot transfer.
        "mIoU": {
            "Vision Banana": 0.699,
            "SAM 3": 0.652,
            "X-Decoder": 0.520,
            "OpenSeeD": 0.478,
            "APE-D": 0.442,
        },
    },
    "instance": {
        # Table 3, SA-Co/Gold, zero-shot transfer. Vision Banana is paired with
        # Gemini 3.1 Flash-Lite for the positive/negative filter.
        "cgF1": {
            "Vision Banana": 47.5,
            "OWLv2": 24.6,
            "DINO-X": 21.3,
            "APE-D": 16.4,
        },
    },
    "referring": {
        # Table 4, RefCOCOg val (UMD), zero-shot transfer.
        "cIoU": {
            "Vision Banana": 0.738,
            "SAM 3 + Gemini 2.5 Pro": 0.734,
            "LocalizationHeads": 0.677,
            "HybridGL": 0.513,
        },
    },
}

_LOWER_IS_BETTER = frozenset(
    {
        "abs_rel",
        "AbsRel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "mean_angular_error",
        "median_angular_error",
        "Mean Angular Error (°)",
        "Median Angular Error (°)",
    }
)


def format_sota_comparison(
    task: str,
    metrics: dict[str, float],
    *,
    model_name: str = "Ours",
) -> str:
    """Return a markdown-style comparison table for *metrics* vs published SOTA."""
    baselines = SOTA_BASELINES.get(task, {})
    if not baselines:
        return f"No SOTA baselines registered for task={task!r}"

    lines = [
        f"## SOTA Comparison ({task})",
        "",
        "| Metric | " + " | ".join(["Ours"] + _baseline_methods(task)) + " |",
        "|--------|" + "|".join(["--------"] * (1 + len(_baseline_methods(task)))) + "|",
    ]

    for metric_name, sota_vals in baselines.items():
        ours = metrics.get(metric_name)
        if ours is None:
            continue
        row = [metric_name, f"{ours:.4f}"]
        for method in _baseline_methods(task):
            if method in sota_vals:
                row.append(f"{sota_vals[method]:.4f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

        if metric_name in _LOWER_IS_BETTER:
            best = min([ours] + list(sota_vals.values()))
        else:
            best = max([ours] + list(sota_vals.values()))
        if abs(ours - best) < 1e-9:
            lines.append(f"  → **Ours leads on {metric_name}**")

    return "\n".join(lines)


def _baseline_methods(task: str) -> list[str]:
    baselines = SOTA_BASELINES.get(task, {})
    methods: list[str] = []
    for vals in baselines.values():
        for name in vals:
            if name != "Vision Banana" and name not in methods:
                methods.append(name)
    return methods


__all__ = ["SOTA_BASELINES", "format_sota_comparison"]
