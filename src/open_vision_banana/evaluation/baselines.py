"""Published SOTA baseline metrics for automatic comparison.

Values sourced from the Vision Banana paper (Table 1) and README benchmarks.
Used by ``scripts/evaluate.py --compare-sota`` and ``scripts/benchmark.py``.
"""

from __future__ import annotations

# metric_name -> {method: value}
# Lower-is-better metrics marked with _lower suffix in compare logic.
SOTA_BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "depth": {
        "delta_1.25": {
            "Vision Banana": 0.882,
            "UniK3D": 0.823,
            "DPT": 0.780,
        },
        "abs_rel": {
            "Vision Banana": 0.058,
            "UniK3D": 0.072,
            "DPT": 0.089,
        },
        "rmse": {
            "Vision Banana": 0.312,
            "UniK3D": 0.385,
            "DPT": 0.420,
        },
    },
    "normal": {
        "mean_angular_error": {
            "Vision Banana": 15.05,
            "Lotus-2": 15.72,
            "Omnidata": 18.20,
        },
    },
    "semantic": {
        "miou": {
            "Vision Banana": 69.9,
            "SAM 3": 65.2,
            "Mask2Former": 62.1,
        },
    },
    "instance": {
        "ap_50": {
            "Vision Banana": 47.5,
            "OWLv2": 24.6,
            "SAM 2": 35.0,
        },
    },
    "referring": {
        "ciou": {
            "Vision Banana": 73.8,
            "LocHeads": 67.7,
            "SAM3+GPT": 71.0,
        },
    },
}

_LOWER_IS_BETTER = frozenset({
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "mean_angular_error",
    "median_angular_error",
})


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
