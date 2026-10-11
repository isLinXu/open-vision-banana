#!/usr/bin/env python3
"""Benchmark result comparison tool for performance regression CI.

Compares current benchmark results against a baseline and reports regressions.

Usage:
    # Compare two result files
    python scripts/compare_benchmarks.py baseline.json current.json

    # With custom threshold (default 10%)
    python scripts/compare_benchmarks.py baseline.json current.json --threshold 0.15

    # Generate markdown report
    python scripts/compare_benchmarks.py baseline.json current.json --report report.md

Exit codes:
    0  — no significant regressions
    1  — one or more regressions exceeded threshold
    2  — input file missing or malformed
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LATENCY_METRICS = {"median_ms", "p95_ms", "mean_ms"}
THROUGHPUT_METRICS = {"throughput_img_per_sec", "throughput"}
COMPARABLE_METRICS = LATENCY_METRICS | THROUGHPUT_METRICS


@dataclass
class ComparisonResult:
    """Single benchmark comparison entry."""

    operation: str
    metric: str
    baseline: float
    current: float
    change_pct: float
    regression: bool


@dataclass
class RegressionReport:
    """Full regression report."""

    comparisons: list[ComparisonResult] = field(default_factory=list)
    regressions: list[ComparisonResult] = field(default_factory=list)
    improvements: list[ComparisonResult] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        return any(r.regression for r in self.comparisons)

    def to_markdown(self) -> str:
        lines = [
            "# Performance Regression Report\n",
            "| Operation | Metric | Baseline | Current | Change % | Status |\n",
            "|-----------|--------|----------|---------|----------|--------|\n",
        ]
        for c in self.comparisons:
            status = "🔴 REGRESSION" if c.regression else "🟢 OK"
            lines.append(
                f"| {c.operation} | {c.metric} | {c.baseline:.2f} | "
                f"{c.current:.2f} | {c.change_pct:+.1f}% | {status} |\n"
            )
        lines.append("\n")
        if self.regressions:
            lines.append(f"**{len(self.regressions)} regression(s) detected.**\n")
        else:
            lines.append("**No regressions detected.**\n")
        return "".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON benchmark file, raising clear errors on failure."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data


def _compare_metric(
    operation: str,
    metric: str,
    baseline: float,
    current: float,
    threshold: float,
) -> ComparisonResult:
    """Compare latency (lower is better) or throughput (higher is better)."""
    if any(not math.isfinite(value) or value < 0 for value in (baseline, current, threshold)):
        raise ValueError("Benchmark values and regression threshold must be finite and nonnegative")
    if baseline == 0:
        change_pct = 0.0 if current == 0 else float("inf")
    else:
        change_pct = (current - baseline) / baseline * 100.0

    regression = (
        change_pct < -threshold * 100.0
        if metric in THROUGHPUT_METRICS
        else change_pct > threshold * 100.0
    )
    return ComparisonResult(
        operation=operation,
        metric=metric,
        baseline=baseline,
        current=current,
        change_pct=change_pct,
        regression=regression,
    )


def compare_benchmarks(
    baseline_data: dict[str, Any],
    current_data: dict[str, Any],
    threshold: float = 0.10,
) -> RegressionReport:
    """Compare two benchmark dictionaries and produce a regression report.

    Expects both inputs to be dicts with operation names as keys and
    metric dicts as values, e.g.:

        {
            "Depth encode": {"median_ms": 12.3, "p95_ms": 15.0},
            "Depth decode": {"median_ms": 8.5, "p95_ms": 10.2},
        }
    """
    report = RegressionReport()
    baseline_ops = set(baseline_data.keys())
    current_ops = set(current_data.keys())

    all_ops = sorted(baseline_ops | current_ops)
    for op in all_ops:
        if op not in baseline_data:
            report.comparisons.append(
                ComparisonResult(
                    operation=op,
                    metric="missing",
                    baseline=0.0,
                    current=0.0,
                    change_pct=0.0,
                    regression=False,
                )
            )
            continue
        if op not in current_data:
            report.comparisons.append(
                ComparisonResult(
                    operation=op,
                    metric="missing",
                    baseline=0.0,
                    current=0.0,
                    change_pct=0.0,
                    regression=True,
                )
            )
            report.regressions.append(report.comparisons[-1])
            continue

        b_metrics = baseline_data[op]
        c_metrics = current_data[op]
        if not isinstance(b_metrics, dict) or not isinstance(c_metrics, dict):
            continue

        for metric in sorted(b_metrics):
            b_val = b_metrics[metric]
            if not isinstance(b_val, int | float):
                continue
            c_val = c_metrics.get(metric)
            if not isinstance(c_val, int | float):
                missing = ComparisonResult(op, metric, float(b_val), 0.0, 0.0, True)
                report.comparisons.append(missing)
                report.regressions.append(missing)
                continue
            comp = _compare_metric(op, metric, float(b_val), float(c_val), threshold)
            report.comparisons.append(comp)
            if comp.regression:
                report.regressions.append(comp)
            elif (
                comp.change_pct > threshold * 100.0
                if metric in THROUGHPUT_METRICS
                else comp.change_pct < -threshold * 100.0
            ):
                report.improvements.append(comp)

    return report


def _benchmark_file_to_dict(path: Path) -> dict[str, Any]:
    """Normalise either a flat dict or a list of BenchmarkResult objects."""
    data = _load_json(path)
    if isinstance(data, list):
        # List of benchmark results (e.g. from benchmark_pipeline.py)
        flat: dict[str, Any] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Every benchmark result must be a metric mapping")
            op_name = item.get("task", item.get("label", "unknown"))
            if not isinstance(op_name, str) or not op_name:
                raise ValueError("Benchmark operation names must be nonempty strings")
            if op_name in flat:
                raise ValueError(f"Duplicate benchmark operation: {op_name}")
            aliases = {
                "median_ms": "median_latency_ms",
                "p95_ms": "p95_latency_ms",
                "mean_ms": "mean_latency_ms",
                "throughput_img_per_sec": "throughput",
            }
            flat[op_name] = {
                metric: item[metric] if metric in item else item[alias]
                for metric, alias in aliases.items()
                if metric in item or alias in item
            }
        data = flat
    if not isinstance(data, dict):
        raise ValueError("Benchmark JSON must be an operation mapping or result list")
    normalized = {}
    for operation, metrics in data.items():
        if operation.startswith("_"):
            continue
        if not isinstance(metrics, dict):
            raise ValueError(f"Benchmark operation {operation} must have a metric mapping")
        # Dispersion/extrema and run metadata are diagnostic, not latency gates.
        selected = {key: value for key, value in metrics.items() if key in COMPARABLE_METRICS}
        if not selected:
            raise ValueError(f"No comparable metrics for benchmark operation {operation}")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
            for value in selected.values()
        ):
            raise ValueError(f"Invalid numeric benchmark metric for {operation}")
        normalized[operation] = selected
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark results for regressions")
    parser.add_argument("baseline", type=Path, help="Path to baseline JSON")
    parser.add_argument("current", type=Path, help="Path to current JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Regression threshold as a fraction (e.g. 0.10 = 10%%)",
    )
    parser.add_argument("--report", type=Path, default=None, help="Markdown report output path")
    parser.add_argument("--json", type=Path, default=None, help="Structured JSON output path")
    parser.add_argument(
        "--allow-empty-baseline",
        action="store_true",
        help="Explicit first-main-run bootstrap; reports that no regression comparison occurred",
    )
    args = parser.parse_args()

    try:
        baseline = _benchmark_file_to_dict(args.baseline)
        current = _benchmark_file_to_dict(args.current)
        if not current:
            raise ValueError("Current benchmark has no measured operations")
        if not baseline and not args.allow_empty_baseline:
            raise ValueError("Baseline has no measured operations; comparison cannot be performed")
        if not math.isfinite(args.threshold) or args.threshold < 0:
            raise ValueError("Regression threshold must be finite and nonnegative")
        report = compare_benchmarks(baseline, current, threshold=args.threshold)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    markdown = (
        report.to_markdown()
        if baseline
        else (
            "# Performance baseline initialization\n\n"
            "No previous baseline is available. Current measurements were validated, "
            "but performance regression was **not assessed**.\n"
        )
    )
    print(markdown)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Report saved to: {args.report}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline_available": bool(baseline),
                    "comparison_status": "compared" if baseline else "baseline_initialization",
                    "has_regression": report.has_regression,
                    "regression_count": len(report.regressions),
                    "comparisons": [
                        {
                            "operation": c.operation,
                            "metric": c.metric,
                            "baseline": c.baseline,
                            "current": c.current,
                            "change_pct": c.change_pct,
                            "regression": c.regression,
                        }
                        for c in report.comparisons
                    ],
                },
                f,
                indent=2,
            )
        print(f"JSON saved to: {args.json}")

    return 1 if report.has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
