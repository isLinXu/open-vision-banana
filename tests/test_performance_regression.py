from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.compare_benchmarks import (
    ComparisonResult,
    RegressionReport,
    _benchmark_file_to_dict,
    _compare_metric,
    compare_benchmarks,
)


class TestCompareMetric:
    """Unit tests for _compare_metric."""

    def test_no_change(self) -> None:
        result = _compare_metric("op", "median_ms", 10.0, 10.0, 0.10)
        assert result.change_pct == 0.0
        assert not result.regression

    def test_small_change_within_threshold(self) -> None:
        result = _compare_metric("op", "median_ms", 10.0, 10.5, 0.10)
        assert result.change_pct == 5.0
        assert not result.regression

    def test_regression_above_threshold(self) -> None:
        result = _compare_metric("op", "median_ms", 10.0, 12.0, 0.10)
        assert result.change_pct == 20.0
        assert result.regression

    def test_improvement_negative_change(self) -> None:
        result = _compare_metric("op", "median_ms", 10.0, 8.0, 0.10)
        assert result.change_pct == -20.0
        assert not result.regression

    def test_zero_baseline(self) -> None:
        result = _compare_metric("op", "median_ms", 0.0, 5.0, 0.10)
        assert result.change_pct == float("inf")
        assert result.regression

    def test_zero_baseline_and_current(self) -> None:
        result = _compare_metric("op", "median_ms", 0.0, 0.0, 0.10)
        assert result.change_pct == 0.0
        assert not result.regression

    def test_throughput_increase_is_an_improvement(self) -> None:
        result = _compare_metric("depth", "throughput_img_per_sec", 100.0, 130.0, 0.15)
        assert result.change_pct == 30.0
        assert not result.regression

    def test_throughput_decrease_is_a_regression(self) -> None:
        result = _compare_metric("depth", "throughput_img_per_sec", 100.0, 80.0, 0.15)
        assert result.regression

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
    def test_invalid_timing_fails(self, value) -> None:
        with pytest.raises(ValueError, match="finite and nonnegative"):
            _compare_metric("depth", "median_ms", 1.0, value, 0.15)


class TestCompareBenchmarks:
    """Unit tests for compare_benchmarks."""

    def test_empty_inputs(self) -> None:
        report = compare_benchmarks({}, {}, threshold=0.10)
        assert not report.has_regression
        assert report.comparisons == []

    def test_single_operation_no_regression(self) -> None:
        baseline = {"Depth encode": {"median_ms": 10.0, "p95_ms": 12.0}}
        current = {"Depth encode": {"median_ms": 10.5, "p95_ms": 12.5}}
        report = compare_benchmarks(baseline, current, threshold=0.10)
        assert not report.has_regression
        assert len(report.comparisons) == 2  # median_ms + p95_ms

    def test_single_operation_regression(self) -> None:
        baseline = {"Depth encode": {"median_ms": 10.0}}
        current = {"Depth encode": {"median_ms": 12.0}}
        report = compare_benchmarks(baseline, current, threshold=0.10)
        assert report.has_regression
        assert len(report.regressions) == 1
        assert report.regressions[0].change_pct == 20.0

    def test_missing_current_operation(self) -> None:
        baseline = {"Depth encode": {"median_ms": 10.0}}
        current = {}
        report = compare_benchmarks(baseline, current, threshold=0.10)
        assert report.has_regression
        assert len(report.regressions) == 1
        assert report.regressions[0].operation == "Depth encode"

    def test_missing_current_metric_is_a_regression(self) -> None:
        report = compare_benchmarks(
            {"depth": {"median_ms": 1.0, "p95_ms": 2.0}},
            {"depth": {"median_ms": 1.0}},
            threshold=0.15,
        )
        assert report.has_regression
        assert report.regressions[0].metric == "p95_ms"

    def test_missing_baseline_operation(self) -> None:
        baseline = {}
        current = {"Depth encode": {"median_ms": 10.0}}
        report = compare_benchmarks(baseline, current, threshold=0.10)
        # Missing baseline should not be a regression, just a note
        assert not report.has_regression

    def test_non_numeric_metric_skipped(self) -> None:
        baseline = {"Depth encode": {"median_ms": 10.0, "label": "foo"}}
        current = {"Depth encode": {"median_ms": 12.0, "label": "bar"}}
        report = compare_benchmarks(baseline, current, threshold=0.10)
        # Only numeric metrics should be compared
        assert len(report.comparisons) == 1

    def test_multiple_operations(self) -> None:
        baseline = {
            "Depth encode": {"median_ms": 10.0},
            "Depth decode": {"median_ms": 8.0},
        }
        current = {
            "Depth encode": {"median_ms": 10.0},  # no change
            "Depth decode": {"median_ms": 10.0},  # 25% regression
        }
        report = compare_benchmarks(baseline, current, threshold=0.10)
        assert report.has_regression
        assert len(report.regressions) == 1
        assert report.regressions[0].operation == "Depth decode"

    def test_improvement_tracked(self) -> None:
        baseline = {"Depth encode": {"median_ms": 10.0}}
        current = {"Depth encode": {"median_ms": 7.0}}
        report = compare_benchmarks(baseline, current, threshold=0.10)
        assert not report.has_regression
        assert len(report.improvements) == 1
        assert report.improvements[0].change_pct == -30.0

    def test_different_thresholds(self) -> None:
        baseline = {"Depth encode": {"median_ms": 10.0}}
        current = {"Depth encode": {"median_ms": 12.0}}
        # 20% change with 0.25 threshold -> not a regression
        report = compare_benchmarks(baseline, current, threshold=0.25)
        assert not report.has_regression
        # 20% change with 0.10 threshold -> regression
        report = compare_benchmarks(baseline, current, threshold=0.10)
        assert report.has_regression


class TestBenchmarkFileToDict:
    """Unit tests for _benchmark_file_to_dict."""

    def test_flat_dict(self) -> None:
        data = {"op1": {"median_ms": 1.0}, "op2": {"median_ms": 2.0}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            result = _benchmark_file_to_dict(path)
            assert result == data
        finally:
            path.unlink()

    def test_list_format(self) -> None:
        data = [
            {
                "task": "depth",
                "median_latency_ms": 12.0,
                "p95_latency_ms": 15.0,
                "mean_latency_ms": 13.0,
                "throughput_img_per_sec": 80.0,
            },
            {
                "task": "normal",
                "median_latency_ms": 10.0,
                "p95_latency_ms": 12.0,
                "mean_latency_ms": 11.0,
                "throughput_img_per_sec": 100.0,
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            result = _benchmark_file_to_dict(path)
            assert "depth" in result
            assert "normal" in result
            assert result["depth"]["median_ms"] == 12.0
            assert result["depth"]["p95_ms"] == 15.0
            assert result["normal"]["throughput_img_per_sec"] == 100.0
        finally:
            path.unlink()

    def test_list_format_with_label(self) -> None:
        data = [
            {
                "label": "Depth encode",
                "median_ms": 5.0,
                "p95_ms": 6.0,
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)
        try:
            result = _benchmark_file_to_dict(path)
            assert "Depth encode" in result
            assert result["Depth encode"]["median_ms"] == 5.0
        finally:
            path.unlink()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            _benchmark_file_to_dict(Path("/nonexistent/benchmark.json"))

    def test_core_json_excludes_dispersion_and_metadata(self, tmp_path) -> None:
        path = tmp_path / "core.json"
        path.write_text(
            json.dumps(
                {
                    "depth": {
                        "median_ms": 1.0,
                        "p95_ms": 2.0,
                        "mean_ms": 1.2,
                        "std_ms": 0.2,
                        "min_ms": 0.9,
                        "max_ms": 2.2,
                    },
                    "_meta": {"iterations": 20, "resolution": 256},
                }
            )
        )
        assert _benchmark_file_to_dict(path) == {
            "depth": {"median_ms": 1.0, "p95_ms": 2.0, "mean_ms": 1.2},
        }

    def test_invalid_numeric_metric_rejected(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"depth":{"median_ms":NaN}}')
        with pytest.raises(ValueError, match="Invalid numeric"):
            _benchmark_file_to_dict(path)

    def test_cli_rejects_empty_current(self, tmp_path, monkeypatch) -> None:
        from scripts.compare_benchmarks import main

        path = tmp_path / "empty.json"
        path.write_text("{}")
        monkeypatch.setattr("sys.argv", ["compare_benchmarks.py", str(path), str(path)])
        assert main() == 2

    def test_cli_requires_measured_baseline(self, tmp_path, monkeypatch) -> None:
        from scripts.compare_benchmarks import main

        baseline = tmp_path / "baseline.json"
        current = tmp_path / "current.json"
        baseline.write_text("{}")
        current.write_text('{"depth":{"median_ms":1.0}}')
        monkeypatch.setattr("sys.argv", ["compare_benchmarks.py", str(baseline), str(current)])
        assert main() == 2

    def test_bootstrap_reports_absent_comparison(self, tmp_path, monkeypatch) -> None:
        from scripts.compare_benchmarks import main

        baseline = tmp_path / "baseline.json"
        current = tmp_path / "current.json"
        report = tmp_path / "report.json"
        baseline.write_text("{}")
        current.write_text('{"depth":{"median_ms":1.0}}')
        monkeypatch.setattr(
            "sys.argv",
            [
                "compare_benchmarks.py",
                str(baseline),
                str(current),
                "--allow-empty-baseline",
                "--json",
                str(report),
            ],
        )
        assert main() == 0
        assert json.loads(report.read_text())["comparison_status"] == "baseline_initialization"
        assert json.loads(report.read_text())["baseline_available"] is False


class TestRegressionReport:
    """Unit tests for RegressionReport."""

    def test_empty_report(self) -> None:
        report = RegressionReport()
        assert not report.has_regression
        md = report.to_markdown()
        assert "No regressions detected" in md

    def test_report_with_regression(self) -> None:
        report = RegressionReport()
        report.comparisons.append(
            ComparisonResult(
                operation="Depth encode",
                metric="median_ms",
                baseline=10.0,
                current=15.0,
                change_pct=50.0,
                regression=True,
            )
        )
        report.regressions.append(report.comparisons[0])
        assert report.has_regression
        md = report.to_markdown()
        assert "REGRESSION" in md
        assert "1 regression(s) detected" in md

    def test_report_markdown_format(self) -> None:
        report = RegressionReport()
        report.comparisons.append(
            ComparisonResult(
                operation="Depth encode",
                metric="median_ms",
                baseline=10.0,
                current=10.0,
                change_pct=0.0,
                regression=False,
            )
        )
        md = report.to_markdown()
        assert "# Performance Regression Report" in md
        assert "| Operation | Metric | Baseline | Current | Change % | Status |" in md
        assert "Depth encode" in md
        assert "🟢 OK" in md
