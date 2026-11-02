"""Focused checks for required experiment artifacts and benchmark coverage."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.check_experiment_gates import E1_E2_BENCHMARKS, GateReport, check_data, check_eval

ROOT = Path(__file__).resolve().parents[1]


def _statuses(report: GateReport) -> dict[str, str]:
    return {result.name: result.status for result in report.results}


def test_data_gate_fails_on_missing_required_artifacts(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    report = check_data(
        merged_audit=None,
        final_manifest=missing,
        hypersim_manifest=None,
        tasks=["depth"],
        model_name="qwen-image-2.1",
    )
    statuses = _statuses(report)
    assert statuses["merge.audit_present"] == "FAIL"
    assert statuses["final.audit_present"] == "FAIL"
    assert statuses["hypersim.manifest_present"] == "FAIL"
    assert statuses["contract.no_errors"] == "FAIL"
    assert "0/4 gates passed" in report.render()

    cli = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_experiment_gates.py"),
         "data", "--final-manifest", str(missing)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 1


def test_data_gate_checks_manifest_bytes_against_audit_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text('{"task":"depth"}\n', encoding="utf-8")
    audit = {"manifest_sha256": "0" * 64, "task_split_counts": {"depth/train": 1, "depth/val": 1}}
    manifest.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps({"per_task_split": {"depth": {"train": 1, "val": 1}}}), encoding="utf-8")
    hypersim = tmp_path / "hypersim.json"
    hypersim.write_text(json.dumps({"metric_depth": {"convention": "planar_z_depth"},
                                    "data_contract": {"encoded_roundtrip_abs_error": {"mean": 0.001}}}), encoding="utf-8")
    validation = types.ModuleType("open_vision_banana.data.validation")
    validation.validate_manifest_contract = lambda *args, **kwargs: SimpleNamespace(errors=[], warnings=[])
    monkeypatch.setitem(sys.modules, validation.__name__, validation)

    kwargs = dict(merged_audit=merged, final_manifest=manifest, hypersim_manifest=hypersim,
                  tasks=["depth"], model_name="qwen-image-2.1")
    assert _statuses(check_data(**kwargs))["final.manifest_sha256"] == "FAIL"

    audit["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")
    assert _statuses(check_data(**kwargs))["final.manifest_sha256"] == "PASS"


def _paper_results() -> dict:
    return {
        "benchmarks": {
            key: {"split": "val", "samples": 2, "source_samples": 2,
                  "decoded_metrics": {metric: 0.5 for metric in metrics}}
            for key, metrics in E1_E2_BENCHMARKS.items()
        }
    }


def test_e1_e2_gate_accepts_complete_paper_coverage(tmp_path: Path) -> None:
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(_paper_results()), encoding="utf-8")
    for profile in ("e1", "e2"):
        report = check_eval(eval_json=path, profile=profile)
        assert not report.failed, report.render()


@pytest.mark.parametrize("change", ["missing", "wrong_dataset", "null_metric", "zero_samples", "train_split", "capped"])
def test_e1_e2_gate_rejects_incomplete_paper_coverage(tmp_path: Path, change: str) -> None:
    results = _paper_results()
    record = results["benchmarks"]["semantic/cityscapes"]
    if change == "missing":
        del results["benchmarks"]["semantic/cityscapes"]
    elif change == "wrong_dataset":
        results["benchmarks"]["semantic/cityscapes8"] = results["benchmarks"].pop("semantic/cityscapes")
    elif change == "null_metric":
        record["decoded_metrics"]["mIoU"] = None
    elif change == "zero_samples":
        record["samples"] = 0
    elif change == "capped":
        record["source_samples"] = 200
    else:
        record["split"] = "train"
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    report = check_eval(eval_json=path, profile="e1")
    assert any(gate.name.startswith("eval.semantic/cityscapes.") for gate in report.failed), report.render()
