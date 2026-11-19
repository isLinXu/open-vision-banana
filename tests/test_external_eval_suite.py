"""Portable plan generation must not load weights or launch GPU processes."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("suite", Path(__file__).parents[1] / "scripts/run_external_eval_suite.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def test_dry_run_inventory_and_arguments(tmp_path, monkeypatch, capsys):
    config = dict(reasonseg_annotation_mode="lisa_v1", models=[dict(name="base", base_path="/models/base")],
                  data_roots={d: f"/data/{d}" for d in suite.SPECS})
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["suite", "--config", str(path), "--output", str(tmp_path / "out")])
    monkeypatch.setattr(suite, "run_process", lambda *args: pytest.fail("Dry run launched a process"))
    suite.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["samples_per_model"] == 7173 and len(plan["gpus"]) == 8
    assert not (tmp_path / "out").exists()
    args = suite.command(config, config["models"][0], "reasonseg", tmp_path / "result.json")
    assert args[args.index("--reasonseg-annotation-mode") + 1] == "lisa_v1"
    assert args[args.index("--split") + 1] == "val"
    assert args[args.index("--batch-size") + 1] == "1"
    assert args[args.index("--model-name") + 1] == "qwen-image-2.1"  # default backbone
    other = dict(config["models"][0], model_name="longcat-image-edit")
    args = suite.command(config, other, "nyu", tmp_path / "result.json")
    assert args[args.index("--model-name") + 1] == "longcat-image-edit"


def test_reject_incomplete_or_nonfinite_result(tmp_path):
    path = tmp_path / "result.json"
    record = dict(samples=1, decoded_metrics=dict(samples=1, cIoU=.5))
    path.write_text(json.dumps(dict(benchmarks={"referring/reasonseg": record})))
    suite.validate(path, "reasonseg", 1)
    record["dataset_fingerprint"] = "actual"
    path.write_text(json.dumps(dict(benchmarks={"referring/reasonseg": record})))
    with pytest.raises(ValueError, match="Dataset changed"):
        suite.validate(path, "reasonseg", 1, fingerprint="other")
    record["decoded_metrics"]["annotation_variants"] = {
        key: dict(samples=1, gIoU=.5, valid_pixels=3) for key in ("legacy_union", "lisa_v1")
    }
    metadata = dict(reasonseg_annotation_mode="lisa_v1",
                    resolution_verification=dict(raw_pipeline_output=[512, 512], every_prediction_checked=True),
                    referring_prompt_verification=dict(version="query_v2", every_prediction_checked=True))
    path.write_text(json.dumps(dict(metadata=metadata, benchmarks={"referring/reasonseg": record})))
    suite.validate(path, "reasonseg", 1, fingerprint="actual", config=dict(reasonseg_annotation_mode="lisa_v1"))
    metadata["resolution_verification"]["raw_pipeline_output"] = [768, 768]
    path.write_text(json.dumps(dict(metadata=metadata, benchmarks={"referring/reasonseg": record})))
    with pytest.raises(ValueError, match="resolution"):
        suite.validate(path, "reasonseg", 1, config=dict(reasonseg_annotation_mode="lisa_v1"))
    record["decoded_metrics"]["cIoU"] = None
    path.write_text(json.dumps(dict(benchmarks={"referring/reasonseg": record})))
    with pytest.raises(ValueError):
        suite.validate(path, "reasonseg", 1)


def test_invalid_semantic_predictions_cannot_reduce_denominator():
    import torch

    from open_vision_banana.evaluation.benchmark import SemanticAccumulator

    accumulator = SemanticAccumulator(2)
    gt = torch.tensor([[0, 1, 255]])
    accumulator.update(torch.tensor([[0, 0, 255]]), gt)
    assert accumulator.compute()["valid_pixels"] == 2
    assert accumulator.compute()["mIoU"] == pytest.approx(.25)
    with pytest.raises(ValueError, match="Predicted class"):
        accumulator.update(torch.tensor([[0, 255, 0]]), gt)


def test_semantic_smoke_allows_absent_classes_but_not_invalid_headline(tmp_path):
    import torch

    from open_vision_banana.evaluation.benchmark import SemanticAccumulator
    from open_vision_banana.evaluation.results_io import write_results

    accumulator = SemanticAccumulator(19)
    accumulator.update(torch.zeros(2, 2), torch.zeros(2, 2))
    path = tmp_path / "smoke.json"
    record = dict(samples=1, decoded_metrics=accumulator.compute())
    result = dict(benchmarks={"semantic/cityscapes": record})
    write_results(path, result)
    suite.validate(path, "cityscapes", 1)
    record["decoded_metrics"]["mIoU"] = float("nan")
    write_results(path, result)
    with pytest.raises(ValueError, match="invalid metrics"):
        suite.validate(path, "cityscapes", 1)
