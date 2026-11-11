"""Default dual-annotation scoring shares inference and merges exact raw counts."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from open_vision_banana.data.paper_eval_datasets import ReasonSegDataset
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.evaluation.results_io import write_metrics_csv
from open_vision_banana.evaluation.shards import merged_metrics
from scripts.evaluate_paper_benchmarks import parse_args

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).parents[1] / "scripts/export_external_eval_comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class WhiteModel:
    calls = 0

    def eval(self):
        return self

    def predict(self, **kwargs):
        self.calls += 1
        return {"rgb_output": torch.ones(1, 3, 6, 6)}


@pytest.mark.parametrize("native", [False, True])
def test_two_annotation_scores_use_one_prediction_and_merge_with_empty_shard(tmp_path, native):
    Image.new("RGB", (12, 12)).save(tmp_path / "a.jpg")
    shapes = [dict(label=label, points=[[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
              for label, lo, hi in [("target", 0, 9), ("ignore", 2, 7), ("flag", 0, 11)]]
    (tmp_path / "a.json").write_text(json.dumps(dict(text=["first", "second"], shapes=shapes)))
    ds = ReasonSegDataset(tmp_path, resolution=6, native_gt=native, report_annotation_variants=True)
    model = WhiteModel()
    evaluator = VisionEvaluator(model, device="cpu")
    report = evaluator.evaluate(DataLoader(ds, batch_size=1), task="referring", native_resolution=native)
    assert model.calls == len(ds) == 2
    variants = report["annotation_variants"]
    assert variants["legacy_union"]["gIoU"] == report["gIoU"]
    assert variants["lisa_v1"]["gIoU"] != report["gIoU"]
    assert variants["lisa_v1"]["valid_pixels"] < variants["legacy_union"]["valid_pixels"]
    states = []
    for indices in ([0], [1], []):
        ev = VisionEvaluator(WhiteModel(), device="cpu")
        ev.evaluate(DataLoader(Subset(ds, indices), batch_size=1), task="referring", native_resolution=native)
        states.append(json.loads(json.dumps(ev.last_accumulator_states)))
    rebuilt = merged_metrics(states)
    for mode in variants:
        assert rebuilt["annotation_variants"][mode] == pytest.approx(variants[mode])
    states[-1].pop("annotation_variants")
    with pytest.raises(ValueError, match="annotation variants"):
        merged_metrics(states)


def test_cli_default_enables_both_and_explicit_opt_out(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["eval", "--output-json", "unused.json"])
    assert parse_args().reasonseg_annotation_variants
    monkeypatch.setattr(sys, "argv", ["eval", "--output-json", "unused.json", "--no-reasonseg-annotation-variants"])
    assert not parse_args().reasonseg_annotation_variants


def test_long_csv_keeps_nested_metrics_and_missing_class_diagnostics(tmp_path):
    result = dict(benchmarks={"referring/reasonseg": dict(decoded_metrics={
        "gIoU": .1, "per_class_iou": [.2, None],
        "annotation_variants": {"lisa_v1": {"gIoU": .8, "F-score": .9}},
    })})
    path = tmp_path / "full.csv"
    write_metrics_csv(path, result)
    with path.open() as handle:
        rows = {r["metric"]: r["value"] for r in csv.DictReader(handle)}
    assert rows["gIoU"] == "0.1"
    assert rows["annotation_variants.lisa_v1.F-score"] == "0.9"
    assert rows["per_class_iou.1"] == ""


def record(dataset="reasonseg", mode="lisa_v1"):
    return dict(model="m<script>", step=1, resolution=512, dataset=dataset, samples=2,
                protocol=dict(reasonseg_annotation_mode=mode, referring_prompt_version="query_v2"),
                metrics=dict(AbsRel=.5, gIoU=.8, cIoU=.7))


def test_named_columns_never_relabel_or_invent_scores(tmp_path):
    single = record()
    row = comparison.comparison_row(single)
    assert row["gIoU"] is None and row["ReasonSeg_lisa_v1_gIoU"] == .8
    assert row.get("AbsRel_per_image") is None
    single["metrics"]["annotation_variants"] = {
        "legacy_union": dict(gIoU=.2, cIoU=.3, valid_pixels=99),
        "lisa_v1": dict(gIoU=.8, cIoU=.7, valid_pixels=80, **{"F-score": .9}),
    }
    legacy = record(mode="legacy_union")
    rows = comparison.export([single, legacy], tmp_path)
    assert rows[0]["gIoU"] == .8  # the independent legacy record is preserved
    dual = next(r for r in rows if r["reasonseg_annotation_mode"] == "lisa_v1")
    assert dual["gIoU"] == .2 and dual["ReasonSeg_lisa_v1_gIoU"] == .8
    assert dual["reasonseg_lisa_v1_valid_pixels"] == 80
    text = (tmp_path / "comparison.html").read_text()
    assert "m&lt;script&gt;" in text and "m<script>" not in text
    assert "</td><details>" not in text
    with (tmp_path / "comparison.csv").open(encoding="utf-8-sig") as handle:
        output = list(csv.DictReader(handle))
    assert any(r["annotation_variants.lisa_v1.F-score"] == "0.9" for r in output)


def test_export_refuses_unmerged_shards(tmp_path):
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(dict(benchmarks={"referring/reasonseg": dict(shard={})})))
    with pytest.raises(ValueError, match="Merge shards"):
        list(comparison.load_full_result(path))
