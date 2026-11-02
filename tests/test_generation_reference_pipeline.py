"""GenAI prompt references and fixed-step checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from scripts.build_generation_reference import build_reference
from scripts.check_eval_checkpoint import check_checkpoint
from scripts.check_experiment_gates import check_generation


def _statuses(report) -> dict[str, str]:
    return {result.name: result.status for result in report.results}


def test_reference_builds_paired_images_and_reuses_identical_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts_root = tmp_path / "prompts"
    prompts_root.mkdir()
    (prompts_root / "prompts.jsonl").write_text(
        '{"id":"a","prompt":"a bright house"}\n{"id":"b","prompt":"a red car"}\n', encoding="utf-8"
    )
    calls = []

    class FakeGenerator:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def generate(self, **kwargs):
            calls.append(("generate", kwargs))
            return torch.ones(1, 3, kwargs["height"], kwargs["width"])

    module = types.ModuleType("open_vision_banana.inference.generator")
    module.BaseGenerator = FakeGenerator
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "reference"
    kwargs = dict(prompts_root=prompts_root, output_dir=output, model_name="qwen-image-2.1",
                  dtype="float32", resolution=8, steps=2, seed=42, guidance_scale=4.0)
    spec = build_reference(**kwargs)
    assert spec["samples"] == 2
    assert [call[1]["seed"] for call in calls if call[0] == "generate"] == [42, 43]
    assert all(call[1]["guidance_scale"] == 4.0 for call in calls if call[0] == "generate")
    assert len(list((output / "images").glob("*.png"))) == 2
    assert len((output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    build_reference(**kwargs)
    assert len([call for call in calls if call[0] == "generate"]) == 2
    with pytest.raises(ValueError, match="parameters changed"):
        build_reference(**{**kwargs, "guidance_scale": 0.0})


def test_generation_gate_rejects_incomplete_or_mismatched_result(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    (reference / "images").mkdir(parents=True)
    (reference / "images" / "000000.png").write_bytes(b"png")
    (reference / "metadata.jsonl").write_text('{"image":"000000.png","prompt":"test"}\n', encoding="utf-8")
    spec = {"samples": 1, "model_name": "qwen-image-2.1", "seed": 42,
            "num_inference_steps": 50, "guidance_scale": 4.0, "dtype": "bfloat16",
            "prompts_sha256": hashlib.sha256(b'["test"]').hexdigest()}
    (reference / "reference_config.json").write_text(json.dumps(spec), encoding="utf-8")
    report_path = tmp_path / "gen.json"
    report = {
        "metadata": {"dataset_root": str(reference), "model_name": "qwen-image-2.1", "seed": 42,
                     "num_inference_steps": 50, "guidance_scale": 4.0, "dtype": "bfloat16",
                     "requested_samples": 1, "successful_samples": 1, "failed_samples": 0},
        "metrics": {"FID_proxy": 1.0, "diversity_score": 0.0, "perceptual_similarity": 0.1},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert not check_generation(eval_json=report_path, reference_dir=reference).failed
    report["metadata"]["successful_samples"] = 0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert _statuses(check_generation(eval_json=report_path, reference_dir=reference))["generation.all_samples_scored"] == "FAIL"


def test_checkpoint_must_be_complete_online_and_match_report(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-10000"
    adapter = checkpoint / "lora_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    metadata = checkpoint / "metadata.json"
    metadata.write_text('{"adapter_weight_source":"online"}', encoding="utf-8")
    report = tmp_path / "eval.json"
    report.write_text(json.dumps({"metadata": {"lora_path": str(checkpoint), "dtype": "bfloat16",
                                               "inference_protocol": {"num_inference_steps": 50,
                                                                      "guidance_scale": 4.0, "seed": 42}}}), encoding="utf-8")
    check_checkpoint(checkpoint, report, expected_steps=50, expected_guidance=4.0,
                     expected_seed=42, expected_dtype="bfloat16")
    with pytest.raises(ValueError, match="guidance_scale"):
        check_checkpoint(checkpoint, report, expected_guidance=0.0)
    report.write_text(json.dumps({"metadata": {"lora_path": str(tmp_path / "other")}}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected"):
        check_checkpoint(checkpoint, report)
    metadata.write_text('{"adapter_weight_source":"ema"}', encoding="utf-8")
    with pytest.raises(ValueError, match="online"):
        check_checkpoint(checkpoint)
