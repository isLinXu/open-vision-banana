import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from open_vision_banana.referring_contract import build_binary_referring_prompt

spec = importlib.util.spec_from_file_location("audit_entry", Path(__file__).parents[1] / "scripts/evaluate_external_audited.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("size,prompt,valid", [
    (512, build_binary_referring_prompt("the red car"), True),
    (768, build_binary_referring_prompt("the red car"), False),
    (512, "wrong template", False),
])
def test_audit_checks_raw_output_and_actual_prompt(tmp_path, monkeypatch, size, prompt, valid):
    class Pipeline:
        def __call__(self, prompt=None, output_resolution=None, **kwargs):
            return types.SimpleNamespace(images=[Image.new("RGB", (size, size))])

    original = Pipeline.__call__
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(QwenImage21Pipeline=Pipeline))
    output = tmp_path / "result.json"

    def evaluate():
        Pipeline()(prompt=prompt, output_resolution=512)
        output.write_text(json.dumps(dict(metadata={})))
        return 0

    monkeypatch.setattr(audit.runpy, "run_path", lambda *args, **kwargs: {"main": evaluate})
    monkeypatch.setattr(sys, "argv", ["audit", "--resolution", "512", "--benchmark", "referring",
                                    "--referring-prompt-version", "query_v2", "--output-json", str(output)])
    if valid:
        audit.main()
        result = json.loads(output.read_text())
        assert result["metadata"]["resolution_verification"]["every_prediction_checked"]
        proof = json.loads(output.with_suffix(".resolution-audit.json").read_text())
        assert proof["predictions_verified"] == proof["referring_prompts_verified"] == 1
    else:
        with pytest.raises(ValueError):
            audit.main()
    assert Pipeline.__call__ is original


@pytest.mark.parametrize("height,valid", [(512, True), (768, False)])
def test_audit_intercepts_the_requested_editor_pipeline(tmp_path, monkeypatch, height, valid):
    class Pipeline:
        def __call__(self, **kwargs):
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512))])

    original = Pipeline.__call__
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(Flux2KleinPipeline=Pipeline))
    output = tmp_path / "result.json"

    def evaluate():
        Pipeline()(prompt="depth", height=height, width=512)
        output.write_text(json.dumps(dict(metadata={})))
        return 0

    monkeypatch.setattr(audit.runpy, "run_path", lambda *args, **kwargs: {"main": evaluate})
    monkeypatch.setattr(sys, "argv", ["audit", "--model-name", "flux2-klein", "--resolution", "512",
                                    "--benchmark", "depth", "--output-json", str(output)])
    if valid:
        audit.main()
        assert json.loads(output.read_text())["metadata"]["resolution_verification"]["every_prediction_checked"]
    else:
        with pytest.raises(ValueError, match="flux2-klein"):
            audit.main()
    assert Pipeline.__call__ is original


def test_qwen_calls_must_request_output_resolution(tmp_path, monkeypatch):
    class Pipeline:
        def __call__(self, prompt=None, output_resolution=None, **kwargs):
            return types.SimpleNamespace(images=[Image.new("RGB", (512, 512))])

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(QwenImage21Pipeline=Pipeline))
    output = tmp_path / "result.json"
    monkeypatch.setattr(audit.runpy, "run_path", lambda *a, **k: {"main": lambda: Pipeline()(prompt="x", height=512, width=512)})
    monkeypatch.setattr(sys, "argv", ["audit", "--resolution", "512", "--benchmark", "depth", "--output-json", str(output)])
    with pytest.raises(ValueError, match="resolution"):
        audit.main()


def test_backbones_outside_diffusers_are_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace())
    monkeypatch.setattr(sys, "argv", ["audit", "--model-name", "moss-image", "--resolution", "512",
                                    "--benchmark", "depth", "--output-json", str(tmp_path / "r.json")])
    with pytest.raises(ValueError, match="MOSSImagePipeline is not a diffusers pipeline"):
        audit.main()
