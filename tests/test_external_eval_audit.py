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
        def __call__(self, prompt=None, height=None, width=None, **kwargs):
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
    monkeypatch.setattr(sys, "argv", ["audit", "--model-name", "flux2-klein", "--resolution", "512",
                                    "--benchmark", "depth", "--output-json", str(tmp_path / "r.json")])
    with pytest.raises(ValueError, match="Flux2KleinPipeline is not a diffusers pipeline"):
        audit.main()


class _Panel:  # the decoded [B, C, H, W] batch the generator returns
    def __init__(self, sizes):
        self.sizes = sizes

    def dim(self):
        return 4

    def __len__(self):
        return len(self.sizes)

    def __iter__(self):
        return iter(types.SimpleNamespace(shape=(3, h, w)) for w, h in self.sizes)


@pytest.mark.parametrize("sizes,valid", [([(512, 512)], True), ([(512, 512), (768, 512)], False)])
def test_moss_is_audited_at_the_generator_it_actually_uses(tmp_path, monkeypatch, sizes, valid):
    # MOSS-Image never calls a pipeline __call__: the generator denoises the right half of an
    # in-context panel itself, so the audit has to intercept that method instead.
    from open_vision_banana.inference.generator import BaseGenerator

    assert hasattr(BaseGenerator, "_generate_moss_panel"), "the audited generator method was renamed"
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace())
    monkeypatch.setattr(BaseGenerator, "_generate_moss_panel", lambda self, **kwargs: _Panel(sizes))
    stub = BaseGenerator._generate_moss_panel
    output = tmp_path / "result.json"

    def evaluate():
        BaseGenerator._generate_moss_panel(object(), prompt=["depth"] * len(sizes))
        output.write_text(json.dumps(dict(metadata={})))
        return 0

    monkeypatch.setattr(audit.runpy, "run_path", lambda *a, **k: {"main": evaluate})
    monkeypatch.setattr(sys, "argv", ["audit", "--model-name", "moss-image", "--resolution", "512",
                                    "--benchmark", "depth", "--output-json", str(output)])
    if valid:
        audit.main()
        meta = json.loads(output.read_text())["metadata"]
        assert meta["resolution_verification"] == {"raw_pipeline_output": [512, 512], "every_prediction_checked": True}
        assert json.loads(output.with_suffix(".resolution-audit.json").read_text())["predictions_verified"] == 1
    else:
        with pytest.raises(ValueError, match="Wrong raw moss-image output size"):
            audit.main()
    assert BaseGenerator._generate_moss_panel is stub  # the audit put the method back


def test_pipelines_without_a_canvas_argument_record_the_native_output(tmp_path, monkeypatch):
    class Pipeline:  # LongCat-Image-Edit: the pipeline picks its own canvas
        def __call__(self, image=None, prompt=None, num_inference_steps=50, guidance_scale=4.0):
            return types.SimpleNamespace(images=[Image.new("RGB", (1024, 1024))])

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(LongCatImageEditPipeline=Pipeline))
    output = tmp_path / "result.json"

    def evaluate():
        Pipeline()(prompt="depth")
        output.write_text(json.dumps(dict(metadata={})))
        return 0

    monkeypatch.setattr(audit.runpy, "run_path", lambda *a, **k: {"main": evaluate})
    monkeypatch.setattr(sys, "argv", ["audit", "--model-name", "longcat-image-edit", "--resolution", "512",
                                    "--benchmark", "depth", "--output-json", str(output)])
    audit.main()
    meta = json.loads(output.read_text())["metadata"]
    assert meta["resolution_verification"] == {"raw_pipeline_output": "pipeline_native", "every_prediction_checked": True}
