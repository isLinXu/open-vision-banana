"""Kontext depth-gate experiment: gates, config wiring and inference branch."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from open_vision_banana.evaluation.gates import evaluate_depth_gate

ROOT = Path(__file__).resolve().parent.parent

BASELINES = {
    "constant@3m": {"delta_1.25": 0.30, "AbsRel": 0.50},
    "prior_mean_map": {"delta_1.25": 0.42, "AbsRel": 0.34},
    "oracle_constant": {"delta_1.25": 0.45, "AbsRel": 0.30},
    "oracle_image_median": {"delta_1.25": 0.60, "AbsRel": 0.25},
}


def _model(delta1=0.75, absrel=0.20, aligned=0.80):
    return {"delta_1.25": delta1, "AbsRel": absrel, "delta_1.25_aligned": aligned}


def test_gate_passes_when_both_conditions_hold():
    r = evaluate_depth_gate(_model(), BASELINES, margin=0.10)
    assert r.passed and r.metric_gate and r.structure_gate
    assert r.reference_baseline == "prior_mean_map"  # oracles never define the bar
    assert r.required_delta1 == pytest.approx(0.52)


def test_gate_ignores_oracle_baselines_for_the_bar():
    # beating only the oracle-constant (0.45) must not be enough: bar is prior_mean_map + margin
    r = evaluate_depth_gate(_model(delta1=0.50), BASELINES, margin=0.10)
    assert not r.metric_gate and not r.passed


def test_gate_fails_on_absrel_even_with_good_delta():
    r = evaluate_depth_gate(_model(absrel=0.40), BASELINES, margin=0.10)
    assert not r.metric_gate


def test_structure_gate_catches_correct_scale_blob():
    # right scale (high delta1) but aligned delta1 below "perfect-scale constant" => no structure
    r = evaluate_depth_gate(_model(delta1=0.58, aligned=0.55), BASELINES, margin=0.10)
    assert r.metric_gate and not r.structure_gate and not r.passed


def test_gate_input_validation():
    with pytest.raises(ValueError):
        evaluate_depth_gate(_model(), {"oracle_image_median": {"delta_1.25": 0.6, "AbsRel": 0.2}})
    with pytest.raises(ValueError):
        evaluate_depth_gate(_model(), {"constant@3m": BASELINES["constant@3m"]})
    with pytest.raises(KeyError):
        evaluate_depth_gate({"delta_1.25": 0.7}, BASELINES)


def test_check_depth_gate_cli_exit_codes(tmp_path):
    def run(model_metrics):
        model_json = tmp_path / "model.json"
        model_json.write_text(
            json.dumps({"benchmarks": {"depth/nyu": {"decoded_metrics": model_metrics}}})
        )
        base_json = tmp_path / "base.json"
        base_json.write_text(json.dumps({"baselines": BASELINES}))
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_depth_gate.py"),
                "--model-json",
                str(model_json),
                "--baseline-json",
                str(base_json),
            ],
            capture_output=True,
            text=True,
        )

    assert run(_model()).returncode == 0
    assert run(_model(delta1=0.45)).returncode == 1


def test_experiment_yaml_maps_to_training_config():
    from open_vision_banana.training.config import load_config

    cfg = load_config(config_path=ROOT / "configs" / "experiments" / "flux_kontext_depth_gate.yaml")
    assert cfg.model_name == "flux-kontext"
    assert cfg.quantize == "4bit" and cfg.tasks == ["depth"]
    assert cfg.img2img_flow is False and cfg.vision_ratio == 1.0
    assert cfg.min_snr is True and cfg.lora_rank == 32 and cfg.device == "cuda"
    from open_vision_banana.model_registry import get_model_config

    assert get_model_config(cfg.model_name).in_context_conditioning


def test_generate_kontext_uses_official_pipeline_conventions():
    from open_vision_banana.inference.generator import BaseGenerator
    from open_vision_banana.model_registry import get_model_config

    calls = []

    class _Out:
        def __init__(self, size):
            self.images = [Image.new("RGB", size, (10, 20, 30))]

    class _Pipe:
        def __call__(self, **kw):
            calls.append(kw)
            return _Out((kw["width"], kw["height"]))

    gen = object.__new__(BaseGenerator)
    gen.model_config = get_model_config("flux-kontext")
    gen._t2i_pipe = _Pipe()
    gen.device = "cpu"

    src = Image.new("RGB", (70, 50))  # not multiples of 16 -> floored for the pipe
    out = gen._generate_kontext(
        prompt="depth", image=src, num_inference_steps=1, guidance_scale=0.0, seed=3
    )
    kw = calls[0]
    assert kw["_auto_resize"] is False  # keep the input raster; no preferred-size resize
    assert (kw["height"], kw["width"]) == (48, 64)
    assert (
        kw["num_inference_steps"] == 30
    )  # 1 means "unspecified": model default, never single step
    assert kw["guidance_scale"] == 2.5
    assert out.shape == (1, 3, 50, 70)  # resized back to the input raster
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_generate_kontext_prompt_count_mismatch():
    from open_vision_banana.inference.generator import BaseGenerator
    from open_vision_banana.model_registry import get_model_config

    gen = object.__new__(BaseGenerator)
    gen.model_config = get_model_config("flux-kontext")
    gen._t2i_pipe = object()
    gen.device = "cpu"
    with pytest.raises(ValueError, match="prompts"):
        gen._generate_kontext(
            prompt=["a", "b"],
            image=Image.new("RGB", (32, 32)),
            num_inference_steps=20,
            guidance_scale=2.0,
            seed=None,
        )
    assert torch  # silence unused import in minimal envs
