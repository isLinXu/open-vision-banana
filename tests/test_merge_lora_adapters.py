"""scripts/merge_lora_adapters.py reproduces sum_i w_i * delta_W_i exactly."""

import importlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from open_vision_banana.inference.checkpoint import read_task_prompt_stats, read_training_prompts

merge = importlib.import_module("scripts.merge_lora_adapters")

MODULES = ("transformer_blocks.0.attn.to_q", "transformer_blocks.0.img_mlp.out")


def _write_adapter(path, seed, rank=8, alpha=8, rslora=True, modules=MODULES):
    gen = torch.Generator().manual_seed(seed)
    state = {}
    for module in modules:
        prefix = f"base_model.model.{module}"
        state[f"{prefix}.lora_A.weight"] = torch.randn(rank, 16, generator=gen).to(torch.bfloat16)
        state[f"{prefix}.lora_B.weight"] = torch.randn(24, rank, generator=gen).to(torch.bfloat16)
    path.mkdir(parents=True)
    save_file(state, str(path / "adapter_model.safetensors"))
    config = {"r": rank, "lora_alpha": alpha, "use_rslora": rslora,
              "target_modules": ["to_q", "img_mlp.out"]}
    (path / "adapter_config.json").write_text(json.dumps(config))
    return path


def _delta(path, module):
    config = json.loads((path / "adapter_config.json").read_text())
    state = load_file(str(path / "adapter_model.safetensors"))
    rank, alpha = config["r"], config["lora_alpha"]
    scale = alpha / rank**0.5 if config.get("use_rslora") else alpha / rank
    prefix = f"base_model.model.{module}"
    a, b = state[f"{prefix}.lora_A.weight"], state[f"{prefix}.lora_B.weight"]
    return scale * b.float() @ a.float()


def _merge(out, *specs):
    merge.write_merged([merge.load_adapter(*merge.parse_spec(str(s))) for s in specs], out)
    return out


def test_weighted_merge_is_exact(tmp_path):
    a = _write_adapter(tmp_path / "a", seed=0)
    b = _write_adapter(tmp_path / "b", seed=1, rank=4, alpha=16, rslora=False)
    out = _merge(tmp_path / "m", f"{a}:0.3", f"{b}:0.7")
    config = json.loads((out / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["use_rslora"]) == (12, 12, False)
    for module in MODULES:
        expected = 0.3 * _delta(a, module) + 0.7 * _delta(b, module)
        torch.testing.assert_close(_delta(out, module), expected, rtol=1e-5, atol=1e-5)
    recipe = json.loads((out / "merge_recipe.json").read_text())
    assert [i["weight"] for i in recipe["inputs"]] == [0.3, 0.7]


def test_single_adapter_scales_strength(tmp_path):
    a = _write_adapter(tmp_path / "a", seed=0)
    unit = _merge(tmp_path / "unit", a)
    half = _merge(tmp_path / "half", f"{a}:0.5")
    for module in MODULES:
        torch.testing.assert_close(_delta(unit, module), _delta(a, module), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            _delta(half, module), 0.5 * _delta(a, module), rtol=1e-5, atol=1e-5
        )


def test_parse_spec_keeps_non_numeric_suffix_in_path():
    assert merge.parse_spec("ckpt/lora_adapter:0.25") == (merge.Path("ckpt/lora_adapter"), 0.25)
    assert merge.parse_spec("ckpt/lora_adapter") == (merge.Path("ckpt/lora_adapter"), 1.0)
    assert merge.parse_spec("C:/x") == (merge.Path("C:/x"), 1.0)


def test_mismatched_modules_are_rejected(tmp_path):
    a = _write_adapter(tmp_path / "a", seed=0)
    b = _write_adapter(tmp_path / "b", seed=1, modules=MODULES[:1])
    with pytest.raises(ValueError, match="adapted modules differ"):
        _merge(tmp_path / "m", a, b)


def _write_metadata(adapter_dir, prompts, step):
    meta = {"step": step, "model_name": "qwen-image-2.1", "task_prompts": prompts,
            "task_prompt_stats": {"semantic": {"palette": "cityscapes19"}}}
    (adapter_dir.parent / "metadata.json").write_text(json.dumps(meta))


def test_training_prompts_are_carried_over(tmp_path):
    a = _write_adapter(tmp_path / "ck20" / "lora_adapter", seed=0)
    b = _write_adapter(tmp_path / "ck100" / "lora_adapter", seed=1)
    prompts = {"depth": "Generate a metric depth visualization of this image."}
    _write_metadata(a, prompts, 20000)
    _write_metadata(b, prompts, 100000)
    out = _merge(tmp_path / "m", f"{a}:0.5", f"{b}:0.5")
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["task_prompts"] == prompts
    assert meta["task_prompt_stats"]["semantic"]["palette"] == "cityscapes19"
    assert meta["source_model_name"] == "qwen-image-2.1"
    assert [m["step"] for m in meta["merged_from"]] == [20000, 100000]
    assert read_training_prompts(out) == prompts
    assert read_task_prompt_stats(out)["semantic"]["palette"] == "cityscapes19"


def test_different_training_prompts_are_rejected(tmp_path):
    a = _write_adapter(tmp_path / "ck20" / "lora_adapter", seed=0)
    b = _write_adapter(tmp_path / "ck100" / "lora_adapter", seed=1)
    _write_metadata(a, {"depth": "old prompt"}, 20000)
    _write_metadata(b, {"depth": "new prompt"}, 100000)
    with pytest.raises(ValueError, match="different task_prompts"):
        _merge(tmp_path / "m", a, b)
