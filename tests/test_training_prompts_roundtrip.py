"""A LoRA from the unified trainer must be prompted at inference as it was trained.

The trainer's default task prompts ("Generate a relative depth map. ...") differ from
PromptBuilder's ("Generate a metric depth visualization of this image."), so checkpoints
record the prompt of each task trained with a single prompt in metadata.json, and the
inference pipeline reads them back.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.inference import pipeline as pipeline_module
from open_vision_banana.inference.checkpoint import read_training_prompts
from open_vision_banana.inference.pipeline import VisionBanana
from open_vision_banana.inference.prompt import PromptBuilder
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT
from open_vision_banana.training.callbacks import TrainingCallbacks
from open_vision_banana.training.trainer import TASK_PROMPTS, uniform_task_prompts


def save_checkpoint(tmp_path, task_prompts):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    callbacks = TrainingCallbacks(output_dir=tmp_path, task_prompts=task_prompts)
    ckpt_dir = callbacks.on_checkpoint(3, model, optimizer, scheduler)
    (ckpt_dir / "lora_adapter").mkdir(exist_ok=True)
    return ckpt_dir


def make_pipeline(training_prompts=None) -> VisionBanana:
    model = object.__new__(VisionBanana)
    model.prompt_builder = PromptBuilder()
    model.training_prompts = dict(training_prompts or {})
    return model


def test_checkpoint_metadata_records_task_prompts(tmp_path):
    ckpt_dir = save_checkpoint(tmp_path, TASK_PROMPTS)
    meta = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["task_prompts"] == TASK_PROMPTS
    assert meta["step"] == 3


def test_checkpoint_metadata_without_prompts_is_unchanged(tmp_path):
    ckpt_dir = save_checkpoint(tmp_path, None)
    meta = json.loads((ckpt_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "task_prompts" not in meta


def test_read_training_prompts_from_checkpoint_dir_adapter_dir_and_file(tmp_path):
    ckpt_dir = save_checkpoint(tmp_path, TASK_PROMPTS)
    assert read_training_prompts(str(ckpt_dir)) == TASK_PROMPTS
    assert read_training_prompts(str(ckpt_dir / "lora_adapter")) == TASK_PROMPTS
    assert read_training_prompts(str(ckpt_dir / "lora.pt")) == TASK_PROMPTS


def test_parent_metadata_is_only_read_for_the_adapter_subdirectory(tmp_path):
    ckpt_dir = save_checkpoint(tmp_path, TASK_PROMPTS)
    other = ckpt_dir / "someone_elses_lora"
    other.mkdir()
    assert read_training_prompts(str(other)) == {}


def test_metric_depth_rows_train_as_depth_with_the_row_prompt(tmp_path):
    root = tmp_path / "hypersim"
    (root / "images").mkdir(parents=True)
    (root / "targets").mkdir()
    (root / "images" / "a.png").write_bytes(b"x")
    (root / "targets" / "a.png").write_bytes(b"x")
    prompt = "Generate a metric depth visualization of this image."
    manifest = root / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task": "metric_depth",
                "input": "images/a.png",
                "target": "targets/a.png",
                "split": "train",
                "prompt": prompt,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    from open_vision_banana.training.trainer import ManifestVisionDataset

    dataset = ManifestVisionDataset(manifest, 8, "train", {"depth"}, data_root=root)
    assert len(dataset) == 1
    assert dataset.samples[0]["task"] == "depth"
    assert dataset.samples[0]["prompt"] == prompt


def test_uniform_task_prompts_skips_tasks_with_per_sample_prompts():
    depth = SimpleNamespace(samples=[{"task": "depth", "prompt": "D"}] * 3)
    seg = SimpleNamespace(
        samples=[
            {"task": "semantic", "prompt": "legend A"},
            {"task": "semantic", "prompt": "legend B"},
        ]
    )
    empty = SimpleNamespace(samples=[{"task": "normal", "prompt": ""}])
    assert uniform_task_prompts(SimpleNamespace(datasets=[depth, seg, empty])) == {"depth": "D"}
    assert uniform_task_prompts(depth) == {"depth": "D"}  # a single, unconcatenated dataset


def test_read_training_prompts_is_empty_when_there_is_no_record(tmp_path):
    assert read_training_prompts(None) == {}
    assert read_training_prompts("hf:org/some-lora") == {}
    assert read_training_prompts(str(tmp_path)) == {}
    assert read_training_prompts(str(save_checkpoint(tmp_path, None))) == {}
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "metadata.json").write_text("{not json", encoding="utf-8")
    assert read_training_prompts(str(broken)) == {}
    (broken / "metadata.json").write_text('{"task_prompts": ["depth"]}', encoding="utf-8")
    assert read_training_prompts(str(broken)) == {}


def test_pipeline_prefers_training_prompts():
    model = make_pipeline({"depth": TASK_PROMPTS["depth"], "semantic": TASK_PROMPTS["semantic"]})
    assert model._build_prompt("depth") == TASK_PROMPTS["depth"]
    assert model._build_prompt("semantic") == TASK_PROMPTS["semantic"]
    # no record for this task: PromptBuilder's default
    assert model._build_prompt("normal") == PromptBuilder.build_normal_prompt()
    # an explicit legend is the caller's prompt choice
    legend = {"road": (128, 64, 128)}
    assert model._build_prompt("semantic", class_color_map=legend) == (
        PromptBuilder().build_semantic_prompt(legend)
    )


def test_from_pretrained_loads_the_checkpoint_prompts(tmp_path, monkeypatch):
    ckpt_dir = save_checkpoint(tmp_path, TASK_PROMPTS)
    generator = MagicMock()
    generator.source_conditioning_config = None
    monkeypatch.setattr(pipeline_module, "BaseGenerator", MagicMock(return_value=generator))

    model = VisionBanana.from_pretrained(
        model_name="sana-1600m", lora_path=str(ckpt_dir / "lora_adapter"), device="cpu"
    )

    assert model.training_prompts == TASK_PROMPTS
    assert model._build_prompt("depth") == TASK_PROMPTS["depth"]


def test_multi_colormap_training_names_the_decode_colormap_in_metadata():
    """cube_edge rows get an explicit palette prompt, and the recorded inference
    prompt names the default decode colormap — no bare 'metric depth' ambiguity."""
    from open_vision_banana.training.trainer import depth_colormap_prompt

    assert depth_colormap_prompt("cube_edge", "metric_power") == (
        "Generate a metric depth visualization of this image using the cube_edge colormap."
    )
    assert depth_colormap_prompt("viridis", "log_relative") == (
        "Generate a relative depth map of this image using the viridis RGB palette."
    )


def test_task_prompt_stats_records_palette_contract_flags(tmp_path):
    from open_vision_banana.training.trainer import task_prompt_stats

    part = SimpleNamespace(
        samples=[
            {"task": "semantic", "prompt": SEMANTIC_SEGMENTATION_PROMPT},
            {"task": "semantic", "prompt": "Generate a 150-class semantic segmentation mask."},
            {"task": "referring", "prompt": "the red car"},
            {"task": "depth", "prompt": "Generate a metric depth visualization of this image."},
        ]
    )
    stats = task_prompt_stats(SimpleNamespace(datasets=[part]))
    assert stats["semantic"]["ade150_palette"] is True
    assert stats["semantic"]["cityscapes_legend"] is True
    assert stats["semantic"]["distinct_prompts"] == 2
    assert stats["depth"]["ade150_palette"] is False
    assert stats["referring"]["distinct_prompts"] == 1


def test_read_task_prompt_stats_from_metadata(tmp_path):
    ckpt_dir = save_checkpoint(tmp_path, TASK_PROMPTS)
    meta_path = ckpt_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["task_prompt_stats"] = {"semantic": {"ade150_palette": True}}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    from open_vision_banana.inference.checkpoint import read_task_prompt_stats

    assert read_task_prompt_stats(str(ckpt_dir))["semantic"]["ade150_palette"] is True
    assert read_task_prompt_stats(str(ckpt_dir / "missing_meta")) == {}


def test_cube_edge_train_rows_get_a_named_colormap_prompt(tmp_path):
    """With depth_colormaps active, even the default-colormap rows name their palette."""
    img_dir = tmp_path / "images"
    tgt_dir = tmp_path / "targets"
    img_dir.mkdir()
    tgt_dir.mkdir()
    Image.new("RGB", (32, 32), (90, 90, 90)).save(img_dir / "a.png")
    depth_rgb = DepthRGBMapper(colormap="cube_edge").depth_to_rgb(torch.full((1, 1, 32, 32), 2.0))[0]
    Image.fromarray((depth_rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(
        tgt_dir / "d.png"
    )
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task": "depth",
                "input": "images/a.png",
                "target": "targets/d.png",
                "split": "train",
                "prompt": "Generate a metric depth visualization of this image.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    from open_vision_banana.training.trainer import ManifestVisionDataset

    ds = ManifestVisionDataset(
        manifest, 32, "train", {"depth"}, data_root=tmp_path, depth_colormaps=["cube_edge"], seed=1
    )
    assert ds[0]["prompt"] == (
        "Generate a metric depth visualization of this image using the cube_edge colormap."
    )
