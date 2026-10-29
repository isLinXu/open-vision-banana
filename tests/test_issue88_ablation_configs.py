"""Issue #88 continuation ablations change only task composition, ratio, depth palettes and checkpoint cadence."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "configs/experiments/qwen_image_21_main_depth_reasoning.yaml"
PALETTES = ["cube_edge", "plasma", "inferno", "viridis", "grayscale"]
EXPERIMENTS = {
    "full_replay10": (
        ["depth", "normal", "semantic", "instance", "referring", "generation"],
        0.9,
        PALETTES,
    ),
    "no_normal": (["depth", "semantic", "instance", "referring", "generation"], 0.9, PALETTES),
    "depth_replay": (["depth", "generation"], 0.75, PALETTES),
    "seg_replay": (["semantic", "instance", "referring", "generation"], 0.75, PALETTES),
    "depth_cube100": (["depth", "generation"], 0.75, ["cube_edge"]),
}


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_issue88_config_settings(name):
    from open_vision_banana.training.config import load_config

    tasks, vision_ratio, palettes = EXPERIMENTS[name]
    config = load_config(
        config_path=ROOT / f"configs/experiments/qwen_image_21_ablation_{name}.yaml"
    )
    assert config.tasks == tasks
    assert config.vision_ratio == vision_ratio
    assert config.task_sampling == "uniform" and config.task_sampling_weights == {}
    assert config.depth_colormaps == palettes
    assert config.depth_edge_aux_weight == 0.0
    assert config.resume_weights_only is True
    assert (config.steps, config.save_every, config.lr, config.seed) == (10000, 2000, 3e-5, 42)
    assert (config.lora_rank, config.lora_alpha, config.batch_size) == (32, 64, 1)
    assert config.output_dir == f"output/qwen_image_21_ablation_{name}"


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_issue88_config_matches_main_recipe_otherwise(name):
    main = yaml.safe_load(MAIN.read_text())
    raw = yaml.safe_load(
        (ROOT / f"configs/experiments/qwen_image_21_ablation_{name}.yaml").read_text()
    )
    for recipe in (main, raw):
        for key in ("tasks", "vision_ratio", "depth_colormaps"):
            del recipe["data"][key]
        for key in ("output_dir", "save_every"):
            del recipe["checkpoint"][key]
    assert raw == main
