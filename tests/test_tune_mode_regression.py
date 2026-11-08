"""Regressions for the existing tune_mode configuration field; no model loading."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from open_vision_banana.training.config import TrainingConfig, load_config
from open_vision_banana.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def test_default_mode_validates():
    config = load_config()
    config.validate()
    assert config.tune_mode == "lora"
    assert not hasattr(config, "train_mode")


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_yaml_mode_reaches_trainer_property_without_model_loading(tmp_path, mode):
    path = tmp_path / "mode.yaml"
    path.write_text(yaml.safe_dump({"training": {"tune_mode": mode}}))
    config = load_config(config_path=path)
    trainer = object.__new__(Trainer)
    trainer.config = config
    assert trainer.full is (mode == "full")


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_effective_config_entry_exits_without_trainer_or_output(tmp_path, mode):
    output = tmp_path / "must-not-exist"
    path = tmp_path / "mode.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "training": {"tune_mode": mode},
                "checkpoint": {"output_dir": str(output)},
            }
        )
    )
    code = (
        "import runpy,sys; "
        "import open_vision_banana.training.config; "
        "sys.modules['open_vision_banana.training.trainer']=None; "
        f"sys.argv=['train.py','--config',{str(path)!r},'--print-effective-config']; "
        f"runpy.run_path({str(ROOT / 'scripts/train.py')!r},run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["tune_mode"] == mode
    assert not output.exists()


def test_fusion_proposal_loads_real_configuration():
    config = load_config(config_path=ROOT / "configs/experiments/fusion_r64_large_scale.yaml")
    assert config.tune_mode == "lora"
    assert config.lora_rank == 64
    assert config.steps == 100000
    assert config.save_every == 10000
    assert config.eval_every == 2500
    assert "normal" not in config.tasks
    assert config.task_sampling_weights["referring"] == 0.15


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_invalid_mode_is_not_silently_defaulted(mode):
    config = TrainingConfig(tune_mode=mode + "_typo")
    with pytest.raises(ValueError, match="tune_mode must be lora or full"):
        config.validate()
