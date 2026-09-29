"""FLUX.1-dev training configs must request 4-bit loading.

``scripts/train.py`` only takes the quantized FLUX path when ``quantize: "4bit"``
is set; otherwise ``ModelBackend`` loads the 12B transformer unquantized.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from open_vision_banana.training.config import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
FLUX_DEV = {
    path.name: config
    for path in sorted(CONFIGS.glob("flux_*.yaml"))
    if (config := load_config(config_path=path)).model_name == "flux-dev"
}
assert FLUX_DEV, "no flux-dev configs found under configs/"


@pytest.mark.parametrize("name", sorted(FLUX_DEV))
def test_flux_dev_config_quantized(name: str) -> None:
    assert FLUX_DEV[name].quantize == "4bit"


def test_training_extra_declares_bitsandbytes() -> None:
    """The 4-bit path uses BitsAndBytesConfig, so [training] must install bitsandbytes."""
    text = (CONFIGS.parent / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^training = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert block, "training extra not found in pyproject.toml"
    deps = [Requirement(d) for d in re.findall(r'"([^"]+)"', block.group(1))]
    assert any(r.name == "bitsandbytes" for r in deps)
