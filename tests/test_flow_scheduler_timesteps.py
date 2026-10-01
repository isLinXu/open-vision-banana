"""Flow-matching timesteps must cover the whole [0, 1] path.

Rectified-flow families start inference at t=0 (the source image), so training has to
see small t too. Logit-normal sampling is sigmoid(N(0, 1)); sigmoid of a uniform
variable only reaches [0.5, 0.731] and left FLUX untrained on the first half of the path.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch
from _paths import REPO_ROOT

from open_vision_banana.training.flow_matching import FlowScheduler

N = 20000
CPU = torch.device("cpu")


def assert_logit_normal(t: torch.Tensor) -> None:
    assert t.min() > 0 and t.max() < 1
    assert (t < 0.3).float().mean() > 0.1
    assert (t > 0.7).float().mean() > 0.1
    assert abs(t.median().item() - 0.5) < 0.02


def load_train_flux():
    pytest.importorskip("wandb")
    path = REPO_ROOT / "scripts" / "train_flux.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rectified_logit_normal_covers_both_ends():
    torch.manual_seed(0)
    assert_logit_normal(FlowScheduler.sample_timesteps(N, CPU, "flux", logit_normal=True))


def test_rectified_uniform_covers_unit_interval():
    torch.manual_seed(0)
    t = FlowScheduler.sample_timesteps(N, CPU, "flux", logit_normal=False)
    assert t.min() < 0.01 and t.max() > 0.99


def test_sana_applies_flow_shift():
    torch.manual_seed(0)
    unshifted = FlowScheduler.sample_timesteps(N, CPU, "sana", logit_normal=True, flow_shift=1.0)
    torch.manual_seed(0)
    shifted = FlowScheduler.sample_timesteps(N, CPU, "sana", logit_normal=True, flow_shift=3.0)
    torch.testing.assert_close(shifted, 3.0 * unshifted / (1.0 + 2.0 * unshifted))


def test_legacy_train_flux_logit_normal_covers_both_ends():
    module = load_train_flux()
    torch.manual_seed(0)
    assert_logit_normal(module.sample_rectified_flow_timesteps(N, CPU, logit_normal=True))
    t = module.sample_rectified_flow_timesteps(N, CPU, logit_normal=False)
    assert t.min() < 0.01 and t.max() > 0.99
