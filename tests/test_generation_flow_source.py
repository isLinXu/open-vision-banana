"""Per-sample flow source: generation-replay rows start from noise even inside mixed batches.

Review item 6a (EXPERIMENT_REVIEW_2026-10-22): the trainer only switched to a
Gaussian start when the *whole* batch was generation, so in a mixed batch on a
T2I backbone (img2img_flow=True) the T2I replay rows had source == target and a
zero velocity target; validation never switched at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from open_vision_banana.training.flow_matching import FlowScheduler
from open_vision_banana.training.model_backend import ModelBackend


FAMILY = "flux"  # rectified flow: v = target - source (SANA uses the opposite sign; the mask logic is identical)


def _backend() -> SimpleNamespace:
    return SimpleNamespace(
        flow_scheduler=FlowScheduler(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        family=FAMILY,
    )


def _call(img2img_flow, flow_sigma=0.0):
    torch.manual_seed(0)
    target = torch.randn(4, 3, 4, 4)
    inputs = target.clone()  # T2I replay: input == target (the dataset mirrors target into input)
    inputs[:2] = torch.randn(2, 3, 4, 4)  # two vision rows with a genuinely different source
    return (
        ModelBackend.prepare_flow_batch(
            _backend(), target, inputs, 4, img2img_flow=img2img_flow, flow_sigma=flow_sigma
        ),
        target,
        inputs,
    )


def test_mixed_batch_mask_gives_replay_rows_a_noise_source():
    mask = torch.tensor([True, True, False, False])
    out, target, inputs = _call(mask)
    assert out["img2img_mask"].tolist() == mask.tolist()
    # vision rows: v = target - input
    assert torch.allclose(out["velocity_target"][:2], target[:2] - inputs[:2], atol=1e-6)
    # replay rows: v = target - noise, not zero
    assert torch.allclose(out["velocity_target"][2:], target[2:] - out["noise"][2:], atol=1e-6)
    assert out["velocity_target"][2:].abs().mean() > 0.5


def test_batch_wide_flag_reproduces_the_old_bug_only_when_asked():
    out, target, inputs = _call(True)  # the pre-fix behaviour, now opt-in via a plain bool
    assert torch.allclose(out["velocity_target"][2:], torch.zeros_like(target[2:]), atol=1e-6)
    out, target, _ = _call(False)
    assert torch.allclose(out["velocity_target"], target - out["noise"], atol=1e-6)


def test_flow_sigma_only_mixes_noise_into_img2img_rows():
    mask = torch.tensor([True, True, False, False])
    out, target, inputs = _call(mask, flow_sigma=0.3)
    expected_src = 0.7 * inputs[:2] + 0.3 * out["noise"][:2]
    assert torch.allclose(out["velocity_target"][:2], target[:2] - expected_src, atol=1e-5)
    assert torch.allclose(out["velocity_target"][2:], target[2:] - out["noise"][2:], atol=1e-6)


def test_mask_length_mismatch_rejected():
    with pytest.raises(ValueError, match="mask has 3 entries"):
        _call(torch.tensor([True, False, True]))


def test_trainer_mask_rule_shared_by_train_and_validation():
    from open_vision_banana.training.trainer import Trainer

    trainer = object.__new__(Trainer)
    trainer.config = SimpleNamespace(img2img_flow=True)
    tasks = ["depth", "generation", "semantic", "generation"]
    mask = Trainer._img2img_mask(trainer, tasks, None, torch.device("cpu"))
    assert mask.tolist() == [True, False, True, False]
    # editors: everything from noise (source is conditioning, not the flow start)
    assert (
        Trainer._img2img_mask(trainer, tasks, "sequence_concat", torch.device("cpu")).tolist()
        == [False] * 4
    )
    trainer.config = SimpleNamespace(img2img_flow=False)
    assert Trainer._img2img_mask(trainer, tasks, None, torch.device("cpu")).tolist() == [False] * 4
