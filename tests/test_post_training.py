"""OPD / GRPO math, config gates, and Trainer step dispatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from open_vision_banana.training.config import TrainingConfig, load_config
from open_vision_banana.training.grpo import (
    clipped_surrogate,
    decoded_metric_reward,
    gaussian_transition_log_prob,
    group_advantages,
    to_unit_rgb,
)
from open_vision_banana.training.opd import (
    TeacherBank,
    build_teacher_bank,
    drift_update,
    high_noise_indices,
    opd_loss,
    rollout_euler,
)
from open_vision_banana.training.post_training import (
    SFT_GATE_PROTOCOL,
    grpo_sft_gate_status,
    parse_teacher_map,
    sft_gate_from_eval,
    write_sft_gate,
)
from open_vision_banana.training.trainer import Trainer
from scripts.check_experiment_gates import check_post_training

ROOT = Path(__file__).resolve().parents[1]


def test_group_advantages_are_zero_mean_unit_std():
    rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
    adv = group_advantages(rewards)
    assert adv.shape == rewards.shape
    assert adv.mean().abs().item() < 1e-5
    assert adv.std(unbiased=False).item() == pytest.approx(1.0)


def test_clipped_surrogate_matches_unclipped_at_ratio_one():
    logp = torch.zeros(4)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss = clipped_surrogate(logp, logp, advantages, clip_eps=0.2)
    assert loss.item() == pytest.approx((-advantages).mean().item())


def test_decoded_metric_reward_is_perfect_on_identical_depth():
    from open_vision_banana.core.depth_mapper import DepthRGBMapper

    depth = torch.full((2, 1, 16, 16), 3.0)
    rgb = DepthRGBMapper().depth_to_rgb(depth)
    reward = decoded_metric_reward("depth", rgb, rgb)
    assert reward.shape == (2,)
    assert torch.all(reward > 0.9)


def test_decoded_metric_reward_rejects_generation():
    rgb = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="generation"):
        decoded_metric_reward("generation", rgb, rgb)


def test_opd_loss_and_euler_rollout_rectified_reaches_constant_field():
    noise = torch.zeros(1, 4, 2, 2)
    target = torch.ones(1, 4, 2, 2)
    velocity = target - noise

    def predict(latents, timesteps, text_cond, batch_size, context_latents=None):
        return velocity.expand_as(latents)

    traj, times = rollout_euler(predict, noise, steps=4, family="flux", text_cond={}, context_latents=None)
    torch.testing.assert_close(traj[-1], target)
    assert times[0].item() == 0.0
    assert times[-1].item() == pytest.approx(1.0)
    assert opd_loss(velocity, velocity).item() == pytest.approx(0.0)


def test_sana_euler_steps_backward_in_time():
    noise = torch.ones(1, 2, 1, 1)
    data = torch.zeros(1, 2, 1, 1)
    velocity = noise - data  # SANA v = source - target

    def predict(latents, timesteps, text_cond, batch_size, context_latents=None):
        return velocity.expand_as(latents)

    traj, times = rollout_euler(predict, noise, steps=5, family="sana", text_cond={})
    torch.testing.assert_close(traj[-1], data, atol=1e-5, rtol=1e-5)
    assert times[0].item() == pytest.approx(1.0)
    assert times[-1].item() == pytest.approx(0.0, abs=1e-6)


def test_high_noise_indices_are_unique_and_in_range():
    idx = high_noise_indices(8, 3, generator=torch.Generator().manual_seed(0))
    assert idx.numel() == 3
    assert len(set(idx.tolist())) == 3
    assert int(idx.min()) >= 0 and int(idx.max()) < 8


def test_teacher_bank_falls_back_to_star():
    def depth_fn(*_a, **_k):
        return torch.tensor(1.0)

    def default_fn(*_a, **_k):
        return torch.tensor(9.0)

    bank = TeacherBank({"depth": depth_fn, "*": default_fn})
    assert bank.predict("depth", None, None, None, 1).item() == 1.0
    assert bank.predict("semantic", None, None, None, 1).item() == 9.0


def test_sft_gate_allows_scale_drift_and_rejects_palette_failure():
    ok, _ = grpo_sft_gate_status(
        {
            "protocol": SFT_GATE_PROTOCOL,
            "colormap_ok": True,
            "raw_delta_1.25": 0.14,
            "aligned_delta_1.25": 0.71,
        }
    )
    assert ok
    bad, detail = grpo_sft_gate_status({"colormap_ok": False, "raw_delta_1.25": 0.1, "aligned_delta_1.25": 0.7})
    assert not bad and "colormap" in detail
    scale_ok, _ = grpo_sft_gate_status({"colormap_ok": True, "raw_delta_1.25": 0.8, "aligned_delta_1.25": 0.7})
    assert not scale_ok


def test_config_rejects_aesthetic_reward_and_generation_grpo():
    with pytest.raises(ValueError, match="decoded_metric"):
        TrainingConfig(objective="grpo", tasks=["depth"], grpo_reward="aesthetic", allow_grpo_without_sft_gate=True).validate()
    with pytest.raises(ValueError, match="generation"):
        TrainingConfig(
            objective="grpo",
            tasks=["depth", "generation"],
            vision_ratio=0.5,
            allow_grpo_without_sft_gate=True,
        ).validate()
    with pytest.raises(ValueError, match="teachers"):
        TrainingConfig(objective="opd", tasks=["depth"]).validate()
    TrainingConfig(
        objective="opd", tasks=["depth"], opd_teachers={"depth": "self"}
    ).validate()
    TrainingConfig(
        objective="grpo", tasks=["depth"], allow_grpo_without_sft_gate=True
    ).validate()


def test_experiment_yamls_load():
    full = load_config(config_path=ROOT / "configs/experiments/qwen_image_21_fullft.yaml")
    assert full.tune_mode == "full" and full.objective == "sft" and full.lr == pytest.approx(1e-5)
    opd = load_config(config_path=ROOT / "configs/experiments/qwen_image_21_opd.yaml")
    assert opd.objective == "opd" and set(opd.opd_teachers) == set(opd.tasks)
    assert opd.opd_teachers["depth"] == "output/specialists/depth"
    assert opd.opd_teachers["semantic"] == opd.opd_teachers["referring"] == "output/specialists/2d"
    assert opd.opd_teachers["generation"] == "output/specialists/generation"
    grpo = load_config(config_path=ROOT / "configs/experiments/qwen_image_21_grpo_depth.yaml")
    assert grpo.objective == "grpo" and grpo.tasks == ["depth"]
    assert grpo.sft_gate_path.endswith("post_sft_gate.json")
    specialists = {}
    for name in ("depth", "2d", "generation"):
        cfg = load_config(config_path=ROOT / f"configs/experiments/qwen_image_21_specialist_{name}.yaml")
        specialists[name] = cfg
        assert cfg.objective == "sft" and cfg.tune_mode == "lora"
        assert cfg.output_dir == f"output/specialists/{name}"
    assert specialists["depth"].tasks == ["depth", "generation"]
    assert specialists["2d"].tasks == ["semantic", "instance", "referring", "generation"]
    assert specialists["generation"].tasks == ["generation"]


def test_post_gate_refuses_missing_teachers_and_ungated_grpo(tmp_path: Path):
    opd_report = check_post_training(config_path=ROOT / "configs/experiments/qwen_image_21_opd.yaml")
    statuses = {r.name: r.status for r in opd_report.results}
    assert statuses["post.teachers_cover_tasks"] == "PASS"
    assert any(name.startswith("post.teacher.") and result == "FAIL" for name, result in statuses.items())
    assert opd_report.failed

    grpo_report = check_post_training(config_path=ROOT / "configs/experiments/qwen_image_21_grpo_depth.yaml")
    assert {r.name: r.status for r in grpo_report.results}["post.sft_gate"] == "FAIL"

    gate = tmp_path / "post_sft_gate.json"
    gate.write_text(json.dumps({
        "protocol": SFT_GATE_PROTOCOL,
        "colormap_ok": True,
        "raw_delta_1.25": 0.12,
        "aligned_delta_1.25": 0.70,
    }), encoding="utf-8")
    passed = check_post_training(
        config_path=ROOT / "configs/experiments/qwen_image_21_grpo_depth.yaml",
        sft_gate=gate,
    )
    assert {r.name: r.status for r in passed.results}["post.sft_gate"] == "PASS"
    assert not passed.failed


def test_parse_teacher_map_from_cli_string():
    assert parse_teacher_map("depth=a,semantic=b") == {"depth": "a", "semantic": "b"}


def _tiny_backend(family: str = "sana"):
    def forward(latents, timesteps, text_cond, batch_size, context_latents=None, **_k):
        # Constant velocity in the SANA frame so a short rollout stays finite.
        return torch.zeros_like(latents)

    return SimpleNamespace(
        family=family,
        dtype=torch.float32,
        encode_image=lambda pixels: torch.zeros(
            pixels.shape[0], 4, 4, 4, device=pixels.device, dtype=pixels.dtype
        ),
        decode_latents=lambda latents: torch.zeros(
            latents.shape[0], 3, 8, 8, device=latents.device, dtype=latents.dtype
        ),
        forward=forward,
        peft_model=torch.nn.Linear(1, 1),
        model_config=SimpleNamespace(pipeline_class="SanaPipeline"),
    )


def _step_device_ids():
    names = ["cpu"]
    if torch.backends.mps.is_available():
        names.append("mps")
    return names


@pytest.mark.parametrize("device_name", _step_device_ids())
def test_opd_step_returns_finite_loss(device_name):
    device = torch.device(device_name)
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(
        objective="opd",
        tasks=["depth"],
        opd_teachers={"depth": "self"},
        opd_rollout_steps=4,
        opd_train_timesteps=2,
        device=device_name,
        dtype="float32",
    )
    trainer.device = device
    trainer.dtype = torch.float32
    trainer.backend = _tiny_backend()
    trainer._teacher_bank = TeacherBank({"depth": trainer.backend.forward})
    trainer._encode_edit_text = lambda *a, **k: {}
    trainer._edit_layout = lambda: None
    batch = {
        "prompt": ["p"],
        "target_rgb": torch.rand(1, 3, 8, 8, device=device),
        "image": torch.rand(1, 3, 8, 8, device=device) * 2 - 1,
        "task": ["depth"],
    }
    out = trainer._opd_step(batch, global_step=1)
    assert torch.isfinite(out["loss"].cpu())
    assert out["task_loss_acc"]["depth"]
    assert out["loss"].device.type == device_name


@pytest.mark.parametrize("device_name", _step_device_ids())
def test_grpo_step_returns_finite_loss(device_name):
    device = torch.device(device_name)
    trainer = object.__new__(Trainer)
    trainer.config = TrainingConfig(
        objective="grpo",
        tasks=["depth"],
        allow_grpo_without_sft_gate=True,
        grpo_group_size=2,
        opd_rollout_steps=2,
        opd_train_timesteps=1,
        grpo_sde_sigma=0.5,
        device=device_name,
        dtype="float32",
        batch_size=1,
    )
    trainer.device = device
    trainer.dtype = torch.float32
    trainer.backend = _tiny_backend()
    trainer._encode_edit_text = lambda *a, **k: {}
    trainer._edit_layout = lambda: None
    batch = {
        "prompt": ["p"],
        "target_rgb": torch.rand(1, 3, 8, 8, device=device),
        "image": torch.rand(1, 3, 8, 8, device=device) * 2 - 1,
        "task": ["depth"],
    }
    out = trainer._grpo_step(batch, global_step=1)
    assert torch.isfinite(out["loss"].cpu())
    assert "reward_mean" in out
    assert out["loss"].device.type == device_name


def test_gaussian_log_prob_prefers_the_true_mean():
    x = torch.zeros(2, 3, 2, 2)
    v = torch.ones_like(x)
    dt = 0.25
    mean = drift_update(x, v, dt, "flux")
    true_lp = gaussian_transition_log_prob(mean, x, v, dt, 0.5, "flux")
    far = gaussian_transition_log_prob(mean + 3, x, v, dt, 0.5, "flux")
    assert torch.all(true_lp > far)


def test_to_unit_rgb_roundtrip():
    rgb = to_unit_rgb(torch.tensor([-1.0, 1.0]))
    assert rgb.tolist() == [0.0, 1.0]


def test_get_device_prefers_mps_and_falls_back_from_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert TrainingConfig(device=None).get_device().type == "mps"
    assert TrainingConfig(device="cuda").get_device().type == "mps"
    assert TrainingConfig(device="cpu").get_device().type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert TrainingConfig(device=None).get_device().type == "cpu"
    with pytest.raises(ValueError, match="CUDA is not available"):
        TrainingConfig(device="cuda").get_device()
    with pytest.raises(ValueError, match="MPS is not available"):
        TrainingConfig(device="mps").get_device()


def test_get_device_prefers_mps_and_falls_back_from_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert TrainingConfig(device=None).get_device().type == "mps"
    assert TrainingConfig(device="cuda").get_device().type == "mps"
    assert TrainingConfig(device="cpu").get_device().type == "cpu"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert TrainingConfig(device=None).get_device().type == "cpu"
    with pytest.raises(ValueError, match="CUDA is not available"):
        TrainingConfig(device="cuda").get_device()
    with pytest.raises(ValueError, match="MPS is not available"):
        TrainingConfig(device="mps").get_device()


def test_unknown_post_training_yaml_field_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"post_training": {"foo": 1}}))
    with pytest.raises(ValueError, match="post_training.foo"):
        load_config(config_path=path)


def _nyu_eval(*, hist=None, raw=0.14, aligned=0.71, absrel=0.4):
    metrics = {"delta_1.25": raw, "delta_1.25_aligned": aligned, "AbsRel": absrel}
    if hist is not None:
        metrics["oracle_colormap_hist"] = hist
    return {
        "metadata": {"depth_colormap": "cube_edge", "model_name": "qwen-image-2.1"},
        "benchmarks": {"depth/nyu": {"dataset": "nyu", "decoded_metrics": metrics}},
    }


def test_sft_gate_from_eval_majority_palette_and_scale(tmp_path: Path):
    gate = sft_gate_from_eval(_nyu_eval(hist={"cube_edge": 80, "plasma": 20}), source="eval.json")
    assert gate["colormap_ok"] is True
    assert gate["raw_delta_1.25"] == pytest.approx(0.14)
    ok, _ = grpo_sft_gate_status(gate)
    assert ok
    path = write_sft_gate(tmp_path / "post_sft_gate.json", gate)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["protocol"] == SFT_GATE_PROTOCOL
    assert loaded["colormap_ok"] is True


def test_sft_gate_from_eval_missing_hist_or_wrong_majority():
    missing = sft_gate_from_eval(_nyu_eval(hist=None))
    assert missing["colormap_ok"] is False
    assert "oracle_colormap_hist missing" in missing["colormap_detail"]
    wrong = sft_gate_from_eval(_nyu_eval(hist={"plasma": 90, "cube_edge": 10}))
    assert wrong["colormap_ok"] is False
    with pytest.raises(ValueError, match="depth/nyu"):
        sft_gate_from_eval({"benchmarks": {}})


def test_build_teacher_bank_dedups_shared_paths_and_offloads(monkeypatch):
    loads: list[object] = []

    class FakeTeacher:
        def __init__(self):
            self.peft_model = torch.nn.Linear(1, 1)
            self.device = torch.device("cpu")

        def forward(self, latents, *a, **k):
            return torch.zeros_like(latents)

    def fake_load(*_a, **kwargs):
        loads.append(kwargs.get("device"))
        return FakeTeacher()

    monkeypatch.setattr("open_vision_banana.training.opd.load_frozen_teacher", fake_load)
    student = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        forward=lambda latents, *a, **k: torch.zeros_like(latents),
    )
    config = SimpleNamespace(
        model_name="qwen-image-2.1",
        lora_rank=32,
        lora_alpha=64,
        rslora=True,
        lora_targets=None,
        opd_offload_teachers=True,
        opd_teachers={
            "semantic": "output/specialists/2d",
            "referring": "output/specialists/2d",
            "depth": "self",
        },
    )
    bank = build_teacher_bank(config, student)
    assert len(loads) == 1
    assert loads[0] == torch.device("cpu")
    assert bank.backends["semantic"] is bank.backends["referring"]
    assert "depth" not in bank.backends
    assert bank.offload is True


def test_teacher_bank_offload_moves_backend_to_compute_then_cpu():
    class TrackingLinear(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self.seen: list[str] = []

        def to(self, device, *args, **kwargs):  # noqa: ANN001
            target = torch.device(device)
            self.seen.append(target.type)
            return super().to(device, *args, **kwargs)

    mod = TrackingLinear()
    backend = SimpleNamespace(
        peft_model=mod,
        device=torch.device("cpu"),
        forward=lambda *a, **k: torch.zeros(1),
    )
    bank = TeacherBank(
        {"depth": lambda *a, **k: backend.forward(*a, **k)},
        backends={"depth": backend},
        offload=True,
        compute_device=torch.device("cpu"),
    )
    assert bank.predict("depth", torch.zeros(1), torch.zeros(1), {}, 1).shape == (1,)
    assert mod.seen == ["cpu", "cpu"]
    assert backend.device == torch.device("cpu")


def test_check_run_defaults_objective_and_tune_mode_for_legacy_effective(tmp_path: Path):
    from scripts.check_experiment_gates import check_run

    run = tmp_path / "run"
    run.mkdir()
    (run / "effective_config.json").write_text(
        json.dumps(
            {
                "tasks": ["depth"],
                "vision_ratio": 1.0,
                "weight_decay": 0.01,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                "preflight": {"skipped": False, "errors": []},
            }
        ),
        encoding="utf-8",
    )
    report = check_run(run_dir=run, tasks=["depth"], vision_ratio=1.0)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["run.objective"] == "PASS"
    assert statuses["run.tune_mode"] == "PASS"


def test_write_sft_gate_cli(tmp_path: Path, monkeypatch):
    from scripts.write_sft_gate import main as write_main

    eval_json = tmp_path / "eval.json"
    eval_json.write_text(json.dumps(_nyu_eval(hist={"cube_edge": 12, "plasma": 3})), encoding="utf-8")
    out = tmp_path / "post_sft_gate.json"
    monkeypatch.setattr(sys, "argv", ["write_sft_gate.py", str(eval_json), "--output", str(out)])
    assert write_main() == 0
    gate = json.loads(out.read_text(encoding="utf-8"))
    assert gate["colormap_ok"] is True
    assert gate["protocol"] == SFT_GATE_PROTOCOL
