"""On-policy distillation: match a teacher velocity on the student's own trajectory.

Qwen-Image-2.0-RL (arXiv:2606.27608) merges task-specialized teachers into one
student by trajectory-level velocity MSE. This module is that loss on top of
the unified ``ModelBackend`` flow frame.

Repo frame (non-SANA): t=0 noise → t=1 data, v = data - noise.
SANA: t=1 noise → t=0 data, v = noise - data.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

VelocityFn = Callable[..., torch.Tensor]


def drift_update(
    latents: torch.Tensor,
    velocity: torch.Tensor,
    dt: float | torch.Tensor,
    family: str,
) -> torch.Tensor:
    """One Euler step of the probability-flow ODE in this repo's flow frame."""
    if family == "sana":
        return latents - dt * velocity
    return latents + dt * velocity


def high_noise_indices(steps: int, count: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample ``count`` rollout slots, biased toward high-noise (early) steps."""
    if count < 1 or count > steps:
        raise ValueError(f"train timestep count must be in [1, {steps}], got {count}")
    weights = torch.linspace(float(steps), 1.0, steps)
    return torch.multinomial(weights, count, replacement=False, generator=generator)


@torch.no_grad()
def rollout_euler(
    predict_fn: VelocityFn,
    noise: torch.Tensor,
    steps: int,
    family: str,
    text_cond: dict[str, Any],
    context_latents: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Integrate the student ODE and return ``(latents_at_t, timesteps)`` of length steps+1."""
    if steps < 1:
        raise ValueError("rollout_steps must be positive")
    dt = 1.0 / steps
    batch = noise.shape[0]
    device, dtype = noise.device, noise.dtype
    latents = noise
    if family == "sana":
        time = torch.ones(batch, device=device, dtype=dtype)
        sign = -1.0
    else:
        time = torch.zeros(batch, device=device, dtype=dtype)
        sign = 1.0
    trajectory = [latents]
    times = [time]
    for _ in range(steps):
        velocity = predict_fn(
            latents, time, text_cond, batch, context_latents=context_latents
        )
        latents = latents + sign * dt * velocity
        time = time + sign * dt
        trajectory.append(latents)
        times.append(time)
    return trajectory, times


def opd_loss(student_velocity: torch.Tensor, teacher_velocity: torch.Tensor) -> torch.Tensor:
    """Trajectory-level velocity matching (Eq. 14 of the RL report)."""
    return F.mse_loss(student_velocity.float(), teacher_velocity.float())


def _self_predict(backend) -> VelocityFn:
    def predict(latents, timesteps, text_cond, batch_size, context_latents=None, **kwargs):
        with torch.no_grad():
            return backend.forward(
                latents, timesteps, text_cond, batch_size, context_latents=context_latents, **kwargs
            )

    return predict


def _backend_predict(backend) -> VelocityFn:
    def predict(latents, timesteps, text_cond, batch_size, context_latents=None, **kwargs):
        with torch.no_grad():
            return backend.forward(
                latents, timesteps, text_cond, batch_size, context_latents=context_latents, **kwargs
            )

    return predict


def _move_backend(backend, device: torch.device | str) -> None:
    """Move a frozen teacher backbone between CPU and the student compute device."""
    target = torch.device(device)
    model = getattr(backend, "peft_model", None)
    if model is not None:
        model.to(target)
    backend.device = target


class TeacherBank:
    """Task → frozen velocity predictor. ``*`` is the fallback teacher.

    When ``offload`` is set, non-self backends live on CPU and are moved to
    ``compute_device`` only for the forward that needs them.
    """

    def __init__(
        self,
        predictors: dict[str, VelocityFn],
        *,
        backends: dict[str, Any] | None = None,
        offload: bool = False,
        compute_device: torch.device | str | None = None,
    ):
        if not predictors:
            raise ValueError("OPD needs at least one teacher")
        self.predictors = dict(predictors)
        self.backends = dict(backends or {})
        self.offload = bool(offload)
        self.compute_device = torch.device(compute_device) if compute_device is not None else None

    def predict(
        self,
        task: str,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_cond: dict[str, Any],
        batch_size: int,
        context_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fn = self.predictors.get(task, self.predictors.get("*"))
        if fn is None:
            raise KeyError(f"No OPD teacher for task {task!r}")
        backend = self.backends.get(task)
        if backend is None and task not in self.predictors:
            backend = self.backends.get("*")
        if self.offload and backend is not None and self.compute_device is not None:
            _move_backend(backend, self.compute_device)
            try:
                return fn(latents, timesteps, text_cond, batch_size, context_latents=context_latents)
            finally:
                _move_backend(backend, "cpu")
        return fn(latents, timesteps, text_cond, batch_size, context_latents=context_latents)


def load_frozen_teacher(
    model_name: str,
    checkpoint: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    lora_rank: int,
    lora_alpha: int,
    use_rslora: bool,
    lora_targets: list[str] | None,
    gradient_checkpointing: bool,
):
    """Load a second LoRA backend and freeze it. ``checkpoint`` is a dir or ``lora.pt``."""
    from .model_backend import ModelBackend

    path = Path(checkpoint)
    if path.is_dir():
        path = path / "lora.pt"
    if not path.is_file():
        raise FileNotFoundError(f"OPD teacher checkpoint not found: {path}")
    backend = ModelBackend(
        model_name=model_name,
        device=device,
        dtype=dtype,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        use_rslora=use_rslora,
        lora_targets=lora_targets,
        enable_gradient_checkpointing=gradient_checkpointing,
        tune_mode="lora",
    )
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("lora_state_dict", ckpt)
    params = {n: p for n, p in backend.peft_model.named_parameters() if p.requires_grad}
    missing = sorted(params.keys() - state.keys())
    unexpected = sorted(state.keys() - params.keys())
    if missing or unexpected:
        raise ValueError(
            f"{path} does not match the student LoRA layout: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))
    backend.peft_model.eval()
    for param in backend.peft_model.parameters():
        param.requires_grad_(False)
    return backend


def build_teacher_bank(config, student_backend) -> TeacherBank:
    """Build predictors from ``config.opd_teachers`` (``self`` reuses the student, no extra load).

    Identical checkpoint paths are loaded once (semantic + referring can share the
    2D specialist). With ``opd_offload_teachers``, frozen teachers stay on CPU
    except for the forward that needs them.
    """
    predictors: dict[str, VelocityFn] = {}
    backends: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    offload = bool(getattr(config, "opd_offload_teachers", True))
    compute_device = student_backend.device
    load_device = torch.device("cpu") if offload else compute_device
    for task, path in config.opd_teachers.items():
        if path == "self":
            predictors[task] = _self_predict(student_backend)
            continue
        key = str(Path(path))
        if key not in loaded:
            loaded[key] = load_frozen_teacher(
                config.model_name,
                path,
                device=load_device,
                dtype=student_backend.dtype,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                use_rslora=config.rslora,
                lora_targets=config.lora_targets,
                gradient_checkpointing=False,
            )
        teacher = loaded[key]
        backends[task] = teacher
        predictors[task] = _backend_predict(teacher)
    return TeacherBank(
        predictors,
        backends=backends,
        offload=offload,
        compute_device=compute_device,
    )
