"""Inference API for frozen SSD/VAE scalar relative-depth probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..core.depth_mapper import DepthRGBMapper
from ..depth_probe_checkpoint import DepthProbeArtifact, load_depth_probe_artifact
from ..training.depth_probe import (
    DepthFeatureContract,
    FrozenSDXLDepthFeatureExtractor,
    ScalarDepthProbe,
)
from .generator import BaseGenerator


class FrozenDepthEstimator:
    """Deterministic relative-depth estimator backed by frozen SSD features."""

    def __init__(
        self,
        generator: Any,
        artifact: DepthProbeArtifact,
    ) -> None:
        model_name = str(getattr(generator, "model_name", artifact.model))
        if model_name != artifact.model:
            raise ValueError(
                f"Checkpoint requires model {artifact.model!r}, got {model_name!r}"
            )
        self.generator = generator
        self.artifact = artifact
        self.contract = DepthFeatureContract.from_dict(artifact.feature_contract)
        self.extractor = FrozenSDXLDepthFeatureExtractor(
            generator,
            timestep=artifact.timestep,
        )
        unet = self.extractor.unet
        parameter = next(unet.parameters())
        self.device = parameter.device
        self.head = ScalarDepthProbe(
            self.contract,
            width=artifact.width,
            upsample_stages=artifact.upsample_stages,
        ).to(device=self.device, dtype=torch.float32)
        self.head.load_state_dict(artifact.head_state_dict, strict=True)
        self.head.eval().requires_grad_(False)
        self.mapper = DepthRGBMapper(artifact.colormap)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        dtype: torch.dtype | None = None,
        model: str | None = None,
    ) -> FrozenDepthEstimator:
        """Load a checkpoint and its required frozen SSD base model."""
        artifact = load_depth_probe_artifact(checkpoint)
        model_name = model or artifact.model
        if model is not None and model != artifact.model:
            raise ValueError(
                f"Checkpoint requires model {artifact.model!r}, got override {model!r}"
            )
        resolved_dtype = dtype
        if resolved_dtype is None:
            resolved_device = BaseGenerator._resolve_device(device)
            resolved_type = torch.device(resolved_device).type
            resolved_dtype = (
                torch.bfloat16 if resolved_type in {"mps", "cuda"} else torch.float32
            )
        generator = BaseGenerator(
            model_name=model_name,
            device=device,
            dtype=resolved_dtype,
        )
        return cls(generator, artifact)

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if images.shape[0] <= 0 or min(images.shape[-2:]) <= 0:
            raise ValueError("images must have non-empty batch and spatial dimensions")
        if not torch.isfinite(images).all():
            raise ValueError("images contain non-finite values")
        minimum = float(images.detach().amin().cpu())
        maximum = float(images.detach().amax().cpu())
        if minimum < -1.001 or maximum > 1.001:
            raise ValueError("images must be normalized to [-1, 1]")

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> dict[str, Any]:
        """Predict normalized relative depth and its checkpoint colormap rendering."""
        self._validate_images(images)
        original_size = tuple(int(value) for value in images.shape[-2:])
        inference_size = (self.artifact.resolution, self.artifact.resolution)
        prepared = images.detach().float().cpu()
        if original_size != inference_size:
            prepared = F.interpolate(
                prepared,
                size=inference_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        contract, features = self.extractor.extract(
            prepared,
            prompt=self.artifact.prompt,
        )
        if contract != self.contract:
            raise ValueError(
                f"Runtime feature contract {contract.to_dict()} does not match "
                f"checkpoint {self.contract.to_dict()}"
            )
        relative_depth = self.head(
            features,
            output_size=inference_size,
            feature_mask=self.artifact.feature_mask,
        )
        if original_size != inference_size:
            relative_depth = F.interpolate(
                relative_depth,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        relative_depth = relative_depth.clamp(0, 1)
        visualization = self.mapper.normalized_to_rgb(relative_depth)
        return {
            "relative_depth": relative_depth,
            "visualization": visualization,
            "metadata": {
                "checkpoint": str(self.artifact.path),
                "model": self.artifact.model,
                "resolution": self.artifact.resolution,
                "timestep": self.artifact.timestep,
                "selected_step": self.artifact.selected_step,
                "colormap": self.artifact.colormap,
                "legacy_checkpoint": self.artifact.legacy,
                "input_size": list(original_size),
            },
        }


__all__ = ["FrozenDepthEstimator"]
