"""ONNX export utilities for Vision Banana model components."""

from __future__ import annotations

import inspect
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..core.depth_mapper import CUBE_EDGE_KEYPOINTS

logger = logging.getLogger(__name__)


class DepthRGBMapperONNX(nn.Module):
    """ONNX-exportable wrapper for DepthRGBMapper.

    Wraps the power transform + cube-edge colormap as a single
    torch.nn.Module suitable for torch.onnx.export().
    """

    def __init__(self, lam: float = -3.0, c: float = 10.0 / 3.0):
        super().__init__()
        self.lam = lam
        self.c = c
        self.register_buffer("_positions", CUBE_EDGE_KEYPOINTS[:, 0].contiguous())
        self.register_buffer("_colors", CUBE_EDGE_KEYPOINTS[:, 1:].contiguous())

    def depth_to_normalized(self, depth: torch.Tensor) -> torch.Tensor:
        """Power transform: depth → normalized [0, 1)."""
        depth = depth.clamp(min=0.0)
        ratio = depth / (self.lam * self.c)
        ratio = ratio.clamp(max=1.0 - 1e-7)
        base = 1.0 - ratio
        normalized = 1.0 - torch.pow(base, self.lam + 1)
        return normalized.clamp(0.0, 1.0)

    def normalized_to_depth(self, normalized: torch.Tensor) -> torch.Tensor:
        """Inverse: normalized → depth."""
        exponent = 1.0 / (self.lam + 1)
        base = 1.0 - normalized
        ratio = 1.0 - torch.pow(base.clamp(min=1e-10), exponent)
        depth = self.lam * self.c * ratio
        return depth.clamp(min=0.0)

    def normalized_to_rgb(self, normalized: torch.Tensor) -> torch.Tensor:
        """Spectral colormap: normalized → RGB (ONNX-compatible)."""
        positions = self._positions
        colors = self._colors

        orig_shape = normalized.shape
        n_flat = normalized.reshape(-1)

        le_mask = n_flat.unsqueeze(1) <= positions.unsqueeze(0)
        first_true = le_mask.float().argmax(dim=1)
        seg_idx = (first_true - 1).clamp(0, positions.shape[0] - 2)

        p_start = positions[seg_idx]
        p_end = positions[seg_idx + 1]
        local_t = (n_flat - p_start) / (p_end - p_start + 1e-10)
        local_t = local_t.clamp(0.0, 1.0)

        c_start = colors[seg_idx]
        c_end = colors[seg_idx + 1]
        rgb_flat = c_start + local_t.unsqueeze(-1) * (c_end - c_start)

        rgb = rgb_flat.reshape(*orig_shape, 3)
        if rgb.dim() == 5:
            rgb = rgb.permute(0, 4, 1, 2, 3).squeeze(2)
        elif rgb.dim() == 4:
            rgb = rgb.permute(0, 3, 1, 2)

        return rgb.clamp(0.0, 1.0)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """Forward: depth → RGB (default export direction)."""
        return self.normalized_to_rgb(self.depth_to_normalized(depth))


class NormalRGBMapperONNX(nn.Module):
    """ONNX-exportable wrapper for NormalRGBMapper."""

    def forward(self, normal: torch.Tensor) -> torch.Tensor:
        """Normal → RGB: R=(1-x)/2, G=(1+y)/2, B=(1+z)/2."""
        x, y, z = normal[:, 0:1], normal[:, 1:2], normal[:, 2:3]
        r = (1.0 - x) / 2.0
        g = (1.0 + y) / 2.0
        b = (1.0 + z) / 2.0
        return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


def export_mapper_onnx(
    mapper_type: str,
    output_path: str | Path,
    input_shape: tuple = (1, 1, 256, 256),
    opset_version: int = 18,
) -> None:
    """Export an RGB mapper to ONNX format.

    Args:
        mapper_type: One of "depth_to_rgb", "normal_to_rgb",
            "rgb_to_depth", "rgb_to_normal".
        output_path: Path to save the ONNX model.
        input_shape: Input tensor shape for tracing.
        opset_version: ONNX opset version.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mapper_type == "depth_to_rgb":
        model: nn.Module = DepthRGBMapperONNX()
        model.eval()
        dummy_input = torch.rand(*input_shape).clamp(0, 100)   # depth in metres

    elif mapper_type == "normal_to_rgb":
        model = NormalRGBMapperONNX()
        model.eval()
        shape = (input_shape[0], 3, *input_shape[2:])
        dummy_input = torch.randn(*shape)
        dummy_input = dummy_input / dummy_input.norm(dim=1, keepdim=True).clamp(min=1e-8)

    elif mapper_type == "rgb_to_depth":
        # Export the inverse: RGB visualisation → metric depth
        class RGBToDepthONNX(nn.Module):
            def __init__(self):
                super().__init__()
                self.mapper = DepthRGBMapperONNX()

            def forward(self, rgb: torch.Tensor) -> torch.Tensor:
                # Use DepthRGBMapper (non-ONNX) for the inverse since it's
                # not used in hot-path inference — only needed for evaluation.
                from ..core.depth_mapper import DepthRGBMapper
                dm = DepthRGBMapper(lam=self.mapper.lam, c=self.mapper.c)
                return dm.rgb_to_depth(rgb)

        model = RGBToDepthONNX()
        model.eval()
        shape = (input_shape[0], 3, *input_shape[2:])
        dummy_input = torch.rand(*shape)

    elif mapper_type == "rgb_to_normal":
        class RGBToNormalONNX(nn.Module):
            def forward(self, rgb: torch.Tensor) -> torch.Tensor:
                r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
                x = 1.0 - 2.0 * r
                y = 2.0 * g - 1.0
                z = 2.0 * b - 1.0
                normal = torch.cat([x, y, z], dim=1)
                return normal / normal.norm(dim=1, keepdim=True).clamp(min=1e-8)

        model = RGBToNormalONNX()
        model.eval()
        shape = (input_shape[0], 3, *input_shape[2:])
        dummy_input = torch.rand(*shape)

    else:
        raise ValueError(
            f"Unsupported mapper type: {mapper_type!r}. "
            "Choose from: depth_to_rgb, normal_to_rgb, rgb_to_depth, rgb_to_normal"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*legacy.*", category=DeprecationWarning)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
            dynamic_axes={
                "input":  {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            },
        )
    logger.info(f"Exported {mapper_type} to {output_path}")


class _FluxTransformerONNXWrapper(nn.Module):
    """Normalize diffusers' FLUX output to a single tensor."""

    def __init__(self, transformer: nn.Module, guidance_embeds: bool):
        super().__init__()
        self.transformer = transformer
        self.guidance_embeds = guidance_embeds

    @staticmethod
    def _sample(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple):
            return output[0]
        sample = getattr(output, "sample", None)
        if isinstance(sample, torch.Tensor):
            return sample
        raise TypeError(
            "FLUX transformer must return a Tensor, tuple, or object with a Tensor .sample"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "return_dict": False,
        }
        if self.guidance_embeds:
            kwargs["guidance"] = guidance
        return self._sample(self.transformer(**kwargs))


def _resolve_flux_transformer(model: Any) -> nn.Module:
    """Resolve a FLUX transformer without relying on pipeline truthiness."""
    transformer = getattr(model, "transformer", None)
    pipe = None

    if transformer is None:
        generator = getattr(model, "generator", None)
        if generator is not None:
            pipe = getattr(generator, "pipe", None)
            if pipe is None:
                pipe = getattr(generator, "_t2i_pipe", None)
            if pipe is not None:
                transformer = getattr(pipe, "transformer", None)

    if transformer is None:
        if pipe is not None and getattr(pipe, "unet", None) is not None:
            raise TypeError(
                "ONNX component export currently supports FLUX transformers only; "
                "the resolved pipeline uses a UNet (for example SDXL/SSD-1B)."
            )
        raise AttributeError(
            "Could not resolve a FLUX transformer. Checked model.transformer, "
            "model.generator.pipe.transformer, and the legacy "
            "model.generator._t2i_pipe.transformer path."
        )
    if not isinstance(transformer, nn.Module):
        raise TypeError(f"Resolved transformer is not a torch.nn.Module: {type(transformer)!r}")

    signature = inspect.signature(transformer.forward)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    required = {
        "hidden_states",
        "encoder_hidden_states",
        "pooled_projections",
        "timestep",
        "img_ids",
        "txt_ids",
    }
    missing = required.difference(parameters)
    if missing and not accepts_kwargs:
        raise TypeError(
            "Resolved transformer does not expose the FLUX forward contract; "
            f"missing parameters: {', '.join(sorted(missing))}."
        )
    return transformer


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def export_flux_transformer_onnx(
    model: Any,
    output_dir: str | Path,
    opset_version: int = 18,
    image_sequence_length: int = 1024,
    text_sequence_length: int = 512,
) -> Path:
    """Export the loaded FLUX transformer component to ONNX.

    This does not export a runnable text-to-image pipeline. Tokenizers, text
    encoders, the scheduler, VAE, and RGB mappers remain separate components.

    Args:
        model: A FLUX transformer or VisionBanana-style object containing one.
        output_dir: Directory for ``transformer.onnx`` and its export manifest.
        opset_version: ONNX opset version.
        image_sequence_length: Packed latent token count (1024 for 512px FLUX).
        text_sequence_length: Text conditioning token count.

    Returns:
        Path to the exported ``transformer.onnx`` file.
    """
    if image_sequence_length <= 0 or text_sequence_length <= 0:
        raise ValueError("ONNX sequence lengths must be positive")

    transformer = _resolve_flux_transformer(model)
    config = getattr(transformer, "config", None)
    in_channels = int(_config_value(config, "in_channels", 64))
    joint_attention_dim = int(_config_value(config, "joint_attention_dim", 4096))
    pooled_projection_dim = int(_config_value(config, "pooled_projection_dim", 768))
    guidance_embeds = bool(_config_value(config, "guidance_embeds", False))

    first_parameter = next(transformer.parameters(), None)
    device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    dtype = (
        first_parameter.dtype
        if first_parameter is not None and first_parameter.is_floating_point()
        else torch.float32
    )
    if device.type == "meta":
        raise ValueError("Cannot export a transformer with parameters on the meta device")

    wrapper = _FluxTransformerONNXWrapper(transformer, guidance_embeds).eval()
    dummy_inputs: list[torch.Tensor] = [
        torch.randn(1, image_sequence_length, in_channels, device=device, dtype=dtype),
        torch.randn(1, text_sequence_length, joint_attention_dim, device=device, dtype=dtype),
        torch.randn(1, pooled_projection_dim, device=device, dtype=dtype),
        torch.tensor([0.5], device=device, dtype=dtype),
        torch.zeros(image_sequence_length, 3, device=device, dtype=dtype),
        torch.zeros(text_sequence_length, 3, device=device, dtype=dtype),
    ]
    input_names = [
        "hidden_states",
        "encoder_hidden_states",
        "pooled_projections",
        "timestep",
        "img_ids",
        "txt_ids",
    ]
    dynamic_axes = {
        "hidden_states": {0: "batch", 1: "image_seq_len"},
        "encoder_hidden_states": {0: "batch", 1: "text_seq_len"},
        "pooled_projections": {0: "batch"},
        "timestep": {0: "batch"},
        "img_ids": {0: "image_seq_len"},
        "txt_ids": {0: "text_seq_len"},
        "output": {0: "batch", 1: "image_seq_len"},
    }
    if guidance_embeds:
        dummy_inputs.append(torch.zeros(1, device=device, dtype=dtype))
        input_names.append("guidance")
        dynamic_axes["guidance"] = {0: "batch"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transformer.onnx"

    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs),
        str(output_path),
        opset_version=opset_version,
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    manifest = {
        "schema_version": 1,
        "format": "onnx",
        "component": "flux_transformer",
        "file": output_path.name,
        "opset_version": opset_version,
        "guidance_input": guidance_embeds,
        "image_sequence_length": image_sequence_length,
        "text_sequence_length": text_sequence_length,
        "included": ["transformer"],
        "not_included": ["tokenizers", "text_encoders", "scheduler", "vae", "rgb_mappers"],
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Exported FLUX transformer component to %s", output_path)
    return output_path


def export_full_pipeline_onnx(
    model: Any,
    output_dir: str | Path,
    opset_version: int = 18,
) -> Path:
    """Deprecated alias for :func:`export_flux_transformer_onnx`.

    The historical name was inaccurate: the function never exported the VAE,
    scheduler, text encoders, or tokenizers required by a full pipeline.
    """
    warnings.warn(
        "export_full_pipeline_onnx() exports only the FLUX transformer component; "
        "use export_flux_transformer_onnx() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return export_flux_transformer_onnx(model, output_dir, opset_version)


__all__ = [
    "DepthRGBMapperONNX",
    "NormalRGBMapperONNX",
    "export_mapper_onnx",
    "export_flux_transformer_onnx",
    "export_full_pipeline_onnx",
]
