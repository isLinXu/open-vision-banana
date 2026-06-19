"""ONNX export utilities for Vision Banana models.

Exports the trained LoRA-augmented FLUX.1 transformer and
the RGB mapping decoders to ONNX format for deployment.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from ..core.depth_mapper import SPECTRAL_KEYPOINTS

logger = logging.getLogger(__name__)


class DepthRGBMapperONNX(nn.Module):
    """ONNX-exportable wrapper for DepthRGBMapper.

    Wraps the power transform + spectral colormap as a single
    torch.nn.Module suitable for torch.onnx.export().
    """

    def __init__(self, lam: float = -3.0, c: float = 10.0 / 3.0):
        super().__init__()
        self.lam = lam
        self.c = c
        self.register_buffer("_positions", SPECTRAL_KEYPOINTS[:, 0].contiguous())
        self.register_buffer("_colors", SPECTRAL_KEYPOINTS[:, 1:].contiguous())

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
    output_path: str,
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
        model = DepthRGBMapperONNX()
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
                dm = DepthRGBMapper(self.mapper.lam, self.mapper.c)
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

    import warnings
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


def export_full_pipeline_onnx(
    model,
    output_dir: str,
    opset_version: int = 18,
) -> None:
    """Export the full trained pipeline (transformer + VAE) to ONNX.

    Args:
        model: A VisionBanana instance whose ``.transformer`` attribute is
            a FLUX transformer (nn.Module).
        output_dir: Directory to write ``transformer.onnx`` into.
        opset_version: ONNX opset version (default 17).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dummy tensors matching FLUX transformer's expected input shapes:
    #   hidden_states      [B, 16, H/8, W/8]  — VAE latent (16-channel)
    #   timestep           [B]                  — continuous timestep in [0, 1]
    #   encoder_hidden_states [B, seq, 4096]   — T5 text embeddings
    dummy_hidden = torch.randn(1, 16, 64, 64)
    dummy_timestep = torch.tensor([0.5])
    dummy_text_emb = torch.randn(1, 77, 4096)

    torch.onnx.export(
        model.transformer,
        (dummy_hidden, dummy_timestep, dummy_text_emb),
        str(output_dir / "transformer.onnx"),
        opset_version=opset_version,
        input_names=["hidden_states", "timestep", "encoder_hidden_states"],
        output_names=["output"],
        dynamic_axes={
            "hidden_states":          {0: "batch", 2: "latent_h", 3: "latent_w"},
            "timestep":               {0: "batch"},
            "encoder_hidden_states":  {0: "batch", 1: "seq_len"},
            "output":                 {0: "batch"},
        },
    )
    logger.info(f"Exported transformer to {output_dir / 'transformer.onnx'}")


__all__ = [
    "DepthRGBMapperONNX",
    "NormalRGBMapperONNX",
    "export_mapper_onnx",
    "export_full_pipeline_onnx",
]
