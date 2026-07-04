"""Inference optimization utilities for Vision Banana.

Provides memory/speed trade-offs for FLUX.1 pipelines and optional INT8
quantization for lightweight RGB mapper modules (not the 12B transformer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class InferenceOptions:
    """Configuration for inference-time optimizations."""

    enable_sequential_cpu_offload: bool = False
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_attention_slicing: bool = False
    attention_slice_size: int | str = "auto"
    enable_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    quantize_mappers_int8: bool = False
    # Full-pipeline quantization options
    dtype: str = "bfloat16"  # float32 | float16 | bfloat16 | int8
    enable_model_cpu_offload: bool = False


def apply_pipeline_optimizations(pipe, options: InferenceOptions) -> list[str]:
    """Apply diffusers memory/speed optimizations to a pipeline.

    Returns:
        List of human-readable descriptions of applied optimizations.
    """
    applied: list[str] = []

    if options.enable_model_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
        applied.append("model_cpu_offload")

    if options.enable_sequential_cpu_offload and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
        applied.append("sequential_cpu_offload")

    if options.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        applied.append("vae_slicing")

    if options.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
        applied.append("vae_tiling")

    if options.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(options.attention_slice_size)
        applied.append(f"attention_slicing({options.attention_slice_size})")

    if options.enable_torch_compile and hasattr(torch, "compile"):
        if hasattr(pipe, "transformer") and pipe.transformer is not None:
            pipe.transformer = torch.compile(
                pipe.transformer,
                mode=options.torch_compile_mode,
                fullgraph=False,
            )
            applied.append(f"torch_compile(transformer, mode={options.torch_compile_mode})")
        else:
            logger.warning("torch.compile requested but pipeline has no transformer")

    # Dtype conversion for full pipeline
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if options.dtype in dtype_map:
        target_dtype = dtype_map[options.dtype]
        for component_name in ("transformer", "vae", "text_encoder", "text_encoder_2"):
            component = getattr(pipe, component_name, None)
            if component is not None:
                component.to(target_dtype)
        applied.append(f"dtype={options.dtype}")

    return applied


def quantize_mapper_int8(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to a small mapper module.

    Suitable for ``DepthRGBMapperONNX`` / ``NormalRGBMapperONNX`` only.
    The FLUX transformer is not supported here.

    Uses ``torchao.quantization.quantize_`` when available (PyTorch ≥ 2.10
    removed ``torch.ao.quantization.quantize_dynamic``).  Falls back to the
    legacy API on older PyTorch versions.
    """
    model.eval()

    # Collect Linear/Conv2d module FQNs for torchao
    module_fqns: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module_fqns.append(name)

    # If no quantizable layers exist (e.g. pure-tensor ONNX mappers),
    # return the model unchanged — quantization is a no-op.
    if not module_fqns:
        logger.debug(
            "quantize_mapper_int8: no nn.Linear/nn.Conv2d found; returning unchanged model"
        )
        return model

    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

        config = Int8DynamicActivationInt8WeightConfig(module_fqns)
        quantize_(model, config)
        return model
    except (ImportError, AttributeError, RuntimeError, NotImplementedError) as e:
        # Fallback for environments without torchao or without Triton (e.g. MPS).
        # Also catches torchao kernel errors on non-CUDA platforms.
        logger.debug(f"torchao quantization unavailable ({e!r}); falling back to legacy API")
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8,
            )


def estimate_vram_savings(options: InferenceOptions) -> dict[str, str]:
    """Return rough VRAM impact notes for selected options."""
    notes: dict[str, str] = {}
    if options.enable_model_cpu_offload:
        notes["model_cpu_offload"] = "~40-50% VRAM reduction, single GPU comfortable"
    if options.enable_sequential_cpu_offload:
        notes["sequential_cpu_offload"] = "~50-70% VRAM reduction, ~3-4× slower"
    if options.enable_vae_slicing:
        notes["vae_slicing"] = "Moderate VRAM reduction during VAE decode"
    if options.enable_vae_tiling:
        notes["vae_tiling"] = "Large VRAM reduction for high-res VAE, slower"
    if options.enable_attention_slicing:
        notes["attention_slicing"] = "Small-moderate VRAM reduction"
    if options.enable_torch_compile:
        notes["torch_compile"] = "First run slower; steady-state speedup on CUDA"
    if options.dtype == "float16":
        notes["dtype_float16"] = "~50% VRAM reduction vs float32, slight quality trade-off"
    elif options.dtype == "bfloat16":
        notes["dtype_bfloat16"] = "~50% VRAM reduction, better stability than float16"
    if options.quantize_mappers_int8:
        notes["quantize_mappers_int8"] = "Negligible VRAM impact (mappers are tiny)"
    return notes


def options_from_dict(d: dict) -> InferenceOptions:
    """Build InferenceOptions from a plain dict (YAML/CLI friendly)."""
    valid = set(InferenceOptions.__dataclass_fields__)
    filtered = {k: v for k, v in d.items() if k in valid}
    return InferenceOptions(**filtered)


__all__ = [
    "InferenceOptions",
    "apply_pipeline_optimizations",
    "estimate_vram_savings",
    "options_from_dict",
    "quantize_mapper_int8",
]
