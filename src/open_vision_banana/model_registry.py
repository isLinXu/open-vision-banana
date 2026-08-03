#!/usr/bin/env python3
"""Unified model registry for multi-backbone Vision Banana training/inference.

Supports 23 base model families with varying architectures:
  - SD 1.5 / SD 2.1  (single text encoder, no added_cond_kwargs)
  - SSD-1B / SDXL     (dual text encoder, added_cond_kwargs required)
  - SDXL-Turbo        (same as SDXL, fewer inference steps)
  - FLUX-dev/schnell  (dual transformer, different pipeline)
  - SD3               (triple text encoder, MMDiT, pooled_projections)
  - PixArt-Alpha/Sigma (T5 encoder, DiT, resolution added_cond_kwargs)
  - Kolors            (ChatGLM encoder, UNet, added_cond_kwargs)
  - HunyuanDiT        (bilingual CLIP + mT5, DiT, image_meta_size)
  - Kandinsky3        (FLAN-UL2 encoder, UNet, attention_mask)
  - SanaSprint        (T5 encoder, DiT, guidance embeds, 2-4 step)

Each model family defines:
  - HF pipeline class & model_id
  - Text encoder count (1 or 2)
  - Whether added_cond_kwargs is needed (SDXL+)
  - VAE scaling factor
  - LoRA target modules (attention + FFN)
  - Default LoRA rank/alpha
  - Default resolution
  - Memory budget (for auto-selection on constrained devices)

Usage:
  from open_vision_banana.model_registry import get_model_config, load_pipeline

  cfg = get_model_config("ssd-1b")
  pipe, components = load_pipeline(cfg, device="mps", dtype=torch.float16)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger("vb_model_registry")


# ─── Model family configs ──────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Configuration for a base diffusion model family."""

    name: str
    model_id: str
    family: str  # "sd15" | "sd2" | "sdxl" | "flux" | "qwen-image" | "z-image" | "sana" | "lumina2" | "chroma" | "flux2" | "cogview4" | "aura-flow" | "sd3" | "pixart" | "kolors" | "hunyuandit" | "kandinsky3" | "sana-sprint"

    # Text encoder
    num_text_encoders: int  # 1 for SD1.5/SD2, 2 for SDXL/FLUX, 3 for SD3

    # Backbone type: "unet" for SD family, "transformer" for DiT-based models
    backbone_type: str = "unet"
    text_encoder_max_length: int = 77

    # Architecture flags
    needs_added_cond_kwargs: bool = False  # SDXL+ requires pooled embeds + time_ids
    cross_attention_dim: int = 768  # SD1.5=768, SD2=1024, SDXL=2048

    # VAE
    vae_scaling_factor: float = 0.18215  # value scaling; None = read from vae.config.scaling_factor
    vae_shift_factor: float | None = None  # shift factor (FLUX=0.1159, Z-Image=0.1159, SD=None)
    vae_downsample_factor: int = 8  # spatial downsample
    vae_latent_channels: int = 4  # SD=4, FLUX/Z-Image=16

    # LoRA
    lora_target_modules: list = field(
        default_factory=lambda: [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
        ]
    )
    default_lora_rank: int = 16
    default_lora_alpha: int = 32

    # Training defaults
    default_resolution: int = 256
    default_steps: int = 3000

    # Memory budget (GB) — approximate VRAM needed at fp16
    memory_budget_gb: float = 7.0

    # Pipeline class name (lazy import)
    pipeline_class: str = "StableDiffusionXLPipeline"

    # Inference defaults
    default_num_inference_steps: int = 30
    default_guidance_scale: float = 2.0

    # Transformer-specific flags (ignored for UNet models)
    needs_guidance_in_forward: bool = (
        False  # Qwen-Image, FLUX.2, SanaSprint pass guidance to transformer
    )
    needs_attention_mask: bool = (
        False  # SANA, Lumina2, PixArt, Kandinsky3 need encoder_attention_mask
    )
    needs_position_ids: bool = False  # Chroma, FLUX.2 need txt_ids/img_ids
    needs_size_kwargs: bool = False  # CogView4 needs original_size/target_size/crop_coords
    needs_pooled_projections: bool = False  # SD3, Kandinsky5 use pooled_projections
    needs_image_meta_size: bool = False  # HunyuanDiT needs image_meta_size + style
    needs_t5_and_clip: bool = False  # HunyuanDiT, HiDream use both T5 and CLIP encoders
    needs_added_cond_kwargs_dit: bool = (
        False  # PixArt uses added_cond_kwargs in transformer forward
    )

    # Whether VAE encode returns latent_dist (False for some custom VAEs)
    vae_has_latent_dist: bool = True

    # Description
    description: str = ""


# ─── Registry (lazy initialization) ───────────────────────────────────────

# Store factory functions instead of instantiated ModelConfig objects.
# Configs are created on first access via get_model_config(), reducing import
# overhead and enabling future dynamic registration without touching this file.
_REGISTRY_FACTORIES: dict[str, Callable[[], ModelConfig]] = {}

# Cache for already-instantiated configs
_REGISTRY_CACHE: dict[str, ModelConfig] = {}


def _register_model(name: str, factory: Callable[[], ModelConfig]) -> None:
    """Register a model config factory."""
    _REGISTRY_FACTORIES[name] = factory


def _build_registry() -> None:
    """Populate factory registry. Called once at module init."""
    # Only register once
    if _REGISTRY_FACTORIES:
        return

    _register_model(
        "sd15",
        lambda: ModelConfig(
            name="sd15",
            model_id="runwayml/stable-diffusion-v1-5",
            family="sd15",
            backbone_type="unet",
            num_text_encoders=1,
            cross_attention_dim=768,
            vae_scaling_factor=0.18215,
            memory_budget_gb=4.0,
            pipeline_class="StableDiffusionPipeline",
            default_resolution=256,
            default_guidance_scale=7.5,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            description="Stable Diffusion 1.5 (860M UNet) — lightweight, widely compatible",
        ),
    )

    _register_model(
        "sd2",
        lambda: ModelConfig(
            name="sd2",
            model_id="stabilityai/stable-diffusion-2-1",
            family="sd2",
            backbone_type="unet",
            num_text_encoders=1,
            cross_attention_dim=1024,
            vae_scaling_factor=0.18215,
            memory_budget_gb=5.0,
            pipeline_class="StableDiffusionPipeline",
            default_resolution=256,
            default_guidance_scale=7.5,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            description="Stable Diffusion 2.1 (866M UNet) — improved text understanding",
        ),
    )

    _register_model(
        "ssd-1b",
        lambda: ModelConfig(
            name="ssd-1b",
            model_id="segmind/SSD-1B",
            family="sdxl",
            backbone_type="unet",
            num_text_encoders=2,
            needs_added_cond_kwargs=True,
            cross_attention_dim=2048,
            vae_scaling_factor=0.18215,
            memory_budget_gb=5.5,
            pipeline_class="StableDiffusionXLPipeline",
            default_resolution=256,
            default_guidance_scale=2.0,
            description="SSD-1B (1.3B UNet) — distilled SDXL, best quality/memory tradeoff",
        ),
    )

    _register_model(
        "sdxl",
        lambda: ModelConfig(
            name="sdxl",
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            family="sdxl",
            backbone_type="unet",
            num_text_encoders=2,
            needs_added_cond_kwargs=True,
            cross_attention_dim=2048,
            vae_scaling_factor=0.18215,
            memory_budget_gb=7.0,
            pipeline_class="StableDiffusionXLPipeline",
            default_resolution=256,
            default_guidance_scale=2.0,
            description="SDXL (2.6B UNet) — full capacity, needs ≥16GB VRAM",
        ),
    )

    _register_model(
        "sdxl-turbo",
        lambda: ModelConfig(
            name="sdxl-turbo",
            model_id="stabilityai/sdxl-turbo",
            family="sdxl",
            backbone_type="unet",
            num_text_encoders=2,
            needs_added_cond_kwargs=True,
            cross_attention_dim=2048,
            vae_scaling_factor=0.18215,
            memory_budget_gb=7.0,
            pipeline_class="StableDiffusionXLPipeline",
            default_resolution=256,
            default_guidance_scale=0.0,
            default_num_inference_steps=4,
            description="SDXL Turbo (2.6B UNet) — 1-4 step generation, no CFG",
        ),
    )

    _register_model(
        "flux-dev",
        lambda: ModelConfig(
            name="flux-dev",
            model_id="black-forest-labs/FLUX.1-dev",
            family="flux",
            backbone_type="transformer",
            num_text_encoders=2,
            needs_added_cond_kwargs=False,
            cross_attention_dim=3072,
            vae_scaling_factor=0.3611,
            vae_downsample_factor=16,
            memory_budget_gb=24.0,
            pipeline_class="FluxPipeline",
            default_resolution=512,
            default_guidance_scale=3.5,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "to_q_context",
                "to_k_context",
                "to_v_context",
                "to_out.0_context",
            ],
            description="FLUX.1-dev (12B transformer) — state-of-the-art, needs ≥24GB VRAM",
        ),
    )

    _register_model(
        "flux-schnell",
        lambda: ModelConfig(
            name="flux-schnell",
            model_id="black-forest-labs/FLUX.1-schnell",
            family="flux",
            backbone_type="transformer",
            num_text_encoders=2,
            needs_added_cond_kwargs=False,
            cross_attention_dim=3072,
            vae_scaling_factor=0.3611,
            vae_downsample_factor=16,
            memory_budget_gb=24.0,
            pipeline_class="FluxPipeline",
            default_resolution=512,
            default_guidance_scale=0.0,
            default_num_inference_steps=4,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "to_q_context",
                "to_k_context",
                "to_v_context",
                "to_out.0_context",
            ],
            description="FLUX.1-schnell (12B) — 4-step generation, Apache-2.0",
        ),
    )

    _register_model(
        "qwen-image",
        lambda: ModelConfig(
            name="qwen-image",
            model_id="Qwen/Qwen-Image",
            family="qwen-image",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=4096,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=20.0,
            pipeline_class="QwenImagePipeline",
            default_resolution=1024,
            default_guidance_scale=4.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "q_proj",
                "k_proj",
                "v_proj",
                "proj_out",
                "img_in",
                "txt_in",
            ],
            needs_guidance_in_forward=True,
            needs_attention_mask=True,
            vae_has_latent_dist=False,
            description="Qwen-Image — text rendering, CFG, Apache-2.0",
        ),
    )

    _register_model(
        "z-image",
        lambda: ModelConfig(
            name="z-image",
            model_id="Tongyi-MAI/Z-Image",
            family="z-image",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2560,
            vae_scaling_factor=0.3611,
            vae_shift_factor=0.1159,
            vae_downsample_factor=8,
            vae_latent_channels=16,
            memory_budget_gb=24.0,
            pipeline_class="ZImagePipeline",
            default_resolution=1024,
            default_guidance_scale=5.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "w1",
                "w2",
                "w3",
            ],
            vae_has_latent_dist=False,
            description="Z-Image (6B base) — Alibaba Tongyi-MAI, single-stream DiT with Qwen3 text encoder, Apache-2.0",
        ),
    )

    _register_model(
        "sana-600m",
        lambda: ModelConfig(
            name="sana-600m",
            model_id="Efficient-Large-Model/SANA_600M_1024px_diffusers",
            family="sana",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=6.0,
            pipeline_class="SanaPipeline",
            default_resolution=1024,
            default_guidance_scale=4.5,
            default_num_inference_steps=20,
            default_lora_rank=16,
            default_lora_alpha=32,
            lora_target_modules=[
                "linear",
                "proj_out",
                "conv_inverted",
                "conv_point",
            ],
            needs_attention_mask=True,
            vae_has_latent_dist=False,
            description="SANA-600M — NVIDIA, linear DiT, efficient 1024px",
        ),
    )

    _register_model(
        "sana-600m-512px",
        lambda: ModelConfig(
            name="sana-600m-512px",
            model_id="Efficient-Large-Model/Sana_600M_512px_diffusers",
            family="sana",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=32,
            memory_budget_gb=5.0,
            pipeline_class="SanaPipeline",
            text_encoder_max_length=300,
            default_resolution=512,
            default_guidance_scale=4.5,
            default_num_inference_steps=20,
            default_lora_rank=16,
            default_lora_alpha=32,
            lora_target_modules=[
                "linear",
                "proj_out",
                "conv_inverted",
                "conv_point",
            ],
            needs_attention_mask=True,
            vae_has_latent_dist=False,
            description="SANA-600M 512px — NVIDIA, linear DiT, DC-AE f32c32 VAE, Gemma-2-2B text encoder",
        ),
    )

    _register_model(
        "sana-1600m",
        lambda: ModelConfig(
            name="sana-1600m",
            model_id="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            family="sana",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=10.0,
            pipeline_class="SanaPipeline",
            default_resolution=1024,
            default_guidance_scale=4.5,
            default_num_inference_steps=20,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "linear",
                "proj_out",
                "conv_inverted",
                "conv_point",
            ],
            needs_attention_mask=True,
            vae_has_latent_dist=False,
            description="SANA-1600M — NVIDIA, linear DiT, higher quality 1024px",
        ),
    )

    _register_model(
        "lumina2",
        lambda: ModelConfig(
            name="lumina2",
            model_id="Alpha-VLLM/Lumina-Image-2.0",
            family="lumina2",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=16,
            memory_budget_gb=12.0,
            pipeline_class="Lumina2Pipeline",
            default_resolution=1024,
            default_guidance_scale=4.0,
            default_num_inference_steps=30,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "x_embedder",
                "proj_out",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            needs_attention_mask=True,
            vae_has_latent_dist=False,
            description="Lumina-Image-2.0 (2B) — Alpha-VLLM, flow DiT",
        ),
    )

    _register_model(
        "chroma",
        lambda: ModelConfig(
            name="chroma",
            model_id="lodestones/Chroma1-HD",
            family="chroma",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=3072,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=20.0,
            pipeline_class="ChromaPipeline",
            default_resolution=1024,
            default_guidance_scale=5.0,
            default_num_inference_steps=35,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "in_proj",
                "out_proj",
                "proj_mlp",
                "proj_out",
                "context_embedder",
                "x_embedder",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ],
            needs_position_ids=True,
            vae_has_latent_dist=True,
            description="Chroma1-HD (8.9B) — FLUX.1-schnell base, Apache-2.0",
        ),
    )

    _register_model(
        "flux2-dev",
        lambda: ModelConfig(
            name="flux2-dev",
            model_id="black-forest-labs/FLUX.2-dev",
            family="flux2",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=3072,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=30.0,
            pipeline_class="Flux2Pipeline",
            default_resolution=1024,
            default_guidance_scale=3.5,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out",
                "to_add_out",
                "to_qkv_mlp_proj",
                "x_embedder",
                "context_embedder",
                "proj_out",
                "linear",
                "linear_in",
                "linear_out",
            ],
            needs_guidance_in_forward=True,
            needs_position_ids=True,
            vae_has_latent_dist=True,
            description="FLUX.2-dev — Black Forest Labs, next-gen DiT",
        ),
    )

    _register_model(
        "cogview4",
        lambda: ModelConfig(
            name="cogview4",
            model_id="THUDM/CogView4-6B",
            family="cogview4",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=4096,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=24.0,
            pipeline_class="CogView4Pipeline",
            default_resolution=1024,
            default_guidance_scale=5.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "proj",
                "text_proj",
                "linear",
                "proj_out",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
            ],
            needs_size_kwargs=True,
            vae_has_latent_dist=False,
            description="CogView4-6B — THUDM/Tsinghua, Apache-2.0",
        ),
    )

    _register_model(
        "aura-flow",
        lambda: ModelConfig(
            name="aura-flow",
            model_id="fal/AuraFlow",
            family="aura-flow",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=15.0,
            pipeline_class="AuraFlowPipeline",
            default_resolution=1024,
            default_guidance_scale=3.5,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            lora_target_modules=[
                "linear_1",
                "linear_2",
                "out_projection",
                "linear",
                "context_embedder",
                "proj_out",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ],
            vae_has_latent_dist=False,
            description="AuraFlow — fal.ai, MMDiT, Apache-2.0",
        ),
    )

    _register_model(
        "sd3",
        lambda: ModelConfig(
            name="sd3",
            model_id="stabilityai/stable-diffusion-3-medium",
            family="sd3",
            backbone_type="transformer",
            num_text_encoders=3,
            needs_added_cond_kwargs=False,
            cross_attention_dim=4096,
            vae_scaling_factor=0.3611,
            vae_shift_factor=0.1159,
            vae_latent_channels=16,
            vae_downsample_factor=8,
            memory_budget_gb=20.0,
            pipeline_class="StableDiffusion3Pipeline",
            default_resolution=1024,
            default_guidance_scale=7.0,
            default_num_inference_steps=28,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=256,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "to_q_context",
                "to_k_context",
                "to_v_context",
                "to_out_context",
                "context_embedder",
                "proj_out",
                "linear_1",
                "linear_2",
            ],
            needs_pooled_projections=True,
            needs_attention_mask=True,
            description="Stable Diffusion 3 (2B MMDiT) — triple encoder (CLIP×2 + T5), flow matching",
        ),
    )

    _register_model(
        "pixart-alpha",
        lambda: ModelConfig(
            name="pixart-alpha",
            model_id="PixArt-alpha/PixArt-XL-2-1024-MS",
            family="pixart",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=4096,
            vae_scaling_factor=0.13025,
            vae_downsample_factor=8,
            memory_budget_gb=10.0,
            pipeline_class="PixArtAlphaPipeline",
            default_resolution=1024,
            default_guidance_scale=4.5,
            default_num_inference_steps=20,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=120,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "scale_shift_table",
            ],
            needs_attention_mask=True,
            needs_added_cond_kwargs_dit=True,
            description="PixArt-Alpha (600M DiT) — T5 encoder, 1024px, efficient training",
        ),
    )

    _register_model(
        "pixart-sigma",
        lambda: ModelConfig(
            name="pixart-sigma",
            model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            family="pixart",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=4096,
            vae_scaling_factor=0.13025,
            vae_downsample_factor=8,
            memory_budget_gb=10.0,
            pipeline_class="PixArtSigmaPipeline",
            default_resolution=1024,
            default_guidance_scale=4.5,
            default_num_inference_steps=20,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=300,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "scale_shift_table",
            ],
            needs_attention_mask=True,
            needs_added_cond_kwargs_dit=True,
            description="PixArt-Sigma (600M DiT) — stronger text rendering, 4K-capable",
        ),
    )

    _register_model(
        "kolors",
        lambda: ModelConfig(
            name="kolors",
            model_id="Kwai-Kolors/Kolors",
            family="kolors",
            backbone_type="unet",
            num_text_encoders=1,
            needs_added_cond_kwargs=True,
            cross_attention_dim=4096,
            vae_scaling_factor=0.18215,
            vae_downsample_factor=8,
            memory_budget_gb=18.0,
            pipeline_class="KolorsPipeline",
            default_resolution=1024,
            default_guidance_scale=5.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=256,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            description="Kolors (2.6B UNet) — Kuaishou, ChatGLM-6B text encoder, Chinese-friendly",
        ),
    )

    _register_model(
        "hunyuandit",
        lambda: ModelConfig(
            name="hunyuandit",
            model_id="Tencent-Hunyuan/HunyuanDiT",
            family="hunyuandit",
            backbone_type="transformer",
            num_text_encoders=2,
            needs_added_cond_kwargs=False,
            cross_attention_dim=1024,
            vae_scaling_factor=0.13025,
            vae_downsample_factor=8,
            memory_budget_gb=15.0,
            pipeline_class="HunyuanDiTPipeline",
            default_resolution=1024,
            default_guidance_scale=6.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=77,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "txt_in",
                "img_in",
            ],
            needs_attention_mask=True,
            needs_t5_and_clip=True,
            needs_image_meta_size=True,
            vae_has_latent_dist=False,
            description="HunyuanDiT (1.5B) — Tencent, bilingual CLIP+mT5, Chinese+English",
        ),
    )

    _register_model(
        "kandinsky3",
        lambda: ModelConfig(
            name="kandinsky3",
            model_id="ai-forever/Kandinsky3.1",
            family="kandinsky3",
            backbone_type="unet",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=4096,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=12.0,
            pipeline_class="Kandinsky3Pipeline",
            default_resolution=1024,
            default_guidance_scale=4.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=256,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            needs_attention_mask=True,
            description="Kandinsky3 (3B UNet) — Sber, FLAN-UL2 encoder, Apache-2.0",
        ),
    )

    _register_model(
        "hunyuan-image",
        lambda: ModelConfig(
            name="hunyuan-image",
            model_id="tencent/HunyuanImage",
            family="hunyuan-image",
            backbone_type="transformer",
            num_text_encoders=2,
            needs_added_cond_kwargs=False,
            cross_attention_dim=4096,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=20.0,
            pipeline_class="HunyuanImagePipeline",
            default_resolution=1024,
            default_guidance_scale=5.5,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=256,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "txt_in",
                "img_in",
            ],
            needs_attention_mask=True,
            needs_t5_and_clip=True,
            vae_has_latent_dist=False,
            description="HunyuanImage — Tencent, dual-encoder DiT, bilingual, Apache-2.0",
        ),
    )

    _register_model(
        "sana-sprint",
        lambda: ModelConfig(
            name="sana-sprint",
            model_id="Efficient-Large-Model/SanaSprint",
            family="sana-sprint",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=8,
            memory_budget_gb=8.0,
            pipeline_class="SanaSprintPipeline",
            default_resolution=1024,
            default_guidance_scale=0.0,
            default_num_inference_steps=2,
            default_lora_rank=16,
            default_lora_alpha=32,
            lora_target_modules=[
                "linear",
                "proj_out",
                "conv_inverted",
                "conv_point",
            ],
            needs_attention_mask=True,
            needs_guidance_in_forward=True,
            vae_has_latent_dist=False,
            description="SanaSprint — NVIDIA, 2-4 step generation, ultra fast",
        ),
    )

    _register_model(
        "longcat-image",
        lambda: ModelConfig(
            name="longcat-image",
            model_id="meituan-longcat/LongCat-Image",
            family="longcat-image",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=16,
            memory_budget_gb=16.0,
            pipeline_class="LongCatImagePipeline",
            default_resolution=1024,
            default_guidance_scale=4.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=512,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "txt_in",
                "img_in",
            ],
            needs_attention_mask=False,
            needs_guidance_in_forward=True,
            needs_position_ids=True,
            vae_has_latent_dist=True,
            description="LongCat-Image — Meituan, Qwen2.5-VL encoder, Apache-2.0",
        ),
    )

    _register_model(
        "lumina",
        lambda: ModelConfig(
            name="lumina",
            model_id="Alpha-VLLM/Lumina-Next-SFT-diffusers",
            family="lumina",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=0.18215,
            vae_downsample_factor=8,
            memory_budget_gb=12.0,
            pipeline_class="LuminaPipeline",
            default_resolution=1024,
            default_guidance_scale=6.0,
            default_num_inference_steps=30,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=256,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "adaLN_modulation",
            ],
            needs_attention_mask=True,
            vae_has_latent_dist=True,
            description="Lumina-Next — Alpha-VLLM, Gemma encoder, Apache-2.0",
        ),
    )

    _register_model(
        "nucleusmoe-image",
        lambda: ModelConfig(
            name="nucleusmoe-image",
            model_id="NucleusAI/NucleusMoE-Image",
            family="nucleusmoe-image",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=16,
            memory_budget_gb=16.0,
            pipeline_class="NucleusMoEImagePipeline",
            default_resolution=1024,
            default_guidance_scale=4.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=256,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "gate_up_proj",
                "down_proj",
            ],
            needs_attention_mask=True,
            vae_has_latent_dist=False,
            description="NucleusMoE-Image — NucleusAI, Qwen3-VL encoder, MoE DiT",
        ),
    )

    _register_model(
        "ernie-image",
        lambda: ModelConfig(
            name="ernie-image",
            model_id="baidu/Ernie-Image",
            family="ernie-image",
            backbone_type="transformer",
            num_text_encoders=1,
            needs_added_cond_kwargs=False,
            cross_attention_dim=2048,
            vae_scaling_factor=None,
            vae_downsample_factor=16,
            memory_budget_gb=16.0,
            pipeline_class="ErnieImagePipeline",
            default_resolution=1024,
            default_guidance_scale=3.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=256,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "txt_in",
                "img_in",
            ],
            needs_attention_mask=True,
            vae_has_latent_dist=False,
            description="Ernie-Image — Baidu, Qwen encoder, Flux2 VAE, flow-matching",
        ),
    )

    _register_model(
        "hidream-image",
        lambda: ModelConfig(
            name="hidream-image",
            model_id="HiDream-ai/HiDream-I1-Full",
            family="hidream-image",
            backbone_type="transformer",
            num_text_encoders=2,
            needs_added_cond_kwargs=False,
            cross_attention_dim=4096,
            vae_scaling_factor=None,
            vae_downsample_factor=16,
            memory_budget_gb=30.0,
            pipeline_class="HiDreamImagePipeline",
            default_resolution=1024,
            default_guidance_scale=5.0,
            default_num_inference_steps=50,
            default_lora_rank=32,
            default_lora_alpha=64,
            text_encoder_max_length=128,
            lora_target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "linear_1",
                "linear_2",
                "txt_in",
                "img_in",
            ],
            needs_attention_mask=False,
            needs_t5_and_clip=True,
            vae_has_latent_dist=False,
            description="HiDream-I1 — HiDream-ai, 4-encoder (CLIP×2+T5+Llama3), MoE DiT",
        ),
    )


# Build factory registry at module import time (lightweight — just lambdas, no ModelConfig instances)
_build_registry()


# Public API: backward-compatible REGISTRY dict
# Accessing REGISTRY keys triggers lazy instantiation
_registry_lock = threading.Lock()


class _LazyRegistry(dict):
    """Dict that lazily instantiates ModelConfig on first value access."""

    def __getitem__(self, key):
        with _registry_lock:
            if key not in _REGISTRY_CACHE and key in _REGISTRY_FACTORIES:
                _REGISTRY_CACHE[key] = _REGISTRY_FACTORIES[key]()
            return _REGISTRY_CACHE[key]

    def __contains__(self, key):
        return key in _REGISTRY_FACTORIES

    def __iter__(self):
        return iter(_REGISTRY_FACTORIES)

    def __len__(self):
        return len(_REGISTRY_FACTORIES)

    def keys(self):
        return _REGISTRY_FACTORIES.keys()

    def values(self):
        for k in _REGISTRY_FACTORIES:
            yield self[k]

    def items(self):
        for k in _REGISTRY_FACTORIES:
            yield k, self[k]

    def get(self, key, default=None):
        if key in _REGISTRY_FACTORIES:
            return self[key]
        return default


REGISTRY: dict[str, ModelConfig] = _LazyRegistry()


def get_model_config(name: str) -> ModelConfig:
    """Look up a model config by name. Raises KeyError if not found."""
    if name not in REGISTRY:
        available = ", ".join(sorted(list_models()))
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return REGISTRY[name]


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(REGISTRY.keys())


def get_models_by_family(family: str) -> list[str]:
    """Return model names belonging to a family."""
    return [k for k in list_models() if get_model_config(k).family == family]


def resolve_model_config(model_name_or_path: str) -> ModelConfig:
    """Resolve a ModelConfig from a model name, HuggingFace ID, or local path.

    Resolution order:
    1. Exact match in the registry (e.g. "flux-schnell", "z-image")
    2. Registry model_id match (e.g. "black-forest-labs/FLUX.1-schnell")
    3. Local path basename match against registry model_id basename
    4. Auto-detect by inspecting the model's config files (model_index.json)

    Falls back to a generic FLUX config if no match is found.
    """
    # 1. Exact registry name match
    if model_name_or_path in REGISTRY:
        return REGISTRY[model_name_or_path]

    # 2. Registry model_id match
    for name in list_models():
        cfg = REGISTRY[name]
        if cfg.model_id == model_name_or_path:
            return cfg

    # 3. Local path basename match (exact + partial + name/family)
    import os

    basename = os.path.basename(os.path.normpath(model_name_or_path.rstrip("/")))

    # Normalize for comparison: lowercase and strip hyphens/underscores
    def _norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "")

    basename_norm = _norm(basename)
    for name in list_models():
        cfg = REGISTRY[name]
        # Check against model_id basename, registry name, and family
        for candidate in (os.path.basename(cfg.model_id), cfg.name, cfg.family):
            if basename.lower() == candidate.lower() or basename_norm == _norm(candidate):
                return _clone_config_with_path(cfg, model_name_or_path)

    # 3b. Partial basename match (e.g. "Z-Image" matches "Z-Image-Turbo")
    basename_lower = basename.lower()
    for name in list_models():
        cfg = REGISTRY[name]
        for candidate in (os.path.basename(cfg.model_id), cfg.name, cfg.family):
            cand_lower = candidate.lower()
            cand_norm = _norm(candidate)
            # Prefix match on original or normalized form
            if (
                cand_lower.startswith(basename_lower)
                or basename_lower.startswith(cand_lower)
                or cand_norm.startswith(basename_norm)
                or basename_norm.startswith(cand_norm)
            ):
                return _clone_config_with_path(cfg, model_name_or_path)

    # 4. Auto-detect from model_index.json
    detected = _auto_detect_family(model_name_or_path)
    if detected is not None:
        return detected

    # Fallback: treat as FLUX (preserves backward compatibility)
    logger.warning(
        f"Could not resolve model '{model_name_or_path}' to a known family. "
        "Falling back to FLUX defaults. To use a different model, pass "
        "--model-name with a registered name (e.g. 'z-image', 'qwen-image')."
    )
    cfg = REGISTRY["flux-schnell"]
    return _clone_config_with_path(cfg, model_name_or_path)


def _clone_config_with_path(cfg: ModelConfig, model_id: str) -> ModelConfig:
    """Clone a ModelConfig with a new model_id (for local paths)."""
    import copy
    import os

    new_cfg = copy.copy(cfg)
    new_cfg.model_id = model_id
    new_cfg.name = os.path.basename(os.path.normpath(model_id.rstrip("/"))).lower()
    return new_cfg


def _auto_detect_family(model_path: str) -> ModelConfig | None:
    """Try to detect model family from model_index.json."""
    import json
    import os

    # Check for model_index.json (diffusers format)
    index_path = os.path.join(model_path, "model_index.json")
    if not os.path.isfile(index_path):
        return None

    try:
        with open(index_path) as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Check for pipeline class in model_index.json
    # diffusers uses "_class_name" (with leading underscore) in model_index.json
    pipe_class = index.get("_class_name", "") or index.get("pipeline_class", "")

    # Map pipeline class names to registry entries
    _class_to_name = {
        "FluxPipeline": "flux-schnell",
        "StableDiffusionXLPipeline": "sdxl",
        "StableDiffusionPipeline": "sd15",
        "QwenImagePipeline": "qwen-image",
        "ZImagePipeline": "z-image",
        "SanaPipeline": "sana-600m",
        "Lumina2Pipeline": "lumina2",
        "ChromaPipeline": "chroma",
        "Flux2Pipeline": "flux2-dev",
        "CogView4Pipeline": "cogview4",
        "AuraFlowPipeline": "aura-flow",
        "StableDiffusion3Pipeline": "sd3",
        "PixArtAlphaPipeline": "pixart-alpha",
        "PixArtSigmaPipeline": "pixart-sigma",
        "KolorsPipeline": "kolors",
        "HunyuanDiTPipeline": "hunyuandit",
        "HunyuanImagePipeline": "hunyuan-image",
        "Kandinsky3Pipeline": "kandinsky3",
        "SanaSprintPipeline": "sana-sprint",
    }

    name = _class_to_name.get(pipe_class)
    if name is not None and name in REGISTRY:
        return _clone_config_with_path(REGISTRY[name], model_path)

    return None


def auto_select_model(memory_gb: float) -> str:
    """Auto-select the best model that fits within a memory budget.

    Preference order (best quality within budget):
      sdxl > ssd-1b > sana-600m > sd2 > sd15
    (Large DiT models excluded from auto-select — require explicit choice)
    """
    candidates = [
        ("sdxl", 7.0),
        ("ssd-1b", 5.5),
        ("sana-600m", 6.0),
        ("sd2", 5.0),
        ("sd15", 4.0),
    ]
    for name, budget in candidates:
        if memory_gb >= budget:
            return name
    return "sd15"  # fallback to smallest


# ─── Pipeline loading ──────────────────────────────────────────────────────


@dataclass
class PipelineComponents:
    """Container for loaded pipeline components."""

    unet: torch.nn.Module  # UNet or Transformer (aliased as 'unet' for backward compat)
    vae: torch.nn.Module
    text_encoder: torch.nn.Module | None
    text_encoder_2: torch.nn.Module | None
    tokenizer: object
    config: ModelConfig = None
    text_encoder_3: torch.nn.Module | None = None  # SD3 uses 3 text encoders
    tokenizer_2: object | None = None
    tokenizer_3: object | None = None
    scheduler: object = None
    # Extended components for new model families
    text_encoder_4: torch.nn.Module | None = None  # HiDream uses 4 text encoders
    tokenizer_4: object | None = None
    processor: object | None = None  # NucleusMoE/LongCat use processor instead of tokenizer

    @property
    def transformer(self):
        """Alias for unet — returns the backbone (UNet or Transformer)."""
        return self.unet


def load_pipeline(
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    enable_gradient_checkpointing: bool = True,
) -> PipelineComponents:
    """Load a diffusion pipeline and extract components.

    Handles architecture differences across model families:
      - SD 1.5/2.1: single text encoder, UNet, no added_cond_kwargs
      - SDXL/SSD-1B/Turbo: dual text encoder, UNet, added_cond_kwargs required
      - FLUX.1: dual text encoder, Transformer, no added_cond_kwargs
      - Qwen-Image/Z-Image/SANA/Lumina2/Chroma/FLUX.2/CogView4/AuraFlow:
        single text encoder, Transformer, various forward signatures

    Returns PipelineComponents with backbone (as 'unet'), vae, text_encoder(s), tokenizer.
    """
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    logger.info(f"Loading {config.model_id} ({config.family}) @ {dtype} ...")

    # Lazy import the correct pipeline class
    _pipeline_imports = {
        "StableDiffusionXLPipeline": "diffusers",
        "StableDiffusionPipeline": "diffusers",
        "FluxPipeline": "diffusers",
        "QwenImagePipeline": "diffusers",
        "ZImagePipeline": "diffusers",
        "SanaPipeline": "diffusers",
        "Lumina2Pipeline": "diffusers",
        "ChromaPipeline": "diffusers",
        "Flux2Pipeline": "diffusers",
        "CogView4Pipeline": "diffusers",
        "AuraFlowPipeline": "diffusers",
        "StableDiffusion3Pipeline": "diffusers",
        "PixArtAlphaPipeline": "diffusers",
        "PixArtSigmaPipeline": "diffusers",
        "KolorsPipeline": "diffusers",
        "HunyuanDiTPipeline": "diffusers",
        "HunyuanImagePipeline": "diffusers",
        "Kandinsky3Pipeline": "diffusers",
        "SanaSprintPipeline": "diffusers",
        "LongCatImagePipeline": "diffusers.pipelines.longcat_image",
        "LuminaPipeline": "diffusers.pipelines.lumina",
        "NucleusMoEImagePipeline": "diffusers.pipelines.nucleusmoe_image",
        "ErnieImagePipeline": "diffusers.pipelines.ernie_image",
        "HiDreamImagePipeline": "diffusers.pipelines.hidream_image",
    }

    cls_name = config.pipeline_class
    if cls_name in _pipeline_imports:
        import importlib

        mod = importlib.import_module(_pipeline_imports[cls_name])
        PipeClass = getattr(mod, cls_name)
    else:
        raise ValueError(f"Unknown pipeline class: {cls_name}")

    pipe = PipeClass.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
    )

    # Extract backbone (unet for SD family, transformer for DiT models)
    if config.backbone_type == "transformer":
        backbone = getattr(pipe, "transformer", None)
        if backbone is None:
            # Some pipelines might still use 'unet' attribute
            backbone = pipe.unet
    else:
        backbone = pipe.unet

    backbone = backbone.to(device)
    vae = pipe.vae.to(device)
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None:
        text_encoder = text_encoder.to(device)
    text_encoder_2 = None
    text_encoder_3 = None
    tokenizer = getattr(pipe, "tokenizer", None)
    tokenizer_2 = getattr(pipe, "tokenizer_2", None)
    tokenizer_3 = getattr(pipe, "tokenizer_3", None)

    if config.num_text_encoders >= 2:
        text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        if text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to(device)

    if config.num_text_encoders >= 3:
        text_encoder_3 = getattr(pipe, "text_encoder_3", None)
        if text_encoder_3 is not None:
            text_encoder_3 = text_encoder_3.to(device)

    # Kolors: ChatGLM text encoder is in a custom module, not standard text_encoder attr
    if config.family == "kolors" and text_encoder is None:
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None:
            text_encoder = text_encoder.to(device)

    # Freeze base model
    backbone.requires_grad_(False)
    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)
    if text_encoder_3 is not None:
        text_encoder_3.requires_grad_(False)

    # Extract extended components for new model families
    text_encoder_4 = None
    tokenizer_4 = None
    processor = None

    if config.family == "hidream-image":
        text_encoder_4 = getattr(pipe, "text_encoder_4", None)
        if text_encoder_4 is not None:
            text_encoder_4 = text_encoder_4.to(device)
            text_encoder_4.requires_grad_(False)
        tokenizer_4 = getattr(pipe, "tokenizer_4", None)

    if config.family in ("nucleusmoe-image", "longcat-image"):
        processor = getattr(pipe, "processor", None) or getattr(pipe, "text_processor", None)

    # Enable gradient checkpointing for memory efficiency
    if enable_gradient_checkpointing and hasattr(backbone, "enable_gradient_checkpointing"):
        backbone.enable_gradient_checkpointing()

    backbone_params = sum(p.numel() for p in backbone.parameters())
    logger.info(
        f"Backbone ({config.backbone_type}) params: {backbone_params / 1e6:.1f}M | "
        f"Text encoders: {config.num_text_encoders} | "
        f"added_cond: {config.needs_added_cond_kwargs}"
    )

    return PipelineComponents(
        unet=backbone,  # stored as 'unet' for backward compatibility
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        scheduler=getattr(pipe, "scheduler", None),
        config=config,
        text_encoder_4=text_encoder_4,
        tokenizer_4=tokenizer_4,
        processor=processor,
    )


# ─── Text encoding helpers ─────────────────────────────────────────────────


def encode_text(
    components: PipelineComponents,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    negative: bool = False,
) -> tuple[torch.Tensor, Any]:
    """Encode text prompts into hidden states + optional pooled embeddings.

    Handles:
      - Single-encoder (SD1.5/2.1, all new DiT models, PixArt, SanaSprint)
      - Dual-encoder (SDXL: concatenated hidden states + pooled; FLUX.1: T5 hidden states;
                      HunyuanDiT: bilingual CLIP + mT5)
      - Triple-encoder (SD3: CLIP×2 + T5, pooled_projections)

    For DiT models (Qwen-Image, Z-Image, SANA, Lumina2, PixArt, Kandinsky3, etc.), returns
    (encoder_hidden_states, attention_mask) where attention_mask replaces pooled_embeds
    for models that need it.

    Returns:
      (encoder_hidden_states, pooled_embeds_or_mask)
      - SD1.5/2.1: (hidden_states, None)
      - SDXL: (concatenated_hidden_states, pooled_embeds)
      - FLUX.1: (t5_hidden_states, None)
      - SD3: (concatenated_clip_hidden_states, pooled_projections)
      - PixArt: (t5_hidden_states, attention_mask)
      - Kolors: (chatglm_hidden_states, attention_mask)
      - HunyuanDiT: (clip_hidden_states, attention_mask) — mT5 handled separately
      - Kandinsky3: (flan_hidden_states, attention_mask)
      - SanaSprint: (t5_hidden_states, attention_mask)
    """
    tokenizer = components.tokenizer
    text_encoder = components.text_encoder
    text_encoder_2 = components.text_encoder_2
    text_encoder_3 = components.text_encoder_3
    config = components.config
    max_length = config.text_encoder_max_length

    def _hidden_states(output: Any, *, penultimate: bool = False) -> torch.Tensor:
        """Normalize tuple/model-output variants from HF and test doubles."""
        hidden_states = getattr(output, "hidden_states", None)
        if penultimate and isinstance(hidden_states, (tuple, list)) and len(hidden_states) >= 2:
            candidate = hidden_states[-2]
            if isinstance(candidate, torch.Tensor):
                return candidate
        candidate = getattr(output, "last_hidden_state", None)
        if isinstance(candidate, torch.Tensor):
            return candidate
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        candidate = output[0]
        if not isinstance(candidate, torch.Tensor):
            raise TypeError(f"Text encoder returned no tensor hidden states: {type(output)!r}")
        return candidate

    def _pooled(output: Any) -> torch.Tensor | None:
        candidate = getattr(output, "text_embeds", None)
        return candidate if isinstance(candidate, torch.Tensor) else None

    # ── SD3: triple text encoder (CLIP×2 + T5) ──
    if config.family == "sd3":
        # CLIP text encoders (tokenizer + tokenizer_2)
        prompt_embeds_list = []
        pooled_list = []

        for te, tok in [
            (text_encoder, tokenizer),
            (text_encoder_2, components.tokenizer_2),
        ]:
            if te is None:
                continue
            tokenized = tok(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)
            # SD3 uses penultimate hidden state, not last_hidden_state
            output = te(input_ids, output_hidden_states=True)
            prompt_embeds_list.append(_hidden_states(output, penultimate=True))
            # Collect pooled embeddings from both CLIP encoders
            pooled = _pooled(output)
            if pooled is not None:
                pooled_list.append(pooled)

        # Concatenate CLIP hidden states along feature dim: [768 + 1280 = 2048]
        encoder_hidden_states = torch.cat(prompt_embeds_list, dim=-1)

        # Pooled: concatenate CLIP-L and CLIP-G pooled embeddings
        pooled_prompt_embeds = torch.cat(pooled_list, dim=-1) if pooled_list else None

        # T5 encoder (tokenizer_3)
        if text_encoder_3 is not None:
            tok3 = components.tokenizer_3
            tokenized_3 = tok3(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids_3 = tokenized_3.input_ids.to(device)
            t5_output = text_encoder_3(input_ids_3)
            # Pad/truncate T5 to same length as CLIP
            t5_hidden = _hidden_states(t5_output)
            if t5_hidden.shape[1] < max_length:
                pad_len = max_length - t5_hidden.shape[1]
                t5_hidden = torch.nn.functional.pad(t5_hidden, (0, 0, 0, pad_len))
            elif t5_hidden.shape[1] > max_length:
                t5_hidden = t5_hidden[:, :max_length]
            encoder_hidden_states = torch.cat([encoder_hidden_states, t5_hidden], dim=-2)

        return encoder_hidden_states.to(dtype), pooled_prompt_embeds.to(
            dtype
        ) if pooled_prompt_embeds is not None else None

    # ── HunyuanDiT: bilingual CLIP + mT5 ──
    if config.family == "hunyuandit":
        # CLIP (text_encoder) — bilingual CLIP
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        clip_hidden = text_encoder(input_ids)[0]

        # mT5 (text_encoder_2)
        if text_encoder_2 is not None:
            tok2 = components.tokenizer_2
            tokenized_2 = tok2(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids_2 = tokenized_2.input_ids.to(device)
            t5_hidden = text_encoder_2(input_ids_2)[0]
            # Return concatenated as (clip_hidden, t5_hidden, clip_mask, t5_mask)
            clip_mask = tokenized.attention_mask.to(device)
            t5_mask = tokenized_2.attention_mask.to(device)
            # For HunyuanDiT, encoder_hidden_states is clip_hidden, t5 is passed separately
            return clip_hidden.to(dtype), (t5_hidden.to(dtype), clip_mask, t5_mask)

        return clip_hidden.to(dtype), tokenized.attention_mask.to(device)

    # ── Kolors: ChatGLM text encoder ──
    if config.family == "kolors":
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        # ChatGLM returns (hidden_states, attention_mask)
        encoder_hidden_states = text_encoder(input_ids)[0]
        attention_mask = tokenized.attention_mask.to(device)
        return encoder_hidden_states.to(dtype), attention_mask

    # ── HunyuanImage: dual-encoder (CLIP + mT5) with attention mask ──
    if config.family == "hunyuan-image":
        # CLIP (text_encoder) — primary encoder
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        clip_hidden = text_encoder(input_ids)[0]
        clip_mask = tokenized.attention_mask.to(device)

        # mT5 (text_encoder_2) — secondary encoder
        if text_encoder_2 is not None:
            tok2 = components.tokenizer_2
            tokenized_2 = tok2(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids_2 = tokenized_2.input_ids.to(device)
            t5_hidden = text_encoder_2(input_ids_2)[0]
            t5_mask = tokenized_2.attention_mask.to(device)
            # Return concatenated hidden states and combined mask
            # HunyuanImage uses both encoders; backbone_forward gets attention_mask
            return clip_hidden.to(dtype), (clip_mask, t5_hidden.to(dtype), t5_mask)

        return clip_hidden.to(dtype), clip_mask

    # ── HiDream: 4-encoder (CLIP×2 + T5 + Llama3) ──
    if config.family == "hidream-image":
        # T5 encoder (text_encoder_3 in pipeline, stored as text_encoder_2 in our components)
        t5_hidden = None
        t5_mask = None
        llama_hidden = None
        llama_mask = None
        pooled_embeds = None

        # CLIP encoders (tokenizer + tokenizer_2) → pooled embeds
        for i, (te, tok) in enumerate(
            [
                (text_encoder, tokenizer),
                (text_encoder_2, components.tokenizer_2),
            ]
        ):
            if te is None:
                continue
            tokenized = tok(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)
            output = te(input_ids, output_hidden_states=False)
            if i == 0:
                pooled_embeds = output.text_embeds

        # T5 encoder (tokenizer_3)
        tok3 = components.tokenizer_3
        if tok3 is not None:
            tokenized_3 = tok3(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids_3 = tokenized_3.input_ids.to(device)
            t5_hidden = text_encoder_3(input_ids_3)[0] if text_encoder_3 is not None else None
            t5_mask = tokenized_3.attention_mask.to(device)

        # Llama3 encoder — needs special handling (generate then embed)
        # For training, we use the tokenizer_4 if available
        getattr(components, "tokenizer_4", None) or getattr(components, "tokenizer_3", None)
        # Note: Llama3 encoder is stored in a non-standard attribute; for training
        # we fall back to using T5 hidden states if Llama3 is not separately loaded
        llama_hidden = t5_hidden  # fallback: use T5 as llama hidden
        llama_mask = t5_mask

        return (
            t5_hidden.to(dtype)
            if t5_hidden is not None
            else torch.zeros(len(prompts), 1, 4096, device=device, dtype=dtype),
            (
                pooled_embeds.to(dtype) if pooled_embeds is not None else None,
                llama_hidden.to(dtype) if llama_hidden is not None else None,
                llama_mask,
            ),
        )

    # ── LongCat-Image: Qwen2.5-VL encoder with prompt template ──
    if config.family == "longcat-image":
        # LongCat uses prompt template wrapping + Qwen2.5-VL text generation
        # For training, we use the tokenizer directly for simplicity
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        # Qwen2.5-VL: get hidden states from output
        outputs = text_encoder(
            input_ids, attention_mask=tokenized.attention_mask.to(device), output_hidden_states=True
        )
        encoder_hidden_states = outputs.hidden_states[-1]
        attention_mask = tokenized.attention_mask.to(device)
        return encoder_hidden_states.to(dtype), attention_mask

    # ── Lumina: Gemma encoder ──
    if config.family == "lumina":
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        outputs = text_encoder(input_ids, attention_mask=tokenized.attention_mask.to(device))
        encoder_hidden_states = outputs.last_hidden_state
        attention_mask = tokenized.attention_mask.to(device)
        return encoder_hidden_states.to(dtype), attention_mask

    # ── NucleusMoE: Qwen3-VL encoder ──
    if config.family == "nucleusmoe-image":
        # NucleusMoE uses a processor (Qwen3VLProcessor) instead of tokenizer
        processor = getattr(components, "processor", None) or tokenizer
        if callable(processor) and not hasattr(tokenizer, "model_max_length"):
            # Use processor for tokenization
            inputs = processor(text=prompts, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
        else:
            tokenized = tokenizer(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

        outputs = text_encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # NucleusMoE uses hidden_states at a specific index
        encoder_hidden_states = outputs.hidden_states[-1]
        return encoder_hidden_states.to(dtype), attention_mask

    # ── Ernie-Image: Qwen encoder ──
    if config.family == "ernie-image":
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        outputs = text_encoder(input_ids, attention_mask=attention_mask)
        encoder_hidden_states = (
            outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        )
        # Ernie needs text_lens (sequence lengths per sample)
        text_lens = attention_mask.sum(dim=1).to(dtype=torch.long)
        return encoder_hidden_states.to(dtype), (attention_mask, text_lens)

    # ── Z-Image: Qwen3 text encoder with chat template + hidden_states[-2] ──
    if config.family == "z-image":
        # Z-Image uses Qwen3Model as text encoder with chat template + thinking mode
        # It takes hidden_states[-2] (penultimate layer) and filters by attention_mask
        # Apply chat template (matching ZImagePipeline._encode_prompt)
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        for i, prompt_item in enumerate(prompt_list):
            messages = [{"role": "user", "content": prompt_item}]
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise TypeError("Z-Image tokenizer must provide apply_chat_template()")
            prompt_list[i] = apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

        text_inputs = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        # Qwen3: output_hidden_states=True via kwargs, take penultimate layer
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        encoder_hidden_states = outputs.hidden_states[-2]
        return encoder_hidden_states.to(dtype), attention_mask

    # ── Standard tokenization for all other models ──
    tokenized = tokenizer(
        prompts, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt"
    )
    input_ids = tokenized.input_ids.to(device)

    encoder_hidden_states = text_encoder(input_ids)[0]
    pooled_embeds = None

    if text_encoder_2 is not None:
        te2_out = text_encoder_2(input_ids)
        # SDXL: concatenate hidden states along last dim
        if config.family == "sdxl":
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, te2_out.last_hidden_state], dim=-1
            )
            pooled_embeds = te2_out.text_embeds  # [B, 1280]
        elif config.family == "flux":
            # FLUX: uses CLIP hidden states + T5 for joint attention
            encoder_hidden_states = te2_out.last_hidden_state  # T5 hidden states
            pooled_embeds = None  # FLUX doesn't use pooled embeds like SDXL

    # For DiT models that need attention mask
    if config.needs_attention_mask:
        attention_mask = tokenized.attention_mask.to(device)
        return encoder_hidden_states.to(dtype), attention_mask

    return encoder_hidden_states.to(dtype), (
        pooled_embeds.to(dtype) if pooled_embeds is not None else None
    )


def build_added_cond_kwargs(
    config: ModelConfig,
    pooled_embeds: torch.Tensor | None,
    resolution: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> dict | None:
    """Build added_cond_kwargs for SDXL/Kolors-family models.

    Returns None for non-SDXL/Kolors models.
    SDXL: {"text_embeds": pooled, "time_ids": [h,w,0,0,h,w]}
    Kolors: {"text_embeds": pooled, "time_ids": [h,w,0,0,h,w]} (same structure)
    """
    if not config.needs_added_cond_kwargs:
        return None

    if pooled_embeds is None:
        return None

    h, w = resolution
    add_time_ids = (
        torch.tensor(
            [h, w, 0, 0, h, w],
            device=device,
            dtype=dtype,
        )
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    return {
        "text_embeds": pooled_embeds,
        "time_ids": add_time_ids,
    }


# ─── LoRA setup ────────────────────────────────────────────────────────────


def apply_lora(
    unet: torch.nn.Module,
    config: ModelConfig,
    rank: int | None = None,
    alpha: int | None = None,
    use_rslora: bool = True,
) -> torch.nn.Module:
    """Apply LoRA adapters to backbone (UNet or Transformer) with model-appropriate target modules."""
    from peft import LoraConfig, get_peft_model

    r = rank or config.default_lora_rank
    a = alpha or config.default_lora_alpha

    lora_config = LoraConfig(
        r=r,
        lora_alpha=a,
        target_modules=config.lora_target_modules,
        lora_dropout=0.0,
        bias="none",
        use_rslora=use_rslora,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


# ─── Backbone forward helper ───────────────────────────────────────────────


def backbone_forward(
    backbone: torch.nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    config: ModelConfig,
    added_cond_kwargs: dict | None = None,
    guidance: float | None = None,
    attention_mask: torch.Tensor | tuple | None = None,
    pooled_embeds: torch.Tensor | None = None,
    txt_ids: torch.Tensor | None = None,
    img_ids: torch.Tensor | None = None,
    original_size: tuple | None = None,
    target_size: tuple | None = None,
    crop_coords: tuple | None = None,
    down_intrablock_additional_residuals: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Unified backbone forward that handles architecture differences.

    SD 1.5/2.1: unet(latents, t, enc_hidden)
    SDXL/SSD-1B: unet(latents, t, enc_hidden, added_cond_kwargs=...)
    Kolors: unet(latents, t, enc_hidden, added_cond_kwargs=...) — same as SDXL
    Kandinsky3: unet(latents, t, enc_hidden, encoder_attention_mask=...)
    FLUX.1: transformer(hidden_states, timestep, guidance, pooled_projections, enc_hidden, txt_ids, img_ids)
    SD3: transformer(hidden_states, timestep, enc_hidden, pooled_projections)
    PixArt: transformer(latents, enc_hidden, timestep, encoder_attention_mask, added_cond_kwargs)
    Qwen-Image: transformer(hidden_states, timestep/1000, guidance, enc_hidden, enc_mask, img_shapes)
    Z-Image: transformer(latent_input, timestep, enc_hidden)
    SANA: transformer(latents, enc_hidden, timestep, encoder_attention_mask)
    SanaSprint: transformer(latents, enc_hidden, timestep, encoder_attention_mask, guidance)
    Lumina2: transformer(hidden_states, timestep, enc_hidden, encoder_attention_mask)
    Chroma: transformer(hidden_states, timestep/1000, enc_hidden, txt_ids, img_ids)
    FLUX.2: transformer(hidden_states, timestep/1000, guidance, enc_hidden, txt_ids, img_ids)
    CogView4: transformer(hidden_states, enc_hidden, timestep, original_size, target_size, crop_coords)
    AuraFlow: transformer(latents, enc_hidden, timestep)
    HunyuanDiT: transformer(latents, t, enc_hidden, text_embedding_mask, enc_hidden_t5, text_embedding_mask_t5, image_meta_size, style, image_rotary_emb)
    """
    if config.backbone_type == "unet":
        unet_kwargs: dict[str, Any] = {}
        if config.needs_added_cond_kwargs and added_cond_kwargs is not None:
            unet_kwargs["added_cond_kwargs"] = added_cond_kwargs
        if config.needs_attention_mask and attention_mask is not None:
            unet_kwargs["encoder_attention_mask"] = attention_mask
        if down_intrablock_additional_residuals is not None:
            unet_kwargs["down_intrablock_additional_residuals"] = (
                down_intrablock_additional_residuals
            )
        return backbone(latents, timesteps, encoder_hidden_states, **unet_kwargs).sample

    # ── Transformer-based models ──
    if config.family == "flux":
        # FLUX.1
        return backbone(
            hidden_states=latents,
            timestep=timesteps,
            guidance=guidance if guidance is not None else 0.0,
            pooled_projections=pooled_embeds,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    elif config.family == "qwen-image":
        return backbone(
            hidden_states=latents,
            timestep=timesteps / 1000,
            guidance=guidance if guidance is not None else 0.0,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=attention_mask,
            return_dict=False,
        )[0]

    elif config.family == "z-image":
        # Z-Image: transformer expects list inputs with frame dimension
        # Pipeline does: latents.unsqueeze(2) → [B, C, 1, H, W] → unbind(0) → [[C, 1, H, W], ...]
        latents_5d = latents.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W]
        latent_list = list(latents_5d.unbind(dim=0))
        # cap_feats: passed as single tensor [B, seq, dim] (matching pipeline behavior)
        output = backbone(
            latent_list,
            timesteps,
            encoder_hidden_states,
            return_dict=False,
        )[0]
        # Z-Image returns list[Tensor] — stack into batch
        if isinstance(output, (list, tuple)):
            output = torch.stack(output, dim=0)
        # Output shape: [B, C, 1, H, W] → squeeze frame dim → [B, C, H, W]
        if output.dim() == 5 and output.shape[2] == 1:
            output = output.squeeze(2)
        return output

    elif config.family == "sana":
        return backbone(
            latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            timestep=timesteps,
            return_dict=False,
        )[0]

    elif config.family == "lumina2":
        return backbone(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=False,
        )[0]

    elif config.family == "chroma":
        return backbone(
            hidden_states=latents,
            timestep=timesteps / 1000,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    elif config.family == "flux2":
        return backbone(
            hidden_states=latents,
            timestep=timesteps / 1000,
            guidance=guidance if guidance is not None else 0.0,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    elif config.family == "cogview4":
        cogview_kwargs = {}
        if original_size is not None:
            cogview_kwargs["original_size"] = original_size
        if target_size is not None:
            cogview_kwargs["target_size"] = target_size
        if crop_coords is not None:
            cogview_kwargs["crop_coords"] = crop_coords
        return backbone(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            return_dict=False,
            **cogview_kwargs,
        )[0]

    elif config.family == "aura-flow":
        return backbone(
            latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            return_dict=False,
        )[0]

    elif config.family == "sd3":
        # SD3: transformer(hidden_states, timestep, encoder_hidden_states, pooled_projections)
        return backbone(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    elif config.family == "pixart":
        # PixArt-Alpha/Sigma: transformer(latents, encoder_hidden_states, timestep, encoder_attention_mask, added_cond_kwargs)
        kwargs: dict[str, Any] = {}
        if config.needs_added_cond_kwargs_dit:
            # PixArt uses resolution-based added_cond_kwargs
            kwargs["added_cond_kwargs"] = {"resolution": None, "aspect_ratio": None}
        return backbone(
            latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            timestep=timesteps,
            return_dict=False,
            **kwargs,
        )[0]

    elif config.family == "kandinsky3":
        # Kandinsky3: unet(latents, t, encoder_hidden_states, encoder_attention_mask)
        return backbone(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=False,
        )[0]

    elif config.family == "hunyuandit":
        # HunyuanDiT: transformer(latents, t, encoder_hidden_states, text_embedding_mask,
        #   encoder_hidden_states_t5, text_embedding_mask_t5, image_meta_size, style, image_rotary_emb)
        # aux_output is (t5_hidden, clip_mask, t5_mask) tuple from encode_text
        t5_hidden = None
        clip_mask = attention_mask
        t5_mask = None
        if isinstance(attention_mask, tuple):
            t5_hidden, clip_mask, t5_mask = attention_mask

        # image_meta_size: [B, 6] (orig_h, orig_w, crop_h, crop_w, target_h, target_w)
        batch_size = latents.shape[0]
        res = latents.shape[-1] * config.vae_downsample_factor
        image_meta_size = (
            torch.tensor(
                [res, res, res, res, res, res],
                device=latents.device,
                dtype=encoder_hidden_states.dtype,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        style = torch.zeros(batch_size, device=latents.device, dtype=encoder_hidden_states.dtype)

        return backbone(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            text_embedding_mask=clip_mask,
            encoder_hidden_states_t5=t5_hidden,
            text_embedding_mask_t5=t5_mask,
            image_meta_size=image_meta_size,
            style=style,
            image_rotary_emb=None,
            return_dict=False,
        )[0]

    elif config.family == "sana-sprint":
        # SanaSprint: same as SANA but with guidance embeds
        return backbone(
            latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            guidance=guidance if guidance is not None else 0.0,
            timestep=timesteps,
            return_dict=False,
        )[0]

    elif config.family == "hunyuan-image":
        # HunyuanImage: dual-encoder DiT, needs attention_mask + guidance
        # attention_mask may be a tuple (clip_mask, t5_hidden, t5_mask) from encode_text
        # or a plain tensor (clip_mask only)
        enc_mask = attention_mask
        enc_hidden_t5 = None
        enc_mask_t5 = None
        if isinstance(attention_mask, tuple):
            enc_mask, enc_hidden_t5, enc_mask_t5 = attention_mask

        kwargs = {
            "hidden_states": latents,
            "timestep": timesteps / 1000,
            "guidance": guidance if guidance is not None else 0.0,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": enc_mask,
            "return_dict": False,
        }
        if enc_hidden_t5 is not None:
            kwargs["encoder_hidden_states_t5"] = enc_hidden_t5
        if enc_mask_t5 is not None:
            kwargs["encoder_hidden_states_mask_t5"] = enc_mask_t5
        return backbone(**kwargs)[0]

    elif config.family == "longcat-image":
        # LongCat: transformer(hidden_states, encoder_hidden_states, timestep/1000, img_ids, txt_ids, guidance)
        kwargs = {
            "hidden_states": latents,
            "timestep": timesteps / 1000,
            "encoder_hidden_states": encoder_hidden_states,
            "txt_ids": txt_ids
            if txt_ids is not None
            else torch.zeros(
                latents.shape[0],
                encoder_hidden_states.shape[1],
                3,
                device=latents.device,
                dtype=encoder_hidden_states.dtype,
            ),
            "img_ids": img_ids
            if img_ids is not None
            else torch.zeros(
                latents.shape[0],
                latents.shape[1],
                3,
                device=latents.device,
                dtype=encoder_hidden_states.dtype,
            ),
            "return_dict": False,
        }
        if config.needs_guidance_in_forward:
            kwargs["guidance"] = guidance if guidance is not None else 0.0
        return backbone(**kwargs)[0]

    elif config.family == "lumina":
        # Lumina-Next: transformer(hidden_states, timestep, encoder_hidden_states, encoder_mask, image_rotary_emb)
        kwargs = {
            "hidden_states": latents,
            "timestep": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_mask": attention_mask if attention_mask is not None else None,
            "return_dict": False,
        }
        return backbone(**kwargs)[0]

    elif config.family == "nucleusmoe-image":
        # NucleusMoE: transformer(hidden_states, img_shapes, encoder_hidden_states, encoder_mask, timestep/num_train_timesteps)
        batch_size = latents.shape[0]
        # NucleusMoE uses sequence-based latents; img_shapes describes spatial dims
        # latents shape: [B, seq_len, C] — derive spatial from seq_len
        patch_size = getattr(backbone.config, "patch_size", 2)
        vae_sf = config.vae_downsample_factor
        # Compute spatial dims from latent shape
        if latents.dim() == 4:
            # [B, C, H, W] format
            latent_h, latent_w = latents.shape[2], latents.shape[3]
            img_h = latent_h * vae_sf
            img_w = latent_w * vae_sf
            # seq dimensions derived implicitly via img_shapes below
        else:
            # [B, seq_len, C] format — assume square
            seq_len = latents.shape[1]
            seq_side = int(seq_len**0.5)
            img_h = seq_side * patch_size * vae_sf
            img_w = img_h
        img_shapes = [
            (1, img_h // (vae_sf * patch_size), img_w // (vae_sf * patch_size))
        ] * batch_size

        mask = attention_mask
        if isinstance(attention_mask, tuple):
            mask = attention_mask[0]

        return backbone(
            hidden_states=latents,
            img_shapes=img_shapes,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=mask,
            timestep=timesteps,
            return_dict=False,
        )[0]

    elif config.family == "ernie-image":
        # Ernie: transformer(hidden_states, timestep, text_bth, text_lens)
        mask = attention_mask
        text_lens = None
        if isinstance(attention_mask, tuple):
            mask, text_lens = attention_mask

        if text_lens is None and mask is not None and isinstance(mask, torch.Tensor):
            text_lens = mask.sum(dim=1).to(dtype=torch.long)

        return backbone(
            hidden_states=latents,
            timestep=timesteps,
            text_bth=encoder_hidden_states,
            text_lens=text_lens,
            return_dict=False,
        )[0]

    elif config.family == "hidream-image":
        # HiDream: transformer(hidden_states, timesteps, enc_t5, enc_llama3, pooled_embeds, img_ids, img_sizes, hidden_states_masks)
        pooled = None
        llama_hidden = None
        llama_mask = None
        if isinstance(attention_mask, tuple):
            # encode_text returns (pooled_embeds, llama_hidden, llama_mask)
            pooled, llama_hidden, llama_mask = attention_mask

        batch_size = latents.shape[0]
        # HiDream uses 2x2 patch packing; img_ids and img_sizes are needed
        vae_sf = config.vae_downsample_factor
        patch_size = 2  # HiDream packs 2x2
        if latents.dim() == 4:
            latent_h, latent_w = latents.shape[2], latents.shape[3]
            img_ids = torch.zeros(
                batch_size,
                latent_h * latent_w,
                3,
                device=latents.device,
                dtype=encoder_hidden_states.dtype,
            )
            img_sizes = [(latent_h, latent_w)] * batch_size
        else:
            seq_len = latents.shape[1]
            side = int(seq_len**0.5)
            img_ids = torch.zeros(
                batch_size, seq_len, 3, device=latents.device, dtype=encoder_hidden_states.dtype
            )
            img_sizes = [(side, side)] * batch_size

        kwargs = {
            "hidden_states": latents,
            "timesteps": timesteps,
            "encoder_hidden_states_t5": encoder_hidden_states,
            "return_dict": False,
        }
        if llama_hidden is not None:
            kwargs["encoder_hidden_states_llama3"] = llama_hidden
        if pooled is not None:
            kwargs["pooled_embeds"] = pooled
        kwargs["img_ids"] = img_ids
        kwargs["img_sizes"] = img_sizes
        if llama_mask is not None:
            kwargs["hidden_states_masks"] = llama_mask

        return backbone(**kwargs)[0]

    else:
        raise ValueError(f"Unsupported family for backbone_forward: {config.family}")


# ─── Backward-compat alias ─────────────────────────────────────────────────


def unet_forward(
    unet: torch.nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    config: ModelConfig,
    added_cond_kwargs: dict | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for backbone_forward (UNet models only)."""
    return backbone_forward(
        backbone=unet,
        latents=latents,
        timesteps=timesteps,
        encoder_hidden_states=encoder_hidden_states,
        config=config,
        added_cond_kwargs=added_cond_kwargs,
    )


# ─── VAE encode/decode helpers ─────────────────────────────────────────────


def vae_encode(
    vae: torch.nn.Module,
    images: torch.Tensor,
    config: ModelConfig,
) -> torch.Tensor:
    """Encode images to latents with model-appropriate scaling.

    SD family: vae.encode(img).latent_dist.mode() * scaling_factor
    SANA/AuraFlow: vae.encode(img).latent_dist.mode() (scaling applied at decode)
    Z-Image/Lumina2/Chroma: (vae.encode(img).latent_dist.mode() - shift) * scaling
    Qwen-Image: vae.encode(img).latent_dist.mode() * scaling
    FLUX.2: retrieve_latents(vae.encode(img)) then patchify
    CogView4: vae.encode(img).latent_dist.mode() * scaling
    """

    # Encode
    if config.vae_has_latent_dist:
        latents = vae.encode(images).latent_dist.mode()
    else:
        # Some custom VAEs (e.g. SANA's AutoencoderDC) may return sample directly
        enc_out = vae.encode(images)
        latents = enc_out.latent_dist.mode() if hasattr(enc_out, "latent_dist") else enc_out.sample

    # Apply scaling (and shift for FLUX/Z-Image style VAEs)
    scaling = config.vae_scaling_factor
    if scaling is None:
        cfg_scaling = getattr(vae.config, "scaling_factor", None)
        scaling = cfg_scaling if cfg_scaling is not None else 1.0

    shift = config.vae_shift_factor
    if shift is None:
        shift = getattr(vae.config, "shift_factor", None) or 0.0

    # Z-Image/FLUX VAE: latents = (encoded - shift) * scaling
    latents = (latents - shift) * scaling if shift != 0.0 else latents * scaling

    return latents


def vae_decode(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    config: ModelConfig,
) -> torch.Tensor:
    """Decode latents to images with model-appropriate unscaling.

    SD family: vae.decode(latents / scaling_factor).sample
    SANA/AuraFlow: vae.decode(latents / scaling_factor)[0]
    Z-Image/Lumina2/Chroma: (latents / scaling) + shift, then vae.decode(...)
    CogView4: latents / scaling, then vae.decode(...)
    """
    scaling = config.vae_scaling_factor
    shift = 0.0

    if scaling is None:
        scaling = getattr(vae.config, "scaling_factor", 1.0)
    if scaling is None:
        scaling = 1.0
    shift = getattr(vae.config, "shift_factor", None)
    if shift is None:
        shift = 0.0

    # Apply unscaling + shift
    latents = latents / scaling + shift

    # Decode
    decoded = vae.decode(latents, return_dict=False)
    sample = decoded[0] if isinstance(decoded, tuple) else decoded.sample

    return sample


# ─── Position IDs helpers (for FLUX/Chroma/FLUX.2) ────────────────────────


def prepare_position_ids(
    latents: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple:
    """Prepare txt_ids and img_ids for models that need them.

    Returns (txt_ids, img_ids) or (None, None) if not needed.
    """
    if not config.needs_position_ids:
        return None, None

    batch_size = latents.shape[0]

    # txt_ids: [B, text_seq_len, 4] for FLUX.2, [B, text_seq_len, 3] for Chroma
    text_seq_len = encoder_hidden_states.shape[1]
    id_dim = 4 if config.family == "flux2" else 3
    txt_ids = torch.zeros(batch_size, text_seq_len, id_dim, device=device, dtype=dtype)

    # img_ids: depends on latent shape
    if config.family in ("flux", "flux2"):
        # FLUX uses packed latents: seq_len = (H/16) * (W/16) * 4
        # But for training, we use the actual latent sequence length
        img_seq_len = (
            latents.shape[1] if latents.dim() == 3 else latents.shape[2] * latents.shape[3]
        )
        img_ids = torch.zeros(batch_size, img_seq_len, id_dim, device=device, dtype=dtype)
    elif config.family == "chroma":
        img_seq_len = (
            latents.shape[1] if latents.dim() == 3 else latents.shape[2] * latents.shape[3]
        )
        img_ids = torch.zeros(batch_size, img_seq_len, id_dim, device=device, dtype=dtype)
    else:
        img_ids = None

    return txt_ids, img_ids
