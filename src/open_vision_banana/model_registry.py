#!/usr/bin/env python3
"""Unified model registry for multi-backbone Vision Banana training/inference.

Supports different base model families with varying architectures:
  - SD 1.5 / SD 2.1  (single text encoder, no added_cond_kwargs)
  - SSD-1B / SDXL     (dual text encoder, added_cond_kwargs required)
  - SDXL-Turbo        (same as SDXL, fewer inference steps)
  - FLUX-dev/schnell  (dual transformer, different pipeline)

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
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger("vb_model_registry")


# ─── Model family configs ──────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Configuration for a base diffusion model family."""
    name: str
    model_id: str
    family: str  # "sd15" | "sd2" | "sdxl" | "flux"

    # Text encoder
    num_text_encoders: int  # 1 for SD1.5/SD2, 2 for SDXL/FLUX
    text_encoder_max_length: int = 77

    # Architecture flags
    needs_added_cond_kwargs: bool = False  # SDXL+ requires pooled embeds + time_ids
    cross_attention_dim: int = 768  # SD1.5=768, SD2=1024, SDXL=2048

    # VAE
    vae_scaling_factor: float = 0.18215
    vae_downsample_factor: int = 8

    # LoRA
    lora_target_modules: list = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
    ])
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

    # Description
    description: str = ""


# ─── Registry ───────────────────────────────────────────────────────────────

REGISTRY: dict[str, ModelConfig] = {

    "sd15": ModelConfig(
        name="sd15",
        model_id="runwayml/stable-diffusion-v1-5",
        family="sd15",
        num_text_encoders=1,
        cross_attention_dim=768,
        vae_scaling_factor=0.18215,
        memory_budget_gb=4.0,
        pipeline_class="StableDiffusionPipeline",
        default_resolution=256,
        default_guidance_scale=7.5,
        lora_target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
        description="Stable Diffusion 1.5 (860M UNet) — lightweight, widely compatible",
    ),

    "sd2": ModelConfig(
        name="sd2",
        model_id="stabilityai/stable-diffusion-2-1",
        family="sd2",
        num_text_encoders=1,
        cross_attention_dim=1024,
        vae_scaling_factor=0.18215,
        memory_budget_gb=5.0,
        pipeline_class="StableDiffusionPipeline",
        default_resolution=256,
        default_guidance_scale=7.5,
        lora_target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
        description="Stable Diffusion 2.1 (866M UNet) — improved text understanding",
    ),

    "ssd-1b": ModelConfig(
        name="ssd-1b",
        model_id="segmind/SSD-1B",
        family="sdxl",
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

    "sdxl": ModelConfig(
        name="sdxl",
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        family="sdxl",
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

    "sdxl-turbo": ModelConfig(
        name="sdxl-turbo",
        model_id="stabilityai/sdxl-turbo",
        family="sdxl",
        num_text_encoders=2,
        needs_added_cond_kwargs=True,
        cross_attention_dim=2048,
        vae_scaling_factor=0.18215,
        memory_budget_gb=7.0,
        pipeline_class="StableDiffusionXLPipeline",
        default_resolution=256,
        default_guidance_scale=0.0,  # Turbo: no CFG
        default_num_inference_steps=4,
        description="SDXL Turbo (2.6B UNet) — 1-4 step generation, no CFG",
    ),

    "flux-dev": ModelConfig(
        name="flux-dev",
        model_id="black-forest-labs/FLUX.1-dev",
        family="flux",
        num_text_encoders=2,
        needs_added_cond_kwargs=False,  # FLUX uses different conditioning
        cross_attention_dim=3072,
        vae_scaling_factor=0.3611,
        vae_downsample_factor=16,  # FLUX VAE has 16x downsample
        memory_budget_gb=24.0,
        pipeline_class="FluxPipeline",
        default_resolution=512,
        default_guidance_scale=3.5,
        default_lora_rank=32,
        default_lora_alpha=64,
        lora_target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            # FLUX transformer also has:
            "to_q_context", "to_k_context", "to_v_context", "to_out.0_context",
        ],
        description="FLUX.1-dev (12B transformer) — state-of-the-art, needs ≥24GB VRAM",
    ),

    "flux-schnell": ModelConfig(
        name="flux-schnell",
        model_id="black-forest-labs/FLUX.1-schnell",
        family="flux",
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
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "to_q_context", "to_k_context", "to_v_context", "to_out.0_context",
        ],
        description="FLUX.1-schnell (12B) — 4-step generation, Apache-2.0",
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Look up a model config by name. Raises KeyError if not found."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY.keys()))
        raise KeyError(
            f"Unknown model '{name}'. Available: {available}"
        )
    return REGISTRY[name]


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(REGISTRY.keys())


def get_models_by_family(family: str) -> list[str]:
    """Return model names belonging to a family."""
    return [k for k, v in REGISTRY.items() if v.family == family]


def auto_select_model(memory_gb: float) -> str:
    """Auto-select the best model that fits within a memory budget.

    Preference order (best quality within budget):
      sdxl > ssd-1b > sd2 > sd15
    (FLUX models excluded from auto-select — require explicit choice)
    """
    candidates = [
        ("sdxl", 7.0),
        ("ssd-1b", 5.5),
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
    unet: torch.nn.Module
    vae: torch.nn.Module
    text_encoder: Optional[torch.nn.Module]
    text_encoder_2: Optional[torch.nn.Module]
    tokenizer: object
    scheduler: object = None
    config: ModelConfig = None


def load_pipeline(
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    enable_gradient_checkpointing: bool = True,
) -> PipelineComponents:
    """Load a diffusion pipeline and extract components.

    Handles architecture differences across model families:
      - SD 1.5/2.1: single text encoder, no added_cond_kwargs
      - SDXL/SSD-1B/Turbo: dual text encoder, added_cond_kwargs required
      - FLUX: dual text encoder, different transformer, no added_cond_kwargs

    Returns PipelineComponents with unet, vae, text_encoder(s), tokenizer.
    """
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    logger.info(f"Loading {config.model_id} ({config.family}) @ {dtype} ...")

    # Lazy import the correct pipeline class
    if config.pipeline_class == "StableDiffusionXLPipeline":
        from diffusers import StableDiffusionXLPipeline as PipeClass
    elif config.pipeline_class == "StableDiffusionPipeline":
        from diffusers import StableDiffusionPipeline as PipeClass
    elif config.pipeline_class == "FluxPipeline":
        from diffusers import FluxPipeline as PipeClass
    else:
        raise ValueError(f"Unknown pipeline class: {config.pipeline_class}")

    pipe = PipeClass.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
    )

    # Extract components
    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    text_encoder_2 = None
    tokenizer = pipe.tokenizer

    if config.num_text_encoders >= 2:
        text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        if text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to(device)

    # Freeze base model
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)

    # Enable gradient checkpointing for memory efficiency
    if enable_gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    unet_params = sum(p.numel() for p in unet.parameters())
    logger.info(
        f"UNet params: {unet_params / 1e6:.1f}M | "
        f"Text encoders: {config.num_text_encoders} | "
        f"added_cond: {config.needs_added_cond_kwargs}"
    )

    return PipelineComponents(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        scheduler=getattr(pipe, "scheduler", None),
        config=config,
    )


# ─── Text encoding helpers ─────────────────────────────────────────────────

def encode_text(
    components: PipelineComponents,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    negative: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Encode text prompts into hidden states + optional pooled embeddings.

    Handles both single-encoder (SD1.5/2.1) and dual-encoder (SDXL/FLUX) architectures.

    Returns:
      (encoder_hidden_states, pooled_embeds)
      - pooled_embeds is None for single-encoder models
    """
    tokenizer = components.tokenizer
    text_encoder = components.text_encoder
    text_encoder_2 = components.text_encoder_2
    max_length = components.config.text_encoder_max_length

    tokenized = tokenizer(
        prompts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)

    encoder_hidden_states = text_encoder(tokenized)[0]
    pooled_embeds = None

    if text_encoder_2 is not None:
        te2_out = text_encoder_2(tokenized)
        # SDXL: concatenate hidden states along last dim
        if components.config.family == "sdxl":
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states, te2_out.last_hidden_state], dim=-1
            )
            pooled_embeds = te2_out.text_embeds  # [B, 1280]
        elif components.config.family == "flux":
            # FLUX: uses CLIP hidden states + T5 for joint attention
            encoder_hidden_states = te2_out.last_hidden_state  # T5 hidden states
            pooled_embeds = None  # FLUX doesn't use pooled embeds like SDXL

    return encoder_hidden_states.to(dtype), (
        pooled_embeds.to(dtype) if pooled_embeds is not None else None
    )


def build_added_cond_kwargs(
    config: ModelConfig,
    pooled_embeds: Optional[torch.Tensor],
    resolution: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 1,
) -> Optional[dict]:
    """Build added_cond_kwargs for SDXL-family UNets.

    Returns None for non-SDXL models (SD1.5/2.1/FLUX).
    """
    if not config.needs_added_cond_kwargs:
        return None

    if pooled_embeds is None:
        return None

    h, w = resolution
    add_time_ids = torch.tensor(
        [h, w, 0, 0, h, w],
        device=device, dtype=dtype,
    ).unsqueeze(0).expand(batch_size, -1)

    return {
        "text_embeds": pooled_embeds,
        "time_ids": add_time_ids,
    }


# ─── LoRA setup ────────────────────────────────────────────────────────────

def apply_lora(
    unet: torch.nn.Module,
    config: ModelConfig,
    rank: Optional[int] = None,
    alpha: Optional[int] = None,
    use_rslora: bool = True,
) -> torch.nn.Module:
    """Apply LoRA adapters to UNet with model-appropriate target modules."""
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


# ─── UNet forward helper ──────────────────────────────────────────────────

def unet_forward(
    unet: torch.nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    config: ModelConfig,
    added_cond_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """Unified UNet forward that handles architecture differences.

    SD 1.5/2.1: no added_cond_kwargs
    SDXL/SSD-1B: added_cond_kwargs required
    FLUX: no added_cond_kwargs (uses different conditioning)
    """
    kwargs = {}
    if config.needs_added_cond_kwargs and added_cond_kwargs is not None:
        kwargs["added_cond_kwargs"] = added_cond_kwargs

    return unet(
        latents, timesteps, encoder_hidden_states, **kwargs
    ).sample
