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
    family: str  # "sd15" | "sd2" | "sdxl" | "flux" | "qwen-image" | "z-image" | "sana" | "lumina2" | "chroma" | "flux2" | "cogview4" | "aura-flow"

    # Text encoder
    num_text_encoders: int  # 1 for SD1.5/SD2, 2 for SDXL/FLUX

    # Backbone type: "unet" for SD family, "transformer" for DiT-based models
    backbone_type: str = "unet"
    text_encoder_max_length: int = 77

    # Architecture flags
    needs_added_cond_kwargs: bool = False  # SDXL+ requires pooled embeds + time_ids
    cross_attention_dim: int = 768  # SD1.5=768, SD2=1024, SDXL=2048

    # VAE
    vae_scaling_factor: float = 0.18215  # value scaling; None = read from vae.config.scaling_factor
    vae_downsample_factor: int = 8  # spatial downsample

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

    # Transformer-specific flags (ignored for UNet models)
    needs_guidance_in_forward: bool = False  # Qwen-Image, FLUX.2 pass guidance to transformer
    needs_attention_mask: bool = False  # SANA, Lumina2 need encoder_attention_mask
    needs_position_ids: bool = False  # Chroma, FLUX.2 need txt_ids/img_ids
    needs_size_kwargs: bool = False  # CogView4 needs original_size/target_size/crop_coords

    # Whether VAE encode returns latent_dist (False for some custom VAEs)
    vae_has_latent_dist: bool = True

    # Description
    description: str = ""


# ─── Registry ───────────────────────────────────────────────────────────────

REGISTRY: dict[str, ModelConfig] = {

    "sd15": ModelConfig(
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
        backbone_type="unet",
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

    "sdxl": ModelConfig(
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

    "sdxl-turbo": ModelConfig(
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
        default_guidance_scale=0.0,  # Turbo: no CFG
        default_num_inference_steps=4,
        description="SDXL Turbo (2.6B UNet) — 1-4 step generation, no CFG",
    ),

    "flux-dev": ModelConfig(
        name="flux-dev",
        model_id="black-forest-labs/FLUX.1-dev",
        family="flux",
        backbone_type="transformer",
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
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "to_q_context", "to_k_context", "to_v_context", "to_out.0_context",
        ],
        description="FLUX.1-schnell (12B) — 4-step generation, Apache-2.0",
    ),

    # ──── New DiT-based models (diffusers 0.38+) ────

    "qwen-image": ModelConfig(
        name="qwen-image",
        model_id="Qwen/Qwen-Image",
        family="qwen-image",
        backbone_type="transformer",
        num_text_encoders=1,
        needs_added_cond_kwargs=False,
        cross_attention_dim=4096,
        vae_scaling_factor=None,  # read from vae.config
        vae_downsample_factor=8,
        memory_budget_gb=20.0,
        pipeline_class="QwenImagePipeline",
        default_resolution=1024,
        default_guidance_scale=4.0,
        default_num_inference_steps=50,
        default_lora_rank=32,
        default_lora_alpha=64,
        lora_target_modules=[
            "to_q", "to_k", "to_v",
            "q_proj", "k_proj", "v_proj",
            "proj_out",
            "img_in", "txt_in",
        ],
        needs_guidance_in_forward=True,
        needs_attention_mask=True,
        vae_has_latent_dist=False,
        description="Qwen-Image — text rendering, CFG, Apache-2.0",
    ),

    "z-image": ModelConfig(
        name="z-image",
        model_id="Z-a-o/Z-Image-Turbo",
        family="z-image",
        backbone_type="transformer",
        num_text_encoders=1,
        needs_added_cond_kwargs=False,
        cross_attention_dim=4096,
        vae_scaling_factor=None,
        vae_downsample_factor=8,
        memory_budget_gb=24.0,
        pipeline_class="ZImagePipeline",
        default_resolution=1024,
        default_guidance_scale=5.0,
        default_num_inference_steps=50,
        default_lora_rank=32,
        default_lora_alpha=64,
        lora_target_modules=[
            "w1", "w2", "w3",
            "linear",
            "in_proj", "out_proj",
        ],
        vae_has_latent_dist=False,
        description="Z-Image-Turbo (6B) — Alibaba, single-stream DiT, Apache-2.0",
    ),

    "sana-600m": ModelConfig(
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
            "linear", "proj_out",
            "conv_inverted", "conv_point",
        ],
        needs_attention_mask=True,
        vae_has_latent_dist=False,
        description="SANA-600M — NVIDIA, linear DiT, efficient 1024px",
    ),

    "sana-1600m": ModelConfig(
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
            "linear", "proj_out",
            "conv_inverted", "conv_point",
        ],
        needs_attention_mask=True,
        vae_has_latent_dist=False,
        description="SANA-1600M — NVIDIA, linear DiT, higher quality 1024px",
    ),

    "lumina2": ModelConfig(
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
            "x_embedder", "proj_out",
            "to_q", "to_k", "to_v", "to_out.0",
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        needs_attention_mask=True,
        vae_has_latent_dist=False,
        description="Lumina-Image-2.0 (2B) — Alpha-VLLM, flow DiT",
    ),

    "chroma": ModelConfig(
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
            "in_proj", "out_proj", "proj_mlp", "proj_out",
            "context_embedder", "x_embedder",
            "ff_context.net.0.proj", "ff_context.net.2",
        ],
        needs_position_ids=True,
        vae_has_latent_dist=True,
        description="Chroma1-HD (8.9B) — FLUX.1-schnell base, Apache-2.0",
    ),

    "flux2-dev": ModelConfig(
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
            "to_q", "to_k", "to_v", "to_out", "to_add_out",
            "to_qkv_mlp_proj",
            "x_embedder", "context_embedder", "proj_out",
            "linear", "linear_in", "linear_out",
        ],
        needs_guidance_in_forward=True,
        needs_position_ids=True,
        vae_has_latent_dist=True,
        description="FLUX.2-dev — Black Forest Labs, next-gen DiT",
    ),

    "cogview4": ModelConfig(
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
            "proj", "text_proj", "linear", "proj_out",
            "to_q", "to_k", "to_v", "to_out.0",
        ],
        needs_size_kwargs=True,
        vae_has_latent_dist=False,
        description="CogView4-6B — THUDM/Tsinghua, Apache-2.0",
    ),

    "aura-flow": ModelConfig(
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
            "linear_1", "linear_2", "out_projection", "linear",
            "context_embedder", "proj_out",
            "to_q", "to_k", "to_v", "to_out.0",
            "ff_context.net.0.proj", "ff_context.net.2",
        ],
        vae_has_latent_dist=False,
        description="AuraFlow — fal.ai, MMDiT, Apache-2.0",
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
    text_encoder: Optional[torch.nn.Module]
    text_encoder_2: Optional[torch.nn.Module]
    tokenizer: object
    scheduler: object = None
    config: ModelConfig = None

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
            backbone = getattr(pipe, "unet")
    else:
        backbone = pipe.unet

    backbone = backbone.to(device)
    vae = pipe.vae.to(device)
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is not None:
        text_encoder = text_encoder.to(device)
    text_encoder_2 = None
    tokenizer = getattr(pipe, "tokenizer", None)

    if config.num_text_encoders >= 2:
        text_encoder_2 = getattr(pipe, "text_encoder_2", None)
        if text_encoder_2 is not None:
            text_encoder_2 = text_encoder_2.to(device)

    # Freeze base model
    backbone.requires_grad_(False)
    vae.requires_grad_(False)
    if text_encoder is not None:
        text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)

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

    Handles:
      - Single-encoder (SD1.5/2.1, all new DiT models)
      - Dual-encoder (SDXL: concatenated hidden states + pooled; FLUX.1: T5 hidden states)

    For DiT models (Qwen-Image, Z-Image, SANA, Lumina2, etc.), returns
    (encoder_hidden_states, attention_mask) where attention_mask replaces pooled_embeds
    for models that need it.

    Returns:
      (encoder_hidden_states, pooled_embeds_or_mask)
      - SD1.5/2.1: (hidden_states, None)
      - SDXL: (concatenated_hidden_states, pooled_embeds)
      - FLUX.1: (t5_hidden_states, None)
      - DiT models with needs_attention_mask: (hidden_states, attention_mask)
      - Other DiT models: (hidden_states, None)
    """
    tokenizer = components.tokenizer
    text_encoder = components.text_encoder
    text_encoder_2 = components.text_encoder_2
    config = components.config
    max_length = config.text_encoder_max_length

    # Tokenize — most models use standard padding
    tokenized = tokenizer(
        prompts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
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
    added_cond_kwargs: Optional[dict] = None,
    guidance: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
    pooled_embeds: Optional[torch.Tensor] = None,
    txt_ids: Optional[torch.Tensor] = None,
    img_ids: Optional[torch.Tensor] = None,
    original_size: Optional[tuple] = None,
    target_size: Optional[tuple] = None,
    crop_coords: Optional[tuple] = None,
) -> torch.Tensor:
    """Unified backbone forward that handles architecture differences.

    SD 1.5/2.1: unet(latents, t, enc_hidden)
    SDXL/SSD-1B: unet(latents, t, enc_hidden, added_cond_kwargs=...)
    FLUX.1: transformer(hidden_states, timestep, guidance, pooled_projections, enc_hidden, txt_ids, img_ids)
    Qwen-Image: transformer(hidden_states, timestep/1000, guidance, enc_hidden, enc_mask, img_shapes)
    Z-Image: transformer(latent_input, timestep, enc_hidden)
    SANA: transformer(latents, enc_hidden, timestep, encoder_attention_mask)
    Lumina2: transformer(hidden_states, timestep, enc_hidden, encoder_attention_mask)
    Chroma: transformer(hidden_states, timestep/1000, enc_hidden, txt_ids, img_ids)
    FLUX.2: transformer(hidden_states, timestep/1000, guidance, enc_hidden, txt_ids, img_ids)
    CogView4: transformer(hidden_states, enc_hidden, timestep, original_size, target_size, crop_coords)
    AuraFlow: transformer(latents, enc_hidden, timestep)
    """
    if config.backbone_type == "unet":
        kwargs = {}
        if config.needs_added_cond_kwargs and added_cond_kwargs is not None:
            kwargs["added_cond_kwargs"] = added_cond_kwargs
        return backbone(latents, timesteps, encoder_hidden_states, **kwargs).sample

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
        # Z-Image: positional args (latent_input, timestep, enc_hidden)
        return backbone(
            latents, timesteps, encoder_hidden_states,
            return_dict=False,
        )[0]

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
        kwargs = {}
        if original_size is not None:
            kwargs["original_size"] = original_size
        if target_size is not None:
            kwargs["target_size"] = target_size
        if crop_coords is not None:
            kwargs["crop_coords"] = crop_coords
        return backbone(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            return_dict=False,
            **kwargs,
        )[0]

    elif config.family == "aura-flow":
        return backbone(
            latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            return_dict=False,
        )[0]

    else:
        raise ValueError(f"Unsupported family for backbone_forward: {config.family}")


# ─── Backward-compat alias ─────────────────────────────────────────────────

def unet_forward(
    unet: torch.nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    config: ModelConfig,
    added_cond_kwargs: Optional[dict] = None,
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
    Z-Image/Lumina2/Chroma: vae.encode(img).latent_dist.mode() * scaling (shift at decode)
    Qwen-Image: vae.encode(img).latent_dist.mode() * scaling
    FLUX.2: retrieve_latents(vae.encode(img)) then patchify
    CogView4: vae.encode(img).latent_dist.mode() * scaling
    """
    from diffusers.utils.torch_utils import randn_tensor

    # Encode
    if config.vae_has_latent_dist:
        latents = vae.encode(images).latent_dist.mode()
    else:
        # Some custom VAEs (e.g. SANA's AutoencoderDC) may return sample directly
        enc_out = vae.encode(images)
        if hasattr(enc_out, 'latent_dist'):
            latents = enc_out.latent_dist.mode()
        else:
            latents = enc_out.sample

    # Apply scaling
    scaling = config.vae_scaling_factor
    if scaling is not None:
        latents = latents * scaling
    else:
        # Read from vae.config if available
        cfg_scaling = getattr(vae.config, 'scaling_factor', None)
        if cfg_scaling is not None:
            latents = latents * cfg_scaling

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
        scaling = getattr(vae.config, 'scaling_factor', 1.0)
    shift = getattr(vae.config, 'shift_factor', 0.0)

    # Apply unscaling + shift
    latents = latents / scaling + shift

    # Decode
    decoded = vae.decode(latents, return_dict=False)
    if isinstance(decoded, tuple):
        sample = decoded[0]
    else:
        sample = decoded.sample

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
        img_seq_len = latents.shape[1] if latents.dim() == 3 else latents.shape[2] * latents.shape[3]
        img_ids = torch.zeros(batch_size, img_seq_len, id_dim, device=device, dtype=dtype)
    elif config.family == "chroma":
        img_seq_len = latents.shape[1] if latents.dim() == 3 else latents.shape[2] * latents.shape[3]
        img_ids = torch.zeros(batch_size, img_seq_len, id_dim, device=device, dtype=dtype)
    else:
        img_ids = None

    return txt_ids, img_ids
