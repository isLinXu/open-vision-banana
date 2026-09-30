"""Unified model backend for multi-model training.

ModelBackend wraps model_registry functions (encode_text, backbone_forward,
vae_encode/vae_decode, apply_lora, load_pipeline) and adds model-specific
logic: quantization, accelerate hook removal, latent packing, guidance, etc.

This eliminates the need for model-specific training scripts — a single
Trainer class can work with any registered model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from ..model_registry import (
    PipelineComponents,
    apply_lora,
    backbone_forward,
    load_pipeline,
    resolve_model_config,
)
from ..model_registry import (
    encode_text as registry_encode_text,
)
from ..model_registry import (
    vae_decode as registry_vae_decode,
)
from ..model_registry import (
    vae_encode as registry_vae_encode,
)
from .flow_matching import (
    FlowScheduler,
    pack_latents,
    prepare_latent_image_ids,
    unpack_latents,
)
from .sana_flow import native_sana_timestep

logger = logging.getLogger(__name__)

# bitsandbytes (0.50+) logs "MatMul8bitLt: inputs will be cast from ... to float16" on
# every LLM.int8 forward with non-fp16 inputs. The 8-bit T5 runs every training step,
# so a FLUX run writes hundreds of thousands of identical lines per hour into train.log
# and wandb. (Older releases used warnings.warn, which Python already shows once.)
_BNB_INT8_LOGGER = "bitsandbytes.autograd._functions"
_BNB_INT8_CAST_MSG = "MatMul8bitLt: inputs will be cast"


class _CastWarningOnceFilter(logging.Filter):
    """Let the first int8 cast warning through and drop its repeats; other records pass."""

    def __init__(self) -> None:
        super().__init__()
        self._seen = False

    def filter(self, record: logging.LogRecord) -> bool:
        if not str(record.msg).startswith(_BNB_INT8_CAST_MSG):
            return True
        if self._seen:
            return False
        self._seen = True
        return True


def _log_bnb_int8_cast_warning_once() -> None:
    bnb_logger = logging.getLogger(_BNB_INT8_LOGGER)
    if not any(isinstance(f, _CastWarningOnceFilter) for f in bnb_logger.filters):
        bnb_logger.addFilter(_CastWarningOnceFilter())


class ModelBackend:
    """Unified model interface for training.

    Wraps model_registry functions and adds model-specific logic:
      - Quantization (4-bit/8-bit for large models like FLUX-12B)
      - Accelerate hook removal (FLUX)
      - Latent packing (FLUX, HiDream)
      - Guidance parameter handling
      - FLUX-specific text encoding (pooled_projections, text_ids)
      - SANA-specific VAE latent normalization
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        quantize: str | None = None,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        use_rslora: bool = True,
        lora_targets: list[str] | None = None,
        enable_gradient_checkpointing: bool = True,
    ):
        """Load pipeline and setup backend.

        Args:
            model_name: Registry name (e.g. "flux-dev", "sana-1600m") or HF model ID.
            device: Target device.
            dtype: Model dtype.
            quantize: None | "4bit" | "8bit" — for large models like FLUX-12B.
            lora_rank: LoRA rank.
            lora_alpha: LoRA alpha.
            use_rslora: Whether to use rsLoRA scaling.
            lora_targets: Override LoRA target modules.
            enable_gradient_checkpointing: Enable gradient checkpointing on backbone.
        """
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.quantize = quantize

        # Resolve model config
        self.model_config = resolve_model_config(model_name)
        self.family = self.model_config.family

        logger.info(
            "Loading model '%s' (family=%s, backbone=%s) on %s @ %s",
            model_name,
            self.family,
            self.model_config.backbone_type,
            device,
            dtype,
        )

        # Load pipeline components
        if quantize:
            self._load_quantized(quantize)
        else:
            self.components = load_pipeline(
                self.model_config,
                device=device,
                dtype=dtype,
                enable_gradient_checkpointing=enable_gradient_checkpointing,
            )

        # Store references
        self.backbone = self.components.unet  # backbone stored as 'unet' for compat
        self.vae = self.components.vae
        self.pipe = getattr(self, "_pipe", None)  # Full pipe for FLUX encode_prompt

        # Apply LoRA
        logger.info(
            "Applying LoRA (rank=%d, alpha=%d, rsLoRA=%s)...", lora_rank, lora_alpha, use_rslora
        )
        self.peft_model = apply_lora(
            self.backbone,
            config=self.model_config,
            rank=lora_rank,
            alpha=lora_alpha,
            use_rslora=use_rslora,
        )

        trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.peft_model.parameters())
        logger.info(
            "Trainable params: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100 * trainable / total,
        )

        # Flow scheduler
        self.flow_scheduler = FlowScheduler()

    def _load_quantized(self, quantize: str) -> None:
        """Load pipeline with quantization for large models."""
        import os

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        model_path = os.environ.get(
            "OPEN_VISION_BANANA_MODEL_PATH",
            self.model_config.model_id,
        )
        if not Path(model_path).is_dir():
            model_path = self.model_config.model_id

        if self.family == "flux" and quantize == "4bit":
            self._load_flux_quantized(model_path)
        else:
            # Fallback: load normally
            logger.warning(
                "Quantization '%s' not supported for family '%s', loading unquantized",
                quantize,
                self.family,
            )
            self.components = load_pipeline(
                self.model_config,
                device=self.device,
                dtype=self.dtype,
                enable_gradient_checkpointing=True,
            )

    def _load_flux_quantized(self, model_path: str) -> None:
        """Load FLUX with 4-bit transformer + 8-bit T5 for 24GB VRAM."""
        from diffusers import FluxPipeline, FluxTransformer2DModel
        from transformers import BitsAndBytesConfig, T5EncoderModel

        # 4-bit transformer
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
            quantization_config=quant_config,
        )

        # 8-bit T5
        logger.info("Loading T5 text encoder in 8-bit...")
        t5_quant = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            torch_dtype=self.dtype,
            quantization_config=t5_quant,
        )
        _log_bnb_int8_cast_warning_once()

        # Full pipeline
        pipe = FluxPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=self.dtype,
        )
        pipe = pipe.to(self.device)

        # Remove accelerate hooks (corrupt FLUX-specific kwargs)
        from accelerate.hooks import remove_hook_from_module

        for submod_name in ["transformer", "vae", "text_encoder", "text_encoder_2"]:
            submod = getattr(pipe, submod_name, None)
            if submod is not None:
                remove_hook_from_module(submod, recurse=True)
        logger.info("Removed accelerate hooks from all pipeline components.")

        # Freeze
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)
        pipe.transformer.requires_grad_(False)
        torch.cuda.empty_cache()

        # Enable gradient checkpointing
        if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
            pipe.transformer.enable_gradient_checkpointing()

        # Store the full pipe for encode_prompt
        self._pipe = pipe

        # Build PipelineComponents manually
        self.components = PipelineComponents(
            unet=pipe.transformer,
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            config=self.model_config,
        )

        backbone_params = sum(p.numel() for p in pipe.transformer.parameters())
        logger.info("FLUX backbone params: %.1fM", backbone_params / 1e6)
        logger.info("GPU mem: %.1f GB", torch.cuda.memory_allocated() / 1024**3)

    # ── Text encoding ──────────────────────────────────────────────────────

    def encode_text(
        self, prompts: list[str], device: torch.device, dtype: torch.dtype
    ) -> dict[str, Any]:
        """Encode text prompts into a dict of conditioning tensors.

        Returns dict with keys depending on model family:
          - All: "encoder_hidden_states"
          - SDXL: + "pooled_embeds"
          - FLUX: + "pooled_projections", "text_ids"
          - SANA: + "attention_mask"
          - SD3: + "pooled_projections"
        """
        if self.family == "flux" and self.pipe is not None:
            return self._flux_encode_text(prompts, device, dtype)

        # Use model_registry's unified encode_text
        hidden_states, second = registry_encode_text(self.components, prompts, device, dtype)

        result = {"encoder_hidden_states": hidden_states}

        # Parse the second return value based on family
        if self.model_config.needs_attention_mask and isinstance(second, torch.Tensor):
            result["attention_mask"] = second
        elif second is not None and isinstance(second, torch.Tensor):
            result["pooled_embeds"] = second
        elif second is not None and isinstance(second, tuple):
            # Various tuple returns for complex models
            if self.family == "sd3":
                result["pooled_projections"] = second
            elif self.family == "hunyuandit":
                result["attention_mask"] = second  # (t5_hidden, clip_mask, t5_mask)
            else:
                result["auxiliary"] = second

        return result

    def _flux_encode_text(
        self, prompts: list[str], device: torch.device, dtype: torch.dtype, max_length: int = 512
    ) -> dict:
        """FLUX dual text encoding via pipeline.encode_prompt (CLIP + T5)."""
        pipe = self.pipe
        with torch.no_grad():
            outputs = pipe.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_length,
            )

        prompt_embeds = outputs[0]
        pooled = outputs[1] if len(outputs) > 1 else None
        text_ids = outputs[2] if len(outputs) > 2 else None

        return {
            "encoder_hidden_states": prompt_embeds.to(device=device, dtype=dtype),
            "pooled_projections": pooled.to(device=device, dtype=dtype)
            if pooled is not None
            else None,
            "text_ids": text_ids.to(device=device, dtype=dtype) if text_ids is not None else None,
        }

    # ── VAE encode/decode ──────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """VAE encode images to latents with model-appropriate scaling.

        Args:
            images: [B, 3, H, W] in [-1, 1].

        Returns:
            Latents in model-appropriate scale.
        """
        latents = registry_vae_encode(self.vae, images, self.model_config)
        return latents.to(self.dtype)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE decode latents to images.

        Returns:
            Images in [-1, 1].
        """
        return registry_vae_decode(self.vae, latents, self.model_config)

    # ── Forward pass ───────────────────────────────────────────────────────

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_cond: dict[str, Any],
        batch_size: int,
        guidance: float | None = None,
    ) -> torch.Tensor:
        """Unified forward that handles packing, position IDs, guidance.

        Args:
            latents: Noisy latents [B, C, H, W].
            timesteps: Flow timesteps [B].
            text_cond: Dict from encode_text().
            batch_size: Batch size.
            guidance: Guidance scale (for models that need it).

        Returns:
            Predicted velocity [B, C, H, W].
        """
        latent_h, latent_w = latents.shape[-2], latents.shape[-1]

        # Handle latent packing for FLUX/HiDream
        if FlowScheduler.needs_latent_packing(self.family):
            latents = pack_latents(latents, batch_size)
            packed_h, packed_w = latent_h // 2, latent_w // 2
            img_ids = prepare_latent_image_ids(
                batch_size, packed_h, packed_w, self.device, self.dtype
            )
        else:
            img_ids = None

        # Build forward kwargs
        fwd_kwargs: dict[str, Any] = {
            "latents": latents,
            "timesteps": self.model_timesteps(timesteps, batch_size),
            "encoder_hidden_states": text_cond["encoder_hidden_states"],
            "config": self.model_config,
        }

        # Guidance (FLUX, Qwen-Image, SanaSprint, etc.)
        if FlowScheduler.needs_guidance(self.family):
            if guidance is not None:
                fwd_kwargs["guidance"] = guidance
            elif self.family == "flux":
                fwd_kwargs["guidance"] = 50.0  # FLUX default guidance

        # Attention mask (SANA, PixArt, etc.)
        if "attention_mask" in text_cond:
            fwd_kwargs["attention_mask"] = text_cond["attention_mask"]

        # Pooled projections (FLUX, SD3)
        if "pooled_projections" in text_cond:
            pooled = text_cond["pooled_projections"]
            if pooled is not None:
                fwd_kwargs["pooled_embeds"] = pooled
            elif self.family == "flux":
                fwd_kwargs["pooled_embeds"] = torch.zeros(
                    batch_size, 768, device=self.device, dtype=self.dtype
                )

        # Pooled embeds (SDXL)
        if "pooled_embeds" in text_cond:
            fwd_kwargs["pooled_embeds"] = text_cond["pooled_embeds"]

        # Text IDs (FLUX, Chroma)
        if "text_ids" in text_cond and text_cond["text_ids"] is not None:
            fwd_kwargs["txt_ids"] = text_cond["text_ids"]

        # Image IDs (FLUX, Chroma, etc.)
        if img_ids is not None:
            fwd_kwargs["img_ids"] = img_ids

        # Run forward through peft_model (so LoRA adapters participate)
        predicted = self._peft_forward(latents, timesteps, fwd_kwargs)

        # Unpack output if packed
        if FlowScheduler.needs_latent_packing(self.family) and predicted.dim() == 3:
            predicted = unpack_latents(predicted, latent_h, latent_w)

        return predicted

    def model_timesteps(self, timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Convert flow time in [0, 1] to what the backbone's timestep embedding expects.

        SANA's transformer takes scheduler timesteps: sigma * num_train_timesteps, times
        its ``timestep_scale``. SanaPipeline, scripts/train_sana_512.py and the inference
        sampler all pass that; sigma itself tells the model it is always at t ~ 0, and the
        resulting LoRA does not work with them. FLUX's transformer takes [0, 1] as is.
        The conversion lives here, not in backbone_forward, whose other callers already
        pass scheduler timesteps.
        """
        if self.family != "sana":
            return timesteps
        scheduler_config = getattr(getattr(self.components, "scheduler", None), "config", None)
        num_train_timesteps = float(getattr(scheduler_config, "num_train_timesteps", 1000))
        # float32 first: in bf16 the product is only exact to 4 above 512.
        scheduler_timesteps = timesteps.float() * num_train_timesteps
        return native_sana_timestep(scheduler_timesteps, batch_size, self.peft_model)

    def _peft_forward(self, latents, timesteps, fwd_kwargs) -> torch.Tensor:
        """Run forward through peft_model with proper argument passing."""
        # For FLUX, we need to call peft_model directly with specific kwargs
        if self.family == "flux":
            forward_kwargs = {
                "hidden_states": fwd_kwargs["latents"],
                "timestep": fwd_kwargs["timesteps"],
                "encoder_hidden_states": fwd_kwargs["encoder_hidden_states"],
            }
            if "pooled_embeds" in fwd_kwargs and fwd_kwargs["pooled_embeds"] is not None:
                forward_kwargs["pooled_projections"] = fwd_kwargs["pooled_embeds"]
            if "guidance" in fwd_kwargs:
                g = fwd_kwargs["guidance"]
                if isinstance(g, int | float):
                    forward_kwargs["guidance"] = torch.full(
                        (latents.shape[0],), float(g), device=self.device, dtype=self.dtype
                    )
                else:
                    forward_kwargs["guidance"] = g
            if "txt_ids" in fwd_kwargs:
                forward_kwargs["txt_ids"] = fwd_kwargs["txt_ids"]
            if "img_ids" in fwd_kwargs:
                forward_kwargs["img_ids"] = fwd_kwargs["img_ids"]

            result = self.peft_model(**forward_kwargs)
            if hasattr(result, "sample"):
                result = result.sample
            return result

        # For other models, use model_registry's backbone_forward
        # But we need to call it through peft_model
        # Temporarily swap the backbone reference
        return backbone_forward(
            self.peft_model,
            latents=fwd_kwargs["latents"],
            timesteps=fwd_kwargs["timesteps"],
            encoder_hidden_states=fwd_kwargs["encoder_hidden_states"],
            config=fwd_kwargs["config"],
            added_cond_kwargs=fwd_kwargs.get("added_cond_kwargs"),
            guidance=fwd_kwargs.get("guidance"),
            attention_mask=fwd_kwargs.get("attention_mask"),
            pooled_embeds=fwd_kwargs.get("pooled_embeds"),
            txt_ids=fwd_kwargs.get("txt_ids"),
            img_ids=fwd_kwargs.get("img_ids"),
        )

    # ── Flow batch preparation ─────────────────────────────────────────────

    def prepare_flow_batch(
        self,
        target_latents: torch.Tensor,
        input_latents: torch.Tensor,
        batch_size: int,
        img2img_flow: bool = True,
        flow_sigma: float = 0.0,
        logit_normal: bool = True,
        flow_shift: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Sample timesteps, create noisy latents and velocity targets.

        Returns dict with keys:
            - "timesteps": [B]
            - "noisy_latents": [B, C, H, W]
            - "velocity_target": [B, C, H, W]
            - "noise": [B, C, H, W]
        """
        noise = torch.randn_like(target_latents)

        # Determine source latents
        if img2img_flow and input_latents is not None:
            source_latents = input_latents
            if flow_sigma > 0:
                source_latents = (1 - flow_sigma) * source_latents + flow_sigma * noise
        else:
            source_latents = noise

        # Sample timesteps
        timesteps = self.flow_scheduler.sample_timesteps(
            batch_size,
            self.device,
            self.family,
            logit_normal=logit_normal,
            flow_shift=flow_shift,
        )
        timesteps = timesteps.to(self.dtype)

        # Create noisy latents and velocity target
        noisy_latents, velocity_target = self.flow_scheduler.create_noisy_latents(
            target_latents,
            source_latents,
            noise,
            timesteps,
            self.family,
            flow_sigma=flow_sigma,
        )

        return {
            "timesteps": timesteps,
            "noisy_latents": noisy_latents,
            "velocity_target": velocity_target,
            "noise": noise,
        }

    # ── CFG dropout ────────────────────────────────────────────────────────

    def apply_cfg_dropout(
        self,
        text_cond: dict[str, Any],
        cfg_dropout_rate: float,
        batch_size: int,
    ) -> dict[str, Any]:
        """Apply classifier-free guidance dropout to text conditioning."""
        if cfg_dropout_rate <= 0:
            return text_cond

        drop_mask = torch.rand(batch_size, device=self.device) < cfg_dropout_rate
        if not drop_mask.any():
            return text_cond

        # Re-encode with empty prompts for dropped samples
        empty_prompts = [""] * batch_size
        uncond_cond = self.encode_text(empty_prompts, self.device, self.dtype)

        result = {}
        for key in text_cond:
            val = text_cond[key]
            uncond_val = uncond_cond.get(key)
            if val is None or uncond_val is None:
                result[key] = val
                continue
            if isinstance(val, torch.Tensor) and isinstance(uncond_val, torch.Tensor):
                drop = drop_mask.view(-1, *([1] * (val.dim() - 1)))
                result[key] = torch.where(drop, uncond_val, val)
            else:
                result[key] = val

        return result
