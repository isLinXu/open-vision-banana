"""Unified model backend for multi-model training.

ModelBackend wraps model_registry functions (encode_text, backbone_forward,
vae_encode/vae_decode, apply_lora, load_pipeline) and adds model-specific
logic: quantization, accelerate hook removal, latent packing, guidance, etc.

This eliminates the need for model-specific training scripts — a single
Trainer class can work with any registered model.
"""

from __future__ import annotations

import logging
from copy import deepcopy
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
from . import qwen_image21
from .edit_forward import (
    channel_concat_edit_forward,
    edit_latent_layout,
    flux2_edit_forward,
    hunyuan_edit_forward,
    joy_edit_forward,
    longcat_edit_forward,
    qwen21_edit_forward,
    qwen_edit_forward,
)
from .flow_matching import (
    FlowScheduler,
    pack_latents,
    prepare_latent_image_ids,
    unpack_latents,
)
from .sana_flow import native_sana_timestep

logger = logging.getLogger(__name__)

# Edit layouts whose training forward runs in the pipeline's native sigma frame.
_FLOW_EDIT_FORWARDS = {
    "qwen_packed": qwen_edit_forward,
    "longcat_packed": longcat_edit_forward,
    "flux2_packed": flux2_edit_forward,
    "joy_slots": joy_edit_forward,
    "hunyuan": hunyuan_edit_forward,
}

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


def _tensors_to_pils(images: torch.Tensor):
    """Convert a [-1, 1] image batch into one PIL image per sample."""
    from PIL import Image

    pixels = ((images.detach().float().cpu() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return [Image.fromarray(pixels[index].permute(1, 2, 0).numpy()) for index in range(pixels.shape[0])]


def reject_unimplemented_instruction_edit(model_config) -> None:
    """Refuse an instruction editor that has no training forward."""
    if getattr(model_config, "instruction_editing", False) and edit_latent_layout(model_config) is None:
        raise ValueError(
            f"{model_config.name} is an instruction-editing model without a training forward. "
            f"Inference uses {model_config.pipeline_class}."
        )


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

        # Registry configs are shared; keep backend overrides and mutable fields local.
        self.model_config = deepcopy(resolve_model_config(model_name))
        if lora_targets is not None:
            self.model_config.lora_target_modules = list(lora_targets)
        self.family = self.model_config.family
        reject_unimplemented_instruction_edit(self.model_config)
        if edit_latent_layout(self.model_config) == "hunyuan":
            self._load_hunyuan_instruct()
            return

        logger.info(
            "Loading model '%s' (family=%s, backbone=%s) on %s @ %s",
            model_name,
            self.family,
            self.model_config.backbone_type,
            device,
            dtype,
        )

        # Load pipeline components
        if quantize and edit_latent_layout(self.model_config) == "qwen21":
            raise ValueError("Quantized loading is not implemented for Qwen-Image-2.1")
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
        layout = edit_latent_layout(self.model_config)
        if layout == "qwen21":
            # Its prompt encoder takes the source image; built once around the loaded parts.
            self.pipe = qwen_image21.build_pipeline(self.components)
            self._qwen21_logged = (
                torch.distributed.is_initialized() and torch.distributed.get_rank() > 0
            )
        elif layout == "qwen_packed" and self.pipe is None:
            self.pipe = qwen_image21.build_qwen_edit_pipeline(
                self.components, self.model_config.pipeline_class
            )

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
        logger.info(
            "Trainable parameter names: %s",
            [name for name, parameter in self.peft_model.named_parameters() if parameter.requires_grad],
        )

        # Flow scheduler
        self.flow_scheduler = FlowScheduler()

    def _load_hunyuan_instruct(self) -> None:
        """Load HunyuanImage-3.0-Instruct for its published ``forward``.

        The checkpoint is an 80B mixture-of-experts causal model. This path does
        not wrap it in a diffusion LoRA; the training forward is
        ``hunyuan_edit_forward``.
        """
        from ..inference.generator import load_hunyuan_image3_instruct
        from ..model_registry import resolve_pretrained_source

        source = resolve_pretrained_source(self.model_config, self.model_name)
        model = load_hunyuan_image3_instruct(source, self.dtype)
        self.peft_model = model
        self.backbone = model
        self.vae = getattr(model, "vae", None)
        self.components = None
        self.pipe = None
        self.flow_scheduler = FlowScheduler()
        logger.info("Loaded HunyuanImage-3.0-Instruct for the edit training forward")

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
        import diffusers
        from diffusers import FluxTransformer2DModel
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

        # Kontext's model_index names FluxKontextPipeline. Loading it as
        # FluxPipeline rejects the repo. Training still uses ModelBackend.forward
        # with context latents; the pipeline class only has to assemble weights.
        pipeline_name = getattr(self.model_config, "pipeline_class", "FluxPipeline")
        pipeline_cls = getattr(diffusers, pipeline_name, None)
        if pipeline_cls is None:
            raise ImportError(f"diffusers has no pipeline class {pipeline_name}")
        pipe = pipeline_cls.from_pretrained(
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
        if edit_latent_layout(self.model_config) == "hunyuan":
            raise ValueError(
                "HunyuanImage-3.0-Instruct text encoding needs the source image. "
                "Use encode_hunyuan_edit(prompts, images)."
            )

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

    def encode_hunyuan_edit(self, prompts: list[str], images: torch.Tensor) -> dict[str, Any]:
        """Build the token batch HunyuanImage-3.0-Instruct.forward expects.

        ``images`` is [B, 3, H, W] in [-1, 1]. Each sample is one reference image.
        """
        model = self.peft_model
        if not hasattr(model, "prepare_model_inputs"):
            raise RuntimeError(
                "HunyuanImage-3.0-Instruct is not loaded; its prepare_model_inputs is missing"
            )
        prepared = model.prepare_model_inputs(
            prompt=list(prompts),
            image=[[pil] for pil in _tensors_to_pils(images)],
            mode="gen_image",
            bot_task="image",
            image_size="auto",
            device=self.device,
        )
        token_ids = prepared["input_ids"]
        return {
            "encoder_hidden_states": token_ids,
            "hunyuan_inputs": {
                "input_ids": token_ids,
                "attention_mask": prepared.get("attention_mask"),
                "rope_image_info": prepared.get("rope_image_info"),
                "image_mask": prepared["image_mask"],
                "timesteps_index": prepared["timesteps_index"],
                "cond_vae_image_mask": prepared["cond_vae_image_mask"],
                "cond_timesteps_index": prepared["cond_timesteps_index"],
            },
        }

    def encode_edit_text(
        self, prompts: list[str], images: torch.Tensor, cfg_dropout_rate: float = 0.0
    ) -> dict[str, Any]:
        """Text condition for a training forward; editors that read the source get it here.

        ``images`` is the [B, 3, H, W] source batch in [-1, 1]. Qwen-Image-2.1 and
        Qwen-Image-Edit drop the prompt to "" with the same source image, as their
        pipelines' unconditional branches do; text-only conditions drop through
        ``apply_cfg_dropout``.
        """
        layout = edit_latent_layout(self.model_config)
        if layout == "hunyuan":
            return self.encode_hunyuan_edit(prompts, images)
        if layout == "qwen21":
            if cfg_dropout_rate > 0:
                drop = torch.rand(len(prompts), device=self.device) < cfg_dropout_rate
                prompts = ["" if d else p for p, d in zip(prompts, drop.tolist(), strict=True)]
            text_cond = qwen_image21.encode_condition(
                self.pipe, prompts, _tensors_to_pils(images), self.device, self.dtype
            )
            if not getattr(self, "_qwen21_logged", True):
                self._qwen21_logged = True
                logger.info(
                    "Qwen-Image-2.1 conditioning: source image %s -> %d Qwen3-VL visual tokens "
                    "in a %d-token condition sequence",
                    tuple(images.shape[-2:]),
                    int(text_cond["image_pad_mask"][0].sum()),
                    text_cond["encoder_hidden_states"].shape[1],
                )
            return text_cond
        if layout == "qwen_packed" and getattr(self, "pipe", None) is not None and hasattr(
            self.pipe, "_get_qwen_prompt_embeds"
        ):
            if cfg_dropout_rate > 0:
                drop = torch.rand(len(prompts), device=self.device) < cfg_dropout_rate
                prompts = ["" if d else p for p, d in zip(prompts, drop.tolist(), strict=True)]
            return qwen_image21.encode_qwen_edit_condition(
                self.pipe, prompts, _tensors_to_pils(images), self.device, self.dtype
            )
        text_cond = self.encode_text(prompts, self.device, self.dtype)
        return self.apply_cfg_dropout(text_cond, cfg_dropout_rate, len(prompts))

    # ── VAE encode/decode ──────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """VAE encode images to latents with model-appropriate scaling.

        Args:
            images: [B, 3, H, W] in [-1, 1].

        Returns:
            Latents in model-appropriate scale.
        """
        if edit_latent_layout(self.model_config) == "hunyuan" and hasattr(self.peft_model, "vae_encode"):
            _timestep, latents = self.peft_model.vae_encode(images)
            return latents.to(self.dtype)
        latents = registry_vae_encode(self.vae, images, self.model_config)
        return latents.to(self.dtype)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """VAE decode latents to images.

        Keeps the graph: the normal angular aux loss backpropagates through the (frozen)
        VAE to the latents. Callers that only want images should run under no_grad.

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
        context_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unified forward that handles packing, position IDs, guidance.

        Args:
            latents: Noisy latents [B, C, H, W].
            timesteps: Flow timesteps [B].
            text_cond: Dict from encode_text().
            batch_size: Batch size.
            guidance: Guidance scale (for models that need it).
            context_latents: Clean source-image latents [B, C, H, W] for
                in-context (Kontext-style) conditioning. Only valid when
                ``model_config.in_context_conditioning`` is set. The tokens are
                appended after the target tokens with position-id axis 0 = 1
                (target tokens use 0), and only the target tokens' prediction
                is returned.

        Returns:
            Predicted velocity [B, C, H, W] in this repo's flow frame.
        """
        reject_unimplemented_instruction_edit(self.model_config)
        layout = edit_latent_layout(self.model_config)
        if layout not in (None, "kontext"):
            if context_latents is None:
                raise ValueError(
                    f"{self.model_name} training forward needs the source image as context_latents"
                )
            return self._forward_edit(
                latents, timesteps, text_cond, guidance, context_latents, layout
            )
        latent_h, latent_w = latents.shape[-2], latents.shape[-1]
        in_context = context_latents is not None
        if in_context and not (self.model_config.in_context_conditioning and self.family == "flux"):
            raise ValueError(
                f"context_latents given but model '{self.model_name}' does not use "
                "in-context conditioning (needs a Kontext-style flux model)"
            )
        n_target_tokens = (latent_h // 2) * (latent_w // 2)

        # Handle latent packing for FLUX/HiDream
        if FlowScheduler.needs_latent_packing(self.family):
            latents = pack_latents(latents, batch_size)
            packed_h, packed_w = latent_h // 2, latent_w // 2
            img_ids = prepare_latent_image_ids(
                batch_size, packed_h, packed_w, self.device, self.dtype
            )
            if in_context:
                context_tokens = pack_latents(context_latents.to(latents.dtype), batch_size)
                context_ids = prepare_latent_image_ids(
                    batch_size,
                    context_latents.shape[-2] // 2,
                    context_latents.shape[-1] // 2,
                    self.device,
                    self.dtype,
                )
                context_ids[:, 0] = 1  # axis 0 marks "this token belongs to the conditioning image"
                latents = torch.cat([latents, context_tokens], dim=1)
                img_ids = torch.cat([img_ids, context_ids], dim=0)
        else:
            img_ids = None

        # Build forward kwargs
        fwd_kwargs: dict[str, Any] = {
            "latents": latents,
            # FLUX's native timestep is sigma (1 = pure noise). This repo's flow
            # time t runs 0 (source=noise) -> 1 (target), so in-context mode feeds
            # 1 - t and flips the output sign below to stay in the repo's frame.
            "timesteps": self.model_timesteps(1.0 - timesteps if in_context else timesteps, batch_size),
            "encoder_hidden_states": text_cond["encoder_hidden_states"],
            "config": self.model_config,
        }

        # Guidance (FLUX, Qwen-Image, SanaSprint, etc.)
        if FlowScheduler.needs_guidance(self.family):
            if guidance is not None:
                fwd_kwargs["guidance"] = guidance
            elif in_context:
                fwd_kwargs["guidance"] = float(self.model_config.default_guidance_scale)
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

        if in_context:
            # Drop the context tokens' outputs; native v = noise - target is the
            # negative of this repo's v = target - source (source = noise).
            predicted = -predicted[:, :n_target_tokens]

        # Unpack output if packed
        if FlowScheduler.needs_latent_packing(self.family) and predicted.dim() == 3:
            predicted = unpack_latents(predicted, latent_h, latent_w)

        return predicted

    def _forward_edit(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_cond: dict[str, Any],
        guidance: float | None,
        context_latents: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        context_latents = context_latents.to(dtype=latents.dtype)
        if layout == "qwen21":
            return qwen21_edit_forward(
                self.peft_model, latents, context_latents, timesteps, text_cond
            )
        if layout == "channel_concat":
            return channel_concat_edit_forward(
                self.peft_model, latents, context_latents, timesteps, text_cond, self.model_config
            )
        forward = _FLOW_EDIT_FORWARDS.get(layout)
        if forward is None:
            raise ValueError(f"No training forward for edit layout {layout}")
        # Flow editors' pipelines noise as sigma * noise + (1 - sigma) * x0 and integrate a
        # transformer output of noise - x0. The repo's t runs from 0 (noise) to 1 (target)
        # with velocity x0 - noise: feed sigma = 1 - t and negate, as Kontext does.
        extra = (guidance,) if layout == "qwen_packed" else ()
        sigma = 1.0 - timesteps
        return -forward(self.peft_model, latents, context_latents, sigma, text_cond, *extra)

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
        img2img_flow: bool | torch.Tensor = True,
        flow_sigma: float = 0.0,
        logit_normal: bool = True,
        flow_shift: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Sample timesteps, create noisy latents and velocity targets.

        ``img2img_flow`` may be a bool for the whole batch or a ``[B]`` bool tensor
        selecting *per sample* whether the flow starts from the (optionally
        noise-mixed) input latents or from pure Gaussian noise. Mixed batches of
        vision rows (img2img) and generation-replay rows (noise -> image) need the
        per-sample form; a batch-wide flag would give the replay rows a zero
        velocity target because their input and target are the same image.

        Returns dict with keys:
            - "timesteps": [B]
            - "noisy_latents": [B, C, H, W]
            - "velocity_target": [B, C, H, W]
            - "noise": [B, C, H, W]
            - "img2img_mask": [B] bool, which samples started from the input
        """
        noise = torch.randn_like(target_latents)

        # Determine source latents (per sample)
        if isinstance(img2img_flow, torch.Tensor):
            mask = img2img_flow.to(device=target_latents.device, dtype=torch.bool).reshape(-1)
            if mask.numel() != target_latents.shape[0]:
                raise ValueError(f"img2img_flow mask has {mask.numel()} entries for batch {target_latents.shape[0]}")
        else:
            mask = torch.full((target_latents.shape[0],), bool(img2img_flow), device=target_latents.device)
        if input_latents is None:
            mask = torch.zeros_like(mask)
        if mask.any():
            img_source = input_latents
            if flow_sigma > 0:
                img_source = (1 - flow_sigma) * img_source + flow_sigma * noise
            view = mask.view(-1, *([1] * (target_latents.dim() - 1)))
            source_latents = torch.where(view, img_source.to(noise.dtype), noise)
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
            "img2img_mask": mask,
        }

    # ── CFG dropout ────────────────────────────────────────────────────────

    def apply_cfg_dropout(
        self,
        text_cond: dict[str, Any],
        cfg_dropout_rate: float,
        batch_size: int,
    ) -> dict[str, Any]:
        """Apply classifier-free guidance dropout to text conditioning."""
        if cfg_dropout_rate <= 0 or "hunyuan_inputs" in text_cond:
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
