"""FLUX.1 base generation wrapper with LoRA instruction alignment.

This module provides the base generation layer using FLUX.1 [schnell]
with parameter-efficient fine-tuning via LoRA (rsLoRA scaling).

For vision-understanding tasks, Vision Banana operates in image-to-image
mode: the input image is encoded via the VAE, partially noised, and then
denoised conditioned on a task prompt.  This is implemented with
FluxImg2ImgPipeline when an input image is provided, and the standard
FluxPipeline for unconditional/text-only generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ..conditioning import (
    CONV_IN_STATE_NAME,
    SINGLE_STEP_X0,
    SourceConditioningConfig,
    calibrate_source_conditioned_output,
    expand_unet_conv_in,
    read_source_conditioning_config,
    set_scheduler_prediction_type,
    sigma_for_timesteps,
    validate_scheduler_prediction_type,
)
from ..model_registry import ModelConfig, resolve_model_config, resolve_pretrained_source
from ..spatial_adapter import (
    SPATIAL_ADAPTER_STATE_NAME,
    create_source_spatial_adapter,
)
from .checkpoint import peft_adapter_state_to_diffusers, resolve_lora_path
from .optimization import InferenceOptions, apply_pipeline_optimizations

logger = logging.getLogger(__name__)

DIFFUSERS_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def _prompt_cache_key(prompt: str | list[str]) -> str:
    """Build a collision-resistant cache key for prompt embedding reuse."""
    if isinstance(prompt, str):
        return prompt if len(prompt) <= 256 else hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return hashlib.sha256(
        json.dumps(prompt, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


# ──────────────────────────────────────────────────────────────
# Bounded LRU cache for prompt embeddings
# ──────────────────────────────────────────────────────────────


class _PromptEmbedCache(OrderedDict):
    """Bounded LRU cache for prompt encoder outputs.

    Evicts least-recently-used entries when capacity is exceeded,
    preventing unbounded VRAM growth during long-running sessions.
    """

    def __init__(self, maxsize: int = 64) -> None:
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value) -> None:
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            evicted_key, _ = self.popitem(last=False)
            logger.debug("Prompt cache evicted '%.60s…' (LRU)", evicted_key)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] tensor in [-1, 1] or [0, 1] to PIL Image."""
    t = t.float().cpu()
    if t.min() < 0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def _restore_peft_lora_scaling(backbone: torch.nn.Module, lora_dir: Path) -> None:
    """Set injected LoRA layers to the adapter's trained scale.

    A converted PEFT state dict carries no ``alpha`` tensors, so
    ``load_lora_weights`` injects every layer with ``lora_alpha = r`` (scale 1.0)
    and drops ``use_rslora``. Training used ``alpha / r`` or ``alpha / sqrt(r)``.
    """
    config_path = lora_dir / "adapter_config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text())
    rank = int(config.get("r", 0))
    alpha = float(config.get("lora_alpha", rank))
    if rank <= 0:
        return
    scale = alpha / rank ** 0.5 if config.get("use_rslora") else alpha / rank
    layers = 0
    for module in backbone.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and hasattr(module, "lora_A"):
            for name in scaling:
                scaling[name] = scale
            layers += 1
    logger.info(
        "LoRA scale set to %.4f (r=%d, alpha=%g, rslora=%s) on %d layers",
        scale, rank, alpha, bool(config.get("use_rslora")), layers,
    )


def _pil_to_tensor(img: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Convert PIL Image to [1, C, H, W] tensor in [0, 1]."""
    import numpy as np

    arr = (
        torch.from_numpy((np.array(img.convert("RGB")) / 255.0).astype("float32"))
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    return arr.to(device=device, dtype=dtype)


def resolve_hunyuan_image3_source(model_source: str) -> str:
    """Return a local HunyuanImage-3.0-Instruct directory whose name has no '.'.

    The official loader cannot import remote code from a directory name that
    contains a dot, including the Hub id ``tencent/HunyuanImage-3.0-Instruct``.
    A local directory is used as given. A Hub id is snapshotted under a
    dot-free cache folder.
    """
    import os

    path = Path(model_source)
    if path.is_dir():
        if "." in path.name:
            raise ValueError(
                f"HunyuanImage-3.0-Instruct directory name {path.name!r} contains '.'. "
                "Rename it, for example HunyuanImage-3-Instruct."
            )
        return str(path)
    safe = model_source.replace("/", "--").replace(".", "-")
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    local_dir = cache_root / "open-vision-banana" / safe
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_source, local_dir=str(local_dir))
    return str(local_dir)


def load_hunyuan_image3_instruct(model_source: str, dtype: torch.dtype):
    """Load HunyuanImage-3.0-Instruct the way its model card specifies."""
    source = resolve_hunyuan_image3_source(model_source)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        source,
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        moe_impl="eager",
        moe_drop_tokens=True,
    )
    model.load_tokenizer(source)
    return model


class BaseGenerator:
    """FLUX.1 base generation wrapper with LoRA support.

    Loads FLUX.1 [schnell] and optionally applies LoRA for instruction
    alignment. The model supports:
    - Text-to-image generation (FluxPipeline)
    - Image-conditioned generation (FluxImg2ImgPipeline) when ``image`` is provided.

    For vision-understanding tasks (depth, normals, segmentation), image
    conditioning is essential — the model generates a *visualization* of the
    input scene, so it must see the scene.

    Memory notes:
    - 12B params require ~24 GB VRAM in bfloat16.
    - Pass ``enable_sequential_cpu_offload=True`` to fit on 12–16 GB GPUs
      at the cost of ~4× slower inference.
    """

    DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lora_path: str | None = None,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        enable_sequential_cpu_offload: bool = False,
        lora_revision: str | None = None,
        inference_options: InferenceOptions | None = None,
    ) -> None:
        self.model_name = model_name
        self.lora_path = lora_path
        self.lora_revision = lora_revision
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.inference_options = inference_options or InferenceOptions(
            enable_sequential_cpu_offload=enable_sequential_cpu_offload,
        )
        # When inference_options is auto-created, sync its dtype to the
        # constructor dtype so apply_pipeline_optimizations doesn't silently
        # convert the pipeline to an unexpected dtype.
        if inference_options is None:
            _dtype_map = {torch.float32: "float32", torch.float16: "float16", torch.bfloat16: "bfloat16"}
            if self.dtype in _dtype_map:
                self.inference_options.dtype = _dtype_map[self.dtype]
        self.enable_sequential_cpu_offload = self.inference_options.enable_sequential_cpu_offload

        # Both pipelines are loaded lazily to save memory when only one is needed.
        self._t2i_pipe: Any = None  # FluxPipeline (text → image)
        self._i2i_pipe: Any = None  # FluxImg2ImgPipeline (image + text → image)

        # KV-cache for text encoder outputs: avoids redundant CLIP/T5
        # forward passes when the same prompt is used for repeated generation.
        # Bounded LRU to prevent unbounded VRAM growth.
        self._prompt_cache: _PromptEmbedCache = _PromptEmbedCache(maxsize=64)

        self._resolved_lora_path: str | None = None
        self.source_conditioning_config: SourceConditioningConfig | None = None
        self.source_spatial_adapter: torch.nn.Module | None = None
        self._load_model()

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to the best available device."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load text-to-image pipeline and optionally apply LoRA.

        Auto-detects the model family from the model_name path/ID using
        the model registry, and loads the appropriate diffusers pipeline.
        Supports FLUX, Z-Image, Qwen-Image, Hunyuan-Image, SDXL, and all
        23+ registered model families.
        """
        # Resolve the model config from the registry
        self.model_config: ModelConfig = resolve_model_config(self.model_name)
        family = self.model_config.family
        pipe_class_name = self.model_config.pipeline_class

        logger.info(
            f"Loading {pipe_class_name} ({family}) from {self.model_name!r} …"
        )

        if pipe_class_name == "HunyuanImage3Instruct":
            if self.lora_path is not None:
                raise ValueError(
                    "HunyuanImage-3.0-Instruct loads the official checkpoint as-is. "
                    "A LoRA path is not applied."
                )
            model_source = resolve_pretrained_source(self.model_config, self.model_name)
            self._t2i_pipe = load_hunyuan_image3_instruct(model_source, self.dtype)
            logger.info(f"{pipe_class_name} loaded successfully (family={family})")
            return

        # Lazy-import the correct pipeline class
        import importlib
        try:
            mod = importlib.import_module("diffusers")
            PipeClass = getattr(mod, pipe_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Pipeline class '{pipe_class_name}' not found in diffusers. "
                f"This may require a newer diffusers version. Error: {e}"
            ) from e

        model_source = resolve_pretrained_source(self.model_config, self.model_name)
        pipe_kwargs: dict[str, Any] = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": False,
        }
        if Path(model_source).is_dir():
            pipe_kwargs["local_files_only"] = True
            fp16_transformer = Path(model_source) / "transformer" / "diffusion_pytorch_model.fp16.safetensors"
            default_transformer = Path(model_source) / "transformer" / "diffusion_pytorch_model.safetensors"
            if fp16_transformer.is_file() and not default_transformer.is_file():
                pipe_kwargs["variant"] = "fp16"
        if getattr(self.inference_options, "quantize", None) == "4bit" and family == "flux":
            pipe = self._load_flux_quantized_pipeline(model_source, PipeClass)
        else:
            pipe = PipeClass.from_pretrained(model_source, **pipe_kwargs)
        if family == "sdxl" and hasattr(pipe, "scheduler"):
            # SSD-1B/SDXL LoRA training uses epsilon prediction. Keep the base
            # pipeline on that contract before any checkpoint-specific setup.
            set_scheduler_prediction_type(pipe.scheduler, "epsilon")
            validate_scheduler_prediction_type(pipe.scheduler, "epsilon")

        if self.lora_path is not None:
            self._resolved_lora_path = resolve_lora_path(
                self.lora_path,
                revision=self.lora_revision,
            )
            logger.info(f"Loading LoRA weights from {self._resolved_lora_path!r} …")
            checkpoint_dir = Path(self._resolved_lora_path)
            lora_dir = checkpoint_dir
            if not (lora_dir / "adapter_model.safetensors").is_file():
                # CheckpointManager uses ``lora_adapter`` while older
                # diffusers/PEFT exports commonly use ``adapter``.  Both are
                # part of the supported checkpoint contract.
                for nested_name in ("lora_adapter", "adapter"):
                    nested_lora_dir = checkpoint_dir / nested_name
                    if (nested_lora_dir / "adapter_model.safetensors").is_file():
                        lora_dir = nested_lora_dir
                        logger.info("Using nested PEFT adapter directory %s", lora_dir)
                        break
            try:
                lora_payload, load_kwargs = self._prepare_lora_load(lora_dir, pipe)
                pipe.load_lora_weights(lora_payload, **load_kwargs)
                # Verify LoRA actually attached (diffusers can silently load 0 layers)
                backbone = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
                lora_count = sum(
                    1 for _, m in backbone.named_modules() if hasattr(m, "lora_A")
                ) if backbone is not None else 0
                if lora_count == 0:
                    raise RuntimeError(
                        "pipe.load_lora_weights returned 0 LoRA layers "
                        "(key prefix mismatch)"
                    )
                logger.info("LoRA weights loaded via pipe.load_lora_weights (%d layers)", lora_count)
                if isinstance(lora_payload, dict):
                    _restore_peft_lora_scaling(backbone, lora_dir)
            except Exception as e:
                logger.warning(
                    "pipe.load_lora_weights failed (%s); "
                    "falling back to manual LoRA weight merge.", e
                )
                merged_count = self._load_and_merge_lora(
                    pipe,
                    str(lora_dir),
                )
                if merged_count == 0:
                    raise RuntimeError(
                        "LoRA fallback did not merge any layers; refusing unadapted inference"
                    ) from e

            self._load_source_conditioning(pipe, str(checkpoint_dir))

        applied = apply_pipeline_optimizations(pipe, self.inference_options)
        if applied:
            logger.info("Applied inference optimizations: %s", ", ".join(applied))
        if (
            not self.inference_options.enable_sequential_cpu_offload
            and getattr(self.inference_options, "quantize", None) != "4bit"
        ):
            pipe = pipe.to(self.device)

        self._t2i_pipe = pipe
        logger.info(f"{pipe_class_name} loaded successfully (family={family})")

    def _load_flux_quantized_pipeline(self, model_source: str, pipeline_cls):
        """Load FLUX.1-dev / Kontext with a 4-bit transformer and 8-bit T5."""
        from diffusers import FluxTransformer2DModel
        from transformers import BitsAndBytesConfig, T5EncoderModel

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            model_source,
            subfolder="transformer",
            torch_dtype=self.dtype,
            quantization_config=quant_config,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_source,
            subfolder="text_encoder_2",
            torch_dtype=self.dtype,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        kwargs: dict[str, Any] = {
            "transformer": transformer,
            "text_encoder_2": text_encoder_2,
            "torch_dtype": self.dtype,
        }
        if Path(model_source).is_dir():
            kwargs["local_files_only"] = True
        pipe = pipeline_cls.from_pretrained(model_source, **kwargs)
        return pipe.to(self.device)

    def _load_source_conditioning(self, pipe, checkpoint_dir: str) -> None:
        """Expand and restore UNet input conditioning declared by a checkpoint."""
        config = read_source_conditioning_config(checkpoint_dir)
        if config is None:
            return
        if self.model_config.family != "sdxl":
            raise ValueError(
                "Source-conditioned checkpoints currently require an SDXL/SSD base model"
            )
        unet = getattr(pipe, "unet", None)
        if unet is None:
            raise AttributeError("Source-conditioned checkpoint requires pipeline.unet")
        conv = expand_unet_conv_in(
            unet,
            condition_channels=config.condition_channels,
        )
        if conv.in_channels != config.in_channels:
            raise ValueError(
                f"Expanded UNet has {conv.in_channels} channels, checkpoint expects "
                f"{config.in_channels}"
            )
        state_path = Path(checkpoint_dir) / CONV_IN_STATE_NAME
        if not state_path.is_file():
            raise FileNotFoundError(
                f"Source-conditioned checkpoint is missing {CONV_IN_STATE_NAME}: "
                f"{checkpoint_dir}"
            )
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        conv.load_state_dict(state, strict=True)
        if config.spatial_adapter_type != "none":
            adapter_path = Path(checkpoint_dir) / SPATIAL_ADAPTER_STATE_NAME
            if not adapter_path.is_file():
                raise FileNotFoundError(
                    "Source-conditioned checkpoint declares a spatial adapter "
                    f"but is missing {SPATIAL_ADAPTER_STATE_NAME}: {checkpoint_dir}"
                )
            adapter = create_source_spatial_adapter(
                unet,
                downscale_factor=config.spatial_adapter_downscale_factor,
            )
            adapter.load_state_dict(
                torch.load(adapter_path, map_location="cpu", weights_only=True),
                strict=True,
            )
            adapter.eval()
            self.source_spatial_adapter = adapter
        set_scheduler_prediction_type(pipe.scheduler, config.prediction_type)
        validate_scheduler_prediction_type(pipe.scheduler, config.prediction_type)
        self.source_conditioning_config = config
        logger.info(
            "Loaded source-conditioned UNet input (%d channels) from %s",
            config.in_channels,
            checkpoint_dir,
        )

    def _get_i2i_pipe(self):
        """Lazily build the img2img pipeline by sharing weights with t2i.

        Maps each model family to its corresponding img2img pipeline class.
        Falls back to reusing the t2i pipeline directly if no img2img variant
        exists (some pipelines support image input via __call__).
        """
        if self._i2i_pipe is not None:
            return self._i2i_pipe

        family = getattr(self, "model_config", None)
        family = family.family if family is not None else "flux"

        # Map family → (img2img_class_name, component_keys)
        _i2i_map = {
            "flux": ("FluxImg2ImgPipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer",
                "text_encoder_2", "tokenizer_2", "transformer",
            ]),
            "z-image": ("ZImagePipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer", "transformer",
            ]),
            "qwen-image": ("QwenImagePipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer", "transformer",
            ]),
            "hunyuan-image": ("HunyuanImagePipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer",
                "text_encoder_2", "tokenizer_2", "transformer",
            ]),
            "sd15": ("StableDiffusionImg2ImgPipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer", "unet",
            ]),
            "sd2": ("StableDiffusionImg2ImgPipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer", "unet",
            ]),
            "sdxl": ("StableDiffusionXLImg2ImgPipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer",
                "text_encoder_2", "tokenizer_2", "unet",
            ]),
            "sana": ("SanaImg2ImgPipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer", "transformer",
            ]),
            "aura-flow": ("AuraFlowPipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer", "transformer",
            ]),
            "sd3": ("StableDiffusion3Img2ImgPipeline", [
                "scheduler", "vae", "text_encoder", "tokenizer",
                "text_encoder_2", "tokenizer_2", "text_encoder_3", "tokenizer_3",
                "transformer",
            ]),
        }

        cls_name, comp_keys = _i2i_map.get(family, (None, None))

        t2i = self._t2i_pipe

        if cls_name is not None:
            import importlib
            try:
                mod = importlib.import_module("diffusers")
                I2IClass = getattr(mod, cls_name)
                components = {k: getattr(t2i, k) for k in comp_keys if hasattr(t2i, k)}
                logger.info(f"Building {cls_name} from shared components …")
                self._i2i_pipe = I2IClass(**components)
            except (ImportError, AttributeError):
                logger.warning(
                    f"{cls_name} not available in diffusers; "
                    "falling back to t2i pipeline for img2img."
                )
                self._i2i_pipe = t2i
        else:
            # No img2img variant — reuse t2i (some pipelines accept image input)
            logger.info(
                f"No img2img pipeline mapping for family '{family}'; "
                "using t2i pipeline directly."
            )
            self._i2i_pipe = t2i

        if self._i2i_pipe is not t2i:
            applied = apply_pipeline_optimizations(self._i2i_pipe, self.inference_options)
            if applied:
                logger.info("Applied inference optimizations to img2img pipeline: %s", ", ".join(applied))
            if not self.inference_options.enable_sequential_cpu_offload:
                self._i2i_pipe = self._i2i_pipe.to(self.device)

        logger.info("Img2img pipeline ready")
        return self._i2i_pipe

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor | None]:
        """Batch-encode text prompts via the pipeline's text encoders.

        Delegates to ``encode_prompt`` on the pipeline when available (FLUX,
        SDXL, etc.). For models without ``encode_prompt``, falls back to
        the registry's ``encode_text`` helper.

        Returns:
            Dict with ``prompt_embeds`` [B, seq, D] and ``pooled_prompt_embeds``.
        """
        if not prompts:
            raise ValueError("prompts must be a non-empty list")

        if len(prompts) == 1:
            key = _prompt_cache_key(prompts[0])
            cached = self._prompt_cache.get(key)
            if cached is not None:
                self._prompt_cache.move_to_end(key)
                return {
                    "prompt_embeds": cached["prompt_embeds"],
                    "pooled_prompt_embeds": cached.get("pooled_prompt_embeds"),
                }

        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        pipe = self._t2i_pipe

        # Use the pipeline's encode_prompt if available (FLUX, SDXL, SD3, etc.)
        if hasattr(pipe, "encode_prompt"):
            family = getattr(self.model_config, "family", "flux")

            # FLUX uses dual-encoder encode_prompt with prompt_2 and max_sequence_length
            if family == "flux":
                text_outputs = pipe.encode_prompt(
                    prompt=prompts,
                    prompt_2=prompts,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                )
            elif family == "hunyuan-image":
                # HunyuanImage also uses dual-encoder but may have different max_length
                max_len = getattr(self.model_config, "text_encoder_max_length", 256)
                text_outputs = pipe.encode_prompt(
                    prompt=prompts,
                    prompt_2=prompts,
                    device=device,
                    num_images_per_prompt=1,
                    max_sequence_length=max_len,
                )
            else:
                # Generic fallback — let the pipeline handle its own encode_prompt
                try:
                    text_outputs = pipe.encode_prompt(
                        prompt=prompts,
                        device=device,
                        num_images_per_prompt=1,
                    )
                except TypeError:
                    # Some pipelines need prompt_2 for dual-encoder
                    text_outputs = pipe.encode_prompt(
                        prompt=prompts,
                        prompt_2=prompts,
                        device=device,
                        num_images_per_prompt=1,
                    )

            if isinstance(text_outputs, (tuple, list)) and len(text_outputs) >= 3:
                prompt_embeds = text_outputs[0]
                pooled_prompt_embeds = text_outputs[2]
            else:
                prompt_embeds = text_outputs[0]
                pooled_prompt_embeds = text_outputs[1] if len(text_outputs) > 1 else None
        else:
            # Fallback: use registry's encode_text for models without encode_prompt
            from ..model_registry import PipelineComponents, encode_text

            components = PipelineComponents(
                unet=getattr(pipe, "transformer", getattr(pipe, "unet", None)),
                vae=pipe.vae,
                text_encoder=getattr(pipe, "text_encoder", None),
                text_encoder_2=getattr(pipe, "text_encoder_2", None),
                text_encoder_3=getattr(pipe, "text_encoder_3", None),
                tokenizer=getattr(pipe, "tokenizer", None),
                tokenizer_2=getattr(pipe, "tokenizer_2", None),
                tokenizer_3=getattr(pipe, "tokenizer_3", None),
                scheduler=getattr(pipe, "scheduler", None),
                config=self.model_config,
            )
            prompt_embeds, pooled_or_mask = encode_text(
                components, prompts, device, self.dtype
            )
            pooled_prompt_embeds = pooled_or_mask

        prompt_embeds = prompt_embeds.to(device)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)

        for i, prompt in enumerate(prompts):
            key = _prompt_cache_key(prompt)
            self._prompt_cache[key] = {
                "prompt_embeds": prompt_embeds[i : i + 1],
                "pooled_prompt_embeds": (
                    pooled_prompt_embeds[i : i + 1]
                    if pooled_prompt_embeds is not None
                    else None
                ),
            }

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @staticmethod
    def _guidance_kwargs(pipe, guidance_scale: float) -> dict:
        """Guidance arguments for ``pipe``, which may not declare ``guidance_scale``.

        Qwen-Image-2.1's pipeline applies classifier-free guidance only through
        ``true_cfg_scale`` against a negative prompt (blank, as in the editing path).
        """
        import inspect

        parameters = inspect.signature(pipe.__call__).parameters
        if "guidance_scale" in parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        ):
            return {"guidance_scale": guidance_scale}
        if "true_cfg_scale" in parameters:
            call = {"true_cfg_scale": guidance_scale}
            if "negative_prompt" in parameters:
                call["negative_prompt"] = " "
            return call
        return {}

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        image: torch.Tensor | Image.Image | None = None,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        strength: float = 0.85,
        height: int = 1024,
        width: int = 1024,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate images from a prompt, optionally conditioned on an input image.

        When ``image`` is provided the img2img pipeline is used, denoising
        from a noisy version of the encoded input.  This is the primary mode
        for all vision-understanding tasks.

        KV-cache reuse: when the same prompt is used for repeated generation
        (e.g. batched inference or multi-task evaluation), the text encoder
        outputs are cached and reused, avoiding redundant CLIP/T5 forward
        passes.  The cache is keyed by the prompt string and invalidated
        when the model weights change (e.g. after LoRA loading).

        Args:
            prompt: Text prompt(s) describing the desired output.
            image: Optional input image.  Accepted formats:
                - torch.Tensor  [B, 3, H, W]  in [-1, 1] (batch)
                - torch.Tensor  [3, H, W]     in [-1, 1] (single)
                - PIL.Image.Image
            num_inference_steps: Denoising steps (1 for schnell).
            guidance_scale: CFG scale (0.0 disables CFG, correct for schnell).
            strength: Img2img noise strength in (0, 1].  Higher → more creativity,
                lower → closer to the input image.  Recommended: 0.75–0.9 for
                vision tasks.
            height: Output image height (used for t2i; img2img uses input size).
            width: Output image width (used for t2i; img2img uses input size).
            seed: Random seed for reproducibility.
            **kwargs: Extra keyword arguments forwarded to the pipeline.

        Returns:
            Generated image tensor [B, 3, H, W] in [0, 1] range.
        """
        if self.source_conditioning_config is not None:
            source_steps = num_inference_steps
            if source_steps == 1:
                # ``1`` is the historical schnell default used by the public
                # pipeline. Source-conditioned SSD checkpoints declare their
                # own distilled schedule (4 steps by default).
                source_steps = (
                    self.source_conditioning_config.default_num_inference_steps
                )
            return self._generate_source_conditioned(
                prompt=prompt,
                image=image,
                num_inference_steps=source_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                height=height,
                width=width,
                seed=seed,
                prompt_embeds=kwargs.pop("prompt_embeds", None),
                pooled_prompt_embeds=kwargs.pop("pooled_prompt_embeds", None),
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        precomputed = "prompt_embeds" in kwargs

        # ── KV-cache reuse for text encoder outputs ────────────────────
        if precomputed:
            encode_prompt = False
        else:
            prompt_key = _prompt_cache_key(prompt)
            cached_prompt_embeds = self._prompt_cache.get(prompt_key)
            if cached_prompt_embeds is not None:
                self._prompt_cache.move_to_end(prompt_key)
                kwargs["prompt_embeds"] = cached_prompt_embeds["prompt_embeds"]
                kwargs["pooled_prompt_embeds"] = cached_prompt_embeds["pooled_prompt_embeds"]
                encode_prompt = False
            else:
                encode_prompt = True

        family = getattr(getattr(self, "model_config", None), "family", None)
        if image is not None and family == "sana":
            return self._generate_sana_img2img(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        if image is not None and (
            getattr(self.model_config, "instruction_editing", False)
            or getattr(self.model_config, "in_context_conditioning", False)
        ):
            return self._generate_instruction_edit(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        if image is not None and family == "flux":
            return self._generate_flux_img2img(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )

        if image is not None:
            pil_images = self._to_pil_list(image)
            pipe = self._get_i2i_pipe()
            # Use native prediction_type (epsilon for SDXL) — unified with training
            output = pipe(
                prompt=prompt if encode_prompt else None,
                image=pil_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                height=height,
                width=width,
                generator=generator,
                **kwargs,
            )
        else:
            # t2i: use native prediction_type (epsilon for SDXL) — unified with training
            output = self._t2i_pipe(
                prompt=prompt if encode_prompt else None,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                **{**self._guidance_kwargs(self._t2i_pipe, guidance_scale), **kwargs},
            )

        output_tensor = self._output_to_tensor(output)
        if image is not None:
            # SD3 img2img can fall back to its native 1024px canvas when the
            # constructed pipeline ignores height/width. Keep task outputs
            # aligned with the input raster for downstream decoders/evaluators.
            input_images = self._to_pil_list(image)
            target_width, target_height = input_images[0].size
            if output_tensor.shape[-2:] != (target_height, target_width):
                output_tensor = torch.nn.functional.interpolate(
                    output_tensor,
                    size=(target_height, target_width),
                    mode="nearest",
                )
        return output_tensor.clamp(0, 1)

    @torch.no_grad()
    def _generate_sana_img2img(
        self,
        *,
        prompt: str | list[str],
        image: torch.Tensor | Image.Image,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> torch.Tensor:
        """SANA vision-task img2img via the training flow-matching ODE."""
        from .sana_img2img import (
            flow_matching_inference,
            images_to_sana_source,
            pipeline_components_from_sana,
            resolve_sana_img2img_settings,
        )

        pipe = self._t2i_pipe
        if pipe is None:
            raise RuntimeError("SANA text-to-image pipeline is not loaded")

        default_steps = int(getattr(self.model_config, "default_num_inference_steps", 20) or 20)
        steps, cfg = resolve_sana_img2img_settings(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            default_steps=default_steps,
        )
        if steps != num_inference_steps or cfg != guidance_scale:
            logger.info(
                "SANA img2img using training contract: steps=%d (was %d), guidance=%.2f (was %.2f)",
                steps,
                num_inference_steps,
                cfg,
                guidance_scale,
            )

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        device = torch.device(self.device)
        source = images_to_sana_source(image, device)
        components = pipeline_components_from_sana(pipe, self.model_config)
        max_length = int(getattr(self.model_config, "text_encoder_max_length", 300) or 300)
        if seed is not None:
            torch.manual_seed(seed)

        return flow_matching_inference(
            peft_model=pipe.transformer,
            vae=pipe.vae,
            components=components,
            input_images=source,
            prompts=prompts,
            device=device,
            model_dtype=self.dtype,
            num_steps=steps,
            use_img2img=True,
            max_length=max_length,
            guidance_scale=cfg,
        )

    @torch.no_grad()
    def _generate_instruction_edit(
        self,
        *,
        prompt: str | list[str],
        image: torch.Tensor | Image.Image,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> torch.Tensor:
        """Image+prompt editing through the model's official pipeline.

        FLUX.1 Kontext keeps its in-context call (``_auto_resize=False``). Other
        editors pass only arguments that pipeline declares: Qwen-Image-Edit uses
        ``true_cfg_scale`` plus a blank negative prompt, and pipelines without
        ``height``/``width`` (LongCat-Image-Edit) are not given them.
        """
        if self.model_config.pipeline_class == "FluxKontextPipeline":
            return self._generate_kontext(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        if self.model_config.pipeline_class == "HunyuanImage3Instruct":
            return self._generate_hunyuan_image3_instruct(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
        pipe = self._t2i_pipe
        if pipe is None:
            raise RuntimeError("Instruction-editing pipeline is not loaded")
        import inspect

        default_steps = int(getattr(self.model_config, "default_num_inference_steps", 28) or 28)
        steps = default_steps if num_inference_steps <= 1 else num_inference_steps
        guidance = (
            guidance_scale if guidance_scale > 0 else float(self.model_config.default_guidance_scale)
        )
        parameters = inspect.signature(pipe.__call__).parameters
        explicit = {
            name
            for name, parameter in parameters.items()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        pil_images = self._to_pil_list(image)
        prompts = [prompt] * len(pil_images) if isinstance(prompt, str) else list(prompt)
        if len(prompts) != len(pil_images):
            raise ValueError(f"{len(prompts)} prompts for {len(pil_images)} images")
        outputs = []
        for index, (pil, text) in enumerate(zip(pil_images, prompts, strict=True)):
            width, height = pil.size
            if "images" in explicit and "image" not in explicit:
                call = {"images": [pil], "prompt": text}
            else:
                call = {"image": pil, "prompt": text}
            if "num_inference_steps" in explicit:
                call["num_inference_steps"] = steps
            if "height" in explicit:
                call["height"] = max(16, height // 16 * 16)
            if "width" in explicit:
                call["width"] = max(16, width // 16 * 16)
            if "output_resolution" in explicit:
                # Qwen-Image-2.1 resizes the source to output_resolution**2 pixels (default
                # 1024**2). Training sees source and target at the dataset resolution, which
                # is the size callers pass here, so keep that area, capped at the model's
                # native resolution.
                side = min(int((width * height) ** 0.5), self.model_config.default_resolution)
                call["output_resolution"] = max(256, side // 32 * 32)
            if "true_cfg_scale" in explicit:
                call["true_cfg_scale"] = guidance
                if "negative_prompt" in explicit:
                    call["negative_prompt"] = " "
            elif "guidance_scale" in explicit:
                call["guidance_scale"] = guidance
            if seed is not None and "generator" in explicit:
                call["generator"] = torch.Generator(device=self.device).manual_seed(seed + index)
            result = pipe(**call)
            tensor = self._output_to_tensor(result)
            if tensor.shape[-2:] != (height, width):
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(height, width), mode="bilinear", align_corners=False
                )
            outputs.append(tensor)
        return torch.cat(outputs, dim=0).clamp(0, 1)

    @torch.no_grad()
    def _generate_hunyuan_image3_instruct(
        self,
        *,
        prompt: str | list[str],
        image: torch.Tensor | Image.Image,
        num_inference_steps: int,
        seed: int | None,
    ) -> torch.Tensor:
        """HunyuanImage-3.0-Instruct editing via ``generate_image``.

        One source image is written to a temporary PNG because the published
        call passes file paths. ``infer_align_image_size`` keeps the edit on
        the input raster; the tensor is resized back if the sample differs.
        """
        import tempfile

        model = self._t2i_pipe
        if model is None or not hasattr(model, "generate_image"):
            raise RuntimeError("HunyuanImage-3.0-Instruct is not loaded")
        default_steps = int(getattr(self.model_config, "default_num_inference_steps", 50) or 50)
        steps = default_steps if num_inference_steps <= 1 else num_inference_steps
        pil_images = self._to_pil_list(image)
        prompts = [prompt] * len(pil_images) if isinstance(prompt, str) else list(prompt)
        if len(prompts) != len(pil_images):
            raise ValueError(f"{len(prompts)} prompts for {len(pil_images)} images")
        outputs = []
        for index, (pil, text) in enumerate(zip(pil_images, prompts, strict=True)):
            width, height = pil.size
            with tempfile.TemporaryDirectory(prefix="hunyuan-image3-") as tmp:
                image_path = Path(tmp) / "source.png"
                pil.save(image_path)
                _cot, samples = model.generate_image(
                    prompt=text,
                    image=[str(image_path)],
                    seed=None if seed is None else seed + index,
                    image_size="auto",
                    use_system_prompt="en_unified",
                    bot_task="think_recaption",
                    infer_align_image_size=True,
                    diff_infer_steps=steps,
                    verbose=0,
                )
                edited = samples[0]
                if isinstance(edited, Image.Image):
                    edited = edited.convert("RGB").copy()
                else:
                    edited = Image.open(edited).convert("RGB")
            tensor = self._output_to_tensor(type("_Out", (), {"images": [edited]})())
            if tensor.shape[-2:] != (height, width):
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(height, width), mode="bilinear", align_corners=False
                )
            outputs.append(tensor)
        return torch.cat(outputs, dim=0).clamp(0, 1)

    @torch.no_grad()
    def _generate_kontext(
        self,
        *,
        prompt: str | list[str],
        image: torch.Tensor | Image.Image,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int | None,
    ) -> torch.Tensor:
        """Kontext-style editing inference: source image as in-context tokens, noise start.

        Uses the official ``FluxKontextPipeline`` call, whose conventions (context
        tokens with position-id axis 0 = 1, sigma 1 -> 0, target tokens only) are
        exactly what ``ModelBackend.forward(context_latents=...)`` trains. Output
        keeps the input raster size so decoders/evaluators stay aligned.
        """
        pipe = self._t2i_pipe
        if pipe is None:
            raise RuntimeError("Kontext pipeline is not loaded")
        default_steps = int(getattr(self.model_config, "default_num_inference_steps", 28) or 28)
        steps = default_steps if num_inference_steps <= 1 else num_inference_steps
        guidance = guidance_scale if guidance_scale > 0 else float(self.model_config.default_guidance_scale)
        pil_images = self._to_pil_list(image)
        prompts = [prompt] * len(pil_images) if isinstance(prompt, str) else list(prompt)
        if len(prompts) != len(pil_images):
            raise ValueError(f"{len(prompts)} prompts for {len(pil_images)} images")
        outputs = []
        for index, (pil, text) in enumerate(zip(pil_images, prompts, strict=True)):
            width, height = pil.size
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed + index)
            result = pipe(
                image=pil,
                prompt=text,
                height=max(16, height // 16 * 16),
                width=max(16, width // 16 * 16),
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                _auto_resize=False,
            )
            tensor = self._output_to_tensor(result)
            if tensor.shape[-2:] != (height, width):
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(height, width), mode="bilinear", align_corners=False
                )
            outputs.append(tensor)
        return torch.cat(outputs, dim=0).clamp(0, 1)

    @torch.no_grad()
    def _generate_flux_img2img(
        self,
        *,
        prompt: str | list[str],
        image: torch.Tensor | Image.Image,
        num_inference_steps: int,
        seed: int | None,
    ) -> torch.Tensor:
        """FLUX vision-task img2img via the training Euler flow."""
        from .flux_img2img import flow_matching_inference, images_to_flux_source

        pipe = self._t2i_pipe
        if pipe is None:
            raise RuntimeError("FLUX text-to-image pipeline is not loaded")
        default_steps = int(getattr(self.model_config, "default_num_inference_steps", 4) or 4)
        steps = default_steps if num_inference_steps <= 1 else num_inference_steps
        if seed is not None:
            torch.manual_seed(seed)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        device = torch.device(self.device)
        source = images_to_flux_source(image, device)
        logger.info(
            "FLUX vision img2img using training Euler flow: steps=%d (requested %d)",
            steps,
            num_inference_steps,
        )
        return flow_matching_inference(
            pipe=pipe,
            config=self.model_config,
            input_images=source,
            prompts=prompts,
            device=device,
            model_dtype=self.dtype,
            num_steps=steps,
            guidance=None,
        )

    @torch.no_grad()
    def _generate_source_conditioned(
        self,
        *,
        prompt: str | list[str] | None,
        image: torch.Tensor | Image.Image | None,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        height: int,
        width: int,
        seed: int | None,
        prompt_embeds: torch.Tensor | None,
        pooled_prompt_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Denoise with a clean source latent or zero source channels for t2i."""
        from ..model_registry import build_added_cond_kwargs, vae_decode, vae_encode

        config = self.source_conditioning_config
        if config is None:
            raise RuntimeError("Source-conditioning config is not loaded")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if guidance_scale != 0.0:
            logger.warning(
                "Source-conditioned checkpoints were trained without CFG; "
                "guidance_scale=%s is ignored",
                guidance_scale,
            )
        if strength != 0.85:
            logger.debug(
                "Source-conditioned inference starts from pure noise; strength=%s is ignored",
                strength,
            )

        pipe = self._t2i_pipe
        unet = pipe.unet
        vae = pipe.vae
        spatial_multiple = (
            config.spatial_adapter_downscale_factor
            if self.source_spatial_adapter is not None
            else 8
        )
        unet_device = next(unet.parameters()).device
        unet_dtype = next(unet.parameters()).dtype
        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype

        adapter_pixels = None
        if image is not None:
            pil_images = self._to_pil_list(image)
            original_sizes = [pil.size for pil in pil_images]
            if len(set(original_sizes)) != 1:
                raise ValueError("Source-conditioned batches require equal image sizes")
            original_width, original_height = original_sizes[0]
            batch_size = len(pil_images)
        else:
            if self.source_spatial_adapter is not None:
                raise ValueError("Spatial-adapter checkpoints require a source image")
            original_width, original_height = width, height
            if prompt_embeds is not None:
                batch_size = int(prompt_embeds.shape[0])
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            elif isinstance(prompt, str):
                batch_size = 1
            else:
                raise ValueError("prompt or prompt_embeds is required")

        model_width = max(
            spatial_multiple,
            (original_width // spatial_multiple) * spatial_multiple,
        )
        model_height = max(
            spatial_multiple,
            (original_height // spatial_multiple) * spatial_multiple,
        )
        if image is not None:
            if (model_width, model_height) != (original_width, original_height):
                pil_images = [
                    pil.resize((model_width, model_height), Image.Resampling.LANCZOS)
                    for pil in pil_images
                ]
            source_pixels = torch.cat(
                [_pil_to_tensor(pil, str(vae_device), vae_dtype) for pil in pil_images],
                dim=0,
            )
            adapter_pixels = source_pixels
            source_pixels = source_pixels * 2.0 - 1.0
            source_latents = vae_encode(vae, source_pixels, self.model_config)
            source_latents = source_latents.to(device=unet_device, dtype=unet_dtype)
        else:
            source_latents = torch.zeros(
                batch_size,
                config.condition_channels,
                model_height // self.model_config.vae_downsample_factor,
                model_width // self.model_config.vae_downsample_factor,
                device=unet_device,
                dtype=unet_dtype,
            )
        if source_latents.shape[1] != config.condition_channels:
            raise RuntimeError(
                f"VAE produced {source_latents.shape[1]} source channels; checkpoint "
                f"expects {config.condition_channels}"
            )

        batch_size = source_latents.shape[0]
        if prompt_embeds is None:
            if prompt is None:
                raise ValueError("prompt or prompt_embeds is required")
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            if len(prompts) == 1 and batch_size > 1:
                prompts *= batch_size
            if len(prompts) != batch_size:
                raise ValueError("Number of prompts must match source-image batch size")
            text = self.encode_prompts(prompts)
            prompt_embeds = text["prompt_embeds"]
            pooled_prompt_embeds = text["pooled_prompt_embeds"]
        if pooled_prompt_embeds is None:
            raise ValueError("SDXL source-conditioned inference requires pooled prompt embeds")
        if prompt_embeds.shape[0] != batch_size:
            raise ValueError("Prompt embedding batch does not match source-image batch")
        prompt_embeds = prompt_embeds.to(device=unet_device, dtype=unet_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(
            device=unet_device,
            dtype=unet_dtype,
        )

        scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
        set_scheduler_prediction_type(scheduler, config.prediction_type)
        validate_scheduler_prediction_type(scheduler, config.prediction_type)
        single_step = config.inference_protocol == SINGLE_STEP_X0
        if single_step:
            # The single-step protocol is only valid at the deployed training
            # schedule; the requested step count is ignored by contract.
            if num_inference_steps != config.training_num_inference_steps:
                logger.debug(
                    "single_step_x0 checkpoints use the trained schedule of %d "
                    "steps; num_inference_steps=%d is ignored",
                    config.training_num_inference_steps,
                    num_inference_steps,
                )
            scheduler.set_timesteps(
                config.training_num_inference_steps, device=unet_device
            )
        else:
            scheduler.set_timesteps(num_inference_steps, device=unet_device)
        random_generator = None
        if seed is not None:
            random_generator = torch.Generator(device=unet_device).manual_seed(seed)
        latents = torch.randn(
            source_latents.shape,
            generator=random_generator,
            device=unet_device,
            dtype=unet_dtype,
        ) * scheduler.init_noise_sigma
        added_cond_kwargs = build_added_cond_kwargs(
            config=self.model_config,
            pooled_embeds=pooled_prompt_embeds,
            resolution=(model_height, model_width),
            device=unet_device,
            dtype=unet_dtype,
            batch_size=batch_size,
        )
        spatial_residuals = None
        if self.source_spatial_adapter is not None:
            if adapter_pixels is None:
                raise RuntimeError("Spatial adapter is missing source pixels")
            self.source_spatial_adapter.to(device=unet_device, dtype=unet_dtype)
            spatial_residuals = [
                residual * config.spatial_adapter_conditioning_scale
                for residual in self.source_spatial_adapter(
                    adapter_pixels.to(device=unet_device, dtype=unet_dtype)
                )
            ]

        if not single_step:
            for timestep in scheduler.timesteps:
                scaled_target = scheduler.scale_model_input(latents, timestep)
                model_input = torch.cat((scaled_target, source_latents), dim=1)
                if model_input.shape[1] != config.in_channels:
                    raise RuntimeError(
                        f"Source-conditioned UNet expects {config.in_channels} input "
                        f"channels, received {model_input.shape[1]}"
                    )
                predicted_noise = unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    down_intrablock_additional_residuals=(
                        list(spatial_residuals) if spatial_residuals is not None else None
                    ),
                    return_dict=False,
                )[0]
                latents = scheduler.step(
                    predicted_noise,
                    timestep,
                    latents,
                    return_dict=False,
                )[0]

        if single_step:
            # One UNet forward at the contracted deployed timestep; the
            # (implied) x0 is decoded directly without scheduler iteration.
            assert config.single_step_timestep_index is not None
            timestep = scheduler.timesteps[config.single_step_timestep_index]
            timestep_batch = timestep.reshape(1).to(unet_device)
            base_latents = (
                source_latents
                if config.prediction_type == "sample"
                else torch.zeros_like(source_latents)
            )
            noise = torch.randn(
                source_latents.shape,
                generator=random_generator,
                device=unet_device,
                dtype=unet_dtype,
            )
            noisy = scheduler.add_noise(base_latents, noise, timestep_batch)
            sigma = sigma_for_timesteps(scheduler, timestep_batch, noisy)
            scaled = noisy / torch.sqrt(sigma.square() + 1.0)
            model_input = torch.cat((scaled, source_latents), dim=1)
            if model_input.shape[1] != config.in_channels:
                raise RuntimeError(
                    f"Source-conditioned UNet expects {config.in_channels} input "
                    f"channels, received {model_input.shape[1]}"
                )
            prediction = unet(
                model_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                down_intrablock_additional_residuals=(
                    list(spatial_residuals) if spatial_residuals is not None else None
                ),
                return_dict=False,
            )[0]
            latents = (
                prediction
                if config.prediction_type == "sample"
                else noisy - sigma * prediction
            )

        decoded = vae_decode(
            vae,
            latents.to(device=vae_device, dtype=vae_dtype),
            self.model_config,
        )
        output = (decoded.float().cpu() / 2.0 + 0.5).clamp(0, 1)
        output = calibrate_source_conditioned_output(output, config)
        if output.shape[-2:] != (original_height, original_width):
            output = torch.nn.functional.interpolate(
                output,
                size=(original_height, original_width),
                mode="nearest",
            )
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_pil_list(self, image: torch.Tensor | Image.Image) -> list[Image.Image]:
        """Normalise various image formats to a list of PIL images."""
        if isinstance(image, Image.Image):
            return [image]

        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return [_tensor_to_pil(image[i]) for i in range(image.shape[0])]

        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _output_to_tensor(output) -> torch.Tensor:
        """Convert diffusers pipeline output to [B, 3, H, W] float tensor in [0, 1]."""
        images = output.images
        import numpy as np

        # Handle numpy array output directly (diffusers output_type="np" or special configs)
        if isinstance(images, np.ndarray):
            arr = images.astype("float32") / 255.0 if images.dtype == np.uint8 else images.astype("float32")
            tensor = torch.from_numpy(arr)
            if tensor.dim() == 3:  # [H, W, 3]
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            elif tensor.dim() == 4:  # [B, H, W, 3]
                tensor = tensor.permute(0, 3, 1, 2)
            return tensor.clamp(0, 1)

        if not isinstance(images, list):
            return images  # already a tensor in some diffusers versions

        tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                arr = np.array(img.convert("RGB")).astype("float32") / 255.0
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            elif isinstance(img, np.ndarray):
                arr = img.astype("float32") / 255.0 if img.dtype == np.uint8 else img.astype("float32")
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            elif isinstance(img, torch.Tensor):
                tensors.append(img.float())
            else:
                raise TypeError(f"Unexpected image type in pipeline output: {type(img)}")

        return torch.stack(tensors)  # [B, 3, H, W]

    # ------------------------------------------------------------------
    # LoRA management
    # ------------------------------------------------------------------

    @property
    def pipe(self):
        """Return the primary (t2i) pipeline for backward compatibility."""
        return self._t2i_pipe

    def save_lora(self, output_dir: str | Path) -> None:
        """Save LoRA weights to a directory."""
        if self._t2i_pipe is None:
            raise RuntimeError("Model not loaded")
        self._t2i_pipe.save_lora_weights(str(output_dir))
        logger.info(f"LoRA weights saved to {output_dir}")

    def unload_lora(self) -> None:
        """Unload LoRA weights, reverting to the base model."""
        if self._t2i_pipe is not None:
            self._t2i_pipe.unload_lora_weights()
            self._i2i_pipe = None  # force rebuild without LoRA
            self._prompt_cache.clear()  # invalidate cached embeddings
            logger.info("LoRA weights unloaded")

    def _prepare_lora_load(self, lora_dir: Path, pipe) -> tuple[str | dict, dict]:
        """Return a Diffusers-compatible LoRA payload.

        PEFT ``adapter_model.safetensors`` keys start with ``base_model.model.``.
        FluxPipeline treats that prefix as Fal Kontext / BFL format and looks
        for ``double_blocks.*.img_mod.lin``, which fails. Convert those keys
        to ``transformer.*`` / ``unet.*`` so ``load_lora_weights`` injects
        adapters instead of falling through to a destructive merge.
        """
        load_kwargs: dict = {}
        companion = lora_dir / DIFFUSERS_LORA_WEIGHT_NAME
        if companion.is_file():
            load_kwargs["weight_name"] = DIFFUSERS_LORA_WEIGHT_NAME
            return str(lora_dir), load_kwargs

        adapter = lora_dir / "adapter_model.safetensors"
        if adapter.is_file():
            from safetensors.torch import load_file

            raw = load_file(str(adapter))
            if any(key.startswith("base_model.model.") for key in raw):
                family = getattr(getattr(self, "model_config", None), "family", None)
                # FLUX needed PEFT→Diffusers conversion to avoid Fal Kontext routing.
                # SANA PEFT adapters historically rely on load_lora failing (0 layers)
                # then manual merge with rsLoRA scale=alpha/sqrt(r). Converting SANA
                # keys makes load_lora "succeed" without that scale and collapses
                # metric-depth δ1 (Hypersim-500: ~0.36 → ~0.02 under current stack).
                if family in {"sana", "sana-sprint"}:
                    logger.info(
                        "Skipping PEFT→Diffusers LoRA conversion for family=%s; "
                        "prefer manual rsLoRA merge fallback",
                        family,
                    )
                    return str(lora_dir), load_kwargs
                component = (
                    "transformer"
                    if getattr(pipe, "transformer", None) is not None
                    else "unet"
                )
                converted = peft_adapter_state_to_diffusers(raw, component)
                logger.info(
                    "Converted PEFT adapter keys to Diffusers %s LoRA format (%d tensors)",
                    component,
                    len(converted),
                )
                return converted, load_kwargs
        return str(lora_dir), load_kwargs

    def _load_and_merge_lora(self, pipe, lora_path: str) -> int:
        """Load LoRA weights from safetensors and merge into base model.

        Bypasses the PEFT wrapper entirely — reads the adapter weights
        directly and applies W += scale * B @ A to the target modules.
        This is the fallback when pipe.load_lora_weights() triggers the
        diffusers PEFT mis-route bug.

        Args:
            pipe: The diffusers FluxPipeline instance.
            lora_path: Path to the directory containing adapter_model.safetensors.
        """
        from safetensors.torch import load_file

        adapter_path = Path(lora_path)
        weights_file = adapter_path / "adapter_model.safetensors"
        if not weights_file.exists():
            raise FileNotFoundError(f"LoRA weights not found: {weights_file}")

        # Read adapter config for scaling
        config_path = adapter_path / "adapter_config.json"
        import json
        with open(config_path) as f:
            config = json.load(f)

        lora_rank = config.get("r", 64)
        lora_alpha = config.get("lora_alpha", 128)
        use_rslora = config.get("use_rslora", False)
        target_modules = config.get("target_modules", [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2",
        ])

        # rsLoRA scale = alpha / sqrt(rank); standard LoRA scale = alpha / rank
        scale = lora_alpha / lora_rank ** 0.5 if use_rslora else lora_alpha / lora_rank

        logger.info(
            "Manual LoRA merge: rank=%d, alpha=%d, rslora=%s, scale=%.4f, targets=%s",
            lora_rank, lora_alpha, use_rslora, scale, target_modules
        )

        state_dict = load_file(str(weights_file))

        # SD3/FLUX use pipe.transformer; SDXL/SD1.5/SD2 use pipe.unet
        if hasattr(pipe, "transformer"):
            backbone = pipe.transformer
            backbone_attr = "transformer"
        elif hasattr(pipe, "unet"):
            backbone = pipe.unet
            backbone_attr = "unet"
        else:
            raise AttributeError(
                f"Pipeline has neither 'transformer' nor 'unet': {type(pipe).__name__}"
            )
        logger.info("Using backbone '%s' (%s) for LoRA merge", backbone_attr, type(backbone).__name__)

        # Group LoRA weights by module name
        # Key format: base_model.model.{module_name}.lora_A.weight
        #             base_model.model.{module_name}.lora_B.weight
        lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in state_dict.items():
            # Extract module name and A/B
            parts = key.split(".")
            # Find lora_A or lora_B in the key
            for i, part in enumerate(parts):
                if part in ("lora_A", "lora_B"):
                    module_name = ".".join(parts[parts.index("model") + 1:i]) if "model" in parts else ".".join(parts[:-2])
                    if module_name not in lora_pairs:
                        lora_pairs[module_name] = {}
                    lora_pairs[module_name][part] = tensor
                    break

        merged_count = 0
        for module_name, pair in lora_pairs.items():
            if "lora_A" not in pair or "lora_B" not in pair:
                continue

            # Navigate to the target module
            module = backbone
            for attr in module_name.split("."):
                if hasattr(module, attr):
                    module = getattr(module, attr)
                else:
                    logger.warning("Module '%s' not found in %s, skipping", module_name, backbone_attr)
                    module = None
                    break

            if module is None:
                continue

            if not hasattr(module, "weight"):
                logger.warning("Module '%s' has no 'weight' attribute, skipping", module_name)
                continue

            # Checkpoints are loaded on CPU while the pipeline may already be
            # on MPS/CUDA; move adapter factors to the target layer first.
            A = pair["lora_A"].to(device=module.weight.device, dtype=module.weight.dtype)
            B = pair["lora_B"].to(device=module.weight.device, dtype=module.weight.dtype)

            # W += scale * B @ A. PEFT uses 2-D factors for Linear
            # modules and 4-D factors for Conv2d modules. Most convolutional
            # LoRA B factors are 1x1, so composing B with A is an einsum over
            # rank while preserving A's spatial kernel.
            if A.ndim == 2 and B.ndim == 2:
                delta = scale * (B @ A)
            elif A.ndim == 4 and B.ndim == 4 and B.shape[2:] == (1, 1):
                delta = scale * torch.einsum("or,rihw->oihw", B[:, :, 0, 0], A)
            else:
                logger.warning(
                    "Unsupported LoRA factor shapes for '%s': A=%s B=%s, skipping",
                    module_name,
                    tuple(A.shape),
                    tuple(B.shape),
                )
                continue
            if module.weight.shape == delta.shape:
                module.weight.data += delta
                merged_count += 1
            else:
                logger.warning(
                    "Shape mismatch for '%s': weight %s vs delta %s, skipping",
                    module_name, tuple(module.weight.shape), tuple(delta.shape)
                )

        logger.info("Manually merged %d LoRA layers into base model", merged_count)
        return merged_count

    def enable_cpu_offload(self) -> None:
        """Enable sequential CPU offload on already-loaded pipelines."""
        for pipe in [self._t2i_pipe, self._i2i_pipe]:
            if pipe is not None:
                pipe.enable_sequential_cpu_offload()
        self.enable_sequential_cpu_offload = True


__all__ = ["BaseGenerator"]
