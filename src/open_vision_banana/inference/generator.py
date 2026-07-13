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

from ..model_registry import ModelConfig, resolve_model_config
from .checkpoint import resolve_lora_path
from .optimization import InferenceOptions, apply_pipeline_optimizations

logger = logging.getLogger(__name__)


def _prompt_cache_key(prompt: str | list[str]) -> str:
    """Build a collision-resistant cache key for prompt embedding reuse."""
    if isinstance(prompt, str):
        return prompt
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


def _pil_to_tensor(img: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Convert PIL Image to [1, C, H, W] tensor in [0, 1]."""
    import numpy as np

    arr = (
        torch.from_numpy((np.array(img.convert("RGB")) / 255.0).astype("float32"))
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    return arr.to(device=device, dtype=dtype)


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

        pipe = PipeClass.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )

        if self.lora_path is not None:
            self._resolved_lora_path = resolve_lora_path(
                self.lora_path,
                revision=self.lora_revision,
            )
            logger.info(f"Loading LoRA weights from {self._resolved_lora_path!r} …")
            try:
                pipe.load_lora_weights(self._resolved_lora_path)
                logger.info("LoRA weights loaded via pipe.load_lora_weights")
            except Exception as e:
                logger.warning(
                    "pipe.load_lora_weights failed (%s); "
                    "falling back to manual LoRA weight merge.", e
                )
                self._load_and_merge_lora(pipe, self._resolved_lora_path)

        applied = apply_pipeline_optimizations(pipe, self.inference_options)
        if applied:
            logger.info("Applied inference optimizations: %s", ", ".join(applied))
        if not self.inference_options.enable_sequential_cpu_offload:
            pipe = pipe.to(self.device)

        self._t2i_pipe = pipe
        logger.info(f"{pipe_class_name} loaded successfully (family={family})")

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

        if image is not None:
            pil_images = self._to_pil_list(image)
            pipe = self._get_i2i_pipe()
            output = pipe(
                prompt=prompt if encode_prompt else None,
                image=pil_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
                **kwargs,
            )
        else:
            output = self._t2i_pipe(
                prompt=prompt if encode_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                **kwargs,
            )

        return self._output_to_tensor(output)

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

    def _load_and_merge_lora(self, pipe, lora_path: str) -> None:
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
        target_modules = config.get("target_modules", ["to_k", "to_v"])

        # rsLoRA scale = alpha / sqrt(rank); standard LoRA scale = alpha / rank
        scale = lora_alpha / lora_rank ** 0.5 if use_rslora else lora_alpha / lora_rank

        logger.info(
            "Manual LoRA merge: rank=%d, alpha=%d, rslora=%s, scale=%.4f, targets=%s",
            lora_rank, lora_alpha, use_rslora, scale, target_modules
        )

        state_dict = load_file(str(weights_file))
        transformer = pipe.transformer

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
            module = transformer
            for attr in module_name.split("."):
                if hasattr(module, attr):
                    module = getattr(module, attr)
                else:
                    logger.warning("Module '%s' not found in transformer, skipping", module_name)
                    module = None
                    break

            if module is None:
                continue

            if not hasattr(module, "weight"):
                logger.warning("Module '%s' has no 'weight' attribute, skipping", module_name)
                continue

            A = pair["lora_A"].to(module.weight.dtype)  # [rank, in_features]
            B = pair["lora_B"].to(module.weight.dtype)  # [out_features, rank]

            # W += scale * B @ A
            delta = scale * (B @ A)
            if module.weight.shape == delta.shape:
                module.weight.data += delta
                merged_count += 1
            else:
                logger.warning(
                    "Shape mismatch for '%s': weight %s vs delta %s, skipping",
                    module_name, tuple(module.weight.shape), tuple(delta.shape)
                )

        logger.info("Manually merged %d LoRA layers into base model", merged_count)

    def enable_cpu_offload(self) -> None:
        """Enable sequential CPU offload on already-loaded pipelines."""
        for pipe in [self._t2i_pipe, self._i2i_pipe]:
            if pipe is not None:
                pipe.enable_sequential_cpu_offload()
        self.enable_sequential_cpu_offload = True


__all__ = ["BaseGenerator"]
