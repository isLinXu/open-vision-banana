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

import torch
from PIL import Image

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
        self._t2i_pipe = None  # FluxPipeline (text → image)
        self._i2i_pipe = None  # FluxImg2ImgPipeline (image + text → image)

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
        """Load FLUX.1 text-to-image pipeline and optionally apply LoRA."""
        from diffusers import FluxPipeline

        logger.info(f"Loading FLUX.1 pipeline from {self.model_name!r} …")
        pipe = FluxPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )

        if self.lora_path is not None:
            self._resolved_lora_path = resolve_lora_path(
                self.lora_path,
                revision=self.lora_revision,
            )
            logger.info(f"Loading LoRA weights from {self._resolved_lora_path!r} …")
            pipe.load_lora_weights(self._resolved_lora_path)

        applied = apply_pipeline_optimizations(pipe, self.inference_options)
        if applied:
            logger.info("Applied inference optimizations: %s", ", ".join(applied))
        if not self.inference_options.enable_sequential_cpu_offload:
            pipe = pipe.to(self.device)

        self._t2i_pipe = pipe
        logger.info("FLUX.1 pipeline loaded successfully")

    def _get_i2i_pipe(self):
        """Lazily build the img2img pipeline by sharing weights with t2i."""
        if self._i2i_pipe is not None:
            return self._i2i_pipe

        from diffusers import FluxImg2ImgPipeline

        logger.info("Building img2img pipeline from shared components …")
        t2i = self._t2i_pipe
        self._i2i_pipe = FluxImg2ImgPipeline(
            scheduler=t2i.scheduler,
            vae=t2i.vae,
            text_encoder=t2i.text_encoder,
            tokenizer=t2i.tokenizer,
            text_encoder_2=t2i.text_encoder_2,
            tokenizer_2=t2i.tokenizer_2,
            transformer=t2i.transformer,
        )

        applied = apply_pipeline_optimizations(self._i2i_pipe, self.inference_options)
        if applied:
            logger.info("Applied inference optimizations to img2img pipeline: %s", ", ".join(applied))
        if not self.inference_options.enable_sequential_cpu_offload:
            self._i2i_pipe = self._i2i_pipe.to(self.device)

        logger.info("Img2img pipeline ready")
        return self._i2i_pipe

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> dict[str, torch.Tensor | None]:
        """Batch-encode text prompts via FLUX dual encoders (P1-12).

        Encodes all prompts in one ``encode_prompt`` call when possible, and
        caches each prompt's embeddings individually for LRU reuse.

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
        text_outputs = self._t2i_pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if isinstance(text_outputs, (tuple, list)) and len(text_outputs) >= 3:
            prompt_embeds = text_outputs[0]
            pooled_prompt_embeds = text_outputs[2]
        else:
            prompt_embeds = text_outputs[0]
            pooled_prompt_embeds = text_outputs[1] if len(text_outputs) > 1 else None

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

    def enable_cpu_offload(self) -> None:
        """Enable sequential CPU offload on already-loaded pipelines."""
        for pipe in [self._t2i_pipe, self._i2i_pipe]:
            if pipe is not None:
                pipe.enable_sequential_cpu_offload()
        self.enable_sequential_cpu_offload = True


__all__ = ["BaseGenerator"]
