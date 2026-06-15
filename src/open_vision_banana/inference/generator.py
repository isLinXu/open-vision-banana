"""FLUX.1 base generation wrapper with LoRA instruction alignment.

This module provides the base generation layer using FLUX.1 [schnell]
with parameter-efficient fine-tuning via LoRA (rsLoRA scaling).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class BaseGenerator:
    """FLUX.1 base generation wrapper with LoRA support.

    Loads FLUX.1 [schnell] and optionally applies LoRA for instruction
    alignment. The model supports single-step generation (schnell variant)
    and can be fine-tuned with only 0.5%-1% of parameters via LoRA.
    """

    DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        lora_path: str | None = None,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_name = model_name
        self.lora_path = lora_path
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.pipe = None
        self._load_model()

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

    def _load_model(self) -> None:
        """Load FLUX.1 pipeline and optionally apply LoRA weights."""
        from diffusers import FluxPipeline

        logger.info(f"Loading FLUX.1 model from {self.model_name}")
        self.pipe = FluxPipeline.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )

        if self.lora_path is not None:
            logger.info(f"Loading LoRA weights from {self.lora_path}")
            self.pipe.load_lora_weights(self.lora_path)

        self.pipe = self.pipe.to(self.device)
        logger.info("Model loaded successfully")

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[str],
        image: torch.Tensor | None = None,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        seed: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate images from text prompts.

        Args:
            prompt: Text prompt(s) for generation.
            image: Optional input image for image-conditioned tasks.
            num_inference_steps: Number of denoising steps (1 for schnell).
            guidance_scale: Classifier-free guidance scale (0 for schnell).
            height: Output image height.
            width: Output image width.
            seed: Random seed for reproducibility.

        Returns:
            Generated image tensor [B, 3, H, W] in [-1, 1] range.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            **kwargs,
        )

        images = output.images
        if isinstance(images, list) and len(images) > 0:
            from torchvision.transforms.functional import pil_to_tensor
            tensors = []
            for img in images:
                t = pil_to_tensor(img).float() / 255.0
                t = t * 2.0 - 1.0
                tensors.append(t)
            return torch.stack(tensors)
        return images

    def save_lora(self, output_dir: str | Path) -> None:
        """Save LoRA weights to directory."""
        if self.pipe is None:
            raise RuntimeError("Model not loaded")
        self.pipe.save_lora_weights(output_dir)
        logger.info(f"LoRA weights saved to {output_dir}")

    def unload_lora(self) -> None:
        """Unload LoRA weights, reverting to base model."""
        if self.pipe is not None:
            self.pipe.unload_lora_weights()
            logger.info("LoRA weights unloaded")


__all__ = ["BaseGenerator"]
