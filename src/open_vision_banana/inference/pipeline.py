"""Vision Banana: Unified multi-task vision model.

The main model class that ties together:
- BaseGenerator (FLUX.1 + LoRA)
- RGBMapper (reversible depth/normal/segmentation encoding)
- VisualizationDecoder (prompt building + output decoding)

Vision Banana supports 5 vision tasks via prompt-based task selection:
1. Semantic Segmentation (2D)
2. Instance Segmentation (2D)
3. Referring Expression Segmentation (2D)
4. Metric Depth Estimation (3D)
5. Surface Normal Estimation (3D)

All tasks share the same model weights — only the prompt changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..core.rgb_mapper import RGBMapper
from .decoder import VisualizationDecoder
from .generator import BaseGenerator
from .prompt import PromptBuilder

logger = logging.getLogger(__name__)


class VisionBanana:
    """Vision Banana: A generalist vision model for generation and understanding.

    Built by instruction-tuning FLUX.1 [schnell] on a mixture of its
    original training data alongside a small amount of vision task data.
    By parameterizing the output space of vision tasks as RGB images,
    we seamlessly reframe perception as image generation.

    Features:
    - 5 vision tasks via prompt-based selection
    - Batched inference with automatic result splitting
    - KV-cache reuse for repeated generation
    - TorchScript export for deployment
    """

    SUPPORTED_TASKS = [
        "semantic", "instance", "referring", "depth", "normal"
    ]

    def __init__(
        self,
        base_generator: BaseGenerator,
        rgb_mapper: RGBMapper | None = None,
        decoder: VisualizationDecoder | None = None,
        enable_cache: bool = True,
        cache_size: int = 64,
    ) -> None:
        self.generator = base_generator
        self.rgb_mapper = rgb_mapper or RGBMapper()
        self.decoder = decoder or VisualizationDecoder(self.rgb_mapper)
        self.prompt_builder = PromptBuilder()
        self.enable_cache = enable_cache
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_size = cache_size

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
        lora_path: str | None = None,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ) -> VisionBanana:
        """Load a pretrained Vision Banana model."""
        generator = BaseGenerator(
            model_name=model_name,
            lora_path=lora_path,
            device=device,
            dtype=dtype,
        )
        return cls(base_generator=generator)

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        task: str,
        class_color_map: dict | None = None,
        object_name: str | None = None,
        expression: str | None = None,
        background_color: tuple[int, int, int] = (0, 0, 0),
        num_inference_steps: int = 1,
        height: int = 1024,
        width: int = 1024,
        seed: int | None = None,
    ) -> dict:
        """Run inference for a vision task.

        Args:
            image: Input image tensor [B, 3, H, W] in [-1, 1].
            task: Vision task name.
            class_color_map: Required for semantic segmentation.
            object_name: Required for instance segmentation.
            expression: Required for referring segmentation.
            background_color: Background color for instance segmentation.
            num_inference_steps: Denoising steps (1 for schnell).
            height: Output image height.
            width: Output image width.
            seed: Random seed for reproducibility.

        Returns:
            Task-specific result dict with decoded outputs.
        """
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Unsupported task: {task}. "
                f"Choose from: {self.SUPPORTED_TASKS}"
            )

        prompt = self._build_prompt(
            task=task,
            class_color_map=class_color_map,
            object_name=object_name,
            expression=expression,
        )

        logger.info(f"Running {task} prediction with prompt: {prompt[:100]}...")

        rgb_output = self.generator.generate(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            seed=seed,
        )

        result = self._decode_output(
            rgb_output=rgb_output,
            task=task,
            class_color_map=class_color_map,
            background_color=background_color,
        )

        result["prompt"] = prompt
        result["rgb_output"] = rgb_output

        return result

    def predict_batch(
        self,
        images: torch.Tensor,
        task: str,
        **kwargs,
    ) -> list[dict]:
        """Run batched inference, returning per-image results.

        Args:
            images: Input images [B, 3, H, W] in [-1, 1].
            task: Vision task name.
            **kwargs: Additional arguments passed to predict().

        Returns:
            List of result dicts, one per image.
        """
        result = self.predict(image=images, task=task, **kwargs)
        B = images.shape[0]

        # Split batch results into per-image dicts
        per_image = []
        for i in range(B):
            item = {}
            for key, val in result.items():
                if isinstance(val, torch.Tensor) and val.shape[0] == B:
                    item[key] = val[i:i+1]
                elif isinstance(val, str):
                    item[key] = val
                else:
                    item[key] = val
            per_image.append(item)

        return per_image

    def _build_prompt(
        self,
        task: str,
        class_color_map: dict | None = None,
        object_name: str | None = None,
        expression: str | None = None,
    ) -> str:
        """Build instruction prompt for the specified task."""
        if task == "semantic":
            if class_color_map is None:
                raise ValueError("class_color_map required for semantic segmentation")
            return self.prompt_builder.build_semantic_prompt(class_color_map)

        elif task == "instance":
            if object_name is None:
                raise ValueError("object_name required for instance segmentation")
            return self.prompt_builder.build_instance_prompt(object_name)

        elif task == "referring":
            if expression is None:
                raise ValueError("expression required for referring segmentation")
            return self.prompt_builder.build_referring_prompt(expression)

        elif task == "depth":
            return self.prompt_builder.build_depth_prompt()

        elif task == "normal":
            return self.prompt_builder.build_normal_prompt()

        raise ValueError(f"Unknown task: {task}")

    def _decode_output(
        self,
        rgb_output: torch.Tensor,
        task: str,
        class_color_map: dict | None = None,
        background_color: tuple[int, int, int] = (0, 0, 0),
    ) -> dict:
        """Decode RGB output to vision task result."""
        if task == "depth":
            return self.decoder.decode_depth(rgb_output)

        elif task == "normal":
            return self.decoder.decode_normal(rgb_output)

        elif task == "semantic":
            if class_color_map is None:
                raise ValueError("class_color_map required for semantic decoding")
            return self.decoder.decode_semantic(rgb_output, class_color_map)

        elif task == "instance":
            return self.decoder.decode_instance(rgb_output, background_color)

        elif task == "referring":
            if class_color_map is None:
                class_color_map = {
                    "target": (255, 255, 255),
                    "background": (0, 0, 0),
                }
            return self.decoder.decode_semantic(rgb_output, class_color_map)

        raise ValueError(f"Unknown task: {task}")

    def save(self, output_dir: str | Path) -> None:
        """Save Vision Banana model (LoRA weights)."""
        self.generator.save_lora(output_dir)
        logger.info(f"Vision Banana saved to {output_dir}")

    def clear_cache(self) -> None:
        """Clear the prompt embedding cache."""
        self._cache.clear()

    def export_torchscript(
        self,
        output_path: str | Path,
        task: str = "depth",
        input_shape: tuple = (1, 3, 512, 512),
    ) -> None:
        """Export the RGB decoding pipeline as TorchScript.

        Only exports the decoder (RGB → vision output), not the
        full diffusion model (which is too large for TorchScript).

        Args:
            output_path: Path to save the TorchScript model.
            task: Task type for the decoder.
            input_shape: Input tensor shape for tracing.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if task == "depth":
            from ..core.depth_mapper import DepthRGBMapper
            mapper = DepthRGBMapper()

            class DepthDecoderScript(torch.nn.Module):
                def __init__(self, mapper):
                    super().__init__()
                    self.mapper = mapper

                def forward(self, rgb: torch.Tensor) -> torch.Tensor:
                    return self.mapper.rgb_to_depth(rgb)

            script_model = DepthDecoderScript(mapper)
            dummy_input = torch.rand(*input_shape)
            scripted = torch.jit.trace(script_model, dummy_input)
            scripted.save(str(output_path))

        elif task == "normal":
            from ..core.normal_mapper import NormalRGBMapper
            mapper = NormalRGBMapper()

            class NormalDecoderScript(torch.nn.Module):
                def __init__(self, mapper):
                    super().__init__()
                    self.mapper = mapper

                def forward(self, rgb: torch.Tensor) -> torch.Tensor:
                    return self.mapper.rgb_to_normal(rgb)

            script_model = NormalDecoderScript(mapper)
            dummy_input = torch.rand(*input_shape)
            scripted = torch.jit.trace(script_model, dummy_input)
            scripted.save(str(output_path))

        else:
            raise ValueError(
                f"TorchScript export not supported for task: {task}. "
                f"Choose from: depth, normal"
            )

        logger.info(f"TorchScript model exported to {output_path}")


__all__ = ["VisionBanana"]
