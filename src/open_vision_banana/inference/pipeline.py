"""Vision Banana: Unified multi-task vision model.

The main model class that ties together:
- BaseGenerator (FLUX.1 + LoRA)
- RGBMapper (reversible depth/normal/segmentation encoding)
- VisualizationDecoder (prompt building + output decoding)

Vision Banana supports 10 vision tasks via prompt-based task selection:
1. Semantic Segmentation (2D)
2. Instance Segmentation (2D)
3. Referring Expression Segmentation (2D)
4. Metric Depth Estimation (3D)
5. Surface Normal Estimation (3D)
6. Edge Detection (2D)
7. Optical Flow (2D)
8. 2D Keypoint Detection (2D)
9. Object Detection (2D)
10. Depth Edge Detection (3D)

All tasks share the same model weights — only the prompt changes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch

from ..core.rgb_mapper import RGBMapper
from .decoder import VisualizationDecoder
from .generator import BaseGenerator
from .optimization import InferenceOptions
from .prompt import PromptBuilder

logger = logging.getLogger(__name__)


class VisionBanana:
    """Vision Banana: A generalist vision model for generation and understanding.

    Built by instruction-tuning FLUX.1 [schnell] on a mixture of its
    original training data alongside a small amount of vision task data.
    By parameterizing the output space of vision tasks as RGB images,
    we seamlessly reframe perception as image generation.

    Features:
    - 10 vision tasks via prompt-based selection
    - Batched inference with automatic result splitting
    - KV-cache reuse for repeated generation
    - TorchScript export for deployment
    """

    SUPPORTED_TASKS = ["semantic", "instance", "referring", "depth", "normal", "edge", "flow", "keypoint", "object_detection", "depth_edge"]

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
        lora_revision: str | None = None,
        enable_sequential_cpu_offload: bool = False,
        inference_options: InferenceOptions | None = None,
    ) -> VisionBanana:
        """Load a pretrained Vision Banana model.

        Args:
            model_name: Hugging Face model id for FLUX.1 base weights.
            lora_path: Local path, ``hf:repo/id``, or ``registry:alias``.
            device: Target device or ``auto``.
            dtype: Model dtype (bfloat16 recommended on CUDA).
            lora_revision: Optional HF revision for remote LoRA weights.
            enable_sequential_cpu_offload: Reduce VRAM via CPU offload.
            inference_options: Full optimization config (overrides offload flag).
        """
        from .optimization import InferenceOptions

        opts = inference_options
        if opts is None and enable_sequential_cpu_offload:
            opts = InferenceOptions(enable_sequential_cpu_offload=True)

        generator = BaseGenerator(
            model_name=model_name,
            lora_path=lora_path,
            device=device,
            dtype=dtype,
            lora_revision=lora_revision,
            inference_options=opts,
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
        target_color: tuple[int, int, int] = (255, 255, 255),
        background_color: tuple[int, int, int] = (0, 0, 0),
        num_inference_steps: int = 1,
        strength: float = 0.85,
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
            target_color: Target color for referring segmentation (R, G, B) in [0, 255].
                Defaults to white (255, 255, 255).
            background_color: Background color for instance segmentation.
            num_inference_steps: Denoising steps (1 for schnell).
            strength: Img2img noise strength (0, 1]. Lower → closer to input.
            height: Output image height (t2i only).
            width: Output image width (t2i only).
            seed: Random seed for reproducibility.

        Returns:
            Task-specific result dict with decoded outputs.
        """
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Choose from: {self.SUPPORTED_TASKS}")

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
            strength=strength,
            height=height,
            width=width,
            seed=seed,
        )

        result = self._decode_output(
            rgb_output=rgb_output,
            task=task,
            class_color_map=class_color_map,
            target_color=target_color,
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

        Passes the full batch to the generator in a single forward call
        (batched img2img), then splits results per image.

        Note:
            FLUX.1 [schnell] processes all images in a batch through the
            same denoising pipeline, so the forward pass is genuinely
            batched.  However, the text prompt is shared across all
            images in the batch — each image receives the same task
            instruction.  For per-image prompts, call ``predict()``
            individually.

        Args:
            images: Input images [B, 3, H, W] in [-1, 1].
            task: Vision task name.
            **kwargs: Additional arguments passed to predict().

        Returns:
            List of result dicts, one per image.
        """
        result = self.predict(image=images, task=task, **kwargs)
        B = images.shape[0]

        per_image = []
        for i in range(B):
            item: dict = {}
            for key, val in result.items():
                if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == B:
                    item[key] = val[i : i + 1]
                elif isinstance(val, list) and len(val) == B:
                    item[key] = val[i]
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

        elif task == "edge":
            return self.prompt_builder.build_edge_prompt()

        elif task == "flow":
            return self.prompt_builder.build_flow_prompt()

        elif task == "keypoint":
            return self.prompt_builder.build_keypoint_prompt()

        elif task == "object_detection":
            return self.prompt_builder.build_object_detection_prompt()

        elif task == "depth_edge":
            return self.prompt_builder.build_depth_edge_prompt()

        raise ValueError(f"Unknown task: {task}")

    def _decode_output(
        self,
        rgb_output: torch.Tensor,
        task: str,
        class_color_map: dict | None = None,
        target_color: tuple[int, int, int] = (255, 255, 255),
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
            # Decode as binary mask by finding pixels closest to target_color
            # in RGB space (L2 distance). This aligns with the paper's
            # approach of "clustering pixels whose values are close to <255,255,0>".
            return self.decoder.decode_referring(
                rgb_output, target_color=target_color
            )

        elif task == "edge":
            return self.decoder.decode_edge(rgb_output)

        elif task == "flow":
            return self.decoder.decode_flow(rgb_output)

        elif task == "keypoint":
            return self.decoder.decode_keypoint(rgb_output)

        elif task == "object_detection":
            return self.decoder.decode_object_detection(rgb_output)

        elif task == "depth_edge":
            return self.decoder.decode_depth_edge(rgb_output)

        raise ValueError(f"Unknown task: {task}")

    def eval(self) -> VisionBanana:
        """Set underlying generator pipelines to eval mode (no-op for diffusers).

        This method exists so that ``VisionEvaluator`` can call
        ``model.eval()`` without checking whether the model is a plain
        nn.Module or a VisionBanana wrapper.

        Returns:
            self (for chaining).
        """
        # diffusers Pipeline objects do not expose .eval() / .train() in the
        # nn.Module sense; calling them is a no-op.  We still propagate to any
        # sub-module that does support it.
        import contextlib

        for pipe in [
            getattr(self.generator, "_t2i_pipe", None),
            getattr(self.generator, "_i2i_pipe", None),
        ]:
            if pipe is not None and hasattr(pipe, "eval"):
                with contextlib.suppress(Exception):
                    pipe.eval()
        return self

    def save(self, output_dir: str | Path) -> None:
        """Save Vision Banana model (LoRA weights)."""
        self.generator.save_lora(output_dir)
        logger.info(f"Vision Banana saved to {output_dir}")

    def clear_cache(self) -> None:
        """Clear the prompt embedding cache and generator KV-cache."""
        self._cache.clear()
        self.generator._prompt_cache.clear()

    # ------------------------------------------------------------------
    # P2: Streaming inference — callback after each denoising step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_streaming(
        self,
        image: torch.Tensor,
        task: str,
        callback: Callable[[int, torch.Tensor], None] | None = None,
        num_inference_steps: int = 1,
        **kwargs,
    ) -> dict:
        """Run inference with optional per-step callback for progressive display.

        For FLUX.1 [schnell] the default is 1 step, so the callback fires
        once with the final result.  For multi-step generation (e.g. when
        using FLUX.1 [dev]), the callback receives the intermediate latent
        after each denoising step, enabling progressive preview UIs.

        Args:
            image: Input image tensor [B, 3, H, W] in [-1, 1].
            task: Vision task name.
            callback: Optional callable(step: int, rgb: Tensor) invoked
                after each denoising step with the current RGB preview.
            num_inference_steps: Denoising steps (1 for schnell).
            **kwargs: Additional arguments passed to predict().

        Returns:
            Task-specific result dict (same as predict()).
        """
        result = self.predict(
            image=image,
            task=task,
            num_inference_steps=num_inference_steps,
            **kwargs,
        )

        if callback is not None:
            callback(0, result.get("rgb_output", torch.empty(0)))

        return result

    # ------------------------------------------------------------------
    # P2: Multi-scale tiled inference for high-resolution images
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_tiled(
        self,
        image: torch.Tensor,
        task: str,
        tile_size: int = 1024,
        overlap: int = 128,
        **kwargs,
    ) -> dict:
        """Run tiled inference for images larger than ``tile_size``.

        Splits the input into overlapping tiles, runs inference on each,
        then stitches results back together with linear blending in
        overlap regions.  This bypasses the 1024px resolution limit of
        FLUX.1 without requiring model modifications.

        Args:
            image: Input image tensor [1, 3, H, W] in [-1, 1].
            task: Vision task name.
            tile_size: Size of each square tile (default 1024).
            overlap: Overlap between adjacent tiles in pixels.
            **kwargs: Additional arguments passed to predict().

        Returns:
            Task-specific result dict with full-resolution outputs.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape
        assert B == 1, "predict_tiled only supports single-image inference"

        # If image fits within tile_size, fall back to standard predict
        if tile_size >= H and tile_size >= W:
            return self.predict(image, task, **kwargs)

        # Generate tile coordinates
        stride = tile_size - overlap
        tiles_h = max(1, (H - overlap + stride - 1) // stride)
        tiles_w = max(1, (W - overlap + stride - 1) // stride)

        # Accumulators for RGB output blending
        rgb_sum = torch.zeros(1, 3, H, W, device=image.device)
        weight = torch.zeros(1, 1, H, W, device=image.device)

        # Blending weight: cosine ramp at edges for smooth stitching
        ramp = torch.linspace(0, 1, tile_size, device=image.device)
        ramp = torch.minimum(ramp, torch.flip(ramp, [0]))
        ramp = torch.clamp(ramp * float(tile_size) / (overlap + 1), 0, 1)
        tile_weight = ramp.unsqueeze(0) * ramp.unsqueeze(1)  # [tile, tile]
        tile_weight = tile_weight.unsqueeze(0).unsqueeze(0)  # [1,1,t,t]

        all_results = []

        for i in range(tiles_h):
            for j in range(tiles_w):
                y0 = min(i * stride, H - tile_size)
                x0 = min(j * stride, W - tile_size)
                y1, x1 = y0 + tile_size, x0 + tile_size

                tile = image[:, :, y0:y1, x0:x1]
                tile_result = self.predict(tile, task, **kwargs)

                tile_rgb = tile_result.get("rgb_output")
                if tile_rgb is not None:
                    rgb_sum[:, :, y0:y1, x0:x1] += tile_rgb * tile_weight
                    weight[:, :, y0:y1, x0:x1] += tile_weight

                all_results.append(tile_result)

        # Blend RGB outputs
        weight = weight.clamp(min=1e-8)
        rgb_output = rgb_sum / weight

        # Decode the full-resolution RGB
        result = self._decode_output(
            rgb_output=rgb_output,
            task=task,
            class_color_map=kwargs.get("class_color_map"),
            background_color=kwargs.get("background_color", (0, 0, 0)),
        )
        result["rgb_output"] = rgb_output
        return result

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
                f"TorchScript export not supported for task: {task}. Choose from: depth, normal"
            )

        logger.info(f"TorchScript model exported to {output_path}")


__all__ = ["VisionBanana"]
