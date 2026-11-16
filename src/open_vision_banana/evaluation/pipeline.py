"""Unified evaluation pipeline for multi-task vision models.

Wraps ``BaseGenerator`` (Sana-1600M + LoRA) with the encoding contract
to ensure prompts, colormap decode, and metric computation are all
consistent with training.

Usage::

    pipeline = EvalPipeline(
        checkpoint_path="output/v5_multitask/checkpoint-10500",
        device="cuda",
    )
    metrics = pipeline.evaluate_task("depth", dataloader)
    # or single-image:
    result = pipeline.predict_single(pil_image, task="depth")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .contract import EncodingContract, load_encoding_contract

logger = logging.getLogger(__name__)


class EvalPipeline:
    """Contract-aware evaluation pipeline.

    Loads a Sana-1600M (+ optional LoRA) model via ``BaseGenerator``,
    reads the encoding contract for prompts / colormap / palette, and
    provides unified inference + metric computation for depth, normal,
    and semantic segmentation tasks.

    Args:
        checkpoint_path: Path to a training checkpoint directory containing
            a ``lora/`` sub-directory.  If *None*, the base model is used
            without LoRA adapters.
        contract_path: Path to the encoding contract YAML.  If *None*, the
            default ``configs/encoding_contract.yaml`` is used.
        device: Torch device string (e.g. ``"cuda"``, ``"mps"``).
        dtype: Model dtype.  Defaults to ``float16``.
        num_inference_steps: Default diffusion steps for inference.
        guidance_scale: Default CFG scale.
        strength: Default img2img noise strength.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        contract_path: str | Path | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        strength: float = 0.85,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.strength = strength

        # ── Load encoding contract ────────────────────────────────────
        self.contract: EncodingContract = load_encoding_contract(contract_path)
        logger.info("Loaded encoding contract: depth=%s, semantic=%s",
                     self.contract.depth.colormap, self.contract.semantic.palette_source)

        # ── Load model ────────────────────────────────────────────────
        from ..inference.generator import BaseGenerator

        model_id = self.contract.model.get("model_id",
                                           "Efficient-Large-Model/Sana_1600M_512px_diffusers")
        lora_path = None
        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            # Look for LoRA adapter in standard locations
            for candidate in (
                ckpt / "lora" / "adapter_model.safetensors",
                ckpt / "lora_adapter" / "adapter_model.safetensors",
                ckpt / "adapter" / "adapter_model.safetensors",
            ):
                if candidate.is_file():
                    lora_path = str(candidate.parent.parent)
                    break
            if lora_path is None:
                # Maybe the checkpoint dir itself contains LoRA weights
                for pattern in ("**/adapter_model.safetensors", "**/pytorch_lora_weights.safetensors"):
                    matches = list(ckpt.glob(pattern))
                    if matches:
                        lora_path = str(ckpt)
                        break

        logger.info("Loading model: %s (LoRA: %s)", model_id, lora_path or "none")
        self.generator = BaseGenerator(
            model_name=model_id,
            lora_path=lora_path,
            device=device,
            dtype=dtype,
        )

        # ── Lazy-init decoders ────────────────────────────────────────
        self._depth_mapper = None
        self._normal_mapper = None
        self._seg_decoder = None

    # ─── Properties for lazy-loaded decoders ─────────────────────────────

    @property
    def depth_mapper(self):
        if self._depth_mapper is None:
            from ..core.depth_mapper import DepthRGBMapper
            colormap = self.contract.depth.colormap or "turbo"
            self._depth_mapper = DepthRGBMapper(
                colormap=colormap,
                lam=self.contract.depth.power_lam,
                c=self.contract.depth.power_c,
                curve=self.contract.depth.curve,
                log_alpha=self.contract.depth.log_alpha,
            )
        return self._depth_mapper

    @property
    def normal_mapper(self):
        if self._normal_mapper is None:
            from ..core.normal_mapper import NormalRGBMapper
            self._normal_mapper = NormalRGBMapper()
        return self._normal_mapper

    @property
    def seg_decoder(self):
        if self._seg_decoder is None:
            from ..core.segmentation import SegmentationDecoder
            self._seg_decoder = SegmentationDecoder()
        return self._seg_decoder

    # ─── Prompt resolution ───────────────────────────────────────────────

    def get_prompt(self, task: str) -> str:
        """Get the canonical training prompt for a task.

        Uses the encoding contract to ensure the evaluation prompt matches
        what the model was trained with.
        """
        prompt = self.contract.get_prompt(task)
        if not prompt:
            logger.warning("No contract prompt for task '%s'; using fallback", task)
            fallback = {
                "depth": "Generate a metric depth visualization of this image.",
                "normal": "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
                "semantic": "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
            }
            prompt = fallback.get(task, f"Generate {task} output.")
        return prompt

    # ─── Single-image inference ──────────────────────────────────────────

    @torch.no_grad()
    def predict_single(
        self,
        image: Image.Image | torch.Tensor,
        task: str,
        *,
        prompt: str | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        strength: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run inference on a single image and return decoded output + metrics.

        Args:
            image: Input PIL Image or ``[3, H, W]`` / ``[1, 3, H, W]`` tensor.
            task: One of ``"depth"``, ``"normal"``, ``"semantic"``.
            prompt: Override the contract prompt.  If *None*, the canonical
                training prompt from the encoding contract is used.
            num_inference_steps: Override default diffusion steps.
            guidance_scale: Override default CFG scale.
            strength: Override default img2img strength.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary with keys:
            - ``"rgb_output"``: Raw RGB tensor ``[1, 3, H, W]`` in ``[0, 1]``.
            - ``"decoded"``: Task-specific decodedoded output (depth map, normal map,
              or segmentation labels).
            - ``"prompt"``: The prompt used for inference.
            - ``"task"``: Task name.
        """
        if prompt is None:
            prompt = self.get_prompt(task)

        # Convert PIL → tensor if needed
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

        # Ensure image is on the right device and in [-1, 1] for the generator
        image_tensor = image_tensor.to(device=self.device, dtype=self.dtype)
        image_for_gen = image_tensor * 2.0 - 1.0 if image_tensor.min() >= 0 else image_tensor

        # Generate
        rgb_output = self.generator.generate(
            prompt=prompt,
            image=image_for_gen,
            num_inference_steps=num_inference_steps or self.num_inference_steps,
            guidance_scale=guidance_scale or self.guidance_scale,
            strength=strength or self.strength,
            seed=seed,
        )

        # Decode based on task
        decoded = self._decode(rgb_output, task)

        return {
            "rgb_output": rgb_output,
            "decoded": decoded,
            "prompt": prompt,
            "task": task,
        }

    # ─── Dataset-level evaluation ────────────────────────────────────────

    @torch.no_grad()
    def evaluate_task(
        self,
        task: str,
        dataloader,
        *,
        max_batches: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        strength: float | None = None,
    ) -> dict[str, Any]:
        """Evaluate a task over a full dataset.

        The dataloader should yield batches with keys:
        - ``"image"``: Input images ``[B, 3, H, W]`` in ``[-1, 1]`` or ``[0, 1]``.
        - ``"target_rgb"``: Target RGB visualization ``[B, 3, H, W]`` in ``[0, 1]``.
        - For semantic: ``"class_color_map"`` (optional, falls back to contract palette).

        Args:
            task: Task name.
            dataloader: Iterable producing batches.
            max_batches: Limit evaluation to N batches.
            num_inference_steps: Override default steps.
            guidance_scale: Override default CFG.
            strength: Override default strength.

        Returns:
            Dictionary of averaged metrics.
        """
        from .depth import evaluate_depth
        from .normal import evaluate_normals
        from .semantic import evaluate_semantic

        all_metrics: list[dict] = []
        prompt = self.get_prompt(task)

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            image = batch["image"].to(self.device, dtype=self.dtype)
            target_rgb = batch["target_rgb"].to(self.device, dtype=self.dtype)

            # Normalize image to [-1, 1] for generator
            image_for_gen = image * 2.0 - 1.0 if image.min() >= 0 else image

            # Generate predictions
            rgb_output = self.generator.generate(
                prompt=prompt,
                image=image_for_gen,
                num_inference_steps=num_inference_steps or self.num_inference_steps,
                guidance_scale=guidance_scale or self.guidance_scale,
                strength=strength or self.strength,
            )

            # Compute task-specific metrics
            if task == "depth":
                pred_depth = self.depth_mapper.rgb_to_depth(rgb_output)
                gt_depth = self.depth_mapper.rgb_to_depth(target_rgb)
                metrics = evaluate_depth(
                    pred_depth, gt_depth,
                    colormap=self.contract.depth.colormap,
                )

            elif task == "normal":
                pred_normal = self.normal_mapper.rgb_to_normal(rgb_output)
                gt_normal = self.normal_mapper.rgb_to_normal(target_rgb)
                metrics = evaluate_normals(pred_normal, gt_normal)

            elif task == "semantic":
                class_color_map = batch.get("class_color_map")
                if class_color_map is None:
                    class_color_map = self.contract.semantic.load_palette()
                pred_labels = self.seg_decoder.decode_semantic(rgb_output, class_color_map)
                gt_labels = self.seg_decoder.decode_semantic(target_rgb, class_color_map)
                n_classes = self.contract.semantic.n_classes
                metrics = evaluate_semantic(pred_labels, gt_labels, n_classes)

            else:
                raise ValueError(f"Unsupported task for EvalPipeline: {task}")

            all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Average metrics across batches
        avg_metrics: dict[str, Any] = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if key in m and isinstance(m[key], (int, float))]
            if values:
                avg_metrics[key] = float(np.mean(values))

        avg_metrics["n_batches"] = len(all_metrics)
        return avg_metrics

    # ─── Internal decode ─────────────────────────────────────────────────

    def _decode(self, rgb_output: torch.Tensor, task: str) -> torch.Tensor:
        """Decode raw RGB output to task-specific representation."""
        if task == "depth":
            return self.depth_mapper.rgb_to_depth(rgb_output)
        elif task == "normal":
            return self.normal_mapper.rgb_to_normal(rgb_output)
        elif task == "semantic":
            palette = self.contract.semantic.load_palette()
            return self.seg_decoder.decode_semantic(rgb_output, palette)
        else:
            # Return raw RGB for unknown tasks
            return rgb_output


__all__ = ["EvalPipeline"]
