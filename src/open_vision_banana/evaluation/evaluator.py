"""Unified evaluation runner for all vision tasks.

Handles model inference + metric computation for:
- Depth estimation
- Normal estimation
- Semantic segmentation
- Instance segmentation
- Referring expression segmentation
"""

from __future__ import annotations

import logging

import torch

from ..core.depth_mapper import DepthRGBMapper
from ..core.normal_mapper import NormalRGBMapper
from ..core.segmentation import SegmentationDecoder
from .depth import evaluate_depth
from .instance import evaluate_instance
from .normal import evaluate_normals
from .referring import evaluate_referring
from .semantic import evaluate_semantic

logger = logging.getLogger(__name__)


class VisionEvaluator:
    """Unified evaluator for all vision tasks.

    Handles model inference + metric computation.
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        task: str,
        num_classes: int | None = None,
        max_batches: int | None = None,
        class_color_map: dict | None = None,
    ) -> dict[str, float]:
        """Run evaluation on a dataset.

        Args:
            dataloader: DataLoader producing batches with 'image' and 'target_rgb' keys.
            task: One of "depth", "normal", "semantic", "instance", "referring".
            num_classes: Required for semantic segmentation (defaults to 19 for Cityscapes).
            max_batches: Limit evaluation to N batches (useful for quick sanity checks).
            class_color_map: Required for semantic segmentation decoding. Must be a dict
                mapping class name → (R, G, B) tuple. If None and task == "semantic",
                will attempt to read from batch["class_color_map"].

        Returns:
            Dictionary of metric name → value (averaged over all evaluated batches).
        """
        self.model.eval()
        all_metrics: dict[str, float] = {}
        n_batches = 0

        # Pre-instantiate shared decoders / mappers once (not per batch)
        depth_mapper = DepthRGBMapper() if task == "depth" else None
        normal_mapper = NormalRGBMapper() if task == "normal" else None
        seg_decoder = SegmentationDecoder() if task in ("semantic", "instance") else None

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            image = batch["image"].to(self.device)
            target_rgb = batch["target_rgb"].to(self.device)
            pred_rgb = self.model(image)

            if task == "depth":
                pred_depth = depth_mapper.rgb_to_depth(pred_rgb)
                gt_depth = depth_mapper.rgb_to_depth(target_rgb)
                metrics = evaluate_depth(pred_depth, gt_depth)

            elif task == "normal":
                pred_normal = normal_mapper.rgb_to_normal(pred_rgb)
                gt_normal = normal_mapper.rgb_to_normal(target_rgb)
                metrics = evaluate_normals(pred_normal, gt_normal)

            elif task == "semantic":
                # Resolve class_color_map: argument > batch field > error
                cmap = class_color_map or batch.get("class_color_map")
                if not cmap:
                    raise ValueError(
                        "class_color_map is required for semantic evaluation. "
                        "Pass it as an argument or include it in the dataloader batch."
                    )
                pred_labels = seg_decoder.decode_semantic(pred_rgb, cmap)
                gt_labels = seg_decoder.decode_semantic(target_rgb, cmap)
                metrics = evaluate_semantic(pred_labels, gt_labels, num_classes or len(cmap))

            elif task == "instance":
                pred_inst = seg_decoder.decode_instance(pred_rgb)[0]
                gt_inst = seg_decoder.decode_instance(target_rgb)[0]
                metrics = evaluate_instance(pred_inst, gt_inst)

            elif task == "referring":
                # Binary mask: target is white (all channels > 0.5) vs background
                pred_mask = pred_rgb.mean(dim=1) > 0.5
                gt_mask = target_rgb.mean(dim=1) > 0.5
                metrics = evaluate_referring(pred_mask, gt_mask)

            else:
                raise ValueError(f"Unknown task: {task!r}. Choose from: depth, normal, semantic, instance, referring")

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k] = all_metrics.get(k, 0.0) + v

            n_batches += 1

        if n_batches == 0:
            logger.warning("evaluate() called on an empty dataloader — returning empty metrics")
            return {}

        for k in all_metrics:
            all_metrics[k] /= n_batches

        return all_metrics


__all__ = ["VisionEvaluator"]
