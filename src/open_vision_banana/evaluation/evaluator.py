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
    ) -> dict[str, float]:
        """Run evaluation on a dataset.

        Args:
            dataloader: DataLoader producing (image, target) pairs.
            task: One of "depth", "normal", "semantic", "instance", "referring".
            num_classes: Required for semantic segmentation.
            max_batches: Limit evaluation to N batches.

        Returns:
            Dictionary of metric name → value.
        """
        self.model.eval()
        all_metrics = {}

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            image = batch["image"].to(self.device)
            target_rgb = batch["target_rgb"].to(self.device)
            pred_rgb = self.model(image)

            if task == "depth":
                mapper = DepthRGBMapper()
                pred_depth = mapper.rgb_to_depth(pred_rgb)
                gt_depth = mapper.rgb_to_depth(target_rgb)
                metrics = evaluate_depth(pred_depth, gt_depth)

            elif task == "normal":
                mapper = NormalRGBMapper()
                pred_normal = mapper.rgb_to_normal(pred_rgb)
                gt_normal = mapper.rgb_to_normal(target_rgb)
                metrics = evaluate_normals(pred_normal, gt_normal)

            elif task == "semantic":
                decoder = SegmentationDecoder()
                pred_labels = decoder.decode_semantic(pred_rgb, {})
                gt_labels = decoder.decode_semantic(target_rgb, {})
                metrics = evaluate_semantic(pred_labels, gt_labels, num_classes or 19)

            elif task == "instance":
                decoder = SegmentationDecoder()
                pred_inst = decoder.decode_instance(pred_rgb)[0]
                gt_inst = decoder.decode_instance(target_rgb)[0]
                metrics = evaluate_instance(pred_inst, gt_inst)

            elif task == "referring":
                metrics = evaluate_referring(
                    pred_rgb.mean(dim=1, keepdim=False) > 0.5,
                    target_rgb.mean(dim=1, keepdim=False) > 0.5,
                )

            else:
                raise ValueError(f"Unknown task: {task}")

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k] = all_metrics.get(k, 0.0) + v

        n_batches = batch_idx + 1
        for k in all_metrics:
            all_metrics[k] /= n_batches

        return all_metrics


__all__ = ["VisionEvaluator"]
