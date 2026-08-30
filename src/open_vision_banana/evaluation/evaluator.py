"""Unified evaluation runner for all vision tasks.

Handles model inference + metric computation for all 10 supported tasks:
- Depth estimation
- Normal estimation
- Semantic segmentation
- Instance segmentation
- Referring expression segmentation
- Edge detection
- Optical flow
- 2D keypoint detection
- Object detection
- Depth edge detection
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from ..core.depth_mapper import DepthRGBMapper
from ..core.flow_mapper import FlowRGBMapper
from ..core.keypoint_mapper import KeypointRGBMapper
from ..core.normal_mapper import NormalRGBMapper
from ..core.object_detection_mapper import ObjectDetectionRGBMapper
from ..core.segmentation import SegmentationDecoder
from .benchmark import (
    DepthAccumulator,
    NormalAccumulator,
    SemanticAccumulator,
    apply_depth_benchmark_protocol,
)
from .depth import evaluate_depth
from .edge import evaluate_edge
from .flow import evaluate_flow
from .generation_preservation import evaluate_generation_preservation
from .instance import SACoAccumulator, evaluate_instance
from .keypoint import evaluate_keypoints
from .normal import evaluate_normals
from .object_detection import evaluate_object_detection
from .referring import ReferringAccumulator, evaluate_referring
from .semantic import evaluate_semantic

logger = logging.getLogger(__name__)


def _batch_bool(value: Any, index: int, default: bool) -> bool:
    """Read a scalar or batched presence label without constraining collate type."""
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        flat = value.reshape(-1)
        return bool(flat[min(index, flat.numel() - 1)].item())
    if isinstance(value, (list, tuple)):
        return bool(value[min(index, len(value) - 1)])
    return bool(value)


class VisionEvaluator:
    """Unified evaluator for all vision tasks.

    Handles model inference + metric computation.

    The evaluator routes inference through the full ``VisionBanana.predict()``
    pipeline (prompt building → diffusion generation → RGB decoding) when the
    model exposes that API.  This ensures that evaluation metrics reflect the
    same code path used during actual deployment.

    For models that do NOT implement ``predict()`` (e.g. lightweight mocks in
    unit tests), the evaluator falls back to a direct ``model(image)`` call
    and decodes the output with the shared decoders.  A warning is logged
    whenever the fallback path is taken.
    """

    def __init__(self, model, device: str = "cuda", strict_nonfinite: bool = True):
        self.model = model
        self.device = device
        self.strict_nonfinite = bool(strict_nonfinite)
        # Detect whether the model exposes the VisionBanana.predict() API.
        self._has_predict = callable(getattr(model, "predict", None))
        if not self._has_predict:
            logger.warning(
                "Model does not expose a `predict()` method. "
                "Falling back to direct model(image) call — "
                "evaluation results may not reflect the full inference pipeline."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_batch(
        self,
        image: torch.Tensor,
        target_rgb: torch.Tensor,
        task: str,
        class_color_map: dict | None = None,
        object_name: str | None = None,
        expression: str | None = None,
        num_inference_steps: int = 1,
        strength: float = 0.85,
        seed: int | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a single batch without iterating a DataLoader.

        Convenience wrapper around ``evaluate()`` for ad-hoc batch evaluation.
        Useful for interactive evaluation, debugging, or custom data pipelines
        that do not use a DataLoader.

        Args:
            image: Input images [B, 3, H, W] in [-1, 1].
            target_rgb: Target RGB visualizations [B, 3, H, W] in [0, 1].
            task: One of the 10 supported tasks.
            class_color_map: Required for semantic segmentation.
            object_name: Forwarded to predict() for instance segmentation.
            expression: Forwarded to predict() for referring segmentation.
            num_inference_steps: Diffusion steps forwarded to predict().
            strength: img2img noise strength forwarded to predict().

        Returns:
            Dictionary of metric name to value for this batch.
        """
        batch: dict[str, Any] = {
            "image": image.to(self.device),
            "target_rgb": target_rgb.to(self.device),
        }
        if class_color_map is not None:
            batch["class_color_map"] = class_color_map

        # Build a single-batch dataloader-like iterable
        single_batch_loader = [batch]

        metrics = self.evaluate(
            dataloader=single_batch_loader,
            task=task,
            class_color_map=class_color_map,
            object_name=object_name,
            expression=expression,
            num_inference_steps=num_inference_steps,
            strength=strength,
            seed=seed,
            dataset_name=dataset_name,
            max_batches=1,
        )
        # Preserve the convenience API's historical scalar-float contract;
        # dataset-level evaluation keeps integer sample counts intact.
        return {
            key: float(value) if isinstance(value, int | float) else value
            for key, value in metrics.items()
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        task: str,
        num_classes: int | None = None,
        max_batches: int | None = None,
        class_color_map: dict | None = None,
        # VisionBanana.predict() extra kwargs forwarded verbatim
        object_name: str | None = None,
        expression: str | None = None,
        num_inference_steps: int = 1,
        strength: float = 0.85,
        seed: int | None = None,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Run evaluation on a dataset.

        Supports all 10 vision tasks:
        ``depth``, ``normal``, ``semantic``, ``instance``, ``referring``,
        ``edge``, ``flow``, ``keypoint``, ``object_detection``, ``depth_edge``.

        Args:
            dataloader: DataLoader producing batches with 'image' and
                'target_rgb' keys.  For semantic segmentation the batch
                may also carry a 'class_color_map' entry. For keypoint and
                object detection tasks, additional keys are expected.
            task: One of the 10 supported vision tasks.
            num_classes: Required for semantic segmentation (defaults to
                19 for Cityscapes, or len(class_color_map) if provided).
            max_batches: Limit evaluation to N batches (useful for quick
                sanity checks).
            class_color_map: Required for semantic segmentation decoding.
                Must be a dict mapping class name → (R, G, B) tuple. If
                None and task == "semantic", will attempt to read from
                batch["class_color_map"].
            object_name: Forwarded to predict() for instance segmentation.
            expression: Forwarded to predict() for referring segmentation.
            num_inference_steps: Diffusion steps forwarded to predict().
            strength: img2img noise strength forwarded to predict().

        Returns:
            Dictionary of metric name → value (averaged over all batches).
        """
        self.model.eval()
        all_metrics: dict[str, Any] = {}
        n_batches = 0
        gen_real_rgb: list[torch.Tensor] = []
        gen_pred_rgb: list[torch.Tensor] = []
        depth_accumulator = DepthAccumulator() if task == "depth" else None
        normal_accumulator = NormalAccumulator() if task == "normal" else None
        semantic_accumulator = (
            SemanticAccumulator(num_classes if num_classes is not None else len(class_color_map or {}))
            if task == "semantic"
            else None
        )
        instance_accumulator = SACoAccumulator() if task == "instance" else None
        referring_accumulator = ReferringAccumulator() if task == "referring" else None

        # Pre-instantiate shared decoders once (not per batch)
        depth_mapper = DepthRGBMapper() if task in ("depth", "depth_edge") else None
        normal_mapper = NormalRGBMapper() if task == "normal" else None
        seg_decoder = SegmentationDecoder() if task in ("semantic", "instance") else None

        skipped_samples = 0
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            image = batch["image"].to(self.device)
            target_rgb = batch["target_rgb"].to(self.device)

            # ── Run model inference ───────────────────────────────────────
            try:
                pred_rgb = self._run_inference(
                    image=image,
                    task=task,
                    class_color_map=class_color_map or batch.get("class_color_map"),
                    object_name=object_name,
                    expression=expression or batch.get("expression"),
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    seed=None if seed is None else seed + batch_idx,
                )
            except ValueError as exc:
                if "NaN or Inf" not in str(exc):
                    raise
                if self.strict_nonfinite:
                    raise RuntimeError(
                        f"Non-finite generator output at evaluation batch {batch_idx}"
                    ) from exc
                skipped = int(image.shape[0]) if image.dim() == 4 else 1
                skipped_samples += skipped
                logger.warning("Skipping batch %d (%d sample(s)): %s", batch_idx, skipped, exc)
                continue

            # ── Decode & compute metrics ──────────────────────────────────
            metrics: dict[str, Any]
            if task == "depth":
                pred_depth = depth_mapper.rgb_to_depth(pred_rgb)
                gt_depth = batch.get("gt_depth")
                if gt_depth is None:
                    gt_depth = depth_mapper.rgb_to_depth(target_rgb)
                else:
                    gt_depth = gt_depth.to(self.device)
                valid_mask = batch.get("valid_mask")
                if valid_mask is not None:
                    valid_mask = valid_mask.to(self.device)
                pred_depth, gt_depth, valid_mask = apply_depth_benchmark_protocol(
                    dataset_name or "", pred_depth, gt_depth, valid_mask
                )
                if depth_accumulator is not None:
                    depth_accumulator.update(pred_depth, gt_depth, valid_mask)
                metrics = evaluate_depth(pred_depth, gt_depth, valid_mask)

            elif task == "normal":
                pred_normal = normal_mapper.rgb_to_normal(pred_rgb)
                gt_normal = batch.get("gt_normal")
                if gt_normal is None:
                    gt_normal = normal_mapper.rgb_to_normal(target_rgb)
                else:
                    gt_normal = gt_normal.to(self.device)
                valid_mask = batch.get("valid_mask")
                if normal_accumulator is not None:
                    normal_accumulator.update(pred_normal, gt_normal, valid_mask)
                metrics = evaluate_normals(pred_normal, gt_normal)

            elif task == "semantic":
                cmap = class_color_map or batch.get("class_color_map")
                if not cmap:
                    raise ValueError(
                        "class_color_map is required for semantic evaluation. "
                        "Pass it as an argument or include it in the dataloader batch."
                    )
                pred_labels = seg_decoder.decode_semantic(pred_rgb, cmap)
                gt_labels = batch.get("gt_labels")
                if gt_labels is None:
                    gt_labels = seg_decoder.decode_semantic(target_rgb, cmap)
                else:
                    gt_labels = gt_labels.to(self.device)
                if semantic_accumulator is not None:
                    semantic_accumulator.update(pred_labels, gt_labels)
                metrics = evaluate_semantic(pred_labels, gt_labels, num_classes if num_classes is not None else len(cmap))

            elif task == "instance":
                pred_batch = seg_decoder.decode_instance(pred_rgb)
                gt_batch = seg_decoder.decode_instance(target_rgb)
                if len(pred_batch) != len(gt_batch):
                    raise RuntimeError("Predicted and target instance batches have different sizes")

                per_query_metrics = []
                for query_idx, (pred_inst, gt_inst) in enumerate(
                    zip(pred_batch, gt_batch, strict=True)
                ):
                    pred_present = _batch_bool(
                        batch.get("pred_present"), query_idx, bool(pred_inst)
                    )
                    gt_present = _batch_bool(
                        batch.get("gt_present"), query_idx, bool(gt_inst)
                    )
                    gated_pred = pred_inst if pred_present else []
                    if instance_accumulator is not None:
                        instance_accumulator.update(
                            gated_pred,
                            gt_inst,
                            pred_present=pred_present,
                            gt_present=gt_present,
                        )
                    per_query_metrics.append(evaluate_instance(gated_pred, gt_inst))

                metrics = {
                    key: sum(item[key] for item in per_query_metrics) / len(per_query_metrics)
                    for key in per_query_metrics[0]
                } if per_query_metrics else {}

            elif task == "referring":
                pred_mask = pred_rgb.mean(dim=1) > 0.5
                gt_mask = target_rgb.mean(dim=1) > 0.5
                if referring_accumulator is not None:
                    referring_accumulator.update(pred_mask, gt_mask)
                metrics = evaluate_referring(pred_mask, gt_mask)

            elif task == "edge":
                # Edge detection: treat RGB as edge probability (grayscale)
                pred_edge = pred_rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                gt_edge = target_rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                metrics = evaluate_edge(pred_edge, gt_edge, compute_ods=True)

            elif task == "depth_edge":
                # Depth edge: same as edge but with depth-specific interpretation
                pred_edge = depth_mapper.rgb_to_depth(pred_rgb)
                gt_edge = depth_mapper.rgb_to_depth(target_rgb)
                # Compute edge from depth discontinuity (simplified: treat as edge map)
                pred_edge = pred_edge / (pred_edge.max() + 1e-10)
                gt_edge = gt_edge / (gt_edge.max() + 1e-10)
                metrics = evaluate_edge(pred_edge, gt_edge, compute_ods=True)

            elif task == "flow":
                # Optical flow: decode HSV-style encoding
                # Vision Banana encodes flow as HSV: direction=hue, magnitude=saturation
                # Simplified: treat RGB channels as (u, v, valid) proxy
                pred_flow = self._decode_flow_rgb(pred_rgb)
                gt_flow = self._decode_flow_rgb(target_rgb)
                metrics = evaluate_flow(pred_flow, gt_flow)

            elif task == "keypoint":
                if "pred_keypoints" in batch:
                    pred_kpts = batch["pred_keypoints"]
                else:
                    pred_kpts, _ = KeypointRGBMapper().rgb_to_keypoints(pred_rgb)
                gt_kpts = batch.get("gt_keypoints")
                if gt_kpts is None:
                    raise ValueError(
                        "gt_keypoints is required for keypoint evaluation. "
                        "Include it in the dataloader batch as 'gt_keypoints'."
                    )
                areas = batch.get("areas")
                metrics = evaluate_keypoints(pred_kpts, gt_kpts, areas=areas)

            elif task == "object_detection":
                gt_boxes = batch.get("gt_boxes", [])
                gt_labels = batch.get("gt_labels", [])
                if not gt_boxes:
                    raise ValueError(
                        "gt_boxes and gt_labels are required for object detection evaluation."
                    )
                pred_boxes: Any
                pred_scores: Any
                pred_labels: Any  # type: ignore[no-redef]
                if batch.get("pred_boxes"):
                    pred_boxes = batch["pred_boxes"]
                    pred_scores = batch["pred_scores"]
                    pred_labels = batch["pred_labels"]
                else:
                    bboxes, class_ids, scores = ObjectDetectionRGBMapper().rgb_to_bboxes(pred_rgb)
                    pred_boxes, pred_scores, pred_labels = self._decode_detection_batch(  # type: ignore[assignment]
                        bboxes, class_ids, scores
                    )
                metrics = evaluate_object_detection(
                    pred_boxes, pred_scores, pred_labels,
                    gt_boxes, gt_labels,
                    num_classes=num_classes if num_classes is not None else (len(class_color_map) if class_color_map else 80),
                )

            elif task == "generation_preservation":
                gen_real_rgb.append(target_rgb.detach().cpu())
                gen_pred_rgb.append(pred_rgb.detach().cpu())
                metrics = {}

            else:
                raise ValueError(
                    f"Unknown task: {task!r}. "
                    "Choose from: depth, normal, semantic, instance, referring, "
                    "edge, depth_edge, flow, keypoint, object_detection, "
                    "generation_preservation"
                )

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics[k] = all_metrics.get(k, 0.0) + v
                elif isinstance(v, list):
                    # Collect lists for per-element averaging (handles nan correctly)
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
                elif isinstance(v, dict):
                    # Collect dicts for per-key averaging
                    if k not in all_metrics:
                        all_metrics[k] = {}
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            all_metrics[k][sub_k] = all_metrics[k].get(sub_k, 0.0) + sub_v

            n_batches += 1

        if task == "generation_preservation":
            if not gen_real_rgb:
                logger.warning("No batches evaluated for generation_preservation")
                return {}
            real_rgb = torch.cat(gen_real_rgb, dim=0)
            pred_rgb_all = torch.cat(gen_pred_rgb, dim=0)
            return evaluate_generation_preservation(
                real_rgb=real_rgb,
                gen_rgb=pred_rgb_all,
                pred_rgb=pred_rgb_all,
                target_rgb=real_rgb,
            )

        if n_batches == 0:
            logger.warning("evaluate() called on an empty dataloader — returning empty metrics")
            return {}

        if task == "depth" and depth_accumulator is not None:
            metrics = depth_accumulator.compute()
            metrics["skipped_samples"] = skipped_samples
            return metrics
        if task == "normal" and normal_accumulator is not None:
            return normal_accumulator.compute()
        if task == "semantic" and semantic_accumulator is not None:
            return semantic_accumulator.compute()
        if task == "referring" and referring_accumulator is not None:
            return referring_accumulator.compute()

        for k in all_metrics:
            metric_value = all_metrics[k]
            if isinstance(metric_value, list):
                # Per-element mean, ignoring nan
                len(metric_value)
                elem_len = len(metric_value[0])
                result = []
                for i in range(elem_len):
                    vals = [
                        batch[i]
                        for batch in metric_value
                        if not (isinstance(batch[i], float) and math.isnan(batch[i]))
                    ]
                    result.append(sum(vals) / len(vals) if vals else float("nan"))
                all_metrics[k] = result
            elif isinstance(metric_value, dict):
                for sub_k in metric_value:
                    if isinstance(metric_value[sub_k], (int, float)):
                        metric_value[sub_k] /= n_batches
            else:
                all_metrics[k] /= n_batches

        if task == "instance" and instance_accumulator is not None:
            all_metrics.update(instance_accumulator.compute())

        return all_metrics

    @staticmethod
    def _decode_flow_rgb(rgb: torch.Tensor, max_flow: float = 20.0) -> torch.Tensor:
        """Decode HSV-style flow encoding using the core FlowRGBMapper.

        Args:
            rgb: [B, 3, H, W] RGB tensor in [0, 1].
            max_flow: Maximum flow magnitude used during encoding.

        Returns:
            Flow tensor [B, 2, H, W] with (u, v) components.
        """
        return FlowRGBMapper(max_flow=max_flow).rgb_to_flow(rgb)

    @staticmethod
    def _decode_detection_batch(
        bboxes: torch.Tensor,
        class_ids: torch.Tensor,
        scores: torch.Tensor,
        score_threshold: float = 0.05,
    ) -> tuple[list, list, list]:
        """Convert decoded tensors to per-image lists for evaluate_object_detection."""
        pred_boxes: list[list[tuple[float, float, float, float]]] = []
        pred_scores: list[list[float]] = []
        pred_labels: list[list[int]] = []

        B = bboxes.shape[0]
        for b in range(B):
            boxes_b: list[tuple[float, float, float, float]] = []
            scores_b: list[float] = []
            labels_b: list[int] = []
            for m in range(bboxes.shape[1]):
                score = float(scores[b, m].item())
                if score < score_threshold:
                    continue
                x, y, w, h = bboxes[b, m].tolist()
                if w <= 0 or h <= 0:
                    continue
                boxes_b.append((x, y, x + w, y + h))
                scores_b.append(score)
                labels_b.append(int(class_ids[b, m].item()))
            pred_boxes.append(boxes_b)
            pred_scores.append(scores_b)
            pred_labels.append(labels_b)
        return pred_boxes, pred_scores, pred_labels

    @staticmethod
    def reduce_metrics(metrics: dict, device: torch.device | str = "cpu") -> dict:
        """All-reduce scalar metrics across DDP ranks (P2-27)."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return metrics

        reduced: dict = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                t = torch.tensor([float(value)], device=device, dtype=torch.float64)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                world = torch.distributed.get_world_size()
                reduced[key] = (t.item() / world)
            elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                arr = torch.tensor(value, device=device, dtype=torch.float64)
                torch.distributed.all_reduce(arr, op=torch.distributed.ReduceOp.SUM)
                reduced[key] = (arr / torch.distributed.get_world_size()).tolist()
            else:
                reduced[key] = value
        return reduced

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        image: torch.Tensor,
        task: str,
        class_color_map: dict | None = None,
        object_name: str | None = None,
        expression: str | None = None,
        num_inference_steps: int = 1,
        strength: float = 0.85,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Run model inference and return a [B, 3, H, W] RGB tensor in [0, 1].

        When the model exposes ``predict()`` (VisionBanana API) the full
        pipeline — prompt building, diffusion generation, RGB output — is
        used.  Otherwise a direct ``model(image)`` call is made as a
        fallback.
        """
        if self._has_predict:
            # Build predict() kwargs based on task requirements
            predict_kwargs: dict = dict(
                image=image,
                task=task,
                num_inference_steps=num_inference_steps,
                strength=strength,
                seed=seed,
            )
            if task == "semantic":
                if class_color_map is None:
                    raise ValueError("class_color_map required for semantic segmentation predict()")
                predict_kwargs["class_color_map"] = class_color_map
            elif task == "instance":
                predict_kwargs["object_name"] = object_name or "object"
            elif task == "referring":
                if expression is None:
                    raise ValueError("expression required for referring segmentation predict()")
                predict_kwargs["expression"] = expression
            elif task in ("edge", "depth_edge", "flow", "keypoint", "object_detection"):
                # These tasks require no additional predict kwargs beyond task name
                pass

            result = self.model.predict(**predict_kwargs)
            # predict() returns a dict; extract the RGB output tensor
            pred_rgb = result.get("rgb_output")
            if pred_rgb is None:
                # Older pipeline versions may not include rgb_output; fall back
                # to re-encoding the decoded output.
                logger.debug(
                    "predict() result missing 'rgb_output'; attempting to extract "
                    "from task-specific keys."
                )
                pred_rgb = self._extract_rgb_from_result(result, task, image)
            pred_rgb = pred_rgb.to(self.device)
            if not torch.isfinite(pred_rgb).all():
                raise ValueError("generator output contains NaN or Inf values")
            return pred_rgb

        # ── Fallback: direct model(image) call ───────────────────────────
        pred_rgb = self.model(image)
        if isinstance(pred_rgb, dict):
            pred_rgb = pred_rgb.get("rgb_output", next(iter(pred_rgb.values())))
        pred_rgb = pred_rgb.to(self.device)
        if not torch.isfinite(pred_rgb).all():
            raise ValueError("generator output contains NaN or Inf values")
        return pred_rgb

    @staticmethod
    def _extract_rgb_from_result(result: dict, task: str, image: torch.Tensor) -> torch.Tensor:
        """Best-effort extraction of a displayable [B,3,H,W] tensor from a result dict."""
        if "visualization" in result:
            return result["visualization"]
        # For depth/normal the decoded tensor can be re-encoded to a display image
        if task == "depth" and "depth" in result:
            from ..core.depth_mapper import DepthRGBMapper

            return DepthRGBMapper().depth_to_rgb(result["depth"])
        if task == "normal" and "normal" in result:
            from ..core.normal_mapper import NormalRGBMapper

            return NormalRGBMapper().normal_to_rgb(result["normal"])
        # For segmentation tasks the mask is binary — return it as a grey image
        if "mask" in result:
            mask = result["mask"].float()
            if mask.dim() == 4 and mask.shape[1] == 1:
                return mask.expand(-1, 3, -1, -1)
        # Last resort: return a zero tensor matching the input spatial dims
        logger.warning(
            "Could not extract rgb_output from predict() result; "
            "returning zero tensor for this batch."
        )
        B, _, H, W = image.shape
        return torch.zeros(B, 3, H, W, device=image.device)


__all__ = ["VisionEvaluator"]
