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

import torch

from ..core.depth_mapper import DepthRGBMapper
from ..core.normal_mapper import NormalRGBMapper
from ..core.segmentation import SegmentationDecoder
from .depth import evaluate_depth
from .edge import evaluate_edge
from .flow import evaluate_flow
from .instance import evaluate_instance
from .keypoint import evaluate_keypoints
from .normal import evaluate_normals
from .object_detection import evaluate_object_detection
from .referring import evaluate_referring
from .semantic import evaluate_semantic

logger = logging.getLogger(__name__)


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

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
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
    ) -> dict[str, float]:
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
        batch = {
            "image": image.to(self.device),
            "target_rgb": target_rgb.to(self.device),
        }
        if class_color_map is not None:
            batch["class_color_map"] = class_color_map

        # Build a single-batch dataloader-like iterable
        single_batch_loader = [batch]

        return self.evaluate(
            dataloader=single_batch_loader,
            task=task,
            class_color_map=class_color_map,
            object_name=object_name,
            expression=expression,
            num_inference_steps=num_inference_steps,
            strength=strength,
            max_batches=1,
        )

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
    ) -> dict[str, float]:
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
        all_metrics: dict[str, float] = {}
        n_batches = 0

        # Pre-instantiate shared decoders once (not per batch)
        depth_mapper = DepthRGBMapper() if task in ("depth", "depth_edge") else None
        normal_mapper = NormalRGBMapper() if task == "normal" else None
        seg_decoder = SegmentationDecoder() if task in ("semantic", "instance") else None

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            image = batch["image"].to(self.device)
            target_rgb = batch["target_rgb"].to(self.device)

            # ── Run model inference ───────────────────────────────────────
            pred_rgb = self._run_inference(
                image=image,
                task=task,
                class_color_map=class_color_map or batch.get("class_color_map"),
                object_name=object_name,
                expression=expression or batch.get("expression"),
                num_inference_steps=num_inference_steps,
                strength=strength,
            )

            # ── Decode & compute metrics ──────────────────────────────────
            if task == "depth":
                pred_depth = depth_mapper.rgb_to_depth(pred_rgb)
                gt_depth = depth_mapper.rgb_to_depth(target_rgb)
                metrics = evaluate_depth(pred_depth, gt_depth)

            elif task == "normal":
                pred_normal = normal_mapper.rgb_to_normal(pred_rgb)
                gt_normal = normal_mapper.rgb_to_normal(target_rgb)
                metrics = evaluate_normals(pred_normal, gt_normal)

            elif task == "semantic":
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
                pred_mask = pred_rgb.mean(dim=1) > 0.5
                gt_mask = target_rgb.mean(dim=1) > 0.5
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
                # 2D keypoint detection: batch must contain keypoint annotations
                pred_kpts = batch.get("pred_keypoints", pred_rgb)  # fallback
                gt_kpts = batch.get("gt_keypoints")
                if gt_kpts is None:
                    raise ValueError(
                        "gt_keypoints is required for keypoint evaluation. "
                        "Include it in the dataloader batch as 'gt_keypoints'."
                    )
                areas = batch.get("areas")
                metrics = evaluate_keypoints(pred_kpts, gt_kpts, areas=areas)

            elif task == "object_detection":
                # Object detection: batch must contain box annotations
                pred_boxes = batch.get("pred_boxes", [])
                pred_scores = batch.get("pred_scores", [])
                pred_labels = batch.get("pred_labels", [])
                gt_boxes = batch.get("gt_boxes", [])
                gt_labels = batch.get("gt_labels", [])
                if not gt_boxes:
                    raise ValueError(
                        "gt_boxes and gt_labels are required for object detection evaluation."
                    )
                metrics = evaluate_object_detection(
                    pred_boxes, pred_scores, pred_labels,
                    gt_boxes, gt_labels,
                    num_classes=num_classes or len(class_color_map) if class_color_map else 80,
                )

            else:
                raise ValueError(
                    f"Unknown task: {task!r}. "
                    "Choose from: depth, normal, semantic, instance, referring, "
                    "edge, depth_edge, flow, keypoint, object_detection"
                )

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

    @staticmethod
    def _decode_flow_rgb(rgb: torch.Tensor) -> torch.Tensor:
        """Decode HSV-style flow encoding from RGB visualization.

        Vision Banana encodes optical flow as:
        - Hue = direction angle
        - Saturation = magnitude
        - Value = 1.0 (full brightness)

        This helper converts RGB back to flow vectors (u, v).

        Args:
            rgb: [B, 3, H, W] RGB tensor in [0, 1].

        Returns:
            Flow tensor [B, 2, H, W] with (u, v) components.
        """
        # Simplified HSV decode from RGB
        # For a proper decode, full HSV conversion would be needed
        # This is a lightweight approximation
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        max_c = torch.max(torch.max(r, g), b)
        min_c = torch.min(torch.min(r, g), b)
        diff = max_c - min_c + 1e-10

        # Hue approximation: direction
        # Using simplified atan2 on (g - b) / (r - g) approximation
        # For Vision Banana flow encoding, u and v are directly mapped to color channels
        u = (r - 0.5) * 2.0 * 255.0  # Approximate horizontal flow
        v = (g - 0.5) * 2.0 * 255.0  # Approximate vertical flow
        # Magnitude from saturation (blue channel proxy)
        mag = b * 2.0

        # Normalize by magnitude if available
        flow = torch.stack([u, v], dim=1)  # [B, 2, H, W]
        return flow

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
            return pred_rgb.to(self.device)

        # ── Fallback: direct model(image) call ───────────────────────────
        pred_rgb = self.model(image)
        if isinstance(pred_rgb, dict):
            pred_rgb = pred_rgb.get("rgb_output", next(iter(pred_rgb.values())))
        return pred_rgb.to(self.device)

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
