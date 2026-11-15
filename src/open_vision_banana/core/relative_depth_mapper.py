"""Relative inverse-depth encoding for the ``depth_rel`` task.

Target: ``d = 1 / z`` on valid pixels, min-max normalised per image to [0, 1] and
replicated into three grey channels (near = bright, far = dark). Pixels without a
valid depth are painted with a fixed sentinel colour that no grey value can take,
and are excluded from the flow loss through the sample's ``loss_mask``.

The encoding has no metric scale by construction: evaluation fits a per-image
affine map from the decoded value to the ground-truth inverse depth on valid
pixels before computing depth metrics (``align_inverse_depth_affine``).

This is intentionally separate from ``DepthRGBMapper`` (the metric cube-edge
contract of the ``depth`` task), so both tasks can be trained from one pool.
"""

from __future__ import annotations

import numpy as np
import torch

DEPTH_REL_TASK = "depth_rel"
DEPTH_REL_PROMPT = (
    "Generate a relative inverse depth map of this image in grayscale: "
    "near surfaces are bright and far surfaces are dark."
)
# Magenta is maximally far from the grey diagonal, so it never equals a valid value.
DEPTH_REL_SENTINEL: tuple[int, int, int] = (255, 0, 255)
# Decoded pixels whose channel spread exceeds this are not on the grey contract.
_GREY_SPREAD_TOLERANCE = 0.25
_MIN_DEPTH_M = 1e-3


class RelativeDepthMapper:
    """Encode/decode the grey relative inverse-depth contract."""

    sentinel = DEPTH_REL_SENTINEL

    @staticmethod
    def normalize_inverse_depth(
        inverse_depth: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Per-image min-max of ``inverse_depth`` over ``valid`` pixels, in [0, 1].

        Shapes: ``[..., H, W]`` for both. Images with a single valid value (or none)
        map to 0.5 on their valid pixels. Invalid pixels are 0.
        """
        valid = valid.bool()
        flat = inverse_depth.float().flatten(-2)
        vflat = valid.flatten(-2)
        big = torch.finfo(flat.dtype).max
        lo = torch.where(vflat, flat, torch.full_like(flat, big)).amin(-1, keepdim=True)
        hi = torch.where(vflat, flat, torch.full_like(flat, -big)).amax(-1, keepdim=True)
        span = hi - lo
        degenerate = ~(span > 1e-12)
        norm = torch.where(
            degenerate, torch.full_like(flat, 0.5), (flat - lo) / span.clamp(min=1e-12)
        )
        norm = torch.where(vflat, norm.clamp(0.0, 1.0), torch.zeros_like(norm))
        return norm.reshape(inverse_depth.shape)

    def encode(self, normalized: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """``[..., H, W]`` normalised inverse depth + validity -> ``[..., 3, H, W]`` RGB in [0, 1]."""
        grey = normalized.float().clamp(0.0, 1.0).unsqueeze(-3).expand(
            *normalized.shape[:-2], 3, *normalized.shape[-2:]
        )
        sentinel = torch.tensor(self.sentinel, dtype=torch.float32, device=normalized.device) / 255.0
        sentinel = sentinel.view(3, 1, 1)
        return torch.where(valid.bool().unsqueeze(-3), grey, sentinel.expand_as(grey)).contiguous()

    def encode_depth(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Metric (or any positive) depth ``[..., H, W]`` -> encoded RGB ``[..., 3, H, W]``."""
        valid = valid.bool() & torch.isfinite(depth) & (depth > _MIN_DEPTH_M)
        inverse = torch.where(valid, 1.0 / depth.clamp(min=_MIN_DEPTH_M), torch.zeros_like(depth))
        return self.encode(self.normalize_inverse_depth(inverse, valid), valid)

    @staticmethod
    def decode(rgb: torch.Tensor) -> torch.Tensor:
        """``[..., 3, H, W]`` RGB in [0, 1] -> normalised inverse depth ``[..., H, W]`` (channel mean)."""
        return rgb.float().clamp(0.0, 1.0).mean(dim=-3)

    @staticmethod
    def off_contract(rgb: torch.Tensor) -> torch.Tensor:
        """Pixels that are not grey (sentinel or a hallucinated colour): ``[..., H, W]`` bool."""
        rgb = rgb.float()
        spread = rgb.amax(dim=-3) - rgb.amin(dim=-3)
        return spread > _GREY_SPREAD_TOLERANCE

    def sentinel_mask(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """Exact sentinel pixels of a ``[..., 3, H, W]`` uint8 target."""
        sentinel = torch.tensor(self.sentinel, dtype=rgb_uint8.dtype, device=rgb_uint8.device)
        return (rgb_uint8 == sentinel.view(3, 1, 1)).all(dim=-3)


def metric_target_inverse_depth(
    target_rgb: np.ndarray, valid_mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a stored cube-edge metric depth target (``uint8 [H, W, 3]``) to ``1 / z``.

    Validity is decoded depth above 1 mm (holes are stored as zero depth), intersected
    with ``valid_mask`` (non-zero = valid) when the row ships one. Returns
    ``(inverse float32 [H, W] (0 where invalid), valid bool [H, W])``.
    """
    from .depth_mapper import DepthRGBMapper

    if target_rgb.dtype != np.uint8 or target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
        raise ValueError("target_rgb must be uint8 [H, W, 3]")
    # Stored targets hold a few thousand distinct colours: decode each once (the
    # per-pixel curve projection costs seconds on a megapixel image).
    packed = (
        (target_rgb[..., 0].astype(np.int32) << 16)
        | (target_rgb[..., 1].astype(np.int32) << 8)
        | target_rgb[..., 2].astype(np.int32)
    )
    colors, index = np.unique(packed.reshape(-1), return_inverse=True)
    unique_rgb = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    rgb = torch.from_numpy(unique_rgb.astype(np.float32) / 255.0).T.reshape(1, 3, 1, -1)
    decoded = DepthRGBMapper(colormap="cube_edge").rgb_to_depth(rgb).reshape(-1).numpy()
    metric = decoded[index].reshape(target_rgb.shape[:2])
    valid = np.isfinite(metric) & (metric > _MIN_DEPTH_M)
    if valid_mask is not None:
        if valid_mask.shape != valid.shape:
            raise ValueError("valid_mask and depth target geometry must match")
        valid &= valid_mask > 0
    inverse = np.where(valid, 1.0 / np.maximum(metric, _MIN_DEPTH_M), 0.0).astype(np.float32)
    return inverse, valid


def align_inverse_depth_affine(
    prediction: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_depth: float | None = None,
) -> torch.Tensor:
    """Per-image least-squares ``a * p + b ~= 1 / gt`` on valid pixels, returned as depth.

    ``prediction`` is the decoded normalised inverse depth; all tensors share the
    shape ``[B, 1, H, W]`` (or ``[B, H, W]``). Images with fewer than two valid pixels
    fall back to a scale-only fit. The aligned inverse depth is floored at the
    smallest GT inverse depth seen in that image (or ``1 / max_depth``) so it never
    becomes non-positive, then inverted.
    """
    if prediction.shape != gt_depth.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, gt_depth and valid must share one shape")
    out = torch.zeros_like(gt_depth, dtype=torch.float32)
    for i in range(prediction.shape[0]):
        p = prediction[i].float()
        gt = gt_depth[i].float()
        mask = valid[i].bool() & torch.isfinite(gt) & (gt > 0) & torch.isfinite(p)
        target = torch.zeros_like(gt)
        target[mask] = 1.0 / gt[mask]
        floor = 1.0 / max_depth if max_depth else None
        if mask.sum() >= 2:
            x = p[mask].double()
            y = target[mask].double()
            design = torch.stack((x, torch.ones_like(x)), dim=1)
            solution = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
            a, b = float(solution[0]), float(solution[1])
            fitted = a * p + b
            low = float(y.min())
        elif mask.any():
            fitted = p * float(target[mask].mean() / p[mask].mean().clamp(min=1e-6))
            low = float(target[mask].min())
        else:
            fitted = torch.full_like(p, 1.0)
            low = 1e-3
        low = max(low, floor) if floor else low
        fitted = fitted.clamp(min=max(low, 1e-6))
        out[i] = 1.0 / fitted
    return out


__all__ = [
    "DEPTH_REL_PROMPT",
    "DEPTH_REL_SENTINEL",
    "DEPTH_REL_TASK",
    "RelativeDepthMapper",
    "align_inverse_depth_affine",
    "metric_target_inverse_depth",
]
