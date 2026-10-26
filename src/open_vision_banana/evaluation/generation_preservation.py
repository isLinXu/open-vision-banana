"""Generation preservation evaluation metrics.

Lightweight metrics to assess whether the tuned model retains its original
image generation capabilities.  The paper reports a 53.5% win rate on
GenAI-Bench; we provide proxy metrics that do not require the full
GenAI-Bench pipeline.

Metrics:
- FID: Fréchet distance via scipy (when installed) on feature statistics.
- FID_proxy: simplified fallback when scipy is unavailable.
- LPIPS: perceptual distance via ``lpips`` package (optional).
- CLIP-score-proxy: semantic similarity between generated images and prompts.
- Diversity score: standard deviation of generated image features.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_lpips_model: torch.nn.Module | None = None
_inception_model: torch.nn.Module | None = None


def evaluate_pairwise_preferences(decisions: Sequence[str]) -> dict[str, float | int]:
    """Compute a GenAI-Bench-style win rate from recorded pairwise judgments.

    Decisions must be ``"candidate"``, ``"baseline"``, or ``"tie"``. Ties
    contribute half a win, matching common pairwise evaluation reporting.
    """
    normalized = [decision.strip().lower() for decision in decisions]
    allowed = {"candidate", "baseline", "tie"}
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise ValueError(f"Unknown pairwise decisions: {invalid}")
    if not normalized:
        raise ValueError("At least one pairwise decision is required")
    wins = normalized.count("candidate")
    losses = normalized.count("baseline")
    ties = normalized.count("tie")
    total = len(normalized)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "total": total,
        "win_rate": (wins + 0.5 * ties) / total,
    }


def _lpips_available() -> bool:
    try:
        import lpips  # noqa: F401
        return True
    except Exception as exc:
        logger.debug("LPIPS is unavailable: %s", exc)
        return False


def _scipy_available() -> bool:
    try:
        import scipy.linalg  # noqa: F401
        return True
    except Exception as exc:
        logger.debug("SciPy is unavailable: %s", exc)
        return False


def _compute_feature_statistics(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and covariance of features."""
    mean = features.mean(dim=0)
    centered = features - mean.unsqueeze(0)
    cov = (centered.T @ centered) / (features.shape[0] - 1 + 1e-10)
    return mean, cov


def compute_fid_proxy(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
) -> float:
    """Simplified Fréchet distance proxy (no scipy)."""
    mu_real, cov_real = _compute_feature_statistics(real_features)
    mu_gen, cov_gen = _compute_feature_statistics(gen_features)
    mean_diff = (mu_real - mu_gen).norm().item()
    cov_diff = torch.norm(cov_real - cov_gen, p="fro").item()
    return mean_diff ** 2 + cov_diff


def compute_fid(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
) -> float:
    """Fréchet Inception Distance on feature vectors.

    Uses scipy matrix square root when available; otherwise falls back to
    ``compute_fid_proxy``.
    """
    if not _scipy_available():
        return compute_fid_proxy(real_features, gen_features)

    import numpy as np
    from scipy.linalg import sqrtm

    mu1, sigma1 = _compute_feature_statistics(real_features)
    mu2, sigma2 = _compute_feature_statistics(gen_features)
    mu1_np = mu1.detach().cpu().numpy()
    mu2_np = mu2.detach().cpu().numpy()
    sigma1_np = sigma1.detach().cpu().numpy()
    sigma2_np = sigma2.detach().cpu().numpy()

    diff = mu1_np - mu2_np
    covmean = sqrtm(sigma1_np @ sigma2_np)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1_np + sigma2_np - 2.0 * covmean))
    return max(fid, 0.0)


def _get_inception(device: torch.device | str) -> torch.nn.Module:
    global _inception_model
    if _inception_model is None:
        from torchvision.models import Inception_V3_Weights, inception_v3

        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        _inception_model = model
    return _inception_model.to(device)


def extract_inception_features(
    rgb: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Extract 2048-d InceptionV3 features from RGB images in [0, 1].

    Args:
        rgb: [N, 3, H, W] tensor.
        device: Target device (defaults to rgb.device).

    Returns:
        [N, 2048] feature tensor.
    """
    if device is None:
        device = rgb.device
    model = _get_inception(device)
    x = F.interpolate(rgb, size=(299, 299), mode="bilinear", align_corners=False)
    # ImageNet normalization used by InceptionV3
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        feats = model(x)
    return feats


def extract_rgb_features(
    rgb: torch.Tensor,
    *,
    pool_size: int = 8,
) -> torch.Tensor:
    """Extract deterministic, dependency-free RGB features.

    This is intentionally a proxy for FID, not an Inception replacement.  It
    is useful for smoke tests and offline health checks where downloading
    ImageNet weights is not acceptable.  Callers should record the feature
    backend alongside the resulting metric.
    """
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape [N,3,H,W], got {tuple(rgb.shape)}")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if not torch.isfinite(rgb).all():
        raise ValueError("rgb contains NaN or Inf values")
    pooled = F.adaptive_avg_pool2d(rgb.float(), (pool_size, pool_size))
    channel_mean = rgb.float().mean(dim=(-2, -1), keepdim=True)
    channel_std = rgb.float().std(dim=(-2, -1), keepdim=True, unbiased=False)
    return torch.cat(
        [pooled.flatten(1), channel_mean.flatten(1), channel_std.flatten(1)],
        dim=1,
    )


def _get_lpips_model(device: torch.device | str, net: str = "alex") -> torch.nn.Module:
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net=net)
    return _lpips_model.to(device)


def _lpips_per_image(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    net: str = "alex",
) -> torch.Tensor:
    """[N] LPIPS distances of ``pred_rgb`` to ``target_rgb`` (requires the ``lpips`` package)."""
    model = _get_lpips_model(pred_rgb.device, net=net)
    with torch.no_grad():
        return model(pred_rgb * 2.0 - 1.0, target_rgb * 2.0 - 1.0).flatten()


def compute_lpips(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    net: str = "alex",
) -> float:
    """Perceptual distance via LPIPS (requires optional ``lpips`` package)."""
    if not _lpips_available():
        return compute_perceptual_similarity(pred_rgb, target_rgb)
    return float(_lpips_per_image(pred_rgb, target_rgb, net=net).mean().item())


def compute_clip_score_proxy(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> float:
    """CLIP-style alignment score between images and text prompts."""
    img_norm = F.normalize(image_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)
    sim = (img_norm * text_norm).sum(dim=-1)
    return sim.mean().item()


def compute_diversity_score(features: torch.Tensor) -> float:
    """Average pairwise distance in normalized feature space."""
    if features.shape[0] < 2:
        return 0.0
    features = F.normalize(features, dim=-1)
    dist_matrix = torch.cdist(features, features, p=2)
    mask = 1.0 - torch.eye(features.shape[0], device=features.device)
    avg_dist = (dist_matrix * mask).sum() / (mask.sum() + 1e-10)
    return avg_dist.item()


def compute_perceptual_similarity(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> float:
    """Pixel-level MSE fallback when LPIPS is unavailable."""
    return F.mse_loss(pred_rgb, target_rgb).item()


def _mse_per_image(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    """[N] pixel MSE of each image; their mean is :func:`compute_perceptual_similarity`."""
    return (pred_rgb - target_rgb).pow(2).flatten(1).mean(dim=1)


def _resize_like(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    """``pred_rgb`` resized (bilinear) to the spatial size of ``target_rgb`` when the shapes differ."""
    if pred_rgb.shape == target_rgb.shape:
        return pred_rgb
    return F.interpolate(pred_rgb, size=target_rgb.shape[-2:], mode="bilinear", align_corners=False)


def evaluate_generation_preservation(
    real_features: torch.Tensor | None = None,
    gen_features: torch.Tensor | None = None,
    text_features: torch.Tensor | None = None,
    pred_rgb: torch.Tensor | None = None,
    target_rgb: torch.Tensor | None = None,
    *,
    real_rgb: torch.Tensor | None = None,
    gen_rgb: torch.Tensor | None = None,
    feature_extractor: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Run generation-preservation evaluation.

    Provide either precomputed feature tensors or RGB images (Inception
    features are extracted automatically from ``real_rgb`` / ``gen_rgb``).
    """
    results: dict[str, float] = {}

    extractor = feature_extractor or extract_inception_features
    if real_features is None and real_rgb is not None:
        real_features = extractor(real_rgb)
    if gen_features is None and gen_rgb is not None:
        gen_features = extractor(gen_rgb)
    if gen_features is None:
        raise ValueError("gen_features or gen_rgb is required")

    if real_features is not None:
        results["FID"] = compute_fid(real_features, gen_features)
        results["FID_proxy"] = compute_fid_proxy(real_features, gen_features)
    else:
        results["FID"] = float("nan")
        results["FID_proxy"] = float("nan")

    results["diversity_score"] = compute_diversity_score(gen_features)

    if text_features is not None:
        results["clip_score_proxy"] = compute_clip_score_proxy(gen_features, text_features)
    else:
        results["clip_score_proxy"] = float("nan")

    if pred_rgb is not None and target_rgb is not None:
        pred_rgb = _resize_like(pred_rgb, target_rgb)
        if _lpips_available():
            results["LPIPS"] = compute_lpips(pred_rgb, target_rgb)
            results["perceptual_similarity"] = results["LPIPS"]
        else:
            results["LPIPS"] = float("nan")
            results["perceptual_similarity"] = compute_perceptual_similarity(pred_rgb, target_rgb)
    else:
        results["LPIPS"] = float("nan")
        results["perceptual_similarity"] = float("nan")

    fid_for_health = results["FID"] if not math.isnan(results["FID"]) else results["FID_proxy"]
    if not math.isnan(fid_for_health):
        fid_norm = min(fid_for_health / 100.0, 1.0) * 100.0
        results["generation_health_score"] = max(0.0, 100.0 - fid_norm)
    else:
        results["generation_health_score"] = float("nan")

    return results


def generation_shard_state(
    real_rgb: torch.Tensor,
    gen_rgb: torch.Tensor,
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor | bool]:
    """Per-image quantities from which :func:`generation_metrics` computes the report.

    Every metric :func:`evaluate_generation_preservation` reports for ``real_rgb`` against
    ``gen_rgb`` is a function of per-image features (Fréchet statistics, diversity) or the mean
    of a per-image distance (LPIPS, else pixel MSE), so the states of any split of the samples
    into blocks give the metrics of all of them.
    """
    pred_rgb = _resize_like(gen_rgb, real_rgb)
    lpips = _lpips_available()
    return {
        "real_features": feature_extractor(real_rgb),
        "gen_features": feature_extractor(gen_rgb),
        "lpips": lpips,
        "perceptual": (
            _lpips_per_image(pred_rgb, real_rgb) if lpips else _mse_per_image(pred_rgb, real_rgb)
        ),
    }


def generation_metrics(states: Sequence[dict[str, torch.Tensor | bool]]) -> dict[str, float]:
    """Metrics of all samples from the :func:`generation_shard_state` of each block, in order.

    They equal what :func:`evaluate_generation_preservation` reports for the same images, up to
    the order in which the per-image distances are averaged.
    """
    if not states:
        raise ValueError("no generation shard states")
    if len({state["lpips"] for state in states}) != 1:
        raise ValueError("shards disagree on whether LPIPS was available")
    results = evaluate_generation_preservation(
        real_features=torch.cat([state["real_features"] for state in states]),
        gen_features=torch.cat([state["gen_features"] for state in states]),
    )
    # without images the call above leaves the perceptual entries NaN
    perceptual = float(torch.cat([state["perceptual"] for state in states]).mean().item())
    results["perceptual_similarity"] = perceptual
    if states[0]["lpips"]:
        results["LPIPS"] = perceptual
    return results


__all__ = [
    "compute_fid",
    "compute_fid_proxy",
    "compute_clip_score_proxy",
    "compute_diversity_score",
    "compute_perceptual_similarity",
    "compute_lpips",
    "extract_inception_features",
    "extract_rgb_features",
    "evaluate_pairwise_preferences",
    "evaluate_generation_preservation",
    "generation_shard_state",
    "generation_metrics",
]
