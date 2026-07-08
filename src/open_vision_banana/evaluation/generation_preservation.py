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

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_lpips_model: torch.nn.Module | None = None
_inception_model: torch.nn.Module | None = None


def _lpips_available() -> bool:
    try:
        import lpips  # noqa: F401
        return True
    except ImportError:
        return False


def _scipy_available() -> bool:
    try:
        import scipy.linalg  # noqa: F401
        return True
    except ImportError:
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


def _get_lpips_model(device: torch.device | str, net: str = "alex") -> torch.nn.Module:
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net=net)
    return _lpips_model.to(device)


def compute_lpips(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    *,
    net: str = "alex",
) -> float:
    """Perceptual distance via LPIPS (requires optional ``lpips`` package)."""
    if not _lpips_available():
        return compute_perceptual_similarity(pred_rgb, target_rgb)

    device = pred_rgb.device
    model = _get_lpips_model(device, net=net)
    pred = pred_rgb * 2.0 - 1.0
    target = target_rgb * 2.0 - 1.0
    with torch.no_grad():
        dist = model(pred, target)
    return float(dist.mean().item())


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


def evaluate_generation_preservation(
    real_features: torch.Tensor | None = None,
    gen_features: torch.Tensor | None = None,
    text_features: torch.Tensor | None = None,
    pred_rgb: torch.Tensor | None = None,
    target_rgb: torch.Tensor | None = None,
    *,
    real_rgb: torch.Tensor | None = None,
    gen_rgb: torch.Tensor | None = None,
) -> dict[str, float]:
    """Run generation-preservation evaluation.

    Provide either precomputed feature tensors or RGB images (Inception
    features are extracted automatically from ``real_rgb`` / ``gen_rgb``).
    """
    results: dict[str, float] = {}

    if real_features is None and real_rgb is not None:
        real_features = extract_inception_features(real_rgb)
    if gen_features is None and gen_rgb is not None:
        gen_features = extract_inception_features(gen_rgb)
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
        if pred_rgb.shape != target_rgb.shape:
            pred_rgb = F.interpolate(
                pred_rgb,
                size=target_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
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


__all__ = [
    "compute_fid",
    "compute_fid_proxy",
    "compute_clip_score_proxy",
    "compute_diversity_score",
    "compute_perceptual_similarity",
    "compute_lpips",
    "extract_inception_features",
    "evaluate_generation_preservation",
]
