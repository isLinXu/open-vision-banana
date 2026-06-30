"""Generation preservation evaluation metrics.

Lightweight metrics to assess whether the tuned model retains its original
image generation capabilities.  The paper reports a 53.5% win rate on
GenAI-Bench; we provide proxy metrics that do not require the full
GenAI-Bench pipeline.

Metrics:
- FID-proxy: Fréchet distance between feature statistics of generated and
  reference images (lightweight InceptionV3 features, not full FID).
- CLIP-score-proxy: semantic similarity between generated images and their
  prompts (uses a small CLIP-like model or precomputed embeddings).
- Diversity score: standard deviation of generated image features.
- Perceptual similarity: LPIPS-style distance between generated and reference.
- Text-to-image alignment: cosine similarity between prompt embeddings and
  image features.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _compute_feature_statistics(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and covariance of features.

    Args:
        features: [N, D] feature vectors.

    Returns:
        (mean [D], cov [D, D]).
    """
    mean = features.mean(dim=0)  # [D]
    centered = features - mean.unsqueeze(0)  # [N, D]
    cov = (centered.T @ centered) / (features.shape[0] - 1 + 1e-10)  # [D, D]
    return mean, cov


def compute_fid_proxy(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
) -> float:
    """Compute lightweight FID proxy between two feature distributions.

    Uses a simplified Fréchet distance between feature statistics.
    This is a proxy, not the full FID (which requires InceptionV3 and
    ~50K samples for reliable statistics).

    Args:
        real_features: [N_real, D] feature vectors from real images.
        gen_features: [N_gen, D] feature vectors from generated images.

    Returns:
        FID proxy (lower is better, 0 = identical distributions).
    """
    mu_real, cov_real = _compute_feature_statistics(real_features)
    mu_gen, cov_gen = _compute_feature_statistics(gen_features)

    # Mean difference
    mean_diff = (mu_real - mu_gen).norm().item()

    # Covariance difference: trace(cov_real + cov_gen - 2 * sqrt(cov_real * cov_gen))
    # Simplified: use ||cov_real - cov_gen||_F as proxy
    cov_diff = torch.norm(cov_real - cov_gen, p="fro").item()

    fid = mean_diff ** 2 + cov_diff
    return fid


def compute_clip_score_proxy(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> float:
    """Compute CLIP-style alignment score between images and text prompts.

    Args:
        image_features: [N, D] image feature vectors (normalized).
        text_features: [N, D] text feature vectors (normalized).

    Returns:
        Mean cosine similarity (higher is better, 1 = perfect alignment).
    """
    # Normalize
    img_norm = F.normalize(image_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)

    # Cosine similarity
    sim = (img_norm * text_norm).sum(dim=-1)  # [N]
    return sim.mean().item()


def compute_diversity_score(
    features: torch.Tensor,
) -> float:
    """Compute diversity score as the average pairwise distance in feature space.

    Higher values indicate more diverse generation.  A diversity score of 0
    means all generated images are identical.

    Args:
        features: [N, D] feature vectors.

    Returns:
        Diversity score (higher is better).
    """
    if features.shape[0] < 2:
        return 0.0

    # Pairwise L2 distance
    features = F.normalize(features, dim=-1)
    dist_matrix = torch.cdist(features, features, p=2)  # [N, N]

    # Exclude diagonal (self-distance = 0)
    mask = 1.0 - torch.eye(features.shape[0], device=features.device)
    avg_dist = (dist_matrix * mask).sum() / (mask.sum() + 1e-10)
    return avg_dist.item()


def compute_perceptual_similarity(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
) -> float:
    """Compute perceptual similarity (LPIPS-style) using pixel-level MSE as proxy.

    For a true LPIPS metric, a pre-trained VGG or AlexNet feature extractor
    would be required.  This function provides a lightweight fallback that
    measures pixel-level MSE; for production use, replace with a real LPIPS
    model.

    Args:
        pred_rgb: [B, 3, H, W] predicted RGB images in [0, 1].
        target_rgb: [B, 3, H, W] target RGB images in [0, 1].

    Returns:
        Perceptual similarity proxy (lower is better, 0 = identical).
    """
    mse = F.mse_loss(pred_rgb, target_rgb).item()
    return mse


def evaluate_generation_preservation(
    real_features: torch.Tensor | None,
    gen_features: torch.Tensor,
    text_features: torch.Tensor | None = None,
    pred_rgb: torch.Tensor | None = None,
    target_rgb: torch.Tensor | None = None,
) -> dict[str, float]:
    """Run a full generation-preservation evaluation suite.

    Args:
        real_features: [N_real, D] features from real / pre-tuning generation.
        gen_features: [N_gen, D] features from post-tuning generation.
        text_features: [N_gen, D] text prompt features (for CLIP score).
        pred_rgb: [B, 3, H, W] generated images (for perceptual similarity).
        target_rgb: [B, 3, H, W] target images (for perceptual similarity).

    Returns:
        Dict with all generation preservation metrics.
    """
    results = {}

    # FID proxy
    if real_features is not None:
        results["FID_proxy"] = compute_fid_proxy(real_features, gen_features)
    else:
        results["FID_proxy"] = float("nan")

    # Diversity
    results["diversity_score"] = compute_diversity_score(gen_features)

    # CLIP score proxy
    if text_features is not None:
        results["clip_score_proxy"] = compute_clip_score_proxy(gen_features, text_features)
    else:
        results["clip_score_proxy"] = float("nan")

    # Perceptual similarity
    if pred_rgb is not None and target_rgb is not None:
        results["perceptual_similarity"] = compute_perceptual_similarity(pred_rgb, target_rgb)
    else:
        results["perceptual_similarity"] = float("nan")

    # Generation win rate proxy (based on FID)
    if not math.isnan(results["FID_proxy"]):
        # Heuristic: lower FID = better preservation
        # Normalize to a 0-100 scale (0 = perfect, 100 = terrible)
        fid_norm = min(results["FID_proxy"] / 100.0, 1.0) * 100.0
        results["generation_health_score"] = max(0.0, 100.0 - fid_norm)
    else:
        results["generation_health_score"] = float("nan")

    return results


__all__ = [
    "compute_fid_proxy",
    "compute_clip_score_proxy",
    "compute_diversity_score",
    "compute_perceptual_similarity",
    "evaluate_generation_preservation",
]