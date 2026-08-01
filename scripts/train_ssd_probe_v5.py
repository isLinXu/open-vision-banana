#!/usr/bin/env python3
"""Train SSD-1B frozen-feature depth probe V5 — enhanced loss + skip decoder + early stopping.

Key improvements over V4 (`train_ssd_probe_v4.py`):
1. **SSIM + Edge-aware loss** — structural similarity captures spatial structure
   (addresses edge blurring); Sobel edge maps weight boundary errors higher.
2. **Skip-connection decoder** — `ScalarDepthProbeV5` injects frozen features
   at multiple decoder stages, preserving fine spatial detail.
3. **Early stopping (patience=30)** — stops training when val score plateaus,
   saving ~45% training time vs V4's fixed 300 epochs.
4. **Extended augmentation** — brightness jitter + vertical flip in addition
   to V4's horizontal flip, applied on cached features (zero re-extraction cost).
5. **EMA probe weights** — exponential moving average of probe parameters for
   smoother inference and better generalization.

Usage:
    python scripts/train_ssd_probe_v5.py \
        --data-dir data/vb_snvc_train_v3 \
        --output-dir output/ssd_probe_v5 \
        --epochs 250 --batch-size 8 --lr 2e-3 --width 128 \
        --patience 30 --ssim-weight 0.2 --edge-weight 0.15 \
        --aug-flip-prob 0.5 --aug-brightness 0.1

Note: V5 reuses the V4 feature cache if present (`--cache-dir` points to V4 cache).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.data.depth_encoding import (  # noqa: E402
    audit_depth_target_encoding,
    relative_depth_prompt,
)
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.training.depth_probe import (  # noqa: E402
    DepthFeatureContract,
    FrozenSDXLDepthFeatureExtractor,
    ScalarDepthProbeV5,
    evaluate_scalar_depth_predictions,
    v5_composite_loss,
)
from open_vision_banana.training.tuner import VisionTaskDataset  # noqa: E402

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(serialized + "\n", encoding="utf-8")


def declared_depth_names(data_dir: Path) -> tuple[list[str], list[str]]:
    manifest = data_dir / "train_val_split.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing depth split manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    train_names = [str(name) for name in payload.get("train", {}).get("depth", [])]
    val_names = [str(name) for name in payload.get("val", {}).get("depth", [])]
    if not train_names or not val_names:
        raise ValueError("Depth manifest must contain non-empty train and val splits")
    if set(train_names).intersection(val_names):
        raise ValueError("Declared depth train and val splits overlap")
    return train_names, val_names


# ──────────────────────────────────────────────────────────────
# Feature caching (reuse V4 cache format)
# ──────────────────────────────────────────────────────────────


def _cache_features_to_memory(
    generator: BaseGenerator,
    data_dir: Path,
    *,
    names: list[str],
    resolution: int,
    prompt: str,
    timestep: int,
    mapper: DepthRGBMapper,
    storage_dtype: torch.dtype,
    desc: str = "",
) -> tuple[DepthFeatureContract, tuple[torch.Tensor, ...], torch.Tensor]:
    dataset = VisionTaskDataset(
        str(data_dir / "depth"),
        resolution=resolution,
        task="depth",
    )
    by_name = {
        Path(sample["image_path"]).name: index
        for index, sample in enumerate(dataset.samples)
    }
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError(f"Declared depth samples not found: {missing[:8]}")

    extractor = FrozenSDXLDepthFeatureExtractor(generator, timestep=timestep)
    contract: DepthFeatureContract | None = None
    levels: list[list[torch.Tensor]] | None = None
    targets: list[torch.Tensor] = []

    for position, name in enumerate(names, start=1):
        sample = dataset[by_name[name]]
        image = sample["image"].unsqueeze(0)
        sample_contract, sample_features = extractor.extract(image, prompt=prompt)

        if contract is None:
            contract = sample_contract
            levels = [[] for _ in sample_features]
        elif sample_contract != contract:
            raise RuntimeError("Frozen feature contract changed between samples")
        assert levels is not None

        for feature_name, destination, feature in zip(
            sample_contract.names,
            levels,
            sample_features,
            strict=True,
        ):
            stored = feature.to(device="cpu", dtype=storage_dtype)
            if not torch.isfinite(stored).all():
                raise FloatingPointError(
                    f"Feature {feature_name} for {name} is non-finite"
                )
            destination.append(stored)

        target = mapper.rgb_to_normalized(sample["target_rgb"].unsqueeze(0))
        target = target.to(device="cpu", dtype=storage_dtype)
        if not torch.isfinite(target).all():
            raise FloatingPointError(f"Depth target for {name} is non-finite")
        targets.append(target)

        if position % 50 == 0 or position == len(names):
            logger.info("  %s cached %d/%d images", desc, position, len(names))

    if contract is None or levels is None:
        raise RuntimeError("No frozen features were extracted")

    features = tuple(torch.cat(level, dim=0) for level in levels)
    target_tensor = torch.cat(targets, dim=0)
    return contract, features, target_tensor


def _save_feature_cache(
    cache_dir: Path,
    *,
    contract: DepthFeatureContract,
    train_features: tuple[torch.Tensor, ...],
    train_target: torch.Tensor,
    train_names: list[str],
    val_features: tuple[torch.Tensor, ...],
    val_target: torch.Tensor,
    val_names: list[str],
    metadata: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": contract.to_dict(),
            "features": train_features,
            "target": train_target,
            "names": train_names,
        },
        cache_dir / "train_cache.pt",
    )
    torch.save(
        {
            "contract": contract.to_dict(),
            "features": val_features,
            "target": val_target,
            "names": val_names,
        },
        cache_dir / "val_cache.pt",
    )
    (cache_dir / "cache_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logger.info("Feature cache saved to %s", cache_dir)


def _load_feature_cache(
    cache_dir: Path,
) -> tuple[DepthFeatureContract, tuple[torch.Tensor, ...], torch.Tensor, list[str],
              tuple[torch.Tensor, ...], torch.Tensor, list[str]] | None:
    train_cache = cache_dir / "train_cache.pt"
    val_cache = cache_dir / "val_cache.pt"
    meta_file = cache_dir / "cache_meta.json"
    if not (train_cache.is_file() and val_cache.is_file() and meta_file.is_file()):
        return None

    logger.info("Loading feature cache from %s ...", cache_dir)
    train_data = torch.load(str(train_cache), map_location="cpu", weights_only=False)
    val_data = torch.load(str(val_cache), map_location="cpu", weights_only=False)
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    contract = DepthFeatureContract.from_dict(train_data["contract"])
    train_features = tuple(train_data["features"])
    val_features = tuple(val_data["features"])

    logger.info(
        "Cache loaded: train=%d, val=%d, resolution=%s, dtype=%s",
        train_features[0].shape[0],
        val_features[0].shape[0],
        meta.get("resolution"),
        meta.get("feature_storage_dtype"),
    )
    return (
        contract,
        train_features,
        train_data["target"],
        train_data["names"],
        val_features,
        val_data["target"],
        val_data["names"],
    )


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def _evaluate_on_set(
    head: ScalarDepthProbeV5,
    features: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    num_samples = target.shape[0]
    output_size = tuple(int(v) for v in target.shape[-2:])
    all_preds = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_features = tuple(
            feat[start:end].to(device=device, dtype=torch.float32)
            for feat in features
        )
        pred = head(batch_features, output_size=output_size)
        all_preds.append(pred.cpu())
    prediction = torch.cat(all_preds, dim=0)
    return evaluate_scalar_depth_predictions(prediction, target.cpu())


def _save_montages(
    output_dir: Path,
    *,
    names: list[str],
    images_tensor: torch.Tensor,
    target_rgb: torch.Tensor,
    prediction: torch.Tensor,
    colormap: str,
    max_count: int = 16,
) -> list[str]:
    mapper = DepthRGBMapper(colormap)
    prediction_rgb = mapper.normalized_to_rgb(prediction.cpu())
    count = min(max_count, len(names))
    paths = []
    for index in range(count):
        source = ((images_tensor[index].clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 0)
        target = target_rgb[index].clamp(0, 1).permute(1, 2, 0)
        predicted = prediction_rgb[index].clamp(0, 1).permute(1, 2, 0)
        panels = [
            Image.fromarray((panel.numpy() * 255).round().astype(np.uint8))
            for panel in (source, target, predicted)
        ]
        montage = Image.new("RGB", (panels[0].width * 3, panels[0].height))
        for panel_index, panel in enumerate(panels):
            montage.paste(panel, (panel_index * panel.width, 0))
        path = output_dir / f"montage_{index:03d}_{Path(names[index]).stem}.png"
        montage.save(path)
        paths.append(str(path.resolve()))
    return paths


# ──────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────


def _lr_schedule(step: int, warmup: int, total: int) -> float:
    """Cosine with linear warmup, floor at 1%."""
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress)) * 0.99 + 0.01


def _augment_batch(
    features: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    batch_size: int,
    *,
    flip_h_prob: float,
    flip_v_prob: float,
    brightness: float,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Apply random augmentations to a batch of cached features + targets."""
    # Horizontal flip
    if flip_h_prob > 0:
        flip_mask = torch.rand(batch_size) < flip_h_prob
        if flip_mask.any():
            flip_idx = flip_mask.nonzero(as_tuple=True)[0]
            features = tuple(
                _flip_dim(feat.clone(), flip_idx, dim=-1) for feat in features
            )
            target = _flip_dim(target.clone(), flip_idx, dim=-1)

    # Vertical flip
    if flip_v_prob > 0:
        flip_mask = torch.rand(batch_size) < flip_v_prob
        if flip_mask.any():
            flip_idx = flip_mask.nonzero(as_tuple=True)[0]
            features = tuple(
                _flip_dim(feat.clone(), flip_idx, dim=-2) for feat in features
            )
            target = _flip_dim(target.clone(), flip_idx, dim=-2)

    # Brightness jitter (additive, applied to all feature channels uniformly)
    if brightness > 0:
        delta = (torch.rand(batch_size, 1, 1, 1) * 2 - 1) * brightness
        delta = delta.to(device=features[0].device, dtype=features[0].dtype)
        features = tuple(feat + delta for feat in features)

    return features, target


def _flip_dim(tensor: torch.Tensor, indices: torch.Tensor, *, dim: int) -> torch.Tensor:
    tensor[indices] = torch.flip(tensor[indices], dims=[dim])
    return tensor


class EMAProbe:
    """Exponential Moving Average of probe parameters."""

    def __init__(self, probe: ScalarDepthProbeV5, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in probe.named_parameters()}

    @torch.no_grad()
    def update(self, probe: ScalarDepthProbeV5) -> None:
        for name, param in probe.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_to(self, probe: ScalarDepthProbeV5) -> ScalarDepthProbeV5:
        """Return a copy of the probe with EMA weights applied."""
        ema_probe = copy.deepcopy(probe)
        for name, param in ema_probe.named_parameters():
            param.data.copy_(self.shadow[name])
        return ema_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="segmind/SSD-1B")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory for feature cache (default: output-dir/feature_cache)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every-epoch", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--upsample-stages", type=int, default=3)
    parser.add_argument("--gradient-weight", type=float, default=0.5)
    parser.add_argument("--pearson-weight", type=float, default=0.3)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--edge-weight", type=float, default=0.15)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--feature-storage-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    # Augmentation
    parser.add_argument("--aug-flip-prob", type=float, default=0.5)
    parser.add_argument("--aug-flip-v-prob", type=float, default=0.1)
    parser.add_argument("--aug-brightness", type=float, default=0.1)
    # Early stopping
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (epochs without improvement)")
    # EMA
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-val-samples", type=int, default=0, help="0=all val")
    # Resume support
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint to resume from (loads head weights)")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="Epoch to start training from (for resume)")
    args = parser.parse_args()

    if any(v <= 0 for v in [args.resolution, args.epochs, args.batch_size, args.lr, args.width]):
        parser.error("resolution, epochs, batch_size, lr, width must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _set_seed(args.seed)
    started = time.perf_counter()
    device = _resolve_device(args.device)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir = (args.cache_dir or output_dir / "feature_cache").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SSD-1B Frozen Probe V5 — Enhanced Loss + Skip Decoder + Early Stopping")
    logger.info("=" * 70)
    logger.info("  model:          %s", args.model)
    logger.info("  data_dir:       %s", data_dir)
    logger.info("  output_dir:     %s", output_dir)
    logger.info("  cache_dir:      %s", cache_dir)
    logger.info("  resolution:     %d", args.resolution)
    logger.info("  epochs:         %d (max)", args.epochs)
    logger.info("  patience:       %d (early stopping)", args.patience)
    logger.info("  batch_size:     %d", args.batch_size)
    logger.info("  lr:             %s", args.lr)
    logger.info("  width:          %d (probe channels)", args.width)
    logger.info("  loss weights:   grad=%.2f pearson=%.2f ssim=%.2f edge=%.2f",
                args.gradient_weight, args.pearson_weight, args.ssim_weight, args.edge_weight)
    logger.info("  augmentation:   flip_h=%.2f flip_v=%.2f brightness=%.3f",
                args.aug_flip_prob, args.aug_flip_v_prob, args.aug_brightness)
    logger.info("  ema_decay:      %s", args.ema_decay)
    logger.info("  device:         %s", device)
    logger.info("=" * 70)

    # ── 1. Load data manifest ──────────────────────────────────
    train_names, val_names = declared_depth_names(data_dir)
    logger.info("Train samples: %d", len(train_names))
    logger.info("Val samples:   %d", len(val_names))

    # ── 2. Audit depth encoding ────────────────────────────────
    all_names_set = set([*train_names, *val_names])
    audit = audit_depth_target_encoding(
        data_dir / "depth" / "targets",
        allowed_names=all_names_set,
    )
    colormap = audit.detected_colormap
    prompt = relative_depth_prompt(colormap)
    mapper = DepthRGBMapper(colormap)
    logger.info("Detected colormap: %s (curve MAE: %.5f)", colormap, audit.mean_curve_mae)

    # ── 3. Try loading cached features ─────────────────────────
    cached = _load_feature_cache(cache_dir)
    storage_dtype = getattr(torch, args.feature_storage_dtype)

    if cached is not None:
        (train_contract, train_features, train_target, train_names_cached,
         val_features, val_target, val_names_cached) = cached
        if len(train_names_cached) == len(train_names):
            train_names = train_names_cached
        if len(val_names_cached) == len(val_names):
            val_names = val_names_cached
        logger.info("Using cached features (skipping extraction)")
    else:
        # ── 3b. Extract features ────────────────────────────────
        logger.info("Loading SSD-1B model...")
        gen_started = time.perf_counter()
        generator = BaseGenerator(
            model_name=args.model,
            device=str(device),
            dtype=torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32,
        )
        logger.info("SSD-1B loaded in %.1fs", time.perf_counter() - gen_started)

        logger.info("Caching frozen features for train set...")
        cache_started = time.perf_counter()
        train_contract, train_features, train_target = _cache_features_to_memory(
            generator, data_dir,
            names=train_names, resolution=args.resolution,
            prompt=prompt, timestep=args.timestep,
            mapper=mapper, storage_dtype=storage_dtype,
            desc="[train]",
        )
        logger.info("Train features cached in %.1fs", time.perf_counter() - cache_started)

        logger.info("Caching frozen features for val set...")
        val_started = time.perf_counter()
        _, val_features, val_target = _cache_features_to_memory(
            generator, data_dir,
            names=val_names, resolution=args.resolution,
            prompt=prompt, timestep=args.timestep,
            mapper=mapper, storage_dtype=storage_dtype,
            desc="[val]",
        )
        logger.info("Val features cached in %.1fs", time.perf_counter() - val_started)

        _save_feature_cache(
            cache_dir,
            contract=train_contract,
            train_features=train_features,
            train_target=train_target,
            train_names=train_names,
            val_features=val_features,
            val_target=val_target,
            val_names=val_names,
            metadata={
                "model": args.model,
                "resolution": args.resolution,
                "timestep": args.timestep,
                "feature_storage_dtype": args.feature_storage_dtype,
                "colormap": colormap,
            },
        )

        del generator
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Feature contract: %s", train_contract.to_dict())
    logger.info("Train feature shapes: %s", [list(f.shape) for f in train_features])

    if args.max_val_samples > 0 and args.max_val_samples < val_target.shape[0]:
        val_features = tuple(f[: args.max_val_samples] for f in val_features)
        val_target = val_target[: args.max_val_samples]

    # ── 4. Build V5 probe head with skip connections ───────────
    head = ScalarDepthProbeV5(
        train_contract,
        width=args.width,
        upsample_stages=args.upsample_stages,
    ).to(device=device, dtype=torch.float32)
    trainable_params = sum(p.numel() for p in head.parameters())
    logger.info("ScalarDepthProbeV5 trainable parameters: %d (%.2fK)", trainable_params, trainable_params / 1000)

    # ── 4b. Resume from checkpoint if specified ────────────────
    resume_epoch = 0
    resume_history: list[dict[str, Any]] = []
    resume_best_score = -float("inf")
    resume_best_epoch = 0
    resume_best_metrics = initial_metrics if 'initial_metrics' in dir() else None
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if resume_path.is_file():
            logger.info("Resuming from checkpoint: %s", resume_path)
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            head.load_state_dict(ckpt["head_state_dict"])
            resume_epoch = ckpt.get("epoch", 0)
            resume_best_metrics = ckpt.get("metrics", None)
            if resume_best_metrics:
                resume_best_score = resume_best_metrics.get("pearson", 0) + resume_best_metrics.get("spearman", 0) - resume_best_metrics.get("mae", 0)
                resume_best_epoch = resume_epoch
            logger.info("  Loaded weights from epoch %d (MAE=%.5f Pearson=%.5f)",
                        resume_epoch,
                        resume_best_metrics.get("mae", 0) if resume_best_metrics else 0,
                        resume_best_metrics.get("pearson", 0) if resume_best_metrics else 0)

            # Load existing metrics.json history if available
            existing_metrics_path = output_dir / "metrics.json"
            if existing_metrics_path.is_file():
                import json as _json
                try:
                    existing = _json.loads(existing_metrics_path.read_text())
                    resume_history = existing.get("history", [])
                    logger.info("  Loaded %d epochs of history from existing metrics.json", len(resume_history))
                except Exception:
                    pass
        else:
            logger.warning("Resume checkpoint not found: %s — starting fresh", resume_path)

    # ── 5. Compute training schedule ───────────────────────────
    num_train = train_target.shape[0]
    steps_per_epoch = num_train // args.batch_size
    # For resume: adjust total_steps and global_step to continue LR schedule correctly
    start_epoch = max(args.start_epoch, resume_epoch + 1)
    global_step = (start_epoch - 1) * steps_per_epoch
    total_steps = steps_per_epoch * args.epochs
    output_size = tuple(int(v) for v in train_target.shape[-2:])

    logger.info("Training plan:")
    logger.info("  samples/epoch:  %d", num_train)
    logger.info("  steps/epoch:    %d (batch_size=%d)", steps_per_epoch, args.batch_size)
    logger.info("  total steps:    %d (%d epochs max)", total_steps, args.epochs)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: _lr_schedule(s, args.warmup_steps, total_steps),
    )

    # Advance scheduler to the correct step for resume
    if global_step > 0:
        for _ in range(global_step):
            scheduler.step()
        logger.info("  Scheduler advanced to step %d (lr=%.6f)", global_step, optimizer.param_groups[0]["lr"])

    # ── 6. EMA ─────────────────────────────────────────────────
    ema = EMAProbe(head, decay=args.ema_decay)

    # ── 7. Initial evaluation ──────────────────────────────────
    initial_metrics = _evaluate_on_set(
        head, val_features, val_target,
        batch_size=args.batch_size, device=device,
    )
    logger.info(
        "Initial val: MAE=%.5f RMSE=%.5f Pearson=%.5f Spearman=%.5f",
        initial_metrics["mae"], initial_metrics["rmse"],
        initial_metrics["pearson"], initial_metrics["spearman"],
    )

    # ── 8. Epoch-based training loop with early stopping ───────
    best_score = resume_best_score
    best_step = global_step
    best_epoch = resume_best_epoch
    best_state = _clone_state_dict(head)
    best_metrics = resume_best_metrics if resume_best_metrics else initial_metrics
    history: list[dict[str, Any]] = list(resume_history)
    epochs_no_improve = 0
    early_stopped = False

    logger.info(
        "Starting V5 training: epochs %d→%d (max %d), early stopping patience=%d",
        start_epoch, args.epochs, args.epochs, args.patience,
    )
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        head.train()
        epoch_indices = torch.randperm(num_train)
        epoch_losses: list[float] = []

        for batch_start in range(0, num_train - args.batch_size + 1, args.batch_size):
            global_step += 1
            indices = epoch_indices[batch_start : batch_start + args.batch_size].tolist()
            batch_features = tuple(
                feat[indices].to(device=device, dtype=torch.float32)
                for feat in train_features
            )
            batch_target = train_target[indices].to(device=device, dtype=torch.float32)

            # Augmentation
            if args.aug_flip_prob > 0 or args.aug_flip_v_prob > 0 or args.aug_brightness > 0:
                batch_features, batch_target = _augment_batch(
                    batch_features, batch_target, args.batch_size,
                    flip_h_prob=args.aug_flip_prob,
                    flip_v_prob=args.aug_flip_v_prob,
                    brightness=args.aug_brightness,
                )

            prediction = head(batch_features, output_size=output_size)
            loss = v5_composite_loss(
                prediction, batch_target,
                gradient_weight=args.gradient_weight,
                pearson_weight=args.pearson_weight,
                ssim_weight=args.ssim_weight,
                edge_weight=args.edge_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update EMA
            ema.update(head)

            epoch_losses.append(float(loss.detach().cpu()))

        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - train_start
        logger.info(
            "Epoch %d/%d | steps=%d | avg_loss=%.5f lr=%.6f | %.1fs (%.1f step/s)",
            epoch, args.epochs, global_step, avg_loss, lr, elapsed,
            global_step / max(elapsed, 1),
        )

        # Evaluate
        if epoch % args.eval_every_epoch == 0 or epoch == args.epochs:
            # Evaluate with EMA weights
            ema_probe = ema.apply_to(head)
            metrics = _evaluate_on_set(
                ema_probe, val_features, val_target,
                batch_size=args.batch_size, device=device,
            )
            score = metrics["pearson"] + metrics["spearman"] - metrics["mae"]
            record = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": avg_loss,
                "lr": lr,
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_pearson": metrics["pearson"],
                "val_spearman": metrics["spearman"],
                "epochs_no_improve": epochs_no_improve,
            }
            history.append(record)
            logger.info(
                "  VAL @ epoch %d: MAE=%.5f RMSE=%.5f Pearson=%.5f Spearman=%.5f (score=%.5f, patience=%d/%d)",
                epoch, metrics["mae"], metrics["rmse"],
                metrics["pearson"], metrics["spearman"], score,
                epochs_no_improve, args.patience,
            )
            if score > best_score:
                best_score = score
                best_step = global_step
                best_epoch = epoch
                best_state = {name: tensor.cpu().clone() for name, tensor in ema_probe.state_dict().items()}
                best_metrics = metrics
                best_path = output_dir / "best_probe.pt"
                torch.save(
                    {
                        "head_state_dict": best_state,
                        "feature_contract": train_contract.to_dict(),
                        "model": args.model,
                        "prompt": prompt,
                        "colormap": colormap,
                        "resolution": args.resolution,
                        "timestep": args.timestep,
                        "width": args.width,
                        "upsample_stages": args.upsample_stages,
                        "epoch": best_epoch,
                        "step": best_step,
                        "metrics": best_metrics,
                        "trainable_parameters": trainable_params,
                        "version": "v5",
                    },
                    best_path,
                )
                logger.info("  ★ New best (score=%.5f) — saved to %s", best_score, best_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    logger.info(
                        "  ‼ Early stopping triggered after %d epochs without improvement",
                        epochs_no_improve,
                    )
                    early_stopped = True
                    break

    # ── 9. Final evaluation with best checkpoint ───────────────
    head.load_state_dict(best_state)
    head.eval()
    final_metrics = _evaluate_on_set(
        head, val_features, val_target,
        batch_size=args.batch_size, device=device,
    )
    train_metrics = _evaluate_on_set(
        head, train_features, train_target,
        batch_size=args.batch_size, device=device,
    )

    # ── 10. Save final checkpoint ──────────────────────────────
    final_path = output_dir / "final_probe.pt"
    torch.save(
        {
            "head_state_dict": _clone_state_dict(head),
            "feature_contract": train_contract.to_dict(),
            "model": args.model,
            "prompt": prompt,
            "colormap": colormap,
            "resolution": args.resolution,
            "timestep": args.timestep,
            "width": args.width,
            "upsample_stages": args.upsample_stages,
            "epoch": epoch if not early_stopped else best_epoch,
            "step": global_step,
            "metrics": final_metrics,
            "trainable_parameters": trainable_params,
            "version": "v5",
        },
        final_path,
    )

    # ── 11. Save visualizations ────────────────────────────────
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    val_dataset = VisionTaskDataset(
        str(data_dir / "depth"),
        resolution=args.resolution,
        task="depth",
    )
    by_name = {
        Path(s["image_path"]).name: i for i, s in enumerate(val_dataset.samples)
    }
    vis_count = min(16, len(val_names))
    vis_images = []
    vis_targets_rgb = []
    vis_preds = []
    with torch.no_grad():
        for vis_idx, name in enumerate(val_names[:vis_count]):
            sample = val_dataset[by_name[name]]
            vis_images.append(sample["image"])
            vis_targets_rgb.append(sample["target_rgb"])
            feat_batch = tuple(
                f[vis_idx : vis_idx + 1].to(device=device, dtype=torch.float32)
                for f in val_features
            )
            pred = head(feat_batch, output_size=output_size)
            vis_preds.append(pred.cpu())

    vis_images_tensor = torch.stack(vis_images)
    vis_targets_tensor = torch.stack(vis_targets_rgb)
    vis_preds_tensor = torch.cat(vis_preds, dim=0)

    montage_paths = _save_montages(
        vis_dir,
        names=val_names[:vis_count],
        images_tensor=vis_images_tensor,
        target_rgb=vis_targets_tensor,
        prediction=vis_preds_tensor,
        colormap=colormap,
        max_count=vis_count,
    )

    # ── 12. Save metrics JSON ──────────────────────────────────
    runtime = time.perf_counter() - started
    result = {
        "model": args.model,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "resolution": args.resolution,
        "seed": args.seed,
        "version": "v5_enhanced",
        "timestep": args.timestep,
        "colormap": colormap,
        "prompt": prompt,
        "epochs_requested": args.epochs,
        "epochs_executed": epoch,
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "total_steps": global_step,
        "trainable_parameters": trainable_params,
        "feature_contract": train_contract.to_dict(),
        "feature_shapes": [list(f.shape) for f in train_features],
        "feature_storage_dtype": args.feature_storage_dtype,
        "splits": {
            "train_count": len(train_names),
            "val_count": len(val_names),
        },
        "training_config": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "gradient_weight": args.gradient_weight,
            "pearson_weight": args.pearson_weight,
            "ssim_weight": args.ssim_weight,
            "edge_weight": args.edge_weight,
            "width": args.width,
            "upsample_stages": args.upsample_stages,
            "aug_flip_prob": args.aug_flip_prob,
            "aug_flip_v_prob": args.aug_flip_v_prob,
            "aug_brightness": args.aug_brightness,
            "patience": args.patience,
            "ema_decay": args.ema_decay,
        },
        "best_val_metrics": best_metrics,
        "final_val_metrics": final_metrics,
        "train_metrics": train_metrics,
        "history": history,
        "runtime_seconds": runtime,
        "checkpoints": {
            "best": str((output_dir / "best_probe.pt").resolve()),
            "final": str(final_path.resolve()),
        },
        "visualizations": montage_paths,
    }
    write_json_report(output_dir / "metrics.json", result)

    # ── Summary ────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("V5 Training Complete%s", " (Early Stopped)" if early_stopped else "")
    logger.info("=" * 70)
    logger.info("  Best epoch:      %d (step %d)", best_epoch, best_step)
    logger.info("  Best val MAE:    %.5f", best_metrics["mae"])
    logger.info("  Best val RMSE:   %.5f", best_metrics["rmse"])
    logger.info("  Best val Pearson: %.5f", best_metrics["pearson"])
    logger.info("  Best val Spearman:%.5f", best_metrics["spearman"])
    logger.info("  Train MAE:       %.5f (sanity)", train_metrics["mae"])
    logger.info("  Train Pearson:   %.5f (sanity)", train_metrics["pearson"])
    logger.info("  Trainable params: %d (%.2fK)", trainable_params, trainable_params / 1000)
    logger.info("  Runtime:         %.1fs (%.1fmin)", runtime, runtime / 60)
    logger.info("=" * 70)

    # Comparison table
    logger.info("\nComparison:")
    logger.info("  %-35s %10s %10s %10s", "Model", "MAE", "Pearson", "Params")
    logger.info("  %-35s %10s %10s %10s", "-" * 35, "-" * 10, "-" * 10, "-" * 10)
    logger.info("  %-35s %10.5f %10.5f %10s",
                "SSD-1B Probe V5 (this)", best_metrics["mae"],
                best_metrics["pearson"], f"{trainable_params / 1e6:.2f}M")
    logger.info("  %-35s %10s %10s %10s", "SSD-1B Probe V4 (prev)", "0.0722", "0.8883", "~1.6M")
    logger.info("  %-35s %10s %10s %10s", "SSD-1B Probe V3", "0.16680", "0.70250", "0.51M")
    logger.info("  %-35s %10s %10s %10s", "Frozen Probe (baseline)", "0.1296", "0.7715", "~0.3M")


if __name__ == "__main__":
    main()
