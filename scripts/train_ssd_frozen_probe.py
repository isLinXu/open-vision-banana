#!/usr/bin/env python3
"""Train SSD-1B frozen-feature depth probe on full v3 dataset at 512px.

This script trains the ScalarDepthProbe (a lightweight FPN decoder) on top
of frozen SSD-1B VAE + UNet features for depth estimation.  Only the probe
head is trainable (~hundreds of K parameters); the ~1B SSD-1B backbone is
fully frozen.

Pipeline:
  1. Load SSD-1B via BaseGenerator, freeze VAE + UNet
  2. Pre-cache frozen features for all train + val images (one-time cost)
  3. Train ScalarDepthProbe with mini-batch SGD + cosine LR schedule
  4. Evaluate on full val split every N steps; save best checkpoint
  5. Final evaluation + metrics JSON + visualization montages

Usage:
    python scripts/train_ssd_frozen_probe.py \
        --data-dir data/vb_snvc_train_v3 \
        --output-dir output/ssd_frozen_probe_512 \
        --resolution 512 \
        --steps 3000 \
        --batch-size 4 \
        --eval-every 100
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
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
    ScalarDepthProbe,
    evaluate_scalar_depth_predictions,
    scalar_depth_probe_loss,
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
    """Read train/val depth names from the split manifest."""
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
# Feature caching
# ──────────────────────────────────────────────────────────────


def _cache_features(
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
    """Extract frozen VAE+UNet features for all named images.

    Returns (contract, features_tuple, target_scalar) where:
      - features_tuple: tuple of [N, C, H, W] tensors on CPU
      - target_scalar: [N, 1, H, W] normalized depth on CPU
    """
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
            logger.info(
                "  %s cached %d/%d images", desc, position, len(names)
            )

    if contract is None or levels is None:
        raise RuntimeError("No frozen features were extracted")

    features = tuple(torch.cat(level, dim=0) for level in levels)
    target_tensor = torch.cat(targets, dim=0)
    return contract, features, target_tensor


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────


def _build_batch_schedule(
    num_samples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> torch.Tensor:
    """Generate a shuffled batch schedule as [steps, batch_size] indices."""
    g = torch.Generator().manual_seed(seed)
    schedule = torch.empty(steps, batch_size, dtype=torch.long)
    for step in range(steps):
        indices = torch.randint(0, num_samples, (batch_size,), generator=g)
        schedule[step] = indices
    return schedule


@torch.no_grad()
def _evaluate_on_set(
    head: ScalarDepthProbe,
    features: tuple[torch.Tensor, ...],
    target: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate probe head on a cached feature set."""
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
    """Save side-by-side montages: input | target | prediction."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="segmind/SSD-1B")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--upsample-stages", type=int, default=3)
    parser.add_argument("--gradient-weight", type=float, default=0.5)
    parser.add_argument("--pearson-weight", type=float, default=0.3)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--feature-storage-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-val-samples", type=int, default=0, help="0=all val")
    args = parser.parse_args()

    # Validate
    positive = {
        "resolution": args.resolution,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "lr": args.lr,
        "width": args.width,
    }
    if any(v <= 0 for v in positive.values()):
        parser.error(f"positive values required: {positive}")
    if args.upsample_stages < 0 or args.timestep < 0:
        parser.error("--upsample-stages and --timestep must be non-negative")
    if args.gradient_weight < 0 or args.pearson_weight < 0:
        parser.error("loss weights must be non-negative")

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
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("SSD-1B Frozen Probe Depth Training")
    logger.info("=" * 70)
    logger.info("  model:          %s", args.model)
    logger.info("  data_dir:       %s", data_dir)
    logger.info("  output_dir:     %s", output_dir)
    logger.info("  resolution:     %d", args.resolution)
    logger.info("  steps:          %d", args.steps)
    logger.info("  batch_size:     %d", args.batch_size)
    logger.info("  lr:             %s", args.lr)
    logger.info("  width:          %d", args.width)
    logger.info("  pearson_weight: %s", args.pearson_weight)
    logger.info("  device:         %s", device)
    logger.info("  storage_dtype:  %s", args.feature_storage_dtype)
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
    logger.info("Prompt: %s", prompt)

    # ── 3. Load SSD-1B generator ───────────────────────────────
    logger.info("Loading SSD-1B model...")
    gen_started = time.perf_counter()
    generator = BaseGenerator(
        model_name=args.model,
        device=str(device),
        dtype=torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32,
    )
    logger.info("SSD-1B loaded in %.1fs", time.perf_counter() - gen_started)

    storage_dtype = getattr(torch, args.feature_storage_dtype)

    # ── 4. Cache frozen features ───────────────────────────────
    logger.info("Caching frozen features for train set...")
    cache_started = time.perf_counter()
    train_contract, train_features, train_target = _cache_features(
        generator,
        data_dir,
        names=train_names,
        resolution=args.resolution,
        prompt=prompt,
        timestep=args.timestep,
        mapper=mapper,
        storage_dtype=storage_dtype,
        desc="[train]",
    )
    logger.info("Train features cached in %.1fs", time.perf_counter() - cache_started)
    logger.info("Feature contract: %s", train_contract.to_dict())
    logger.info(
        "Train feature shapes: %s",
        [list(f.shape) for f in train_features],
    )

    logger.info("Caching frozen features for val set...")
    val_started = time.perf_counter()
    val_contract, val_features, val_target = _cache_features(
        generator,
        data_dir,
        names=val_names,
        resolution=args.resolution,
        prompt=prompt,
        timestep=args.timestep,
        mapper=mapper,
        storage_dtype=storage_dtype,
        desc="[val]",
    )
    logger.info("Val features cached in %.1fs", time.perf_counter() - val_started)

    if val_contract != train_contract:
        raise ValueError("Train and val feature contracts differ")

    # Free SSD-1B memory — only need cached features from here
    del generator
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # Optionally subsample val for speed
    if args.max_val_samples > 0 and args.max_val_samples < val_target.shape[0]:
        logger.info(
            "Subsampling val from %d to %d for faster eval",
            val_target.shape[0],
            args.max_val_samples,
        )
        val_features = tuple(f[: args.max_val_samples] for f in val_features)
        val_target = val_target[: args.max_val_samples]

    # ── 5. Build probe head ────────────────────────────────────
    head = ScalarDepthProbe(
        train_contract,
        width=args.width,
        upsample_stages=args.upsample_stages,
    ).to(device=device, dtype=torch.float32)
    trainable_params = sum(p.numel() for p in head.parameters())
    logger.info("ScalarDepthProbe trainable parameters: %d (%.2fK)", trainable_params, trainable_params / 1000)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.lr * 0.01,
    )

    # ── 6. Initial evaluation ──────────────────────────────────
    output_size = tuple(int(v) for v in train_target.shape[-2:])
    initial_metrics = _evaluate_on_set(
        head, val_features, val_target,
        batch_size=args.batch_size, device=device,
    )
    logger.info(
        "Initial val: MAE=%.5f RMSE=%.5f Pearson=%.5f Spearman=%.5f",
        initial_metrics["mae"],
        initial_metrics["rmse"],
        initial_metrics["pearson"],
        initial_metrics["spearman"],
    )

    # ── 7. Training loop ───────────────────────────────────────
    num_train = train_target.shape[0]
    schedule = _build_batch_schedule(
        num_samples=num_train,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed + 1000,
    )

    best_score = -float("inf")
    best_step = 0
    best_state = _clone_state_dict(head)
    best_metrics = initial_metrics
    history: list[dict[str, Any]] = []
    train_losses: list[float] = []

    logger.info("Starting training: %d steps, batch_size=%d", args.steps, args.batch_size)

    for step in range(1, args.steps + 1):
        head.train()
        indices = schedule[step - 1].tolist()
        batch_features = tuple(
            feat[indices].to(device=device, dtype=torch.float32)
            for feat in train_features
        )
        batch_target = train_target[indices].to(device=device, dtype=torch.float32)

        prediction = head(batch_features, output_size=output_size)
        loss = scalar_depth_probe_loss(
            prediction,
            batch_target,
            gradient_weight=args.gradient_weight,
            pearson_weight=args.pearson_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(float(loss.detach().cpu()))

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = _evaluate_on_set(
                head, val_features, val_target,
                batch_size=args.batch_size, device=device,
            )
            score = metrics["pearson"] + metrics["spearman"] - metrics["mae"]
            record = {
                "step": step,
                "train_loss": float(np.mean(train_losses[-args.eval_every:])),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_pearson": metrics["pearson"],
                "val_spearman": metrics["spearman"],
            }
            history.append(record)
            logger.info(
                "Step %5d/%d | loss=%.5f lr=%.6f | val: MAE=%.5f RMSE=%.5f "
                "Pearson=%.5f Spearman=%.5f",
                step,
                args.steps,
                record["train_loss"],
                record["lr"],
                metrics["mae"],
                metrics["rmse"],
                metrics["pearson"],
                metrics["spearman"],
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_state = _clone_state_dict(head)
                best_metrics = metrics
                # Save best checkpoint
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
                        "step": best_step,
                        "metrics": best_metrics,
                        "trainable_parameters": trainable_params,
                    },
                    best_path,
                )
                logger.info("  ★ New best (score=%.5f) — saved to %s", best_score, best_path)

    # ── 8. Final evaluation with best checkpoint ───────────────
    head.load_state_dict(best_state)
    head.eval()
    final_metrics = _evaluate_on_set(
        head, val_features, val_target,
        batch_size=args.batch_size, device=device,
    )

    # Also evaluate on train set (sanity check)
    train_metrics = _evaluate_on_set(
        head, train_features, train_target,
        batch_size=args.batch_size, device=device,
    )

    # ── 9. Save final checkpoint ───────────────────────────────
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
            "step": best_step,
            "metrics": final_metrics,
            "trainable_parameters": trainable_params,
        },
        final_path,
    )

    # ── 10. Save visualizations ────────────────────────────────
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Generate prediction montages for first N val samples
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
            # Predict using cached features
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

    # ── 11. Save metrics JSON ──────────────────────────────────
    runtime = time.perf_counter() - started
    result = {
        "model": args.model,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "resolution": args.resolution,
        "seed": args.seed,
        "timestep": args.timestep,
        "colormap": colormap,
        "prompt": prompt,
        "steps_requested": args.steps,
        "best_step": best_step,
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
            "gradient_weight": args.gradient_weight,
            "pearson_weight": args.pearson_weight,
            "width": args.width,
            "upsample_stages": args.upsample_stages,
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
    logger.info("Training Complete")
    logger.info("=" * 70)
    logger.info("  Best step:       %d", best_step)
    logger.info("  Best val MAE:    %.5f", best_metrics["mae"])
    logger.info("  Best val RMSE:   %.5f", best_metrics["rmse"])
    logger.info("  Best val Pearson: %.5f", best_metrics["pearson"])
    logger.info("  Best val Spearman:%.5f", best_metrics["spearman"])
    logger.info("  Train MAE:       %.5f (sanity)", train_metrics["mae"])
    logger.info("  Train Pearson:   %.5f (sanity)", train_metrics["pearson"])
    logger.info("  Trainable params: %d (%.2fK)", trainable_params, trainable_params / 1000)
    logger.info("  Runtime:         %.1fs (%.1fmin)", runtime, runtime / 60)
    logger.info("  Best checkpoint: %s", result["checkpoints"]["best"])
    logger.info("=" * 70)

    # Print comparison table
    logger.info("\nComparison with baseline:")
    logger.info("  %-25s %10s %10s %10s", "Model", "MAE", "Pearson", "Params")
    logger.info("  %-25s %10s %10s %10s", "-" * 25, "-" * 10, "-" * 10, "-" * 10)
    logger.info(
        "  %-25s %10.5f %10.5f %10s",
        "SSD-1B Frozen Probe (this)",
        best_metrics["mae"],
        best_metrics["pearson"],
        f"{trainable_params / 1e6:.2f}M",
    )
    logger.info(
        "  %-25s %10s %10s %10s",
        "TinyVisionBanana (prev)",
        "0.1873",
        "0.6198",
        "2.99M",
    )
    logger.info(
        "  %-25s %10s %10s %10s",
        "Frozen Probe (baseline)",
        "0.1296",
        "0.7715",
        "~0.3M",
    )


if __name__ == "__main__":
    main()
