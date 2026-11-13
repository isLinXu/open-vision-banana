#!/usr/bin/env python3
"""Gate RGB-to-normal regression on frozen local DINOv2 ViT-S/14 features.

The original v5 normal PNG targets were shown to be collapsed.  This program
does not reuse them.  It trains a small unit-vector head on the locally stored
DIODE train split, validates against DIODE's separate val split, and compares
the output to a global train-only constant-normal baseline.  It is a gate for
future training, not a change to the deployed specialist normal estimator.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from audit_real_normal_datasets import NormalPair, decode_normal, discover_diode
from segmentation_dinov2_probe import DINO_FEATURE_DIM, DINO_PATCH_SIZE, LocalDINOv2ViTS14
from segmentation_resnet18_probe import load_input, select_device, set_seed


DEFAULT_DIODE_ROOT = Path("data/raw/diode_eval/data/diode-normal-256")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diode-root", type=Path, default=DEFAULT_DIODE_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--dino-repository", type=Path, default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")
    parser.add_argument("--dino-weights", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--min-valid-patch-fraction", type=float, default=0.2)
    parser.add_argument("--normal-length-tolerance", type=float, default=0.12)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size < DINO_PATCH_SIZE or args.image_size % DINO_PATCH_SIZE:
        raise ValueError(f"--image-size must be a positive multiple of {DINO_PATCH_SIZE}")
    if args.steps < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("--steps, --batch-size, and --eval-batch-size must be positive")
    if args.train_samples < 1 or args.val_samples < 1:
        raise ValueError("--train-samples and --val-samples must be positive")
    if not 0.0 < args.min_valid_patch_fraction <= 1.0:
        raise ValueError("--min-valid-patch-fraction must be in (0, 1]")
    if not 0.0 < args.normal_length_tolerance < 1.0:
        raise ValueError("--normal-length-tolerance must be in (0, 1)")
    if not args.dino_repository.is_dir() or not args.dino_weights.is_file():
        raise FileNotFoundError("Local DINOv2 repository or weights are missing")


def select_pairs(root: Path, count: int) -> list[NormalPair]:
    pairs = discover_diode(root)
    if count < 1 or count > len(pairs):
        raise ValueError(f"requested {count} normal pairs, found {len(pairs)} under {root}")
    indices = np.linspace(0, len(pairs) - 1, count, dtype=np.int64)
    return [pairs[int(index)] for index in indices]


def patch_normal_targets(normal: np.ndarray, valid: np.ndarray, grid_size: int,
                         min_valid_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Average valid normal vectors into square DINO patch targets."""
    height, width, channels = normal.shape
    if channels != 3 or valid.shape != (height, width) or height != width or height % grid_size:
        raise ValueError("normal/valid shapes must be square and divisible by grid size")
    stride = height // grid_size
    blocks = normal.reshape(grid_size, stride, grid_size, stride, 3).transpose(0, 2, 1, 3, 4)
    mask = valid.reshape(grid_size, stride, grid_size, stride).transpose(0, 2, 1, 3)
    counts = mask.sum(axis=(2, 3))
    vectors = (blocks * mask[..., None]).sum(axis=(2, 3)) / np.maximum(counts[..., None], 1)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    target = vectors / np.maximum(lengths, 1e-8)
    usable = counts >= min_valid_fraction * stride * stride
    return target.reshape(-1, 3).astype(np.float32), usable.reshape(-1)


def load_target(pair: NormalPair, image_size: int, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    return decode_normal(pair.normal_path, image_size, tolerance)


def cache_features(pairs: list[NormalPair], split: str, cache_dir: Path, backbone: nn.Module,
                   device: torch.device, image_size: int, tolerance: float,
                   min_valid_fraction: float, eval_batch_size: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, int]:
    """Create/reuse DINO features and patch targets without retaining full maps."""
    features, targets, valid_masks = [], [], []
    direction_sum = np.zeros(3, dtype=np.float64)
    direction_count = 0
    grid_size = image_size // DINO_PATCH_SIZE
    split_cache = cache_dir / split
    split_cache.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pairs), eval_batch_size):
        batch_pairs = pairs[start:start + eval_batch_size]
        batch_inputs, cached, cache_paths, batch_targets, batch_masks = [], [], [], [], []
        for offset, pair in enumerate(batch_pairs):
            index = start + offset
            feature_path = split_cache / f"{index:05d}.pt"
            feature = None
            if feature_path.exists():
                try:
                    feature = torch.load(feature_path, map_location="cpu", weights_only=True)
                    if (not isinstance(feature, torch.Tensor) or feature.ndim != 2
                            or feature.shape != (grid_size * grid_size, DINO_FEATURE_DIM)):
                        feature = None
                except (OSError, RuntimeError):
                    feature = None
            normal, valid = load_target(pair, image_size, tolerance)
            token_targets, token_valid = patch_normal_targets(normal, valid, grid_size, min_valid_fraction)
            direction_sum += normal[valid].sum(axis=0, dtype=np.float64)
            direction_count += int(valid.sum())
            batch_inputs.append(load_input(str(pair.input_path), image_size)[0])
            cached.append(feature)
            cache_paths.append(feature_path)
            batch_targets.append(token_targets)
            batch_masks.append(token_valid)
        missing = [index for index, feature in enumerate(cached) if feature is None]
        if missing:
            with torch.no_grad():
                produced = backbone(torch.stack(batch_inputs).to(device)).cpu()
            for index in missing:
                temporary = cache_paths[index].with_name(f".{cache_paths[index].name}.{os.getpid()}.tmp")
                # Save an independent sample tensor. Slicing a batch leaves a
                # view whose backing storage would otherwise inflate every
                # cache file to the complete batch size.
                torch.save(produced[index].clone(), temporary)
                temporary.replace(cache_paths[index])
                cached[index] = produced[index].clone()
            if device.type == "mps":
                torch.mps.empty_cache()
        batch_features = torch.stack([feature.float() for feature in cached])
        features.append(batch_features)
        targets.extend(batch_targets)
        valid_masks.extend(batch_masks)
        print(f"[features] {split} {min(start + len(batch_pairs), len(pairs))}/{len(pairs)}", flush=True)
    return (
        torch.cat(features),
        np.stack(targets),
        np.stack(valid_masks),
        direction_sum,
        direction_count,
    )


class NormalTokenHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.layers(features), dim=-1, eps=1e-8)


@torch.no_grad()
def predict(head: nn.Module, features: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        outputs.append(head(features[start:start + batch_size].to(device)).cpu())
    return torch.cat(outputs).numpy().astype(np.float32)


def angular_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    cosine = np.sum(prediction * target, axis=-1)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def metric_summary(errors: list[np.ndarray]) -> dict[str, float | int]:
    values = np.concatenate(errors) if errors else np.empty(0, dtype=np.float32)
    return {
        "mean_angular_error_deg": float(values.mean()) if len(values) else 0.0,
        "median_angular_error_deg": float(np.median(values)) if len(values) else 0.0,
        "p95_angular_error_deg": float(np.quantile(values, 0.95)) if len(values) else 0.0,
        "within_11_25deg_fraction": float((values < 11.25).mean()) if len(values) else 0.0,
        "within_22_5deg_fraction": float((values < 22.5).mean()) if len(values) else 0.0,
        "within_30deg_fraction": float((values < 30.0).mean()) if len(values) else 0.0,
        "valid_count": int(len(values)),
    }


def grid_metrics(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    return metric_summary([angular_errors(prediction[index][valid[index]], target[index][valid[index]]) for index in range(len(prediction))])


def expanded_normal_grid(prediction: np.ndarray, image_size: int) -> np.ndarray:
    grid_size = int(round(prediction.shape[1] ** 0.5))
    if grid_size * grid_size != prediction.shape[1] or image_size % grid_size:
        raise ValueError("prediction token count is incompatible with requested image size")
    stride = image_size // grid_size
    grid = prediction.reshape(-1, grid_size, grid_size, 3)
    return np.repeat(np.repeat(grid, stride, axis=1), stride, axis=2)


def expanded_metrics(prediction: np.ndarray, pairs: list[NormalPair], image_size: int,
                     tolerance: float) -> dict[str, float | int]:
    expanded = expanded_normal_grid(prediction, image_size)
    errors = []
    for pred, pair in zip(expanded, pairs, strict=True):
        target, valid = load_target(pair, image_size, tolerance)
        errors.append(angular_errors(pred[valid], target[valid]))
    return metric_summary(errors)


def normal_to_rgb(normal: np.ndarray) -> Image.Image:
    encoded = np.rint(np.clip((normal + 1.0) * 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(encoded, mode="RGB")


def write_panel(path: Path, pair: NormalPair, prediction: np.ndarray, image_size: int,
                tolerance: float) -> None:
    with Image.open(pair.input_path) as image:
        source = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
    target, _ = load_target(pair, image_size, tolerance)
    output = normal_to_rgb(expanded_normal_grid(prediction[None], image_size)[0])
    target_rgb = normal_to_rgb(target)
    gap, label_height = 8, 28
    canvas = Image.new("RGB", (3 * image_size + 4 * gap, image_size + label_height + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(("RGB input", "DIODE normal", "DINOv2 prediction")):
        draw.text((gap + index * (image_size + gap), 6), label, fill="black")
    for index, image in enumerate((source, target_rgb, output)):
        canvas.paste(image, (gap + index * (image_size + gap), label_height + gap))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    set_seed(args.seed)
    device = select_device(args.device)
    root = args.diode_root.expanduser().resolve()
    train_pairs = select_pairs(root / "train", args.train_samples)
    val_pairs = select_pairs(root / "val", args.val_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.feature_cache or args.output_dir / "features").resolve()
    backbone = LocalDINOv2ViTS14(args.dino_repository.resolve(), args.dino_weights.resolve()).to(device).eval()
    train_features, train_target, train_valid, train_direction_sum, train_direction_count = cache_features(
        train_pairs, "train", cache_dir, backbone, device, args.image_size, args.normal_length_tolerance,
        args.min_valid_patch_fraction, args.eval_batch_size,
    )
    val_features, val_target, val_valid, _, _ = cache_features(
        val_pairs, "val", cache_dir, backbone, device, args.image_size, args.normal_length_tolerance,
        args.min_valid_patch_fraction, args.eval_batch_size,
    )
    del backbone
    if device.type == "mps":
        torch.mps.empty_cache()

    train_targets = torch.from_numpy(train_target)
    train_masks = torch.from_numpy(train_valid)
    head = NormalTokenHead(DINO_FEATURE_DIM, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_error, best_step, history = None, float("inf"), 0, []
    report_every = max(1, min(100, args.steps // 10))
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train_features), (min(args.batch_size, len(train_features)),))
        batch_features = train_features[indices].to(device)
        batch_targets = train_targets[indices].to(device)
        batch_masks = train_masks[indices].to(device)
        if not bool(batch_masks.any()):
            continue
        prediction = head(batch_features)
        loss = (1.0 - (prediction * batch_targets).sum(dim=-1))[batch_masks].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % report_every == 0 or step == args.steps:
            train_prediction = predict(head, train_features, device, args.eval_batch_size)
            val_prediction = predict(head, val_features, device, args.eval_batch_size)
            train_grid = grid_metrics(train_prediction, train_target, train_valid)
            val_grid = grid_metrics(val_prediction, val_target, val_valid)
            val_expanded = expanded_metrics(val_prediction, val_pairs, args.image_size, args.normal_length_tolerance)
            history.append({
                "step": step,
                "loss": float(loss.item()),
                "train_grid": train_grid,
                "val_grid": val_grid,
                "val_expanded_pixel": val_expanded,
            })
            if val_expanded["mean_angular_error_deg"] < best_error:
                best_error = float(val_expanded["mean_angular_error_deg"])
                best_step = step
                best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            print(
                f"[train] step={step}/{args.steps} loss={loss.item():.5f} "
                f"val_grid_deg={val_grid['mean_angular_error_deg']:.2f} "
                f"val_expanded_deg={val_expanded['mean_angular_error_deg']:.2f}",
                flush=True,
            )

    assert best_state is not None
    head.load_state_dict(best_state)
    torch.save(
        {"head": best_state, "step": best_step, "best_val_expanded_angular_error_deg": best_error,
         "backbone": "dinov2_vits14", "image_size": args.image_size},
        args.output_dir / "normal_head.pt",
    )
    train_prediction = predict(head, train_features, device, args.eval_batch_size)
    val_prediction = predict(head, val_features, device, args.eval_batch_size)
    train_direction = train_direction_sum / max(train_direction_count, 1)
    train_direction /= max(float(np.linalg.norm(train_direction)), 1e-8)
    baseline_prediction = np.broadcast_to(train_direction, val_prediction.shape).astype(np.float32)
    results = {
        "train": {
            "grid": grid_metrics(train_prediction, train_target, train_valid),
            "expanded_pixel": expanded_metrics(train_prediction, train_pairs, args.image_size, args.normal_length_tolerance),
        },
        "val": {
            "grid": grid_metrics(val_prediction, val_target, val_valid),
            "expanded_pixel": expanded_metrics(val_prediction, val_pairs, args.image_size, args.normal_length_tolerance),
            "train_only_global_constant_baseline": expanded_metrics(
                baseline_prediction, val_pairs, args.image_size, args.normal_length_tolerance
            ),
        },
    }
    for index, pair in enumerate(val_pairs[:8]):
        write_panel(
            args.output_dir / f"panel_val_{index:02d}.png", pair, val_prediction[index], args.image_size,
            args.normal_length_tolerance,
        )
    candidate = results["val"]["expanded_pixel"]["mean_angular_error_deg"]
    baseline = results["val"]["train_only_global_constant_baseline"]["mean_angular_error_deg"]
    report = {
        "probe": "frozen_local_dinov2_vits14_diode_camera_normal_head",
        "diode_root": str(root),
        "dino_repository": str(args.dino_repository.resolve()),
        "dino_weights": str(args.dino_weights.resolve()),
        "local_files_only": True,
        "device": str(device),
        "image_size": args.image_size,
        "grid_size": args.image_size // DINO_PATCH_SIZE,
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "normal_encoding": "normal = normalize(2 * rgb / 255 - 1)",
        "normal_length_tolerance": args.normal_length_tolerance,
        "min_valid_patch_fraction": args.min_valid_patch_fraction,
        "train_global_constant_direction": [float(value) for value in train_direction],
        "best_step": best_step,
        "history": history,
        "results": results,
        "gate": {
            "candidate_val_expanded_angular_error_deg": candidate,
            "train_only_constant_baseline_angular_error_deg": baseline,
            "improvement_over_baseline_deg": baseline - candidate,
            "minimum_improvement_to_consider_expansion_deg": 5.0,
            "passed": bool(baseline - candidate > 5.0),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"results": results, "gate": report["gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
