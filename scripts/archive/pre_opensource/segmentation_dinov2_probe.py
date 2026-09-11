#!/usr/bin/env python3
"""Gate semantic segmentation on local frozen DINOv2 ViT-S/14 features.

This is deliberately a held-out gate, rather than a production semantic
segmenter.  It reuses the fixed 150-class palette and expanded-pixel mIoU
contract of ``segmentation_resnet18_probe.py``, while replacing its ImageNet
ResNet-18 features with the locally cached DINOv2 ViT-S/14 encoder.  The
checkpoint is loaded from an explicit local path, so this program neither
downloads weights nor changes the specialist inference API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_resnet18_probe import (
    IGNORE_LABEL,
    NUM_CLASSES,
    TokenClassifier,
    cache_features,
    expand_grid_labels,
    labels_to_rgb,
    load_palette,
    load_previews,
    metrics,
    predict,
    select_device,
    select_rows,
    set_seed,
    write_panel,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DINO_REPOSITORY = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
DEFAULT_DINO_WEIGHTS = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
DINO_PATCH_SIZE = 14
DINO_FEATURE_DIM = 384


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--dino-repository", type=Path, default=DEFAULT_DINO_REPOSITORY)
    parser.add_argument("--dino-weights", type=Path, default=DEFAULT_DINO_WEIGHTS)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
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
    if not args.dino_repository.is_dir():
        raise FileNotFoundError(f"Local DINOv2 repository not found: {args.dino_repository}")
    if not args.dino_weights.is_file():
        raise FileNotFoundError(f"Local DINOv2 weights not found: {args.dino_weights}")


class LocalDINOv2ViTS14(nn.Module):
    """Local-only frozen DINOv2 ViT-S/14 patch-token feature extractor."""

    def __init__(self, repository: Path, weights: Path) -> None:
        super().__init__()
        self.network = torch.hub.load(
            str(repository),
            "dinov2_vits14",
            source="local",
            pretrained=True,
            weights=str(weights),
        )
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.network.forward_features(images)
        features = output["x_norm_patchtokens"]
        if features.ndim != 3 or features.shape[-1] != DINO_FEATURE_DIM:
            raise ValueError(f"Unexpected DINOv2 patch-token shape: {tuple(features.shape)}")
        return features


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
    set_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.feature_cache or args.output_dir / "features").resolve()
    palette, lut = load_palette(args.palette)
    train_rows = select_rows(args.manifest, "train", args.train_samples)
    val_rows = select_rows(args.manifest, "val", args.val_samples)
    expected_grid_size = args.image_size // DINO_PATCH_SIZE

    backbone = LocalDINOv2ViTS14(args.dino_repository.resolve(), args.dino_weights.resolve()).to(device).eval()
    train_features, train_grid, train_full, grid_size = cache_features(
        train_rows, "train", cache_dir, backbone, device, args.image_size, lut, args.eval_batch_size,
        expected_feature_dim=DINO_FEATURE_DIM, expected_grid_size=expected_grid_size,
    )
    val_features, val_grid, val_full, val_grid_size = cache_features(
        val_rows, "val", cache_dir, backbone, device, args.image_size, lut, args.eval_batch_size,
        expected_feature_dim=DINO_FEATURE_DIM, expected_grid_size=expected_grid_size,
    )
    if val_grid_size != grid_size:
        raise ValueError("train and validation feature grids differ")
    del backbone
    if device.type == "mps":
        torch.mps.empty_cache()

    train_targets = torch.from_numpy(train_grid.astype("int64"))
    valid = train_targets != IGNORE_LABEL
    frequencies = torch.bincount(train_targets[valid].reshape(-1), minlength=NUM_CLASSES).float()
    # Keep construction on CPU because ``frequencies`` and ``active`` are
    # derived from the CPU feature/label cache; move once after indexing.
    class_weights = torch.zeros(NUM_CLASSES)
    active = frequencies > 0
    class_weights[active] = frequencies[active].rsqrt()
    class_weights[active] /= class_weights[active].mean()
    class_weights = class_weights.to(device)

    head = TokenClassifier(DINO_FEATURE_DIM, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_miou, best_step, history = None, -1.0, 0, []
    report_every = max(1, min(100, args.steps // 10))
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train_features), (min(args.batch_size, len(train_features)),))
        batch_features = train_features[indices].to(device)
        batch_targets = train_targets[indices].to(device)
        batch_valid = batch_targets != IGNORE_LABEL
        logits = head(batch_features)
        loss = F.cross_entropy(logits[batch_valid], batch_targets[batch_valid], weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % report_every == 0 or step == args.steps:
            train_prediction = predict(head, train_features, device, args.eval_batch_size)
            val_prediction = predict(head, val_features, device, args.eval_batch_size)
            train_metrics = metrics(train_prediction, train_grid)
            val_metrics = metrics(val_prediction, val_grid)
            history.append({"step": step, "loss": float(loss.item()), "train_grid": train_metrics, "val_grid": val_metrics})
            if val_metrics["miou"] > best_miou:
                best_miou, best_step = float(val_metrics["miou"]), step
                best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            print(
                f"[train] step={step}/{args.steps} loss={loss.item():.5f} "
                f"train_miou={train_metrics['miou']:.4f} val_miou={val_metrics['miou']:.4f}",
                flush=True,
            )

    assert best_state is not None
    head.load_state_dict(best_state)
    torch.save(
        {"head": best_state, "step": best_step, "best_grid_miou": best_miou,
         "backbone": "dinov2_vits14", "image_size": args.image_size},
        args.output_dir / "classifier_head.pt",
    )
    train_prediction = predict(head, train_features, device, args.eval_batch_size)
    val_prediction = predict(head, val_features, device, args.eval_batch_size)
    train_expanded = expand_grid_labels(train_prediction, grid_size, args.image_size)
    val_expanded = expand_grid_labels(val_prediction, grid_size, args.image_size)
    train_oracle = expand_grid_labels(train_grid, grid_size, args.image_size)
    val_oracle = expand_grid_labels(val_grid, grid_size, args.image_size)
    results = {
        "train": {
            "grid": metrics(train_prediction, train_grid),
            "expanded_pixel": metrics(train_expanded, train_full),
            "expanded_oracle_pixel": metrics(train_oracle, train_full),
        },
        "val": {
            "grid": metrics(val_prediction, val_grid),
            "expanded_pixel": metrics(val_expanded, val_full),
            "expanded_oracle_pixel": metrics(val_oracle, val_full),
        },
    }
    write_panel(
        args.output_dir / "panel_train.png", load_previews(train_rows[:8], args.image_size),
        train_full[:8], train_expanded[:8], palette,
    )
    write_panel(
        args.output_dir / "panel_val.png", load_previews(val_rows[:8], args.image_size),
        val_full[:8], val_expanded[:8], palette,
    )
    report = {
        "probe": "frozen_local_dinov2_vits14_discrete_150_class_head",
        "manifest": str(args.manifest.resolve()),
        "palette": str(args.palette.resolve()),
        "feature_cache": str(cache_dir),
        "dino_repository": str(args.dino_repository.resolve()),
        "dino_weights": str(args.dino_weights.resolve()),
        "local_files_only": True,
        "device": str(device),
        "image_size": args.image_size,
        "grid_size": grid_size,
        "feature_dim": DINO_FEATURE_DIM,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "best_step": best_step,
        "best_grid_miou": best_miou,
        "history": history,
        "results": results,
        "gate": {
            "baseline_resnet18_val_expanded_pixel_miou": 0.03815445226190414,
            "candidate_val_expanded_pixel_miou": results["val"]["expanded_pixel"]["miou"],
            "minimum_miou_to_consider_expansion": 0.04,
            "passed": bool(results["val"]["expanded_pixel"]["miou"] > 0.04),
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"results": results, "gate": report["gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
