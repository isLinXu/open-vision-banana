#!/usr/bin/env python3
"""Gate a discriminative semantic head on frozen ImageNet ResNet-18 features.

The v5 semantic labels use a fixed, valid 150-class palette, but prior MAR
RGB-generation experiments did not generalize.  This intentionally modest
baseline keeps ImageNet features frozen and learns a categorical head directly
on target IDs.  It is a decision gate, not a production segmenter: its 16x16
output grid has a measurable nearest-neighbour upper bound reported alongside
the held-out prediction mIoU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.utils import save_image


NUM_CLASSES = 150
IGNORE_LABEL = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device(value: str | None) -> torch.device:
    if value is not None:
        if value == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available")
        if value == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available")
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_rows(manifest: Path, split: str, count: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    rows = [row for row in rows if row.get("task") == "seg" and row.get("split") == split]
    rows.sort(key=lambda row: row["input"])
    if count < 1 or count > len(rows):
        raise ValueError(f"requested {count} {split} segmentation rows, found {len(rows)}")
    indices = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def load_palette(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    palette = np.asarray(data.get("class_palette"), dtype=np.uint8)
    ignore_color = np.asarray(data.get("ignore_color"), dtype=np.uint8)
    if palette.shape != (NUM_CLASSES, 3) or ignore_color.shape != (3,):
        raise ValueError(f"invalid fixed 150-class palette: {path}")
    lut = np.full(1 << 24, IGNORE_LABEL, dtype=np.uint8)
    packed = ((palette[:, 0].astype(np.uint32) << 16)
              | (palette[:, 1].astype(np.uint32) << 8)
              | palette[:, 2].astype(np.uint32))
    lut[packed] = np.arange(NUM_CLASSES, dtype=np.uint8)
    return palette, lut


def load_input(path: str, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ImageNet-normalised input and an unnormalised RGB preview."""
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
    preview = torch.from_numpy(array).permute(2, 0, 1)
    normalised = (preview - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return normalised, preview


def load_labels(path: str, image_size: int, lut: np.ndarray) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(
            image.convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
    packed = ((array[..., 0].astype(np.uint32) << 16)
              | (array[..., 1].astype(np.uint32) << 8)
              | array[..., 2].astype(np.uint32))
    return lut[packed]


def patch_mode_labels(full_labels: np.ndarray, grid_size: int) -> np.ndarray:
    """Reduce a square semantic map to dominant valid class per feature token."""
    height, width = full_labels.shape
    if height != width or height % grid_size:
        raise ValueError(f"label shape {full_labels.shape} is incompatible with grid {grid_size}")
    stride = height // grid_size
    blocks = (full_labels.reshape(grid_size, stride, grid_size, stride)
              .transpose(0, 2, 1, 3).reshape(grid_size * grid_size, -1))
    output = np.full(len(blocks), IGNORE_LABEL, dtype=np.uint8)
    for index, block in enumerate(blocks):
        valid = block[block != IGNORE_LABEL]
        if len(valid):
            output[index] = np.bincount(valid, minlength=NUM_CLASSES).argmax()
    return output


def expand_grid_labels(labels: np.ndarray, grid_size: int, image_size: int) -> np.ndarray:
    stride = image_size // grid_size
    grid = labels.reshape(-1, grid_size, grid_size)
    return np.repeat(np.repeat(grid, stride, axis=1), stride, axis=2)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} != {target.shape}")
    valid = target.reshape(-1) != IGNORE_LABEL
    pred = prediction.reshape(-1)[valid].astype(np.int64)
    ref = target.reshape(-1)[valid].astype(np.int64)
    confusion = np.bincount(
        NUM_CLASSES * ref + pred,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)
    true_positive = np.diag(confusion).astype(np.float64)
    target_total = confusion.sum(axis=1).astype(np.float64)
    prediction_total = confusion.sum(axis=0).astype(np.float64)
    union = target_total + prediction_total - true_positive
    present = union > 0
    return {
        "pixel_accuracy": float(true_positive.sum() / max(1, int(target_total.sum()))),
        "miou": float((true_positive[present] / union[present]).mean()) if present.any() else 0.0,
        "present_classes": int(present.sum()),
        "valid_pixels": int(target_total.sum()),
    }


class ResNet18Layer3(nn.Module):
    """ImageNet ResNet-18 through its 16x16 layer3 feature grid."""

    def __init__(self) -> None:
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.layer3(self.layer2(self.layer1(self.stem(images))))
        return features.flatten(2).transpose(1, 2)


class TokenClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def cache_features(rows: list[dict], split: str, cache_dir: Path, backbone: nn.Module,
                   device: torch.device, image_size: int, lut: np.ndarray,
                   eval_batch_size: int, expected_feature_dim: int = 256,
                   expected_grid_size: int | None = None) -> tuple[torch.Tensor, np.ndarray, np.ndarray, int]:
    """Create/reuse frozen features and load labels without retaining preview RGBs."""
    features, grids, full_labels = [], [], []
    grid_size = 0
    split_cache = cache_dir / split
    split_cache.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), eval_batch_size):
        batch_rows = rows[start:start + eval_batch_size]
        batch_inputs, batch_labels, batch_paths = [], [], []
        cached = []
        for offset, row in enumerate(batch_rows):
            index = start + offset
            feature_path = split_cache / f"{index:05d}.pt"
            feature = None
            if feature_path.exists():
                try:
                    feature = torch.load(feature_path, map_location="cpu", weights_only=True)
                    if (not isinstance(feature, torch.Tensor) or feature.ndim != 2
                            or feature.shape[-1] != expected_feature_dim):
                        feature = None
                except (OSError, RuntimeError):
                    feature = None
            normalised, _ = load_input(row["input"], image_size)
            batch_inputs.append(normalised)
            batch_labels.append(load_labels(row["target"], image_size, lut))
            batch_paths.append(feature_path)
            cached.append(feature)
        missing = [index for index, feature in enumerate(cached) if feature is None]
        if missing:
            images = torch.stack(batch_inputs).to(device)
            with torch.no_grad():
                produced = backbone(images).cpu()
            for index in missing:
                temporary = batch_paths[index].with_name(f".{batch_paths[index].name}.{os.getpid()}.tmp")
                # ``produced[index]`` is a view into the complete batch.
                # Cloning prevents torch serialization from retaining every
                # other sample's backing storage in each cache file.
                torch.save(produced[index].clone(), temporary)
                temporary.replace(batch_paths[index])
                cached[index] = produced[index].clone()
            del images, produced
            if device.type == "mps":
                torch.mps.empty_cache()
        batch_features = torch.stack([feature.float() for feature in cached])
        if grid_size == 0:
            grid_size = int(round(batch_features.shape[1] ** 0.5))
            if grid_size * grid_size != batch_features.shape[1]:
                raise ValueError(f"feature token count {batch_features.shape[1]} is not square")
            if expected_grid_size is not None and grid_size != expected_grid_size:
                raise ValueError(
                    f"feature grid size {grid_size} does not match expected {expected_grid_size}"
                )
        features.append(batch_features)
        grids.extend(patch_mode_labels(label, grid_size) for label in batch_labels)
        full_labels.extend(batch_labels)
        print(f"[features] {split} {min(start + len(batch_rows), len(rows))}/{len(rows)}", flush=True)
    return (
        torch.cat(features),
        np.stack(grids),
        np.stack(full_labels),
        grid_size,
    )


def load_previews(rows: list[dict], image_size: int) -> torch.Tensor:
    """Load only the small subset needed for output panels."""
    return torch.stack([load_input(row["input"], image_size)[1] for row in rows])


@torch.no_grad()
def predict(head: nn.Module, features: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(features), batch_size):
        outputs.append(head(features[start:start + batch_size].to(device)).argmax(dim=-1).cpu())
    return torch.cat(outputs).numpy().astype(np.uint8)


def labels_to_rgb(labels: np.ndarray, palette: np.ndarray) -> torch.Tensor:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels != IGNORE_LABEL
    rgb[valid] = palette[labels[valid]]
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0


def write_panel(path: Path, previews: torch.Tensor, target: np.ndarray,
                prediction: np.ndarray, palette: np.ndarray) -> None:
    images = torch.stack((previews, labels_to_rgb(target, palette), labels_to_rgb(prediction, palette)), dim=1)
    save_image(images.reshape(-1, 3, target.shape[-2], target.shape[-1]), path, nrow=3)


def main() -> None:
    args = parse_args()
    if args.image_size < 32 or args.image_size % 16 or args.steps < 1 or args.batch_size < 1:
        raise SystemExit("--image-size must be a multiple of 16; --steps and --batch-size must be positive")
    set_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (args.feature_cache or args.output_dir / "features").resolve()
    palette, lut = load_palette(args.palette)
    train_rows = select_rows(args.manifest, "train", args.train_samples)
    val_rows = select_rows(args.manifest, "val", args.val_samples)

    backbone = ResNet18Layer3().to(device).eval()
    train_features, train_grid, train_full, grid_size = cache_features(
        train_rows, "train", cache_dir, backbone, device, args.image_size, lut, args.eval_batch_size
    )
    val_features, val_grid, val_full, val_grid_size = cache_features(
        val_rows, "val", cache_dir, backbone, device, args.image_size, lut, args.eval_batch_size
    )
    if val_grid_size != grid_size:
        raise ValueError("train and validation feature grids differ")
    del backbone
    if device.type == "mps":
        torch.mps.empty_cache()

    train_targets = torch.from_numpy(train_grid.astype(np.int64))
    valid = train_targets != IGNORE_LABEL
    frequencies = torch.bincount(train_targets[valid].reshape(-1), minlength=NUM_CLASSES).float()
    # Build boolean-indexed weights alongside CPU label frequencies, then move
    # the completed tensor to the training device. MPS does not allow a CPU
    # boolean mask to index an MPS tensor.
    class_weights = torch.zeros(NUM_CLASSES)
    active = frequencies > 0
    class_weights[active] = frequencies[active].rsqrt()
    class_weights[active] /= class_weights[active].mean()
    class_weights = class_weights.to(device)

    head = TokenClassifier(train_features.shape[-1], args.hidden_dim).to(device)
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
    torch.save({"head": best_state, "step": best_step, "best_grid_miou": best_miou}, args.output_dir / "classifier_head.pt")
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
        args.output_dir / "panel_train.png",
        load_previews(train_rows[:8], args.image_size),
        train_full[:8],
        train_expanded[:8],
        palette,
    )
    write_panel(
        args.output_dir / "panel_val.png",
        load_previews(val_rows[:8], args.image_size),
        val_full[:8],
        val_expanded[:8],
        palette,
    )
    report = {
        "probe": "frozen_imagenet_resnet18_layer3_discrete_150_class_head",
        "manifest": str(args.manifest.resolve()),
        "palette": str(args.palette.resolve()),
        "feature_cache": str(cache_dir),
        "device": str(device),
        "image_size": args.image_size,
        "grid_size": grid_size,
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
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
