#!/usr/bin/env python3
"""Probe fixed MAR features with a direct 150-class segmentation objective.

The RGB-diffusion segmentation experiment optimizes continuous VAE latents,
then hopes a decoded RGB image will land on a 150-colour semantic palette.
This script tests the smallest alternative: keep the trained MAR conditional
features frozen and train a categorical head directly on semantic labels.

It is intentionally an overfit gate, not a replacement production model:
the head predicts one class per MAR 24x24 token.  Reports include both token
metrics and the pixel metrics after nearest-neighbour expansion, plus an
oracle expansion of the target token labels to quantify the grid's ceiling.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "output/mar_vb_clean_pairs_v4/pairs.jsonl"
NUM_CLASSES = 150
IGNORE_LABEL = 255
SEG_LABEL = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-cache", type=Path,
        help="Per-sample frozen-feature cache. Defaults to OUTPUT/features.",
    )
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--val-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Classifier samples per optimizer step; 0 uses the entire cached train split.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--best-by", choices=("train", "val"), default="train",
        help="Metric split used to retain the classifier-head checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--seg-palette", type=Path)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_module(img_size: int):
    os.environ["VB_IMG_SIZE"] = str(img_size)
    os.environ["VB_NUM_WORKERS"] = "0"
    path = PROJECT_ROOT / "scripts/mar_vb_train_v5_multitask.py"
    spec = importlib.util.spec_from_file_location("mar_vb_seg_probe_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_rows(manifest: Path, split: str, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.open() if line.strip()]
    selected = [row for row in rows if row.get("task") == "seg" and row.get("split") == split]
    selected.sort(key=lambda row: row["input"])
    if len(selected) < limit:
        raise ValueError(f"manifest has only {len(selected)} segmentation {split} rows; requested {limit}")
    if limit == len(selected):
        return selected
    # Evenly spread deterministic rows over the full split.  Stems are sorted,
    # so taking the first N samples would turn a medium-scale gate into a
    # potentially accidental source-order slice.
    indices = np.linspace(0, len(selected) - 1, num=limit, dtype=np.int64)
    return [selected[int(index)] for index in indices]


def load_palette(rows: list[dict], explicit: Path | None) -> tuple[np.ndarray, Path]:
    path = explicit
    if path is None:
        for row in rows:
            value = row.get("segmentation_palette")
            if value:
                path = Path(value)
                break
    if path is None or not path.exists():
        raise FileNotFoundError("a valid fixed segmentation palette is required")
    data = json.loads(path.read_text())
    palette = np.asarray(data.get("class_palette"), dtype=np.uint8)
    if palette.shape != (NUM_CLASSES, 3):
        raise ValueError(f"expected a {NUM_CLASSES}x3 class palette in {path}")
    return palette, path.resolve()


def packed_color_lut(palette: np.ndarray) -> np.ndarray:
    """Create an exact RGB-to-class table; unlisted RGB values are ignore."""
    lut = np.full(1 << 24, IGNORE_LABEL, dtype=np.uint8)
    packed = ((palette[:, 0].astype(np.uint32) << 16)
              | (palette[:, 1].astype(np.uint32) << 8)
              | palette[:, 2].astype(np.uint32))
    lut[packed] = np.arange(NUM_CLASSES, dtype=np.uint8)
    return lut


def load_input(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.BICUBIC)
        array = np.asarray(rgb, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_full_labels(path: str, size: int, lut: np.ndarray) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((size, size), Image.NEAREST)
        array = np.asarray(rgb, dtype=np.uint8)
    packed = ((array[..., 0].astype(np.uint32) << 16)
              | (array[..., 1].astype(np.uint32) << 8)
              | array[..., 2].astype(np.uint32))
    return lut[packed]


def patch_mode_labels(full_labels: np.ndarray, grid_size: int) -> np.ndarray:
    """Reduce an HxW label map to one valid dominant class per token."""
    height, width = full_labels.shape
    if height != width or height % grid_size:
        raise ValueError(f"label shape {full_labels.shape} is incompatible with {grid_size}x{grid_size}")
    stride = height // grid_size
    blocks = (full_labels.reshape(grid_size, stride, grid_size, stride)
              .transpose(0, 2, 1, 3).reshape(grid_size * grid_size, -1))
    reduced = np.full(len(blocks), IGNORE_LABEL, dtype=np.uint8)
    for index, block in enumerate(blocks):
        valid = block[block != IGNORE_LABEL]
        if len(valid):
            reduced[index] = np.bincount(valid, minlength=NUM_CLASSES).argmax()
    return reduced


@torch.no_grad()
def conditional_features(model, context_latent: torch.Tensor) -> torch.Tensor:
    """Decode every target token using only context-image and task conditioning."""
    base = model.m
    batch = context_latent.size(0)
    context_tokens = base.patchify(context_latent)
    target_tokens = torch.zeros_like(context_tokens)
    target_mask = torch.ones(batch, base.seq_len, device=context_latent.device)
    labels = torch.full((batch,), SEG_LABEL, dtype=torch.long, device=context_latent.device)
    class_embedding = base.class_emb(labels)
    encoded = model.encode_vb(context_tokens, target_tokens, target_mask, class_embedding, labels)
    return model.decode_vb(encoded, target_mask)


class TokenClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def confusion_matrix(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    valid = target != IGNORE_LABEL
    return np.bincount(
        NUM_CLASSES * target[valid].astype(np.int64) + prediction[valid].astype(np.int64),
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    confusion = confusion_matrix(prediction.reshape(-1), target.reshape(-1))
    true_positive = np.diag(confusion).astype(np.float64)
    target_total = confusion.sum(axis=1).astype(np.float64)
    predicted_total = confusion.sum(axis=0).astype(np.float64)
    union = target_total + predicted_total - true_positive
    present = union > 0
    valid_pixels = int(target_total.sum())
    return {
        "pixel_accuracy": float(true_positive.sum() / max(1, valid_pixels)),
        "miou": float((true_positive[present] / union[present]).mean()) if present.any() else 0.0,
        "present_classes": int(present.sum()),
        "valid_pixels": valid_pixels,
    }


def expand_grid_labels(labels: np.ndarray, grid_size: int, image_size: int) -> np.ndarray:
    batch = labels.shape[0]
    stride = image_size // grid_size
    grid = labels.reshape(batch, grid_size, grid_size)
    return np.repeat(np.repeat(grid, stride, axis=1), stride, axis=2)


def labels_to_rgb(labels: np.ndarray, palette: np.ndarray) -> torch.Tensor:
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    valid = labels != IGNORE_LABEL
    rgb[valid] = palette[labels[valid]]
    return torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0


def save_panel(path: Path, inputs: torch.Tensor, target: np.ndarray, prediction: np.ndarray,
               palette: np.ndarray) -> None:
    target_rgb = labels_to_rgb(target, palette)
    prediction_rgb = labels_to_rgb(prediction, palette)
    source_rgb = (inputs.cpu().clamp(-1, 1) + 1) / 2
    images = torch.stack((source_rgb, target_rgb, prediction_rgb), dim=1).reshape(-1, 3, *target.shape[-2:])
    save_image(images, path, nrow=3)


def _load_cached_feature(path: Path, expected_shape: tuple[int, int]) -> torch.Tensor | None:
    if not path.exists():
        return None
    try:
        feature = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(feature, torch.Tensor) or tuple(feature.shape) != expected_shape:
        return None
    return feature.float()


def _write_cached_feature(path: Path, feature: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(feature, temporary)
    temporary.replace(path)


def build_feature_split(rows: list[dict], *, model, vae, mt, device: str,
                        image_size: int, grid_size: int, lut: np.ndarray,
                        cache_dir: Path, split: str) -> dict[str, torch.Tensor | np.ndarray]:
    features, inputs, grids, full_labels = [], [], [], []
    expected_shape = (model.m.seq_len, model.m.decoder_norm.normalized_shape[0])
    cache_hits = 0
    for index, row in enumerate(rows, 1):
        image = load_input(row["input"], image_size)
        labels = load_full_labels(row["target"], image_size, lut)
        cache_path = cache_dir / split / f"{index - 1:05d}.pt"
        token_features = _load_cached_feature(cache_path, expected_shape)
        if token_features is None:
            context = mt.encode_latent(vae, image.unsqueeze(0).to(device), sample=False)
            token_features = conditional_features(model, context)[0].cpu()
            _write_cached_feature(cache_path, token_features)
            del context
        else:
            cache_hits += 1
        features.append(token_features.unsqueeze(0))
        inputs.append(image)
        grids.append(patch_mode_labels(labels, grid_size))
        full_labels.append(labels)
        # MPS retains compiled intermediates aggressively.  Feature extraction
        # is sample-wise here, so release them before moving to the next image.
        del token_features
        if device == "mps":
            torch.mps.empty_cache()
        print(f"[features] {split} {index}/{len(rows)} cache_hits={cache_hits}", flush=True)
    return {
        "features": torch.cat(features),
        "inputs": torch.stack(inputs),
        "grid_labels": np.stack(grids),
        "full_labels": np.stack(full_labels),
    }


@torch.no_grad()
def predict_in_batches(head: nn.Module, features: torch.Tensor, *, device: str,
                       batch_size: int) -> np.ndarray:
    predictions = []
    for start in range(0, len(features), batch_size):
        logits = head(features[start:start + batch_size].to(device))
        predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions).numpy().astype(np.uint8)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.train_samples < 1 or args.val_samples < 1:
        raise SystemExit("--steps, --train-samples, and --val-samples must be positive")
    if args.lr <= 0 or args.hidden_dim < 1 or args.batch_size < 0 or args.eval_batch_size < 1:
        raise SystemExit("--lr, --hidden-dim, and --eval-batch-size must be positive; --batch-size cannot be negative")
    if not args.checkpoint.exists() or not args.manifest.exists():
        raise SystemExit("checkpoint or manifest not found")

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    feature_cache = (args.feature_cache or args.output / "features").resolve()
    train_rows = parse_rows(args.manifest, "train", args.train_samples)
    val_rows = parse_rows(args.manifest, "val", args.val_samples)
    palette, palette_path = load_palette(train_rows + val_rows, args.seg_palette)
    lut = packed_color_lut(palette)

    mt = load_training_module(args.img_size)
    device = mt.DEVICE
    base = mt.mar_large(buffer_size=64, diffloss_d=8, diffloss_w=1280,
                        num_sampling_steps="100", img_size=args.img_size,
                        attn_dropout=mt.DROPOUT, proj_dropout=mt.DROPOUT)
    state = load_file(f"{mt.CKPT_DIR}/mar-large.safetensors")
    if args.img_size != 256:
        state = mt.resize_pos_keys(state, "", args.img_size // 16)
    base.load_state_dict(state, strict=True)
    model = mt.MARVBMultiTask(base, num_tasks=len(mt.TASK_LABEL)).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    if not args.no_ema and "ema" in checkpoint:
        ema = mt.EMA(model, decay=checkpoint["ema"].get("decay", 0.999))
        ema.load_state_dict(checkpoint["ema"])
        ema.apply()
    vae = mt.AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                           ckpt_path=f"{mt.CKPT_DIR}/kl16.safetensors").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in vae.parameters():
        parameter.requires_grad_(False)

    grid_size = model.m.seq_h
    print(f"[probe] device={device} grid={grid_size}x{grid_size} checkpoint_step={checkpoint.get('step')}", flush=True)
    with torch.no_grad():
        train = build_feature_split(train_rows, model=model, vae=vae, mt=mt, device=device,
                                    image_size=args.img_size, grid_size=grid_size, lut=lut,
                                    cache_dir=feature_cache, split="train")
        val = build_feature_split(val_rows, model=model, vae=vae, mt=mt, device=device,
                                  image_size=args.img_size, grid_size=grid_size, lut=lut,
                                  cache_dir=feature_cache, split="val")

    # Release the 479M MAR model before head training.  This probe intentionally
    # tests categorical supervision on frozen conditional features.
    del model, vae, base
    if device == "mps":
        torch.mps.empty_cache()

    train_features = train["features"].float()
    train_targets = torch.from_numpy(train["grid_labels"].astype(np.int64))
    val_features = val["features"].float()
    valid_targets = train_targets != IGNORE_LABEL
    frequencies = torch.bincount(train_targets[valid_targets].reshape(-1), minlength=NUM_CLASSES).float()
    class_weights = torch.zeros(NUM_CLASSES, device=device)
    active = frequencies > 0
    class_weights[active] = frequencies[active].rsqrt()
    class_weights[active] /= class_weights[active].mean()

    head = TokenClassifier(train_features.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    batch_size = args.batch_size or args.train_samples
    batch_size = min(batch_size, args.train_samples)
    best_state, best_miou, best_step, history = None, -1.0, 0, []
    report_every = max(1, min(100, args.steps // 10))
    started = time.time()
    for step in range(1, args.steps + 1):
        if batch_size == args.train_samples:
            batch_index = torch.arange(args.train_samples)
        else:
            batch_index = torch.randint(args.train_samples, (batch_size,))
        batch_features = train_features[batch_index].to(device)
        batch_targets = train_targets[batch_index].to(device)
        batch_valid = batch_targets != IGNORE_LABEL
        logits = head(batch_features)
        loss = F.cross_entropy(logits[batch_valid], batch_targets[batch_valid], weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % report_every == 0 or step == args.steps:
            train_prediction = predict_in_batches(head, train_features, device=device,
                                                  batch_size=args.eval_batch_size)
            val_prediction = predict_in_batches(head, val_features, device=device,
                                                batch_size=args.eval_batch_size)
            train_grid = metrics(train_prediction, train["grid_labels"])
            val_grid = metrics(val_prediction, val["grid_labels"])
            item = {
                "step": step,
                "loss": float(loss.item()),
                "train_grid": train_grid,
                "val_grid": val_grid,
            }
            history.append(item)
            selection = train_grid if args.best_by == "train" else val_grid
            if selection["miou"] > best_miou:
                best_miou = selection["miou"]
                best_step = step
                best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
            print(f"[train] step={step}/{args.steps} loss={item['loss']:.5f} "
                  f"train_miou={train_grid['miou']:.4f} val_miou={val_grid['miou']:.4f}", flush=True)

    assert best_state is not None
    head.load_state_dict(best_state)
    torch.save({"head": best_state, "step": best_step, "best_by": args.best_by,
                "best_grid_miou": best_miou}, args.output / "classifier_head.pt")
    train_prediction = predict_in_batches(head, train_features, device=device,
                                          batch_size=args.eval_batch_size)
    val_prediction = predict_in_batches(head, val_features, device=device,
                                        batch_size=args.eval_batch_size)

    results = {}
    for name, split, prediction in (("train", train, train_prediction), ("val", val, val_prediction)):
        expanded_prediction = expand_grid_labels(prediction, grid_size, args.img_size)
        expanded_oracle = expand_grid_labels(split["grid_labels"], grid_size, args.img_size)
        results[name] = {
            "grid": metrics(prediction, split["grid_labels"]),
            "expanded_pixel": metrics(expanded_prediction, split["full_labels"]),
            "expanded_oracle_pixel": metrics(expanded_oracle, split["full_labels"]),
        }
        save_panel(args.output / f"panel_{name}.png", split["inputs"], split["full_labels"],
                   expanded_prediction, palette)
        print(f"[{name}] {json.dumps(results[name], sort_keys=True)}", flush=True)

    report = {
        "probe": "frozen_mar_conditional_features_direct_150_class_head",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "manifest": str(args.manifest.resolve()),
        "palette": str(palette_path),
        "feature_cache": str(feature_cache),
        "img_size": args.img_size,
        "grid_size": grid_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "steps": args.steps,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "batch_size": batch_size,
        "eval_batch_size": args.eval_batch_size,
        "best_by": args.best_by,
        "best_step": best_step,
        "best_grid_miou": best_miou,
        "ema": not args.no_ema,
        "elapsed_seconds": time.time() - started,
        "history": history,
        "results": results,
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
