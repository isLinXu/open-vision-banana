#!/usr/bin/env python3
"""Compare matched VAE-only and VAE+SSD frozen-feature depth probes."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.data.depth_encoding import (  # noqa: E402
    audit_depth_target_encoding,
    relative_depth_prompt,
)
from open_vision_banana.depth_probe_checkpoint import (  # noqa: E402
    save_depth_probe_artifact,
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


def declared_depth_split_names(
    data_dir: Path,
    *,
    train_samples: int,
    val_samples: int,
) -> tuple[list[str], list[str]]:
    """Select disjoint declared train/validation names in manifest order."""
    if train_samples <= 0 or val_samples <= 0:
        raise ValueError("train_samples and val_samples must be positive")
    manifest = data_dir / "train_val_split.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing v3 split manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    train = [str(name) for name in payload.get("train", {}).get("depth", [])]
    val = [str(name) for name in payload.get("val", {}).get("depth", [])]
    if len(train) < train_samples or len(val) < val_samples:
        raise ValueError(
            f"Insufficient declared depth samples: train={len(train)}, val={len(val)}"
        )
    selected_train = train[:train_samples]
    selected_val = val[:val_samples]
    overlap = sorted(set(selected_train).intersection(selected_val))
    if overlap:
        raise ValueError(f"Declared train/validation overlap: {overlap}")
    return selected_train, selected_val


def declared_depth_stage8_split_names(
    data_dir: Path,
    *,
    train_samples: int,
    dev_samples: int,
    val_samples: int,
    split_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Build deterministic train-fit/dev/official-val splits from the manifest."""
    if min(train_samples, dev_samples, val_samples) <= 0:
        raise ValueError("stage8 split sizes must be positive")
    manifest = data_dir / "train_val_split.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing v3 split manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    declared_train = [str(name) for name in payload.get("train", {}).get("depth", [])]
    declared_val = [str(name) for name in payload.get("val", {}).get("depth", [])]
    if len(declared_train) < train_samples + dev_samples:
        raise ValueError(
            "Insufficient declared depth train samples for fit/dev split: "
            f"available={len(declared_train)} requested={train_samples + dev_samples}"
        )
    if len(declared_val) < val_samples:
        raise ValueError(
            f"Insufficient declared depth validation samples: available={len(declared_val)} "
            f"requested={val_samples}"
        )
    selected = declared_train[: train_samples + dev_samples]
    order = list(range(len(selected)))
    random.Random(split_seed).shuffle(order)
    fit = [selected[index] for index in order[:train_samples]]
    dev = [selected[index] for index in order[train_samples:]]
    official = declared_val[:val_samples]
    all_names = [*fit, *dev, *official]
    if len(set(all_names)) != len(all_names):
        raise ValueError("Stage-8 fit/dev/official-val splits overlap")
    return fit, dev, official


def split_names_checksum(
    train_fit_names: list[str],
    dev_names: list[str],
    official_val_names: list[str],
) -> str:
    """Hash split labels and ordered names for an auditable data contract."""
    digest = hashlib.sha256()
    for label, names in (
        ("train_fit", train_fit_names),
        ("dev", dev_names),
        ("official_val", official_val_names),
    ):
        digest.update(label.encode())
        for name in names:
            digest.update(b"\0")
            digest.update(name.encode())
    return digest.hexdigest()


def select_dev_checkpoint(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best endpoint using dev metrics only; official val is ignored."""
    if not history:
        raise ValueError("cannot select a checkpoint from empty history")
    required = ("step", "dev")
    if any(any(key not in record for key in required) for record in history):
        raise KeyError("checkpoint history must contain step and dev")
    for record in history:
        metrics = record["dev"]
        if any(key not in metrics for key in ("mae", "pearson", "spearman")):
            raise KeyError("dev checkpoint metrics are incomplete")
    return max(
        history,
        key=lambda record: (
            (float(record["dev"]["pearson"]) + float(record["dev"]["spearman"]))
            / 2.0,
            -float(record["dev"]["mae"]),
            int(record["step"]),
        ),
    )


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot aggregate an empty metric list")
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def aggregate_seed_metrics(metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate scalar metrics across seeds without hiding per-seed results."""
    if not metrics:
        raise ValueError("cannot aggregate metrics for zero seeds")
    return {
        key: _mean_std([float(item[key]) for item in metrics])
        for key in ("mae", "rmse", "pearson", "spearman")
    }


def stage8_gate_passed(
    aggregate_delta: dict[str, float],
    per_seed: list[dict[str, Any]],
    *,
    min_correlation_delta: float = 0.05,
    max_mae_regression: float = 0.01,
    required_positive_seeds: int = 2,
) -> bool:
    """Apply the multi-seed contribution gate and dev/official sign check."""
    if min_correlation_delta < 0 or max_mae_regression < 0:
        raise ValueError("stage8 thresholds must be non-negative")
    if required_positive_seeds <= 0:
        raise ValueError("required_positive_seeds must be positive")
    if not per_seed or required_positive_seeds > len(per_seed):
        return False
    required = ("mae", "pearson", "spearman")
    if any(key not in aggregate_delta for key in required):
        raise KeyError("aggregate delta metrics are incomplete")
    positive = 0
    signs_agree = True
    for item in per_seed:
        delta = item.get("delta", {})
        dev_delta = item.get("dev_delta", {})
        if any(key not in delta for key in required) or any(
            key not in dev_delta for key in ("pearson", "spearman")
        ):
            raise KeyError("per-seed stage8 deltas are incomplete")
        if delta["pearson"] > 0 and delta["spearman"] > 0:
            positive += 1
        for key in ("pearson", "spearman"):
            if float(delta[key]) * float(dev_delta[key]) < 0:
                signs_agree = False
    return (
        float(aggregate_delta["pearson"]) >= min_correlation_delta
        and float(aggregate_delta["spearman"]) >= min_correlation_delta
        and float(aggregate_delta["mae"]) <= max_mae_regression
        and positive >= required_positive_seeds
        and signs_agree
    )


def build_batch_schedule(
    *,
    num_samples: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> torch.Tensor:
    """Precompute one replacement-sampling schedule shared by both arms."""
    if num_samples <= 0 or batch_size <= 0 or steps <= 0:
        raise ValueError("schedule dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        0,
        num_samples,
        (steps, batch_size),
        generator=generator,
        dtype=torch.long,
    )


def tensor_checksum(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_dict_checksum(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode())
        digest.update(tensor_checksum(state_dict[name]).encode())
    return digest.hexdigest()


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _load_samples(
    data_dir: Path,
    *,
    names: list[str],
    resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        raise FileNotFoundError(f"Declared samples do not resolve: {missing}")
    samples = [dataset[by_name[name]] for name in names]
    return (
        torch.stack([sample["image"] for sample in samples]),
        torch.stack([sample["target_rgb"] for sample in samples]),
    )


def _cache_features(
    generator: BaseGenerator,
    images: torch.Tensor,
    *,
    prompt: str,
    timestep: int,
    storage_dtype: torch.dtype = torch.bfloat16,
) -> tuple[DepthFeatureContract, tuple[torch.Tensor, ...], list[list[int]]]:
    extractor = FrozenSDXLDepthFeatureExtractor(generator, timestep=timestep)
    contract: DepthFeatureContract | None = None
    levels: list[list[torch.Tensor]] | None = None
    for index in range(images.shape[0]):
        sample_contract, sample_features = extractor.extract(
            images[index : index + 1],
            prompt=prompt,
        )
        if contract is None:
            contract = sample_contract
            levels = [[] for _ in sample_features]
        elif sample_contract != contract:
            raise RuntimeError("Frozen feature contract changed between samples")
        assert levels is not None
        for name, destination, feature in zip(
            sample_contract.names,
            levels,
            sample_features,
            strict=True,
        ):
            stored = feature.to(device="cpu", dtype=storage_dtype)
            if not torch.isfinite(stored).all():
                raise FloatingPointError(
                    f"Feature {name} is non-finite after conversion to {storage_dtype}"
                )
            destination.append(stored)
        if (index + 1) % 8 == 0 or index + 1 == images.shape[0]:
            logger.info("Cached frozen features for %d/%d images", index + 1, images.shape[0])
    if contract is None or levels is None:
        raise RuntimeError("No features were extracted")
    features = tuple(torch.cat(level, dim=0) for level in levels)
    return contract, features, [list(feature.shape) for feature in features]


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _slice_features(
    features: tuple[torch.Tensor, ...],
    indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return tuple(feature[indices] for feature in features)


@torch.no_grad()
def _predict_in_batches(
    head: ScalarDepthProbe,
    features: tuple[torch.Tensor, ...],
    *,
    feature_mask: tuple[bool, ...],
    output_size: tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, features[0].shape[0], batch_size):
        end = min(start + batch_size, features[0].shape[0])
        outputs.append(
            head(
                tuple(feature[start:end] for feature in features),
                output_size=output_size,
                feature_mask=feature_mask,
            ).cpu()
        )
    return torch.cat(outputs, dim=0)


@dataclass
class ArmResult:
    name: str
    feature_mask: tuple[bool, ...]
    initial_checksum: str
    final_state: dict[str, torch.Tensor]
    train_metrics: dict[str, float]
    dev_metrics: dict[str, float]
    official_val_metrics: dict[str, float]
    selected_step: int
    history: list[dict[str, Any]]
    official_val_prediction: torch.Tensor
    runtime_seconds: float
    trainable_parameters: int


def _train_arm(
    *,
    name: str,
    contract: DepthFeatureContract,
    initial_state: dict[str, torch.Tensor],
    train_features: tuple[torch.Tensor, ...],
    dev_features: tuple[torch.Tensor, ...],
    official_val_features: tuple[torch.Tensor, ...],
    train_target: torch.Tensor,
    dev_target: torch.Tensor,
    official_val_target: torch.Tensor,
    batch_schedule: torch.Tensor,
    feature_mask: tuple[bool, ...],
    width: int,
    upsample_stages: int,
    lr: float,
    eval_every: int,
    gradient_weight: float,
    pearson_weight: float,
    seed: int,
    device: torch.device,
) -> ArmResult:
    _set_seed(seed)
    started = time.perf_counter()
    head = ScalarDepthProbe(
        contract,
        width=width,
        upsample_stages=upsample_stages,
    ).to(device=device, dtype=torch.float32)
    head.load_state_dict(initial_state, strict=True)
    initial_checksum = state_dict_checksum(_clone_state_dict(head))
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=batch_schedule.shape[0],
        eta_min=lr * 0.01,
    )
    output_size = tuple(int(size) for size in train_target.shape[-2:])
    history: list[dict[str, Any]] = []
    for step, cpu_indices in enumerate(batch_schedule, start=1):
        indices = cpu_indices.to(device=device)
        prediction = head(
            _slice_features(train_features, cpu_indices),
            output_size=output_size,
            feature_mask=feature_mask,
        )
        loss = scalar_depth_probe_loss(
            prediction,
            train_target[indices],
            gradient_weight=gradient_weight,
            pearson_weight=pearson_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % eval_every == 0 or step == batch_schedule.shape[0]:
            head.eval()
            train_prediction = _predict_in_batches(
                head,
                train_features,
                feature_mask=feature_mask,
                output_size=output_size,
                batch_size=8,
            )
            dev_prediction = _predict_in_batches(
                head,
                dev_features,
                feature_mask=feature_mask,
                output_size=output_size,
                batch_size=8,
            )
            train_metrics = evaluate_scalar_depth_predictions(
                train_prediction,
                train_target,
            )
            dev_metrics = evaluate_scalar_depth_predictions(
                dev_prediction,
                dev_target,
            )
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "dev": dev_metrics,
                "state_dict": _clone_state_dict(head),
            }
            history.append(record)
            logger.info(
                "%s step %d | train p/s=%.4f/%.4f | dev p/s=%.4f/%.4f mae=%.4f",
                name,
                step,
                train_metrics["pearson"],
                train_metrics["spearman"],
                dev_metrics["pearson"],
                dev_metrics["spearman"],
                dev_metrics["mae"],
            )
            head.train()
    selected = select_dev_checkpoint(history)
    head.load_state_dict(selected["state_dict"], strict=True)
    head.eval()
    final_train_prediction = _predict_in_batches(
        head,
        train_features,
        feature_mask=feature_mask,
        output_size=output_size,
        batch_size=8,
    )
    final_dev_prediction = _predict_in_batches(
        head,
        dev_features,
        feature_mask=feature_mask,
        output_size=output_size,
        batch_size=8,
    )
    final_official_val_prediction = _predict_in_batches(
        head,
        official_val_features,
        feature_mask=feature_mask,
        output_size=output_size,
        batch_size=8,
    )
    return ArmResult(
        name=name,
        feature_mask=feature_mask,
        initial_checksum=initial_checksum,
        final_state=_clone_state_dict(head),
        train_metrics=evaluate_scalar_depth_predictions(
            final_train_prediction,
            train_target,
        ),
        dev_metrics=evaluate_scalar_depth_predictions(
            final_dev_prediction,
            dev_target,
        ),
        official_val_metrics=evaluate_scalar_depth_predictions(
            final_official_val_prediction,
            official_val_target,
        ),
        selected_step=int(selected["step"]),
        history=history,
        official_val_prediction=final_official_val_prediction,
        runtime_seconds=time.perf_counter() - started,
        trainable_parameters=trainable_parameters,
    )


def _save_montages(
    output_dir: Path,
    *,
    arm_name: str,
    names: list[str],
    images: torch.Tensor,
    target_rgb: torch.Tensor,
    prediction: torch.Tensor,
    colormap: str,
    limit: int = 4,
) -> list[str]:
    mapper = DepthRGBMapper(colormap)
    predicted_rgb = mapper.normalized_to_rgb(prediction)
    paths = []
    for index, name in enumerate(names[:limit]):
        panels = (
            ((images[index].clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 0),
            target_rgb[index].clamp(0, 1).permute(1, 2, 0),
            predicted_rgb[index].clamp(0, 1).permute(1, 2, 0),
        )
        pil_panels = [
            Image.fromarray((panel.numpy() * 255).round().astype(np.uint8))
            for panel in panels
        ]
        montage = Image.new(
            "RGB",
            (pil_panels[0].width * len(pil_panels), pil_panels[0].height),
        )
        for panel_index, panel in enumerate(pil_panels):
            montage.paste(panel, (panel_index * panel.width, 0))
        path = output_dir / f"{arm_name}_val_{index:02d}_{Path(name).stem}.png"
        montage.save(path)
        paths.append(str(path.resolve()))
    return paths


def _arm_payload(result: ArmResult) -> dict[str, Any]:
    return {
        "feature_mask": list(result.feature_mask),
        "initial_checksum": result.initial_checksum,
        "trainable_parameters": result.trainable_parameters,
        "runtime_seconds": result.runtime_seconds,
        "train_metrics": result.train_metrics,
        "dev_metrics": result.dev_metrics,
        "official_val_metrics": result.official_val_metrics,
        "selected_step": result.selected_step,
        "history": [
            {key: value for key, value in record.items() if key != "state_dict"}
            for record in result.history
        ],
    }


def _parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be a comma-separated integer list") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain at least one unique integer")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="segmind/SSD-1B")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--dev-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--upsample-stages", type=int, default=3)
    parser.add_argument("--gradient-weight", type=float, default=0.5)
    parser.add_argument("--pearson-weight", type=float, default=0.1)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--min-correlation-delta", type=float, default=0.05)
    parser.add_argument("--max-mae-regression", type=float, default=0.01)
    parser.add_argument("--required-positive-seeds", type=int, default=2)
    parser.add_argument(
        "--feature-storage-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="CPU cache dtype; bfloat16 halves memory without FP16 range overflow",
    )
    parser.add_argument(
        "--visualization-count",
        type=int,
        default=4,
        help="Official-val montages generated for every saved weight",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        seeds = _parse_seeds(args.seeds)
    except ValueError as exc:
        parser.error(str(exc))

    positive = {
        "train_samples": args.train_samples,
        "dev_samples": args.dev_samples,
        "val_samples": args.val_samples,
        "resolution": args.resolution,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "lr": args.lr,
        "width": args.width,
        "visualization_count": args.visualization_count,
    }
    if any(value <= 0 for value in positive.values()):
        parser.error(f"positive values required: {positive}")
    if args.required_positive_seeds <= 0 or args.required_positive_seeds > len(seeds):
        parser.error("--required-positive-seeds must be in [1, number of seeds]")
    if args.upsample_stages < 0 or args.timestep < 0:
        parser.error("--upsample-stages and --timestep must be non-negative")
    if any(value < 0 for value in (args.gradient_weight, args.pearson_weight,
                                   args.min_correlation_delta, args.max_mae_regression)):
        parser.error("loss and contribution thresholds must be non-negative")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    started = time.perf_counter()
    device = _resolve_device(args.device)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_names, dev_names, official_names = declared_depth_stage8_split_names(
        data_dir,
        train_samples=args.train_samples,
        dev_samples=args.dev_samples,
        val_samples=args.val_samples,
        split_seed=args.split_seed,
    )
    fit_images, fit_rgb = _load_samples(data_dir, names=fit_names, resolution=args.resolution)
    dev_images, dev_rgb = _load_samples(data_dir, names=dev_names, resolution=args.resolution)
    official_images, official_rgb = _load_samples(data_dir, names=official_names, resolution=args.resolution)
    audit = audit_depth_target_encoding(data_dir / "depth" / "targets", allowed_names=set([*fit_names, *dev_names, *official_names]))
    colormap = audit.detected_colormap
    prompt = relative_depth_prompt(colormap)
    mapper = DepthRGBMapper(colormap)
    fit_target = mapper.rgb_to_normalized(fit_rgb)
    dev_target = mapper.rgb_to_normalized(dev_rgb)
    official_target = mapper.rgb_to_normalized(official_rgb)
    all_images = torch.cat((fit_images, dev_images, official_images), dim=0)
    visualization_count = min(args.visualization_count, len(official_names))
    visualization_images = official_images[:visualization_count].clone()
    visualization_rgb = official_rgb[:visualization_count].clone()
    logger.info("Stage-8 data: fit=%d dev=%d official_val=%d seeds=%s resolution=%d device=%s", len(fit_names), len(dev_names), len(official_names), seeds, args.resolution, device)

    generator = BaseGenerator(model_name=args.model, device=str(device), dtype=torch.bfloat16 if device.type == "mps" else torch.float32)
    storage_dtype = getattr(torch, args.feature_storage_dtype)
    contract, all_features, feature_shapes = _cache_features(
        generator,
        all_images,
        prompt=prompt,
        timestep=args.timestep,
        storage_dtype=storage_dtype,
    )
    del generator
    # Release image/target staging tensors; features remain on CPU and stream to MPS.
    del fit_images, dev_images, official_images, all_images
    del fit_rgb, dev_rgb, official_rgb
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    fit_count, dev_count = len(fit_names), len(dev_names)
    fit_features = tuple(feature[:fit_count] for feature in all_features)
    dev_features = tuple(feature[fit_count:fit_count + dev_count] for feature in all_features)
    official_features = tuple(feature[fit_count + dev_count:] for feature in all_features)
    fit_target = fit_target.to(device=device, dtype=torch.float32)
    dev_target = dev_target.to(device=device, dtype=torch.float32)
    official_target = official_target.to(device=device, dtype=torch.float32)
    vae_mask = (True,) + (False,) * (len(contract.names) - 1)
    full_mask = (True,) * len(contract.names)
    seed_reports: list[dict[str, Any]] = []
    all_initial_checksums: list[str] = []
    all_schedule_checksums: list[str] = []
    for seed in seeds:
        _set_seed(seed)
        template = ScalarDepthProbe(contract, width=args.width, upsample_stages=args.upsample_stages)
        initial_state = _clone_state_dict(template)
        initial_checksum = state_dict_checksum(initial_state)
        del template
        schedule = build_batch_schedule(num_samples=fit_count, batch_size=args.batch_size, steps=args.steps, seed=seed + 1000)
        schedule_checksum = tensor_checksum(schedule)
        all_initial_checksums.append(initial_checksum)
        all_schedule_checksums.append(schedule_checksum)
        vae_only = _train_arm(name="vae_only", contract=contract, initial_state=initial_state, train_features=fit_features, dev_features=dev_features, official_val_features=official_features, train_target=fit_target, dev_target=dev_target, official_val_target=official_target, batch_schedule=schedule, feature_mask=vae_mask, width=args.width, upsample_stages=args.upsample_stages, lr=args.lr, eval_every=args.eval_every, gradient_weight=args.gradient_weight, pearson_weight=args.pearson_weight, seed=seed, device=device)
        vae_ssd = _train_arm(name="vae_ssd", contract=contract, initial_state=initial_state, train_features=fit_features, dev_features=dev_features, official_val_features=official_features, train_target=fit_target, dev_target=dev_target, official_val_target=official_target, batch_schedule=schedule, feature_mask=full_mask, width=args.width, upsample_stages=args.upsample_stages, lr=args.lr, eval_every=args.eval_every, gradient_weight=args.gradient_weight, pearson_weight=args.pearson_weight, seed=seed, device=device)
        if vae_only.initial_checksum != vae_ssd.initial_checksum or vae_only.trainable_parameters != vae_ssd.trainable_parameters:
            raise RuntimeError("matched arm initialization or parameter count mismatch")
        delta = {key: vae_ssd.official_val_metrics[key] - vae_only.official_val_metrics[key] for key in ("mae", "rmse", "pearson", "spearman")}
        dev_delta = {key: vae_ssd.dev_metrics[key] - vae_only.dev_metrics[key] for key in ("mae", "rmse", "pearson", "spearman")}
        artifacts: dict[str, Any] = {}
        visualization_dir = output_dir / "visualizations"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        for arm in (vae_only, vae_ssd):
            artifact_dir = save_depth_probe_artifact(
                output_dir / f"seed{seed}_{arm.name}",
                head_state_dict=arm.final_state,
                feature_contract=contract.to_dict(),
                feature_mask=arm.feature_mask,
                model=args.model,
                prompt=prompt,
                colormap=colormap,
                resolution=args.resolution,
                timestep=args.timestep,
                width=args.width,
                upsample_stages=args.upsample_stages,
                selected_step=arm.selected_step,
                metrics={
                    "train": arm.train_metrics,
                    "dev": arm.dev_metrics,
                    "official_val": arm.official_val_metrics,
                },
                splits={
                    "train_fit_names": fit_names,
                    "dev_names": dev_names,
                    "official_val_names": official_names,
                    "split_seed": args.split_seed,
                    "checksum": split_names_checksum(fit_names, dev_names, official_names),
                },
                training={
                    "seed": seed,
                    "steps": args.steps,
                    "selected_step": arm.selected_step,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "eval_every": args.eval_every,
                    "gradient_weight": args.gradient_weight,
                    "pearson_weight": args.pearson_weight,
                    "feature_storage_dtype": args.feature_storage_dtype,
                    "initial_state_checksum": initial_checksum,
                    "batch_schedule_checksum": schedule_checksum,
                },
            )
            visualization_paths = _save_montages(
                visualization_dir,
                arm_name=f"seed{seed}_{arm.name}",
                names=official_names[:visualization_count],
                images=visualization_images,
                target_rgb=visualization_rgb,
                prediction=arm.official_val_prediction[:visualization_count],
                colormap=colormap,
                limit=visualization_count,
            )
            artifacts[arm.name] = {
                "checkpoint": str(artifact_dir),
                "visualizations": visualization_paths,
            }
        seed_reports.append(
            {
                "seed": seed,
                "matching": {
                    "initial_state_checksum": initial_checksum,
                    "batch_schedule_checksum": schedule_checksum,
                    "parameter_count": vae_only.trainable_parameters,
                },
                "arms": {
                    "vae_only": _arm_payload(vae_only),
                    "vae_ssd": _arm_payload(vae_ssd),
                },
                "artifacts": artifacts,
                "delta": delta,
                "dev_delta": dev_delta,
            }
        )

    def aggregate_arm(name: str, split: str) -> dict[str, dict[str, float]]:
        return aggregate_seed_metrics([report["arms"][name][f"{split}_metrics"] for report in seed_reports])

    aggregate_vae = {split: aggregate_arm("vae_only", split) for split in ("train", "dev", "official_val")}
    aggregate_ssd = {split: aggregate_arm("vae_ssd", split) for split in ("train", "dev", "official_val")}
    aggregate_delta = {key: float(np.mean([report["delta"][key] for report in seed_reports])) for key in ("mae", "rmse", "pearson", "spearman")}
    gate_passed = stage8_gate_passed(aggregate_delta, seed_reports, min_correlation_delta=args.min_correlation_delta, max_mae_regression=args.max_mae_regression, required_positive_seeds=args.required_positive_seeds)
    report = {"stage": 8, "gate_passed": gate_passed, "decision": "promote_ssd_features" if gate_passed else "ssd_feature_contribution_unproven", "model": args.model, "data_dir": str(data_dir), "splits": {"train_fit_names": fit_names, "dev_names": dev_names, "official_val_names": official_names, "split_seed": args.split_seed, "checksum": split_names_checksum(fit_names, dev_names, official_names)}, "seeds": seeds, "resolution": args.resolution, "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr, "eval_every": args.eval_every, "timestep": args.timestep, "colormap": colormap, "prompt": prompt, "feature_contract": contract.to_dict(), "feature_shapes": feature_shapes, "feature_storage_dtype": args.feature_storage_dtype, "matching": {"initial_state_checksums": all_initial_checksums, "batch_schedule_checksums": all_schedule_checksums}, "thresholds": {"min_correlation_delta": args.min_correlation_delta, "max_mae_regression": args.max_mae_regression, "required_positive_seeds": args.required_positive_seeds}, "aggregate": {"vae_only": aggregate_vae, "vae_ssd": aggregate_ssd, "delta_official_val": aggregate_delta}, "per_seed": seed_reports, "runtime_seconds": time.perf_counter() - started}
    write_json_report(output_dir / "result.json", report)
    logger.info("Stage-8 official-val delta mean: %s", json.dumps(aggregate_delta, sort_keys=True))
    logger.info("Gate passed: %s | decision=%s", gate_passed, report["decision"])


if __name__ == "__main__":
    main()
