#!/usr/bin/env python3
"""Overfit a scalar-depth head on cached frozen SSD/VAE features."""

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
    depth_probe_gate_passed,
    evaluate_scalar_depth_predictions,
    scalar_depth_probe_loss,
)
from open_vision_banana.training.tuner import VisionTaskDataset  # noqa: E402

logger = logging.getLogger(__name__)


def declared_depth_train_names(data_dir: Path, *, num_samples: int) -> list[str]:
    """Read the first declared train names without falling back to sorted files."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    manifest = data_dir / "train_val_split.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing v3 split manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    names = payload.get("train", {}).get("depth", [])
    if len(names) < num_samples:
        raise ValueError(
            f"Manifest declares only {len(names)} depth train samples; "
            f"requested {num_samples}"
        )
    return [str(name) for name in names[:num_samples]]


def write_json_report(path: Path, payload: dict[str, Any]) -> None:
    """Write standards-compliant JSON, rejecting NaN and infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(serialized + "\n", encoding="utf-8")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_declared_samples(
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
    samples_by_name = {
        Path(sample["image_path"]).name: index
        for index, sample in enumerate(dataset.samples)
    }
    missing = [name for name in names if name not in samples_by_name]
    if missing:
        raise FileNotFoundError(f"Declared depth samples do not resolve: {missing}")
    samples = [dataset[samples_by_name[name]] for name in names]
    images = torch.stack([sample["image"] for sample in samples])
    targets = torch.stack([sample["target_rgb"] for sample in samples])
    return images, targets


def _cache_frozen_features(
    generator: BaseGenerator,
    images: torch.Tensor,
    *,
    prompt: str,
    timestep: int,
) -> tuple[DepthFeatureContract, tuple[torch.Tensor, ...], list[list[int]]]:
    extractor = FrozenSDXLDepthFeatureExtractor(generator, timestep=timestep)
    per_level: list[list[torch.Tensor]] | None = None
    expected_contract: DepthFeatureContract | None = None
    for index in range(images.shape[0]):
        contract, features = extractor.extract(
            images[index : index + 1],
            prompt=prompt,
        )
        if expected_contract is None:
            expected_contract = contract
            per_level = [[] for _ in features]
        elif contract != expected_contract:
            raise RuntimeError("Frozen feature contract changed between samples")
        assert per_level is not None
        for destination, feature in zip(per_level, features, strict=True):
            destination.append(feature.cpu())
        logger.info("Cached frozen features for sample %d/%d", index + 1, images.shape[0])
    if expected_contract is None or per_level is None:
        raise RuntimeError("No frozen features were extracted")
    batched = tuple(torch.cat(level, dim=0) for level in per_level)
    shapes = [list(feature.shape) for feature in batched]
    return expected_contract, batched, shapes


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _per_sample_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    names: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "sample": name,
            **evaluate_scalar_depth_predictions(
                prediction[index : index + 1],
                target[index : index + 1],
            ),
        }
        for index, name in enumerate(names)
    ]


def _all_metrics_pass_gate(
    aggregate: dict[str, float],
    per_sample: list[dict[str, Any]],
    *,
    max_mae: float,
    min_pearson: float,
    min_spearman: float,
) -> bool:
    keyword_args = {
        "max_mae": max_mae,
        "min_pearson": min_pearson,
        "min_spearman": min_spearman,
    }
    return depth_probe_gate_passed(aggregate, **keyword_args) and all(
        depth_probe_gate_passed(sample_metrics, **keyword_args)
        for sample_metrics in per_sample
    )


def _save_prediction_montages(
    output_dir: Path,
    *,
    names: list[str],
    images: torch.Tensor,
    target_rgb: torch.Tensor,
    prediction: torch.Tensor,
    colormap: str,
) -> list[str]:
    mapper = DepthRGBMapper(colormap)
    prediction_rgb = mapper.normalized_to_rgb(prediction.cpu())
    paths = []
    for index, name in enumerate(names):
        source = ((images[index].clamp(-1, 1) + 1.0) / 2.0).permute(1, 2, 0)
        target = target_rgb[index].clamp(0, 1).permute(1, 2, 0)
        predicted = prediction_rgb[index].clamp(0, 1).permute(1, 2, 0)
        panels = [
            Image.fromarray((panel.numpy() * 255).round().astype(np.uint8))
            for panel in (source, target, predicted)
        ]
        montage = Image.new("RGB", (panels[0].width * 3, panels[0].height))
        for panel_index, panel in enumerate(panels):
            montage.paste(panel, (panel_index * panel.width, 0))
        path = output_dir / f"sample_{index:02d}_{Path(name).stem}.png"
        montage.save(path)
        paths.append(str(path.resolve()))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="segmind/SSD-1B")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--upsample-stages", type=int, default=3)
    parser.add_argument("--gradient-weight", type=float, default=0.5)
    parser.add_argument("--pearson-weight", type=float, default=0.1)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--max-mae", type=float, default=0.05)
    parser.add_argument("--min-pearson", type=float, default=0.95)
    parser.add_argument("--min-spearman", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    positive = {
        "num_samples": args.num_samples,
        "resolution": args.resolution,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "lr": args.lr,
        "width": args.width,
    }
    if any(value <= 0 for value in positive.values()):
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

    names = declared_depth_train_names(data_dir, num_samples=args.num_samples)
    images, target_rgb = _load_declared_samples(
        data_dir,
        names=names,
        resolution=args.resolution,
    )
    audit = audit_depth_target_encoding(
        data_dir / "depth" / "targets",
        allowed_names=set(names),
    )
    colormap = audit.detected_colormap
    prompt = relative_depth_prompt(colormap)
    target_position = DepthRGBMapper(colormap).rgb_to_normalized(target_rgb)
    logger.info(
        "Stage-6 data: samples=%d resolution=%d colormap=%s device=%s",
        len(names),
        args.resolution,
        colormap,
        device,
    )

    generator = BaseGenerator(
        model_name=args.model,
        device=str(device),
        dtype=torch.bfloat16 if device.type == "mps" else torch.float32,
    )
    contract, cpu_features, feature_shapes = _cache_frozen_features(
        generator,
        images,
        prompt=prompt,
        timestep=args.timestep,
    )
    logger.info("Frozen feature contract: %s", contract.to_dict())
    logger.info("Frozen feature shapes: %s", feature_shapes)
    del generator
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    features = tuple(feature.to(device=device, dtype=torch.float32) for feature in cpu_features)
    target = target_position.to(device=device, dtype=torch.float32)
    head = ScalarDepthProbe(
        contract,
        width=args.width,
        upsample_stages=args.upsample_stages,
    ).to(device=device, dtype=torch.float32)
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.steps,
        eta_min=args.lr * 0.01,
    )
    logger.info("Trainable scalar-depth head parameters: %d", trainable_parameters)

    best_score = -float("inf")
    best_step = 0
    best_state = _clone_state_dict(head)
    best_metrics = evaluate_scalar_depth_predictions(
        head(features, output_size=target.shape[-2:]),
        target,
    )
    history: list[dict[str, Any]] = []
    gate_passed = False
    steps_run = 0
    for step in range(1, args.steps + 1):
        head.train()
        prediction = head(features, output_size=target.shape[-2:])
        loss = scalar_depth_probe_loss(
            prediction,
            target,
            gradient_weight=args.gradient_weight,
            pearson_weight=args.pearson_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        steps_run = step

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            head.eval()
            with torch.no_grad():
                evaluated = head(features, output_size=target.shape[-2:])
            metrics = evaluate_scalar_depth_predictions(evaluated, target)
            score = metrics["pearson"] + metrics["spearman"] - metrics["mae"]
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                **metrics,
            }
            history.append(record)
            logger.info(
                "Step %d | loss=%.6f mae=%.5f pearson=%.5f spearman=%.5f",
                step,
                record["loss"],
                metrics["mae"],
                metrics["pearson"],
                metrics["spearman"],
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_state = _clone_state_dict(head)
                best_metrics = metrics
            gate_passed = _all_metrics_pass_gate(
                metrics,
                _per_sample_metrics(evaluated, target, names),
                max_mae=args.max_mae,
                min_pearson=args.min_pearson,
                min_spearman=args.min_spearman,
            )
            if gate_passed:
                best_step = step
                best_state = _clone_state_dict(head)
                best_metrics = metrics
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        final_prediction = head(features, output_size=target.shape[-2:])
    final_metrics = evaluate_scalar_depth_predictions(final_prediction, target)
    per_sample_metrics = _per_sample_metrics(final_prediction, target, names)
    gate_passed = _all_metrics_pass_gate(
        final_metrics,
        per_sample_metrics,
        max_mae=args.max_mae,
        min_pearson=args.min_pearson,
        min_spearman=args.min_spearman,
    )
    checkpoint_path = output_dir / "depth_probe.pt"
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "feature_contract": contract.to_dict(),
            "model": args.model,
            "prompt": prompt,
            "colormap": colormap,
            "resolution": args.resolution,
            "timestep": args.timestep,
            "width": args.width,
            "upsample_stages": args.upsample_stages,
            "sample_names": names,
            "metrics": final_metrics,
            "per_sample_metrics": per_sample_metrics,
        },
        checkpoint_path,
    )
    montage_paths = _save_prediction_montages(
        output_dir,
        names=names,
        images=images,
        target_rgb=target_rgb,
        prediction=final_prediction.cpu(),
        colormap=colormap,
    )
    result = {
        "gate_passed": gate_passed,
        "decision": "continue_ssd_depth" if gate_passed else "stop_ssd_depth_mainline",
        "model": args.model,
        "data_dir": str(data_dir),
        "sample_names": names,
        "resolution": args.resolution,
        "seed": args.seed,
        "timestep": args.timestep,
        "colormap": colormap,
        "prompt": prompt,
        "steps_requested": args.steps,
        "steps_run": steps_run,
        "best_step": best_step,
        "trainable_parameters": trainable_parameters,
        "feature_contract": contract.to_dict(),
        "feature_shapes": feature_shapes,
        "thresholds": {
            "max_mae": args.max_mae,
            "min_pearson": args.min_pearson,
            "min_spearman": args.min_spearman,
        },
        "metrics": final_metrics,
        "per_sample_metrics": per_sample_metrics,
        "best_metrics_during_training": best_metrics,
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path.resolve()),
        "prediction_montages": montage_paths,
    }
    write_json_report(output_dir / "result.json", result)
    logger.info("Stage-6 gate result: %s", json.dumps(result["metrics"], sort_keys=True))
    logger.info("Gate passed: %s | decision=%s", gate_passed, result["decision"])


if __name__ == "__main__":
    main()
