#!/usr/bin/env python3
"""Run inference and save visualizations for frozen SSD depth-probe checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_frozen_depth_features import (  # noqa: E402
    _cache_features,
    _load_samples,
    _predict_in_batches,
)

from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.depth_probe_checkpoint import (  # noqa: E402
    DepthProbeArtifact,
    load_depth_probe_artifact,
)
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.training.depth_probe import (  # noqa: E402
    DepthFeatureContract,
    ScalarDepthProbe,
    evaluate_scalar_depth_predictions,
)

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _labelled_panel(image: Image.Image, label: str) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 22), fill=(0, 0, 0))
    draw.text((6, 4), label, fill=(255, 255, 255))
    return result


def _save_triptych(
    path: Path,
    *,
    source: torch.Tensor,
    target_rgb: torch.Tensor,
    prediction_rgb: torch.Tensor,
    size: int,
) -> None:
    panels = [
        _labelled_panel(_to_pil((source.clamp(-1, 1) + 1) / 2), "input"),
        _labelled_panel(_to_pil(target_rgb), "ground truth"),
        _labelled_panel(_to_pil(prediction_rgb), "prediction"),
    ]
    panels = [panel.resize((size, size), Image.Resampling.BILINEAR) for panel in panels]
    canvas = Image.new("RGB", (size * 3, size), (255, 255, 255))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * size, 0))
    canvas.save(path)


def _save_contact_sheet(
    path: Path,
    *,
    names: list[str],
    source: torch.Tensor,
    target_rgb: torch.Tensor,
    prediction_rgb: torch.Tensor,
    indices: list[int],
    tile_size: int,
    columns: int,
) -> None:
    rows = (len(indices) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size * 3, rows * tile_size), "white")
    for position, index in enumerate(indices):
        tile_path = path.parent / f".contact_{position:03d}.png"
        _save_triptych(
            tile_path,
            source=source[index],
            target_rgb=target_rgb[index],
            prediction_rgb=prediction_rgb[index],
            size=tile_size,
        )
        tile = Image.open(tile_path).convert("RGB")
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((6, tile_size - 18), Path(names[index]).stem, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        x = (position % columns) * tile_size * 3
        y = (position // columns) * tile_size
        sheet.paste(tile, (x, y))
        tile_path.unlink()
    sheet.save(path)


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer values")
    return seeds


def _resolve_checkpoint(checkpoint_dir: Path, seed: int) -> Path:
    artifact_dir = checkpoint_dir / f"seed{seed}_vae_ssd"
    if artifact_dir.is_dir():
        return artifact_dir
    final_artifact_dir = checkpoint_dir / f"seed{seed}_vae_ssd_final"
    if final_artifact_dir.is_dir():
        return final_artifact_dir
    legacy_path = checkpoint_dir / f"seed{seed}_vae_ssd.pt"
    if legacy_path.is_file():
        return legacy_path
    raise FileNotFoundError(f"No VAE+SSD depth probe found for seed {seed}")


def _official_val_names(artifact: DepthProbeArtifact) -> list[str]:
    names = artifact.splits.get("official_val_names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"Checkpoint {artifact.path} has no official validation split")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means all official-val samples")
    parser.add_argument("--contact-count", type=int, default=12)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    seeds = _parse_seeds(args.seeds)
    if args.limit < 0 or args.contact_count <= 0 or args.contact_columns <= 0 or args.size <= 0:
        parser.error("limit must be non-negative; contact and size values must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    first_checkpoint = _resolve_checkpoint(checkpoint_dir, seeds[0])
    metadata = load_depth_probe_artifact(first_checkpoint)
    names = _official_val_names(metadata)
    if args.limit:
        names = names[: args.limit]
    resolution = metadata.resolution
    model_name = args.model or metadata.model
    prompt = metadata.prompt
    colormap = metadata.colormap
    mapper = DepthRGBMapper(colormap)
    images, target_rgb = _load_samples(data_dir, names=names, resolution=resolution)
    source_images = images.clone()
    target = mapper.rgb_to_normalized(target_rgb)
    device = _resolve_device(args.device)
    logger.info("Loading %s and extracting %d official-val feature tuples", model_name, len(names))
    generator = BaseGenerator(model_name=model_name, device=str(device), dtype=torch.bfloat16 if device.type == "mps" else torch.float32)
    contract, cpu_features, feature_shapes = _cache_features(
        generator,
        images,
        prompt=prompt,
        timestep=metadata.timestep,
    )
    del generator, images
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    features = tuple(feature.to(device=device, dtype=torch.float32) for feature in cpu_features)
    del cpu_features
    target_device = target.to(device=device, dtype=torch.float32)
    output_size = tuple(int(size) for size in target.shape[-2:])
    all_reports: dict[str, Any] = {"model": model_name, "prompt": prompt, "colormap": colormap, "resolution": resolution, "names": names, "feature_contract": contract.to_dict(), "feature_shapes": feature_shapes, "checkpoints": {}}
    for seed in seeds:
        checkpoint_path = _resolve_checkpoint(checkpoint_dir, seed)
        checkpoint = load_depth_probe_artifact(checkpoint_path)
        saved_contract = DepthFeatureContract.from_dict(checkpoint.feature_contract)
        if saved_contract != contract:
            raise ValueError(f"Feature contract mismatch in {checkpoint_path}")
        head = ScalarDepthProbe(
            contract,
            width=checkpoint.width,
            upsample_stages=checkpoint.upsample_stages,
        ).to(device=device, dtype=torch.float32)
        head.load_state_dict(checkpoint.head_state_dict, strict=True)
        prediction = _predict_in_batches(
            head,
            features,
            feature_mask=checkpoint.feature_mask,
            output_size=output_size,
            batch_size=8,
        )
        prediction_rgb = mapper.normalized_to_rgb(prediction)
        seed_dir = output_dir / f"seed{seed}_vae_ssd"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(names):
            _save_triptych(seed_dir / f"{index:03d}_{Path(name).stem}.png", source=source_images[index], target_rgb=target_rgb[index], prediction_rgb=prediction_rgb[index], size=args.size)
        contact_indices = np.linspace(0, len(names) - 1, min(args.contact_count, len(names)), dtype=int).tolist()
        _save_contact_sheet(seed_dir / "contact_sheet.png", names=names, source=source_images, target_rgb=target_rgb, prediction_rgb=prediction_rgb, indices=contact_indices, tile_size=args.size // 3, columns=args.contact_columns)
        torch.save({"names": names, "prediction": prediction.cpu(), "target": target.cpu(), "selected_step": checkpoint.selected_step, "checkpoint": str(checkpoint_path)}, seed_dir / "predictions.pt")
        metrics = evaluate_scalar_depth_predictions(prediction, target_device)
        all_reports["checkpoints"][str(seed)] = {"path": str(checkpoint_path), "selected_step": checkpoint.selected_step, "metrics": metrics, "output_dir": str(seed_dir)}
        del head, prediction, prediction_rgb
    _write_json(output_dir / "visualization_result.json", all_reports)
    logger.info("Saved frozen depth visualizations under %s", output_dir)


if __name__ == "__main__":
    main()
