#!/usr/bin/env python3
"""Run inference and visualize SSD-1B Probe V5 depth predictions.

Loads V5 best_probe.pt checkpoint, extracts frozen SSD-1B features,
runs ScalarDepthProbeV5 inference on validation samples, and generates
montage visualizations (input | ground truth | prediction) + contact sheet.

Usage:
    python scripts/visualize_probe_v5.py \
        --checkpoint output/ssd_probe_v5/best_probe.pt \
        --data-dir data/vb_snvc_train_v3 \
        --output-dir output/ssd_probe_v5/visualizations \
        --max-count 20
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

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
)

logger = logging.getLogger(__name__)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _load_image(path: Path, resolution: int, range_: str = "01") -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    if range_ == "11":
        t = t * 2.0 - 1.0
    return t


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _labelled_panel(image: Image.Image, label: str) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, result.width, 24), fill=(0, 0, 0))
    draw.text((8, 4), label, fill=(255, 255, 255))
    return result


def _save_triptych(path: Path, *, source: torch.Tensor, target_rgb: torch.Tensor, prediction_rgb: torch.Tensor, size: int) -> None:
    panels = [
        _labelled_panel(_to_pil((source.clamp(-1, 1) + 1) / 2), "input"),
        _labelled_panel(_to_pil(target_rgb), "ground truth"),
        _labelled_panel(_to_pil(prediction_rgb), "prediction"),
    ]
    panels = [p.resize((size, size), Image.Resampling.BILINEAR) for p in panels]
    canvas = Image.new("RGB", (size * 3, size), (255, 255, 255))
    for i, p in enumerate(panels):
        canvas.paste(p, (i * size, 0))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to V5 best_probe.pt")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Override model name (default: from checkpoint)")
    parser.add_argument("--max-count", type=int, default=20, help="Max samples to visualize")
    parser.add_argument("--size", type=int, default=256, help="Output image size per panel")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    device = _resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load checkpoint ──────────────────────────────────────
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model_name = args.model or ckpt["model"]
    prompt = ckpt["prompt"]
    colormap = ckpt["colormap"]
    resolution = ckpt["resolution"]
    timestep = ckpt["timestep"]
    width = ckpt["width"]
    upsample_stages = ckpt["upsample_stages"]
    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})

    logger.info("  model: %s | colormap: %s | resolution: %d", model_name, colormap, resolution)
    logger.info("  epoch: %s | width: %d | upsample_stages: %d", epoch, width, upsample_stages)
    logger.info("  checkpoint metrics: MAE=%.5f Pearson=%.5f", metrics.get("mae", 0), metrics.get("pearson", 0))

    # ── 2. Load data manifest ───────────────────────────────────
    data_dir = args.data_dir.expanduser().resolve()
    split_path = data_dir / "train_val_split.json"
    with open(split_path) as f:
        split = json.load(f)
    val_names = split.get("val", {}).get("depth", [])

    if not val_names:
        logger.error("No val names found in train_val_split.json")
        sys.exit(1)

    logger.info("Val samples available: %d", len(val_names))

    # ── 3. Audit depth encoding ────────────────────────────────
    audit = audit_depth_target_encoding(
        data_dir / "depth" / "targets",
        allowed_names=set(val_names),
    )
    assert audit.detected_colormap == colormap, f"Colormap mismatch: {audit.detected_colormap} vs {colormap}"
    mapper = DepthRGBMapper(colormap)
    logger.info("Colormap verified: %s", colormap)

    # ── 4. Load images and targets ─────────────────────────────
    max_count = min(args.max_count, len(val_names))
    names = val_names[:max_count]
    logger.info("Loading %d val images...", max_count)

    images = []
    target_rgbs = []
    for name in names:
        # Name may already include extension (.jpg) or be bare stem
        stem = name.rsplit(".", 1)[0] if "." in name else name
        img_path = data_dir / "depth" / "images" / f"{stem}.jpg"
        tgt_path = data_dir / "depth" / "targets" / f"{stem}.png"
        if not img_path.exists():
            logger.warning("Missing image: %s", img_path)
            continue
        images.append(_load_image(img_path, resolution, "11"))
        target_rgbs.append(_load_image(tgt_path, resolution, "01"))

    images_tensor = torch.stack(images)
    target_rgb_tensor = torch.stack(target_rgbs)
    target = mapper.rgb_to_normalized(target_rgb_tensor)
    logger.info("Loaded %d samples, image shape: %s", len(images), images_tensor.shape)

    # ── 5. Extract frozen features ─────────────────────────────
    logger.info("Loading %s and extracting features...", model_name)
    generator = BaseGenerator(
        model_name=model_name,
        device=str(device),
        dtype=torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32,
    )

    contract_dict = ckpt["feature_contract"]
    contract = DepthFeatureContract.from_dict(contract_dict)
    extractor = FrozenSDXLDepthFeatureExtractor(generator, timestep=timestep)

    logger.info("Extracting features for %d images...", len(images))
    extracted_contract, features = extractor.extract(images_tensor, prompt=prompt)

    # Verify contract matches checkpoint
    if extracted_contract != contract:
        logger.warning("Feature contract mismatch — using extracted contract")
    contract = extracted_contract

    # Free generator memory
    del generator, extractor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # ── 6. Load V5 probe and run inference ─────────────────────
    head = ScalarDepthProbeV5(
        contract,
        width=width,
        upsample_stages=upsample_stages,
    ).to(device=device, dtype=torch.float32)
    head.load_state_dict(ckpt["head_state_dict"], strict=True)
    head.eval()

    logger.info("Running inference...")
    output_size = tuple(int(v) for v in target.shape[-2:])
    features_device = tuple(f.to(device=device, dtype=torch.float32) for f in features)
    del features

    all_preds = []
    batch_size = 8
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch_features = tuple(f[start:end] for f in features_device)
            pred = head(batch_features, output_size=output_size)
            all_preds.append(pred.cpu())

    prediction = torch.cat(all_preds, dim=0)
    prediction_rgb = mapper.normalized_to_rgb(prediction)

    # ── 7. Evaluate ────────────────────────────────────────────
    target_device = target.to(device=device, dtype=torch.float32)
    pred_device = prediction.to(device=device, dtype=torch.float32)
    eval_metrics = evaluate_scalar_depth_predictions(pred_device, target_device)
    logger.info("Evaluation metrics:")
    for k, v in eval_metrics.items():
        logger.info("  %s: %.5f", k, v)

    # ── 8. Save montages ───────────────────────────────────────
    logger.info("Saving montages to %s", output_dir)
    count = min(max_count, len(names))
    montage_paths = []
    for index in range(count):
        path = output_dir / f"montage_{index:03d}_{Path(names[index]).stem}.png"
        _save_triptych(
            path,
            source=images_tensor[index],
            target_rgb=target_rgb_tensor[index],
            prediction_rgb=prediction_rgb[index],
            size=args.size,
        )
        montage_paths.append(str(path))

    # ── 9. Save contact sheet ──────────────────────────────────
    contact_count = min(12, count)
    contact_indices = np.linspace(0, count - 1, contact_count, dtype=int).tolist()
    tile_size = args.size
    columns = 2
    rows = (len(contact_indices) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size * 3, rows * tile_size), "white")

    for pos, idx in enumerate(contact_indices):
        tile_path = output_dir / f".contact_{pos:03d}.png"
        _save_triptych(
            tile_path,
            source=images_tensor[idx],
            target_rgb=target_rgb_tensor[idx],
            prediction_rgb=prediction_rgb[idx],
            size=tile_size,
        )
        tile = Image.open(tile_path).convert("RGB")
        draw = ImageDraw.Draw(tile)
        draw.text((8, tile_size - 20), Path(names[idx]).stem, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        x = (pos % columns) * tile_size * 3
        y = (pos // columns) * tile_size
        sheet.paste(tile, (x, y))
        tile_path.unlink()

    contact_path = output_dir / "contact_sheet.png"
    sheet.save(contact_path)
    logger.info("Contact sheet: %s", contact_path)

    # ── 10. Save metrics JSON ──────────────────────────────────
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": epoch,
        "model": model_name,
        "colormap": colormap,
        "resolution": resolution,
        "sample_count": count,
        "metrics": eval_metrics,
        "montage_paths": montage_paths,
        "contact_sheet": str(contact_path.resolve()),
    }
    metrics_path = output_dir / "visualization_result.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results: %s", metrics_path)

    print("\n" + "=" * 60)
    print("V5 Inference Visualization Complete")
    print("=" * 60)
    print(f"  Samples:     {count}")
    print(f"  MAE:         {eval_metrics.get('mae', 0):.5f}")
    print(f"  Pearson:     {eval_metrics.get('pearson', 0):.5f}")
    print(f"  Spearman:    {eval_metrics.get('spearman', 0):.5f}")
    print(f"  RMSE:        {eval_metrics.get('rmse', 0):.5f}")
    print(f"  Montages:    {count} files in {output_dir}")
    print(f"  Contact:     {contact_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
