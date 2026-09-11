#!/usr/bin/env python3
"""Compare SANA structure checkpoints with tolerance-aware edge metrics.

The structure task uses sparse binary edges, so RGB reconstruction loss alone
does not show whether generated edges align with the target.  This evaluator
uses a fixed, evenly spaced validation subset and reports precision, recall,
and F1 after allowing a small spatial tolerance around each edge.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        required=True,
        help="PEFT lora_adapter directories, one per checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--tolerance-px", type=float, default=2.0)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Luminance thresholds used to binarize generated edge maps.",
    )
    parser.add_argument("--model-name", default="sana-600m-512px")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Choose deterministic, evenly distributed sample indices without repeats."""
    if total < 1:
        return []
    if count < 1:
        raise ValueError("num_samples must be positive")
    return np.linspace(0, total - 1, min(total, count), dtype=np.int64).tolist()


def rgb_luminance(image: np.ndarray) -> np.ndarray:
    """Convert a [H, W, 3] RGB image in [0, 1] to luminance."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def _distance_to_edges(edges: np.ndarray) -> np.ndarray:
    """Return distance from every pixel to the nearest true edge pixel."""
    return cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 5)


def edge_counts(prediction: np.ndarray, target: np.ndarray, tolerance_px: float) -> dict[str, int]:
    """Compute non-one-to-one tolerance matching counts for binary edge maps."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {prediction.shape} != {target.shape}")
    if prediction.ndim != 2:
        raise ValueError(f"Expected 2D masks, got {prediction.shape}")
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be non-negative")

    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    pred_total = int(prediction.sum())
    target_total = int(target.sum())
    if pred_total == 0 or target_total == 0:
        return {
            "matched_prediction": 0,
            "prediction_total": pred_total,
            "matched_target": 0,
            "target_total": target_total,
        }

    pred_distance = _distance_to_edges(prediction)
    target_distance = _distance_to_edges(target)
    return {
        "matched_prediction": int((target_distance[prediction] <= tolerance_px).sum()),
        "prediction_total": pred_total,
        "matched_target": int((pred_distance[target] <= tolerance_px).sum()),
        "target_total": target_total,
    }


def edge_scores(counts: dict[str, int]) -> dict[str, float]:
    """Convert aggregated tolerance counts to precision, recall, and F1."""
    pred_total = counts["prediction_total"]
    target_total = counts["target_total"]
    precision = counts["matched_prediction"] / pred_total if pred_total else 0.0
    recall = counts["matched_target"] / target_total if target_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def add_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: left.get(key, 0) + right.get(key, 0) for key in right}


def tensor_to_numpy_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert a [3, H, W] tensor in [0, 1] to a float RGB numpy array."""
    return image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()


def checkpoint_label(checkpoint: Path) -> str:
    """Use the checkpoint directory instead of the shared lora_adapter suffix."""
    return checkpoint.parent.name if checkpoint.name == "lora_adapter" else checkpoint.name


def evaluate_checkpoint(args, checkpoint: Path, dataset, indices: list[int]) -> dict:
    """Generate a fixed validation subset and summarize every threshold."""
    from infer_sana_512 import (
        compute_l1,
        compute_psnr,
        create_comparison_grid,
        flow_matching_inference,
        get_dtype,
        load_model_and_lora,
    )

    model_args = argparse.Namespace(
        model_name=args.model_name,
        checkpoint=str(checkpoint),
        lora_rank=32,
        lora_alpha=64,
        no_rslora=False,
        dtype=args.dtype,
    )
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps"))
    dtype = get_dtype(args.dtype)
    peft_model, vae, components, model_cfg = load_model_and_lora(model_args, device, dtype)
    model_dtype = next(peft_model.parameters()).dtype
    output_dir = args.output_dir / checkpoint_label(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts_by_threshold = {
        threshold: {
            "matched_prediction": 0,
            "prediction_total": 0,
            "matched_target": 0,
            "target_total": 0,
        }
        for threshold in args.thresholds
    }
    psnr_values: list[float] = []
    l1_values: list[float] = []
    positive_fractions: dict[float, list[float]] = {threshold: [] for threshold in args.thresholds}
    samples: list[dict] = []

    for result_index, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        input_image = sample["image"]
        target_image = sample["target_rgb"]
        prompt = sample["prompt"]
        start = time.monotonic()
        prediction = flow_matching_inference(
            peft_model,
            vae,
            components,
            input_image.unsqueeze(0).to(device=device, dtype=model_dtype),
            [prompt],
            device,
            model_dtype,
            num_steps=args.num_inference_steps,
            use_img2img=True,
            max_length=model_cfg.text_encoder_max_length,
            guidance_scale=args.guidance_scale,
        )[0].cpu()
        elapsed = time.monotonic() - start
        target_rgb = tensor_to_numpy_rgb(target_image)
        prediction_rgb = tensor_to_numpy_rgb(prediction)
        target_binary = rgb_luminance(target_rgb) >= 0.5
        prediction_luminance = rgb_luminance(prediction_rgb)

        psnr = compute_psnr(prediction.unsqueeze(0), target_image.unsqueeze(0))
        l1 = compute_l1(prediction.unsqueeze(0), target_image.unsqueeze(0))
        psnr_values.append(psnr)
        l1_values.append(l1)
        sample_result = {
            "dataset_index": dataset_index,
            "psnr": psnr,
            "l1": l1,
            "seconds": elapsed,
            "target_positive_fraction": float(target_binary.mean()),
        }

        for threshold in args.thresholds:
            prediction_binary = prediction_luminance >= threshold
            counts = edge_counts(prediction_binary, target_binary, args.tolerance_px)
            counts_by_threshold[threshold] = add_counts(counts_by_threshold[threshold], counts)
            positive_fractions[threshold].append(float(prediction_binary.mean()))

        binary_prediction = torch.from_numpy(
            (prediction_luminance >= 0.5).astype(np.float32)
        ).unsqueeze(0).repeat(3, 1, 1)
        input_display = ((input_image + 1.0) / 2.0).clamp(0, 1)
        grid = create_comparison_grid(
            input_display,
            binary_prediction,
            target_image,
            "structure_binary",
            f"{checkpoint_label(checkpoint)} | dataset index {dataset_index}",
            result_index,
        )
        grid.save(output_dir / f"sample_{result_index:02d}_index_{dataset_index:04d}.png")
        samples.append(sample_result)
        LOGGER.info(
            "%s sample %d/%d: index=%d PSNR=%.2f L1=%.4f %.1fs",
            checkpoint_label(checkpoint),
            result_index + 1,
            len(indices),
            dataset_index,
            psnr,
            l1,
            elapsed,
        )

    threshold_metrics = []
    for threshold in args.thresholds:
        counts = counts_by_threshold[threshold]
        threshold_metrics.append(
            {
                "threshold": threshold,
                **edge_scores(counts),
                **counts,
                "mean_prediction_positive_fraction": float(np.mean(positive_fractions[threshold])),
            }
        )
    best_threshold = max(threshold_metrics, key=lambda item: item["f1"])
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_label": checkpoint_label(checkpoint),
        "num_samples": len(indices),
        "sample_indices": indices,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "tolerance_px": args.tolerance_px,
        "mean_psnr": float(np.mean(psnr_values)),
        "mean_l1": float(np.mean(l1_values)),
        "threshold_metrics": threshold_metrics,
        "best_threshold": best_threshold,
        "samples": samples,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    del peft_model, vae, components
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.tolerance_px < 0:
        raise SystemExit("--tolerance-px must be non-negative")
    if not args.pair_manifest.is_file():
        raise SystemExit(f"Pair manifest not found: {args.pair_manifest}")
    if not args.checkpoints:
        raise SystemExit("At least one checkpoint is required")
    for checkpoint in args.checkpoints:
        if not (checkpoint / "adapter_model.safetensors").is_file():
            raise SystemExit(f"Missing PEFT adapter: {checkpoint / 'adapter_model.safetensors'}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    torch.manual_seed(args.seed)
    from train_sana_512 import ManifestVisionDataset

    dataset = ManifestVisionDataset(args.pair_manifest, args.resolution, "val", {"structure"})
    indices = select_evenly_spaced_indices(len(dataset), args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_checkpoint(args, checkpoint, dataset, indices) for checkpoint in args.checkpoints]
    ranking = sorted(results, key=lambda item: item["best_threshold"]["f1"], reverse=True)
    report = {
        "selection_metric": "best_threshold.f1",
        "note": "Threshold sweep is applied identically to each checkpoint; use the chosen threshold for deployment.",
        "ranking": [
            {
                "checkpoint_label": item["checkpoint_label"],
                "best_threshold": item["best_threshold"],
                "mean_psnr": item["mean_psnr"],
                "mean_l1": item["mean_l1"],
            }
            for item in ranking
        ],
        "results": results,
    }
    (args.output_dir / "checkpoint_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    LOGGER.info("Checkpoint ranking by tolerance F1:")
    for position, item in enumerate(ranking, 1):
        best = item["best_threshold"]
        LOGGER.info(
            "  %d. %s F1=%.4f P=%.4f R=%.4f threshold=%.2f",
            position,
            item["checkpoint_label"],
            best["f1"],
            best["precision"],
            best["recall"],
            best["threshold"],
        )


if __name__ == "__main__":
    main()
