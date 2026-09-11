#!/usr/bin/env python3
"""Run the validated specialist paths on the exact SANA evaluation subset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_sana_structure_binary_targets import build_binary_structure_map
from infer_depth_pseudo import (
    depth_to_rgb,
    estimate_depth,
    load_depth_estimator,
    normalize_depth_to_uint8,
    select_device,
)
from infer_depth_normal import normal_from_relative_depth
from infer_task_specialists import (
    LocalDINOv2ViTS14,
    dino_features,
    load_checkpoint,
    load_normal_head,
    load_semantic_head,
    normal_to_diode_rgb,
    semantic_to_rgb,
    expand_normal_grid,
)
from mar_vb_eval_v6 import edge_f1_ods, normal_angle_deg, segmentation_metrics, quantize_palette
from open_vision_banana.evaluation.relative_depth import evaluate_relative_depth_rgb


TASKS = ("depth", "normal", "semantic", "structure")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair-manifest", type=Path, required=True)
    p.add_argument("--sana-metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-per-task", type=int, default=32)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None)
    p.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    p.add_argument("--normal-checkpoint", type=Path, required=True)
    p.add_argument("--semantic-checkpoint", type=Path, required=True)
    p.add_argument("--semantic-palette", type=Path, required=True)
    p.add_argument("--dino-repository", type=Path, default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main")
    p.add_argument("--dino-weights", type=Path, default=Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth")
    return p.parse_args()


def load_rows(path: Path, n: int) -> dict[str, list[dict]]:
    rows = {task: [] for task in TASKS}
    aliases = {"seg": "semantic", "segmentation": "semantic"}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != "val":
            continue
        task = aliases.get(row.get("task"), row.get("task"))
        if task in rows:
            rows[task].append(row)
    for task in TASKS:
        rows[task].sort(key=lambda row: row["input"])
        rows[task] = rows[task][:n]
    return {task: values for task, values in rows.items() if values}


def load_rgb(path: str, resolution: int, nearest: bool = False) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)


def palette_metrics(pred: torch.Tensor, target: torch.Tensor, palette_info: dict) -> dict:
    palette, ignore = palette_info["palette"], palette_info["ignore"]
    pred_labels = quantize_palette(pred, palette)
    target_pixels = target.permute(1, 2, 0).numpy().reshape(-1, 3)
    valid = np.max(np.abs(target_pixels - ignore), axis=1) > 0.5 / 255.0
    target_labels = quantize_palette(target, palette)
    confusion = np.bincount(len(palette) * target_labels[valid] + pred_labels[valid], minlength=len(palette) ** 2).reshape(len(palette), len(palette))
    return segmentation_metrics(confusion)


def main() -> None:
    a = args()
    rows = load_rows(a.pair_manifest, a.n_per_task)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = select_device(a.device)
    palette_data = json.loads(a.semantic_palette.read_text(encoding="utf-8"))
    palette_info = {
        "palette": np.asarray(palette_data["class_palette"], dtype=np.float32) / 255.0,
        "ignore": np.asarray(palette_data.get("ignore_color", [0, 0, 0]), dtype=np.float32) / 255.0,
    }
    report: dict = {
        "model": "validated specialists",
        "subset_manifest": str(a.pair_manifest.resolve()),
        "sana_metrics": str(a.sana_metrics.resolve()),
        "selection": "same sorted val rows as SANA; first n_per_task",
        "config": {"n_per_task": a.n_per_task, "resolution": a.resolution, "device": str(device)},
        "per_task": {},
        "samples": {},
    }
    start_all = time.monotonic()

    depth_model, depth_processor = load_depth_estimator(a.depth_model, device)
    normal_ckpt = load_checkpoint(a.normal_checkpoint, "normal")
    semantic_ckpt = load_checkpoint(a.semantic_checkpoint, "semantic")
    dino_size = int(semantic_ckpt["image_size"])
    backbone = LocalDINOv2ViTS14(a.dino_repository.resolve(), a.dino_weights.resolve()).to(device).eval()
    normal_head = load_normal_head(normal_ckpt, device)
    semantic_head = load_semantic_head(semantic_ckpt, device)

    for task, task_rows in rows.items():
        values = []
        for index, row in enumerate(task_rows):
            source = load_rgb(row["input"], a.resolution)
            target = load_rgb(row["target"], a.resolution, nearest=task == "semantic")
            source_image = Image.open(row["input"]).convert("RGB")
            if task == "depth":
                depth = estimate_depth(source_image, depth_model, depth_processor, device)
            if task in ("normal", "semantic"):
                tokens = dino_features(source_image, dino_size, backbone, device)
            source_image.close()
            if task == "depth":
                pred = depth_to_rgb(normalize_depth_to_uint8(depth)).resize((a.resolution, a.resolution), Image.Resampling.LANCZOS)
                prediction = torch.from_numpy(np.asarray(pred, dtype=np.float32) / 255.0).permute(2, 0, 1)
                metric = evaluate_relative_depth_rgb(prediction[None], target[None], colormap="turbo")
                metric = {f"specialist_{key}": value for key, value in metric.items()}
            elif task == "normal":
                with torch.no_grad():
                    normal_tokens = normal_head(torch.from_numpy(tokens).to(device)).cpu().numpy()
                normal = expand_normal_grid(normal_tokens, dino_size)[0]
                prediction = torch.from_numpy((normal + 1.0) * 0.5).permute(2, 0, 1)
                prediction = torch.nn.functional.interpolate(prediction[None], size=(a.resolution, a.resolution), mode="nearest")[0]
                # The v5 normal target is collapsed; keep this number only as an audit.
                metric = {"specialist_normal_mean_angular_error_deg": normal_angle_deg(prediction, target)}
            elif task == "semantic":
                with torch.no_grad():
                    labels = semantic_head(torch.from_numpy(tokens).to(device)).argmax(-1).cpu().numpy().reshape(dino_size // 14, dino_size // 14)
                prediction_image = semantic_to_rgb(labels, (palette_info["palette"] * 255).astype(np.uint8)).resize((a.resolution, a.resolution), Image.Resampling.NEAREST)
                prediction = torch.from_numpy(np.asarray(prediction_image, dtype=np.float32) / 255.0).permute(2, 0, 1)
                metric = {f"specialist_{key}": value for key, value in palette_metrics(prediction, target, palette_info).items()}
            else:
                binary = build_binary_structure_map(row["input"], a.resolution, 0.08, 3)
                prediction = torch.from_numpy(binary.astype(np.float32) / 255.0).permute(2, 0, 1)
                metric = {"specialist_structure_edge_f1_ods": edge_f1_ods(prediction, target), "specialist_structure_positive_fraction": float((prediction.mean(0) >= 0.5).float().mean())}
            sample = {"index": index, "input": row["input"], "target": row["target"], **metric}
            values.append(sample)
        report["samples"][task] = values
        numeric = sorted({key for row in values for key in row if isinstance(row.get(key), (int, float))})
        report["per_task"][task] = {key: float(np.mean([row[key] for row in values])) for key in numeric}
        report["per_task"][task]["count"] = len(values)
    report["elapsed_sec"] = time.monotonic() - start_all
    report["notes"] = {"normal": "depth-derived specialist compared against collapsed v5 normal target; audit only.", "depth": "Depth-Anything v2 pseudo-depth is the target generator family.", "semantic": "DINOv2 head uses fixed 150-class palette.", "structure": "Canny/LSD rule-based map."}
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"per_task": report["per_task"], "metrics": str((out / "metrics.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
