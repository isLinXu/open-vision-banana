#!/usr/bin/env python3
"""Diagnose why official Cityscapes mIoU is near zero.

ovb-fix_ctrl_seg was trained on SNVC ADE-150 (seed=42) colors. Paper-protocol
eval decodes with the official 19-class Cityscapes palette. This script scores
a small val slice under both alphabets so the two numbers are not confused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.datasets import CityscapesSegDataset
from open_vision_banana.evaluation.semantic import evaluate_semantic
from open_vision_banana.inference.pipeline import VisionBanana
from open_vision_banana.core.segmentation.decoder import SegmentationDecoder
from open_vision_banana.task_prompts import CITYSCAPES_RGB_LEGEND, SEMANTIC_SEGMENTATION_PROMPT

TRAIN_PROMPT = (
    "Generate a 150-class semantic segmentation mask. Use only the fixed "
    "training palette, with one solid class color per pixel and sharp object boundaries."
)

# ADE20K index (0-149) -> Cityscapes trainId. Diagnostic only, not paper protocol.
ADE_TO_CS = {
    0: 3, 1: 2, 2: 10, 4: 8, 6: 0, 9: 8, 11: 1, 12: 11, 13: 9, 17: 8,
    20: 13, 32: 4, 43: 7, 52: 1, 72: 8, 80: 15, 83: 14, 87: 5, 93: 5,
    102: 13, 116: 17, 127: 18, 136: 6,
}


def load_ade_palette(path: Path) -> dict[str, tuple[int, int, int]]:
    raw = json.loads(path.read_text())
    return {name: tuple(rgb) for name, rgb in raw["palette"].items()}


def mean_nn_distance(rgb: torch.Tensor, palette: dict[str, tuple[int, int, int]]) -> float:
    colors = torch.tensor(list(palette.values()), dtype=torch.float32, device=rgb.device) / 255.0
    pixels = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    dist = torch.cdist(pixels, colors).min(dim=-1).values
    return float(dist.mean().item())


def _linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        used_rows, used_cols = set(), set()
        pairs: list[tuple[int, int]] = []
        flat = [((i, j), cost[i, j]) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
        for (i, j), _ in sorted(flat, key=lambda item: item[1]):
            if i in used_rows or j in used_cols:
                continue
            used_rows.add(i)
            used_cols.add(j)
            pairs.append((i, j))
        row = np.array([p[0] for p in pairs], dtype=np.int64)
        col = np.array([p[1] for p in pairs], dtype=np.int64)
        return row, col
    return linear_sum_assignment(cost)


def hungarian_miou(pred: torch.Tensor, gt: torch.Tensor, num_classes: int) -> float:
    pred = pred.reshape(-1).long()
    gt = gt.reshape(-1).long()
    valid = gt != 255
    pred = pred[valid]
    gt = gt[valid]
    confusion = torch.bincount(gt * num_classes + pred, minlength=num_classes * num_classes)
    confusion = confusion[: num_classes * num_classes].reshape(num_classes, num_classes).cpu().numpy()
    cost = confusion.max() - confusion
    row, col = _linear_sum_assignment(cost)
    remap = {int(c): int(r) for r, c in zip(row, col)}
    remapped = torch.tensor(
        [remap.get(int(p), 255) for p in pred.tolist()],
        dtype=torch.long,
        device=gt.device,
    )
    ious = []
    for k in range(num_classes):
        p = remapped == k
        g = gt == k
        inter = (p & g).sum().item()
        union = (p | g).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def map_ade_to_cs(ade_labels: torch.Tensor) -> torch.Tensor:
    mapped = torch.full_like(ade_labels, 255)
    for ade_id, cs_id in ADE_TO_CS.items():
        mapped[ade_labels == ade_id] = cs_id
    return mapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes-root", type=Path, required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--ade-palette", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = CityscapesSegDataset(str(args.cityscapes_root), split="val", resolution=512)
    n = min(args.max_samples, len(dataset))
    loader = DataLoader(Subset(dataset, range(n)), batch_size=1, shuffle=False)
    model = VisionBanana.from_pretrained(
        model_name="sana-1600m",
        lora_path=args.lora_path,
        device="cuda",
        dtype=torch.float32,
    )
    decoder = SegmentationDecoder()
    cs_map = dict(CITYSCAPES_RGB_LEGEND)
    ade_map = load_ade_palette(args.ade_palette)

    cs_conf = torch.zeros(19, 19, dtype=torch.long)
    ade_conf = torch.zeros(19, 19, dtype=torch.long)
    cs_dists, ade_dists = [], []
    hung_ious = []
    previews = []

    for i, batch in enumerate(loader):
        image = batch["image"].to("cuda")
        gt = batch["gt_labels"].to("cuda")
        for prompt_name, prompt in (
            ("cityscapes_legend", SEMANTIC_SEGMENTATION_PROMPT),
            ("train_ade150", TRAIN_PROMPT),
        ):
            out = model.predict(
                image,
                task="semantic",
                prompt=prompt,
                class_color_map=cs_map if prompt_name == "cityscapes_legend" else ade_map,
                num_inference_steps=20,
                strength=0.85,
                seed=42,
            )
            pred_rgb = out["rgb_output"]
            if pred_rgb.dim() == 3:
                pred_rgb = pred_rgb.unsqueeze(0)
            if prompt_name == "cityscapes_legend":
                pred_cs = decoder.decode_semantic(pred_rgb, cs_map)
                metrics = evaluate_semantic(pred_cs, gt, 19)
                pred_flat = pred_cs.reshape(-1).long()
                gt_flat = gt.reshape(-1).long()
                valid = gt_flat != 255
                idx = gt_flat[valid] * 19 + pred_flat[valid]
                cs_conf += torch.bincount(idx.cpu(), minlength=361).reshape(19, 19)
                cs_dists.append(mean_nn_distance(pred_rgb, cs_map))
                hung_ious.append(hungarian_miou(pred_cs, gt, 19))
                if args.preview_dir is not None and i < 4:
                    vis = (pred_rgb[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
                    previews.append((i, vis))
            else:
                pred_ade = decoder.decode_semantic(pred_rgb, ade_map)
                pred_mapped = map_ade_to_cs(pred_ade)
                pred_flat = pred_mapped.reshape(-1).long()
                gt_flat = gt.reshape(-1).long()
                valid = (gt_flat != 255) & (pred_flat != 255)
                idx = gt_flat[valid] * 19 + pred_flat[valid]
                ade_conf += torch.bincount(idx.cpu(), minlength=361).reshape(19, 19)
                ade_dists.append(mean_nn_distance(pred_rgb, ade_map))

    def miou_from_conf(conf: torch.Tensor) -> dict:
        tp = torch.diag(conf).float()
        denom = conf.sum(1).float() + conf.sum(0).float() - tp
        ious = torch.where(denom > 0, tp / denom, torch.nan)
        return {
            "mIoU": float(ious.nanmean().item()),
            "per_class_iou": [None if torch.isnan(v) else float(v) for v in ious],
        }

    result = {
        "n": n,
        "training": {
            "manifest": "SNVC seg, ADE20K 150-class seed=42 palette",
            "prompt": TRAIN_PROMPT,
        },
        "paper_eval_was": {
            "palette": "official Cityscapes 19-class",
            "prompt": "Cityscapes RGB legend",
            "note": "cross-palette; not comparable to paper 0.699",
        },
        "cityscapes_legend_prompt": {
            "cs_nn_decode": miou_from_conf(cs_conf),
            "hungarian_remap_miou_mean": float(np.mean(hung_ious)),
            "mean_nn_dist_to_cs_palette": float(np.mean(cs_dists)),
        },
        "train_ade150_prompt": {
            "ade_decode_then_ade2cs": miou_from_conf(ade_conf),
            "mean_nn_dist_to_ade_palette": float(np.mean(ade_dists)),
            "note": "diagnostic mapping only; not paper Cityscapes mIoU",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2))
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        for idx, vis in previews:
            Image.fromarray(vis).save(args.preview_dir / f"pred_{idx:02d}.png")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
