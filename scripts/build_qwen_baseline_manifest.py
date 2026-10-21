#!/usr/bin/env python3
"""Build a paper-aligned, de-biased multi-task manifest for the Qwen-Image baseline.

Solves the two data bottlenecks identified in the SNVC shared-baseline ablation:
1. Depth supervision quality: replaces or filters out SNVC relative/grayscale
   pseudo-depth targets, keeping only valid ``metric_power`` + ``cube_edge``
   synthetic metric depth targets (e.g. Hypersim / Virtual KITTI 2) and unifying
   their prompt to ``PromptBuilder.DEPTH_PROMPT``.
2. Semantic class/domain balance: Cityscapes was only 311 / 4460 (6.97%) of
   semantic train rows in the raw SNVC pack. This builder either merges an
   official Cityscapes-19 manifest or deterministically upsamples Cityscapes
   train rows to reach ``--min-cityscapes-share`` (default 0.50), while keeping
   validation splits strictly un-duplicated.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.data.manifest import resolve_manifest_path  # noqa: E402
from open_vision_banana.inference.prompt import PromptBuilder  # noqa: E402
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT  # noqa: E402


TASK_ALIASES = {
    "segmentation": "semantic",
    "seg": "semantic",
    "metric_depth": "depth",
}


def is_valid_metric_cube_edge_target(
    target_path: Path,
    mapper: DepthRGBMapper,
    *,
    max_mae: float = 2.0 / 255.0,
    max_grayscale_fraction: float = 0.90,
) -> bool:
    """Return True if ``target_path`` is a valid non-grayscale cube-edge depth target."""
    if not target_path.is_file():
        return False
    try:
        with Image.open(target_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except OSError:
        return False
    rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    channel_span = (rgb[:, 0] - rgb[:, 1]).abs() + (rgb[:, 1] - rgb[:, 2]).abs()
    gray_frac = float((channel_span < 1.0 / 255.0).float().mean().item())
    if gray_frac > max_grayscale_fraction:
        return False
    decoded = mapper.rgb_to_depth(rgb)
    if float(decoded.max().item()) <= 0.0:
        return False
    recon = mapper.depth_to_rgb(decoded)
    mae = float((recon - rgb).abs().mean().item())
    return mae <= max_mae


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_qwen_baseline_manifest(
    base_manifest: Path,
    output_manifest: Path,
    *,
    data_root: Path | None = None,
    metric_depth_manifest: Path | None = None,
    cityscapes_manifest: Path | None = None,
    min_cityscapes_share: float = 0.50,
    filter_non_metric_depth: bool = True,
    seed: int = 42,
) -> dict:
    """Build a cleaned, rebalanced manifest and return an audit summary."""
    if not 0.0 <= min_cityscapes_share < 1.0:
        raise ValueError("min_cityscapes_share must be in [0, 1)")

    rng = random.Random(seed)
    anchor = data_root.resolve() if data_root else base_manifest.resolve()
    mapper = DepthRGBMapper(colormap="cube_edge", curve="metric_power")

    base_rows = load_jsonl(base_manifest)
    kept_rows: list[dict] = []
    rejected_depth = 0

    for row in base_rows:
        raw_task = row.get("task", "")
        task = TASK_ALIASES.get(raw_task, raw_task)
        row = dict(row, task=task)

        if task == "depth":
            if metric_depth_manifest is not None:
                continue
            if filter_non_metric_depth:
                tgt_path = resolve_manifest_path(row.get("target", ""), anchor)
                if not is_valid_metric_cube_edge_target(tgt_path, mapper):
                    rejected_depth += 1
                    continue
            row["prompt"] = PromptBuilder.DEPTH_PROMPT

        kept_rows.append(row)

    if metric_depth_manifest is not None:
        for row in load_jsonl(metric_depth_manifest):
            row = dict(row, task="depth", prompt=PromptBuilder.DEPTH_PROMPT)
            kept_rows.append(row)

    if cityscapes_manifest is not None:
        for row in load_jsonl(cityscapes_manifest):
            row = dict(row, task="semantic", prompt=SEMANTIC_SEGMENTATION_PROMPT)
            kept_rows.append(row)

    # Rebalance Cityscapes train share within semantic
    cs_train = [
        r for r in kept_rows
        if r.get("task") == "semantic"
        and r.get("split") == "train"
        and r.get("prompt") == SEMANTIC_SEGMENTATION_PROMPT
    ]
    other_sem_train = [
        r for r in kept_rows
        if r.get("task") == "semantic"
        and r.get("split") == "train"
        and r.get("prompt") != SEMANTIC_SEGMENTATION_PROMPT
    ]
    upsampled_cs_added = 0
    if cs_train and other_sem_train and min_cityscapes_share > 0.0:
        target_cs_count = int(
            round(min_cityscapes_share * len(other_sem_train) / (1.0 - min_cityscapes_share))
        )
        if len(cs_train) < target_cs_count:
            needed = target_cs_count - len(cs_train)
            full_copies, remainder = divmod(needed, len(cs_train))
            extra = cs_train * full_copies + rng.sample(cs_train, remainder)
            kept_rows.extend(dict(r) for r in extra)
            upsampled_cs_added = len(extra)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sem_train_total = len(cs_train) + upsampled_cs_added + len(other_sem_train)
    final_cs_train = len(cs_train) + upsampled_cs_added
    task_split_counts: dict[str, int] = {}
    for row in kept_rows:
        key = f"{row.get('task')}/{row.get('split')}"
        task_split_counts[key] = task_split_counts.get(key, 0) + 1

    audit = {
        "output_manifest": str(output_manifest.resolve()),
        "total_rows": len(kept_rows),
        "task_split_counts": task_split_counts,
        "rejected_non_metric_depth_rows": rejected_depth,
        "cityscapes_train_initial": len(cs_train),
        "cityscapes_train_upsampled_added": upsampled_cs_added,
        "cityscapes_train_final": final_cs_train,
        "cityscapes_train_share": (
            final_cs_train / sem_train_total if sem_train_total > 0 else 0.0
        ),
    }
    output_manifest.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--metric-depth-manifest", type=Path, default=None)
    parser.add_argument("--cityscapes-manifest", type=Path, default=None)
    parser.add_argument("--min-cityscapes-share", type=float, default=0.50)
    parser.add_argument("--no-filter-non-metric-depth", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_qwen_baseline_manifest(
        args.base_manifest,
        args.output,
        data_root=args.data_root,
        metric_depth_manifest=args.metric_depth_manifest,
        cityscapes_manifest=args.cityscapes_manifest,
        min_cityscapes_share=args.min_cityscapes_share,
        filter_non_metric_depth=not args.no_filter_non_metric_depth,
        seed=args.seed,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
