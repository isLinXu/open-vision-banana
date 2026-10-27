#!/usr/bin/env python3
"""Load every supplement row with the proposed training augmentations, without a model."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from open_vision_banana.training.trainer import ManifestVisionDataset, _collate_fn


def validate(manifest: Path, output: Path, release: Path, *, workers: int = 16) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    dataset = ManifestVisionDataset(
        manifest,
        512,
        "train",
        {"depth", "referring"},
        aspect_preserving_crop=True,
        depth_colormaps=["cube_edge", "plasma", "inferno", "viridis", "grayscale"],
        seed=42,
    )
    increment = json.loads(release.read_text(encoding="utf-8"))["increment"]
    expected = {task: increment[task] for task in ("depth", "referring")}
    if len(dataset) != sum(expected.values()):
        raise ValueError(f"Expected {sum(expected.values())} supplement rows, got {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=16, num_workers=workers, collate_fn=_collate_fn, shuffle=False
    )
    counts: Counter[str] = Counter()
    empty = 0
    started = time.monotonic()
    for index, batch in enumerate(loader):
        image, target = batch["image"], batch["target_rgb"]
        if image.shape != target.shape or list(image.shape[1:]) != [3, 512, 512]:
            raise ValueError("Unexpected input/target dimensions")
        if image.dtype != torch.float32 or target.dtype != torch.float32:
            raise ValueError("Unexpected input/target dtype")
        if not torch.isfinite(image).all() or not torch.isfinite(target).all():
            raise ValueError("Nonfinite training tensor")
        if image.min() < -1 or image.max() > 1 or target.min() < 0 or target.max() > 1:
            raise ValueError("Input must be [-1,1]; target must be [0,1]")
        counts.update(batch["task"])
        for offset, task in enumerate(batch["task"]):
            if task == "referring":
                mask = target[offset]
                if not ((mask == 0) | (mask == 1)).all():
                    raise ValueError("Nonbinary referring target")
                empty += int(not mask.any())
        if index % 128 == 0:
            print(json.dumps({"loaded": sum(counts.values())}), flush=True)
    if counts != expected:
        raise ValueError(f"Unexpected task counts: {counts}")
    if empty:
        raise ValueError(f"Filtered supplement still produces {empty} empty referring crops")
    report = {
        "status": "passed",
        "rows": sum(counts.values()),
        "task_counts": dict(counts),
        "resolution": 512,
        "workers": workers,
        "seed": 42,
        "aspect_preserving_crop": True,
        "depth_colormaps": dataset.depth_colormaps,
        "empty_referring_masks_after_configured_random_crop": empty,
        "nonfinite_values": 0,
        "input_range": [-1, 1],
        "target_range": [0, 1],
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Full supplement loading/collation and augmentation; CPU, no model or accuracy evaluation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--release", type=Path, default=Path("docs/main_depth_reasoning_release.json")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.manifest, args.output, args.release, workers=args.workers), indent=2
        )
    )
