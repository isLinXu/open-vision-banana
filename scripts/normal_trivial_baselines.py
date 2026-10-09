#!/usr/bin/env python3
"""Score trivial surface-normal predictors with the paper-benchmark accumulator.

    python scripts/normal_trivial_baselines.py \
        --dataset-key nyu_normal --root /data/nyu_normals \
        --output-json output/eval/nyu_normal_trivial.json

    python scripts/normal_trivial_baselines.py \
        --dataset-key diode_normal --root /data/diode \
        --dataset-kwargs '{"scene_type": "indoors"}' \
        --native-resolution \
        --output-json output/eval/diode_indoor_normal_trivial.json

NYU, ScanNet and VKITTI need a community normal directory. This script does
not derive normals from depth. ``oracle_*`` rows use the evaluation ground
truth and are upper bounds, not zero-shot methods.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data import build_dataset  # noqa: E402
from open_vision_banana.evaluation.normal_baselines import (  # noqa: E402
    evaluate_normal_baselines,
    mean_direction,
)


def _load(key, root, split, resolution, limit, batch_size, **dataset_kwargs):
    dataset = build_dataset(key, str(root), split=split, resolution=resolution, **dataset_kwargs)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset: {key} at {root} (split={split})")
    if limit is not None and len(dataset) > limit:
        dataset = Subset(dataset, range(limit))
    return list(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-key", required=True, help="nyu_normal | diode_normal | normal | vkitti_normal")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--native-resolution", action="store_true")
    p.add_argument("--dataset-kwargs", default="{}")
    p.add_argument("--prior-key", default=None)
    p.add_argument("--prior-root", type=Path, default=None)
    p.add_argument("--prior-split", default="train")
    p.add_argument("--prior-max-samples", type=int, default=2000)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    extra = json.loads(args.dataset_kwargs)
    if args.native_resolution:
        extra["native_gt"] = True
    batch_size = 1 if args.native_resolution else args.batch_size
    batches = _load(args.dataset_key, args.root, args.split, args.resolution, args.max_samples, batch_size, **extra)
    if args.native_resolution:
        for batch in batches:
            batch["gt_normal"] = batch["gt_normal_native"]
            batch["valid_mask"] = batch["valid_mask_native"]

    prior_direction = None
    if args.prior_key and args.prior_root:
        prior_batches = _load(
            args.prior_key,
            args.prior_root,
            args.prior_split,
            args.resolution,
            args.prior_max_samples,
            args.batch_size,
        )
        prior_direction = mean_direction(prior_batches)

    results = evaluate_normal_baselines(batches, prior_direction=prior_direction)
    payload = {
        "dataset_key": args.dataset_key,
        "split": args.split,
        "native_resolution": args.native_resolution,
        "dataset_kwargs": extra,
        "samples": sum(int(b["gt_normal"].shape[0]) for b in batches),
        "note": (
            "oracle_* baselines use evaluation GT. NYU/ScanNet/VKITTI normals are "
            "community maps; this script does not derive normals from depth."
        ),
        "baselines": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'baseline':<24} {'mean°':>8} {'median°':>8}")
    for name, metrics in results.items():
        print(
            f"{name:<24} {metrics['Mean Angular Error (°)']:>8.3f} "
            f"{metrics['Median Angular Error (°)']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    torch.manual_seed(0)
    raise SystemExit(main())
