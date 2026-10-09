#!/usr/bin/env python3
"""Score trivial depth predictors under the paper-benchmark protocol.

Run this next to ``evaluate_paper_benchmarks.py`` to put a model's delta_1.25 /
AbsRel in context. Example (NYU val, with a synthetic-data prior map):

    python scripts/depth_trivial_baselines.py \
        --dataset-key nyu_depth --dataset-name nyu --root /data/nyu \
        --prior-key hypersim_depth --prior-root data/hypersim_depth_512 \
        --output-json output/eval/nyu_trivial_baselines.json

The prior set must NOT be an evaluation benchmark; use the same synthetic data
the model is trained on to get a like-for-like "dataset prior" reference.
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
from open_vision_banana.evaluation.trivial_baselines import (  # noqa: E402
    evaluate_trivial_baselines,
    mean_depth_map,
)


def _load(
    key: str,
    root: Path,
    split: str,
    resolution: int,
    limit: int | None,
    batch_size: int,
    **dataset_kwargs,
) -> list[dict]:
    dataset = build_dataset(key, str(root), split=split, resolution=resolution, **dataset_kwargs)
    if len(dataset) == 0:
        raise RuntimeError(f"Empty dataset: {key} at {root} (split={split})")
    if limit is not None and len(dataset) > limit:
        dataset = Subset(dataset, range(limit))
    return list(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-key", required=True, help="build_dataset key, e.g. nyu_depth")
    p.add_argument("--dataset-name", required=True, help="protocol name: nyu | kitti | eth3d | other")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--native-resolution",
        action="store_true",
        help="Score at the original GT resolution, matching evaluate_paper_benchmarks.py --native-resolution.",
    )
    p.add_argument(
        "--dataset-kwargs",
        default="{}",
        help='JSON of extra adapter kwargs, e.g. \'{"scene_type": "indoors"}\' for DIODE-Indoor.',
    )
    p.add_argument("--constants", type=float, nargs="*", default=[1.0, 2.0, 3.0, 5.0, 10.0])
    p.add_argument("--prior-key", default=None)
    p.add_argument("--prior-root", type=Path, default=None)
    p.add_argument("--prior-split", default="train")
    p.add_argument("--prior-max-samples", type=int, default=2000)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    extra = json.loads(args.dataset_kwargs)
    if args.native_resolution:
        extra["native_gt"] = True
    batch_size = 1 if args.native_resolution else args.batch_size  # native sizes can differ per image
    batches = _load(
        args.dataset_key, args.root, args.split, args.resolution, args.max_samples, batch_size, **extra
    )
    if args.native_resolution:
        for batch in batches:
            batch["gt_depth"] = batch["gt_depth_native"]
            batch["valid_mask"] = batch["valid_mask_native"]

    prior_map = None
    if args.prior_key and args.prior_root:
        prior_batches = _load(
            args.prior_key, args.prior_root, args.prior_split, args.resolution,
            args.prior_max_samples, args.batch_size,
        )
        prior_map = mean_depth_map(prior_batches, (args.resolution, args.resolution))

    results = evaluate_trivial_baselines(
        batches, dataset_name=args.dataset_name, constants=args.constants, prior_map=prior_map
    )
    payload = {
        "dataset_key": args.dataset_key,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "native_resolution": args.native_resolution,
        "dataset_kwargs": extra,
        "samples": sum(int(b["gt_depth"].shape[0]) for b in batches),
        "note": "oracle_* baselines use evaluation GT and are upper bounds for their predictor class.",
        "baselines": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'baseline':<22} {'delta1':>8} {'AbsRel':>8} {'RMSE':>8}")
    for name, m in results.items():
        print(f"{name:<22} {m['delta_1.25']:>8.4f} {m['AbsRel']:>8.4f} {m['RMSE']:>8.4f}")
    return 0


if __name__ == "__main__":
    torch.manual_seed(0)
    raise SystemExit(main())
