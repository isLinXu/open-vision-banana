#!/usr/bin/env python3
"""Evaluate a downloaded RES bundle with native GT; --preflight loads no model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_vision_banana.data.res_dataset import RESManifestDataset  # noqa: E402
from open_vision_banana.referring_contract import BINARY_REFERRING_ENCODING  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", default="pairs.jsonl")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--source-dataset")
    parser.add_argument("--source-protocol")
    parser.add_argument("--source-split", choices=("val", "test", "testA", "testB"))
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate all selected images/masks and write counts without loading weights",
    )
    parser.add_argument("--model-name")
    parser.add_argument("--lora-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--quantize", choices=("4bit",))
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Score only the first N selected rows; all selected assets are still validated",
    )
    parser.add_argument("--output-json", type=Path, default=Path("res_metrics.json"))
    args = parser.parse_args(argv)
    if not args.preflight and not args.model_name:
        parser.error("--model-name is required unless --preflight is selected")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.num_inference_steps <= 0 or not 0 < args.strength <= 1:
        parser.error("Inference steps must be positive and strength must be in (0,1]")
    dataset = RESManifestDataset(
        args.dataset_root,
        split=args.split,
        manifest=args.manifest,
        source_dataset=args.source_dataset,
        source_protocol=args.source_protocol,
        source_split=args.source_split,
        resolution=args.resolution,
    )
    counts = dataset.validate_assets()
    results = {
        "status": "preflight_passed",
        "manifest": args.manifest,
        "manifest_sha256": dataset.manifest_sha256,
        "filters": dataset.filters,
        "counts": counts,
        "protocol": {
            "task": "referring",
            "encoding": BINARY_REFERRING_ENCODING,
            "native_resolution": True,
            "batch_size": 1,
            "ground_truth": "native lossless binary PNG; one sample per expression",
            "source_splits": "official val/test/testA/testB retained; no train rows",
        },
    }
    if not args.preflight:
        from open_vision_banana.evaluation.evaluator import VisionEvaluator
        from open_vision_banana.inference.pipeline import VisionBanana

        device = args.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        model = VisionBanana.from_pretrained(
            model_name=args.model_name,
            lora_path=args.lora_path,
            device=device,
            dtype=getattr(torch, args.dtype),
            enable_sequential_cpu_offload=args.cpu_offload,
            quantize=args.quantize,
        )
        sample_count = min(len(dataset), args.max_samples or len(dataset))
        loader = DataLoader(
            Subset(dataset, range(sample_count)), batch_size=1, shuffle=False, num_workers=0
        )
        results["inference"] = {
            "model_name": args.model_name,
            "lora_path": args.lora_path,
            "resolution": args.resolution,
            "device": device,
            "dtype": args.dtype,
            "num_inference_steps": args.num_inference_steps,
            "strength": args.strength,
            "seed": args.seed,
            "evaluated_samples": sample_count,
        }
        results["metrics"] = VisionEvaluator(model, device=device).evaluate(
            loader,
            task="referring",
            native_resolution=True,
            referring_encoding=BINARY_REFERRING_ENCODING,
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            seed=args.seed,
        )
        results["status"] = "evaluated"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": results["status"],
                "samples": counts["samples"],
                "output": str(args.output_json),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
