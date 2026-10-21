#!/usr/bin/env python3
"""RefCOCOg UMD val cIoU. Avoids deploy-incompatible evaluate() kwargs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.datasets import RefCOCOgDataset
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.inference.pipeline import VisionBanana


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refcocog-root", type=Path, required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--model-name", default="sana-1600m")
    parser.add_argument("--quantize", choices=["4bit"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--num-inference-steps", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = RefCOCOgDataset(
        str(args.refcocog_root),
        split="val",
        resolution=512,
        native_gt=True,
        protocol="umd",
    )
    if len(dataset) == 0:
        raise SystemExit(f"empty RefCOCOg val at {args.refcocog_root}")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = VisionBanana.from_pretrained(
        model_name=args.model_name,
        lora_path=args.lora_path,
        device="cuda",
        dtype=dtype,
        quantize=args.quantize,
    )
    evaluator = VisionEvaluator(model, device="cuda")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics = evaluator.evaluate(
        loader,
        task="referring",
        num_inference_steps=args.num_inference_steps,
        strength=0.85,
        seed=42,
        dataset_name="refcocog",
        native_resolution=True,
    )
    payload = {
        "model_name": args.model_name,
        "lora_path": args.lora_path,
        "samples": len(dataset),
        "protocol": "umd",
        "native_resolution": True,
        "quantize": args.quantize,
        "decoded_metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
