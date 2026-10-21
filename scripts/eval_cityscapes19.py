#!/usr/bin/env python3
"""Official Cityscapes val mIoU for the 19-color specialist."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.datasets import CityscapesSegDataset
from open_vision_banana.evaluation.evaluator import VisionEvaluator
from open_vision_banana.inference.pipeline import VisionBanana
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cityscapes-root", type=Path, required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = CityscapesSegDataset(str(args.cityscapes_root), split="val", resolution=512)
    if len(dataset) != 500:
        raise SystemExit(f"expected 500 official val pairs, got {len(dataset)}")
    model = VisionBanana.from_pretrained(
        model_name="sana-1600m",
        lora_path=args.lora_path,
        device="cuda",
        dtype=torch.float32,
    )
    recorded = (model.training_prompts or {}).get("semantic", "")
    if recorded and "150-class" in recorded:
        raise SystemExit("ADE-150 LoRA is not paper-protocol Cityscapes")
    evaluator = VisionEvaluator(model, device="cuda")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics = evaluator.evaluate(
        loader,
        task="semantic",
        num_classes=19,
        class_color_map=dict(CityscapesSegDataset.CITYSCAPES_COLORS),
        num_inference_steps=20,
        strength=0.85,
        seed=42,
        dataset_name="cityscapes",
    )
    payload = {
        "model_name": "sana-1600m",
        "lora_path": args.lora_path,
        "samples": 500,
        "prompt": SEMANTIC_SEGMENTATION_PROMPT,
        "palette": "cityscapes_19_official",
        "decoded_metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
