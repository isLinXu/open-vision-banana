#!/usr/bin/env python3
"""Require a complete online-weight adapter and matching evaluation provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_checkpoint(
    checkpoint: Path, eval_json: Path | None = None, *, expected_steps: int | None = None,
    expected_guidance: float | None = None, expected_seed: int | None = None,
    expected_dtype: str | None = None,
) -> None:
    metadata_path = checkpoint / "metadata.json"
    adapter_path = checkpoint / "lora_adapter"
    if not metadata_path.is_file() or not adapter_path.is_dir() or not any(adapter_path.iterdir()):
        raise ValueError(f"Incomplete evaluation checkpoint: {checkpoint}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("adapter_weight_source") != "online":
        raise ValueError(f"Expected online adapter at {checkpoint}, got {metadata.get('adapter_weight_source')!r}")
    if eval_json is not None:
        report = json.loads(eval_json.read_text(encoding="utf-8"))
        report_meta = report.get("metadata", {})
        recorded = report_meta.get("lora_path")
        if not isinstance(recorded, str) or Path(recorded).resolve() != checkpoint.resolve():
            raise ValueError(f"{eval_json} used {recorded!r}, expected {checkpoint}")
        protocol = report_meta.get("inference_protocol", report_meta)
        expected = {
            "num_inference_steps": expected_steps,
            "guidance_scale": expected_guidance,
            "seed": expected_seed,
        }
        for key, value in expected.items():
            if value is not None and protocol.get(key) != value:
                raise ValueError(f"{eval_json} has {key}={protocol.get(key)!r}, expected {value!r}")
        if expected_dtype is not None and report_meta.get("dtype") != expected_dtype:
            raise ValueError(f"{eval_json} has dtype={report_meta.get('dtype')!r}, expected {expected_dtype!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path)
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-guidance", type=float)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-dtype")
    args = parser.parse_args()
    check_checkpoint(
        args.checkpoint, args.eval_json, expected_steps=args.expected_steps,
        expected_guidance=args.expected_guidance, expected_seed=args.expected_seed,
        expected_dtype=args.expected_dtype,
    )
    print(f"online adapter verified: {args.checkpoint}")


if __name__ == "__main__":
    main()
