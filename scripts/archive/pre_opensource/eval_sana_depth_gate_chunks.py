#!/usr/bin/env python3
"""Run stable chunked depth evaluation for a SANA gate checkpoint on MPS.

Each chunk is evaluated in a fresh Python process.  This avoids a macOS MPS
Metal command-buffer stall observed after many consecutive SANA img2img runs.
The resulting ``metrics.json`` preserves the same per-sample metric contract as
``eval_sana_unified.py`` while recording each child result for auditability.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = PROJECT_ROOT / "scripts" / "eval_sana_unified.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--timeout-sec", type=int, default=300)
    return parser.parse_args()


def mean_numeric(samples: list[dict]) -> dict[str, float]:
    numeric_keys = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if isinstance(value, (int, float)) and key != "index"
        }
    )
    return {
        key: float(np.mean([sample[key] for sample in samples if key in sample]))
        for key in numeric_keys
    }


def main() -> None:
    args = parse_args()
    if args.n_samples < 1 or args.chunk_size < 1:
        raise SystemExit("--n-samples and --chunk-size must be positive")
    if not args.pair_manifest.is_file():
        raise SystemExit(f"manifest not found: {args.pair_manifest}")
    if not args.checkpoint.is_dir():
        raise SystemExit(f"adapter directory not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    all_samples: list[dict] = []
    chunk_reports: list[dict] = []
    started = time.monotonic()

    for start in range(0, args.n_samples, args.chunk_size):
        count = min(args.chunk_size, args.n_samples - start)
        chunk_dir = chunks_dir / f"{args.split}_{start:03d}_{start + count - 1:03d}"
        command = [
            sys.executable,
            str(EVALUATOR),
            "--data-dir", str(args.data_dir),
            "--pair-manifest", str(args.pair_manifest),
            "--checkpoint", str(args.checkpoint),
            "--output-dir", str(chunk_dir),
            "--split", args.split,
            "--preserve-manifest-order",
            "--tasks", "depth",
            "--n-per-task", str(count),
            "--start-index", str(start),
            "--batch-size", "1",
            "--resolution", str(args.resolution),
            "--num-inference-steps", str(args.num_inference_steps),
            "--guidance-scale", str(args.guidance_scale),
            "--seed", str(args.seed),
            "--device", args.device,
            "--dtype", args.dtype,
            "--no-save-panels",
        ]
        print(f"[{args.split}] evaluating rows {start}:{start + count}", flush=True)
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True, timeout=args.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(f"chunk {start}:{start + count} exceeded {args.timeout_sec}s") from exc

        report_path = chunk_dir / "metrics.json"
        if not report_path.is_file():
            raise SystemExit(f"chunk {start}:{start + count} did not write {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        samples = report.get("samples", {}).get("depth", [])
        if len(samples) != count:
            raise SystemExit(f"chunk {start}:{start + count} returned {len(samples)} samples")
        all_samples.extend(samples)
        chunk_reports.append(
            {
                "start_index": start,
                "count": count,
                "metrics": report.get("per_task", {}).get("depth", {}),
                "path": str(report_path.resolve()),
            }
        )

    aggregate = mean_numeric(all_samples)
    aggregate["count"] = len(all_samples)
    result = {
        "model": "SANA checkpoint",
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.pair_manifest.resolve()),
        "split": args.split,
        "selection": "manifest order, chunked into fresh MPS processes",
        "config": {
            "n_samples": args.n_samples,
            "chunk_size": args.chunk_size,
            "resolution": args.resolution,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
        },
        "per_task": {"depth": aggregate},
        "samples": {"depth": all_samples},
        "chunks": chunk_reports,
        "elapsed_sec": time.monotonic() - started,
    }
    result_path = args.output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"per_task": result["per_task"], "metrics": str(result_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
