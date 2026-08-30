#!/usr/bin/env python3
"""Run a reproducible offline generation-preservation health evaluation.

This command evaluates generated images against the reference image
distribution supplied by ``GenerationDataset``.  It is deliberately reported
as a proxy and must not be cited as the GenAI-Bench win rate.  Failed samples
are excluded from metric computation and written to a JSONL sidecar so a run
cannot silently hide model/runtime failures.

Example (requires local model weights and a local generation dataset)::

    python scripts/evaluate_generation_preservation.py \
        --model-name black-forest-labs/FLUX.1-schnell \
        --dataset-root data/vb_generation_5k \
        --output-json reports/generation_health.json \
        --max-samples 32
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data import build_generation_dataset  # noqa: E402
from open_vision_banana.data.manifest import fingerprint_records  # noqa: E402
from open_vision_banana.evaluation.generation_preservation import (  # noqa: E402
    evaluate_generation_preservation,
    extract_inception_features,
    extract_rgb_features,
)
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402

logger = logging.getLogger("generation_preservation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--failure-jsonl", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Keep a proxy report despite failed samples (never use for an acceptance gate).",
    )
    parser.add_argument(
        "--feature-backend",
        choices=["rgb_proxy", "inception"],
        default="rgb_proxy",
        help="rgb_proxy is offline and deterministic; inception requires torchvision weights.",
    )
    return parser.parse_args()


def resolve_device(value: str | None) -> str:
    if value:
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(value: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def resolve_runtime_dtype(value: str, device: str) -> torch.dtype:
    """Reject unstable reduced-precision generation on Apple MPS."""
    dtype = resolve_dtype(value)
    if device == "mps" and dtype != torch.float32:
        raise ValueError(
            "Generation replay on MPS requires --dtype float32; fp16/bf16 can "
            "produce non-finite diffusion samples."
        )
    return dtype


def dataset_fingerprint(dataset: Any) -> str:
    root = Path(getattr(dataset, "root_dir", "."))
    root_abs = root.absolute()
    records = []
    for sample in getattr(dataset, "samples", []):
        path = Path(sample["image_path"])
        # Keep the manifest path lexical: ``generation/images`` is allowed to
        # contain symlinks to a separately stored image corpus.  Resolving the
        # symlink before ``relative_to`` makes an otherwise valid portable
        # manifest look external and aborts report generation.
        image_rel = Path(os.path.relpath(path.absolute(), root_abs)).as_posix()
        records.append(
            {
                "task": "generation",
                "image": image_rel,
                "prompt": sample.get("prompt", ""),
            }
        )
    return fingerprint_records(records, root)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_failure(handle, *, index: int, prompt: str, exc: Exception, elapsed_ms: float) -> None:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "index": index,
        "prompt": prompt,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "latency_ms": elapsed_ms,
    }
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.num_inference_steps < 1:
        raise SystemExit("--batch-size and --num-inference-steps must be positive")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    dataset = build_generation_dataset(
        str(args.dataset_root),
        resolution=args.resolution,
        require_captions=True,
    )
    if not len(dataset):
        raise SystemExit(f"No generation samples found under {args.dataset_root}")
    selected = dataset if args.max_samples is None else Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if device == "cuda" else 0,
    )

    runtime_dtype = resolve_runtime_dtype(args.dtype, device)
    model = BaseGenerator(
        model_name=args.model_name,
        lora_path=args.lora_path,
        device=device,
        dtype=runtime_dtype,
    )
    failure_path = args.failure_jsonl or args.output_json.with_suffix(".failures.jsonl")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    real_images: list[torch.Tensor] = []
    generated_images: list[torch.Tensor] = []
    failures = 0
    started = time.perf_counter()

    with failure_path.open("w", encoding="utf-8") as failure_handle:
        sample_index = 0
        for batch in loader:
            prompts = batch["prompt"]
            target_rgb = batch["target_rgb"].float()
            for offset, prompt in enumerate(prompts):
                index = sample_index + offset
                sample_started = time.perf_counter()
                try:
                    generated = model.generate(
                        prompt=str(prompt),
                        image=None,
                        num_inference_steps=args.num_inference_steps,
                        height=args.resolution,
                        width=args.resolution,
                        seed=args.seed + index,
                    )
                    if generated.dim() != 4 or generated.shape[0] != 1 or generated.shape[1] != 3:
                        raise ValueError(f"unexpected generated shape: {tuple(generated.shape)}")
                    if not torch.isfinite(generated).all():
                        raise ValueError("generated image contains NaN or Inf")
                    real_images.append(target_rgb[offset : offset + 1].cpu())
                    generated_images.append(generated.cpu().clamp(0, 1))
                except Exception as exc:  # keep evaluating independent samples
                    failures += 1
                    _write_failure(
                        failure_handle,
                        index=index,
                        prompt=str(prompt),
                        exc=exc,
                        elapsed_ms=(time.perf_counter() - sample_started) * 1000.0,
                    )
                    logger.warning("sample %d failed: %s", index, exc)
            sample_index += len(prompts)

    if failures and not args.allow_failures:
        raise RuntimeError(
            f"{failures} generation samples failed; refusing an incomplete acceptance report. "
            f"See {failure_path} or pass --allow-failures for a non-gating proxy."
        )

    if not generated_images:
        raise RuntimeError(f"All {failures} generation samples failed; see {failure_path}")

    real_rgb = torch.cat(real_images)
    gen_rgb = torch.cat(generated_images)
    if args.feature_backend == "rgb_proxy":
        feature_extractor = extract_rgb_features
    else:
        feature_extractor = extract_inception_features
    metrics = evaluate_generation_preservation(
        real_rgb=real_rgb,
        gen_rgb=gen_rgb,
        pred_rgb=gen_rgb,
        target_rgb=real_rgb,
        feature_extractor=feature_extractor,
    )
    elapsed = time.perf_counter() - started
    report = {
        "metadata": {
            "evaluation_mode": "generation_preservation_proxy",
            "formal_genai_bench": False,
            "metric_scope": "reference_distribution_and_pixel_perceptual_proxy",
            "feature_backend": args.feature_backend,
            "model_name": args.model_name,
            "lora_path": args.lora_path,
            "dataset_root": str(args.dataset_root),
            "dataset_fingerprint": dataset_fingerprint(dataset),
            "device": device,
            "dtype": str(runtime_dtype).replace("torch.", ""),
            "resolution": args.resolution,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "requested_samples": len(selected),
            "successful_samples": len(generated_images),
            "failed_samples": failures,
            "failure_jsonl": str(failure_path),
            "elapsed_seconds": elapsed,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "metrics": _json_safe(metrics),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("generation health report written to %s", args.output_json)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
