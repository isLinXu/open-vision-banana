#!/usr/bin/env python3
"""Generate a reusable base-model reference set from GenAI-Bench prompts.

The prompt-only benchmark cannot be passed directly to the paired-image proxy.
This command creates ``images/`` and ``metadata.jsonl`` for that proxy. The
result measures drift from the base model, not the paper's human win rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_prompts(root: Path, max_samples: int | None = None) -> list[str]:
    path = root / "prompts.jsonl"
    if not path.is_file():
        raise ValueError(f"GenAI-Bench prompts not found: {path}")
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing prompt at {path}:{number}")
            prompts.append(prompt.strip())
            if max_samples is not None and len(prompts) >= max_samples:
                break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def reference_spec(
    prompts: list[str], *, model_name: str, dtype: str, resolution: int, steps: int, seed: int,
    guidance_scale: float,
) -> dict:
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "model_name": model_name,
        "dtype": dtype,
        "resolution": resolution,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "samples": len(prompts),
        "prompts_sha256": hashlib.sha256(payload).hexdigest(),
        "evaluation_mode": "base_generation_reference",
        "formal_genai_bench": False,
    }


def build_reference(
    *, prompts_root: Path, output_dir: Path, model_name: str, dtype: str,
    resolution: int, steps: int, seed: int, guidance_scale: float = 4.0,
    max_samples: int | None = None,
    device: str | None = None,
) -> dict:
    if resolution < 1 or steps < 1 or max_samples is not None and max_samples < 1:
        raise ValueError("resolution, steps, and max_samples must be positive")
    prompts = load_prompts(prompts_root, max_samples)
    spec = reference_spec(
        prompts, model_name=model_name, dtype=dtype, resolution=resolution, steps=steps,
        seed=seed, guidance_scale=guidance_scale,
    )
    spec_path = output_dir / "reference_config.json"
    images_dir = output_dir / "images"
    manifest_path = output_dir / "metadata.jsonl"
    if spec_path.exists():
        saved = json.loads(spec_path.read_text(encoding="utf-8"))
        if saved != spec:
            raise ValueError(f"Reference parameters changed; use a new output directory: {output_dir}")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Unverified reference files exist in {output_dir}; use a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = [{"image": f"{index:06d}.png", "prompt": prompt} for index, prompt in enumerate(prompts)]
    missing = [row for row in rows if not (images_dir / row["image"]).is_file()]
    if missing:
        from open_vision_banana.inference.generator import BaseGenerator

        runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if runtime_device == "mps" and dtype != "float32":
            raise ValueError("MPS reference generation requires float32")
        model = BaseGenerator(
            model_name=model_name,
            device=runtime_device,
            dtype={"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype],
        )
        for index, row in enumerate(rows):
            image_path = images_dir / row["image"]
            if image_path.is_file():
                continue
            generated = model.generate(
                prompt=row["prompt"], image=None, num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=resolution, width=resolution, seed=seed + index,
            )
            if generated.shape != (1, 3, resolution, resolution) or not torch.isfinite(generated).all():
                raise ValueError(f"Invalid base generation for prompt {index}: {tuple(generated.shape)}")
            rgb = (generated[0].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb, mode="RGB").save(image_path)
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    spec = build_reference(
        prompts_root=args.prompts_root, output_dir=args.output_dir, model_name=args.model_name,
        dtype=args.dtype, resolution=args.resolution, steps=args.num_inference_steps,
        seed=args.seed, guidance_scale=args.guidance_scale,
        max_samples=args.max_samples, device=args.device,
    )
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
