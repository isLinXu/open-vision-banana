#!/usr/bin/env python3
"""Exact weighted merge of PEFT LoRA adapters: delta_W = sum_i w_i * s_i * B_i @ A_i.

Averaging the A and B factors separately does not average delta_W, because B @ A is
bilinear. This script concatenates the factors instead, so the merged adapter has rank
sum(r_i) and reproduces sum_i w_i * delta_W_i exactly:

    A' = [A_1; A_2; ...]                     (rows stacked)
    B' = [w_1 s_1 B_1, w_2 s_2 B_2, ...]     (columns stacked)

s_i is each input's own scale (alpha / sqrt(r) for rsLoRA, alpha / r otherwise). The output
is written with use_rslora=false and lora_alpha=r, so the loader applies a scale of 1, and
the tensors are stored in float32 so the merge adds no rounding of its own.

A single input with weight lam scales the adapter strength: delta_W -> lam * delta_W
(lam=0 is the base model, lam=1 the original checkpoint).

    python scripts/merge_lora_adapters.py --out output/merged/b20k_b100k \\
        --adapter ckpt/checkpoint-20000_best/lora_adapter:0.5 \\
        --adapter ckpt/checkpoint-100000_final/lora_adapter:0.5

All inputs must share target_modules and the same set of adapted modules. The output
directory also gets merge_recipe.json (input paths, weights, ranks and scales) and a
metadata.json carrying the inputs' recorded training prompts (task_prompts and
task_prompt_stats from each checkpoint's metadata.json), so evaluation reads the same
prompts as for the original checkpoints. Inputs whose recorded prompts differ are rejected.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


@dataclass
class Adapter:
    path: Path
    config: dict
    weight: float
    state: dict[str, torch.Tensor]
    metadata: dict

    @property
    def scale(self) -> float:
        rank, alpha = self.config["r"], self.config["lora_alpha"]
        return alpha / rank**0.5 if self.config.get("use_rslora", False) else alpha / rank


def parse_spec(spec: str) -> tuple[Path, float]:
    """Split ``DIR[:WEIGHT]``; a trailing component that is not a number belongs to DIR."""
    path, sep, weight = spec.rpartition(":")
    if sep:
        try:
            return Path(path), float(weight)
        except ValueError:
            pass
    return Path(spec), 1.0


def read_metadata(path: Path) -> dict:
    """Checkpoint metadata.json next to the adapter, or one level up for ``lora_adapter/``."""
    for candidate in (path, path.parent):
        meta = candidate / "metadata.json"
        if meta.is_file():
            return json.loads(meta.read_text())
    return {}


def load_adapter(path: Path, weight: float) -> Adapter:
    config = json.loads((path / "adapter_config.json").read_text())
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(f"{path}: per-module rank_pattern / alpha_pattern is not supported")
    state = load_file(str(path / "adapter_model.safetensors"))
    return Adapter(path, config, weight, state, read_metadata(path))


PROMPT_KEYS = ("task_prompts", "task_prompt_stats")


def merged_metadata(adapters: list[Adapter]) -> dict:
    """Training-prompt record shared by every input, plus the merge provenance."""
    has_prompts = [any(k in a.metadata for k in PROMPT_KEYS) for a in adapters]
    recorded = [a for a, has in zip(adapters, has_prompts) if has]
    if recorded and len(recorded) != len(adapters):
        missing = [str(a.path) for a, has in zip(adapters, has_prompts) if not has]
        raise ValueError(f"training prompts are recorded for some inputs but not {missing}")
    metadata: dict = {}
    for key in PROMPT_KEYS:
        values = [a.metadata.get(key) for a in recorded]
        if any(v != values[0] for v in values):
            raise ValueError(f"inputs were trained with different {key}; refusing to merge")
        if values and values[0] is not None:
            metadata[key] = values[0]
    for key in ("model_name", "lora_rank", "lora_alpha"):
        values = {json.dumps(a.metadata.get(key)) for a in adapters}
        if len(values) == 1 and adapters[0].metadata.get(key) is not None:
            metadata[f"source_{key}"] = adapters[0].metadata[key]
    metadata["merged_from"] = [
        {"path": str(a.path), "weight": a.weight, "step": a.metadata.get("step")}
        for a in adapters
    ]
    return metadata


def merge_adapters(adapters: list[Adapter]) -> tuple[dict[str, torch.Tensor], dict]:
    """Return the merged state dict and adapter config."""
    if not adapters:
        raise ValueError("need at least one adapter")
    first = adapters[0]
    a_keys = sorted(k for k in first.state if ".lora_A." in k)
    if not a_keys:
        raise ValueError(f"{first.path}: no lora_A tensors found")
    for adapter in adapters:
        if sorted(adapter.config["target_modules"]) != sorted(first.config["target_modules"]):
            raise ValueError(f"{adapter.path}: target_modules differ from {first.path}")
        if sorted(k for k in adapter.state if ".lora_A." in k) != a_keys:
            raise ValueError(f"{adapter.path}: adapted modules differ from {first.path}")

    merged: dict[str, torch.Tensor] = {}
    for a_key in a_keys:
        b_key = a_key.replace(".lora_A.", ".lora_B.")
        a_parts, b_parts = [], []
        for adapter in adapters:
            a, b = adapter.state[a_key].float(), adapter.state[b_key].float()
            rank = adapter.config["r"]
            if a.shape[0] != rank or b.shape[1] != rank:
                raise ValueError(f"{adapter.path}: {a_key} does not have rank {rank}")
            in_out = (first.state[a_key].shape[1], first.state[b_key].shape[0])
            if (a.shape[1], b.shape[0]) != in_out:
                raise ValueError(f"{adapter.path}: {a_key} shape differs from {first.path}")
            a_parts.append(a)
            b_parts.append(b * (adapter.weight * adapter.scale))
        merged[a_key] = torch.cat(a_parts, dim=0).contiguous()
        merged[b_key] = torch.cat(b_parts, dim=1).contiguous()

    rank = sum(adapter.config["r"] for adapter in adapters)
    config = dict(first.config, r=rank, lora_alpha=rank, use_rslora=False)
    return merged, config


def write_merged(adapters: list[Adapter], out: Path) -> None:
    merged, config = merge_adapters(adapters)
    metadata = merged_metadata(adapters)
    if "task_prompts" not in metadata:
        print("warning: no input records task_prompts; evaluation will use default prompts")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    save_file(merged, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    recipe = {
        "formula": "delta_W = sum_i weight_i * scale_i * B_i @ A_i",
        "inputs": [
            {"path": str(a.path), "weight": a.weight, "r": a.config["r"],
             "lora_alpha": a.config["lora_alpha"],
             "use_rslora": a.config.get("use_rslora", False), "scale": a.scale}
            for a in adapters
        ],
        "merged_rank": config["r"],
    }
    (out / "merge_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(f"wrote {out} (rank {config['r']}, {len(merged) // 2} modules)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--adapter", action="append", required=True, metavar="DIR[:WEIGHT]",
        help="PEFT adapter directory with an optional weight (default 1.0); repeatable",
    )
    parser.add_argument("--out", type=Path, required=True, help="output adapter directory")
    args = parser.parse_args()
    write_merged([load_adapter(*parse_spec(spec)) for spec in args.adapter], args.out)


if __name__ == "__main__":
    main()
