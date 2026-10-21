#!/usr/bin/env python3
"""Write a referring pair manifest from Rea2Seg-16K parquet files.

The training prompt is the question. ``cot_answer`` is kept in the row for
inspection and is not the prompt. Targets are white on black, matching
referring template 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from open_vision_banana.inference.prompt import PromptBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def referring_prompt(question: str) -> str:
    # The canonical white/black prompt (paper Fig. 4 (c)/(d) template); RefCOCOg and
    # ReasonSeg evaluation build the very same string from the expression.
    return PromptBuilder.build_referring_prompt(question)


def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    binary = mask.astype(bool)
    rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rgb[binary] = 255
    return Image.fromarray(rgb, mode="RGB")


def decode_rle(payload: str | dict) -> np.ndarray:
    from pycocotools import mask as mask_utils

    encoded = json.loads(payload) if isinstance(payload, str) else payload
    if "rle" in encoded and isinstance(encoded["rle"], dict):
        encoded = encoded["rle"]
    counts = encoded["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    decoded = mask_utils.decode({"size": encoded["size"], "counts": counts})
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded


def image_bytes(value: object) -> bytes:
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is None:
            raise ValueError("image struct has no bytes")
        return raw
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"unsupported image value: {type(value).__name__}")


def assign_splits(
    categories: list[str],
    val_fraction: float,
    seed: int,
) -> list[str]:
    rng = np.random.default_rng(seed)
    splits = ["train"] * len(categories)
    for category in sorted(set(categories)):
        indexes = [i for i, name in enumerate(categories) if name == category]
        holdout = max(1, int(round(len(indexes) * val_fraction)))
        holdout = min(holdout, len(indexes) - 1)
        chosen = rng.choice(indexes, size=holdout, replace=False)
        for index in np.atleast_1d(chosen):
            splits[int(index)] = "val"
    return splits


def write_manifest(parquet_dir: Path, output_dir: Path, val_fraction: float, seed: int) -> dict:
    import pyarrow.parquet as pq

    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {parquet_dir}")
    rows: list[dict] = []
    for path in files:
        table = pq.read_table(path)
        columns = table.to_pydict()
        count = table.num_rows
        for index in range(count):
            rows.append({name: columns[name][index] for name in columns})
    categories = [str(row["category"]) for row in rows]
    splits = assign_splits(categories, val_fraction, seed)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "pairs.jsonl"
    counts = {"train": 0, "val": 0}
    by_category: dict[str, dict[str, int]] = {}
    with manifest.open("w", encoding="utf-8") as handle:
        for index, (row, split) in enumerate(zip(rows, splits, strict=True)):
            category = str(row["category"])
            stem = f"{category}_{index:06d}"
            image_path = image_dir / f"{stem}.jpg"
            mask_path = mask_dir / f"{stem}.png"
            image_path.write_bytes(image_bytes(row["image"]))
            mask_to_rgb(decode_rle(row["mask"])).save(mask_path)
            payload = {
                "input": str(image_path),
                "target": str(mask_path),
                "task": "referring",
                "split": split,
                "prompt": referring_prompt(str(row["question"])),
                "category": category,
                "cot_answer": str(row.get("cot_answer") or ""),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[split] += 1
            bucket = by_category.setdefault(category, {"train": 0, "val": 0})
            bucket[split] += 1
    summary = {"rows": len(rows), "counts": counts, "by_category": by_category, "manifest": str(manifest)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    summary = write_manifest(args.parquet_dir, args.output_dir, args.val_fraction, args.seed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
