#!/usr/bin/env python3
"""Convert SN-VC-50M official JSONL annotations to training manifest format.

Reads the JSONL files from ``dense_geometric_prediction/``, ``segmentation/``,
etc. and produces a single JSONL manifest with the format::

    {"task": "depth", "input": "coco2017/train2017/000009.jpg",
     "target": "coco2017/depth/000009_depth.png", "split": "train",
     "prompt": "Predict relative depth..."}

Supported task types:
- **depth** / **normal** — ``image: [input, target]``
- **seg** (genseg / gcg) — ``image: input, seg: target``

Usage::

    python scripts/build_snvc50m_manifest.py \\
        --dataset-root data/raw/SenseNova-Vision-Corpus-50M \\
        --output output/snvc50m_manifest/pairs_snvc50m.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_manifest")

# Default prompts matching the TarArchiveDataset TASKS definitions
DEFAULT_PROMPTS = {
    "depth": (
        "Generate a relative depth map. Preserve object boundaries and encode "
        "continuous relative depth with the training color map."
    ),
    "normal": (
        "Generate an RGB surface-normal map. Encode XYZ surface orientation "
        "in the RGB channels and preserve fine geometry."
    ),
    "seg": (
        "Generate a semantic segmentation mask. Use only the fixed training "
        "palette, with one solid class color per pixel and sharp object boundaries."
    ),
}

VAL_RATIO = 0.05  # 5% validation


def _assign_split(record_id: str) -> str:
    """Deterministic train/val split based on hash of record id."""
    h = hashlib.md5(record_id.encode()).hexdigest()
    return "val" if int(h[:8], 16) % 100 < int(VAL_RATIO * 100) else "train"


def convert_depth_normal(
    jsonl_path: Path,
    task: str,
) -> list[dict]:
    """Convert depth or normal JSONL (image: [input, target] format)."""
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            images = d.get("image", [])
            if len(images) < 2:
                continue
            input_path = images[0]
            target_path = images[1]

            # Extract prompt from conversations if available
            prompt = DEFAULT_PROMPTS[task]
            convs = d.get("conversations", [])
            if convs and convs[0].get("from") == "human":
                raw_prompt = convs[0]["value"]
                # Strip <image> prefix
                if raw_prompt.startswith("<image>"):
                    raw_prompt = raw_prompt[len("<image>"):]
                prompt = raw_prompt.strip()

            rec_id = f"{task}_{d.get('id', '')}_{input_path}"
            records.append({
                "task": task,
                "input": input_path,
                "target": target_path,
                "split": _assign_split(rec_id),
                "prompt": prompt,
            })
    return records


def convert_seg_genseg(jsonl_path: Path) -> list[dict]:
    """Convert genseg JSONL (image + seg format)."""
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            input_path = d.get("image", "")
            seg_path = d.get("seg", "")
            if not input_path or not seg_path:
                continue

            rec_id = f"seg_{d.get('image_id', '')}_{input_path}"
            records.append({
                "task": "seg",
                "input": input_path,
                "target": seg_path,
                "split": _assign_split(rec_id),
                "prompt": DEFAULT_PROMPTS["seg"],
            })
    return records


def convert_seg_gcg(jsonl_path: Path) -> list[dict]:
    """Convert gcg JSONL (image + seg format, with caption)."""
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            input_path = d.get("image", "")
            seg_path = d.get("seg", "")
            if not input_path or not seg_path:
                continue

            rec_id = f"seg_gcg_{d.get('image_id', '')}_{input_path}"
            records.append({
                "task": "seg",
                "input": input_path,
                "target": seg_path,
                "split": _assign_split(rec_id),
                "prompt": DEFAULT_PROMPTS["seg"],
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SN-VC-50M JSONL annotations to training manifest"
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("data/raw/SenseNova-Vision-Corpus-50M"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/snvc50m_manifest/pairs_snvc50m.jsonl"),
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["depth", "normal", "seg"],
        help="Which tasks to include",
    )
    parser.add_argument(
        "--max-per-task", type=int, default=None,
        help="Limit samples per task (for smoke testing)",
    )
    args = parser.parse_args()

    all_records: list[dict] = []
    root = args.dataset_root

    # ── Depth & Normal ────────────────────────────────────────────────
    geo_dir = root / "dense_geometric_prediction"
    if geo_dir.is_dir():
        for jsonl_file in sorted(geo_dir.glob("*_depth_*.jsonl")):
            if "depth" in args.tasks:
                recs = convert_depth_normal(jsonl_file, "depth")
                logger.info("  depth: %s → %d records", jsonl_file.name, len(recs))
                all_records.extend(recs)

        for jsonl_file in sorted(geo_dir.glob("*_normal_*.jsonl")):
            if "normal" in args.tasks:
                recs = convert_depth_normal(jsonl_file, "normal")
                logger.info("  normal: %s → %d records", jsonl_file.name, len(recs))
                all_records.extend(recs)

    # ── Segmentation ──────────────────────────────────────────────────
    seg_dir = root / "segmentation"
    if seg_dir.is_dir():
        if "seg" in args.tasks:
            for jsonl_file in sorted(seg_dir.glob("*_genseg_*.jsonl")):
                recs = convert_seg_genseg(jsonl_file)
                logger.info("  seg(genseg): %s → %d records", jsonl_file.name, len(recs))
                all_records.extend(recs)

            for jsonl_file in sorted(seg_dir.glob("*_gcg_*.jsonl")):
                recs = convert_seg_gcg(jsonl_file)
                logger.info("  seg(gcg): %s → %d records", jsonl_file.name, len(recs))
                all_records.extend(recs)

    # ── Apply per-task limit ──────────────────────────────────────────
    if args.max_per_task:
        limited: list[dict] = []
        for task in args.tasks:
            task_recs = [r for r in all_records if r["task"] == task]
            limited.extend(task_recs[: args.max_per_task])
            if len(task_recs) > args.max_per_task:
                logger.info("  %s: limited to %d / %d", task, args.max_per_task, len(task_recs))
        all_records = limited

    # ── Stats ─────────────────────────────────────────────────────────
    task_counts: dict[str, dict[str, int]] = {}
    for rec in all_records:
        t = rec["task"]
        s = rec["split"]
        task_counts.setdefault(t, {"train": 0, "val": 0})
        task_counts[t][s] += 1

    logger.info("=" * 60)
    logger.info("Total records: %s", f"{len(all_records):,}")
    for task, splits in sorted(task_counts.items()):
        logger.info("  %s: train=%s, val=%s", task, f"{splits['train']:,}", f"{splits['val']:,}")

    # ── Write ─────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Written → %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
