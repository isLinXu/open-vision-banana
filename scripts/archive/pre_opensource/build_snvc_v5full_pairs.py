#!/usr/bin/env python3
"""Build annotation-backed SANA RGB->RGB pair manifests from the sharded
``open-vision-banana-snvc-train-full`` (v5_full) dataset.

The released v5_full pack stores every image (input *and* generated target)
flat inside sharded task directories ``images/<family>/shardN/`` with the name
pattern ``{dataset}_{md5(orig)[:8]}_{basename}``.  Segmentation targets live in
``targets/segmentation`` and are addressed by the hash of the annotation's
``seg``/``mask`` path.  The generic ``build_mar_v5_clean_manifest.py`` assumes a
non-sharded layout with ``targets/depth`` and a macOS ``V5_ROOT``, so it cannot
resolve this dataset.  This script re-derives the exact same flattened names,
indexes the sharded directories once, and emits a ``pairs.jsonl`` compatible
with ``scripts/train_sana_512.py``.

Only image-to-image tasks usable by the Vision-Banana (prompt-conditioned
generation) route are emitted: depth, normal, structure.  Segmentation is
excluded because this pack ships ``targets/segmentation`` as source *photos*
(or missing), which is broken supervision.  ``structure`` pairs each wildrgbd
RGB frame (``image[i]``) with its black-background edge map (``depth[i]`` in
``targets/structure``).

Usage:
    python scripts/build_snvc_v5full_pairs.py \
        --root /media/linxu/thinkHub/datasets/open-vision-banana-snvc-train-full \
        --output output/snvc_v5full_pairs/pairs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_ROOT = Path("/media/linxu/thinkHub/datasets/open-vision-banana-snvc-train-full")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def get_dataset_name(jsonl_name: str) -> str:
    name = jsonl_name.replace("_sampled.jsonl", "").replace(".jsonl", "")
    for suffix in (
        "_gcg",
        "_genseg",
        "_point",
        "_bbox",
        "_visual",
        "_depth",
        "_normal",
        "_recon",
        "_und",
        "_edit_regenerated",
        "_total_unified",
    ):
        name = name.replace(suffix, "")
    return name


def get_task_family(jsonl_name: str) -> str:
    name = jsonl_name.replace("_sampled.jsonl", "")
    if "depth" in name:
        return "depth"
    if "normal" in name:
        return "normal"
    if "gcg" in name or "genseg" in name:
        return "segmentation"
    if "point" in name or "bbox" in name:
        return "detection"
    if "recon" in name:
        return "structure"
    if "visual" in name:
        return "recognition"
    return "other"


def flat_name(orig: str, dataset: str) -> str:
    digest = hashlib.md5(orig.encode()).hexdigest()[:8]
    return f"{dataset}_{digest}_{os.path.basename(orig)}"


def index_dir(base: Path) -> dict[str, Path]:
    """Map file basename -> absolute path for one directory tree (recursive).

    Handles both sharded ``images/<family>/shardN/*`` and flat
    ``targets/<family>/*`` layouts.
    """
    index: dict[str, Path] = {}
    if not base.is_dir():
        return index
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name, path)
    return index


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_rows(root: Path) -> tuple[list[dict], Counter]:
    ann_dir = root / "annotations"
    depth_idx = index_dir(root / "images" / "depth")
    normal_idx = index_dir(root / "images" / "normal")
    seg_input_idx = index_dir(root / "images" / "segmentation")
    seg_target_idx = index_dir(root / "targets" / "segmentation")
    structure_input_idx = index_dir(root / "images" / "structure")
    structure_target_idx = index_dir(root / "targets" / "structure")
    rejected: Counter = Counter()
    rows: list[dict] = []

    for ann in sorted(ann_dir.glob("*.jsonl")):
        family = get_task_family(ann.name)
        dataset = get_dataset_name(ann.name)
        if family not in {"depth", "normal", "structure"}:
            continue

        for line_no, item in _read_jsonl(ann):
            if family in {"depth", "normal"}:
                images = item.get("image")
                if not isinstance(images, list) or len(images) < 2:
                    rejected[f"{family}_bad_image_field"] += 1
                    continue
                inp_name = flat_name(images[0], dataset)
                tgt_name = flat_name(images[1], dataset)
                idx = depth_idx if family == "depth" else normal_idx
                inp, tgt = idx.get(inp_name), idx.get(tgt_name)
                if inp is None or tgt is None:
                    rejected[f"{family}_missing_pair"] += 1
                    continue
                if inp.resolve() == tgt.resolve():
                    rejected[f"{family}_identity_pair"] += 1
                    continue
                rows.append(
                    {
                        "task": family,
                        "input": str(inp),
                        "target": str(tgt),
                        "source": images[0],
                        "annotation": ann.name,
                        "line": line_no,
                    }
                )
            elif family == "structure":
                images = item.get("image")
                depths = item.get("depth")
                if not isinstance(images, list) or not isinstance(depths, list):
                    rejected["structure_bad_fields"] += 1
                    continue
                for rgb, dep in zip(images, depths):
                    if not isinstance(rgb, str) or not isinstance(dep, str):
                        rejected["structure_bad_fields"] += 1
                        continue
                    inp = structure_input_idx.get(flat_name(rgb, dataset))
                    tgt = structure_target_idx.get(flat_name(dep, dataset))
                    if inp is None or tgt is None:
                        rejected["structure_missing_pair"] += 1
                        continue
                    if inp.resolve() == tgt.resolve():
                        rejected["structure_identity_pair"] += 1
                        continue
                    rows.append(
                        {
                            "task": "structure",
                            "input": str(inp),
                            "target": str(tgt),
                            "source": rgb,
                            "annotation": ann.name,
                            "line": line_no,
                        }
                    )
            else:  # segmentation
                source = item.get("image")
                mask = item.get("seg") or item.get("mask")
                if not isinstance(source, str) or not isinstance(mask, str):
                    rejected["seg_bad_fields"] += 1
                    continue
                inp = seg_input_idx.get(flat_name(source, dataset))
                tgt = seg_target_idx.get(flat_name(mask, dataset))
                if inp is None or tgt is None:
                    rejected["seg_missing_pair"] += 1
                    continue
                if inp.resolve() == tgt.resolve():
                    rejected["seg_identity_pair"] += 1
                    continue
                rows.append(
                    {
                        "task": "seg",
                        "input": str(inp),
                        "target": str(tgt),
                        "source": source,
                        "annotation": ann.name,
                        "line": line_no,
                    }
                )
    return rows, rejected


def assign_split(rows: list[dict], val_percent: int) -> None:
    for row in rows:
        group = os.path.realpath(row["input"])
        bucket = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) % 100
        row["split"] = "val" if bucket < val_percent else "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/snvc_v5full_pairs/pairs.jsonl"),
    )
    parser.add_argument("--val-percent", type=int, default=5)
    args = parser.parse_args()

    if not 1 <= args.val_percent <= 50:
        raise SystemExit("--val-percent must be between 1 and 50")
    if not (args.root / "annotations").is_dir():
        raise SystemExit(f"No annotations dir under {args.root}")

    rows, rejected = collect_rows(args.root)
    assign_split(rows, args.val_percent)
    rows.sort(key=lambda r: (r["task"], r["split"], r["input"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["task"]][row["split"]] += 1
    audit = {
        "manifest": str(args.output.resolve()),
        "source_root": str(args.root.resolve()),
        "val_percent": args.val_percent,
        "total": len(rows),
        "counts": {task: dict(counts[task]) for task in sorted(counts)},
        "rejected": dict(rejected),
    }
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
