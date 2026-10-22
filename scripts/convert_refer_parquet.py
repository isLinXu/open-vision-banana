#!/usr/bin/env python3
"""Rebuild REFER-format RefCOCO / RefCOCO+ / RefCOCOg files from the parquet mirrors.

The original UNC host (bvisionweb1.cs.unc.edu) no longer resolves (checked
2026-10-22) and the Wayback copy is too slow to be practical. The ModelScope /
Hugging Face parquet conversions (``swift/refcoco``, ``swift/refcocoplus``,
``swift/refcocog`` = ``jxu124/refcoco*``) carry every field of the original
``refs(*).p`` + ``instances.json`` (``raw_anns`` holds the segmentation polygons),
so this script writes those two files back out. Everything downstream
(``scripts/build_res_manifest.py``, ``RefCOCOgDataset(protocol="umd")``,
``build_qwen_baseline_manifest.py --exclude-image-ids``) keeps working unchanged.

    python scripts/convert_refer_parquet.py --input data/raw/refer_parquet/refcocog \\
        --output data/raw/refer/refcocog --split-by umd
    python scripts/convert_refer_parquet.py --input data/raw/refer_parquet/refcoco \\
        --output data/raw/refer/refcoco --split-by unc

Split names follow REFER: refcocog(umd) = train / val / test; refcoco(+)(unc) =
train / val / testA / testB (the parquet's ``test`` shard is testA).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

SPLIT_FROM_SHARD = {"validation": "val", "test": "testA", "testB": "testB", "train": "train"}


def _as_obj(value: Any) -> Any:
    """Parquet cells may hold JSON text or already-structured values."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _shard_split(path: Path) -> str:
    stem = path.name.split("-")[0]
    return SPLIT_FROM_SHARD.get(stem, stem)


def convert_refer_parquet(
    input_dir: Path, output_dir: Path, *, split_by: str, dataset_name: str | None = None
) -> dict[str, int]:
    """Write ``instances.json`` and ``refs(<split_by>).p`` from ``input_dir/**/*.parquet``."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("pyarrow is required: pip install pyarrow") from exc

    shards = sorted(p for p in Path(input_dir).rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {input_dir}")
    if split_by == "umd":
        allowed_splits = {"train", "val", "test"}
    elif split_by in {"unc", "google"}:
        allowed_splits = {"train", "val", "testA", "testB", "test"}
    else:
        raise ValueError("split_by must be umd, unc or google")

    refs: list[dict[str, Any]] = []
    images: dict[int, dict[str, Any]] = {}
    annotations: dict[int, dict[str, Any]] = {}
    category_ids: set[int] = set()
    counts: dict[str, int] = {}

    for shard in shards:
        shard_split = _shard_split(shard)
        table = pq.read_table(shard).to_pylist()
        for row in table:
            split = str(row.get("split") or shard_split)
            if split == "test" and split_by != "umd" and shard_split == "testA":
                split = "testA"
            if split not in allowed_splits:
                raise ValueError(f"{shard.name}: unexpected split {split!r} for split_by={split_by}")
            image_id = int(row["image_id"])
            ann_id = int(row["ann_id"])
            category_id = int(row["category_id"])
            category_ids.add(category_id)

            info = _as_obj(row.get("raw_image_info")) or {}
            if image_id not in images:
                images[image_id] = {
                    "id": image_id,
                    "file_name": info.get("file_name") or f"COCO_train2014_{image_id:012d}.jpg",
                    "width": int(info["width"]),
                    "height": int(info["height"]),
                    **{k: info[k] for k in ("license", "coco_url", "flickr_url", "date_captured") if k in info},
                }
            ann = _as_obj(row.get("raw_anns")) or {}
            if ann_id not in annotations:
                if "segmentation" not in ann:
                    raise ValueError(f"ref {row.get('ref_id')} (ann {ann_id}) has no segmentation")
                annotations[ann_id] = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": ann["segmentation"],
                    "area": float(ann.get("area", 0.0)),
                    "bbox": list(ann.get("bbox") or row.get("bbox") or []),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            sentences = _as_obj(row.get("sentences")) or _as_obj(row.get("raw_sentences")) or []
            sentences = [
                {
                    "sent_id": int(s["sent_id"]),
                    "sent": s.get("sent", s.get("raw", "")),
                    "raw": s.get("raw", s.get("sent", "")),
                    "tokens": list(s.get("tokens") or str(s.get("sent", "")).split()),
                }
                for s in sentences
            ]
            if not sentences:
                raise ValueError(f"ref {row.get('ref_id')} has no sentences")
            refs.append(
                {
                    "ref_id": int(row["ref_id"]),
                    "ann_id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "split": split,
                    "file_name": str(row.get("file_name") or images[image_id]["file_name"]),
                    "sentences": sentences,
                    "sent_ids": [s["sent_id"] for s in sentences],
                }
            )
            counts[split] = counts.get(split, 0) + 1

    # REFER's instances.json lists the COCO 2014 categories; names are not needed downstream,
    # but keep ids so pycocotools-style readers stay happy.
    categories = [{"id": cid, "name": str(cid), "supercategory": "object"} for cid in sorted(category_ids)]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "instances.json").write_text(
        json.dumps(
            {
                "images": sorted(images.values(), key=lambda i: i["id"]),
                "annotations": sorted(annotations.values(), key=lambda a: a["id"]),
                "categories": categories,
                "info": {"description": f"REFER {dataset_name or input_dir.name} rebuilt from parquet"},
            }
        ),
        encoding="utf-8",
    )
    with (output_dir / f"refs({split_by}).p").open("wb") as handle:
        pickle.dump(sorted(refs, key=lambda r: r["ref_id"]), handle, protocol=2)
    summary = {"refs": len(refs), "images": len(images), "annotations": len(annotations), **counts}
    (output_dir / "convert_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="directory holding data/*.parquet")
    parser.add_argument("--output", type=Path, required=True, help="REFER directory to write")
    parser.add_argument("--split-by", choices=("umd", "unc", "google"), required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    summary = convert_refer_parquet(args.input, args.output, split_by=args.split_by, dataset_name=args.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
