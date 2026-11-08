#!/usr/bin/env python3
"""Finalize an LLMSeg component with the existing whole-image crop-visibility policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.data.supplement_quality import excluded_images, inspect_referring_row
from open_vision_banana.res_bundle import safe_path
from open_vision_banana.res_preparation import validate_prepared_pairs


def finalize(source: Path, output: Path, *, resolution: int = 512, workers: int = 4) -> dict:
    """Keep all rows on images whose targets survive every allowed crop; never relabel splits."""
    source, output = source.resolve(), output.resolve()
    if resolution <= 0 or workers <= 0:
        raise ValueError("resolution and workers must be positive")
    if source == output or output.exists():
        raise ValueError("Use a fresh output directory, separate from the prepared source")
    if (source / "BUILD_INCOMPLETE").exists():
        raise ValueError("Source preparation is incomplete")
    manifest = source / "training.jsonl"
    payload = manifest.read_bytes()
    original = json.loads((source / "report.json").read_text(encoding="utf-8"))
    before_hash = hashlib.sha256(payload).hexdigest()
    if original["manifest_sha256"] != before_hash:
        raise ValueError("Source manifest differs from preparation report")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not rows or any(
        row.get("source_dataset") != "wangjunchi/LLMSeg"
        or row.get("split") != "train"
        or row.get("source_split") != "train"
        for row in rows
    ):
        raise ValueError("Expected only original LLMSeg training rows")
    for row in rows:
        for key in ("input", "target"):
            safe_path(source, row[key])
    # Verify every source target before filtering, including targets later removed.
    target_hashes = {}
    for row in rows:
        name = row["target"]
        if name not in target_hashes:
            with Image.open(safe_path(source, name)) as image:
                image = image.convert("RGB")
                digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode())
                digest.update(image.tobytes())
                target_hashes[name] = digest.hexdigest()
        if target_hashes[name] != row["target_sha256"]:
            raise ValueError("Source target pixels changed")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inspected = list(
            pool.map(lambda row: inspect_referring_row(row, manifest, resolution), rows)
        )
    if any(
        item["rgb_pixels_sha256"] != row["image_sha256"]
        for row, item in zip(rows, inspected, strict=False)
    ):
        raise ValueError("Source image pixels changed")
    pixels, coco_ids = excluded_images(inspected)
    removed_ids = {
        item["sample_id"]
        for item in inspected
        if item["rgb_pixels_sha256"] in pixels or coco_ids.intersection(item["coco_image_ids"])
    }
    kept = [row for row in rows if row["sample_id"] not in removed_ids]
    if not kept:
        raise ValueError("No rows survive the crop policy")
    output.mkdir(parents=True)
    marker = output / "BUILD_INCOMPLETE"
    marker.write_text("Finalizing LLMSeg crop component\n", encoding="utf-8")
    for name in sorted({row[key] for row in kept for key in ("input", "target")}):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Expected safe relative prepared asset names")
        src, dst = safe_path(source, name), safe_path(output, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    (output / "training.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept),
        encoding="utf-8",
        newline="\n",
    )
    with (output / "rejected.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write((source / "rejected.jsonl").read_text(encoding="utf-8"))
        for row in rows:
            if row["sample_id"] in removed_ids:
                stream.write(
                    json.dumps(
                        {
                            "sample_id": row["sample_id"],
                            "source_row": row["source_row"],
                            "source_image": row["source_image"],
                            "reason": "crop_whole_image_closure",
                        }
                    )
                    + "\n"
                )
    crop = {
        "resolution": resolution,
        "input_rows": len(rows),
        "retained_rows": len(kept),
        "excluded_rows": len(rows) - len(kept),
        "directly_unsafe_rows": sum(
            not item["minimum_crop_foreground_pixels"] for item in inspected
        ),
        "closure_scope": "within this LLMSeg component, not propagated to other sources",
        "excluded_coco_ids": sorted(coco_ids),
        "excluded_pixel_hashes": sorted(pixels),
    }
    report = dict(original)
    report.update(
        llmseg_rows=len(kept),
        counts={"train": len(kept)},
        llmseg_category_counts=dict(Counter(row["category"] for row in kept)),
        unique_images=len({row["input"] for row in kept}),
        unique_targets=len({row["target"] for row in kept}),
        manifest_sha256=hashlib.sha256((output / "training.jsonl").read_bytes()).hexdigest(),
        rejected_rows=original["rejected_rows"] + len(rows) - len(kept),
        pre_crop={"rows": len(rows), "manifest_sha256": before_hash},
        crop_filter=crop,
    )
    report["rejected_by_reason"] = dict(original["rejected_by_reason"])
    report["rejected_by_reason"]["crop_whole_image_closure"] = len(rows) - len(kept)
    for name, value in (("report.json", report), ("crop_filter_report.json", crop)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    (output / "crop_filter_ledger.jsonl").write_text(
        "".join(
            json.dumps(dict(item, excluded=item["sample_id"] in removed_ids), sort_keys=True) + "\n"
            for item in inspected
        ),
        encoding="utf-8",
        newline="\n",
    )
    # The generic validator requires the marker absent; restore it on any failure.
    marker.unlink()
    try:
        validation = validate_prepared_pairs(output)
    except Exception:
        marker.write_text("Final native validation failed\n", encoding="utf-8")
        raise
    return {
        "crop_filter": crop,
        "validation": validation,
        "manifest_sha256": report["manifest_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(
                args.data_dir, args.output_dir, resolution=args.resolution, workers=args.workers
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
