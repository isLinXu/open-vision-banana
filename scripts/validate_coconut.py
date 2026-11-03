#!/usr/bin/env python3
"""Independently validate every released asset and source-to-target pixel union."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.coconut_preparation import POLICY_VERSION
from open_vision_banana.res_bundle import safe_path, sha256_file


def check_group(job):
    root, source, rows, assets = job
    root, source = Path(root), Path(source)
    first = rows[0]
    image = Image.open(safe_path(root, first["input"])).convert("RGB")
    pixel_hash = hashlib.sha256(
        f"{image.width}x{image.height}:".encode() + image.tobytes()
    ).hexdigest()
    if pixel_hash != first["rgb_pixels_sha256"]:
        raise ValueError("Decoded image fingerprint mismatch")
    rgb = np.array(Image.open(safe_path(source, first["source_mask"])).convert("RGB"))
    for name in {r[k] for r in rows for k in ("input", "target")}:
        path = safe_path(root, name)
        if (
            path.stat().st_size != assets[name]["bytes"]
            or sha256_file(path) != assets[name]["sha256"]
        ):
            raise ValueError(f"Changed release asset: {name}")
    from open_vision_banana.data.openvocab_semantic import parse_prompt_mapping
    from open_vision_banana.inference.prompt import PromptBuilder

    for row in rows:
        target_im = Image.open(safe_path(root, row["target"]))
        if target_im.mode != "RGB" or target_im.format != "PNG" or target_im.size != image.size:
            raise ValueError("Invalid target mode/size/format")
        target = np.array(target_im)
        if row["encoding"] != POLICY_VERSION or row["rgb_pixels_sha256"] != pixel_hash:
            raise ValueError("Invalid row identity/encoding")
        if row["task"] == "semantic":
            mapping = {k: tuple(v) for k, v in row["mapping"].items()}
            if parse_prompt_mapping(row["prompt"]) != mapping or row["absent_classes"]:
                raise ValueError("Semantic prompt contract mismatch")
            expected = np.empty_like(rgb)
            expected[:] = mapping["background"]
            for e in row["source_legend"]:
                if e["category"] in row["categories"]:
                    expected[np.all(rgb == e["rgb"], axis=2)] = mapping[e["category"]]
            if not np.array_equal(expected, target):
                raise ValueError("Semantic pixels differ from source category unions")
        elif row["task"] == "instance":
            if row["prompt"] != PromptBuilder.build_instance_prompt(row["category"]):
                raise ValueError("Instance prompt mismatch")
            used = set()
            expected_fg = np.zeros(rgb.shape[:2], bool)
            for e in row["source_legend"]:
                if e["category"] != row["category"]:
                    continue
                mask = np.all(rgb == e["rgb"], axis=2)
                expected_fg |= mask
                color = tuple(target[mask][0])
                if color == (0, 0, 0) or color in used or not np.all(target[mask] == color):
                    raise ValueError("Instance region is empty, merged, or non-solid")
                used.add(color)
            if len(used) != row["num_instances"] or not np.array_equal(
                expected_fg, np.any(target != 0, axis=2)
            ):
                raise ValueError("Instance region count/foreground mismatch")
        else:
            raise ValueError("Unknown task")
    return pixel_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dir", type=Path)
    p.add_argument("--source-media", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    root = args.data_dir
    if (root / "BUILD_INCOMPLETE").exists():
        raise ValueError("Incomplete build")
    build = json.loads((root / "build_report.json").read_text(encoding="utf-8"))
    for name, meta in build["files"].items():
        path = safe_path(root, name)
        if path.stat().st_size != meta["bytes"] or sha256_file(path) != meta["sha256"]:
            raise ValueError(f"Changed metadata: {name}")
    excluded = set(json.loads((root / "evaluation_image_ids.json").read_text(encoding="utf-8")))
    groups = defaultdict(list)
    counts = Counter()
    names = set()
    sample_ids = set()
    with (root / "pairs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            idx = row["source_image_id"]
            if (
                idx in excluded
                or row["split"] not in {"train", "val"}
                or row["sample_id"] in sample_ids
            ):
                raise ValueError("Heldout leak/invalid split/duplicate sample")
            sample_ids.add(row["sample_id"])
            groups[idx].append(row)
            counts[f"{row['task']}/{row['split']}"] += 1
            names.update((row["input"], row["target"]))
    counts["source_images_retained"] = len(groups)
    if dict(counts) != build["counts"]:
        raise ValueError("Count mismatch")
    with (root / "assets.jsonl").open(encoding="utf-8") as f:
        asset_rows = [json.loads(line) for line in f]
    assets = {a["path"]: a for a in asset_rows}
    if set(assets) != names or len(assets) != len(asset_rows):
        raise ValueError("Asset inventory mismatch")
    for rows in groups.values():
        if len({r["split"] for r in rows}) != 1:
            raise ValueError("Source image crosses splits")
    jobs = (
        (
            str(root),
            str(args.source_media),
            rows,
            {n: assets[n] for r in rows for n in (r["input"], r["target"])},
        )
        for rows in groups.values()
    )
    hashes = set()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, digest in enumerate(pool.map(check_group, jobs, chunksize=16), 1):
            if digest in hashes:
                raise ValueError("Duplicate decoded RGB")
            hashes.add(digest)
            if i % 2000 == 0:
                print(json.dumps({"validated_images": i}), flush=True)
    result = {
        "status": "passed",
        "counts": dict(counts),
        "dataset_fingerprint": build["dataset_fingerprint"],
        "asset_files": len(assets),
        "source_pixel_unions_verified": True,
        "excluded_ids": len(excluded),
    }
    (root / "validation_report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
