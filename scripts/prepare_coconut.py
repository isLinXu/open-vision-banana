#!/usr/bin/env python3
"""Build a provenance-preserving semantic/instance supplement from pinned COCONut-B."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.coconut_preparation import (
    ALIASES,
    POLICY_VERSION,
    REVISION,
    STUFF,
    THINGS,
    eligible_groups,
    image_split,
    parse_legend,
    render_targets,
)
from open_vision_banana.heldout import coco_image_id_from_path
from open_vision_banana.res_bundle import safe_path, sha256_file


def lines(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def work(job):
    row, source, out, split_map, val_fraction = job
    idx = row["image_id"]
    try:
        inp = safe_path(Path(source), row["image"])
        mask = safe_path(Path(source), row["seg"])
        with Image.open(inp) as im:
            original = im.convert("RGB")
            digest = hashlib.sha256(
                f"{original.width}x{original.height}:".encode() + original.tobytes()
            ).hexdigest()
        with Image.open(mask) as im:
            if im.format != "PNG" or im.mode != "RGB" or im.size != original.size:
                raise ValueError("target must be same-size lossless RGB PNG")
            rgb = np.array(im)
        entries = parse_legend(row)
        groups, reasons, black = eligible_groups(rgb, entries)
        split = image_split(digest, idx, split_map, val_fraction)
        rows, assets = [], []
        root = Path(out)
        input_name = f"images/{idx:012d}.jpg"
        seed = int(hashlib.sha256(f"42:{idx}".encode()).hexdigest()[:16], 16)
        for n, (meta, target) in enumerate(render_targets(rgb, groups, seed)):
            target_name = f"targets/{idx:012d}_{meta['task']}_{n}.png"
            dest = root / target_name
            Image.fromarray(target).save(dest, compress_level=3)
            packed = (
                (target[..., 0].astype(np.uint32) << 16)
                | (target[..., 1].astype(np.uint32) << 8)
                | target[..., 2]
            )
            color_ids = np.unique(packed)
            if meta["task"] == "semantic":
                allowed = {(r << 16) | (g << 8) | b for r, g, b in meta["mapping"].values()}
                if not set(map(int, color_ids)) <= allowed:
                    raise ValueError("semantic target/prompt mismatch")
            elif len(color_ids) != meta["num_instances"] + int(0 in color_ids):
                raise ValueError("instance count mismatch")
            rows.append(
                dict(
                    meta,
                    input=input_name,
                    target=target_name,
                    split=split,
                    sample_id=f"coconut_b:{idx}:{meta['task']}:{n}",
                    source_dataset="coconut_b",
                    source_image_id=idx,
                    source_revision=REVISION,
                    source_mask=row["seg"],
                    source_input=row["image"],
                    source_legend=entries,
                    source_black_fraction=black,
                    encoding=POLICY_VERSION,
                    background_policy="outside_selected_source_categories",
                    rgb_pixels_sha256=digest,
                )
            )
            assets.append(
                {"path": target_name, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}
            )
        if rows:
            dest = root / input_name
            shutil.copyfile(inp, dest)
            assets.append(
                {"path": input_name, "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}
            )
        return {
            "id": idx,
            "pixel_hash": digest,
            "rows": rows,
            "assets": assets,
            "category_rejections": reasons,
            "source_mask_sha256": sha256_file(mask),
            "source_input_sha256": sha256_file(inp),
        }
    except (ValueError, OSError, KeyError) as e:
        return {"id": idx, "error": str(e)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--exclude-image-ids", type=Path, required=True)
    p.add_argument("--reference-manifest", type=Path, action="append", default=[])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-images", type=int)
    p.add_argument("--val-fraction", type=float, default=0.02)
    args = p.parse_args()
    if not 0 <= args.val_fraction < 1:
        p.error("val-fraction must be in [0,1)")
    if args.workers < 1 or (args.max_images is not None and args.max_images < 1):
        p.error("workers and max-images must be positive")
    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output must be empty")
    root.mkdir(parents=True, exist_ok=True)
    (root / "BUILD_INCOMPLETE").touch()
    for name in ("images", "targets"):
        (root / name).mkdir()
    exclude = set(json.loads(args.exclude_image_ids.read_text()))
    prior = {}
    references = []
    for path in args.reference_manifest:
        for r in lines(path):
            idx = coco_image_id_from_path(r.get("input", ""))
            if idx is None:
                continue
            if r["split"] != "train":
                exclude.add(idx)
            else:
                prior[idx] = "train"
        references.append({"name": path.name, "sha256": sha256_file(path)})
    source = args.source_root
    source_manifest = (
        source
        / "training/cleaned_manifests/segmentation/coconut_b_total_unified_genseg_edit_regenerated.jsonl"
    )
    paired = source / "image_text_image/manifests/coconut_b.jsonl"
    inventory_path = source / "source_inventory.json"
    if not args.max_images:
        inventory = json.loads(inventory_path.read_text())
        if inventory["revision"] != REVISION or len(inventory["shards"]) != 10:
            raise ValueError("Expected complete pinned source acquisition")
        entries = {e["path"]: e for e in inventory["files"]}
        for path in (source_manifest, paired):
            entry = entries[path.relative_to(source).as_posix()]
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError("Source manifest changed after acquisition")
    pair_index = {}
    for r in lines(paired):
        if r["id"] in pair_index:
            raise ValueError("Duplicate pair id")
        pair_index[r["id"]] = r["image"]
    records, seen = [], set()
    rejected = []
    for r in lines(source_manifest):
        idx = r["image_id"]
        if coco_image_id_from_path(r["image"]) != idx or coco_image_id_from_path(r["seg"]) != idx:
            raise ValueError("Source paths and COCO image id disagree")
        if idx in seen:
            raise ValueError("Duplicate raw image id")
        seen.add(idx)
        if pair_index.get(idx) != [r["image"], r["seg"]]:
            rejected.append({"id": idx, "reason": "pair_join_mismatch"})
        elif idx in exclude:
            rejected.append({"id": idx, "reason": "heldout_image_id"})
        else:
            records.append(r)
    if seen != set(pair_index):
        raise ValueError("Source and pair populations differ")
    records.sort(key=lambda r: r["image_id"])
    if args.max_images:
        records = records[: args.max_images]
    counts, hashes, asset_count = Counter(), set(), 0
    class_counts = Counter()
    # Map order is deterministic regardless of worker completion order.
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        (root / "pairs.jsonl").open("w", encoding="utf-8") as manifest,
        (root / "assets.jsonl").open("w", encoding="utf-8") as inventory,
        (root / "source_audit.jsonl").open("w", encoding="utf-8") as audit,
    ):
        jobs = ((r, str(source / "media"), str(root), prior, args.val_fraction) for r in records)
        for n, result in enumerate(pool.map(work, jobs, chunksize=16), 1):
            rows = result.pop("rows", [])
            assets = result.pop("assets", [])
            if "error" in result:
                rejected.append(
                    {"id": result["id"], "reason": "invalid_source", "detail": result["error"]}
                )
            elif result["pixel_hash"] in hashes:
                rejected.append({"id": result["id"], "reason": "duplicate_decoded_rgb"})
                rows, assets = [], []
            else:
                hashes.add(result["pixel_hash"])
                if not rows:
                    rejected.append({"id": result["id"], "reason": "no_eligible_category"})
            for row in rows:
                manifest.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"{row['task']}/{row['split']}"] += 1
                class_counts.update(row.get("categories", [row.get("category")]))
            for asset in assets:
                inventory.write(json.dumps(asset, sort_keys=True) + "\n")
            asset_count += len(assets)
            counts["source_images_retained"] += bool(rows)
            audit.write(json.dumps(dict(result, rows_emitted=len(rows)), ensure_ascii=False) + "\n")
            if n % 1000 == 0:
                print(json.dumps({"processed": n, "counts": counts}), flush=True)
    dump(root / "rejected.json", rejected)
    dump(root / "evaluation_image_ids.json", sorted(exclude))
    dump(
        root / "policy.json",
        {
            "version": POLICY_VERSION,
            "things": sorted(THINGS),
            "stuff": sorted(STUFF),
            "aliases": ALIASES,
            "min_region_pixels": 64,
            "min_region_fraction": 2e-4,
            "max_semantic_classes": 12,
            "max_instance_rows_per_image": 3,
            "absent_classes": 0,
            "black_fraction_rejection": False,
            "drop_entire_category_if_any_region_empty_or_tiny": True,
            "semantic_variants": 1,
            "seed": 42,
            "val_fraction": args.val_fraction,
        },
    )
    dump(
        root / "provenance.json",
        {
            "source_repo": "https://modelscope.cn/datasets/isLinXu/open-vision-banana",
            "source_revision": REVISION,
            "raw_manifest_sha256": sha256_file(source_manifest),
            "pair_manifest_sha256": sha256_file(paired),
            "exclude_ids_input_sha256": sha256_file(args.exclude_image_ids),
            "reference_manifests": references,
            "source_population": len(seen),
            "processed_images": len(records),
            "partial_build": bool(args.max_images),
            "source_inventory_sha256": sha256_file(inventory_path)
            if inventory_path.exists()
            else None,
            "class_counts": class_counts,
            "limitations": [
                "Source black outside selected categories is treated as task background.",
                "No independent proof of exhaustive instance or crowd annotations; source omissions can remain.",
                "Heldout filtering uses COCO image identity; decoded-pixel deduplication is within this supplement.",
            ],
        },
    )
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in root.iterdir()
        if p.is_file() and p.name != "BUILD_INCOMPLETE"
    }
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "status": "complete",
        "encoding": POLICY_VERSION,
        "counts": dict(counts),
        "asset_count": asset_count,
        "rejections": dict(Counter(x["reason"] for x in rejected)),
        "files": files,
        "dataset_fingerprint": fingerprint,
    }
    dump(root / "build_report.json", report)
    (root / "BUILD_INCOMPLETE").unlink()
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
