#!/usr/bin/env python3
"""Filter the initial depth20k and pinned reasoning sources before main training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from open_vision_banana.data.hypersim_supplement import file_sha256, sha256, write_json
from open_vision_banana.data.manifest import fingerprint_records, read_jsonl
from open_vision_banana.data.supplement_quality import excluded_images, inspect_referring_row


def filter_depth(source: Path, output: Path, *, expected_initial_count: int = 20000) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Keep the initial preparation separate from the filtered release")
    original = json.loads((source / "manifest.json").read_text())
    if (source / "BUILD_INCOMPLETE").exists() or original[
        "pairs_written"
    ] != expected_initial_count:
        raise ValueError("Expected the complete initial 20k preparation")
    for name, key in (
        ("pairs.jsonl", "manifest_sha256"),
        ("provenance.jsonl", "provenance_sha256"),
    ):
        if file_sha256(source / name) != original[key]:
            raise ValueError("Initial preparation metadata changed")
    records = read_jsonl(source / "provenance.jsonl")
    if len(records) != expected_initial_count or [r["row"] for r in records] != read_jsonl(
        source / "pairs.jsonl"
    ):
        raise ValueError("Initial manifest/provenance counts or rows differ")
    kept = [r for r in records if r["roundtrip_error_max"] <= 0.05]
    removed = [r for r in records if r["roundtrip_error_max"] > 0.05]
    if not kept:
        raise ValueError("No depth frames pass the maximum-error gate")
    output.mkdir(parents=True, exist_ok=True)
    (output / "BUILD_INCOMPLETE").write_text("Filtering and verifying native depth assets\n")
    for record in kept:
        for asset in record["assets"]:
            src, dst = source / asset["path"], output / asset["path"]
            if file_sha256(src) != asset["sha256"]:
                raise ValueError(f"Changed source asset: {asset['path']}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copyfile(src, dst)
            if file_sha256(dst) != asset["sha256"]:
                raise ValueError(f"Changed output asset: {asset['path']}")
    rows = [r["row"] for r in kept]
    rejected = read_jsonl(source / "rejected.jsonl") + [
        {
            "source": r["source"],
            "sample_id": r["row"]["sample_id"],
            "rejected": "frame_max_encoded_error_above_5cm",
            "encoded_max_error_m": r["roundtrip_error_max"],
        }
        for r in removed
    ]
    for name, entries in (
        ("pairs.jsonl", rows),
        ("provenance.jsonl", kept),
        ("rejected.jsonl", rejected),
    ):
        (output / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in entries),
            encoding="utf-8",
        )
    policy = dict(original["quality_policy"], maximum_frame_encoded_max_abs_error_m=0.05)
    selection = dict(
        original["selection"],
        count=len(rows),
        selection_mode="maximum_error_filtered_initial_20k",
        parent_manifest_sha256=original["manifest_sha256"],
    )
    write_json(output / "selection_spec.json", selection)
    write_json(output / "quality_policy.json", policy)
    valid = sum(r["valid_pixels"] for r in kept)
    report = dict(original)
    report.update(
        pairs_written=len(rows),
        selection=selection,
        quality_policy=policy,
        pre_filter={
            "rows": expected_initial_count,
            "manifest_sha256": original["manifest_sha256"],
            "source_fingerprint": original["source_fingerprint"],
            "removed_max_error": len(removed),
        },
        manifest_sha256=file_sha256(output / "pairs.jsonl"),
        provenance_sha256=file_sha256(output / "provenance.jsonl"),
        dataset_fingerprint=fingerprint_records(rows, output),
        source_fingerprint=sha256(
            json.dumps(
                [
                    [r["row"]["sample_id"], r["source_rgb_sha256"], r["source_depth_sha256"]]
                    for r in kept
                ],
                separators=(",", ":"),
            ).encode()
        ),
        unique_images=len(kept),
        unique_scenes=len({r["source_scene"] for r in rows}),
        unique_cameras=len({(r["source_scene"], r["source_camera"]) for r in rows}),
        scene_counts=dict(sorted(Counter(r["source_scene"] for r in rows).items())),
        rejected_frames=len(rejected),
        asset_bytes=sum(a["bytes"] for r in kept for a in r["assets"]),
        tasks={
            "tasks": ["depth"],
            "rows_per_task": {"depth": len(rows)},
            "normal_backfacing_frames": 0,
        },
        data_contract={
            "valid_pixels": valid,
            "invalid_pixels": sum(r["invalid_pixels"] for r in kept),
            "depth_meters": {
                "min": min(r["depth_min"] for r in kept),
                "max": max(r["depth_max"] for r in kept),
                "mean": sum(r["depth_sum"] for r in kept) / valid,
            },
            "encoded_roundtrip_abs_error": {
                "mean": sum(r["roundtrip_error_sum"] for r in kept) / valid,
                "max": max(r["roundtrip_error_max"] for r in kept),
            },
        },
    )
    write_json(output / "manifest.json", report)
    scenes = defaultdict(lambda: defaultdict(list))
    for row in rows:
        scenes[row["source_scene"]][row["source_camera"]].append(row["source_frame"])
    (output / "selected_frames.json").write_text(
        json.dumps(
            {"schema_version": 1, "count": len(rows), "scenes": scenes}, separators=(",", ":")
        )
        + "\n"
    )
    (output / "BUILD_INCOMPLETE").unlink()
    return report


def filter_referring(
    sources: dict[str, Path], output: Path, ledger: Path, *, workers: int = 16
) -> dict:
    rows = {name: read_jsonl(path) for name, path in sources.items()}
    inspected = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, path in sources.items():
            inspected[name] = list(
                pool.map(lambda r, path=path: inspect_referring_row(r, path), rows[name])
            )
    pixels, coco = excluded_images([r for values in inspected.values() for r in values])
    report = {
        "schema_version": 1,
        "resolution": 512,
        "policy": "Exclude whole images across sources if any target is empty for any allowed integer crop after the existing nearest-neighbour resize",
        "excluded_unique_decoded_images": len(pixels),
        "excluded_coco_ids": sorted(coco),
        "sources": {},
    }
    details = []
    for name, values in inspected.items():
        excluded = {
            r["sample_id"]
            for r in values
            if r["rgb_pixels_sha256"] in pixels or coco.intersection(r["coco_image_ids"])
        }
        kept = [r for r in rows[name] if r["sample_id"] not in excluded]
        report["sources"][name] = {
            "input_manifest_sha256": file_sha256(sources[name]),
            "input_rows": len(rows[name]),
            "retained_rows": len(kept),
            "excluded_rows": len(excluded),
            "directly_unsafe_rows": sum(not r["minimum_crop_foreground_pixels"] for r in values),
            "retained_fingerprint": fingerprint_records(kept, sources[name].parent),
            "excluded_sample_ids": sorted(excluded),
        }
        for record in values:
            details.append(dict(record, source=name, excluded=record["sample_id"] in excluded))
    write_json(output, report)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in details), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypersim", type=Path, required=True)
    parser.add_argument("--depth-output", type=Path, required=True)
    parser.add_argument("--rea2seg", type=Path, required=True)
    parser.add_argument("--llmseg", type=Path, required=True)
    parser.add_argument("--ref-filter", type=Path, required=True)
    parser.add_argument("--ref-ledger", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--release", type=Path, default=Path("docs/main_depth_reasoning_release.json")
    )
    args = parser.parse_args()
    expected = json.loads(args.release.read_text(encoding="utf-8"))
    initial = expected["hypersim"].get("pre_filter", expected["hypersim"])
    if file_sha256(args.hypersim / "pairs.jsonl") != initial["manifest_sha256"]:
        raise ValueError("Initial depth manifest differs from the pinned preparation")
    for name in ("rea2seg", "llmseg"):
        if file_sha256(getattr(args, name)) != expected[name]["manifest_sha256"]:
            raise ValueError(f"Reasoning manifest differs from the pinned release: {name}")
    ref = filter_referring(
        {name: getattr(args, name) for name in ("rea2seg", "llmseg")},
        args.ref_filter,
        args.ref_ledger,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                name: {k: v for k, v in info.items() if k != "excluded_sample_ids"}
                for name, info in ref["sources"].items()
            }
        ),
        flush=True,
    )
    depth = filter_depth(args.hypersim, args.depth_output)
    print(
        json.dumps({"depth_rows": depth["pairs_written"], "data_contract": depth["data_contract"]}),
        flush=True,
    )
