#!/usr/bin/env python3
"""Join the pinned depth/reasoning supplements, optionally with the six-task main set.

An optional main manifest must pass an exact decoded-image train/held-out audit.
Existing held-out rows retain their order; this script never creates new splits,
caps a task, cleans away overlaps or treats the R2 single-task dataset as main.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from open_vision_banana.data.hypersim_supplement import file_sha256, write_json
from open_vision_banana.data.manifest import fingerprint_records, read_jsonl, resolve_manifest_path
from open_vision_banana.res_preparation import _pixels
from scripts.build_qwen_baseline_manifest import (
    benchmark_leak,
    coco_image_id_from_path,
    load_excluded_image_ids,
)
from scripts.merge_multitask_manifest import merge_manifests

SOURCES = {
    "hypersim": (20000, "depth", "hypersim"),
    "rea2seg": (15275, "referring", "snowball521/Rea2Seg-16K"),
    "llmseg": (5000, "referring", "wangjunchi/LLMSeg"),
}
MAIN_TASKS = {"depth", "normal", "semantic", "instance", "referring", "generation"}


def coco_ids(row: dict) -> set[int]:
    ids = {
        coco_image_id_from_path(str(row.get(key, ""))) for key in ("input", "image", "source_image")
    }
    source = str(row.get("source_dataset", "")).lower()
    # Rea2Seg image_id is a COCO id: from a COCO file name, or Visual Genome via --vg-image-data
    if row.get("image_id") is not None and any(
        s in source for s in ("refcoco", "llmseg", "rea2seg")
    ):
        ids.add(int(row["image_id"]))
    return {i for i in ids if i is not None}


def assert_no_heldout_overlap(rows: list[dict], manifest: Path) -> dict:
    """Check actual pixels across all tasks; metadata hashes alone are insufficient."""
    cache: dict[Path, str] = {}
    identities: dict[str, set[str]] = {"train": set(), "heldout": set()}
    image_ids: dict[str, set[int]] = {"train": set(), "heldout": set()}
    for row in rows:
        path = resolve_manifest_path(row["input"], manifest).resolve()
        if path not in cache:
            with Image.open(path) as image:
                cache[path] = _pixels(image.convert("RGB"))
        group = "train" if row["split"] == "train" else "heldout"
        identities[group].add(cache[path])
        image_ids[group].update(coco_ids(row))
    overlap = identities["train"] & identities["heldout"]
    id_overlap = image_ids["train"] & image_ids["heldout"]
    if overlap or id_overlap:
        raise ValueError(
            f"{len(overlap)} decoded images / {len(id_overlap)} COCO IDs occur in both train and held-out splits; "
            "rebuild/audit the base first (no rows or validation splits were silently changed)"
        )
    return {"unique_train_images": len(identities["train"]), "train_heldout_overlap": 0}


def merge(
    sources: dict[str, Path],
    output: Path,
    release: Path,
    exclude_image_ids: Path,
    *,
    base: Path | None = None,
) -> dict:
    expected = json.loads(release.read_text(encoding="utf-8"))
    excluded = load_excluded_image_ids([exclude_image_ids])
    if not excluded:
        raise ValueError("An official held-out COCO image-id union is required")
    hashes = {}
    seen_ids = set()
    source_rows = {}
    for name, (count, task, source_name) in SOURCES.items():
        count = expected[name].get("rows", count)
        path = sources[name]
        if (path.parent / "BUILD_INCOMPLETE").exists():
            raise ValueError(f"Incomplete dataset: {name}")
        rows = read_jsonl(path)
        source_rows[name] = rows
        hashes[name] = file_sha256(path)
        if name == "hypersim":
            actual = fingerprint_records(rows, path.parent)
            if actual != expected[name]["dataset_fingerprint"]:
                raise ValueError("Hypersim selection differs from the pinned release")
        elif hashes[name] != expected[name]["manifest_sha256"]:
            raise ValueError(f"Manifest differs from the pinned release: {name}")
        if len(rows) != count:
            raise ValueError(f"Expected {count} rows from {name}, got {len(rows)}")
        for row in rows:
            if (row.get("task"), row.get("split"), row.get("source_dataset")) != (
                task,
                "train",
                source_name,
            ):
                raise ValueError(f"Unexpected task/split/source: {name}")
            identity = row["sample_id"]
            if identity in seen_ids:
                raise ValueError(f"Repeated sample identity: {identity}")
            seen_ids.add(identity)
            enriched = dict(row, source=f"{source_name} {row.get('category', '')}")
            if benchmark_leak(enriched, set()) is not None:
                raise ValueError(f"Benchmark-source row in supplement: {identity}")
            if coco_ids(row) & excluded:
                raise ValueError(f"Held-out COCO image in supplement: {identity}")

    before_filter = {name: len(rows) for name, rows in source_rows.items()}
    quality = expected.get("referring_filter")
    if quality:
        filter_path = Path(__file__).resolve().parents[1] / quality["path"]
        if file_sha256(filter_path) != quality["sha256"]:
            raise ValueError("Referring quality filter differs from the pinned release")
        filters = json.loads(filter_path.read_text(encoding="utf-8"))
        if filters["resolution"] != 512:
            raise ValueError("Referring crop filter must match the 512px training configuration")
        for name in ("rea2seg", "llmseg"):
            rule = filters["sources"][name]
            if hashes[name] != rule["input_manifest_sha256"]:
                raise ValueError("Referring quality filter belongs to a different source")
            excluded_samples = set(rule["excluded_sample_ids"])
            if not excluded_samples.issubset({r["sample_id"] for r in source_rows[name]}):
                raise ValueError("Referring quality filter contains unknown sample IDs")
            source_rows[name] = [
                r for r in source_rows[name] if r["sample_id"] not in excluded_samples
            ]
            if (
                len(source_rows[name]) != rule["retained_rows"]
                or fingerprint_records(source_rows[name], sources[name].parent)
                != rule["retained_fingerprint"]
            ):
                raise ValueError("Filtered referring rows differ from the pinned release")
    supplement_count = sum(len(rows) for rows in source_rows.values())
    if base:
        audit_path = base.with_suffix(".audit.json")
        if not audit_path.is_file():
            raise ValueError("Base requires its existing metric/provenance audit JSON")
        base_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if base_audit.get("manifest_sha256") != file_sha256(base):
            raise ValueError("Base audit does not match the actual manifest")
        if base_audit.get("rejected_non_metric_depth_rows") != 0:
            raise ValueError("Base must pass the existing metric-depth filter before joining")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination, so the existing merger's relative paths stay valid.
    with tempfile.TemporaryDirectory(prefix="depth-reasoning-", dir=output.parent) as temp:
        from scripts.merge_multitask_manifest import PATH_KEYS, _rebase

        inputs = [base] if base else []
        for name, rows in source_rows.items():
            filtered_path = Path(temp) / f"{name}.jsonl"
            rebased = []
            for original in rows:
                row = dict(original)
                for key in PATH_KEYS:
                    if row.get(key):
                        row[key] = _rebase(
                            resolve_manifest_path(row[key], sources[name]),
                            filtered_path.parent,
                            False,
                        )
                rebased.append(row)
            filtered_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rebased),
                encoding="utf-8",
                newline="\n",
            )
            inputs.append(filtered_path)
        staged = Path(temp) / "pairs.jsonl"
        summary = merge_manifests(inputs, staged, cap=0, val_percent=0)
        expected_count = supplement_count
        if base:
            base_rows = read_jsonl(base)
            expected_count += len(base_rows)
            if {r["task"] for r in base_rows if r.get("split") == "train"} != MAIN_TASKS:
                raise ValueError("Base must contain exactly the six main training tasks")
            # New Hypersim scenes must also be disjoint from this actual base, not only E1 logs.
            import re

            old_scenes = set()
            for row in base_rows:
                for key in ("input", "target", "source_scene"):
                    old_scenes.update(re.findall(r"ai_\d{3}_\d{3}", str(row.get(key, ""))))
            new_scenes = {r["source_scene"] for r in read_jsonl(sources["hypersim"])}
            if old_scenes & new_scenes:
                raise ValueError("Supplement Hypersim scenes overlap the actual main manifest")
        if summary["total_emitted"] != expected_count or summary["per_task_dropped"]:
            raise ValueError(f"Missing assets or dropped rows: {summary}")
        merged = read_jsonl(staged)
        if any(coco_ids(row) & excluded for row in merged if row["split"] == "train"):
            raise ValueError(
                "Training rows overlap official held-out COCO IDs; audit the base first"
            )
        split_audit = assert_no_heldout_overlap(merged, staged) if base else {}
        # Rebase once more to the final anchor. Preserve the base held-out order explicitly.
        for row in merged:
            for key in PATH_KEYS:
                if row.get(key):
                    row[key] = _rebase(
                        resolve_manifest_path(row[key], staged), output.parent, False
                    )
        if base:
            heldout = []
            for original in base_rows:
                if original.get("split") == "train":
                    continue
                row = dict(original, source_manifest=base.name)
                if row["task"] == "generation" and not row.get("input"):
                    row["input"] = row["target"]
                for key in PATH_KEYS:
                    if row.get(key):
                        row[key] = _rebase(
                            resolve_manifest_path(row[key], base), output.parent, False
                        )
                heldout.append(row)
            merged = [r for r in merged if r["split"] == "train"] + heldout
        output.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged),
            encoding="utf-8",
            newline="\n",
        )
    report = {
        "status": "prepared",
        "rows": len(merged),
        "supplement_rows": supplement_count,
        "source_rows_before_ref_filter": before_filter,
        "source_rows_after_ref_filter": {name: len(rows) for name, rows in source_rows.items()},
        "referring_filter_sha256": quality["sha256"] if quality else None,
        "manifest_sha256": file_sha256(output),
        "input_manifest_sha256": hashes,
        "base_manifest_sha256": file_sha256(base) if base else None,
        "task_split_counts": dict(Counter(f"{r['task']}/{r['split']}" for r in merged)),
        "per_task_dropped": {},
        "per_task_split": summary["per_task_split"],
        "cap": 0,
        "validation_reassigned": False,
        "zero_shot_claim_allowed": False,
        "protocol": "supervised_main_extension" if base else "train_only_supplement",
        "source_benchmark_rows": 0,
        "known_coco_heldout_overlap": 0,
        "evaluation_image_id_overlap_by_source": {exclude_image_ids.name: 0},
        "train_val_coco_image_id_overlap": 0,
        "rejected_non_metric_depth_rows": 0,
        "metric_validation_evidence": "Pinned native-depth preparation; hash-matched base metric audit when supplied",
        "evaluation_image_ids_sha256": file_sha256(exclude_image_ids),
        "excluded_coco_image_ids": len(excluded),
        "official_ref_exclusion_evidence": "Pinned reasoning-v1 release: all-family held-out IDs and decoded RGB",
        **split_audit,
    }
    write_json(output.with_suffix(".audit.json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in SOURCES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--release", type=Path, default=Path("docs/main_depth_reasoning_release.json")
    )
    parser.add_argument("--exclude-image-ids", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(
                {name: getattr(args, name) for name in SOURCES},
                args.output,
                args.release,
                args.exclude_image_ids,
                base=args.base_manifest,
            ),
            indent=2,
        )
    )
