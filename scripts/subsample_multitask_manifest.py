#!/usr/bin/env python3
"""Build a deterministic E2b vision-training subset from the audited E1 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


VISION_TASKS = frozenset({"depth", "normal", "semantic", "instance", "referring"})
PATH_KEYS = ("input", "target", "image")
COCO_IMAGE_ID_RE = re.compile(r"^(?:COCO_(?:train|val|test)20\d{2}_)?(\d{12})\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _coco_id(row: dict) -> int | None:
    match = COCO_IMAGE_ID_RE.match(Path(str(row.get("input", ""))).name)
    return int(match.group(1)) if match else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rebase(row: dict, source: Path, output: Path) -> dict:
    if source.parent.resolve() == output.parent.resolve():
        return row
    rebased = dict(row)
    for key in PATH_KEYS:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            continue
        asset = Path(value)
        if not asset.is_absolute():
            asset = source.parent / asset
        rebased[key] = Path(os.path.relpath(asset.resolve(), output.parent.resolve())).as_posix()
    return rebased


def subsample_manifest(source: Path, output: Path, *, fraction: float = 0.25, seed: int = 42) -> dict:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("source and output manifests must differ")
    source_hash = _sha256(source)
    source_audit_path = source.with_suffix(".audit.json")
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("manifest_sha256") != source_hash:
        raise ValueError("E1 manifest does not match its audit SHA256")
    if source_audit.get("protocol") != "strict" or not source_audit.get("zero_shot_claim_allowed"):
        raise ValueError("E2b requires an audited strict E1 manifest")
    overlaps = source_audit.get("evaluation_image_id_overlap_by_source")
    if not isinstance(overlaps, dict) or not overlaps or any(value != 0 for value in overlaps.values()):
        raise ValueError("E1 audit must show zero evaluation image overlap")

    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    vision_train: dict[str, list[tuple[str, int]]] = defaultdict(list)
    before = Counter()
    for index, row in enumerate(rows):
        task, split = row.get("task"), row.get("split")
        if task not in VISION_TASKS | {"generation"} or split not in {"train", "val", "test"}:
            raise ValueError(f"E1 manifest row {index + 1} has unexpected task/split: {task!r}/{split!r}")
        before[f"{task}/{split}"] += 1
        if task in VISION_TASKS and split == "train":
            canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            rank = hashlib.sha256(f"{seed}:{task}:{canonical}".encode("utf-8")).hexdigest()
            vision_train[task].append((rank, index))

    selected: set[int] = set()
    for candidates in vision_train.values():
        count = max(1, int(len(candidates) * fraction + 0.5))
        selected.update(index for _, index in sorted(candidates)[:count])

    retained = [
        _rebase(row, source, output)
        for index, row in enumerate(rows)
        if row["split"] != "train" or row["task"] == "generation" or index in selected
    ]
    after = Counter(f"{row['task']}/{row['split']}" for row in retained)
    train_ids = {_coco_id(row) for row in retained if row["split"] == "train"}
    val_ids = {_coco_id(row) for row in retained if row["split"] != "train"}
    train_ids.discard(None)
    val_ids.discard(None)
    for key, count in before.items():
        task, split = key.split("/")
        if (task == "generation" or split != "train") and after[key] != count:
            raise AssertionError(f"E2b changed unchanged partition {key}")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained)
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if output.exists() and _sha256(output) != payload_hash:
        raise ValueError(f"E2b output differs from existing {output}; choose a new output path")
    output.write_text(payload, encoding="utf-8", newline="\n")
    report = {
        "output_manifest": str(output.resolve()),
        "manifest_sha256": payload_hash,
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": source_hash,
        "source_audit_sha256": _sha256(source_audit_path),
        "protocol": source_audit["protocol"],
        "zero_shot_claim_allowed": source_audit["zero_shot_claim_allowed"],
        "evaluation_image_id_overlap_by_source": overlaps,
        "train_val_coco_image_id_overlap": len(train_ids & val_ids),
        "rejected_non_metric_depth_rows": source_audit.get("rejected_non_metric_depth_rows", 0),
        "sampling": {"scope": "vision_train_only", "fraction": fraction, "seed": seed, "method": "sha256_rank"},
        "task_split_counts_before": dict(sorted(before.items())),
        "task_split_counts": dict(sorted(after.items())),
        "per_task_split": {
            task: {split: after[f"{task}/{split}"] for split in ("train", "val", "test") if after[f"{task}/{split}"]}
            for task in sorted(VISION_TASKS | {"generation"})
        },
        "per_task_dropped": {},
    }
    output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="audited E1 final manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(subsample_manifest(args.source, args.output, fraction=args.fraction, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
