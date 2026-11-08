#!/usr/bin/env python3
"""Compose pinned local manifests, not a quality audit or training readiness gate.

Spec: components maps each ROLES key to manifest, root, sha256, expected_train.
old_hypersim is an identity manifest (sample_id, source_dataset); its assets are
not needed. expected_removed_referring asserts the replaced baseline train rows.
All paths in the spec are relative to the spec directory. No downloads or models.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROLES = (
    "baseline",
    "old_hypersim",
    "full_res",
    "llmseg",
    "rea2seg",
    "clean_depth",
    "lingbot",
    "coconut",
    "indoor",
)
TASKS = {"depth", "normal", "semantic", "instance", "referring", "generation"}
ROLE_TASKS = {
    "old_hypersim": {"depth"},
    "full_res": {"referring"},
    "llmseg": {"referring"},
    "rea2seg": {"referring"},
    "clean_depth": {"depth"},
    "lingbot": {"depth"},
    "coconut": {"semantic", "instance"},
    "indoor": {"depth"},
}
PATH_KEYS = (
    "input",
    "target",
    "metric_depth",
    "valid_mask",
    "image",
    "image_path",
    "target_path",
    "mask",
    "mask_path",
    "conditioning_image",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(row):
    values = tuple(row.get(k) for k in ("source_dataset", "sample_id"))
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError("Replacement identity requires source_dataset and sample_id")
    return values


def local(value, root):
    p = Path(value)
    return (p if p.is_absolute() else root / p).resolve()


def validate_assets(row, root, context):
    required = ("target",) if row["task"] == "generation" else ("input", "target")
    for key in required:
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"{context}: missing {key}")
    for key in PATH_KEYS:
        if key not in row or row[key] is None:
            continue
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"{context}: invalid path {key}")
        asset = local(row[key], root)
        if not asset.is_file():
            raise FileNotFoundError(asset)
        row[key] = str(asset)


def load_component(role, entry, spec_root):
    if set(entry) != {"manifest", "root", "sha256", "expected_train"}:
        raise ValueError(f"{role}: unknown or missing component fields")
    if not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64:
        raise ValueError(f"{role}: SHA must be bound before composition")
    if type(entry["expected_train"]) is not int:
        raise ValueError(f"{role}: expected_train must be bound before composition")
    path = local(entry["manifest"], spec_root)
    root = local(entry["root"], spec_root)
    if digest(path) != entry["sha256"]:
        raise ValueError(f"{role}: manifest SHA mismatch")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{role}:{number}: expected object")
            if row.get("task") not in ROLE_TASKS.get(role, TASKS):
                raise ValueError(f"{role}:{number}: unknown or disallowed task")
            if row.get("split") not in {"train", "val", "test"}:
                raise ValueError(f"{role}:{number}: unknown split")
            source_split = row.get("source_split")
            if isinstance(source_split, str):
                source_split = source_split.lower()
            if row["split"] == "train" and source_split not in (
                None,
                "train",
                "training",
                "official_train",
                "raw_train_derived",
            ):
                raise ValueError(f"{role}:{number}: held-out source in train")
            if role == "old_hypersim":
                if row["split"] != "train":
                    raise ValueError("old_hypersim identity set must be train only")
                identity(row)
            elif role != "baseline":
                validate_assets(row, root, f"{role}:{number}")
            rows.append(row)
    count = sum(r["split"] == "train" for r in rows)
    expected = entry["expected_train"]
    if type(expected) is not int or expected < 0 or count != expected:
        raise ValueError(f"{role}: expected_train mismatch ({count} != {expected})")
    return rows, path


def write_rows(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def compose(spec_path, output):
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if set(spec) != {"components", "expected_removed_referring"}:
        raise ValueError("Unknown or missing spec fields")
    entries = spec["components"]
    if set(entries) != set(ROLES):
        raise ValueError("Supply every fixed component role, and no unknown roles")
    loaded = {role: load_component(role, entries[role], spec_path.parent) for role in ROLES}
    old = [identity(r) for r in loaded["old_hypersim"][0]]
    if len(set(old)) != len(old):
        raise ValueError("Duplicate old Hypersim replacement identity")
    old = set(old)
    matched, ledger, train, heldout = set(), [], [], []
    removed_referring = 0
    for index, row in enumerate(loaded["baseline"][0]):
        if row["split"] != "train":
            heldout.append(row)
            continue
        reason = None
        if row["task"] == "referring":
            reason = "replace_referring"
            removed_referring += 1
        elif row["task"] == "normal":
            reason = "exclude_normal_training"
        elif row["task"] == "depth":
            key = (row.get("source_dataset"), row.get("sample_id"))
            if key in old:
                if key in matched:
                    raise ValueError("Repeated baseline Hypersim replacement identity")
                matched.add(key)
                reason = "replace_hypersim_increment"
        if reason:
            ledger.append(
                {
                    "baseline_row": index + 1,
                    "reason": reason,
                    "sample_id": row.get("sample_id"),
                    "source_dataset": row.get("source_dataset"),
                }
            )
        else:
            train.append(row)
    if matched != old:
        raise ValueError("Baseline does not contain exact old Hypersim identity set")
    if removed_referring != spec["expected_removed_referring"]:
        raise ValueError("Baseline referring replacement count mismatch")
    # Removed baseline rows need identity/provenance, not obsolete assets.
    baseline_root = local(entries["baseline"]["root"], spec_path.parent)
    for row in train + heldout:
        validate_assets(row, baseline_root, "retained baseline")
    component_heldout = []
    for role in ROLES[2:]:
        for row in loaded[role][0]:
            (train if row["split"] == "train" else component_heldout).append(row)
    # Recheck input manifests before publishing; asset content audit is out of scope.
    for role, (_, path) in loaded.items():
        if digest(path) != entries[role]["sha256"]:
            raise ValueError(f"{role}: manifest changed during composition")
    output.mkdir(parents=True, exist_ok=False)
    marker = output / "BUILD_INCOMPLETE"
    marker.write_text("Composition not published\n", encoding="utf-8")
    files = {
        "pairs.jsonl": train,
        "training.jsonl": train + [r for r in heldout if r["split"] == "val"],
        "baseline-heldout.jsonl": heldout,
        "component-heldout.jsonl": component_heldout,
        "replacement-ledger.jsonl": ledger,
    }
    for name, rows in files.items():
        write_rows(output / name, rows)
    report = {
        "status": "COMPOSED_NOT_AUDITED",
        "training_ready": False,
        "spec_sha256": digest(spec_path),
        "sources": entries,
        "counts": dict(Counter(r["task"] for r in train)),
        "train_rows": len(train),
        "baseline_heldout_rows": len(heldout),
        "component_heldout_rows": len(component_heldout),
        "replacements": dict(Counter(r["reason"] for r in ledger)),
        "outputs": {n: digest(output / n) for n in files},
        "not_performed": [
            "asset content/quality audit",
            "cross-source leakage audit",
            "image or semantic deduplication",
            "training loader validation",
        ],
    }
    (output / "COMPOSED.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    marker.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compose(args.spec, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
