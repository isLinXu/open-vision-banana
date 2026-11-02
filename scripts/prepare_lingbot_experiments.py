#!/usr/bin/env python3
"""Freeze paired full-pool manifests; never drop rows or invent validation splits."""

from __future__ import annotations

# ruff: noqa: E402 -- standalone CLI bootstraps the repository source path.
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


from open_vision_banana.data.manifest import read_jsonl, resolve_manifest_path
from open_vision_banana.res_bundle import safe_path, sha256_file
from scripts.merge_main_depth_reasoning import assert_no_heldout_overlap, decoded_identity

REVISION = "78946c7b0aff52b87fe2ad15143e93625a43b139"
FINGERPRINT = "e3a39e02e2b4383d341d59884ed2dc90ef8e5bacdad466744486b2db4c55863c"
PAIRS_SHA = "2a277c9dc0c7523ad434d59407e40b7693b85de4d53558538acf4596f4548b8f"
ASSETS_SHA = "09a39c1fde318e814cd5600f73b2f79eb3c184193e147ee3ab0cc21d20f3c89d"
ROWS = 50000
TASKS = {"depth", "normal", "semantic", "instance", "referring", "generation"}
PATHS = ("input", "target", "image", "metric_depth")
PROMPT = "Generate a metric depth visualization of this image."
# Held-out rows keep their frozen split; the #88 base also carries referring/test.
SPLITS = {"train", "val", "test"}


def materialize(path: Path, source: str) -> list[dict]:
    if (path.parent / "BUILD_INCOMPLETE").exists():
        raise ValueError(f"Incomplete dataset: {path.parent}")
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for index, row in enumerate(rows):
        row["task"] = {"metric_depth": "depth"}.get(row.get("task"), row.get("task"))
        if row["task"] not in TASKS or row.get("split") not in SPLITS:
            raise ValueError(f"Unexpected task/split at {path}:{index + 1}")
        if row["task"] == "generation" and not row.get("input"):
            row["input"] = row.get("target")
        for key in PATHS:
            value = row.get(key)
            if not value:
                if key in {"input", "target"}:
                    raise ValueError(f"Missing {key}: {path}:{index + 1}")
                continue
            if not isinstance(value, str):
                raise ValueError(f"Invalid {key}: {path}:{index + 1}")
            asset = resolve_manifest_path(value, path).resolve()
            if not asset.is_file():
                raise FileNotFoundError(asset)
            row[key] = str(asset)
        if row["task"] == "depth" and row.get("prompt") != PROMPT:
            raise ValueError(
                "Normalize depth prompts in the frozen baseline before preparing both arms"
            )
        row["experiment_source"] = source
        # Stable row identity within this explicitly pinned input; no path-dependent hash.
        row["experiment_sample_id"] = f"{source}:{index}"
    return rows


def validate_release(root: Path, *, hash_assets: bool = False) -> list[dict]:
    if sha256_file(root / "pairs.jsonl") != PAIRS_SHA:
        raise ValueError("LingBot pairs do not match the pinned release")
    if sha256_file(root / "assets.jsonl") != ASSETS_SHA:
        raise ValueError("LingBot asset inventory does not match the pinned release")
    build = json.loads((root / "build_report.json").read_text())
    validation = json.loads((root / "validation_report.json").read_text())
    if (
        build.get("dataset_fingerprint") != FINGERPRINT
        or validation.get("dataset_fingerprint") != FINGERPRINT
        or build.get("status") != "complete"
        or validation.get("status") != "passed"
    ):
        raise ValueError("Release build/validation identity mismatch")
    rows = materialize(root / "pairs.jsonl", "lingbot-depth-v1")
    if len(rows) != ROWS or len({r.get("sample_id") for r in rows}) != ROWS:
        raise ValueError("LingBot row count or sample IDs differ")
    for r in rows:
        if any(
            r.get(k) != v
            for k, v in {
                "task": "depth",
                "split": "train",
                "source_dataset": "lingbot-depth",
                "source_subset": "RobbySim",
                "source_view": "rrt_view",
                "depth_units": "metres",
                "depth_convention": "planar_z_depth",
            }.items()
        ) or not r.get("metric_depth"):
            raise ValueError("Unexpected LingBot training contract")
    inventory = read_jsonl(root / "assets.jsonl")
    expected = {str(safe_path(root, x["path"]).resolve()) for x in inventory}
    actual = {r[k] for r in rows for k in ("input", "target", "metric_depth")}
    if len(inventory) != 3 * ROWS or len(expected) != len(inventory) or actual != expected:
        raise ValueError("Asset inventory and manifest differ")
    for item in inventory:
        p = safe_path(root, item["path"])
        if p.stat().st_size != item["bytes"] or (hash_assets and sha256_file(p) != item["sha256"]):
            raise ValueError(f"Changed release asset: {item['path']}")
    return rows


def pixel_hashes(paths, workers: int = 1) -> dict[Path, str]:
    """Decoded-RGB identity of every input image, decoded once (in parallel when workers > 1)."""
    paths = sorted({str(Path(p).resolve()) for p in paths})
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            digests = list(pool.map(decoded_identity, paths, chunksize=64))
    else:
        digests = [decoded_identity(p) for p in paths]
    return {Path(p): d for p, d in zip(paths, digests, strict=True)}


def write_manifest(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "BUILD_INCOMPLETE").write_text("Paired data build is incomplete\n")
    with path.open("x", encoding="utf-8") as stream:
        for original in rows:
            row = dict(original)
            for key in PATHS:
                if row.get(key):
                    try:
                        row[key] = Path(os.path.relpath(row[key], path.parent)).as_posix()
                    except ValueError:  # Different Windows drives.
                        row[key] = Path(row[key]).as_posix()
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "sha256": sha256_file(path),
        "rows": len(rows),
        "task_split_counts": dict(Counter(f"{r['task']}/{r['split']}" for r in rows)),
    }


def prepare(
    base: Path,
    lingbot: Path,
    output: Path,
    baseline_sha256: str,
    *,
    depth_rows: int = 60852,
    replay_rows: int = 16492,
    hash_assets: bool = False,
    workers: int = 1,
) -> dict:
    if output.exists():
        raise FileExistsError("Use a new output directory; existing runs are never replaced")
    if sha256_file(base) != baseline_sha256:
        raise ValueError("Baseline manifest SHA mismatch")
    old = materialize(base, "baseline")
    if any(r.get("source_dataset") == "lingbot-depth" for r in old):
        raise ValueError("Baseline already contains LingBot")
    counts = Counter((r["task"], r["split"]) for r in old)
    if counts["depth", "train"] != depth_rows or counts["generation", "train"] != replay_rows:
        raise ValueError("Baseline training counts differ from the declared full pool")
    if any(counts[t, s] < 1 for t in TASKS for s in ("train", "val")):
        raise ValueError("Baseline must contain all six tasks with frozen train and val splits")
    new = validate_release(lingbot, hash_assets=hash_assets)
    combined = old + new
    # Keys are resolved paths, the same keys the overlap helper looks up.
    cache = pixel_hashes((r["input"] for r in combined), workers)
    # Existing audited helper checks decoded pixels AND COCO IDs across all tasks.
    overlap = assert_no_heldout_overlap(combined, base, pixel_cache=cache)
    old_pixels = {cache[Path(r["input"])] for r in old}
    new_pixels = {cache[Path(r["input"])] for r in new}
    if old_pixels & new_pixels:
        raise ValueError("LingBot duplicates baseline decoded RGB; no automatic dropping")
    if sha256_file(base) != baseline_sha256:
        raise ValueError("Baseline manifest changed during validation")
    # All validation precedes publication of either experiment manifest.
    output.mkdir(parents=True, exist_ok=False)
    marker = output / "BUILD_INCOMPLETE"
    marker.write_text("Paired manifests are incomplete\n")
    report = {
        "status": "complete",
        "lingbot_revision": REVISION,
        "lingbot_fingerprint": FINGERPRINT,
        "baseline_input_sha256": baseline_sha256,
        "lingbot_asset_check": "sha256" if hash_assets else "size (restore already checks hashes)",
        "internal_overlap_audit": overlap,
        "external_benchmark_audit": "not performed here",
        "manifests": {},
    }
    for name, rows in (("full_baseline", old), ("full_plus_lingbot", combined)):
        report["manifests"][name] = write_manifest(output / name / "pairs.jsonl", rows)
    (output / "experiment-data.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in report["manifests"]:
        (output / name / "BUILD_INCOMPLETE").unlink()
    marker.unlink()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--lingbot-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-depth-train", type=int, default=60852)
    p.add_argument("--expected-replay-train", type=int, default=16492)
    p.add_argument("--verify-asset-hashes", action="store_true")
    p.add_argument(
        "--workers", type=int, default=1, help="processes decoding images for the pixel checks"
    )
    a = p.parse_args()
    print(
        json.dumps(
            prepare(
                a.baseline.resolve(),
                a.lingbot_dir.resolve(),
                a.output_dir.resolve(),
                a.baseline_sha256,
                depth_rows=a.expected_depth_train,
                replay_rows=a.expected_replay_train,
                hash_assets=a.verify_asset_hashes,
                workers=a.workers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
