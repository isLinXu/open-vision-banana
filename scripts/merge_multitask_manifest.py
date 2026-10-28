#!/usr/bin/env python3
"""Merge per-task pair files into one validated multi-task manifest.

Each input row keeps its ``task`` label; a row is only emitted when BOTH its
``input`` and ``target`` files exist on disk (this is what silently dropped the
4,700 ``structure`` pairs from the last FLUX run — the trainer skips missing
files at load time, so we surface the loss here instead).

Paths: every preparation script writes paths relative to *its own* manifest
directory (the ``resolve_manifest_path`` convention). This merger resolves each
row against the manifest it came from and rewrites the result relative to the
output manifest's directory (or absolute with ``--absolute``), so the merged file
loads from anywhere. An optional per-task cap balances the sampling
distribution, and ``--val-percent`` gives every task a validation slice when a
source has none (the trainer requires either no validation split at all or one
for every task).

Usage:
    python scripts/merge_multitask_manifest.py \
        --inputs data/hypersim/pairs.jsonl data/instance_coco/instance.jsonl \
                 data/res/pairs.jsonl data/replay/generation.jsonl \
        --out data/combined/pairs.jsonl --cap 20000 --val-percent 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.manifest import resolve_manifest_path  # noqa: E402

PATH_KEYS = ("input", "target", "image", "metric_depth")


_COCO_ID_RE = re.compile(r"^(?:COCO_(?:train|val|test)20\d\d_)?0*(\d{1,12})\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _image_hash(rel_input: str, out_dir: Path) -> int:
    """Stable per-image hash used as the split key.

    COCO-style file names hash by their image id (so copies of the same COCO image
    under different directories share a split); everything else by resolved path.
    """
    name = Path(rel_input).name
    match = _COCO_ID_RE.match(name)
    key = f"coco:{int(match.group(1))}" if match else str(
        (out_dir / rel_input).resolve() if not Path(rel_input).is_absolute() else Path(rel_input)
    )
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:12], 16)


def _rebase(path: Path, out_dir: Path, absolute: bool) -> str:
    path = path.resolve()
    if absolute:
        return str(path)
    return Path(os.path.relpath(path, out_dir.resolve())).as_posix()


def merge_manifests(
    inputs: list[Path],
    out: Path,
    *,
    cap: int = 0,
    val_percent: float = 5.0,
    absolute: bool = False,
    seed: int = 42,
) -> dict:
    """Merge, validate and rebase manifests; return the audit summary."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    stats: dict[str, Counter] = defaultdict(Counter)
    out_dir = out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in inputs:
        if not src.is_file():
            print(f"[warn] missing input {src}", flush=True)
            continue
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                task = row.get("task", "?")
                # generation rows may carry only a target (T2I) -> mirror it into input
                if task == "generation" and not row.get("input") and row.get("target"):
                    row["input"] = row["target"]
                missing = False
                for key in PATH_KEYS:
                    value = row.get(key)
                    if not isinstance(value, str) or not value:
                        if key in ("input", "target"):
                            stats[task][f"{key}_missing"] += 1
                            missing = True
                        continue
                    resolved = resolve_manifest_path(value, src)
                    if not resolved.is_file():
                        stats[task][f"{key}_missing"] += 1
                        missing = True
                        break
                    row[key] = _rebase(resolved, out_dir, absolute)
                if missing:
                    continue
                row["source_manifest"] = src.name
                by_task[task].append(row)
                stats[task]["valid"] += 1

    out_rows: list[dict] = []
    total_before = total_after = 0
    for task in sorted(by_task):
        rows = sorted(by_task[task], key=lambda r: r["input"])
        total_before += len(rows)
        if cap and len(rows) > cap:
            step = len(rows) / cap  # even strided subsample
            rows = [rows[int(i * step)] for i in range(cap)]
        if val_percent > 0 and not any(r.get("split") == "val" for r in rows):
            # Image-grouped, deterministic: every row of the same input image (across tasks and
            # instance classes) lands in the same split, so val loss never sees a training image.
            n_val = 0
            if len(rows) > 1:
                keyed = sorted(rows, key=lambda r: _image_hash(r["input"], out_dir))
                for row in keyed:
                    if _image_hash(row["input"], out_dir) % 10_000 < int(val_percent * 100):
                        row["split"] = "val"
                        n_val += 1
                if n_val == 0:  # tiny tasks: guarantee one val image
                    keyed[0]["split"] = "val"
                    n_val = 1
                if n_val == len(rows):  # ...and one train image
                    keyed[-1]["split"] = "train"
                    n_val -= 1
            stats[task]["val_assigned"] = n_val
        for row in rows:
            row.setdefault("split", "train")
        total_after += len(rows)
        out_rows.extend(rows)

    out_rows.sort(key=lambda r: (r["task"], r["input"]))
    with out.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "out": str(out.resolve()),
        "path_policy": "absolute" if absolute else "relative-to-output-manifest",
        "per_task_valid": {t: int(stats[t]["valid"]) for t in stats},
        "per_task_dropped": {
            t: {k: v for k, v in stats[t].items() if k not in ("valid", "val_assigned")}
            for t in stats
            if any(k not in ("valid", "val_assigned") for k in stats[t])
        },
        "per_task_val_assigned": {t: int(stats[t]["val_assigned"]) for t in stats if stats[t]["val_assigned"]},
        "per_task_emitted": dict(Counter(r["task"] for r in out_rows)),
        "per_task_split": {
            t: dict(Counter(r["split"] for r in out_rows if r["task"] == t)) for t in sorted(by_task)
        },
        "total_valid": total_before,
        "total_emitted": total_after,
        "cap": cap,
    }
    out.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cap", type=int, default=0, help="max rows per task (0 = no cap); applied after validation")
    parser.add_argument("--val-percent", type=float, default=5.0, help="val share for tasks that have no val rows")
    parser.add_argument("--absolute", action="store_true", help="write absolute paths instead of output-relative")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = merge_manifests(
        args.inputs, args.out, cap=args.cap, val_percent=args.val_percent, absolute=args.absolute, seed=args.seed
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
