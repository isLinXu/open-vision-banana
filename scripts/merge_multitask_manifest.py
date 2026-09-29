#!/usr/bin/env python3
"""Merge per-task pair files into one validated multi-task manifest.

Each input row keeps its ``task`` label; a row is only emitted when BOTH its
``input`` and ``target`` files exist on disk (this is what silently dropped the
4,700 ``structure`` pairs from the last FLUX run — the trainer skips missing
files at load time, so we surface the loss here instead).  An optional
per-task cap balances the sampling distribution.

Usage:
    python scripts/merge_multitask_manifest.py \
        --inputs output/snvc_v5full_pairs/pairs_3task.jsonl \
                 output/snvc_v5full_pairs/pairs_detection.jsonl \
                 output/snvc_v5full_pairs/pairs_keypoint.jsonl \
                 output/snvc_v5full_pairs/pairs_refer.jsonl \
        --out output/snvc_v5full_pairs/pairs_multitask6.jsonl \
        --cap 6000
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cap", type=int, default=0,
                        help="max rows per task (0 = no cap); applied after validation")
    parser.add_argument("--val-percent", type=int, default=5)
    args = parser.parse_args()

    by_task: dict[str, list[dict]] = defaultdict(list)
    stats: dict[str, Counter] = defaultdict(Counter)

    for src in args.inputs:
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
                inp = Path(row.get("input", ""))
                tgt = Path(row.get("target", ""))
                if not inp.is_file():
                    stats[task]["input_missing"] += 1
                    continue
                if not tgt.is_file():
                    stats[task]["target_missing"] += 1
                    continue
                by_task[task].append(row)
                stats[task]["valid"] += 1

    total_before = 0
    total_after = 0
    out_rows: list[dict] = []
    for task in sorted(by_task):
        rows = sorted(by_task[task], key=lambda r: r["input"])
        total_before += len(rows)
        if args.cap and len(rows) > args.cap:
            step = len(rows) / args.cap  # even strided subsample
            rows = [rows[int(i * step)] for i in range(args.cap)]
        total_after += len(rows)
        out_rows.extend(rows)

    out_rows.sort(key=lambda r: (r["task"], r["input"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "out": str(args.out.resolve()),
        "per_task_valid": {t: int(stats[t]["valid"]) for t in stats},
        "per_task_dropped": {
            t: {k: v for k, v in stats[t].items() if k != "valid"}
            for t in stats if any(k != "valid" for k, v in stats[t].items())
        },
        "per_task_emitted": dict(Counter(r["task"] for r in out_rows)),
        "total_valid": total_before,
        "total_emitted": total_after,
        "cap": args.cap,
    }
    args.out.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
