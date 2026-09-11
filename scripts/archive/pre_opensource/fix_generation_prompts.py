#!/usr/bin/env python3
"""Fix P0-1: backfill real COCO captions into generation prompts.jsonl.

Root cause: prepare_zimage_data.py failed to locate captions_train2017.json
(glob mismatch) and silently fell back to the generic prompt "a photograph"
for all 18,000 samples. The true captions already exist in targets/*.txt
(one caption per sample), but GenerationDataset only reads prompts.jsonl.

This script rewrites prompts.jsonl in place, mapping each image to its
caption via the numeric suffix (coco_gen_00042.jpg -> gen_00042.txt).
Original prompts.jsonl is backed up to prompts.jsonl.bak before rewriting.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GENERATION_DIRS = [
    PROJECT_ROOT / "data" / "vb_zimage_train" / "generation",
    PROJECT_ROOT / "data" / "vb_snvc_train" / "generation",
]


def find_caption(txt_index: dict[str, Path], image_name: str) -> str | None:
    """Resolve caption text for an image filename via numeric suffix."""
    m = re.search(r"(\d+)\.[a-zA-Z]+$", image_name)
    if not m:
        return None
    num = m.group(1)
    for cand in (f"gen_{num}", f"coco_gen_{num}", Path(image_name).stem):
        p = txt_index.get(cand)
        if p is not None:
            return p.read_text(encoding="utf-8").strip()
    return None


def fix_dir(gen_dir: Path) -> tuple[int, int, int]:
    prompts_file = gen_dir / "prompts.jsonl"
    targets_dir = gen_dir / "targets"
    if not prompts_file.exists() or not targets_dir.exists():
        print(f"[skip] {gen_dir}: missing prompts.jsonl or targets/")
        return 0, 0, 0

    txt_index = {p.stem: p for p in targets_dir.glob("*.txt")}

    entries = [json.loads(line) for line in prompts_file.read_text().splitlines() if line.strip()]
    total = len(entries)

    # Backup once
    backup = prompts_file.with_suffix(".jsonl.bak")
    if not backup.exists():
        shutil.copy2(prompts_file, backup)

    fixed = 0
    still_generic = 0
    out_lines = []
    for e in entries:
        cap = find_caption(txt_index, e["image"])
        if cap:
            e["prompt"] = cap
            fixed += 1
        else:
            still_generic += 1
        out_lines.append(json.dumps(e, ensure_ascii=False))

    prompts_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[done] {gen_dir.name}: total={total}, fixed={fixed}, no_caption={still_generic}")
    return total, fixed, still_generic


def main() -> int:
    grand_total = grand_fixed = 0
    for d in GENERATION_DIRS:
        total, fixed, _ = fix_dir(d)
        grand_total += total
        grand_fixed += fixed

    print(f"\nOverall: {grand_fixed}/{grand_total} prompts backfilled with real captions")

    # Verify uniqueness on the training-relevant prefix (first 3000 used by training)
    for d in GENERATION_DIRS:
        pf = d / "prompts.jsonl"
        if not pf.exists():
            continue
        prompts = [json.loads(l)["prompt"] for l in pf.read_text().splitlines()[:3000]]
        uniq = len(set(prompts))
        print(f"  {d.parent.name}/generation first-3000 unique prompts: {uniq}")
        print(f"    sample: {prompts[0][:90]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
