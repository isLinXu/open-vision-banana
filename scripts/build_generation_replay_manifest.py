#!/usr/bin/env python3
"""Build a ``generation`` replay manifest for generation-preservation mixing.

The paper instruction-tunes on the base model's own generation/editing data plus a
small share of vision-task data (Section 2); fine-tuning on vision data alone is
what erodes generation ability. ``TrainingConfig.vision_ratio < 1`` samples rows
with ``task: generation`` for the remaining share. This script turns a captioned
image set or an instruction-editing pair set into those rows and appends them to
(or writes) a pairs.jsonl.

Two row kinds::

    --kind t2i    {"task": "generation", "target": img, "prompt": caption}
                  Noise -> target; the trainer uses the target as the (unused)
                  source. Only meaningful for text-to-image backbones (SANA,
                  FLUX); the trainer rejects it for instruction editors.
    --kind edit   {"task": "generation", "input": src, "target": edited,
                   "prompt": instruction}
                  Source + instruction -> edited target; the right replay for
                  Qwen-Image-Edit / Qwen-Image-2.1 / Kontext-style editors.

Inputs are a JSONL or CSV whose columns are selected with ``--image-key`` /
``--caption-key`` (t2i) or ``--source-key`` / ``--target-key`` /
``--instruction-key`` (edit). Paths are kept relative to ``--data-root``.

Example::

    python scripts/build_generation_replay_manifest.py \
        --kind edit --records /data/imgedit/train.jsonl --data-root /data \
        --source-key input_image --target-key output_image --instruction-key edit \
        --val-fraction 0.02 --output /data/qwen_baseline/generation_replay.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _relative(path_str: str, data_root: Path | None) -> str:
    path = Path(path_str)
    if data_root is not None and path.is_absolute():
        try:
            return str(path.resolve().relative_to(data_root.resolve()))
        except ValueError:
            return str(path)
    return str(path)


def build_generation_replay_manifest(
    records_path: Path,
    output: Path,
    *,
    kind: str,
    data_root: Path | None = None,
    image_key: str = "image",
    caption_key: str = "caption",
    source_key: str = "source",
    target_key: str = "target",
    instruction_key: str = "instruction",
    val_fraction: float = 0.0,
    max_rows: int | None = None,
    seed: int = 42,
    append: bool = False,
    require_files: bool = True,
) -> dict:
    """Write generation rows and return an audit summary."""
    if kind not in {"t2i", "edit"}:
        raise ValueError("kind must be 't2i' or 'edit'")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    anchor = data_root.resolve() if data_root else records_path.resolve().parent

    rows: list[dict] = []
    skipped_missing_text = 0
    skipped_missing_file = 0
    skipped_identity = 0
    for record in load_records(records_path):
        if kind == "t2i":
            image, text = record.get(image_key), record.get(caption_key)
            if not image or not isinstance(text, str) or not text.strip():
                skipped_missing_text += 1
                continue
            paths = {"target": _relative(str(image), data_root)}
        else:
            source, target, text = (
                record.get(source_key), record.get(target_key), record.get(instruction_key)
            )
            if not source or not target or not isinstance(text, str) or not text.strip():
                skipped_missing_text += 1
                continue
            paths = {
                "input": _relative(str(source), data_root),
                "target": _relative(str(target), data_root),
            }
            if paths["input"] == paths["target"]:
                skipped_identity += 1
                continue
        if require_files and any(
            not (anchor / p).is_file() and not Path(p).is_file() for p in paths.values()
        ):
            skipped_missing_file += 1
            continue
        rows.append({"task": "generation", **paths, "prompt": text.strip()})

    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_rows is not None:
        rows = rows[:max_rows]
    n_val = int(round(len(rows) * val_fraction))
    for index, row in enumerate(rows):
        row["split"] = "val" if index < n_val else "train"
    # Keep validation rows first so a truncated manifest still has a val split.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a" if append else "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "output_manifest": str(output.resolve()),
        "kind": kind,
        "rows_written": len(rows),
        "train_rows": len(rows) - n_val,
        "val_rows": n_val,
        "skipped_missing_text": skipped_missing_text,
        "skipped_missing_file": skipped_missing_file,
        "skipped_identity_pairs": skipped_identity,
        "appended": append,
    }
    output.with_suffix(".generation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kind", choices=("t2i", "edit"), required=True)
    parser.add_argument("--records", type=Path, required=True, help="JSONL or CSV of records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--caption-key", default="caption")
    parser.add_argument("--source-key", default="source")
    parser.add_argument("--target-key", default="target")
    parser.add_argument("--instruction-key", default="instruction")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true", help="Append to an existing pairs.jsonl")
    parser.add_argument(
        "--no-require-files", action="store_true", help="Do not check that image files exist"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_generation_replay_manifest(
        args.records,
        args.output,
        kind=args.kind,
        data_root=args.data_root,
        image_key=args.image_key,
        caption_key=args.caption_key,
        source_key=args.source_key,
        target_key=args.target_key,
        instruction_key=args.instruction_key,
        val_fraction=args.val_fraction,
        max_rows=args.max_rows,
        seed=args.seed,
        append=args.append,
        require_files=not args.no_require_files,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
