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

Inputs are selected with ``--image-key`` / ``--caption-key`` (t2i) or
``--source-key`` / ``--target-key`` / ``--instruction-key`` (edit) and may be:

* a JSONL or CSV of records whose image fields are file paths;
* one or more **parquet** files / a directory of shards (Hugging Face ``Image``
  columns hold ``{"bytes", "path"}``; the bytes are written to ``--images-dir``);
* a **WebDataset tar** (``<key>.jpg`` + ``<key>.txt`` / ``.json`` members), or a
  tar / directory of images with a sidecar ``--captions-file`` (one caption per
  line, indexed by the numeric image stem; BLIP3o-60k layout).

``--preset omniedit`` sets the OmniEdit column names (``src_img`` / ``edited_img`` /
``edited_prompt_list``). List-valued instruction fields use their first entry.
Paths are kept relative to ``--data-root``.

Example::

    python scripts/build_generation_replay_manifest.py \
        --kind edit --preset omniedit --records data/raw/omniedit/data --data-root data \
        --images-dir data/replay/omniedit_images --val-fraction 0.02 \
        --output data/replay/omniedit.jsonl

    python scripts/build_generation_replay_manifest.py \
        --kind t2i --records data/raw/blip3o_60k/journeyDB.tar \
        --captions-file data/raw/blip3o_60k/journeyDB.txt --data-root data \
        --images-dir data/replay/blip3o_images --output data/replay/blip3o.jsonl
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.manifest import relative_manifest_path  # noqa: E402

PRESETS = {
    "omniedit": {"source_key": "src_img", "target_key": "edited_img", "instruction_key": "edited_prompt_list"},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TEXT_SUFFIXES = {".txt", ".caption"}


def _parquet_rows(paths: list[Path]) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("reading parquet needs pyarrow: pip install pyarrow") from exc
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()


def _tar_rows(path: Path, captions: dict[str, str] | None) -> Iterator[dict]:
    """WebDataset grouping: members sharing a stem form one record (image + caption)."""
    pending: dict[str, dict] = {}
    with tarfile.open(path, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name)
            stem, suffix = name.stem, name.suffix.lower()
            record = pending.setdefault(stem, {"__key__": stem})
            data = tar.extractfile(member)
            if data is None:
                continue
            payload = data.read()
            if suffix in IMAGE_SUFFIXES:
                record["image"] = {"bytes": payload, "path": name.name}
            elif suffix in TEXT_SUFFIXES:
                record["caption"] = payload.decode("utf-8", "replace").strip()
            elif suffix == ".json":
                try:
                    meta = json.loads(payload.decode("utf-8"))
                except json.JSONDecodeError:
                    meta = {}
                if isinstance(meta, dict):
                    record.update({k: v for k, v in meta.items() if k not in record})
            if "image" in record and ("caption" in record or captions is not None):
                if captions is not None and "caption" not in record:
                    record["caption"] = captions.get(stem, "")
                yield pending.pop(stem)
    for record in pending.values():  # images whose caption member never arrived
        if captions is not None and "caption" not in record:
            record["caption"] = captions.get(record["__key__"], "")
        yield record


def _dir_rows(path: Path, captions: dict[str, str] | None) -> Iterator[dict]:
    for image in sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        caption = None
        sidecar = image.with_suffix(".txt")
        if sidecar.is_file():
            caption = sidecar.read_text(encoding="utf-8", errors="replace").strip()
        elif captions is not None:
            caption = captions.get(image.stem, "")
        yield {"image": str(image), "caption": caption}


def load_captions(path: Path | None, index_base: int = 0) -> dict[str, str] | None:
    """One caption per line -> {str(index_base + line_no): caption} (BLIP3o-60k sidecar)."""
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {str(index_base + i): line.strip() for i, line in enumerate(lines)}


def iter_records(path: Path, *, captions_file: Path | None = None, caption_index_base: int = 0) -> Iterator[dict]:
    """Yield records from JSONL / CSV / parquet (file or dir) / WebDataset tar / image dir."""
    captions = load_captions(captions_file, caption_index_base)
    if path.is_dir():
        shards = sorted(path.rglob("*.parquet"))
        if shards:
            yield from _parquet_rows(shards)
        else:
            yield from _dir_rows(path, captions)
        return
    suffix = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".parquet":
        yield from _parquet_rows([path])
    elif suffix.endswith((".tar", ".tar.gz", ".tgz")):
        yield from _tar_rows(path, captions)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def load_records(path: Path) -> list[dict]:
    return list(iter_records(path))


def _first_text(value: Any) -> str | None:
    """Instruction/caption fields may be strings or lists of strings."""
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
    return None


def _materialize(value: Any, images_dir: Path | None, stem: str) -> str | None:
    """A path string, or embedded bytes (HF Image struct / raw bytes / PIL) written to images_dir."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    payload: bytes | None = None
    original_name = None
    if isinstance(value, dict):
        payload = value.get("bytes")
        original_name = value.get("path")
        if payload is None and isinstance(original_name, str):
            return original_name
    elif isinstance(value, (bytes, bytearray)):
        payload = bytes(value)
    elif hasattr(value, "save"):  # PIL image
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        payload = buffer.getvalue()
    if payload is None:
        return None
    if images_dir is None:
        raise ValueError("records embed image bytes; pass --images-dir to materialize them")
    suffix = Path(original_name).suffix.lower() if isinstance(original_name, str) and Path(original_name).suffix else ""
    if suffix not in IMAGE_SUFFIXES:
        suffix = ".png" if payload[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    images_dir.mkdir(parents=True, exist_ok=True)
    out = images_dir / f"{stem}{suffix}"
    if not out.exists():
        out.write_bytes(payload)
    # absolute, so _relative rewrites it against --data-root whatever the cwd-relative --images-dir
    return os.path.abspath(out)


def _relative(path_str: str, data_root: Path | None) -> str:
    """Absolute paths (files this script wrote, absolute records) relative to ``data_root``.

    A relative path from a record is already relative to the records' anchor and is kept.
    """
    path = Path(path_str)
    if data_root is None or not path.is_absolute():
        return path_str
    return relative_manifest_path(path, data_root)


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
    images_dir: Path | None = None,
    captions_file: Path | None = None,
    caption_index_base: int = 0,
    preset: str | None = None,
) -> dict:
    """Write generation rows and return an audit summary."""
    if kind not in {"t2i", "edit"}:
        raise ValueError("kind must be 't2i' or 'edit'")
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
        source_key = PRESETS[preset].get("source_key", source_key)
        target_key = PRESETS[preset].get("target_key", target_key)
        instruction_key = PRESETS[preset].get("instruction_key", instruction_key)
        image_key = PRESETS[preset].get("image_key", image_key)
        caption_key = PRESETS[preset].get("caption_key", caption_key)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    anchor = data_root.resolve() if data_root else records_path.resolve().parent

    rows: list[dict] = []
    skipped_missing_text = 0
    skipped_missing_file = 0
    skipped_identity = 0
    for index, record in enumerate(
        iter_records(records_path, captions_file=captions_file, caption_index_base=caption_index_base)
    ):
        key = str(record.get("__key__") or record.get("omni_edit_id") or record.get("id") or f"{index:08d}")
        if kind == "t2i":
            text = _first_text(record.get(caption_key))
            image = _materialize(record.get(image_key), images_dir, key)
            if not image or text is None:
                skipped_missing_text += 1
                continue
            paths = {"target": _relative(str(image), data_root)}
        else:
            text = _first_text(record.get(instruction_key))
            source = _materialize(record.get(source_key), images_dir, f"{key}_src")
            target = _materialize(record.get(target_key), images_dir, f"{key}_edit")
            if not source or not target or text is None:
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
        "preset": preset,
        "images_dir": str(images_dir) if images_dir else None,
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
    parser.add_argument(
        "--records", type=Path, required=True,
        help="JSONL / CSV / parquet file or dir of shards / WebDataset tar / image dir",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="column-name preset")
    parser.add_argument("--images-dir", type=Path, default=None, help="where embedded image bytes are written")
    parser.add_argument("--captions-file", type=Path, default=None, help="one caption per line, keyed by numeric stem")
    parser.add_argument("--caption-index-base", type=int, default=0, help="stem of the first caption line (0 or 1)")
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
        images_dir=args.images_dir,
        captions_file=args.captions_file,
        caption_index_base=args.caption_index_base,
        preset=args.preset,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
