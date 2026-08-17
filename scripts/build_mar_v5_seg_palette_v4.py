#!/usr/bin/env python3
"""Canonicalize Vision-Banana semantic masks for MAR v5 training.

The source masks use a different random RGB color assignment for each image.
``manifest.json`` records the color-to-label legend per sample, which lets this
script recover the stable 150-class label IDs and render them with one fixed,
reversible palette.  The output manifest is a copy of v3 with only the real
semantic-segmentation targets replaced; depth, normal, and structure pairs are
left untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_MANIFEST = Path(
    "/Users/gatilin/.cache/huggingface/hub/"
    "datasets--lllab--lllab-vision-banana-dataset/snapshots/"
    "f074bc5ea17e167d2a91561121f58f5a40eca882/"
    "semantic-segmentation/processed/semantic_all/manifest.json"
)
DEFAULT_PAIRS = PROJECT_ROOT / "output/mar_vb_clean_pairs_v3/pairs.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/mar_vb_clean_seg_palette_v4/targets"
DEFAULT_OUTPUT_MANIFEST = PROJECT_ROOT / "output/mar_vb_clean_pairs_v4/pairs.jsonl"

NUM_CLASSES = 150
IGNORE_LABEL = 255
IGNORE_COLOR = (0, 0, 0)


def pascal_voc_palette(size: int) -> np.ndarray:
    """Return the stable bit-interleaved PASCAL-VOC palette for ``size`` IDs."""
    palette = np.zeros((size, 3), dtype=np.uint8)
    for index in range(size):
        label, bit = index, 0
        while label:
            palette[index, 0] |= ((label >> 0) & 1) << (7 - bit)
            palette[index, 1] |= ((label >> 1) & 1) << (7 - bit)
            palette[index, 2] |= ((label >> 2) & 1) << (7 - bit)
            label >>= 3
            bit += 1
    return palette


# Offset valid labels by one so label 0 never shares black with ignore pixels.
CLASS_PALETTE = pascal_voc_palette(NUM_CLASSES + 1)[1:]


def rgb_key(rgb: list[int] | tuple[int, int, int]) -> int:
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=min(3, os.cpu_count() or 1),
        help="Independent mask conversion workers (default: up to 3).",
    )
    return parser.parse_args()


def load_source_records(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    records: dict[str, dict] = {}
    for record in data.get("records", []):
        name = record.get("input_file")
        if not isinstance(name, str):
            continue
        if name in records:
            raise ValueError(f"duplicate semantic input_file in source manifest: {name}")
        records[name] = record
    if not records:
        raise ValueError(f"no source records found in {path}")
    return records


def label_map(record: dict) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for item in record.get("classes", []):
        color = item.get("rgb")
        label = item.get("label_id")
        if not (isinstance(color, list) and len(color) == 3 and isinstance(label, int)):
            continue
        if label != IGNORE_LABEL and not 0 <= label < NUM_CLASSES:
            raise ValueError(f"unsupported label ID {label} in {record.get('input_file')}")
        key = rgb_key(color)
        if key in mapping and mapping[key] != label:
            raise ValueError(f"ambiguous source RGB color in {record.get('input_file')}")
        mapping[key] = label
    if not mapping:
        raise ValueError(f"no class legend in {record.get('input_file')}")
    return mapping


def remap_target(source: Path, record: dict) -> tuple[np.ndarray, int, Counter]:
    """Recover label IDs from the per-image source legend and recolor them."""
    with Image.open(source) as image:
        raw = np.asarray(image.convert("RGB"), dtype=np.uint8)
    flat = raw.reshape(-1, 3)
    packed = ((flat[:, 0].astype(np.uint32) << 16)
              | (flat[:, 1].astype(np.uint32) << 8)
              | flat[:, 2].astype(np.uint32))
    mapping = label_map(record)
    # Each target has only a few legend colors.  Searching its sorted legend is
    # much faster on large masks than sorting every pixel with ``np.unique``.
    legend_keys = np.asarray(sorted(mapping), dtype=np.uint32)
    legend_labels = np.asarray([mapping[int(key)] for key in legend_keys], dtype=np.int16)
    positions = np.searchsorted(legend_keys, packed)
    in_range = positions < len(legend_keys)
    matched = np.zeros(len(packed), dtype=bool)
    matched[in_range] = legend_keys[positions[in_range]] == packed[in_range]
    unknown = int((~matched).sum())
    if unknown:
        unknown_colors = np.unique(packed[~matched])
        raise ValueError(
            f"{source.name}: {unknown} mask pixel(s) use colors absent from the source legend: "
            f"{', '.join(f'0x{int(color):06x}' for color in unknown_colors[:8])}"
        )
    labels = legend_labels[positions]
    output = np.zeros((len(labels), 3), dtype=np.uint8)
    valid = labels != IGNORE_LABEL
    output[valid] = CLASS_PALETTE[labels[valid]]
    class_ids, class_counts = np.unique(labels, return_counts=True)
    counts = Counter({int(label): int(count) for label, count in zip(class_ids, class_counts)})
    return output.reshape(raw.shape), int(valid.sum()), counts


def write_palette(path: Path, records: dict[str, dict]) -> None:
    class_names: list[str | None] = [None] * NUM_CLASSES
    for record in records.values():
        for item in record.get("classes", []):
            label = item.get("label_id")
            name = item.get("canonical_name")
            if isinstance(label, int) and 0 <= label < NUM_CLASSES and isinstance(name, str):
                if class_names[label] is None:
                    class_names[label] = name
                elif class_names[label] != name:
                    raise ValueError(f"label {label} has inconsistent names: "
                                     f"{class_names[label]!r} vs {name!r}")
    path.write_text(json.dumps({
        "format": "mar_vb_semantic_fixed_palette_v4",
        "num_classes": NUM_CLASSES,
        "class_palette": CLASS_PALETTE.tolist(),
        "class_names": class_names,
        "ignore_label": IGNORE_LABEL,
        "ignore_color": list(IGNORE_COLOR),
        "label_encoding": "class_palette[label_id]; ignore_color for label_id=255",
    }, ensure_ascii=False, indent=2) + "\n")


def process_target(payload: tuple[str, dict, str, bool]) -> tuple[bool, int, dict[int, int]]:
    """Convert one target in a worker process and return compact audit data."""
    source_text, record, destination_text, overwrite = payload
    source, destination = Path(source_text), Path(destination_text)
    if destination.exists() and not overwrite:
        try:
            with Image.open(destination) as existing:
                existing.verify()
            return False, 0, {}
        except OSError:
            # A cancelled process can leave a partial file; rebuild it below.
            pass
    remapped, valid_pixels, counts = remap_target(source, record)
    # Fixed-palette masks are sparse-color PNGs; level 1 is lossless and avoids
    # spending most conversion time on unnecessary compression.
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    Image.fromarray(remapped, mode="RGB").save(
        temporary, format="PNG", compress_level=1,
    )
    temporary.replace(destination)
    return True, valid_pixels, dict(counts)


def main() -> None:
    args = parse_args()
    for path, desc in ((args.source_manifest, "source manifest"), (args.pairs, "v3 pairs")):
        if not path.exists():
            raise SystemExit(f"{desc} not found: {path}")

    source_records = load_source_records(args.source_manifest)
    rows = [json.loads(line) for line in args.pairs.open() if line.strip()]
    semantic_rows = [
        row for row in rows
        if row.get("task") == "seg"
        and row.get("target_kind") == "vision_banana_semantic_rgb"
    ]
    if not semantic_rows:
        raise SystemExit("no Vision-Banana semantic rows found in the input pairs manifest")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    palette_path = args.output_dir.parent / "palette.json"
    write_palette(palette_path, source_records)
    palette_path = palette_path.resolve()

    audit = Counter()
    label_pixels: Counter = Counter()
    fixed_targets: dict[str, Path] = {}
    jobs: list[tuple[str, dict, str, bool]] = []
    job_meta: list[tuple[str, Path]] = []
    for row in semantic_rows:
        source = Path(row["target"])
        input_name = Path(row["input"]).name
        record = source_records.get(input_name)
        if record is None:
            raise ValueError(f"source manifest has no record for {input_name}")
        if not source.exists():
            raise FileNotFoundError(f"semantic source target is missing: {source}")
        destination = args.output_dir / source.name
        jobs.append((str(source), record, str(destination), args.overwrite))
        job_meta.append((str(source), destination.resolve()))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(process_target, jobs, chunksize=1)
        for index, ((source, destination), (written, valid_pixels, counts)) in enumerate(
                zip(job_meta, results), 1):
            if written:
                audit["targets_written"] += 1
                audit["valid_pixels"] += valid_pixels
                label_pixels.update(counts)
            else:
                audit["existing_targets_reused"] += 1
            fixed_targets[source] = destination
            if index % 500 == 0 or index == len(semantic_rows):
                print(f"[{index}/{len(semantic_rows)}] fixed semantic targets ready", flush=True)

    manifest_rows = []
    for row in rows:
        replacement = fixed_targets.get(str(Path(row.get("target", ""))))
        if replacement is not None:
            row = dict(row)
            row["target"] = str(replacement)
            row["target_kind"] = "vision_banana_semantic_fixed_palette_v4"
            row["segmentation_palette"] = str(palette_path)
            row["segmentation_ignore_label"] = IGNORE_LABEL
        manifest_rows.append(row)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter((row["task"], row["split"]) for row in manifest_rows)
    report = {
        "source_manifest": str(args.source_manifest.resolve()),
        "input_pairs": str(args.pairs.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "target_dir": str(args.output_dir.resolve()),
        "palette": str(palette_path),
        "semantic_pairs": len(semantic_rows),
        "all_pair_counts": {f"{task}/{split}": count for (task, split), count in sorted(counts.items())},
        "processing": dict(audit),
        "label_pixels_written": {str(label): count for label, count in sorted(label_pixels.items())},
        "ignore_label": IGNORE_LABEL,
    }
    audit_path = args.output_manifest.with_suffix(".semantic_palette.audit.json")
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
