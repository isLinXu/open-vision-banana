#!/usr/bin/env python3
"""Issue #144: turn segmentation sources of the isLinXu/open-vision-banana ModelScope corpus into OVB rows.

Subcommands (each writes into its own --output-dir, paths in pairs.jsonl are relative to it):

  coconut  COCONut panoptic genseg masks (coconut_b, coconut_l) -> semantic + instance rows, through the same
           coconut_preparation policy (and the same worker) as prepare_coconut.py, which built the pool's
           COCONut-B component. --skip-images-of drops the images another manifest already holds (cross-source
           dedup against the existing pool).
  gcg      PSG grounded captions ("<p>phrase<color>(r, g, b)</color></p>") -> binary referring rows
           (binary_white_black_v1, query_v2 prompt). A phrase whose category occurs more than once in the
           image (person-1, person-2) does not identify one region and is dropped.
  audit    perceptual-hash leakage audit of converted inputs against evaluation images (64-bit DCT pHash,
           Hamming distance <= --max-distance); writes the leaked inputs to drop.
  compose  base manifest + converted train rows (absolute paths, audited leaks removed) -> one manifest.
           Converted val rows are not added, so the validation set stays the base one.

Evaluation images only ever reach this script through --exclude-image-ids and the audit; they are never
converted or added.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from open_vision_banana.coconut_preparation import LEGEND, image_split  # noqa: E402
from open_vision_banana.heldout import coco_image_id_from_path  # noqa: E402
from open_vision_banana.referring_contract import (  # noqa: E402
    BINARY_REFERRING_ENCODING,
    build_binary_referring_prompt,
)
from open_vision_banana.res_bundle import safe_path, sha256_file  # noqa: E402

SEGMENTATION = "training/cleaned_manifests/segmentation"
COCONUT_MANIFEST = SEGMENTATION + "/{subset}_total_unified_genseg_edit_regenerated.jsonl"
GCG_MANIFEST = SEGMENTATION + "/PSG_total_unified_gcg_edit_regenerated.jsonl"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def dump(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8"
    )


def prepare_output(root: Path):
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Output must be empty: {root}")
    for name in ("images", "targets"):
        (root / name).mkdir(parents=True, exist_ok=True)


def file_id(image_id) -> str:
    return (
        f"{image_id:012d}"
        if isinstance(image_id, int)
        else re.sub(r"[^A-Za-z0-9_.-]", "_", str(image_id))
    )


def pool_map(fn, items, workers, chunksize):
    """ProcessPoolExecutor.map, or the builtin map for one worker (no pickling, e.g. in tests)."""
    if workers <= 1:
        yield from map(fn, items)
        return
    with ProcessPoolExecutor(workers) as pool:
        yield from pool.map(fn, items, chunksize=chunksize)


# ── coconut ───────────────────────────────────────────────────────────────────


def _prepare_coconut_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_coconut", ROOT / "scripts/prepare_coconut.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # pool workers unpickle prepare_coconut.work by module name
    spec.loader.exec_module(module)
    return module


def relabel_coconut_rows(rows, subset: str):
    """The shared worker names every row after coconut_b; name it after the converted subset."""
    for row in rows:
        row["sample_id"] = row["sample_id"].replace("coconut_b:", f"{subset}:", 1)
        row["source_dataset"] = subset
        row["source_revision"] = "modelscope:isLinXu/open-vision-banana"
    return rows


def coconut(args):
    out = args.output_dir
    prepare_output(out)
    worker = _prepare_coconut_module().work
    exclude = (
        set(json.loads(args.exclude_image_ids.read_text())) if args.exclude_image_ids else set()
    )
    skip = set()
    for manifest in args.skip_images_of:
        skip |= {r["source_image_id"] for r in read_jsonl(manifest) if "source_image_id" in r}
    records, rejected, seen = [], [], set()
    for row in read_jsonl(args.source_root / COCONUT_MANIFEST.format(subset=args.subset)):
        idx = row["image_id"]
        if idx in seen:
            raise ValueError(f"Duplicate source image id {idx}")
        seen.add(idx)
        coco_id = coco_image_id_from_path(row["image"])
        if coco_id is not None and coco_id in exclude:
            rejected.append({"id": idx, "reason": "heldout_image_id"})
        elif idx in skip:
            rejected.append({"id": idx, "reason": "already_in_reference_pool"})
        elif (
            not (args.media_root / row["image"]).is_file()
            or not (args.media_root / row["seg"]).is_file()
        ):
            rejected.append({"id": idx, "reason": "media_not_available"})
        else:
            records.append(row)
    records.sort(key=lambda r: str(r["image_id"]))
    if args.max_images:
        records = records[: args.max_images]
    counts, hashes = Counter(), set()
    jobs = ((r, str(args.media_root), str(out), {}, args.val_fraction) for r in records)
    with (
        ProcessPoolExecutor(args.workers) as pool,
        (out / "pairs.jsonl").open("w", encoding="utf-8") as fh,
    ):
        for result in pool.map(worker, jobs, chunksize=16):
            rows = relabel_coconut_rows(result.pop("rows", []), args.subset)
            if "error" in result:
                rejected.append(
                    {"id": result["id"], "reason": "invalid_source", "detail": result["error"]}
                )
                continue
            if result["pixel_hash"] in hashes:
                rejected.append({"id": result["id"], "reason": "duplicate_decoded_rgb"})
                continue
            hashes.add(result["pixel_hash"])
            if not rows:
                rejected.append({"id": result["id"], "reason": "no_eligible_category"})
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"{row['task']}/{row['split']}"] += 1
            counts["source_images_retained"] += bool(rows)
    report = dict(
        subset=args.subset,
        source_rows=len(seen),
        converted_images=len(records),
        counts=counts,
        rejections=Counter(r["reason"] for r in rejected),
    )
    dump(out / "rejected.json", rejected)
    dump(out / "build_report.json", report)
    print(json.dumps(report, default=dict), flush=True)


# ── gcg (PSG grounded captions -> referring) ──────────────────────────────────


def parse_gcg_phrases(row):
    """(phrase, rgb) per colored phrase of the single source answer."""
    answers = [c["value"] for c in row["conversations"] if c["from"] == "gpt"]
    if len(answers) != 1:
        raise ValueError("expected one source answer")
    matches = LEGEND.findall(answers[0])
    if not matches or len(matches) != answers[0].count("<color>"):
        raise ValueError("malformed source legend")
    phrases = [(label.strip(), (int(r), int(g), int(b))) for label, r, g, b in matches]
    if len({p for p, _ in phrases}) != len(phrases) or len({c for _, c in phrases}) != len(phrases):
        raise ValueError("duplicate phrase/color")
    if any(c == (0, 0, 0) for _, c in phrases):
        raise ValueError("black legend color")
    return phrases


def unique_phrases(phrases):
    """Keep phrases whose category (the label without an -N instance suffix) occurs once."""
    base = [re.sub(r"-\d+$", "", p) for p, _ in phrases]
    count = Counter(base)
    return [(b, rgb) for b, (_, rgb) in zip(base, phrases, strict=True) if count[b] == 1]


def gcg_work(job):
    row, media, out, val_fraction, max_phrases, min_pixels, min_fraction = job
    idx = coco_image_id_from_path(row["image"])
    try:
        if idx is None:
            raise ValueError("not a COCO image path")
        source_image = safe_path(Path(media), row["image"])
        with Image.open(source_image) as im:
            image = im.convert("RGB")
        digest = hashlib.sha256(
            f"{image.width}x{image.height}:".encode() + image.tobytes()
        ).hexdigest()
        with Image.open(safe_path(Path(media), row["seg"])) as im:
            if im.format != "PNG" or im.size != image.size:
                raise ValueError("mask must be a same-size PNG")
            rgb = np.array(im.convert("RGB"))
        phrases = parse_gcg_phrases(row)
        packed = (
            (rgb[..., 0].astype(np.uint32) << 16)
            | (rgb[..., 1].astype(np.uint32) << 8)
            | rgb[..., 2]
        )
        values, areas = np.unique(packed, return_counts=True)
        observed = dict(zip(map(int, values), map(int, areas), strict=True))
        keys = {(r << 16) | (g << 8) | b for _, (r, g, b) in phrases}
        if set(observed) - keys - {0}:
            raise ValueError("mask contains colors absent from the caption legend")
        limit = max(min_pixels, int(np.ceil(packed.size * min_fraction)))
        candidates = [
            (phrase, color)
            for phrase, color in unique_phrases(phrases)
            if observed.get((color[0] << 16) | (color[1] << 8) | color[2], 0) >= limit
        ]
        random.Random(f"42:{idx}").shuffle(candidates)
        split = image_split(digest, idx, {}, val_fraction)
        rows = []
        input_name = f"images/{idx:012d}.jpg"
        for n, (phrase, color) in enumerate(candidates[:max_phrases]):
            mask = np.all(rgb == color, axis=2)
            target = np.zeros_like(rgb)
            target[mask] = 255
            target_name = f"targets/{idx:012d}_ref_{n}.png"
            Image.fromarray(target).save(Path(out) / target_name, compress_level=3)
            rows.append(
                dict(
                    task="referring",
                    encoding=BINARY_REFERRING_ENCODING,
                    expression=phrase,
                    prompt=build_binary_referring_prompt(phrase),
                    input=input_name,
                    target=target_name,
                    split=split,
                    sample_id=f"psg_gcg:{idx}:{n}",
                    image_id=idx,
                    source_image_id=idx,
                    source_dataset="psg_gcg",
                    supervision="psg_panoptic_mask",
                    height=image.height,
                    width=image.width,
                    geometry_sha256=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
                    source_caption=row.get("caption"),
                    source_mask=row["seg"],
                    source_input=row["image"],
                    rgb_pixels_sha256=digest,
                )
            )
        if rows:
            shutil.copyfile(
                source_image, Path(out) / input_name
            )  # the source bytes, not a re-encode
        return {
            "id": idx,
            "pixel_hash": digest,
            "rows": rows,
            "phrases": len(phrases),
            "unique_phrases": len(candidates),
        }
    except (ValueError, OSError, KeyError) as e:
        return {"id": idx if idx is not None else row.get("image"), "error": str(e)}


def gcg(args):
    out = args.output_dir
    prepare_output(out)
    exclude = (
        set(json.loads(args.exclude_image_ids.read_text())) if args.exclude_image_ids else set()
    )
    records, rejected = [], []
    for row in read_jsonl(args.source_root / GCG_MANIFEST):
        idx = coco_image_id_from_path(row["image"])
        if idx is not None and idx in exclude:
            rejected.append({"id": idx, "reason": "heldout_image_id"})
        else:
            records.append(row)
    if args.max_images:
        records = records[: args.max_images]
    counts, hashes = Counter(), set()
    jobs = (
        (r, str(args.media_root), str(out), args.val_fraction, args.max_phrases, 64, 2e-4)
        for r in records
    )
    with (
        ProcessPoolExecutor(args.workers) as pool,
        (out / "pairs.jsonl").open("w", encoding="utf-8") as fh,
    ):
        for result in pool.map(gcg_work, jobs, chunksize=16):
            rows = result.pop("rows", [])
            if "error" in result:
                rejected.append(
                    {"id": result["id"], "reason": "invalid_source", "detail": result["error"]}
                )
                continue
            if result["pixel_hash"] in hashes:
                rejected.append({"id": result["id"], "reason": "duplicate_decoded_rgb"})
                continue
            hashes.add(result["pixel_hash"])
            counts["source_phrases"] += result["phrases"]
            counts["unique_phrases"] += result["unique_phrases"]
            if not rows:
                rejected.append({"id": result["id"], "reason": "no_unambiguous_phrase"})
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                counts[f"referring/{row['split']}"] += 1
            counts["source_images_retained"] += bool(rows)
    report = dict(
        source="psg_gcg",
        converted_images=len(records),
        counts=counts,
        rejections=Counter(r["reason"] for r in rejected),
    )
    dump(out / "rejected.json", rejected)
    dump(out / "build_report.json", report)
    print(json.dumps(report, default=dict), flush=True)


# ── audit (perceptual hash) ───────────────────────────────────────────────────

_DCT = None


def phash(path) -> int:
    """64-bit DCT perceptual hash (32x32 grey, unnormalised DCT-II, 8x8 low band above its median)."""
    global _DCT
    if _DCT is None:
        n = np.arange(32)
        _DCT = 2 * np.cos(np.pi * np.outer(n, 2 * n + 1) / 64)
    with Image.open(path) as im:
        pixels = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = (_DCT @ pixels @ _DCT.T)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hash_job(path):
    try:
        return str(path), phash(path)
    except OSError:
        return str(path), None


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between uint64 arrays a (n) and b (m) -> (n, m)."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


def audit(args):
    inputs = set()
    for manifest in args.manifest:
        root = Path(manifest).parent
        inputs |= {
            str(root / r["input"]) for r in read_jsonl(manifest) if r.get("split") == "train"
        }
    evals = {}
    for spec in args.eval_root:
        name, path = spec.split("=", 1)
        evals[name] = sorted(p for p in Path(path).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    mine = dict(pool_map(_hash_job, sorted(inputs), args.workers, 64))
    theirs = {
        name: dict(pool_map(_hash_job, paths, args.workers, 16)) for name, paths in evals.items()
    }
    keys = [k for k, v in mine.items() if v is not None]
    hashes = np.array([mine[k] for k in keys], dtype=np.uint64)
    report = {
        "inputs": len(inputs),
        "unreadable_inputs": len(inputs) - len(keys),
        "max_distance": args.max_distance,
        "eval_images": {n: len(v) for n, v in theirs.items()},
        "hits": {},
    }
    drop = set()
    for name, table in theirs.items():
        paths = [k for k, v in table.items() if v is not None]
        if not paths:
            report["hits"][name] = []
            continue
        ref = np.array([table[k] for k in paths], dtype=np.uint64)
        hits = []
        for start in range(0, len(hashes), 256):  # 256 x m x 64 bytes per block
            dist = hamming(hashes[start : start + 256], ref)
            for i, j in zip(*np.nonzero(dist <= args.max_distance), strict=True):
                hits.append(
                    {"input": keys[start + i], "eval": paths[j], "distance": int(dist[i, j])}
                )
                drop.add(keys[start + i])
        report["hits"][name] = hits
    report["dropped_inputs"] = len(drop)
    dump(args.output, report)
    dump(Path(args.output).with_suffix(".drop.json"), sorted(drop))
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "hits"}
            | {"hits": {n: len(h) for n, h in report["hits"].items()}}
        ),
        flush=True,
    )


# ── compose ───────────────────────────────────────────────────────────────────


def absolute_rows(manifest: Path, drop: set[str]):
    root = manifest.parent
    for row in read_jsonl(manifest):
        if row.get("split") != "train":
            continue
        row = dict(row, input=str(root / row["input"]), target=str(root / row["target"]))
        if row["input"] in drop:
            continue
        yield row


def compose(args):
    drop = set(json.loads(args.drop.read_text())) if args.drop else set()
    counts = defaultdict(Counter)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in read_jsonl(args.base):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["base"][f"{row['task']}/{row.get('split', 'train')}"] += 1
        for manifest in args.add:
            for row in absolute_rows(manifest, drop):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["source_dataset"]][row["task"]] += 1
    summary = {
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "drop_list": len(drop),
        "counts": {k: dict(v) for k, v in counts.items()},
    }
    dump(args.output.with_suffix(".compose.json"), summary)
    print(json.dumps(summary), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("coconut", "gcg"):
        s = sub.add_parser(name)
        s.add_argument(
            "--source-root", type=Path, required=True, help="ModelScope download (manifests)"
        )
        s.add_argument(
            "--media-root", type=Path, required=True, help="unpacked media (images, masks)"
        )
        s.add_argument("--output-dir", type=Path, required=True)
        s.add_argument(
            "--exclude-image-ids", type=Path, help="JSON list of evaluation COCO image ids"
        )
        s.add_argument("--workers", type=int, default=32)
        s.add_argument("--max-images", type=int)
        s.add_argument("--val-fraction", type=float, default=0.02)
    sub.choices["coconut"].add_argument(
        "--subset", required=True, choices=("coconut_b", "coconut_l")
    )
    sub.choices["coconut"].add_argument(
        "--skip-images-of",
        type=Path,
        action="append",
        default=[],
        help="manifest whose source_image_id images are already in the pool",
    )
    sub.choices["gcg"].add_argument("--max-phrases", type=int, default=3)
    a = sub.add_parser("audit")
    a.add_argument("--manifest", type=Path, action="append", required=True)
    a.add_argument(
        "--eval-root", action="append", required=True, help="name=directory of evaluation images"
    )
    a.add_argument("--max-distance", type=int, default=6)
    a.add_argument("--workers", type=int, default=32)
    a.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compose")
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--add", type=Path, action="append", required=True)
    c.add_argument("--drop", type=Path)
    c.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    {"coconut": coconut, "gcg": gcg, "audit": audit, "compose": compose}[args.cmd](args)


if __name__ == "__main__":
    main()
