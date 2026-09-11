#!/usr/bin/env python3
"""Generate seg pseudo labels for existing depth + detection images.

Uses SegFormer-B5 (ADE20K, 150 classes) to create segmentation targets for
images that already have depth/normal targets. This enables 3-task training
(depth + normal + seg) with shared input images.

Input sources:
  1. images/depth/shardN/*.jpg (6,065 images, already have depth+normal targets)
  2. images/detection/*.jpg (9,619 images, additional seg-only inputs)

Output:
  - output/snvc_v5full_seg_v2/targets/  (seg pseudo labels)
  - output/snvc_v5full_pairs/pairs_seg_v2.jsonl  (seg pairs)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/media/linxu/thinkHub/datasets/open-vision-banana-snvc-train-full")
MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
NUM_CLASSES = 150
WORK_RES = 512
MODEL_RES = 640

Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def pascal_voc_palette(size: int) -> np.ndarray:
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


CLASS_PALETTE = pascal_voc_palette(NUM_CLASSES + 1)[1:]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((WORK_RES, WORK_RES), Image.Resampling.LANCZOS)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    t = t.unsqueeze(0)
    t = F.interpolate(t, size=(MODEL_RES, MODEL_RES), mode="bilinear", align_corners=False)
    return ((t - MEAN) / STD)[0]


def collect_images() -> list[dict]:
    """Collect all usable input images from depth and detection dirs."""
    items = []
    seen = set()

    # Source 1: depth images (already have depth+normal targets)
    depth_dir = ROOT / "images" / "depth"
    if depth_dir.is_dir():
        for shard in sorted(depth_dir.iterdir()):
            if not shard.is_dir():
                continue
            for f in sorted(shard.iterdir()):
                if f.name.endswith("_depth.png"):
                    continue
                if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                if f.name in seen:
                    continue
                seen.add(f.name)
                items.append({
                    "input": str(f),
                    "name": f.name,
                    "source_group": "depth",
                })

    # Source 2: detection images (seg-only inputs)
    det_dir = ROOT / "images" / "detection"
    if det_dir.is_dir():
        for f in sorted(det_dir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if f.name in seen:
                continue
            seen.add(f.name)
            items.append({
                "input": str(f),
                "name": f.name,
                "source_group": "detection",
            })

    return items


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-targets", type=Path,
                        default=Path("output/snvc_v5full_seg_v2/targets"))
    parser.add_argument("--out-pairs", type=Path,
                        default=Path("output/snvc_v5full_pairs/pairs_seg_v2.jsonl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val-percent", type=int, default=5)
    args = parser.parse_args()

    items = collect_images()
    print(f"collected {len(items)} input images", flush=True)
    src_counts = Counter(it["source_group"] for it in items)
    for src, cnt in src_counts.items():
        print(f"  {src}: {cnt:,}")

    if args.limit:
        items = items[: args.limit]

    args.out_targets.mkdir(parents=True, exist_ok=True)
    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID} on {args.device} ...", flush=True)
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model.eval().to(args.device)
    if args.device == "cuda":
        model.half()
    print(f"  classes: {len(model.config.id2label)}", flush=True)

    palette_t = torch.from_numpy(CLASS_PALETTE.astype(np.uint8))
    counters: Counter = Counter()
    rows: list[dict] = []

    def flush_pending(pending, batch_tensor):
        nonlocal rows
        dtype = torch.float16 if args.device == "cuda" else torch.float32
        logits = model(pixel_values=batch_tensor.to(args.device, dtype=dtype)).logits
        logits = F.interpolate(
            logits.float(), size=(WORK_RES, WORK_RES), mode="bilinear", align_corners=False
        )
        labels = logits.argmax(dim=1).cpu()
        for item, lab in zip(pending, labels):
            rgb = palette_t[lab.reshape(-1)].reshape(WORK_RES, WORK_RES, 3).numpy()
            out_path = args.out_targets / item["name"]
            tmp = out_path.with_name(f".{out_path.name}.tmp")
            Image.fromarray(rgb, mode="RGB").save(tmp, format="PNG", compress_level=1)
            tmp.replace(out_path)
            bucket = int(hashlib.sha1(item["input"].encode()).hexdigest()[:8], 16) % 100
            rows.append({
                "task": "seg",
                "input": item["input"],
                "target": str(out_path.resolve()),
                "source": item["name"],
                "annotation": f"pseudo:{MODEL_ID}",
                "split": "val" if bucket < args.val_percent else "train",
            })
            counters["written"] += 1

    pending: list[dict] = []
    batch_imgs: list[torch.Tensor] = []
    total = len(items)
    t0 = time.time()

    for i, item in enumerate(items):
        out_path = args.out_targets / item["name"]
        if out_path.exists() and out_path.stat().st_size > 100:
            bucket = int(hashlib.sha1(item["input"].encode()).hexdigest()[:8], 16) % 100
            rows.append({
                "task": "seg",
                "input": item["input"],
                "target": str(out_path.resolve()),
                "source": item["name"],
                "annotation": f"pseudo:{MODEL_ID}",
                "split": "val" if bucket < args.val_percent else "train",
            })
            counters["reused"] += 1
            continue
        try:
            with Image.open(item["input"]) as im:
                im.load()
                batch_imgs.append(preprocess(im))
        except Exception as err:
            counters["load_error"] += 1
            if counters["load_error"] <= 10:
                print(f"  [skip] {item['name']}: {err}", flush=True)
            continue
        pending.append(item)
        if len(pending) >= args.batch_size:
            flush_pending(pending, torch.stack(batch_imgs))
            pending, batch_imgs = [], []
        if (i + 1) % 200 == 0 or i + 1 == total:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"[{i + 1}/{total}] written={counters['written']} reused={counters['reused']} "
                  f"errors={counters['load_error']} speed={speed:.1f}/s", flush=True)

    if pending:
        flush_pending(pending, torch.stack(batch_imgs))

    rows.sort(key=lambda r: r["input"])
    with args.out_pairs.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    report = {
        "model": MODEL_ID,
        "total_inputs": total,
        "pairs_written": len(rows),
        "counters": dict(counters),
        "elapsed_sec": round(elapsed, 1),
        "source_groups": dict(src_counts),
    }
    args.out_pairs.with_suffix(".audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\ndone in {elapsed:.0f}s: {len(rows)} pairs -> {args.out_pairs}", flush=True)


if __name__ == "__main__":
    main()
