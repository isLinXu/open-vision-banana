#!/usr/bin/env python3
"""Turn SA-Co/Gold COCO JSON into the queries.jsonl this repo evaluates.

Uses annotator pass ``a`` only. The paper cgF1 takes the best of three
human passes; this file is the downloaded official pass, not that oracle.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    return parser.parse_args()


def find_image(image_root: Path, file_name: str) -> Path | None:
    name = Path(file_name).name
    candidates = (
        image_root / file_name,
        image_root / "metaclip-images" / file_name,
        image_root / "sa1b-images" / name,
        image_root / "sa1b-images" / file_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    args = parse_args()
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(args.annotations.glob("*_a_release_test.json"))
    if not files:
        raise SystemExit(f"no *_a_release_test.json under {args.annotations}")
    written = 0
    missing_images = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            subset = path.name.replace("_merged_a_release_test.json", "")
            by_image: dict[int, list[dict]] = defaultdict(list)
            for ann in payload["annotations"]:
                by_image[int(ann["image_id"])].append(ann)
            for image in payload["images"]:
                image_path = find_image(args.image_root, image["file_name"])
                if image_path is None:
                    missing_images += 1
                    continue
                anns = by_image.get(int(image["id"]), [])
                mask_path = None
                if anns:
                    height = int(image["height"])
                    width = int(image["width"])
                    combined = np.zeros((height, width), dtype=np.uint8)
                    for ann in anns:
                        decoded = mask_utils.decode(ann["segmentation"])
                        if decoded.ndim == 3:
                            decoded = decoded.any(axis=2)
                        combined[decoded.astype(bool)] = 255
                    mask_path = args.mask_dir / f"{subset}_{image['id']}.png"
                    Image.fromarray(combined, mode="L").save(mask_path)
                row = {
                    "image": str(image_path),
                    "query": image["text_input"],
                    "present": bool(anns),
                    "subset": subset,
                    "annotator": "a",
                }
                if mask_path is not None:
                    row["mask"] = str(mask_path)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(json.dumps({"queries": written, "missing_images": missing_images, "files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
