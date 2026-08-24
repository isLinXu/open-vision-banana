#!/usr/bin/env python3
"""Build the FLUX paper-method gate layout under data/flux_paper_gate_512.

Layout expected by scripts/train_sdxl_mps.py:

    depth/images, depth/targets                 Hypersim metric-depth RGB
    segmentation/images, segmentation/targets   Cityscapes 19-class palette
    generation/images, generation/prompts.jsonl captioned RGB photos

Files are symlinked, not copied. KITTI is intentionally omitted so the first
shared-LoRA gate does not mix indoor/outdoor metric depth.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_flux_paper_gate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENERATION = Path(
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_generation_5k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "flux_paper_gate_512",
    )
    parser.add_argument(
        "--hypersim-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "hypersim_depth_512",
    )
    parser.add_argument(
        "--cityscapes-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "cityscapes_19class_512",
    )
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=DEFAULT_GENERATION,
    )
    parser.add_argument("--depth-limit", type=int, default=200)
    return parser.parse_args()


def _symlink(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def link_vision_pairs(
    rows: list[dict],
    dest: Path,
    *,
    limit: int | None = None,
) -> int:
    image_dir = dest / "images"
    target_dir = dest / "targets"
    image_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = rows if limit is None else rows[:limit]
    written = 0
    for row in selected:
        image = Path(row["input"])
        target = Path(row["target"])
        _symlink(image, image_dir / image.name)
        _symlink(target, target_dir / target.name)
        written += 1
    return written


def link_generation(source: Path, dest: Path) -> int:
    captions_path = source / "captions.json"
    image_dir = source / "images"
    if not captions_path.is_file():
        raise FileNotFoundError(f"Missing generation captions: {captions_path}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Missing generation images: {image_dir}")
    captions = json.loads(captions_path.read_text())
    out_images = dest / "images"
    out_images.mkdir(parents=True, exist_ok=True)
    prompts_path = dest / "prompts.jsonl"
    written = 0
    with prompts_path.open("w", encoding="utf-8") as handle:
        for item in captions:
            filename = item["filename"]
            src = image_dir / filename
            if not src.is_file():
                raise FileNotFoundError(f"Captioned image missing: {src}")
            _symlink(src, out_images / filename)
            handle.write(
                json.dumps({"image": filename, "prompt": item["caption"]}, ensure_ascii=False)
                + "\n"
            )
            written += 1
    return written


def prepare_flux_paper_gate(
    output: Path,
    *,
    hypersim_root: Path,
    cityscapes_root: Path,
    generation_root: Path,
    depth_limit: int,
) -> dict:
    hypersim_pairs = _read_jsonl(hypersim_root / "pairs.jsonl")
    cityscapes_pairs = _read_jsonl(cityscapes_root / "pairs.jsonl")
    if len(hypersim_pairs) < depth_limit:
        raise ValueError(
            f"Need {depth_limit} Hypersim pairs, found {len(hypersim_pairs)} in {hypersim_root}"
        )
    depth_n = link_vision_pairs(hypersim_pairs, output / "depth", limit=depth_limit)
    seg_n = link_vision_pairs(cityscapes_pairs, output / "segmentation")
    gen_n = link_generation(generation_root, output / "generation")
    manifest = {
        "output": str(output.resolve()),
        "depth_pairs": depth_n,
        "segmentation_pairs": seg_n,
        "generation_pairs": gen_n,
        "depth_source": str(hypersim_root.resolve()),
        "segmentation_source": str(cityscapes_root.resolve()),
        "generation_source": str(generation_root.resolve()),
        "notes": [
            "Shared LoRA paper-method gate. One adapter, RGB tasks, ~9:1 generation:vision.",
            "KITTI omitted to avoid indoor/outdoor metric-depth clash on the first gate.",
            "Generation images are existing captioned photos, not FLUX-schnell native samples.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Wrote %d depth, %d segmentation, %d generation pairs to %s",
        depth_n,
        seg_n,
        gen_n,
        output,
    )
    return manifest


def main() -> int:
    args = parse_args()
    prepare_flux_paper_gate(
        args.output,
        hypersim_root=args.hypersim_root,
        cityscapes_root=args.cityscapes_root,
        generation_root=args.generation_root,
        depth_limit=args.depth_limit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
