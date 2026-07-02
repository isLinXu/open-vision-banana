#!/usr/bin/env python3
"""Build a generation dataset for Vision Banana 9:1 mix training.

Collects all available image sources, deduplicates, optionally augments,
and produces a unified dataset with {images/*.jpg, captions.json} format.

Usage:
    python scripts/build_generation_dataset.py
    python scripts/build_generation_dataset.py --output data/vb_generation_5k --augment
    python scripts/build_generation_dataset.py --target-count 2000 --augment
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_generation_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

GENERIC_CAPTIONS = [
    "A natural photograph.",
    "A realistic image with rich visual content.",
    "A high quality photograph.",
    "A detailed scene captured in a photograph.",
    "A photorealistic image.",
    "A clear photograph of an everyday scene.",
    "A colorful image with diverse visual elements.",
    "A well-lit photograph showing various objects.",
]

AUGMENTATION_OPS = [
    "hflip",
    "brightness_up",
    "brightness_down",
    "contrast_up",
    "contrast_down",
]


def _collect_coco128() -> list[dict]:
    """Collect COCO128 images with their captions."""
    coco128_dir = DATA_DIR / "coco128"
    captions_path = coco128_dir / "captions.json"
    if not captions_path.exists():
        return []

    with open(captions_path) as f:
        entries = json.load(f)

    img_dir = coco128_dir / "images"
    samples = []
    for entry in entries:
        img_path = img_dir / entry["filename"]
        if img_path.exists():
            samples.append({
                "source": "coco128",
                "src_path": img_path,
                "caption": entry.get("caption", "A photo."),
                "original_filename": entry["filename"],
            })
    logger.info("coco128: %d samples", len(samples))
    return samples


def _collect_from_dir(source_name: str, data_subdir: Path | str) -> list[dict]:
    """Collect images from a generic data directory."""
    target_dir = DATA_DIR / data_subdir if isinstance(data_subdir, str) else data_subdir
    img_dir = target_dir / "images"
    if not img_dir.exists():
        return []

    files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    samples = []
    for img_path in files:
        caption = random.choice(GENERIC_CAPTIONS)
        samples.append({
            "source": source_name,
            "src_path": img_path,
            "caption": caption,
            "original_filename": img_path.name,
        })
    logger.info("%s: %d images", source_name, len(samples))
    return samples


def _deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicate images by original filename."""
    seen: set[str] = set()
    unique = []
    for s in samples:
        if s["original_filename"] not in seen:
            seen.add(s["original_filename"])
            unique.append(s)
    logger.info("After deduplication: %d -> %d", len(samples), len(unique))
    return unique


def _copy_and_index(samples: list[dict], images_dir: Path) -> list[dict]:
    """Copy images to target directory and build caption entries."""
    images_dir.mkdir(parents=True, exist_ok=True)
    captions = []
    for i, sample in enumerate(samples):
        src = sample["src_path"]
        new_name = f"gen_{i:06d}.jpg"
        dst = images_dir / new_name
        try:
            img = Image.open(src).convert("RGB")
            img.save(dst, "JPEG", quality=95)
            captions.append({
                "filename": new_name,
                "caption": sample["caption"],
                "source": sample["source"],
            })
        except Exception as exc:
            logger.warning("Failed to copy %s: %s", src, exc)
    return captions


def _apply_augmentation(img: Image.Image) -> Image.Image:
    """Apply a random subset of light augmentations to an image."""
    chosen = random.sample(AUGMENTATION_OPS, k=random.randint(1, 2))
    aug = img
    for op in chosen:
        if op == "hflip" and random.random() < 0.5:
            aug = aug.transpose(Image.FLIP_LEFT_RIGHT)
        elif op == "brightness_up":
            aug = ImageEnhance.Brightness(aug).enhance(1.15)
        elif op == "brightness_down":
            aug = ImageEnhance.Brightness(aug).enhance(0.85)
        elif op == "contrast_up":
            aug = ImageEnhance.Contrast(aug).enhance(1.15)
        elif op == "contrast_down":
            aug = ImageEnhance.Contrast(aug).enhance(0.85)
    return aug


def _augment_dataset(captions: list[dict], images_dir: Path) -> list[dict]:
    """Generate augmented variants for every existing image."""
    augmented = list(captions)
    for i, entry in enumerate(captions):
        img_path = images_dir / entry["filename"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            aug_img = _apply_augmentation(img)
            new_name = f"aug_{i:06d}.jpg"
            aug_img.save(images_dir / new_name, "JPEG", quality=95)
            augmented.append({
                "filename": new_name,
                "caption": entry["caption"],
                "source": entry.get("source", "unknown") + "_aug",
            })
        except Exception as exc:
            logger.warning("Augmentation failed for %s: %s", img_path, exc)
    return augmented


def build_dataset(
    output_dir: Path,
    *,
    augment: bool = True,
    target_count: int | None = None,
) -> dict:
    """Build the generation dataset from all available sources.

    Args:
        output_dir: Target directory (must contain ``images/`` and ``captions.json``).
        augment: Whether to create augmented copies to double the dataset size.
        target_count: If set, repeatedly augment until at least this many samples.

    Returns:
        Metadata dict with counts and paths.
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Building Generation Dataset")
    logger.info("=" * 60)

    # ── Collect all sources ──
    all_samples: list[dict] = []
    all_samples.extend(_collect_coco128())
    all_samples.extend(_collect_from_dir("vb_v4_depth_2k", "vb_v4_depth_2k"))
    all_samples.extend(_collect_from_dir("vb_nyu_depth_300", "vb_nyu_depth_300"))
    all_samples.extend(_collect_from_dir("vb_distill_428", "vb_distill_428"))
    all_samples.extend(_collect_from_dir("vision_banana_depth", "vision_banana_depth"))

    logger.info("Total gathered before dedup: %d", len(all_samples))

    unique_samples = _deduplicate(all_samples)

    # ── Copy and index ──
    captions = _copy_and_index(unique_samples, images_dir)
    logger.info("Base dataset: %d samples", len(captions))

    # ── Augmentation ──
    if augment:
        captions = _augment_dataset(captions, images_dir)
        logger.info("After augmentation: %d samples", len(captions))

    # ── Target count padding (augment more if needed) ──
    if target_count and len(captions) < target_count:
        logger.info("Target count %d not reached, applying extra augmentation rounds...", target_count)
        base_captions = [c for c in captions if not c["filename"].startswith("aug_")]
        round_num = 0
        while len(captions) < target_count:
            round_num += 1
            for i, entry in enumerate(base_captions):
                if len(captions) >= target_count:
                    break
                img_path = images_dir / entry["filename"]
                if not img_path.exists():
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                    aug_img = _apply_augmentation(img)
                    new_name = f"aug_r{round_num}_{i:06d}.jpg"
                    aug_img.save(images_dir / new_name, "JPEG", quality=95)
                    captions.append({
                        "filename": new_name,
                        "caption": entry["caption"],
                        "source": entry.get("source", "unknown") + f"_aug_r{round_num}",
                    })
                except Exception as exc:
                    logger.warning("Extra augmentation failed: %s", exc)
            if round_num > 10:
                logger.warning("Too many augmentation rounds, breaking")
                break
        logger.info("After target padding: %d samples", len(captions))

    # ── Save captions.json ──
    captions_path = output_dir / "captions.json"
    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    source_counts = Counter(c.get("source", "unknown") for c in captions)
    logger.info("Source distribution:")
    for src, cnt in source_counts.most_common():
        logger.info("  %-30s %d", src, cnt)

    return {
        "total_samples": len(captions),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "captions_path": str(captions_path),
        "source_counts": dict(source_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Build Vision Banana generation dataset")
    parser.add_argument("--output", type=str, default="data/vb_generation_5k",
                        help="Output directory (relative to project root)")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable image augmentation to double dataset size")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentation")
    parser.add_argument("--target-count", type=int, default=None,
                        help="Minimum number of samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = PROJECT_ROOT / args.output
    augment = not args.no_augment

    result = build_dataset(output_dir, augment=augment, target_count=args.target_count)

    print("\n" + "=" * 60)
    print("Generation Dataset Build Complete")
    print("=" * 60)
    print(f"Total samples:   {result['total_samples']}")
    print(f"Images dir:      {result['images_dir']}")
    print(f"Captions file:   {result['captions_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
