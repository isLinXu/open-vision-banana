#!/usr/bin/env python3
"""Install the official Cityscapes val tree for paper-protocol semantic mIoU.

``gtFine`` is Cityscapes' fine pixel annotation package, not a model file.
Paper Cityscapes val needs both:

    leftImg8bit/val/{frankfurt,lindau,munster}/*_leftImg8bit.png
    gtFine/val/{frankfurt,lindau,munster}/*_gtFine_labelIds.png

That is 500 paired images. The 8-image ``cityscapes8`` dump is a different
8-class toy set and must not be used here.

Official zips (Cityscapes Custom license) can be obtained after
registration at https://www.cityscapes-dataset.com/downloads/ or from
the ModelScope mirror https://www.modelscope.cn/datasets/OpenDataLab/CityScapes :

    raw/gtFine_trainvaltest.zip          (~241 MB)
    raw/leftImg8bit_trainvaltest.zip     (~11 GB)

Do not download ``raw/leftImg8bit_trainextra.zip`` (~47 GB) or
``raw/gtCoarse.zip`` for paper val. This script extracts the official ``val`` split by default, or ``train``
and ``val`` when ``--splits train val`` is passed. Only ``*_leftImg8bit.png``
and ``*_gtFine_labelIds.png`` are unpacked. Place the two zips on disk, then::

    python scripts/setup_cityscapes_official.py \\
        --gtfine-zip /data/gtFine_trainvaltest.zip \\
        --leftimg-zip /data/leftImg8bit_trainvaltest.zip \\
        --output /data/paper_benchmarks/cityscapes

    python scripts/evaluate_paper_benchmarks.py --benchmark semantic \\
        --cityscapes-root /data/paper_benchmarks/cityscapes \\
        --output-json output/eval/cityscapes_val.json
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setup_cityscapes")

VAL_CITIES = ("frankfurt", "lindau", "munster")
OFFICIAL_VAL_PAIRS = 500
OFFICIAL_TRAIN_PAIRS = 2975


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtfine-zip", type=Path, default=None)
    parser.add_argument("--leftimg-zip", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/cityscapes"),
        help="Root that CityscapesSegDataset will scan.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val"],
        choices=["train", "val"],
        help="Which official splits to extract or verify. Default: val.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only check an already-extracted official tree.",
    )
    return parser.parse_args()


def _member_under(name: str, *parts: str) -> bool:
    normalized = name.replace("\\", "/").lstrip("./")
    token = "/".join(parts)
    return token in normalized.split("/") or f"/{token}/" in f"/{normalized}/"


def _extract_split_members(
    archive: Path,
    destination: Path,
    folder: str,
    split: str,
    suffix: str,
) -> int:
    extracted = 0
    with zipfile.ZipFile(archive) as zipped:
        selected = [
            name
            for name in zipped.namelist()
            if _member_under(name, folder, split)
            and name.endswith(suffix)
            and not name.endswith("/")
        ]
        if not selected:
            raise FileNotFoundError(
                f"{archive} has no {suffix} members under {folder}/{split}"
            )
        for name in selected:
            target = destination / name
            if target.is_file():
                extracted += 1
                continue
            zipped.extract(name, destination)
            extracted += 1
    return extracted


def _collapse_extracted_root(output: Path, folder: str) -> None:
    """Accept zips that wrap ``Cityscapes/gtFine/...`` or ``gtFine/...``."""
    direct = output / folder
    if direct.is_dir():
        return
    matches = [path for path in output.rglob(folder) if path.is_dir() and path.name == folder]
    if len(matches) != 1:
        return
    source = matches[0]
    if source.resolve() == direct.resolve():
        return
    direct.parent.mkdir(parents=True, exist_ok=True)
    source.rename(direct)


def official_split_pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    image_root = root / "leftImg8bit" / split
    label_root = root / "gtFine" / split
    pairs: list[tuple[Path, Path]] = []
    if not image_root.is_dir() or not label_root.is_dir():
        return pairs
    cities = VAL_CITIES if split == "val" else tuple(
        sorted(path.name for path in image_root.iterdir() if path.is_dir())
    )
    for city in cities:
        for image in sorted((image_root / city).glob("*_leftImg8bit.png")):
            label = label_root / city / image.name.replace("leftImg8bit", "gtFine_labelIds")
            if label.is_file():
                pairs.append((image, label))
    return pairs


def official_val_pairs(root: Path) -> list[tuple[Path, Path]]:
    return official_split_pairs(root, "val")


def official_train_pairs(root: Path) -> list[tuple[Path, Path]]:
    return official_split_pairs(root, "train")


def verify_official_val(root: Path) -> list[tuple[Path, Path]]:
    cities = {
        path.name
        for path in (root / "leftImg8bit" / "val").iterdir()
        if path.is_dir()
    } if (root / "leftImg8bit" / "val").is_dir() else set()
    missing_cities = [city for city in VAL_CITIES if city not in cities]
    if missing_cities:
        raise FileNotFoundError(
            f"{root} is not official Cityscapes val; missing cities {missing_cities}. "
            "Do not substitute cityscapes8 or a 50-image export."
        )
    pairs = official_val_pairs(root)
    if len(pairs) != OFFICIAL_VAL_PAIRS:
        raise FileNotFoundError(
            f"{root} has {len(pairs)} official val pairs; expected {OFFICIAL_VAL_PAIRS}."
        )
    return pairs


def verify_official_train(root: Path) -> list[tuple[Path, Path]]:
    pairs = official_train_pairs(root)
    if len(pairs) != OFFICIAL_TRAIN_PAIRS:
        raise FileNotFoundError(
            f"{root} has {len(pairs)} official train pairs; expected {OFFICIAL_TRAIN_PAIRS}."
        )
    return pairs


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = tuple(dict.fromkeys(args.splits))

    if not args.verify_only:
        if args.gtfine_zip is None or args.leftimg_zip is None:
            raise SystemExit(
                "Pass --gtfine-zip and --leftimg-zip, or --verify-only on an extracted tree."
            )
        if not args.gtfine_zip.is_file() or not args.leftimg_zip.is_file():
            raise SystemExit("Both official zip files must exist")
        for split in splits:
            logger.info("Extracting gtFine %s labelIds from %s", split, args.gtfine_zip)
            _extract_split_members(
                args.gtfine_zip, output, "gtFine", split, "_gtFine_labelIds.png"
            )
            logger.info("Extracting leftImg8bit %s from %s", split, args.leftimg_zip)
            _extract_split_members(
                args.leftimg_zip, output, "leftImg8bit", split, "_leftImg8bit.png"
            )
        _collapse_extracted_root(output, "gtFine")
        _collapse_extracted_root(output, "leftImg8bit")

    if "val" in splits:
        pairs = verify_official_val(output)
        logger.info("Official Cityscapes val ready: %d pairs under %s", len(pairs), output)
    if "train" in splits:
        pairs = verify_official_train(output)
        logger.info("Official Cityscapes train ready: %d pairs under %s", len(pairs), output)
    print(
        "python scripts/evaluate_paper_benchmarks.py --benchmark semantic "
        f"--cityscapes-root {output} --native-resolution "
        "--output-json output/eval/cityscapes_val_official.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
