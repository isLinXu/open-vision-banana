#!/usr/bin/env python3
"""Dataset preparation script that downloads and splits train/val in one go.

Downloads benchmark datasets from public sources, extracts them, and
automatically splits into ``train/`` and ``val/`` subsets in the standard
``images/ + targets/`` layout expected by the Vision Banana training pipeline.

Features:
    - argparse CLI with sub-commands
    - Structured logging (level + file handlers)
    - Resume support via JSON state file
    - Automatic train/val split with reproducible seed
    - Post-download validation via ``validate_vision_dataset``
    - Dry-run mode for safe preview

Usage:
    # list every supported dataset
    python scripts/download_train_val.py list

    # download one dataset and split it 90/10 train/val
    python scripts/download_train_val.py download --dataset nyu --output data/nyu

    # download every supported dataset
    python scripts/download_train_val.py download --dataset all --output data/

    # validation ratio + random seed + resume
    python scripts/download_train_val.py download --dataset diode \
        --output data/diode --val-ratio 0.15 --seed 2024 --resume

    # dry run (print the plan only, run nothing)
    python scripts/download_train_val.py download --dataset all --output data/ --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# try importing the project modules; degrade gracefully if they are unavailable
# ---------------------------------------------------------------------------
try:
    from open_vision_banana.data.validation import validate_vision_dataset
    _HAS_VALIDATION = True
except Exception:
    _HAS_VALIDATION = False

try:
    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    _HAS_MAPPER = True
except Exception:
    _HAS_MAPPER = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATE_FILENAME = ".download_state.json"
CHUNK_SIZE = 8192
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

logger = logging.getLogger("download_train_val")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Config of one dataset."""
    name: str
    task: str
    urls: list[dict[str, Any]]  # [{"url": ..., "filename": ..., "optional": bool}]
    size_mb: int
    description: str
    license: str
    prepare_fn: Callable[[Path, Path, argparse.Namespace], int] | None = None
    huggingface: str | None = None
    notes: str = ""


def _noop_prepare(_raw: Path, _out: Path, _args: argparse.Namespace) -> int:
    """No-op prepare — just count raw files."""
    return 0


DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "nyu": DatasetConfig(
        name="NYU Depth v2",
        task="depth",
        urls=[],
        size_mb=1500,
        description="Indoor RGB-D, 1449 labeled pairs, most common depth benchmark",
        license="Academic",
        prepare_fn=None,
        huggingface="patriziapol/stratified-depth-dataset-nyu-depth-v2",
        notes="The official source needs a manual download; the script tries the HuggingFace mirror first.",
    ),
    "diode": DatasetConfig(
        name="DIODE",
        task="depth",
        urls=[
            {
                "url": "http://diode-dataset.s3.amazonaws.com/train_val.tar.gz",
                "filename": "diode_train_val.tar.gz",
                "optional": False,
            },
        ],
        size_mb=2500,
        description="First real indoor+outdoor dense depth, 25K train / 3.3K val",
        license="Academic",
        prepare_fn=None,
    ),
    "kitti": DatasetConfig(
        name="KITTI Depth",
        task="depth",
        urls=[
            {
                "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip",
                "filename": "data_depth_annotated.zip",
                "optional": False,
            },
        ],
        size_mb=1500,
        description="Autonomous driving sparse LiDAR depth, 23K train / 1K test",
        license="CC BY-NC-SA 3.0",
        prepare_fn=None,
        notes="The official source needs registration; the script gives a direct link, which may break if the license changes.",
    ),
    "ade20k": DatasetConfig(
        name="ADE20K",
        task="semantic",
        urls=[
            {
                "url": "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
                "filename": "ADEChallengeData2016.zip",
                "optional": False,
            },
        ],
        size_mb=700,
        description="Scene parsing standard, 20K training / 150 classes / indoor+outdoor",
        license="BSD",
        prepare_fn=None,
    ),
    "sintel": DatasetConfig(
        name="Sintel Flow",
        task="flow",
        urls=[
            {
                "url": "http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
                "filename": "MPI-Sintel-complete.zip",
                "optional": False,
            },
        ],
        size_mb=6000,
        description="Classic optical flow benchmark, clean/final versions, 1K training frames",
        license="Academic",
        prepare_fn=None,
    ),
    "flyingthings3d": DatasetConfig(
        name="FlyingThings3D",
        task="flow",
        urls=[
            {
                "url": "https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D/FlyingThings3D_subset.zip",
                "filename": "FlyingThings3D_subset.zip",
                "optional": False,
            },
        ],
        size_mb=42000,
        description="Synthetic 3D optical flow, 25K stereo frames, good for pretraining",
        license="Academic",
        prepare_fn=None,
        notes="About 42GB; a subset is recommended.",
    ),
    "hypersim": DatasetConfig(
        name="Hypersim (subset)",
        task="depth",
        urls=[],
        size_mb=33000,
        description="Paper-equivalent synthetic data, first 8 scenes ~33GB",
        license="MIT",
        prepare_fn=None,
        notes="Clone manually from GitHub: https://github.com/apple/ml-hypersim",
    ),
    "vkitti2": DatasetConfig(
        name="Virtual KITTI 2",
        task="depth",
        urls=[],
        size_mb=50000,
        description="Synthetic KITTI with weather/time, 50K frames, perfect GT",
        license="Academic",
        prepare_fn=None,
        notes="Download manually from the official site: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/",
    ),
    "blendedmvs": DatasetConfig(
        name="BlendedMVS",
        task="depth",
        urls=[],
        size_mb=17000,
        description="Multi-view stereo, high-quality depth",
        license="Academic",
        prepare_fn=None,
        notes="GitHub: https://github.com/YoYo000/MVSNet",
    ),
    "scannet": DatasetConfig(
        name="ScanNet Normal",
        task="normal",
        urls=[],
        size_mb=12000,
        description="Real indoor scans with surface normals",
        license="ScanNet Terms",
        prepare_fn=None,
        notes="Registration required: http://www.scan-net.org/",
    ),
    "cityscapes": DatasetConfig(
        name="Cityscapes Semantic",
        task="semantic",
        urls=[],
        size_mb=11000,
        description="Urban street scenes, 5K fine annotations",
        license="Cityscapes Terms",
        prepare_fn=None,
        notes="Registration required: https://www.cityscapes-dataset.com/",
    ),
    "refcocog": DatasetConfig(
        name="RefCOCOg",
        task="referring",
        urls=[],
        size_mb=50,
        description="Referring expression segmentation, 26K images",
        license="MIT",
        prepare_fn=None,
        notes="GitHub: https://github.com/lichengunc/refer",
    ),
}


# ---------------------------------------------------------------------------
# State management (resume support)
# ---------------------------------------------------------------------------
@dataclass
class DownloadState:
    """Resume state."""
    completed_downloads: list[str] = field(default_factory=list)
    completed_splits: list[str] = field(default_factory=list)
    completed_validations: list[str] = field(default_factory=list)
    failed_datasets: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> DownloadState:
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return cls(**data)
            except Exception as exc:
                logger.warning("State file corrupted (%s), starting fresh.", exc)
        return cls()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")

    def is_downloaded(self, dataset: str) -> bool:
        return dataset in self.completed_downloads

    def is_split(self, dataset: str) -> bool:
        return dataset in self.completed_splits

    def is_validated(self, dataset: str) -> bool:
        return dataset in self.completed_validations


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------
def _check_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _file_hash(path: Path, algo: str = "md5") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    output_path: Path,
    *,
    timeout: int = 600,
    resume: bool = False,
) -> bool:
    """Download a single file via wget/curl/urllib with optional resume.

    Prefers wget -c for resumable downloads; falls back to curl, then urllib.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) wget with resume
    if _check_tool("wget"):
        cmd = [
            "wget",
            "-c" if resume else "--no-clobber",
            "-O", str(output_path),
            "--timeout", str(timeout),
            "--tries", "3",
            url,
        ]
        logger.info("[wget] %s → %s", url, output_path)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout + 60
            )
            if result.returncode == 0:
                return True
            logger.warning("wget failed: %s", result.stderr[:500])
        except subprocess.TimeoutExpired:
            logger.error("wget timeout after %ds", timeout)
        except Exception as exc:
            logger.error("wget error: %s", exc)

    # 2) curl fallback (no resume support via CLI flags easily)
    if _check_tool("curl"):
        cmd = [
            "curl", "-L", "-o", str(output_path),
            "--max-time", str(timeout),
            "--retry", "2",
            url,
        ]
        logger.info("[curl] %s → %s", url, output_path)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout + 60
            )
            if result.returncode == 0:
                return True
            logger.warning("curl failed: %s", result.stderr[:500])
        except subprocess.TimeoutExpired:
            logger.error("curl timeout after %ds", timeout)
        except Exception as exc:
            logger.error("curl error: %s", exc)

    # 3) urllib fallback (pure Python, no resume)
    logger.info("[urllib] %s → %s", url, output_path)
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as exc:
        logger.error("urllib download failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------
def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract zip / tar / tar.gz to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = archive_path.suffix.lower()
    name = archive_path.name.lower()

    try:
        if suffix == ".zip" or name.endswith(".zip"):
            logger.info("Extracting ZIP: %s → %s", archive_path, output_dir)
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(output_dir)
            return True

        if suffix in (".tar", ".gz", ".tgz") or name.endswith((".tar.gz", ".tgz")):
            logger.info("Extracting TAR: %s → %s", archive_path, output_dir)
            mode = "r:gz" if name.endswith((".gz", ".tgz")) else "r"
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(output_dir)
            return True

        logger.error("Unknown archive format: %s", suffix)
        return False
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Dataset-specific preparation
# ---------------------------------------------------------------------------
def _save_depth_rgb_pair(
    img: Image.Image,
    depth: torch.Tensor,
    out_img: Path,
    out_tgt: Path,
    resolution: int | None,
) -> None:
    """Encode a depth tensor as RGB and save the image/target pair."""
    if resolution is not None:
        img = img.resize((resolution, resolution), Image.LANCZOS)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(resolution, resolution), mode="nearest"
        ).squeeze(0)

    if _HAS_MAPPER:
        mapper = DepthRGBMapper()
        target_rgb = mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        arr = (target_rgb.clamp(0.0, 1.0) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr, mode="RGB").save(out_tgt)
    else:
        # fallback: save the depth directly as a 16-bit PNG
        depth_np = depth.squeeze().cpu().numpy()
        depth_16 = (depth_np * 1000.0).clip(0, 65535).astype(np.uint16)
        Image.fromarray(depth_16, mode="I;16").save(out_tgt)

    img.save(out_img)


def prepare_diode(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """DIODE: extract image/depth pairs from the extracted train_val directory."""
    # DIODE layout after extraction: train/indoor, train/outdoor, val/indoor, val/outdoor
    # each directory holds *_rgb.png and *_depth.npy
    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    src_dirs = list(raw_dir.rglob("*"))
    rgb_files = sorted(
        p for p in src_dirs
        if p.is_file() and "rgb" in p.stem.lower() and p.suffix.lower() in IMG_EXTS
    )

    count = 0
    for rgb_path in rgb_files:
        depth_path = rgb_path.parent / rgb_path.name.replace("_rgb", "_depth").replace(".png", ".npy")
        if not depth_path.exists():
            continue

        out_img = img_dir / f"diode_{count:06d}.png"
        out_tgt = tgt_dir / f"diode_{count:06d}.png"
        if out_img.exists() and out_tgt.exists() and not args.overwrite:
            count += 1
            continue

        img = Image.open(rgb_path).convert("RGB")
        depth = torch.from_numpy(np.load(depth_path)).float()
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.unsqueeze(0)
        if depth.max() > 1.0:
            depth = depth / 1000.0

        _save_depth_rgb_pair(img, depth, out_img, out_tgt, args.resolution)
        count += 1
        if count % 100 == 0:
            logger.info("DIODE processed %d pairs", count)

    logger.info("DIODE: wrote %d pairs to %s", count, output_dir)
    return count


def prepare_kitti(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """KITTI: extract RGB/depth pairs from data_depth_annotated."""
    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    # find the depth maps and the matching RGB
    depth_files = sorted(raw_dir.rglob("*groundtruth_depth*.png"))
    count = 0
    for depth_path in depth_files:
        # the raw KITTI RGB lives elsewhere; only the downloaded annotated package is handled here
        # that package usually contains only depth, and the RGB must come from the raw data
        # for simplicity, skip when no RGB is found
        rgb_path = depth_path.parent / depth_path.name.replace("groundtruth_depth", "image")
        if not rgb_path.exists():
            continue

        out_img = img_dir / f"kitti_{count:06d}.png"
        out_tgt = tgt_dir / f"kitti_{count:06d}.png"
        if out_img.exists() and out_tgt.exists() and not args.overwrite:
            count += 1
            continue

        img = Image.open(rgb_path).convert("RGB")
        depth = torch.from_numpy(np.array(Image.open(depth_path))).float().unsqueeze(0)
        depth = depth / 256.0  # KITTI depth scaling

        _save_depth_rgb_pair(img, depth, out_img, out_tgt, args.resolution)
        count += 1

    logger.info("KITTI: wrote %d pairs to %s", count, output_dir)
    return count


def prepare_ade20k(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """ADE20K: extract training/validation image + annotation pairs."""
    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    # common paths after extraction
    base = raw_dir / "ADEChallengeData2016"
    if not base.exists():
        base = raw_dir

    train_img_dir = base / "images" / "training"
    train_ann_dir = base / "annotations" / "training"
    val_img_dir = base / "images" / "validation"
    val_ann_dir = base / "annotations" / "validation"

    pairs: list[tuple[Path, Path, str]] = []
    for split, idir, adir in [
        ("train", train_img_dir, train_ann_dir),
        ("val", val_img_dir, val_ann_dir),
    ]:
        if not idir.exists():
            continue
        for img_path in sorted(idir.glob("*.jpg")):
            ann_path = adir / img_path.name.replace(".jpg", ".png")
            if ann_path.exists():
                pairs.append((img_path, ann_path, split))

    count = 0
    for img_path, ann_path, _split in pairs:
        out_img = img_dir / f"ade20k_{count:06d}.png"
        out_tgt = tgt_dir / f"ade20k_{count:06d}.png"
        if out_img.exists() and out_tgt.exists() and not args.overwrite:
            count += 1
            continue

        img = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path).convert("RGB")
        if args.resolution is not None:
            img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
            ann = ann.resize((args.resolution, args.resolution), Image.NEAREST)
        img.save(out_img)
        ann.save(out_tgt)
        count += 1

    logger.info("ADE20K: wrote %d pairs to %s", count, output_dir)
    return count


def prepare_sintel(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """Sintel: extract flow frame pairs."""
    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    # MPI-Sintel-complete layout: training/clean, training/final, training/flow
    base = raw_dir / "MPI-Sintel-complete" / "training"
    if not base.exists():
        base = raw_dir / "training"
    if not base.exists():
        base = raw_dir

    clean_dir = base / "clean"
    flow_dir = base / "flow"
    if not clean_dir.exists() or not flow_dir.exists():
        logger.warning("Sintel expected structure not found under %s", raw_dir)
        return 0

    # for the flow task the target is the flow field; simplified here to saving RGB frame pairs
    # real training needs the .flo files parsed further; this only maps the files
    count = 0
    for scene_dir in sorted(clean_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for img_path in sorted(scene_dir.glob("*.png")):
            out_img = img_dir / f"sintel_{count:06d}.png"
            if out_img.exists() and not args.overwrite:
                count += 1
                continue
            img = Image.open(img_path).convert("RGB")
            if args.resolution is not None:
                img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
            img.save(out_img)
            count += 1

    logger.info("Sintel: wrote %d frames to %s", count, output_dir)
    return count


def prepare_flyingthings3d(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> int:
    """FlyingThings3D subset: extract image + flow/disparity pairs."""
    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    # subset layout: train/image_clean/left/*.png
    base = raw_dir / "FlyingThings3D_subset"
    if not base.exists():
        base = raw_dir

    img_subdirs = list(base.rglob("image_clean/left"))
    count = 0
    for subdir in img_subdirs:
        for img_path in sorted(subdir.glob("*.png")):
            out_img = img_dir / f"ft3d_{count:06d}.png"
            if out_img.exists() and not args.overwrite:
                count += 1
                continue
            img = Image.open(img_path).convert("RGB")
            if args.resolution is not None:
                img = img.resize((args.resolution, args.resolution), Image.LANCZOS)
            img.save(out_img)
            count += 1

    logger.info("FlyingThings3D: wrote %d frames to %s", count, output_dir)
    return count


def prepare_nyu_from_hf(output_dir: Path, args: argparse.Namespace) -> int:
    """Download NYU Depth v2 through the HuggingFace ``datasets`` library."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        logger.error("`datasets` library required for NYU HF download: pip install datasets")
        raise RuntimeError("Missing dependency: datasets") from exc

    logger.info("Loading NYU Depth v2 from HuggingFace: patriziapol/stratified-depth-dataset-nyu-depth-v2")
    ds = load_dataset("patriziapol/stratified-depth-dataset-nyu-depth-v2", split="train")
    logger.info("HF dataset loaded: %d samples", len(ds))

    img_dir = output_dir / "images"
    tgt_dir = output_dir / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    limit = args.max_samples if args.max_samples else len(ds)
    count = 0
    for i, sample in enumerate(ds):
        if i >= limit:
            break
        out_img = img_dir / f"nyu_{count:06d}.png"
        out_tgt = tgt_dir / f"nyu_{count:06d}.png"
        if out_img.exists() and out_tgt.exists() and not args.overwrite:
            count += 1
            continue

        if "image" not in sample or "depth" not in sample:
            continue

        sample["image"].save(out_img)
        depth = np.array(sample["depth"])
        depth_16 = (depth * 1000.0).clip(0, 65535).astype(np.uint16)
        Image.fromarray(depth_16, mode="I;16").save(out_tgt)
        count += 1
        if count % 100 == 0:
            logger.info("NYU HF: saved %d / %d", count, limit)

    logger.info("NYU: wrote %d pairs to %s", count, output_dir)
    return count


# bind prepare_fn
DATASET_REGISTRY["diode"].prepare_fn = prepare_diode
DATASET_REGISTRY["kitti"].prepare_fn = prepare_kitti
DATASET_REGISTRY["ade20k"].prepare_fn = prepare_ade20k
DATASET_REGISTRY["sintel"].prepare_fn = prepare_sintel
DATASET_REGISTRY["flyingthings3d"].prepare_fn = prepare_flyingthings3d


# ---------------------------------------------------------------------------
# Train / Val split
# ---------------------------------------------------------------------------
def split_train_val(
    unified_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    overwrite: bool,
) -> tuple[int, int]:
    """Randomly split unified_dir/images + targets into train / val.

    Returns:
        (train_count, val_count)
    """
    img_dir = unified_dir / "images"
    tgt_dir = unified_dir / "targets"

    if not img_dir.exists() or not tgt_dir.exists():
        logger.error("Cannot split: missing images/ or targets/ in %s", unified_dir)
        return 0, 0

    image_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not image_files:
        logger.warning("No images found in %s", img_dir)
        return 0, 0

    # build the image-target pairs
    pairs: list[tuple[Path, Path]] = []
    for img_path in image_files:
        # prefer an exact match on the target name
        tgt_path = tgt_dir / (img_path.stem + ".png")
        if not tgt_path.exists():
            # fallback: try other extensions with the same stem
            candidates = list(tgt_dir.glob(img_path.stem + ".*"))
            if candidates:
                tgt_path = candidates[0]
            else:
                logger.warning("No target for %s, skipping", img_path.name)
                continue
        pairs.append((img_path, tgt_path))

    n_total = len(pairs)
    if n_total < 2:
        logger.warning("Too few samples (%d) to split", n_total)
        return n_total, 0

    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    if n_train < 1:
        n_train = n_total
        n_val = 0

    # reproducible random split
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    train_idx = set(indices[:n_train])
    set(indices[n_train:])

    # create the output directories
    train_img_dir = output_dir / "train" / "images"
    train_tgt_dir = output_dir / "train" / "targets"
    val_img_dir = output_dir / "val" / "images"
    val_tgt_dir = output_dir / "val" / "targets"
    for d in (train_img_dir, train_tgt_dir, val_img_dir, val_tgt_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0

    for i, (img_path, tgt_path) in enumerate(pairs):
        is_train = i in train_idx
        if is_train:
            dst_img = train_img_dir / img_path.name
            dst_tgt = train_tgt_dir / tgt_path.name
            train_count += 1
        else:
            dst_img = val_img_dir / img_path.name
            dst_tgt = val_tgt_dir / tgt_path.name
            val_count += 1

        if dst_img.exists() and dst_tgt.exists() and not overwrite:
            continue

        # symlink or copy (copy preferred, for independence)
        shutil.copy2(img_path, dst_img)
        shutil.copy2(tgt_path, dst_tgt)

    logger.info(
        "Split complete: train=%d, val=%d (ratio %.2f, seed=%d)",
        train_count, val_count, val_ratio, seed,
    )
    return train_count, val_count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def download_and_prepare(
    dataset_key: str,
    args: argparse.Namespace,
    state: DownloadState,
    state_path: Path,
) -> bool:
    """Download, extract, prepare and split one dataset."""
    config = DATASET_REGISTRY[dataset_key]
    dataset_output = Path(args.output) / dataset_key
    raw_dir = Path(args.download_dir) / dataset_key
    unified_dir = dataset_output / "unified"  # the prepared unified images+targets

    logger.info("=" * 60)
    logger.info("Dataset: %s", config.name)
    logger.info("Task: %s", config.task)
    logger.info("Description: %s", config.description)
    logger.info("License: %s", config.license)
    if config.notes:
        logger.info("Notes: %s", config.notes)
    logger.info("=" * 60)

    # ── 1. Download ──────────────────────────────────────────────────────
    if not state.is_downloaded(dataset_key):
        if args.dry_run:
            logger.info("[DRY-RUN] Would download %d URLs", len(config.urls))
            for u in config.urls:
                logger.info("  → %s", u.get("url", "N/A"))
        else:
            # handle datasets that need a manual download
            if not config.urls and config.huggingface:
                logger.info("Using HuggingFace source: %s", config.huggingface)
                if dataset_key == "nyu":
                    try:
                        prepare_nyu_from_hf(unified_dir, args)
                    except Exception as exc:
                        logger.error("NYU HF prepare failed: %s", exc)
                        state.failed_datasets.append(dataset_key)
                        state.save(state_path)
                        return False
                else:
                    logger.warning("HF auto-download not yet implemented for %s", dataset_key)
                    state.failed_datasets.append(dataset_key)
                    state.save(state_path)
                    return False
            elif not config.urls:
                logger.warning("Dataset '%s' has no auto-download URLs. Manual download required.", dataset_key)
                logger.info("  Expected at: %s", raw_dir)
                state.failed_datasets.append(dataset_key)
                state.save(state_path)
                return False
            else:
                raw_dir.mkdir(parents=True, exist_ok=True)
                for url_info in config.urls:
                    if url_info.get("optional", False):
                        logger.info("Skipping optional: %s", url_info["filename"])
                        continue
                    dest = raw_dir / url_info["filename"]
                    if dest.exists() and not args.overwrite:
                        logger.info("Already present: %s", dest)
                        continue
                    ok = download_file(
                        url_info["url"],
                        dest,
                        resume=args.resume,
                        timeout=args.timeout,
                    )
                    if not ok and not url_info.get("optional", False):
                        logger.error("Failed to download required file: %s", url_info["filename"])
                        state.failed_datasets.append(dataset_key)
                        state.save(state_path)
                        return False

                # extract
                for url_info in config.urls:
                    archive = raw_dir / url_info["filename"]
                    if archive.exists():
                        extract_archive(archive, raw_dir)

        if not args.dry_run:
            state.completed_downloads.append(dataset_key)
            state.save(state_path)
    else:
        logger.info("Download phase already completed (resume).")

    # ── 2. Prepare ───────────────────────────────────────────────────────
    if config.prepare_fn is not None and dataset_key != "nyu":
        if args.dry_run:
            logger.info("[DRY-RUN] Would prepare dataset into %s", unified_dir)
        else:
            try:
                config.prepare_fn(raw_dir, unified_dir, args)
            except Exception as exc:
                logger.error("Preparation failed: %s", exc)
                state.failed_datasets.append(dataset_key)
                state.save(state_path)
                return False
    elif dataset_key == "nyu":
        # NYU was already written to unified_dir through HF in the download stage
        pass
    else:
        # no prepare_fn: assume raw_dir is already images/ + targets/
        if (raw_dir / "images").exists() and (raw_dir / "targets").exists():
            unified_dir = raw_dir
        else:
            logger.warning("No prepare_fn and no images/targets found in %s", raw_dir)

    # ── 3. Train/Val Split ──────────────────────────────────────────────
    if not state.is_split(dataset_key):
        if args.dry_run:
            logger.info("[DRY-RUN] Would split into train/val (ratio %.2f)", args.val_ratio)
        else:
            if not unified_dir.exists():
                logger.error("Unified dir not found after preparation: %s", unified_dir)
                state.failed_datasets.append(dataset_key)
                state.save(state_path)
                return False

            train_n, val_n = split_train_val(
                unified_dir,
                dataset_output,
                args.val_ratio,
                args.seed,
                args.overwrite,
            )
            if train_n == 0:
                logger.error("Split produced zero training samples.")
                state.failed_datasets.append(dataset_key)
                state.save(state_path)
                return False

            state.completed_splits.append(dataset_key)
            state.save(state_path)
    else:
        logger.info("Split phase already completed (resume).")

    # ── 4. Validation ────────────────────────────────────────────────────
    if args.validate and not state.is_validated(dataset_key):
        if args.dry_run:
            logger.info("[DRY-RUN] Would validate train/val splits")
        else:
            if _HAS_VALIDATION:
                for split in ("train", "val"):
                    split_dir = dataset_output / split
                    if split_dir.exists():
                        logger.info("Validating %s/%s ...", dataset_key, split)
                        report = validate_vision_dataset(split_dir)
                        print(report.summary())
                        if not report.ok:
                            logger.warning("Validation issues in %s/%s", dataset_key, split)
                state.completed_validations.append(dataset_key)
                state.save(state_path)
            else:
                logger.info("Validation module unavailable, skipping.")
    else:
        if state.is_validated(dataset_key):
            logger.info("Validation already completed (resume).")

    logger.info("Dataset '%s' finished successfully.", dataset_key)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_train_val",
        description="Download benchmark datasets and auto-split train/val for Vision Banana",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list
  %(prog)s download --dataset nyu --output data/
  %(prog)s download --dataset all --output data/ --resume --validate
  %(prog)s download --dataset diode --output data/diode --val-ratio 0.2 --seed 42
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list
    subparsers.add_parser("list", help="List every supported dataset")

    # download
    dl_parser = subparsers.add_parser("download", help="Download and split datasets")
    dl_parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset key or 'all' (see `list` sub-command)",
    )
    dl_parser.add_argument(
        "--output", type=str, default="data",
        help="Root output directory (default: data)",
    )
    dl_parser.add_argument(
        "--download-dir", type=str, default="downloads",
        help="Temporary directory for raw downloads (default: downloads)",
    )
    dl_parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation set ratio, 0~1 (default: 0.1)",
    )
    dl_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible split (default: 42)",
    )
    dl_parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limit max samples per dataset (for quick testing)",
    )
    dl_parser.add_argument(
        "--resolution", type=int, default=None,
        help="Resize images to square resolution (e.g. 512)",
    )
    dl_parser.add_argument(
        "--resume", action="store_true",
        help="Enable resume mode: skip already-completed phases",
    )
    dl_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files instead of skipping",
    )
    dl_parser.add_argument(
        "--validate", action="store_true",
        help="Run validate_vision_dataset() after preparation",
    )
    dl_parser.add_argument(
        "--dry-run", action="store_true",
        help="Print planned actions without executing",
    )
    dl_parser.add_argument(
        "--timeout", type=int, default=600,
        help="Download timeout in seconds per file (default: 600)",
    )
    dl_parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    dl_parser.add_argument(
        "--log-file", type=str, default=None,
        help="Optional log file path",
    )

    return parser


def setup_logging(level: str, log_file: str | None) -> None:
    """Configure structured logging."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def cmd_list() -> int:
    """Run the list subcommand."""
    print("\nSupported datasets (Dataset Registry):")
    print("=" * 70)
    for key, cfg in sorted(DATASET_REGISTRY.items()):
        status = "🟢" if cfg.urls or cfg.huggingface else "🔴"
        print(f"\n  {status} {key:20s} | {cfg.name}")
        print(f"     {'':20s} Task: {cfg.task}")
        print(f"     {'':20s} Size: ~{cfg.size_mb} MB")
        print(f"     {'':20s} License: {cfg.license}")
        print(f"     {'':20s} {cfg.description}")
        if cfg.notes:
            print(f"     {'':20s} ⚠️  {cfg.notes}")
    print()
    auto = [k for k, c in DATASET_REGISTRY.items() if c.urls or c.huggingface]
    manual = [k for k, c in DATASET_REGISTRY.items() if not c.urls and not c.huggingface]
    print(f"Automatic: {', '.join(auto) or 'None'}")
    print(f"Manual:    {', '.join(manual) or 'None'}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Run the download subcommand."""
    setup_logging(args.log_level, args.log_file)

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / STATE_FILENAME
    state = DownloadState.load(state_path) if args.resume else DownloadState()

    if args.dataset == "all":
        datasets = list(DATASET_REGISTRY.keys())
    else:
        if args.dataset not in DATASET_REGISTRY:
            logger.error("Unknown dataset '%s'. Run `list` to see available options.", args.dataset)
            return 1
        datasets = [args.dataset]

    results: dict[str, bool] = {}
    for ds_key in datasets:
        success = download_and_prepare(ds_key, args, state, state_path)
        results[ds_key] = success

    # ── Summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Download & Split Summary")
    logger.info("=" * 60)
    total_ok = 0
    for ds_key, ok in results.items():
        status = "✅ Success" if ok else "❌ Failed"
        logger.info("  %-20s | %s", ds_key, status)
        if ok:
            total_ok += 1

    logger.info("-" * 60)
    logger.info("Total: %d / %d datasets succeeded", total_ok, len(datasets))

    # clean up the temporary unified directory (optional)
    # if not args.dry_run:
    #     for ds_key in datasets:
    #         unified = output_root / ds_key / "unified"
    #         if unified.exists():
    #             shutil.rmtree(unified)
    #             logger.debug("Cleaned up unified dir: %s", unified)

    return 0 if total_ok == len(datasets) else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "list":
        return cmd_list()
    if args.command == "download":
        return cmd_download(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
