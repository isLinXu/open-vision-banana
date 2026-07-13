#!/usr/bin/env python3
"""一键下载 Vision Banana 全量数据集（训练 / 验证 / 测试）。

论文指定评估基准 + 训练数据集，自动下载、解压、格式化到标准目录结构。

数据集清单
==========
训练数据:
  - COCO 2017 (train/val)          — 图像生成 + 深度估计伪标签
  - ADE20K (train/val)             — 语义分割
  - BSDS500 (train/val/test)       — 边缘检测
  - Hypersim (selected scenes)     — 深度估计 (ground truth)

评估基准 (论文指定):
  - Cityscapes val                 — 语义分割评估
  - RefCOCOg UMD val               — 指代分割评估
  - ReasonSeg val                  — 推理分割评估
  - GenAI-Bench                    — 图像生成评估

Usage:
    # 下载全部数据集（训练 + 评估）
    python scripts/download_all_datasets.py

    # 只下载训练数据
    python scripts/download_all_datasets.py --split train

    # 只下载评估基准
    python scripts/download_all_datasets.py --split eval

    # 指定输出目录
    python scripts/download_all_datasets.py --output-dir /data/datasets

    # Dry-run（只打印计划）
    python scripts/download_all_datasets.py --dry-run

    # 列出所有数据集
    python scripts/download_all_datasets.py --list

Requires: wget/curl, unzip, tar (system tools), huggingface_hub, torch
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── HF mirror for faster downloads in China ──────────────────────────────
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_data")


# ─── Dataclass for dataset specs ──────────────────────────────────────────

@dataclass
class DatasetSpec:
    """Specification for a downloadable dataset."""
    name: str
    task: str
    split: str  # "train" | "eval"
    source_type: str  # "url" | "huggingface" | "manual"
    source: str
    size_gb: float
    output_subdir: str
    description: str
    license: str = "unknown"
    requires_token: bool = False
    prepare_fn: Callable[[Path, Path], int] | None = None
    optional: bool = False


# ─── Output directory helpers ─────────────────────────────────────────────

DEFAULT_OUTPUT = Path("data")

TRAIN_DIR = "train"
EVAL_DIR = "eval"


# ─── Dataset registry ────────────────────────────────────────────────────

DATASETS: list[DatasetSpec] = [
    # ════════════════════════════════════════════════════════════════════
    # TRAINING DATASETS
    # ════════════════════════════════════════════════════════════════════

    DatasetSpec(
        name="coco2017",
        task="generation+depth",
        split="train",
        source_type="url",
        source="http://images.cocodataset.org/zips/train2017.zip",
        size_gb=18.0,
        output_subdir=f"{TRAIN_DIR}/coco2017",
        description="COCO 2017 train images (118K) — generation + pseudo depth",
        license="CC-BY-4.0",
    ),
    DatasetSpec(
        name="coco2017-val",
        task="generation+depth",
        split="train",
        source_type="url",
        source="http://images.cocodataset.org/zips/val2017.zip",
        size_gb=0.8,
        output_subdir=f"{TRAIN_DIR}/coco2017/val2017",
        description="COCO 2017 val images (5K) — generation + pseudo depth",
        license="CC-BY-4.0",
    ),
    DatasetSpec(
        name="coco2017-annotations",
        task="generation",
        split="train",
        source_type="url",
        source="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        size_gb=0.25,
        output_subdir=f"{TRAIN_DIR}/coco2017/annotations",
        description="COCO 2017 captions annotations",
        license="CC-BY-4.0",
    ),
    DatasetSpec(
        name="ade20k",
        task="segmentation",
        split="train",
        source_type="url",
        source="http://data.csail.mit.edu/places/ADEChallenge/ADEChallengeData2016.zip",
        size_gb=0.9,
        output_subdir=f"{TRAIN_DIR}/ade20k",
        description="ADE20K semantic segmentation (20K train + 2K val)",
        license="BSD",
    ),
    DatasetSpec(
        name="bsds500",
        task="edge",
        split="train",
        source_type="url",
        source="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semanticContours/bsdS500PicsSEGM.tgz",
        size_gb=0.03,
        output_subdir=f"{TRAIN_DIR}/bsds500",
        description="BSDS500 boundary detection (200 train + 100 val + 200 test)",
        license="BSD",
    ),
    DatasetSpec(
        name="hypersim",
        task="depth",
        split="train",
        source_type="huggingface",
        source="dl3d/hypersim",
        size_gb=50.0,
        output_subdir=f"{TRAIN_DIR}/hypersim",
        description="Hypersim synthetic indoor depth (selected scenes)",
        license="CC-BY-NC-4.0",
        optional=True,  # Large, can use COCO pseudo depth instead
    ),

    # ════════════════════════════════════════════════════════════════════
    # EVALUATION BENCHMARKS (paper-specified)
    # ════════════════════════════════════════════════════════════════════

    DatasetSpec(
        name="cityscapes",
        task="segmentation",
        split="eval",
        source_type="url",
        source="https://www.cityscapes-dataset.com/file-handling/?packageID=3",  # val images
        size_gb=0.1,
        output_subdir=f"{EVAL_DIR}/cityscapes",
        description="Cityscapes val — semantic segmentation benchmark (paper-specified)",
        license="Cityscapes License (non-commercial)",
        requires_token=True,  # Requires cityscapes login
        optional=False,
    ),
    DatasetSpec(
        name="refcocog",
        task="referring_segmentation",
        split="eval",
        source_type="url",
        source="https://bvisionweb.cs.unc.edu/lichen/projects2/refcoco/refcocog.zip",
        size_gb=1.2,
        output_subdir=f"{EVAL_DIR}/refcocog",
        description="RefCOCOg UMD val — referring expression segmentation (paper-specified)",
        license="BSD",
    ),
    DatasetSpec(
        name="reasonseg",
        task="reasoning_segmentation",
        split="eval",
        source_type="huggingface",
        source="lmsys/ReasonSeg",
        size_gb=2.0,
        output_subdir=f"{EVAL_DIR}/reasonseg",
        description="ReasonSeg val — reasoning segmentation (paper-specified)",
        license="Apache-2.0",
    ),
    DatasetSpec(
        name="genai-bench",
        task="generation",
        split="eval",
        source_type="huggingface",
        source="BaiqiLian/GenAI-Bench",
        size_gb=0.5,
        output_subdir=f"{EVAL_DIR}/genai_bench",
        description="GenAI-Bench — image generation evaluation (paper-specified)",
        license="MIT",
    ),
]


# ─── Download helpers ─────────────────────────────────────────────────────

def _download_url(url: str, dest: Path, dry_run: bool = False) -> bool:
    """Download a file from URL to dest. Returns True on success."""
    if dest.exists():
        logger.info(f"  Already exists: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    if dry_run:
        logger.info(f"  [DRY-RUN] Would download: {url}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading: {url}")

    # Try wget first (better resume support), fallback to urllib
    try:
        result = subprocess.run(
            ["wget", "-c", "-q", "--show-progress", "-O", str(dest), url],
            check=True,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fallback: urllib
    try:
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _download_hf(repo_id: str, dest: Path, dry_run: bool = False) -> bool:
    """Download from HuggingFace Hub."""
    if dest.exists() and any(dest.iterdir()):
        logger.info(f"  Already exists: {dest}")
        return True

    if dry_run:
        logger.info(f"  [DRY-RUN] Would download from HF: {repo_id}")
        return False

    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Downloading from HuggingFace: {repo_id}")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            resume_download=True,
        )
        logger.info(f"  Downloaded to: {path}")
        return True
    except Exception as e:
        logger.error(f"  HF download failed: {e}")
        return False


def _extract_archive(archive: Path, dest: Path, dry_run: bool = False) -> bool:
    """Extract zip/tgz/tar.gz archive to dest directory."""
    if dry_run:
        logger.info(f"  [DRY-RUN] Would extract: {archive.name}")
        return False

    dest.mkdir(parents=True, exist_ok=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(str(dest))
    elif archive.suffix in (".tgz", ".gz") or archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(str(dest))
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r") as tf:
            tf.extractall(str(dest))
    else:
        logger.warning(f"  Unknown archive format: {archive.name}")
        return False

    logger.info(f"  Extracted to: {dest}")
    return True


# ─── Per-dataset preparation ──────────────────────────────────────────────

def prepare_coco2017(raw_dir: Path, out_dir: Path) -> int:
    """Prepare COCO 2017: extract + organize images + captions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # Extract train images
    train_zip = raw_dir / "train2017.zip"
    if train_zip.exists():
        train_dir = out_dir / "train2017"
        if not train_dir.exists():
            _extract_archive(train_zip, out_dir)
        imgs = list((out_dir / "train2017").glob("*.jpg"))
        count += len(imgs)
        logger.info(f"  COCO train2017: {len(imgs)} images")

    # Extract val images
    val_zip = raw_dir / "val2017.zip"
    if val_zip.exists():
        val_dir = out_dir / "val2017"
        if not val_dir.exists():
            _extract_archive(val_zip, out_dir)
        imgs = list((out_dir / "val2017").glob("*.jpg"))
        count += len(imgs)
        logger.info(f"  COCO val2017: {len(imgs)} images")

    # Extract annotations (captions)
    ann_zip = raw_dir / "annotations_trainval2017.zip"
    if ann_zip.exists():
        ann_dir = out_dir / "annotations"
        if not ann_dir.exists():
            _extract_archive(ann_zip, out_dir)
        logger.info(f"  COCO annotations extracted")

    return count


def prepare_ade20k(raw_dir: Path, out_dir: Path) -> int:
    """Prepare ADE20K: extract and organize into images/ + targets/."""
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = raw_dir / "ADEChallengeData2016.zip"
    if not archive.exists():
        # Try already-extracted
        data_dir = out_dir / "ADEChallengeData2016"
        if data_dir.exists():
            return len(list((data_dir / "images" / "training").glob("*.jpg")))
        return 0

    if not (out_dir / "ADEChallengeData2016").exists():
        _extract_archive(archive, out_dir)

    data_dir = out_dir / "ADEChallengeData2016"
    train_count = len(list((data_dir / "images" / "training").glob("*.jpg")))
    val_count = len(list((data_dir / "images" / "validation").glob("*.jpg")))
    logger.info(f"  ADE20K: {train_count} train + {val_count} val")
    return train_count + val_count


def prepare_bsds500(raw_dir: Path, out_dir: Path) -> int:
    """Prepare BSDS500: extract boundary detection data."""
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = raw_dir / "bsdS500PicsSEGM.tgz"
    if not archive.exists():
        return 0

    if not (out_dir / "BSDS500").exists():
        _extract_archive(archive, out_dir)

    bsds = out_dir / "BSDS500"
    # Symlink or copy to expected structure
    for split in ["train", "val", "test"]:
        img_dir = bsds / "data" / split
        if img_dir.exists():
            logger.info(f"  BSDS500 {split}: {len(list(img_dir.glob('*.jpg')))} images")

    return 800  # 200 train + 100 val + 200 test (approx)


# ─── Main download pipeline ──────────────────────────────────────────────

def download_dataset(spec: DatasetSpec, output_dir: Path, dry_run: bool = False) -> bool:
    """Download and prepare a single dataset."""
    dest = output_dir / spec.output_subdir
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {spec.name} ({spec.task})")
    logger.info(f"  Size: ~{spec.size_gb:.1f} GB | Split: {spec.split}")
    logger.info(f"  Output: {dest}")
    logger.info(f"  Source: {spec.source_type} — {spec.source}")
    if spec.requires_token:
        logger.info(f"  ⚠️  Requires login/token for download")
    if spec.optional:
        logger.info(f"  (optional — can skip if not needed)")

    if spec.source_type == "manual":
        logger.warning(f"  Manual download required. Please visit: {spec.source}")
        logger.info(f"  After downloading, place files in: {dest}")
        return False

    # Create raw download dir
    raw_dir = dest / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if spec.source_type == "url":
        filename = spec.source.split("/")[-1]
        # Handle query params in URL
        if "?" in filename:
            filename = filename.split("?")[0]
        archive_path = raw_dir / filename

        success = _download_url(spec.source, archive_path, dry_run)
        if not success and not dry_run:
            return False

        if not dry_run and spec.prepare_fn:
            spec.prepare_fn(raw_dir, dest)
        elif not dry_run:
            # Auto-extract
            if archive_path.exists():
                _extract_archive(archive_path, dest)

    elif spec.source_type == "huggingface":
        success = _download_hf(spec.source, dest, dry_run)
        if not success and not dry_run:
            return False

    logger.info(f"  ✅ Done: {spec.name}")
    return True


def run_download(
    output_dir: Path,
    split: str = "all",
    dry_run: bool = False,
    skip_optional: bool = False,
    only: list[str] | None = None,
) -> None:
    """Download datasets matching the filter criteria."""
    specs = DATASETS

    # Filter by split
    if split == "train":
        specs = [s for s in specs if s.split == "train"]
    elif split == "eval":
        specs = [s for s in specs if s.split == "eval"]

    # Filter by name
    if only:
        specs = [s for s in specs if s.name in only]

    # Skip optional
    if skip_optional:
        specs = [s for s in specs if not s.optional]

    total_size = sum(s.size_gb for s in specs)
    logger.info(f"\n{'='*60}")
    logger.info(f"Vision Banana — Dataset Download Pipeline")
    logger.info(f"{'='*60}")
    logger.info(f"Datasets to download: {len(specs)}")
    logger.info(f"Estimated total size: ~{total_size:.1f} GB")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split: {split} | Dry-run: {dry_run}")
    logger.info(f"{'='*60}\n")

    if dry_run:
        for spec in specs:
            logger.info(f"  [DRY-RUN] {spec.name:25s} | {spec.size_gb:6.1f} GB | {spec.source}")
        logger.info(f"\nTotal: ~{total_size:.1f} GB across {len(specs)} datasets")
        return

    success_count = 0
    fail_count = 0
    skipped = []

    for spec in specs:
        try:
            ok = download_dataset(spec, output_dir, dry_run)
            if ok:
                success_count += 1
            elif spec.optional:
                skipped.append(spec.name)
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"  Failed: {spec.name} — {e}")
            fail_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Download Summary")
    logger.info(f"{'='*60}")
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Failed:  {fail_count}")
    if skipped:
        logger.info(f"  ⏭️  Skipped (optional): {', '.join(skipped)}")
    logger.info(f"  Output: {output_dir}")

    # Print directory structure
    logger.info(f"\n{'='*60}")
    logger.info(f"Directory structure:")
    for spec in specs:
        dest = output_dir / spec.output_subdir
        exists = "✅" if dest.exists() else "❌"
        logger.info(f"  {exists} {spec.output_subdir:40s} ({spec.task})")


def list_datasets() -> None:
    """List all registered datasets in a table."""
    print(f"\n{'='*90}")
    print(f"Vision Banana — Dataset Registry")
    print(f"{'='*90}")
    print(f"{'Name':<25} {'Task':<25} {'Split':<6} {'Size(GB)':<10} {'Source Type':<15} {'Optional'}")
    print(f"{'-'*90}")

    for spec in DATASETS:
        print(f"{spec.name:<25} {spec.task:<25} {spec.split:<6} {spec.size_gb:<10.1f} "
              f"{spec.source_type:<15} {'✓' if spec.optional else ''}")

    total = sum(s.size_gb for s in DATASETS)
    train_total = sum(s.size_gb for s in DATASETS if s.split == "train")
    eval_total = sum(s.size_gb for s in DATASETS if s.split == "eval")
    print(f"{'-'*90}")
    print(f"{'TOTAL':<25} {'':<25} {'':<6} {total:<10.1f}")
    print(f"{'  train':<25} {'':<25} {'':<6} {train_total:<10.1f}")
    print(f"{'  eval':<25} {'':<25} {'':<6} {eval_total:<10.1f}")
    print(f"{'='*90}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="一键下载 Vision Banana 全量数据集（训练 + 评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 下载全部数据集
    python scripts/download_all_datasets.py

    # 只下载训练数据
    python scripts/download_all_datasets.py --split train

    # 只下载评估基准
    python scripts/download_all_datasets.py --split eval

    # 指定输出目录
    python scripts/download_all_datasets.py --output-dir /data/datasets

    # 跳过可选的大数据集（如 Hypersim 50GB）
    python scripts/download_all_datasets.py --skip-optional

    # 只下载特定数据集
    python scripts/download_all_datasets.py --only coco2017 ade20k

    # Dry-run
    python scripts/download_all_datasets.py --dry-run

    # 列出所有数据集
    python scripts/download_all_datasets.py --list
        """,
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory (default: data/)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "eval"],
                        help="Which split to download (default: all)")
    parser.add_argument("--skip-optional", action="store_true",
                        help="Skip optional datasets (e.g. Hypersim 50GB)")
    parser.add_argument("--only", nargs="+", type=str, default=None,
                        help="Only download specific datasets by name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without downloading")
    parser.add_argument("--list", action="store_true",
                        help="List all registered datasets")

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_download(
        output_dir=output_dir,
        split=args.split,
        dry_run=args.dry_run,
        skip_optional=args.skip_optional,
        only=args.only,
    )


if __name__ == "__main__":
    main()
