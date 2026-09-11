#!/usr/bin/env python3
"""Download TartanAir synthetic depth dataset for Vision Banana.

TartanAir is a fully synthetic dataset collected in Unreal Engine 4 environments,
providing RGB images, metric depth maps, semantic segmentation, optical flow,
and camera poses. This script downloads only the depth and RGB modalities needed
for depth-estimation training, ensuring zero real-world depth data is used.

Data source:
  - AirLab Ceph RGW (default): https://airlab-cloud.andrew.cmu.edu:8080
  - HuggingFace (alternative): theairlabcmu/tartanair
  - Cloudflare R2 (alternative): Scale Foundation mirror

The dataset is published under a Creative Commons Attribution 4.0 International License.

Usage:
    # Download all environments (Easy only, left camera, depth+image)
    python scripts/download_tartanair.py --output data/tartanair/

    # Download a small subset for quick validation
    python scripts/download_tartanair.py --output data/tartanair/ --subset tiny

    # Download specific environments with both Easy and Hard difficulties
    python scripts/download_tartanair.py --output data/tartanair/ \
        --envs abandonedfactory gascola hospital --difficulty both

    # Use HuggingFace backend (good for regions with slow AirLab access)
    python scripts/download_tartanair.py --output data/tartanair/ --backend huggingface

    # Dry run: show what would be downloaded without fetching
    python scripts/download_tartanair.py --output data/tartanair/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Built-in TartanAir V1 file manifest (derived from official
# download_training_zipfiles.txt to avoid runtime external dependencies)
# ──────────────────────────────────────────────────────────────

TARTANAIR_V1_ZIPFILES: dict[str, list[tuple[str, float]]] = {
    "abandonedfactory": [
        ("abandonedfactory/Easy/depth_left.zip", 2.998),
        ("abandonedfactory/Easy/image_left.zip", 7.075),
        ("abandonedfactory/Hard/depth_left.zip", 2.195),
        ("abandonedfactory/Hard/image_left.zip", 5.013),
    ],
    "abandonedfactory_night": [
        ("abandonedfactory_night/Easy/depth_left.zip", 2.923),
        ("abandonedfactory_night/Easy/image_left.zip", 6.036),
        ("abandonedfactory_night/Hard/depth_left.zip", 2.070),
        ("abandonedfactory_night/Hard/image_left.zip", 4.154),
    ],
    "amusement": [
        ("amusement/Easy/depth_left.zip", 2.845),
        ("amusement/Easy/image_left.zip", 5.186),
        ("amusement/Hard/depth_left.zip", 2.128),
        ("amusement/Hard/image_left.zip", 3.948),
    ],
    "carwelding": [
        ("carwelding/Easy/depth_left.zip", 0.806),
        ("carwelding/Easy/image_left.zip", 2.001),
        ("carwelding/Hard/depth_left.zip", 0.491),
        ("carwelding/Hard/image_left.zip", 1.155),
    ],
    "endofworld": [
        ("endofworld/Easy/depth_left.zip", 1.355),
        ("endofworld/Easy/image_left.zip", 4.374),
        ("endofworld/Hard/depth_left.zip", 0.436),
        ("endofworld/Hard/image_left.zip", 1.496),
    ],
    "gascola": [
        ("gascola/Easy/depth_left.zip", 4.456),
        ("gascola/Easy/image_left.zip", 7.114),
        ("gascola/Hard/depth_left.zip", 3.528),
        ("gascola/Hard/image_left.zip", 5.446),
    ],
    "hospital": [
        ("hospital/Easy/depth_left.zip", 2.612),
        ("hospital/Easy/image_left.zip", 7.193),
        ("hospital/Hard/depth_left.zip", 1.030),
        ("hospital/Hard/image_left.zip", 2.693),
    ],
    "japanesealley": [
        ("japanesealley/Easy/depth_left.zip", 0.839),
        ("japanesealley/Easy/image_left.zip", 1.734),
        ("japanesealley/Hard/depth_left.zip", 0.516),
        ("japanesealley/Hard/image_left.zip", 1.059),
    ],
    "neighborhood": [
        ("neighborhood/Easy/depth_left.zip", 6.867),
        ("neighborhood/Easy/image_left.zip", 11.177),
        ("neighborhood/Hard/depth_left.zip", 4.348),
        ("neighborhood/Hard/image_left.zip", 6.898),
    ],
    "ocean": [
        ("ocean/Easy/depth_left.zip", 1.835),
        ("ocean/Easy/image_left.zip", 3.441),
        ("ocean/Hard/depth_left.zip", 1.362),
        ("ocean/Hard/image_left.zip", 2.556),
    ],
    "office": [
        ("office/Easy/depth_left.zip", 1.109),
        ("office/Easy/image_left.zip", 3.309),
        ("office/Hard/depth_left.zip", 0.681),
        ("office/Hard/image_left.zip", 1.945),
    ],
    "office2": [
        ("office2/Easy/depth_left.zip", 1.077),
        ("office2/Easy/image_left.zip", 3.082),
        ("office2/Hard/depth_left.zip", 1.003),
        ("office2/Hard/image_left.zip", 2.818),
    ],
    "oldtown": [
        ("oldtown/Easy/depth_left.zip", 1.772),
        ("oldtown/Easy/image_left.zip", 4.463),
        ("oldtown/Hard/depth_left.zip", 1.380),
        ("oldtown/Hard/image_left.zip", 3.633),
    ],
    "seasidetown": [
        ("seasidetown/Easy/depth_left.zip", 1.168),
        ("seasidetown/Easy/image_left.zip", 3.180),
        ("seasidetown/Hard/depth_left.zip", 0.570),
        ("seasidetown/Hard/image_left.zip", 1.441),
    ],
    "seasonsforest": [
        ("seasonsforest/Easy/depth_left.zip", 1.845),
        ("seasonsforest/Easy/image_left.zip", 3.096),
        ("seasonsforest/Hard/depth_left.zip", 0.792),
        ("seasonsforest/Hard/image_left.zip", 1.303),
    ],
    "seasonsforest_winter": [
        ("seasonsforest_winter/Easy/depth_left.zip", 3.493),
        ("seasonsforest_winter/Easy/image_left.zip", 5.621),
        ("seasonsforest_winter/Hard/depth_left.zip", 2.517),
        ("seasonsforest_winter/Hard/image_left.zip", 4.123),
    ],
    "soulcity": [
        ("soulcity/Easy/depth_left.zip", 2.604),
        ("soulcity/Easy/image_left.zip", 6.885),
        ("soulcity/Hard/depth_left.zip", 0.916),
        ("soulcity/Hard/image_left.zip", 2.425),
    ],
    "westerndesert": [
        ("westerndesert/Easy/depth_left.zip", 1.320),
        ("westerndesert/Easy/image_left.zip", 2.090),
        ("westerndesert/Hard/depth_left.zip", 0.896),
        ("westerndesert/Hard/image_left.zip", 1.704),
    ],
}

ENVIRONMENT_SUBSETS = {
    "tiny": ["abandonedfactory"],
    "small": ["abandonedfactory", "gascola", "hospital", "seasonsforest"],
    "medium": [
        "abandonedfactory",
        "amusement",
        "gascola",
        "hospital",
        "neighborhood",
        "seasonsforest",
        "soulcity",
    ],
    "full": list(TARTANAIR_V1_ZIPFILES.keys()),
}


# ──────────────────────────────────────────────────────────────
# Download backends
# ──────────────────────────────────────────────────────────────

@dataclass
class DownloadBackend:
    name: str
    download_fn: Callable[[list[str], Path, int], tuple[bool, list[str]]]


def _make_airlab_downloader() -> DownloadBackend:
    """Create AirLab Ceph RGW downloader using boto3."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for AirLab backend. Install: pip install boto3"
        ) from exc

    endpoint_url = (
        "https://airlab-cloud.andrew.cmu.edu:8080/"
        "swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
    )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        config=Config(signature_version=UNSIGNED),
    )
    bucket = "tartanair"

    def _download(
        filelist: list[str], dest: Path, workers: int
    ) -> tuple[bool, list[str]]:
        dest.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        had_error = False

        def _download_one(key: str) -> tuple[str, str, str]:
            target = dest / key.replace("/", "_")
            if target.exists():
                return key, str(target), "exists"
            try:
                resp = client.get_object(Bucket=bucket, Key=key)
                with open(target, "wb") as f:
                    for chunk in resp["Body"].iter_chunks(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                return key, str(target), "ok"
            except Exception as e:
                logger.error("Failed to download %s: %s", key, e)
                return key, str(target), f"error: {e}"

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_download_one, k): k for k in filelist}
            for fut in as_completed(futs):
                src, tgt, status = fut.result()
                downloaded.append(tgt)
                if status == "ok":
                    logger.info("  ✓ %s", src)
                elif status == "exists":
                    logger.info("  ⏭ %s (exists)", src)
                else:
                    had_error = True
                    logger.error("  ✗ %s: %s", src, status)

        return not had_error, downloaded

    return DownloadBackend(name="airlab", download_fn=_download)


def _make_huggingface_downloader() -> DownloadBackend:
    """Create HuggingFace downloader using huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HuggingFace backend. "
            "Install: pip install huggingface_hub"
        ) from exc

    repo_id = "theairlabcmu/tartanair"
    chunk_size = 50

    def _download(
        filelist: list[str], dest: Path, _workers: int
    ) -> tuple[bool, list[str]]:
        dest.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        had_error = False

        # Build allow_patterns from filelist by converting zip paths to glob patterns
        # HF snapshot_download supports allow_patterns like "abandonedfactory/Easy/*_left.zip"
        patterns = list({f"{path.split('/')[0]}/{path.split('/')[1]}/*_left.zip" for path in filelist})

        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(dest),
                allow_patterns=patterns,
            )
            # Map expected files to local paths
            for path in filelist:
                local = dest / path
                if local.exists():
                    downloaded.append(str(local))
                else:
                    # Try alternate naming (HF may flatten)
                    alt = dest / path.replace("/", "_")
                    if alt.exists():
                        downloaded.append(str(alt))
                    else:
                        had_error = True
                        logger.error("File not found after HF download: %s", path)
        except Exception as e:
            logger.error("HuggingFace download failed: %s", e)
            had_error = True

        return not had_error, downloaded

    return DownloadBackend(name="huggingface", download_fn=_download)


def _make_cloudflare_downloader() -> DownloadBackend:
    """Create Cloudflare R2 downloader."""
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for Cloudflare backend. Install: pip install boto3"
        ) from exc

    access_key = "f1ae9efebbc6a9a7cebbd949ba3a12de"
    secret_key = "0a21fe771089d82e048ed0a1dd6067cb29a5666bf4fe95f7be9ba6f72482ec8b"
    endpoint_url = "https://0a585e9484af268a716f8e6d3be53bbc.r2.cloudflarestorage.com"
    bucket = "tartanair-v1"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
    )

    def _download(
        filelist: list[str], dest: Path, _workers: int
    ) -> tuple[bool, list[str]]:
        dest.mkdir(parents=True, exist_ok=True)
        downloaded: list[str] = []
        had_error = False

        for key in filelist:
            target = dest / key.replace("/", "_")
            if target.exists():
                downloaded.append(str(target))
                logger.info("  ⏭ %s (exists)", key)
                continue
            try:
                # Cloudflare has a "tartanair" prefix folder
                s3_key = f"tartanair/{key}"
                s3.download_file(bucket, s3_key, str(target))
                downloaded.append(str(target))
                logger.info("  ✓ %s", key)
            except Exception as e:
                had_error = True
                logger.error("  ✗ %s: %s", key, e)

        return not had_error, downloaded

    return DownloadBackend(name="cloudflare", download_fn=_download)


def get_backend(name: str) -> DownloadBackend:
    if name == "airlab":
        return _make_airlab_downloader()
    if name == "huggingface":
        return _make_huggingface_downloader()
    if name == "cloudflare":
        return _make_cloudflare_downloader()
    raise ValueError(f"Unknown backend: {name}. Choose from: airlab, huggingface, cloudflare")


# ──────────────────────────────────────────────────────────────
# File selection and filtering
# ──────────────────────────────────────────────────────────────

def select_files(
    envs: list[str],
    difficulty: str,
) -> list[tuple[str, float]]:
    """Select zip files matching the requested environments and difficulty.

    Args:
        envs: List of environment names.
        difficulty: "easy", "hard", or "both".

    Returns:
        List of (zip_path, size_gb) tuples.
    """
    selected: list[tuple[str, float]] = []
    for env in envs:
        if env not in TARTANAIR_V1_ZIPFILES:
            logger.warning("Unknown environment: %s, skipping", env)
            continue
        for path, size in TARTANAIR_V1_ZIPFILES[env]:
            diff = path.split("/")[1].lower()
            if difficulty == "both":
                selected.append((path, size))
            elif difficulty == diff:
                selected.append((path, size))
    return selected


# ──────────────────────────────────────────────────────────────
# Unzip and organize
# ──────────────────────────────────────────────────────────────

def _extract_zip(zip_path: Path, dest: Path) -> bool:
    """Extract a zip file to dest, preserving internal structure."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        return True
    except zipfile.BadZipFile as e:
        logger.error("Bad zip file %s: %s", zip_path, e)
        return False
    except Exception as e:
        logger.error("Failed to extract %s: %s", zip_path, e)
        return False


def organize_extracted(raw_dir: Path, output_dir: Path) -> None:
    """Move extracted zip contents into the canonical TartanAir layout.

    After extracting, zip contents are typically under:
        raw_dir/ENV_NAME/Difficulty/P000/image_left/...

    We ensure the output_dir mirrors this structure.
    """
    # HuggingFace snapshot_download may place files with repo structure.
    # The zip files themselves are already in the correct relative paths
    # when downloaded via AirLab/Cloudflare (they are flattened with _).
    # For HuggingFace, files may be in repo structure.
    pass


def unzip_and_cleanup(zip_paths: list[str], output_dir: Path, delete_zip: bool = False) -> list[str]:
    """Extract all downloaded zip files to output_dir.

    Returns list of extracted zip paths that succeeded.
    """
    success: list[str] = []
    for zp in zip_paths:
        zip_path = Path(zp)
        if not zip_path.exists():
            logger.warning("Zip file not found: %s", zip_path)
            continue

        logger.info("Extracting %s → %s", zip_path.name, output_dir)
        if _extract_zip(zip_path, output_dir):
            success.append(zp)
            if delete_zip:
                zip_path.unlink()
                logger.info("  Deleted %s", zip_path.name)
        else:
            logger.error("  Extraction failed for %s", zip_path.name)

    return success


# ──────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TartanAir synthetic depth dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="medium",
        choices=list(ENVIRONMENT_SUBSETS.keys()),
        help="Subset of environments to download",
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="Override subset: specify individual environment names",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=["easy", "hard", "both"],
        help="Difficulty level to download",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="airlab",
        choices=["airlab", "huggingface", "cloudflare"],
        help="Download backend",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        default=True,
        help="Extract zip files after downloading (default: True)",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        default=False,
        help="Skip extraction (keep zip files only)",
    )
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        default=False,
        help="Delete zip files after successful extraction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be downloaded without downloading",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine environments
    envs = args.envs if args.envs else ENVIRONMENT_SUBSETS[args.subset]
    logger.info("Selected environments (%d): %s", len(envs), ", ".join(envs))
    logger.info("Difficulty: %s", args.difficulty)
    logger.info("Backend: %s", args.backend)

    # Select files
    selected = select_files(envs, args.difficulty)
    if not selected:
        logger.error("No files match the selected criteria.")
        return 1

    total_gb = sum(s for _, s in selected)
    logger.info("Files to download: %d (%.1f GB estimated)", len(selected), total_gb)

    if args.dry_run:
        print("\n[DRY RUN] Would download:\n")
        for path, size in selected:
            print(f"  {path:<60} {size:.2f} GB")
        print(f"\n  Total: {len(selected)} files, {total_gb:.1f} GB")
        print(f"  Output: {output_dir.resolve()}")
        return 0

    # Download
    raw_zip_dir = output_dir / "_zip_downloads"
    raw_zip_dir.mkdir(parents=True, exist_ok=True)

    backend = get_backend(args.backend)
    filelist = [path for path, _ in selected]

    logger.info("Starting download via %s backend...", backend.name)
    success, downloaded = backend.download_fn(filelist, raw_zip_dir, args.workers)

    if not success:
        logger.warning("Some downloads failed. Check logs above.")

    # Extract
    if args.unzip and not args.no_unzip:
        logger.info("Extracting %d zip files...", len(downloaded))
        extracted = unzip_and_cleanup(
            downloaded, output_dir, delete_zip=args.delete_zip
        )
        logger.info("Successfully extracted %d / %d zip files", len(extracted), len(downloaded))
    else:
        logger.info("Download complete. Zip files saved to %s", raw_zip_dir)
        logger.info("Run with --unzip to extract, or use: unzip '*.zip'")

    return 0 if success else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
