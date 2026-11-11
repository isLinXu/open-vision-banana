#!/usr/bin/env python3
"""Download the base diffusion model weights Vision Banana supports into the local HuggingFace cache.

Based on the model configs registered in model_registry, this script handles the pretrained weight
download for all 23 model families, with resume, dry-run and memory-budget filtering.

Usage:
    # list every available model and its memory requirement
    python scripts/download_model.py --list

    # download one model (e.g. FLUX.1-dev)
    python scripts/download_model.py --model flux-dev

    # download into a given local cache directory
    python scripts/download_model.py --model ssd-1b --cache-dir /mnt/hf-cache

    # filter downloadable models by memory budget
    python scripts/download_model.py --list --max-memory 8

    # dry run: show what would be downloaded without downloading
    python scripts/download_model.py --model flux-schnell --dry-run

    # force a fresh download (ignore the existing cache)
    python scripts/download_model.py --model sd15 --force
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup: one format, no duplicate handlers
# ---------------------------------------------------------------------------
_logger_name = "download_model"
logger = logging.getLogger(_logger_name)


def _setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger without clashing with existing handlers."""
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# Automatic HF mirror and token setup
# ---------------------------------------------------------------------------
def _ensure_hf_env() -> None:
    """Set the HF mirror and token automatically (when a local config file exists)."""
    # prefer hf-mirror to speed up downloads from mainland China
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # reuse the token if the user is already logged in with huggingface-cli
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists() and "HF_TOKEN" not in os.environ:
        try:
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                os.environ.setdefault("HF_TOKEN", token)
                os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
        except OSError:
            pass  # silently ignore when the file is not readable


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------
def _get_registry() -> dict[str, dict]:
    """Extract model metadata from model_registry without importing torch directly."""
    # deferred import, so --list / --dry-run do not load torch
    try:
        from open_vision_banana.model_registry import REGISTRY, ModelConfig
    except ImportError as exc:
        logger.error(
            "Cannot import open_vision_banana.model_registry: %s\n"
            "Run from the project root or install in editable mode: pip install -e .",
            exc,
        )
        sys.exit(1)

    entries: dict[str, dict] = {}
    for name in REGISTRY:
        cfg: ModelConfig = REGISTRY[name]
        entries[name] = {
            "model_id": cfg.model_id,
            "family": cfg.family,
            "backbone_type": cfg.backbone_type,
            "memory_budget_gb": cfg.memory_budget_gb,
            "default_resolution": cfg.default_resolution,
            "description": cfg.description,
            "pipeline_class": cfg.pipeline_class,
        }
    return entries


def _download_with_resume(
    repo_id: str,
    *,
    cache_dir: str | None = None,
    resume: bool = True,
    force: bool = False,
    local_files_only: bool = False,
) -> Path:
    """Download with resume support via huggingface_hub.snapshot_download.

    Args:
        repo_id: Model repo ID on the HuggingFace Hub, e.g. "black-forest-labs/FLUX.1-dev".
        cache_dir: Local cache directory; defaults to the HF default path (~/.cache/huggingface/hub).
        resume: Whether to resume interrupted downloads (default True).
        force: When True, ignore the local cache and download again.
        local_files_only: Use only the local cache, without network requests.

    Returns:
        Local path of the downloaded snapshot (Path).

    Raises:
        RuntimeError: When snapshot_download returns None or a network error occurs.
    """
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import LocalEntryNotFoundError, RepositoryNotFoundError
    except ImportError as exc:
        logger.error(
            "Missing the huggingface_hub dependency: %s\n"
            "Install the inference extras: pip install -e '.[inference]'",
            exc,
        )
        raise

    logger.info("Starting download: %s", repo_id)
    if cache_dir:
        logger.info("Cache directory: %s", cache_dir)
    logger.info("Resume: %s", "enabled" if resume else "disabled")

    try:
        local_path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            resume_download=resume,
            local_files_only=local_files_only,
            force_download=force,
        )
    except RepositoryNotFoundError as exc:
        logger.error("Model repo does not exist or access is denied: %s", repo_id)
        logger.error(
            "Hint: if this model needs a HuggingFace login, first run:\n"
            "  huggingface-cli login"
        )
        raise RuntimeError(f"RepositoryNotFoundError: {exc}") from exc
    except LocalEntryNotFoundError as exc:
        logger.error("Model not found in the local cache and local_files_only=True: %s", repo_id)
        raise RuntimeError(f"LocalEntryNotFoundError: {exc}") from exc
    except Exception as exc:
        logger.error("Unknown error during download: %s", exc)
        raise RuntimeError(f"Download failed: {exc}") from exc

    if local_path is None:
        raise RuntimeError("snapshot_download returned None, the download failed.")

    return Path(local_path)


def _calculate_dir_size(path: Path) -> float:
    """Total size of a directory, recursively (GB)."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError as exc:
        logger.warning("Error while computing the directory size: %s", exc)
    return total / 1e9


def _print_model_list(entries: dict[str, dict], max_memory: float | None = None) -> None:
    """Print the model list, optionally filtered by memory budget."""
    if max_memory is not None:
        filtered = {
            k: v for k, v in entries.items()
            if v["memory_budget_gb"] <= max_memory
        }
        if not filtered:
            print(f"⚠️  No model fits the memory budget ≤ {max_memory} GB")
            return
        entries_to_show = filtered
        title_suffix = f"(≤ {max_memory} GB)"
    else:
        entries_to_show = entries
        title_suffix = ""

    print(f"\nAvailable models {title_suffix}:")
    print("-" * 95)
    print(
        f"  {'Name':<18} {'Family':<12} {'Backbone':<11} "
        f"{'VRAM(GB)':<9} {'Resolution':<11} {'Model ID'}"
    )
    print("-" * 95)

    for name, info in sorted(entries_to_show.items(), key=lambda x: x[1]["memory_budget_gb"]):
        print(
            f"  {name:<18} {info['family']:<12} {info['backbone_type']:<11} "
            f"{info['memory_budget_gb']:<9.1f} {info['default_resolution']:<11d} "
            f"{info['model_id']}"
        )

    print("-" * 95)
    print(f"{len(entries_to_show)} models in total")
    print("\nDownload examples:")
    print("  python scripts/download_model.py --model flux-dev")
    print("  python scripts/download_model.py --model ssd-1b --cache-dir /mnt/cache")


def _print_model_detail(name: str, info: dict) -> None:
    """Print the details of one model."""
    print(f"\n{'='*60}")
    print(f"Model name:  {name}")
    print(f"Model ID:    {info['model_id']}")
    print(f"Family:      {info['family']}")
    print(f"Backbone:    {info['backbone_type']}")
    print(f"VRAM budget: {info['memory_budget_gb']:.1f} GB")
    print(f"Default res: {info['default_resolution']} px")
    print(f"Pipeline:    {info['pipeline_class']}")
    if info["description"]:
        print(f"Description: {info['description']}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Build and return the parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download the base diffusion model weights Vision Banana needs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # list every model
  %(prog)s --list

  # download FLUX.1-schnell (Apache-2.0, good for quick experiments)
  %(prog)s --model flux-schnell

  # download SSD-1B (memory friendly, best quality/efficiency balance)
  %(prog)s --model ssd-1b --cache-dir /mnt/hf-cache

  # filter usable models by an 8GB VRAM budget
  %(prog)s --list --max-memory 8

  # only check that the local cache is complete, without network requests
  %(prog)s --model sd15 --local-files-only
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List every model registered in model_registry",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Name of the model to download (e.g. flux-dev, ssd-1b, sd15)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Local HuggingFace Hub cache directory (default: ~/.cache/huggingface/hub)",
    )
    parser.add_argument(
        "--max-memory",
        type=float,
        default=None,
        help="With --list, show only models whose VRAM requirement is at most this value (GB)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be downloaded, without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download, ignoring the existing local cache",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only the local cache without network requests (to check cache integrity)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (enabled by default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log at DEBUG level",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Script entry point."""
    args = parse_args()
    _setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    _ensure_hf_env()

    # resolve model_registry early to get the model list
    try:
        entries = _get_registry()
    except SystemExit:
        return 1

    # ── List mode ─────────────────────────────────────────────────────────
    if args.list:
        _print_model_list(entries, max_memory=args.max_memory)
        return 0

    # ── Argument checks ───────────────────────────────────────────────────
    if not args.model:
        logger.error("Specify --model <name>, or use --list to see the available models")
        return 1

    model_name = args.model.strip().lower()
    if model_name not in entries:
        logger.error("Unknown model: %r", model_name)
        logger.error("Available models: %s", ", ".join(sorted(entries.keys())))
        return 1

    info = entries[model_name]
    _print_model_detail(model_name, info)

    # ── Dry-run mode ──────────────────────────────────────────────────────
    if args.dry_run:
        print("[DRY RUN] Model that would be downloaded:")
        print(f"  Model name: {model_name}")
        print(f"  Repo ID:    {info['model_id']}")
        print(f"  Cache dir:  {args.cache_dir or '(default)'}")
        print(f"  Resume:     {not args.no_resume}")
        print(f"  Force:      {args.force}")
        print("  Nothing was downloaded.")
        return 0

    # ── Download ──────────────────────────────────────────────────────────
    if args.local_files_only:
        logger.info("local-files-only mode: only checking the local cache")

    logger.info("Starting model download: %s (%s)", model_name, info["model_id"])

    try:
        local_path = _download_with_resume(
            repo_id=info["model_id"],
            cache_dir=args.cache_dir,
            resume=not args.no_resume,
            force=args.force,
            local_files_only=args.local_files_only,
        )
    except RuntimeError as exc:
        logger.error("Download failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Download interrupted by the user. The partial download stays in the cache and resumes on the next run.")
        return 130  # standard Unix interrupt exit code

    # ── Post-download summary ─────────────────────────────────────────────
    total_gb = _calculate_dir_size(local_path)
    logger.info("✅ Download complete")
    logger.info("  Local path: %s", local_path)
    logger.info("  Total size: %.2f GB", total_gb)
    logger.info("  Model:      %s (%s)", model_name, info["model_id"])

    # print the bare path (easy for shell scripts to capture)
    print(str(local_path.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
