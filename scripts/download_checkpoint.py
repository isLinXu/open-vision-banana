#!/usr/bin/env python3
"""Download Vision Banana LoRA checkpoints from Hugging Face Hub.

Usage:
    # List registered checkpoints
    python scripts/download_checkpoint.py --list

    # Download by registry alias
    python scripts/download_checkpoint.py --alias vision-banana

    # Download by repo id
    python scripts/download_checkpoint.py --repo open-vision-banana/vision-banana-lora

    # Download to a specific directory
    python scripts/download_checkpoint.py --alias vision-banana --output-dir checkpoints/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from open_vision_banana.inference.checkpoint import (
    CheckpointNotAvailableError,
    download_lora_from_hub,
    get_registry_entry,
    list_registry_checkpoints,
    resolve_lora_repo_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Vision Banana LoRA checkpoints from Hugging Face Hub"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List checkpoints in configs/checkpoint_registry.yaml",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=None,
        help="Registry alias (e.g. vision-banana)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Hugging Face repo id (e.g. org/model-lora)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HF revision (branch, tag, or commit)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Copy downloaded weights to this directory (optional)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Hugging Face cache directory override",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Attempt download even when registry status is placeholder",
    )
    return parser.parse_args()


def _print_registry() -> None:
    entries = list_registry_checkpoints()
    if not entries:
        print("No checkpoints registered.")
        return

    print("Registered checkpoints:\n")
    for entry in entries:
        tasks = ", ".join(entry["tasks"]) if entry["tasks"] else "-"
        print(f"  {entry['alias']}")
        print(f"    repo_id:     {entry['repo_id']}")
        print(f"    status:      {entry['status']}")
        print(f"    tasks:       {tasks}")
        if entry["description"]:
            desc = entry["description"].strip().replace("\n", " ")
            print(f"    description: {desc[:120]}{'…' if len(desc) > 120 else ''}")
        print()


def _resolve_spec(args: argparse.Namespace) -> tuple[str, str | None]:
    if args.alias and args.repo:
        raise ValueError("Specify only one of --alias or --repo")

    if args.alias:
        spec = f"registry:{args.alias}"
        try:
            entry = get_registry_entry(args.alias)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if entry.get("status") == "placeholder" and not args.force:
            raise ValueError(
                f"Checkpoint {args.alias!r} is not yet published (status: placeholder). "
                f"Train locally with scripts/train.py, or use --force to attempt download anyway."
            )
        revision = args.revision or entry.get("revision")
        repo_id = resolve_lora_repo_id(spec, allow_placeholder=args.force)
        return repo_id, revision

    if args.repo:
        return args.repo, args.revision

    raise ValueError("Specify --list, --alias, or --repo")


def main() -> int:
    args = parse_args()

    if args.list:
        _print_registry()
        return 0

    try:
        repo_id, revision = _resolve_spec(args)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    if args.dry_run:
        print(f"Would download: {repo_id}")
        if revision:
            print(f"  revision: {revision}")
        if args.output_dir:
            print(f"  output:   {args.output_dir}")
        return 0

    try:
        local_dir = download_lora_from_hub(
            repo_id,
            revision=revision,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        if isinstance(exc, CheckpointNotAvailableError) or "placeholder" in str(exc).lower():
            logger.info(
                "Train locally with scripts/train.py or publish weights to Hugging Face "
                "and set status: available in configs/checkpoint_registry.yaml."
            )
        else:
            logger.info(
                "If the repo is a placeholder, train locally with scripts/train.py "
                "or publish weights to Hugging Face and update configs/checkpoint_registry.yaml."
            )
        return 1

    logger.info("Downloaded to: %s", local_dir)

    if args.output_dir:
        import shutil
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for item in local_dir.iterdir():
            dest = out / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        logger.info("Copied to: %s", out.resolve())
        print(str(out.resolve()))
    else:
        print(str(local_dir))

    return 0


if __name__ == "__main__":
    sys.exit(main())
