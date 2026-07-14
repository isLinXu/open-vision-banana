#!/usr/bin/env python3
"""Update training configuration to use only pure synthetic depth data.

This script enforces the paper requirement:
  "entirely synthetic depth data from simulation engines, zero real-world depth data"

It reads an existing training config (YAML), updates the data.tasks section to
include ONLY synthetic depth datasets, removes all real-world depth/normal
references, and writes the updated configuration back.

Supported operations:
  • Update existing config in-place (with backup)
  • Create a new synthetic-only config from a template
  • Audit config for real-world data contamination
  • Generate dataset path mappings for the synthetic pipeline

Usage:
    # Update existing config to synthetic-only (backs up original)
    python scripts/update_training_config.py --config configs/training_config.yaml --synthetic-only

    # Create a new synthetic-only config from the default template
    python scripts/update_training_config.py --create configs/synthetic_depth_only.yaml

    # Audit a config for real-world data contamination
    python scripts/update_training_config.py --config configs/training_config.yaml --audit

    # Update config with specific synthetic dataset paths
    python scripts/update_training_config.py --config configs/training_config.yaml \\
        --synthetic-only --dataset-root data/synthetic_depth_prepared/

    # Update with custom depth mapping parameters (paper defaults preserved)
    python scripts/update_training_config.py --config configs/training_config.yaml \\
        --synthetic-only --lam -3.0 --c 3.333

    # List real-world datasets that will be removed
    python scripts/update_training_config.py --list-real-world
"""

from __future__ import annotations

import argparse
import copy
import logging
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants — Synthetic vs Real-World Dataset Classification
# ═══════════════════════════════════════════════════════════════════════════════

# Datasets that are 100% synthetic (simulation engines)
PURE_SYNTHETIC_DATASETS: set[str] = {
    "Hypersim",
    "Virtual_KITTI_2",
    "VirtualKITTI",
    "VirtualKITTI2",
    "BlendedMVS",
    "TartanAir",
    "vkitti",
    "vkitti2",
    "blendedmvs",
    "tartanair",
    "hypersim",
    "vb_depth_pseudo_5k",  # Project's own synthetic data
}

# Datasets that contain real-world captured data (must be excluded)
REAL_WORLD_DATASETS: set[str] = {
    "NYU_Depth_v2",
    "NYU",
    "DIODE",
    "KITTI",
    "ETH3D",
    "ScanNet",
    "Cityscapes",
    "RefCOCOg",
    "SA_Co",
    "SA-Co",
    "COCO",
    "ADE20K",
    "Sintel",
    "OASIS",
    "HRWSI",
    "DIODE_indoor",
    "DIODE_outdoor",
    "nyu",
    "diode",
    "kitti",
    "eth3d",
    "scannet",
    "cityscapes",
    "refcocog",
    "sa1b",
    "coco",
    "ade20k",
    "sintel",
    "oasis",
    "hrwsi",
}

# Task categories that are depth-estimation only (no other vision tasks)
DEPTH_ONLY_TASKS: set[str] = {
    "depth_estimation",
    "depth",
}

# Task categories that require real-world data (to be excluded in pure-synthetic mode)
REAL_WORLD_TASKS: set[str] = {
    "semantic_segmentation",
    "instance_segmentation",
    "referring_segmentation",
    "normal_estimation",
    "edge_detection",
    "optical_flow",
    "keypoint_detection",
    "object_detection",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Config Manipulation
# ═══════════════════════════════════════════════════════════════════════════════


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict on error."""
    if not path.exists():
        logger.error("Config file not found: %s", path)
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return {}


def save_config(path: Path, config: dict[str, Any], *, backup: bool = True) -> bool:
    """Save a config to YAML with optional backup of existing file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and backup:
            backup_path = path.with_suffix(f".yaml.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(path, backup_path)
            logger.info("Backup created: %s", backup_path)
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        logger.info("Config saved: %s", path)
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Audit & Validation
# ─────────────────────────────────────────────────────────────────────────────


def audit_config(config: dict[str, Any]) -> dict[str, Any]:
    """Audit a config for real-world data contamination.

    Returns a dict with:
      - violations: list of (path, reason) tuples
      - is_pure_synthetic: bool
      - summary: str
    """
    violations: list[tuple[str, str]] = []
    data = config.get("data", {})
    tasks = data.get("tasks", {})

    # Check for real-world datasets in depth_estimation
    depth_tasks = {k: v for k, v in tasks.items() if "depth" in k.lower()}
    for task_name, task_spec in depth_tasks.items():
        datasets = task_spec.get("datasets", []) if isinstance(task_spec, dict) else []
        for ds in datasets:
            ds_name = ds.get("name", "") if isinstance(ds, dict) else str(ds)
            # Skip if explicitly known as pure synthetic
            if any(ps.lower() == ds_name.lower() for ps in PURE_SYNTHETIC_DATASETS):
                continue
            # Check for real-world dataset (exact match or word-boundary match)
            ds_lower = ds_name.lower()
            is_real_world = False
            for rw in REAL_WORLD_DATASETS:
                rw_lower = rw.lower()
                # Exact match or word-boundary match (avoid "kitti" matching "virtual_kitti_2")
                if ds_lower == rw_lower:
                    is_real_world = True
                    break
                # Word-boundary: real-world name should appear as a separate token
                if rw_lower in ds_lower:
                    # Ensure it's not part of a longer word (e.g., "vkitti" or "virtualkitti")
                    # by checking prefix/suffix boundaries
                    idx = ds_lower.find(rw_lower)
                    before = idx == 0 or not ds_lower[idx - 1].isalnum()
                    after = idx + len(rw_lower) == len(ds_lower) or not ds_lower[idx + len(rw_lower)].isalnum()
                    if before and after:
                        is_real_world = True
                        break
            if is_real_world:
                violations.append(
                    (f"data.tasks.{task_name}.datasets.{ds_name}",
                     f"Real-world dataset '{ds_name}' found in depth task")
                )

    # Check for non-depth tasks (which typically require real-world data)
    for task_name in tasks:
        if task_name not in depth_tasks and not any(d in task_name.lower() for d in ["depth"]):
            violations.append(
                (f"data.tasks.{task_name}",
                 f"Non-depth task '{task_name}' may require real-world data")
            )

    # Check generation_path (if provided, ensure it's not real-world)
    gen_path = data.get("generation_path")
    if gen_path and isinstance(gen_path, str):
        path_lower = gen_path.lower()
        if any(rw.lower() in path_lower for rw in REAL_WORLD_DATASETS):
            violations.append(
                ("data.generation_path",
                 f"Generation path '{gen_path}' may contain real-world data")
            )

    # Check vision_ratio (if < 1.0, generation data may be real-world photos)
    vision_ratio = data.get("vision_ratio", 0.1)
    if vision_ratio < 1.0 and not gen_path:
        violations.append(
            ("data.vision_ratio",
             f"vision_ratio={vision_ratio} < 1.0 but no generation_path specified — "
             "generation data source unclear")
        )

    return {
        "violations": violations,
        "is_pure_synthetic": len(violations) == 0,
        "summary": (
            f"Audit: {len(violations)} violation(s), "
            f"{'PASS' if len(violations) == 0 else 'FAIL'}"
        ),
    }


def print_audit_report(audit_result: dict[str, Any]) -> None:
    """Pretty-print an audit report."""
    print("\n" + "=" * 70)
    print("  Config Audit Report: Pure Synthetic Depth Data")
    print("=" * 70)
    print(f"\n  Status: {'✅ PURE SYNTHETIC' if audit_result['is_pure_synthetic'] else '❌ CONTAMINATED'}")
    print(f"  Violations: {len(audit_result['violations'])}")
    if audit_result["violations"]:
        print("\n  Issues found:")
        for path, reason in audit_result["violations"]:
            print(f"    • {path}")
            print(f"      → {reason}")
    else:
        print("\n  No real-world data contamination detected.")
    print("\n" + "=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-Only Config Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_synthetic_only_config(
    base_config: dict[str, Any],
    *,
    dataset_root: str | None = None,
    lam: float = -3.0,
    c: float = 10.0 / 3.0,
    vision_ratio: float = 1.0,
    resolution: int = 512,
) -> dict[str, Any]:
    """Build a synthetic-only training config from a base config.

    Args:
        base_config: Existing config dict to use as template.
        dataset_root: Root directory containing prepared synthetic datasets.
        lam: Depth power-transform shape parameter.
        c: Depth power-transform scale parameter.
        vision_ratio: Vision data sampling ratio (1.0 = 100% synthetic depth).
        resolution: Training resolution.

    Returns:
        Updated config dict with only synthetic depth datasets.
    """
    config = copy.deepcopy(base_config)

    # Ensure data section exists
    if "data" not in config:
        config["data"] = {}
    data = config["data"]

    # Set pure synthetic depth parameters
    data["vision_ratio"] = vision_ratio
    data["generation_ratio"] = 0.0 if vision_ratio >= 1.0 else 1.0 - vision_ratio
    data["resolution"] = resolution
    data["augmentation"] = True
    data["val_split"] = 0.0  # No validation split for synthetic-only (optional)
    data["generation_path"] = None  # No generation data mixing

    dataset_root = dataset_root.rstrip("/") if dataset_root else None

    # Build synthetic-only tasks
    synthetic_tasks: dict[str, Any] = {
        "depth_estimation": {
            "datasets": [
                {
                    "name": "Hypersim",
                    "split": "train",
                    "path": f"{dataset_root}/hypersim-depth/depth" if dataset_root else None,
                },
                {
                    "name": "Virtual_KITTI_2",
                    "split": "train",
                    "path": f"{dataset_root}/vkitti2-depth/depth" if dataset_root else None,
                },
                {
                    "name": "BlendedMVS",
                    "split": "train",
                    "path": f"{dataset_root}/blendedmvs-depth/depth" if dataset_root else None,
                },
                {
                    "name": "TartanAir",
                    "split": "train",
                    "path": f"{dataset_root}/tartanair-depth/depth" if dataset_root else None,
                },
            ]
        }
    }

    # Remove None paths for cleaner YAML
    for task_spec in synthetic_tasks.values():
        for ds in task_spec.get("datasets", []):
            if ds.get("path") is None:
                del ds["path"]

    data["tasks"] = synthetic_tasks

    # Update depth mapping parameters (paper defaults)
    if "depth_mapping" not in config:
        config["depth_mapping"] = {}
    config["depth_mapping"]["lam"] = lam
    config["depth_mapping"]["c"] = round(c, 3)

    # Remove normal mapping (not used in pure depth mode)
    if "normal_mapping" in config:
        config["normal_mapping"] = {}

    # Remove segmentation (not used in pure depth mode)
    if "segmentation" in config:
        # Keep for compatibility but could be removed
        pass

    # Add metadata comment
    config["_meta"] = {
        "mode": "synthetic_depth_only",
        "zero_real_world": True,
        "generated_at": datetime.now().isoformat(),
        "paper_alignment": "Section 3.2 — entirely synthetic depth data from simulation engines",
        "datasets": ["Hypersim", "Virtual_KITTI_2", "BlendedMVS", "TartanAir"],
    }

    return config


def build_default_synthetic_config(
    *,
    dataset_root: str = "data/synthetic_depth_prepared",
    output_dir: str = "output/checkpoints_synthetic_depth",
    model_name: str = "black-forest-labs/FLUX.1-schnell",
    resolution: int = 512,
    batch_size: int = 8,
    epochs: int = 10,
    lr: float = 5e-5,
    lora_rank: int = 64,
    lora_alpha: int = 128,
) -> dict[str, Any]:
    """Build a complete default config for pure synthetic depth training."""
    dataset_root = dataset_root.rstrip("/")
    return {
        "_meta": {
            "mode": "synthetic_depth_only",
            "zero_real_world": True,
            "generated_at": datetime.now().isoformat(),
            "paper_alignment": "Section 3.2 — entirely synthetic depth data from simulation engines",
            "datasets": ["Hypersim", "Virtual_KITTI_2", "BlendedMVS", "TartanAir"],
        },
        "model": {
            "name": model_name,
            "dtype": "bfloat16",
        },
        "lora": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "rs_lora": True,
            "target_modules": ["to_k", "to_v"],
            "dropout": 0.0,
        },
        "optimizer": {
            "lr": lr,
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "warmup_steps": 500,
            "max_grad_norm": 1.0,
        },
        "output_dir": output_dir,
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "open-vision-banana",
            "wandb_log_checkpoints": False,
            "tensorboard_enabled": False,
            "tensorboard_dir": "output/runs_synthetic_depth",
            "run_name": None,
        },
        "training": {
            "epochs": epochs,
            "gradient_accumulation_steps": 1,
            "use_amp": True,
            "use_ema": True,
            "ema_decay": 0.9999,
            "ema_warmup_steps": 2000,
            "save_every": 1000,
            "eval_every": 500,
            "max_keep": 5,
            "seed": 42,
            "log_every": 10,
            "early_stopping_patience": None,
            "early_stopping_min_delta": 0.0,
        },
        "data": {
            "vision_ratio": 1.0,
            "generation_ratio": 0.0,
            "resolution": resolution,
            "batch_size": batch_size,
            "num_workers": 4,
            "augmentation": True,
            "val_split": 0.0,
            "generation_path": None,
            "tasks": {
                "depth_estimation": {
                    "datasets": [
                        {
                            "name": "Hypersim",
                            "split": "train",
                            "path": f"{dataset_root}/hypersim-depth/depth",
                        },
                        {
                            "name": "Virtual_KITTI_2",
                            "split": "train",
                            "path": f"{dataset_root}/vkitti2-depth/depth",
                        },
                        {
                            "name": "BlendedMVS",
                            "split": "train",
                            "path": f"{dataset_root}/blendedmvs-depth/depth",
                        },
                        {
                            "name": "TartanAir",
                            "split": "train",
                            "path": f"{dataset_root}/tartanair-depth/depth",
                        },
                    ]
                }
            },
        },
        "depth_mapping": {
            "lam": -3.0,
            "c": 3.333,
        },
        "normal_mapping": {},
        "segmentation": {
            "color_tolerance": 14.0,
            "min_area_fraction": 2.0e-4,
            "erosion_threshold": 0.1,
            "bbox_expansion_factor": 5.0,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update training config for pure synthetic depth data only",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update existing config (backs up original)
  python scripts/update_training_config.py --config configs/training_config.yaml --synthetic-only

  # Create new synthetic-only config
  python scripts/update_training_config.py --create configs/synthetic_depth_only.yaml

  # Audit existing config for real-world contamination
  python scripts/update_training_config.py --config configs/training_config.yaml --audit

  # Update with specific dataset paths and custom parameters
  python scripts/update_training_config.py --config configs/training_config.yaml \\
      --synthetic-only --dataset-root data/prepared/ --lam -3.0 --c 3.333

  # List all real-world datasets that would be removed
  python scripts/update_training_config.py --list-real-world

  # Create a new config with custom training parameters
  python scripts/update_training_config.py --create configs/synthetic_depth_512.yaml \\
      --resolution 512 --batch-size 8 --epochs 10 --lr 5e-5
""",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to existing config YAML")
    parser.add_argument("--create", type=str, default=None, help="Create new synthetic-only config at this path")
    parser.add_argument("--synthetic-only", action="store_true", help="Update config to use only synthetic depth data")
    parser.add_argument("--audit", action="store_true", help="Audit config for real-world data contamination")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root directory for prepared synthetic datasets")
    parser.add_argument("--lam", type=float, default=-3.0, help="Depth power-transform lambda (default: -3.0)")
    parser.add_argument("--c", type=float, default=10.0 / 3.0, help="Depth power-transform c (default: 3.333)")
    parser.add_argument("--vision-ratio", type=float, default=1.0, help="Vision data ratio (default: 1.0 = 100%%)")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution (default: 512)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank (default: 64)")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha (default: 128)")
    parser.add_argument("--output-dir", type=str, default="output/checkpoints_synthetic_depth", help="Output directory")
    parser.add_argument("--model-name", type=str, default="black-forest-labs/FLUX.1-schnell", help="Base model name")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup when updating existing config")
    parser.add_argument("--list-real-world", action="store_true", help="List real-world datasets that will be excluded")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def list_real_world_datasets() -> None:
    """Print all real-world datasets that would be excluded."""
    print("\n" + "=" * 70)
    print("  Real-World Datasets (EXCLUDED from pure synthetic training)")
    print("=" * 70)
    print("\n  Depth Estimation:")
    for ds in sorted(["NYU Depth v2", "DIODE", "KITTI", "ETH3D", "HRWSI"]):
        print(f"    • {ds}")
    print("\n  Normal Estimation:")
    for ds in sorted(["ScanNet", "OASIS"]):
        print(f"    • {ds}")
    print("\n  Segmentation / Detection:")
    for ds in sorted(["Cityscapes", "ADE20K", "COCO", "SA-1B / SA-Co", "RefCOCOg"]):
        print(f"    • {ds}")
    print("\n  Optical Flow:")
    for ds in sorted(["Sintel"]):
        print(f"    • {ds}")
    print("\n" + "=" * 70)
    print("\n  Pure Synthetic Datasets (INCLUDED):")
    for ds in sorted(["Hypersim (Apple / UE4)", "Virtual KITTI 2 (NAVER Labs)", "BlendedMVS (UoA)", "TartanAir (CMU)"]):
        print(f"    ✓ {ds}")
    print("\n" + "=" * 70 + "\n")


def print_config_summary(config: dict[str, Any]) -> None:
    """Print a concise summary of the config."""
    data = config.get("data", {})
    tasks = data.get("tasks", {})

    print("\n" + "=" * 70)
    print("  Synthetic-Only Training Config Summary")
    print("=" * 70)

    # Meta
    meta = config.get("_meta", {})
    if meta:
        print(f"\n  Mode: {meta.get('mode', 'unknown')}")
        print(f"  Zero real-world: {meta.get('zero_real_world', False)}")

    # Model
    model = config.get("model", {})
    print(f"\n  Model: {model.get('name', 'unknown')}")
    print(f"  Dtype: {model.get('dtype', 'unknown')}")

    # LoRA
    lora = config.get("lora", {})
    print(f"\n  LoRA: rank={lora.get('rank', '?')}, alpha={lora.get('alpha', '?')}, rsLoRA={lora.get('rs_lora', False)}")

    # Training
    training = config.get("training", {})
    print(f"\n  Epochs: {training.get('epochs', '?')}")
    print(f"  Batch size: {data.get('batch_size', '?')}")
    print(f"  Resolution: {data.get('resolution', '?')}")
    print(f"  Vision ratio: {data.get('vision_ratio', '?')}")

    # Depth mapping
    dm = config.get("depth_mapping", {})
    print(f"\n  Depth mapping: λ={dm.get('lam', '?')}, c={dm.get('c', '?')}")

    # Datasets
    print(f"\n  Datasets ({len(tasks)} task(s)):")
    for task_name, task_spec in tasks.items():
        datasets = task_spec.get("datasets", []) if isinstance(task_spec, dict) else []
        print(f"    [{task_name}]")
        for ds in datasets:
            ds_name = ds.get("name", "") if isinstance(ds, dict) else str(ds)
            ds_path = ds.get("path", "") if isinstance(ds, dict) else ""
            path_str = f" @ {ds_path}" if ds_path else ""
            print(f"      • {ds_name}{path_str}")

    # Output
    print(f"\n  Output: {config.get('output_dir', 'unknown')}")
    print("=" * 70 + "\n")


def main() -> int:
    args = parse_args()

    if args.list_real_world:
        list_real_world_datasets()
        return 0

    if args.audit and args.config:
        config = load_config(Path(args.config))
        if not config:
            return 1
        audit_result = audit_config(config)
        print_audit_report(audit_result)
        return 0 if audit_result["is_pure_synthetic"] else 1

    if args.create:
        # Create a brand-new synthetic-only config
        logger.info("Creating new synthetic-only config: %s", args.create)
        config = build_default_synthetic_config(
            dataset_root=args.dataset_root or "data/synthetic_depth_prepared",
            output_dir=args.output_dir,
            model_name=args.model_name,
            resolution=args.resolution,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )
        if save_config(Path(args.create), config, backup=False):
            print_config_summary(config)
            return 0
        return 1

    if args.synthetic_only and args.config:
        # Update existing config
        config_path = Path(args.config)
        config = load_config(config_path)
        if not config:
            return 1

        # Audit before changes
        audit_before = audit_config(config)
        if audit_before["is_pure_synthetic"]:
            logger.info("Existing config already passes pure-synthetic audit.")
        else:
            logger.info("Existing config has %d violation(s) — will be fixed.", len(audit_before["violations"]))
            if args.verbose:
                print_audit_report(audit_before)

        updated = build_synthetic_only_config(
            config,
            dataset_root=args.dataset_root,
            lam=args.lam,
            c=args.c,
            vision_ratio=args.vision_ratio,
            resolution=args.resolution,
        )

        if save_config(config_path, updated, backup=not args.no_backup):
            # Audit after changes
            audit_after = audit_config(updated)
            print_audit_report(audit_after)
            print_config_summary(updated)
            return 0 if audit_after["is_pure_synthetic"] else 1
        return 1

    logger.error("No action specified. Use --synthetic-only, --create, --audit, or --list-real-world.")
    return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    sys.exit(main())
