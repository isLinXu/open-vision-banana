#!/usr/bin/env python3
"""Audited five-task SDXL/SSD LoRA training entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from audit_v3_multitask_dataset import SUPPORTED_TASKS, audit_v3_dataset


def build_training_command(args: argparse.Namespace) -> list[str]:
    """Translate the audited runner options to the existing SSD trainer."""
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_sdxl_mps.py")),
        "--model",
        args.model,
        "--data-dir",
        str(args.data_dir),
        "--task",
        "all",
        "--source-conditioned",
        "--prediction-type",
        "epsilon",
        "--output-dir",
        str(args.output_dir),
        "--max-steps",
        str(args.max_steps),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--resolution",
        str(args.resolution),
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.alpha),
        "--lr",
        str(args.lr),
        "--save-every",
        str(args.save_every),
        "--log-every",
        str(args.log_every),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--no-ema",
    ]
    if getattr(args, "task_ratios", None):
        command.extend(["--task-ratios", args.task_ratios])
    if getattr(args, "task_schedule", False):
        command.append("--task-schedule")
    if getattr(args, "no_balance_data", False):
        command.append("--no-balance-data")
    if getattr(args, "eval_every", None):
        command.extend(["--eval-every", str(args.eval_every)])
    if getattr(args, "max_per_task", None) is not None:
        command.extend(["--max-per-task", str(args.max_per_task)])
    if getattr(args, "max_generation", None) is not None:
        command.extend(["--max-generation", str(args.max_generation)])
    if getattr(args, "max_val_per_task", None) is not None:
        command.extend(["--max-val-per-task", str(args.max_val_per_task)])
    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    return command


def _write_audit(output_dir: Path, audit: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "multitask_data_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="depth,normal,semantic,edge,generation")
    parser.add_argument("--model", default="ssd-1b", choices=["ssd-1b", "sdxl"])
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument("--max-val-per-task", type=int, default=64)
    parser.add_argument("--max-generation", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--task-ratios", default=None)
    parser.add_argument("--task-schedule", action="store_true")
    parser.add_argument("--no-balance-data", action="store_true")
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.grad_accum <= 0:
        parser.error("steps, epochs, batch-size and grad-accum must be positive")
    if (args.max_per_task is not None and args.max_per_task <= 0) or (args.max_generation is not None and args.max_generation <= 0) or args.max_val_per_task < 0:
        parser.error("sample caps must be positive")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    audit_tasks = ["edge" if task == "edge" else task for task in tasks]
    unknown = sorted(set(audit_tasks) - SUPPORTED_TASKS)
    if unknown:
        parser.error(f"unsupported tasks: {unknown}")
    if set(audit_tasks) != SUPPORTED_TASKS:
        parser.error(
            "the current runner is intentionally all-five-task only; "
            f"requested={sorted(set(audit_tasks))} required={sorted(SUPPORTED_TASKS)}"
        )
    audit = audit_v3_dataset(args.data_dir.expanduser().resolve(), tasks=audit_tasks)
    audit["requested_tasks"] = tasks
    audit["runner"] = "train_multitask_lora.py"
    audit_path = _write_audit(args.output_dir.expanduser().resolve(), audit)
    print(f"Audit passed: {audit_path}")
    if args.audit_only:
        return

    command = build_training_command(args)
    metadata = args.output_dir.expanduser().resolve() / "multitask_runner_config.json"
    metadata.write_text(
        json.dumps(
            {
                "command": command,
                "requested_tasks": tasks,
                "audit_checksum": audit["manifest_checksum"],
                "audit_path": str(audit_path),
                "task_ratios": json.loads(args.task_ratios) if args.task_ratios else None,
                "task_schedule": args.task_schedule,
                "no_balance_data": args.no_balance_data,
                "eval_every": args.eval_every,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
