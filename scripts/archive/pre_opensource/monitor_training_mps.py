#!/usr/bin/env python3
"""Training monitor for Open-Vision-Banana MPS training.

Checks:
- Process status (PID, running/stopped)
- Latest log entries (iteration, epoch, loss, val_loss)
- Checkpoint saves
- Disk space on project volume
- MPS memory usage (if available)

Usage:
    python3 scripts/monitor_training_mps.py

Output: /Users/gatilin/PycharmProjects/open-vision-banana/output/monitor_report.json
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path("/Users/gatilin/PycharmProjects/open-vision-banana")
LOG_DIR = PROJECT_DIR / "output"
CKPT_DIR = LOG_DIR / "checkpoints_mps"
REPORT_PATH = LOG_DIR / "monitor_report.json"
MONITOR_LOG = LOG_DIR / "monitor.log"

TRAINING_LOG_PATTERNS = [
    LOG_DIR / "training_mps_nohup.log",
    LOG_DIR / "training_mps_launch_*.log",
]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def get_training_process() -> dict:
    """Find the running training process."""
    ps_output = _run(["ps", "aux"])
    for line in ps_output.splitlines():
        if "train.py" in line and "grep" not in line:
            parts = line.split()
            if len(parts) >= 11:
                return {
                    "pid": parts[1],
                    "cpu": parts[2],
                    "mem": parts[3],
                    "status": "running",
                    "command": " ".join(parts[10:]),
                }
    return {"status": "stopped", "pid": None, "cpu": None, "mem": None}


def parse_training_log(log_path: Path) -> dict:
    """Extract latest iteration/epoch/loss from training log."""
    if not log_path.exists():
        return {}

    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return {}

    result = {
        "last_line": "",
        "iter": None,
        "epoch": None,
        "loss": None,
        "val_loss": None,
        "step": None,
        "lines_read": len(lines),
    }

    if not lines:
        return result

    result["last_line"] = lines[-1].strip()

    # Scan from bottom for key metrics
    for line in reversed(lines[-200:]):
        line = line.strip()
        if "loss=" in line or "loss =" in line:
            try:
                # Try to extract loss value
                for part in line.split():
                    if "loss=" in part or "loss=" in part:
                        val = part.split("=")[-1].strip(",")
                        result["loss"] = float(val)
                        break
            except Exception:
                pass
        if "val_loss" in line or "val loss" in line:
            try:
                for part in line.split():
                    if "val_loss=" in part or "val_loss=" in part:
                        val = part.split("=")[-1].strip(",")
                        result["val_loss"] = float(val)
                        break
            except Exception:
                pass
        if "epoch" in line.lower():
            try:
                for part in line.split():
                    if part.lower().startswith("epoch=") or part.lower().startswith("epoch"):
                        val = part.split("=")[-1].strip(",/")
                        result["epoch"] = float(val)
                        break
            except Exception:
                pass
        if "step=" in line or "global_step=" in line:
            try:
                for part in line.split():
                    if "step=" in part or "global_step=" in part:
                        val = part.split("=")[-1].strip(",")
                        result["step"] = int(float(val))
                        break
            except Exception:
                pass

    return result


def get_disk_space() -> dict:
    """Check disk space on project volume."""
    output = _run(["df", "-h", str(PROJECT_DIR)])
    if output:
        lines = output.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 6:
                return {
                    "filesystem": parts[0],
                    "size": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "use_percent": parts[4],
                    "mounted": parts[5],
                }
    return {}


def get_checkpoints() -> dict:
    """List saved checkpoints."""
    if not CKPT_DIR.exists():
        return {"count": 0, "latest": None, "checkpoints": []}

    ckpts = sorted(
        [d.name for d in CKPT_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    )
    return {
        "count": len(ckpts),
        "latest": ckpts[-1] if ckpts else None,
        "checkpoints": ckpts,
    }


def get_mps_memory() -> dict:
    """Check MPS memory usage (macOS specific)."""
    try:
        import torch
        if torch.backends.mps.is_available():
            # MPS doesn't expose memory stats like CUDA, but we can estimate
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / 1e9, 1),
                "available_gb": round(mem.available / 1e9, 1),
                "used_gb": round(mem.used / 1e9, 1),
                "percent": mem.percent,
            }
    except Exception:
        pass
    return {}


def find_latest_log() -> Path | None:
    """Find the most recent training log file."""
    candidates = []
    for pattern in TRAINING_LOG_PATTERNS:
        if "*" in str(pattern):
            import glob
            candidates.extend(glob.glob(str(pattern)))
        else:
            candidates.append(str(pattern))

    existing = [Path(p) for p in candidates if Path(p).exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def main():
    now = datetime.now().isoformat()
    latest_log = find_latest_log()
    log_info = parse_training_log(latest_log) if latest_log else {}

    report = {
        "timestamp": now,
        "project": "open-vision-banana",
        "device": "mps",
        "process": get_training_process(),
        "log_file": str(latest_log) if latest_log else None,
        "log_metrics": log_info,
        "checkpoints": get_checkpoints(),
        "disk_space": get_disk_space(),
        "mps_memory": get_mps_memory(),
    }

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Write monitor log
    with open(MONITOR_LOG, "a") as f:
        f.write(f"[{now}] " + json.dumps(report, default=str) + "\n")

    # Print concise summary
    proc = report["process"]
    metrics = report["log_metrics"]
    ckpts = report["checkpoints"]
    disk = report["disk_space"]

    print("=== Open-Vision-Banana MPS Training Monitor ===")
    print(f"Time: {now}")
    print(f"Process: {proc['status'].upper()}", end="")
    if proc["status"] == "running":
        print(f" (PID={proc['pid']}, CPU={proc['cpu']}%, MEM={proc['mem']}%)")
    else:
        print(" ⚠️")

    print(f"Log: {report['log_file']}")
    if metrics.get("loss") is not None:
        print(f"  Loss: {metrics['loss']:.4f}", end="")
        if metrics.get("val_loss") is not None:
            print(f" | Val Loss: {metrics['val_loss']:.4f}")
        else:
            print()
    if metrics.get("epoch") is not None:
        print(f"  Epoch: {metrics['epoch']}")
    if metrics.get("step") is not None:
        print(f"  Step: {metrics['step']}")
    if metrics.get("last_line"):
        print(f"  Last log: {metrics['last_line'][:100]}")

    print(f"Checkpoints: {ckpts['count']} saved", end="")
    if ckpts["latest"]:
        print(f" (latest: {ckpts['latest']})")
    else:
        print()

    if disk:
        print(f"Disk: {disk.get('used', '?')} / {disk.get('size', '?')} ({disk.get('use_percent', '?')} used)")

    mem = report["mps_memory"]
    if mem:
        print(f"Memory: {mem['used_gb']}GB / {mem['total_gb']}GB ({mem['percent']}% used)")

    print(f"Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
