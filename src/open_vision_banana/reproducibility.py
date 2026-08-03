"""Experiment manifest and dataset fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import VisionBananaConfig

TASK_ALIASES = {
    "depth_estimation": "depth",
    "normal_estimation": "normal",
    "semantic_segmentation": "semantic",
    "instance_segmentation": "instance",
    "referring_segmentation": "referring",
    "edge_detection": "edge",
    "keypoint_detection": "keypoint",
}


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing or storage."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def configured_tasks(config: VisionBananaConfig) -> list[str]:
    """Return canonical task names from list or legacy mapping-style configs."""
    value = config.data.tasks
    if value is None:
        return []
    names = list(value) if isinstance(value, (list, dict)) else []
    if not names:
        raise ValueError("data.tasks must be a non-empty list or mapping")
    result: list[str] = []
    for name in names:
        task = TASK_ALIASES.get(str(name), str(name))
        if task not in result:
            result.append(task)
    return result


def resolve_data_paths(config: VisionBananaConfig) -> dict[str, Path]:
    """Resolve configured task and generation roots without requiring them to exist."""
    paths: dict[str, Path] = {}
    data_root = Path(config.data.root).expanduser() if config.data.root else None
    for task in configured_tasks(config):
        configured = config.data.task_dirs.get(task)
        if configured is None:
            for alias, canonical in TASK_ALIASES.items():
                if canonical == task and alias in config.data.task_dirs:
                    configured = config.data.task_dirs[alias]
                    break
        if configured is not None:
            path = Path(configured).expanduser()
            if not path.is_absolute() and data_root is not None:
                path = data_root / path
        elif data_root is not None:
            candidate = data_root / task
            path = candidate if candidate.exists() else data_root
        else:
            continue
        paths[f"task:{task}"] = path.resolve()

    if config.data.generation_path:
        generation_path = Path(config.data.generation_path).expanduser()
        if not generation_path.is_absolute() and data_root is not None and not generation_path.exists():
            generation_path = data_root / generation_path
        paths["generation"] = generation_path.resolve()
    elif data_root is not None and (data_root / "generation").exists():
        paths["generation"] = (data_root / "generation").resolve()
    return paths


def fingerprint_path(path: Path, mode: str = "metadata") -> dict[str, Any]:
    """Fingerprint a file tree using metadata or complete file content."""
    if mode not in {"metadata", "content"}:
        raise ValueError("fingerprint mode must be 'metadata' or 'content'")
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        stat = file_path.stat()
        file_count += 1
        total_bytes += stat.st_size
        digest.update(relative.encode())
        digest.update(str(stat.st_size).encode())
        if mode == "metadata":
            digest.update(str(stat.st_mtime_ns).encode())
        else:
            with file_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return {
        "path": str(path),
        "mode": mode,
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def _runtime_state() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        runtime.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "mps_available": bool(
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ),
            }
        )
    except ImportError:
        runtime["torch"] = None
    return runtime


def build_experiment_manifest(
    config: VisionBananaConfig,
    *,
    repo_root: str | Path,
    fingerprint_mode: str = "metadata",
    require_data: bool = True,
) -> dict[str, Any]:
    """Build a model-free manifest and fail early on missing configured data."""
    config.validate()
    resolved_paths = resolve_data_paths(config)
    missing = [name for name, path in resolved_paths.items() if not path.exists()]
    if require_data:
        if config.data.root and not configured_tasks(config):
            raise ValueError("data.root is set but data.tasks is empty")
        if configured_tasks(config) and not resolved_paths:
            raise ValueError("data.tasks is set but no data.root/task_dirs could be resolved")
        if missing:
            details = ", ".join(f"{name}={resolved_paths[name]}" for name in missing)
            raise FileNotFoundError(f"Configured data path(s) do not exist: {details}")

    fingerprint_cache: dict[Path, dict[str, Any]] = {}
    datasets = {}
    for name, path in resolved_paths.items():
        if not path.exists():
            continue
        if path not in fingerprint_cache:
            fingerprint_cache[path] = fingerprint_path(path, fingerprint_mode)
        datasets[name] = dict(fingerprint_cache[path])
    if require_data:
        empty = [name for name, details in datasets.items() if details["file_count"] == 0]
        if empty:
            raise ValueError("Configured data path(s) contain no files: " + ", ".join(empty))
    effective_config = config.to_dict()
    warnings = []
    if isinstance(config.data.tasks, dict):
        warnings.append(
            "Mapping-style data.tasks is legacy: only mapping keys select loaders; "
            "nested dataset metadata is not used as a path. Configure data.task_dirs explicitly."
        )
    return {
        "schema_version": 1,
        "config_sha256": sha256_json(effective_config),
        "effective_config": effective_config,
        "git": _git_state(Path(repo_root).resolve()),
        "runtime": _runtime_state(),
        "datasets": datasets,
        "missing_data": {
            name: str(resolved_paths[name]) for name in missing
        },
        "warnings": warnings,
    }


__all__ = [
    "build_experiment_manifest",
    "canonical_json",
    "configured_tasks",
    "fingerprint_path",
    "resolve_data_paths",
    "sha256_json",
]
