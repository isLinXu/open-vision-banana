"""Checkpoint resolution and Hugging Face Hub download utilities.

Supports three LoRA source formats:

- Local directory: ``/path/to/lora``
- Hugging Face repo id: ``hf:org/repo`` or ``org/repo`` (when not a local path)
- Registry alias: ``registry:vision-banana`` (see ``configs/checkpoint_registry.yaml``)
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "configs" / "checkpoint_registry.yaml"
HF_PREFIX = "hf:"
REGISTRY_PREFIX = "registry:"


def load_checkpoint_registry(registry_path: str | Path | None = None) -> dict[str, dict]:
    """Load the checkpoint registry YAML."""
    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    if not path.exists():
        logger.warning("Checkpoint registry not found at %s", path)
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("checkpoints", {})


class CheckpointNotAvailableError(ValueError):
    """Raised when a registry checkpoint is documented but not yet published."""


def get_registry_entry(
    alias: str,
    registry_path: str | Path | None = None,
) -> dict:
    """Return a registry entry dict for *alias*, or raise ValueError."""
    registry = load_checkpoint_registry(registry_path)
    entry = registry.get(alias)
    if entry is None:
        raise ValueError(
            f"Unknown checkpoint alias {alias!r}. "
            f"Available: {sorted(registry.keys()) or '(none)'}"
        )
    return entry


def is_registry_placeholder(alias: str, registry_path: str | Path | None = None) -> bool:
    """Return True when a registry alias exists but weights are not published."""
    entry = get_registry_entry(alias, registry_path=registry_path)
    return entry.get("status", "unknown") == "placeholder"


def resolve_lora_repo_id(
    lora_spec: str,
    registry_path: str | Path | None = None,
    *,
    allow_placeholder: bool = False,
) -> str:
    """Resolve a LoRA spec to a Hugging Face repo id."""
    if lora_spec.startswith(REGISTRY_PREFIX):
        alias = lora_spec[len(REGISTRY_PREFIX):]
        entry = get_registry_entry(alias, registry_path=registry_path)
        if not allow_placeholder and entry.get("status") == "placeholder":
            raise CheckpointNotAvailableError(
                f"Checkpoint {alias!r} is not yet published on Hugging Face "
                f"(status: placeholder). Train locally with scripts/train.py or "
                f"watch {entry.get('repo_id', 'the HF repo')} for releases."
            )
        repo_id = entry.get("repo_id")
        if not repo_id:
            raise ValueError(f"Registry entry {alias!r} has no repo_id")
        return repo_id

    if lora_spec.startswith(HF_PREFIX):
        return lora_spec[len(HF_PREFIX):]

    return lora_spec


def is_local_path(path_str: str) -> bool:
    """Return True when *path_str* refers to an existing local path."""
    if path_str.startswith((HF_PREFIX, REGISTRY_PREFIX)):
        return False
    p = Path(path_str)
    return p.exists()


def download_lora_from_hub(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    """Download LoRA weights from Hugging Face Hub.

    Returns:
        Local directory containing LoRA weight files.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for remote checkpoint download. "
            "Install with: pip install huggingface_hub"
        ) from exc

    logger.info("Downloading LoRA from %s …", repo_id)
    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
        allow_patterns=[
            "*.safetensors",
            "*.bin",
            "*.json",
            "*.txt",
            "adapter_config.json",
            "README.md",
        ],
    )
    return Path(local_dir)


def resolve_lora_path(
    lora_path: str | None,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    token: str | None = None,
    download: bool = True,
    allow_placeholder: bool = False,
) -> str | None:
    """Resolve a LoRA path spec to a local directory string.

    Args:
        lora_path: Local path, ``hf:repo``, ``registry:alias``, or bare repo id.
        revision: Optional HF revision (branch/tag/commit).
        cache_dir: HF cache directory override.
        registry_path: Override path to checkpoint_registry.yaml.
        token: HF access token for gated models.
        download: When False, return repo id for diffusers direct loading.

    Returns:
        Local directory path, or None when *lora_path* is None.
    """
    if lora_path is None:
        return None

    if is_local_path(lora_path):
        return str(Path(lora_path).resolve())

    repo_id = resolve_lora_repo_id(
        lora_path,
        registry_path=registry_path,
        allow_placeholder=allow_placeholder,
    )

    if not download:
        return repo_id

    local = download_lora_from_hub(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )
    return str(local)


def list_registry_checkpoints(registry_path: str | Path | None = None) -> list[dict]:
    """Return registry entries as a list of dicts for CLI display."""
    registry = load_checkpoint_registry(registry_path)
    entries = []
    for alias, meta in registry.items():
        entries.append({
            "alias": alias,
            "repo_id": meta.get("repo_id", ""),
            "revision": meta.get("revision"),
            "description": meta.get("description", ""),
            "tasks": meta.get("tasks", []),
            "status": meta.get("status", "unknown"),
        })
    return entries


__all__ = [
    "CheckpointNotAvailableError",
    "DEFAULT_REGISTRY_PATH",
    "download_lora_from_hub",
    "get_registry_entry",
    "is_registry_placeholder",
    "list_registry_checkpoints",
    "load_checkpoint_registry",
    "resolve_lora_path",
    "resolve_lora_repo_id",
]
