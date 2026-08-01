"""Versioned, checksum-verified artifacts for frozen-feature depth probes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

DEPTH_PROBE_FORMAT = "open-vision-banana.depth-probe"
DEPTH_PROBE_FORMAT_VERSION = 1
DEPTH_PROBE_CONFIG_NAME = "config.json"
DEPTH_PROBE_WEIGHTS_NAME = "model.safetensors"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_feature_contract(payload: Mapping[str, Any]) -> dict[str, list[Any]]:
    names = payload.get("names")
    channels = payload.get("channels")
    if not isinstance(names, (list, tuple)) or not isinstance(channels, (list, tuple)):
        raise ValueError("feature_contract must contain names and channels lists")
    normalized_names = [str(name) for name in names]
    normalized_channels = [int(value) for value in channels]
    if not normalized_names or len(normalized_names) != len(normalized_channels):
        raise ValueError("feature_contract names and channels must be non-empty and aligned")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("feature_contract names must be unique")
    if any(not name for name in normalized_names) or any(value <= 0 for value in normalized_channels):
        raise ValueError("feature_contract names and channels must be valid")
    return {"names": normalized_names, "channels": normalized_channels}


def _validate_state_dict(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not payload:
        raise ValueError("depth probe state_dict must not be empty")
    state_dict: dict[str, torch.Tensor] = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not name:
            raise ValueError("depth probe state_dict keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"depth probe state_dict value {name!r} is not a tensor")
        state_dict[name] = value.detach().cpu().contiguous()
    return state_dict


@dataclass(frozen=True)
class DepthProbeArtifact:
    """Validated metadata and tensors needed to construct a depth estimator."""

    path: Path
    head_state_dict: dict[str, torch.Tensor]
    feature_contract: dict[str, list[Any]]
    feature_mask: tuple[bool, ...]
    model: str
    prompt: str
    colormap: str
    resolution: int
    timestep: int
    width: int
    upsample_stages: int
    selected_step: int
    metrics: dict[str, Any]
    splits: dict[str, Any]
    training: dict[str, Any]
    format_version: int
    legacy: bool = False


def _normalize_metadata(
    *,
    path: Path,
    payload: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    legacy: bool,
) -> DepthProbeArtifact:
    contract = _validate_feature_contract(payload.get("feature_contract", {}))
    feature_mask_raw = payload.get("feature_mask")
    if not isinstance(feature_mask_raw, (list, tuple)):
        raise ValueError("feature_mask must be a list")
    feature_mask = tuple(feature_mask_raw)
    if len(feature_mask) != len(contract["names"]) or any(
        type(value) is not bool for value in feature_mask
    ):
        raise ValueError("feature_mask must contain one boolean per feature")

    required_strings = ("model", "prompt", "colormap")
    strings = {name: payload.get(name) for name in required_strings}
    if any(not isinstance(value, str) or not value for value in strings.values()):
        raise ValueError(f"depth probe metadata requires non-empty strings: {required_strings}")

    integer_fields = {
        "resolution": payload.get("resolution"),
        "timestep": payload.get("timestep"),
        "width": payload.get("width"),
        "upsample_stages": payload.get("upsample_stages"),
        "selected_step": payload.get("selected_step", 0),
    }
    if any(type(value) is not int for value in integer_fields.values()):
        raise ValueError("depth probe numeric metadata must contain integers")
    if integer_fields["resolution"] <= 0 or integer_fields["width"] <= 0:
        raise ValueError("resolution and width must be positive")
    if any(integer_fields[name] < 0 for name in ("timestep", "upsample_stages", "selected_step")):
        raise ValueError("timestep, upsample_stages, and selected_step must be non-negative")

    metrics = payload.get("metrics", {})
    splits = payload.get("splits", {})
    training = payload.get("training", {})
    if not all(isinstance(value, dict) for value in (metrics, splits, training)):
        raise ValueError("metrics, splits, and training metadata must be objects")

    return DepthProbeArtifact(
        path=path,
        head_state_dict=_validate_state_dict(state_dict),
        feature_contract=contract,
        feature_mask=feature_mask,
        model=str(strings["model"]),
        prompt=str(strings["prompt"]),
        colormap=str(strings["colormap"]),
        resolution=int(integer_fields["resolution"]),
        timestep=int(integer_fields["timestep"]),
        width=int(integer_fields["width"]),
        upsample_stages=int(integer_fields["upsample_stages"]),
        selected_step=int(integer_fields["selected_step"]),
        metrics=dict(metrics),
        splits=dict(splits),
        training=dict(training),
        format_version=0 if legacy else DEPTH_PROBE_FORMAT_VERSION,
        legacy=legacy,
    )


def save_depth_probe_artifact(
    path: str | Path,
    *,
    head_state_dict: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    feature_mask: tuple[bool, ...] | list[bool],
    model: str,
    prompt: str,
    colormap: str,
    resolution: int,
    timestep: int,
    width: int,
    upsample_stages: int,
    selected_step: int,
    metrics: Mapping[str, Any] | None = None,
    splits: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write a versioned safetensors probe artifact directory."""
    artifact_dir = Path(path).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_dict = _validate_state_dict(head_state_dict)
    contract = _validate_feature_contract(feature_contract)
    normalized_mask = tuple(feature_mask)
    metadata: dict[str, Any] = {
        "format": DEPTH_PROBE_FORMAT,
        "format_version": DEPTH_PROBE_FORMAT_VERSION,
        "architecture": "frozen-ssd-vae-fpn-scalar-depth",
        "weights": DEPTH_PROBE_WEIGHTS_NAME,
        "model": model,
        "prompt": prompt,
        "colormap": colormap,
        "resolution": resolution,
        "timestep": timestep,
        "width": width,
        "upsample_stages": upsample_stages,
        "selected_step": selected_step,
        "feature_contract": contract,
        "feature_mask": list(normalized_mask),
        "metrics": dict(metrics or {}),
        "splits": dict(splits or {}),
        "training": dict(training or {}),
    }
    _normalize_metadata(
        path=artifact_dir,
        payload=metadata,
        state_dict=state_dict,
        legacy=False,
    )

    weights_path = artifact_dir / DEPTH_PROBE_WEIGHTS_NAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".depth-probe-",
        suffix=".safetensors",
        dir=artifact_dir,
    )
    os.close(descriptor)
    temporary_weights = Path(temporary_name)
    try:
        save_file(state_dict, str(temporary_weights))
        temporary_weights.replace(weights_path)
    finally:
        temporary_weights.unlink(missing_ok=True)

    metadata["weights_sha256"] = _sha256(weights_path)
    serialized = json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".depth-probe-",
        suffix=".json",
        dir=artifact_dir,
    )
    os.close(descriptor)
    temporary_config = Path(temporary_name)
    try:
        temporary_config.write_text(serialized, encoding="utf-8")
        temporary_config.replace(artifact_dir / DEPTH_PROBE_CONFIG_NAME)
    finally:
        temporary_config.unlink(missing_ok=True)
    return artifact_dir


def load_depth_probe_artifact(path: str | Path) -> DepthProbeArtifact:
    """Load a versioned artifact, or a validated legacy ``.pt`` checkpoint."""
    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("legacy depth probe checkpoint must contain a dictionary")
        state_dict = payload.get("head_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("legacy depth probe checkpoint has no head_state_dict")
        legacy_payload = {
            **payload,
            "metrics": {
                key: payload[key]
                for key in ("train_metrics", "dev_metrics", "official_val_metrics")
                if key in payload
            },
            "splits": {
                key: payload[key]
                for key in ("train_fit_names", "dev_names", "official_val_names")
                if key in payload
            },
            "training": {},
        }
        return _normalize_metadata(
            path=checkpoint_path,
            payload=legacy_payload,
            state_dict=state_dict,
            legacy=True,
        )

    config_path = checkpoint_path / DEPTH_PROBE_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing depth probe config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("depth probe config must contain a JSON object")
    if payload.get("format") != DEPTH_PROBE_FORMAT:
        raise ValueError(f"Unsupported depth probe format: {payload.get('format')!r}")
    if payload.get("format_version") != DEPTH_PROBE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported depth probe format version: {payload.get('format_version')!r}"
        )
    weights_name = payload.get("weights")
    if weights_name != DEPTH_PROBE_WEIGHTS_NAME:
        raise ValueError(f"Unexpected depth probe weights filename: {weights_name!r}")
    weights_path = checkpoint_path / weights_name
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing depth probe weights: {weights_path}")
    expected_checksum = payload.get("weights_sha256")
    if not isinstance(expected_checksum, str) or _sha256(weights_path) != expected_checksum:
        raise ValueError("Depth probe weights checksum mismatch")
    state_dict = load_file(str(weights_path), device="cpu")
    return _normalize_metadata(
        path=checkpoint_path,
        payload=payload,
        state_dict=state_dict,
        legacy=False,
    )


__all__ = [
    "DEPTH_PROBE_CONFIG_NAME",
    "DEPTH_PROBE_FORMAT",
    "DEPTH_PROBE_FORMAT_VERSION",
    "DEPTH_PROBE_WEIGHTS_NAME",
    "DepthProbeArtifact",
    "load_depth_probe_artifact",
    "save_depth_probe_artifact",
]
