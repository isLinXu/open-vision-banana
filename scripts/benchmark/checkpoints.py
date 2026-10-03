"""Prepare one revision-pinned HF LoRA without downloading a model backbone.

Network access is opt-in. The same source cache is shared by online and EMA
exports, but their PEFT adapter directories are always distinct. This module
imports neither the project inference package nor CUDA at import time.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen


class CheckpointValidationError(ValueError):
    """A pinned source or its exported adapter failed validation."""


def _portable_metadata(value: Any) -> Any:
    """Avoid copying local roots from training metadata into a shared summary.

    Original adapter config/source bytes remain unchanged and checksummed. This
    only redacts absolute strings in the human-readable metadata snapshot.
    """
    if isinstance(value, str) and (PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive):
        return "<local-path>"
    if isinstance(value, dict):
        return {key: _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_metadata(item) for item in value]
    return value


def resolve_prepared_paths(record: dict, record_path: Path) -> dict:
    """Resolve portable cache paths for runtime use without modifying a record.

    New records are relative to their JSON file's parent directory. Every new
    local path must stay inside that directory, including through symlinks.
    Older records without ``path_base`` remain readable on their original host
    when they contain absolute paths. No model or CUDA imports are needed.
    """
    if not isinstance(record, dict):
        raise CheckpointValidationError("Prepared checkpoint record must be an object")
    resolved = copy.deepcopy(record)
    policy = resolved.get("path_base")
    if policy not in (None, "prepared_checkpoint_directory"):
        raise CheckpointValidationError(f"Unsupported prepared checkpoint path_base: {policy!r}")
    root = Path(record_path).resolve().parent

    def resolve(value: Any) -> str:
        if policy == "prepared_checkpoint_directory":
            relative = _safe_relative(value, "prepared checkpoint path")
            return str(_within(root, relative))
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise CheckpointValidationError("Legacy prepared checkpoint paths must be native absolute paths")
        return str(Path(value).resolve())

    for key in ("adapter_dir", "lora_path", "source_checkpoint"):
        if key in resolved:
            resolved[key] = resolve(resolved[key])
    provenance = resolved.get("provenance", {})
    for group, key in (("source_files", "local_path"), ("export_files", "path")):
        for entry in provenance.get(group, []):
            if not isinstance(entry, dict) or key not in entry:
                raise CheckpointValidationError(f"Prepared checkpoint {group} entry is missing {key}")
            entry[key] = resolve(entry[key])
    return resolved


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in "\\:\x00"):
        raise CheckpointValidationError(f"Invalid {label}: {value!r}")
    parts = value.split("/")
    if PurePosixPath(value).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise CheckpointValidationError(f"Unsafe {label}: {value!r}")
    # Reject Windows alternate streams, device names and normalized aliases too.
    for part in parts:
        if part[-1:] in (".", " ") or any(ord(c) < 32 for c in part):
            raise CheckpointValidationError(f"Unsafe {label}: {value!r}")
        if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part):
            raise CheckpointValidationError(f"Unsafe {label}: {value!r}")
    return value


def _within(root: Path, relative: str) -> Path:
    target = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise CheckpointValidationError(f"Path escapes destination: {relative}")
    return target


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".json-", suffix=".partial",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, record: dict) -> dict:
    size = record.get("size")
    if not isinstance(size, int) or size < 0 or path.stat().st_size != size:
        raise CheckpointValidationError(f"File size mismatch: {path}")
    sha256 = _sha256(path)
    lfs = record.get("lfs")
    if lfs:
        expected = lfs.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise CheckpointValidationError(f"Missing valid LFS SHA256: {path}")
        if sha256 != expected or lfs.get("size", size) != size:
            raise CheckpointValidationError(f"LFS SHA256/size mismatch: {path}")
    else:
        expected = record.get("blobId")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{40}", expected):
            raise CheckpointValidationError(f"Missing Git blob checksum: {path}")
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                blob.update(block)
        if blob.hexdigest() != expected:
            raise CheckpointValidationError(f"Git blob checksum mismatch: {path}")
    return {"size": size, "sha256": sha256,
            "lfs_sha256": lfs.get("sha256") if lfs else None,
            "git_blob_sha1": record.get("blobId")}


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def _download(url: str, target: Path, record: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".dl-",
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        _verify_file(temporary, record)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _canonicalize(state: dict, label: str) -> dict:
    import torch

    if not isinstance(state, dict) or not state:
        raise CheckpointValidationError(f"Missing or empty {label}")
    normalized = {}
    for key, tensor in state.items():
        if not isinstance(key, str):
            raise CheckpointValidationError(f"Non-string key in {label}")
        canonical = key.replace(".lora_A.default.", ".lora_A.").replace(
            ".lora_B.default.", ".lora_B.")
        if canonical in normalized:
            raise CheckpointValidationError(f"Canonical key collision: {canonical}")
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise CheckpointValidationError(f"Expected floating tensor: {label}/{key}")
        if not bool(torch.isfinite(tensor).all()):
            raise CheckpointValidationError(f"Non-finite tensor: {label}/{key}")
        normalized[canonical] = tensor.detach().cpu().contiguous()
    return normalized


def _check_matching(candidate: dict, reference: dict, label: str, *, equal=False) -> None:
    import torch

    if candidate.keys() != reference.keys():
        raise CheckpointValidationError(f"{label} key set differs from adapter")
    for key in reference:
        if candidate[key].shape != reference[key].shape:
            raise CheckpointValidationError(f"{label} shape mismatch: {key}")
        if equal and not torch.equal(candidate[key], reference[key]):
            raise CheckpointValidationError(f"{label} differs from online adapter: {key}")


def _check_lora_pairs(tensors: dict, config: dict) -> int:
    rank = config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise CheckpointValidationError("adapter_config.r must be a positive integer")
    count = 0
    for key, a in tensors.items():
        if ".lora_A." not in key:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in tensors:
            raise CheckpointValidationError(f"Missing LoRA B pair: {key}")
        module = key.split(".lora_A.")[0]
        module_rank = rank
        for pattern, value in (config.get("rank_pattern") or {}).items():
            if re.match(rf"(.*\.)?({pattern})$", module):
                module_rank = value
                break
        b = tensors[b_key]
        if a.ndim < 2 or b.ndim < 2 or a.shape[0] != module_rank or b.shape[1] != module_rank:
            raise CheckpointValidationError(f"LoRA pair/rank shape mismatch: {key}")
        count += 1
    if not count or any(".lora_B." in k and k.replace(".lora_B.", ".lora_A.") not in tensors
                        for k in tensors):
        raise CheckpointValidationError("Missing LoRA A/B pairs")
    return count


def _export_adapter(target: Path, tensors: dict, config_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    if target.exists():
        if not (target / "adapter_config.json").is_file() or not (
                target / "adapter_model.safetensors").is_file():
            raise CheckpointValidationError(f"Incomplete existing adapter directory: {target}")
        if _sha256(target / "adapter_config.json") != _sha256(config_path):
            raise CheckpointValidationError(f"Cached exported config mismatch: {target}")
        cached = _canonicalize(load_file(str(target / "adapter_model.safetensors")), "export")
        _check_matching(cached, tensors, "Cached export", equal=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=target.name + ".", suffix=".partial", dir=target.parent))
    try:
        save_file({k: tensors[k] for k in sorted(tensors)}, str(stage / "adapter_model.safetensors"))
        shutil.copyfile(config_path, stage / "adapter_config.json")
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def prepare_checkpoint(spec: dict, destination: Path, allow_download: bool = False) -> dict:
    """Verify sources and export an explicitly selected online/EMA PEFT adapter.

    ``spec`` requires repo_id, a full 40-hex revision, checkpoint_subdir and
    weights='ema'|'online'; base_model_id is optional. Selection annotations are
    ignored. A false allow_download uses an existing pinned manifest and makes
    no network requests. Corrupt cached sources fail closed, or are atomically
    replaced when downloads are enabled. No backbone compatibility is implied.
    """
    if not isinstance(spec, dict):
        raise CheckpointValidationError("Checkpoint spec must be an object")
    repo = spec.get("repo_id")
    if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo, re.ASCII):
        raise CheckpointValidationError("repo_id must be an owner/model HF identifier")
    _safe_relative(repo, "repo_id")
    revision = spec.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise CheckpointValidationError("revision must be a full 40-hex commit SHA")
    revision = revision.lower()
    subdir = _safe_relative(spec.get("checkpoint_subdir"), "checkpoint_subdir")
    variant = spec.get("weights")
    if variant not in ("ema", "online"):
        raise CheckpointValidationError("weights must explicitly be 'ema' or 'online'")
    base = spec.get("base_model_id")
    if base is not None and not isinstance(base, str):
        raise CheckpointValidationError("base_model_id must be a string")
    destination = Path(destination).resolve()
    cache_root = _within(destination, f"sources/{repo.replace('/', '--')}/{revision}")
    manifest_path = _within(cache_root, "manifest.json")
    manifest_url = f"https://huggingface.co/api/models/{quote(repo, safe='/')}/revision/{revision}?blobs=true"
    if allow_download:
        manifest = _fetch_json(manifest_url)
        if manifest.get("sha") != revision:
            raise CheckpointValidationError("HF manifest revision differs from pinned SHA")
        if manifest.get("id", repo) != repo:
            raise CheckpointValidationError("HF manifest repo differs from requested repo")
        _atomic_json(manifest_path, manifest)
    elif manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("No pinned manifest cache; rerun with allow_download=True")
    if manifest.get("sha") != revision or manifest.get("id", repo) != repo:
        raise CheckpointValidationError("Cached manifest does not match pinned repository/revision")
    entries = manifest.get("siblings")
    if not isinstance(entries, list):
        raise CheckpointValidationError("HF manifest has no file inventory")
    inventory = {}
    for record in entries:
        if not isinstance(record, dict) or not isinstance(record.get("rfilename"), str):
            raise CheckpointValidationError("Invalid manifest file record")
        name = record.get("rfilename")
        if name in inventory:
            raise CheckpointValidationError(f"Duplicate manifest entry: {name}")
        inventory[name] = record
    wanted = ["metadata.json", "lora_adapter/adapter_config.json",
              "lora_adapter/adapter_model.safetensors"]
    if variant == "ema":
        wanted.append("lora.pt")
    files = {}
    provenance_files = []
    for relative in wanted:
        remote = f"{subdir}/{relative}"
        if remote not in inventory:
            raise CheckpointValidationError(f"Pinned checkpoint is missing {remote}")
        target = _within(cache_root, remote)
        record = inventory[remote]
        url = f"https://huggingface.co/{quote(repo, safe='/')}/resolve/{revision}/{quote(remote, safe='/')}"
        if target.is_file():
            try:
                verification = _verify_file(target, record)
            except CheckpointValidationError:
                if not allow_download:
                    raise
                _download(url, target, record)
                verification = _verify_file(target, record)
        elif allow_download:
            _download(url, target, record)
            verification = _verify_file(target, record)
        else:
            raise FileNotFoundError(f"Source not cached: {remote}; enable allow_download")
        files[relative] = target
        provenance_files.append({"remote_path": remote, "source_url": url,
                                 "revision": revision,
                                 "local_path": target.relative_to(destination).as_posix(), **verification})

    metadata = json.loads(files["metadata.json"].read_text(encoding="utf-8"))
    config_path = files["lora_adapter/adapter_config.json"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or not isinstance(config, dict):
        raise CheckpointValidationError("Metadata and adapter config must be JSON objects")
    directory_step = re.fullmatch(r"checkpoint-(\d+)(?:[_-].*)?", PurePosixPath(subdir).name)
    if directory_step and metadata.get("step") is not None:
        if metadata["step"] != int(directory_step.group(1)):
            raise CheckpointValidationError("Metadata step disagrees with checkpoint directory")
    for metadata_key, config_key in (("lora_rank", "r"), ("lora_alpha", "lora_alpha")):
        if metadata_key in metadata and metadata[metadata_key] != config.get(config_key):
            raise CheckpointValidationError(f"Metadata {metadata_key} disagrees with adapter config")
    declared_base = config.get("base_model_name_or_path")
    if base and declared_base and base != declared_base:
        raise CheckpointValidationError("base_model_id disagrees with adapter config")

    from safetensors.torch import load_file

    online = _canonicalize(load_file(str(files["lora_adapter/adapter_model.safetensors"])), "adapter")
    selected = online
    if variant == "ema":
        import torch

        try:
            state = torch.load(files["lora.pt"], map_location="cpu", weights_only=True)
        except TypeError as error:
            raise RuntimeError("EMA export requires PyTorch supporting safe weights_only=True") from error
        if not isinstance(state, dict):
            raise CheckpointValidationError("lora.pt must contain a state dictionary")
        selected = _canonicalize(state.get("ema_state_dict"), "ema_state_dict")
        checkpoint_online = _canonicalize(state.get("lora_state_dict"), "lora_state_dict")
        _check_matching(checkpoint_online, online, "Checkpoint online", equal=True)
        _check_matching(selected, online, "EMA")
        if state.get("step") is not None and metadata.get("step") is not None:
            if state["step"] != metadata["step"]:
                raise CheckpointValidationError("lora.pt step disagrees with metadata")
    pairs = _check_lora_pairs(selected, config)
    adapter_dir = _within(destination, f"adapters/{repo.replace('/', '--')}/{revision}/{subdir}/{variant}")
    _export_adapter(adapter_dir, selected, config_path)
    adapter_relative = adapter_dir.relative_to(destination).as_posix()
    source_checkpoint = (files["lora.pt"] if variant == "ema" else
                         files["lora_adapter/adapter_model.safetensors"])
    result = {
        "schema_version": 1, "path_base": "prepared_checkpoint_directory",
        "repo_id": repo, "revision": revision,
        "checkpoint_subdir": subdir, "weights_variant": variant,
        "adapter_dir": adapter_relative, "lora_path": adapter_relative,
        "adapter_sha256": _sha256(adapter_dir / "adapter_model.safetensors"),
        "source_checkpoint": source_checkpoint.relative_to(destination).as_posix(),
        "base_model_id": ("<local-base-model>" if _portable_metadata(base or declared_base) ==
                          "<local-path>" else base or declared_base),
        "metadata": _portable_metadata(metadata),
        "adapter_configuration": _portable_metadata(config), "adapter_tensor_count": len(selected),
        "lora_parameter_count": sum(t.numel() for t in selected.values()),
        "validated_lora_pairs": pairs,
        "validation": {"safe_weights_only": variant == "ema",
                       "base_architecture_shape_verified": False,
                       "base_weights_downloaded": False, "cuda_initialization_requested": False},
        "provenance": {
            "source_revision": revision, "manifest_url": manifest_url,
            "manifest_sha256": _sha256(manifest_path), "weights_variant": variant,
            "source_files": provenance_files,
            "export_files": [{"path": (adapter_dir / name).relative_to(destination).as_posix(),
                              "sha256": _sha256(adapter_dir / name)}
                             for name in ("adapter_config.json", "adapter_model.safetensors")],
        },
    }
    return result


def _select_spec(document: dict, candidate: str | None = None) -> dict:
    if not isinstance(document, dict):
        raise CheckpointValidationError("Spec document must be a JSON object")
    if "candidates" not in document:
        if candidate is not None:
            raise CheckpointValidationError("--candidate requires a candidates document")
        return document
    selected = candidate or document.get("selected")
    entries = document["candidates"]
    if isinstance(entries, dict):
        if not all(isinstance(spec, dict) for spec in entries.values()):
            raise CheckpointValidationError("Candidates mapping values must be objects")
        entries = [{**spec, "id": key} for key, spec in entries.items()]
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise CheckpointValidationError("Candidates must be a list of objects or an id/object mapping")
    matches = [entry for entry in entries if entry.get("id") == selected]
    if len(matches) != 1:
        raise CheckpointValidationError(f"Candidate must uniquely exist: {selected!r}")
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--output", type=Path, required=True, help="Source/export cache directory")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    document = json.loads(args.spec.read_text(encoding="utf-8"))
    spec = _select_spec(document, args.candidate)
    result = prepare_checkpoint(spec, args.output, args.allow_download)
    result_path = args.output / "prepared-checkpoint.json"
    _atomic_json(result_path, result)
    print(json.dumps({"adapter_dir": result["adapter_dir"],
                      "weights_variant": result["weights_variant"],
                      "path_base": result["path_base"],
                      "provenance_path": "prepared-checkpoint.json"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
