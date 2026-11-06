#!/usr/bin/env python3
"""Hugging Face / ModelScope publishing script for Vision Banana checkpoints.

After training a LoRA checkpoint, use this script to package and publish it
to Hugging Face Hub with an auto-generated model card.

Usage:
    # Publish a trained checkpoint
    python scripts/publish_to_hub.py \
        --checkpoint output/checkpoints/best_model.pt \
        --repo-id open-vision-banana/vision-banana-depth-v1 \
        --task depth \
        --description "Depth estimation LoRA trained on Hypersim"

    # Dry run (show what would be published without uploading)
    python scripts/publish_to_hub.py \
        --checkpoint output/checkpoints/best_model.pt \
        --repo-id open-vision-banana/vision-banana-depth-v1 \
        --dry-run

    # Upload a full merged pipeline to a PRIVATE repo in the authenticated account
    # Token is requested with hidden input; never put it in this script or argv.
    python scripts/publish_to_hub.py \
        --checkpoint output/ovb-e2b_cap10k_s42_10000_merged \
        --merged-pipeline

    # Retry an interrupted merged upload: add --resume to the same command.
    # Hub 2.x with hf_xet handles streamed, deduplicated uploads.

    # Publish the same merged pipeline to ModelScope (魔搭) as a PRIVATE repo.
    # --repo-id is required here because the namespace is not guessed from a token.
    python scripts/publish_to_hub.py \
        --checkpoint output/ovb-e2b_cap10k_s42_10000_merged \
        --merged-pipeline --target modelscope \
        --repo-id <your-modelscope-username>/ovb-e2b_cap10k_s42_10000_merged

    # Update registry after successful publish
    python scripts/publish_to_hub.py \
        --checkpoint output/checkpoints/best_model.pt \
        --repo-id open-vision-banana/vision-banana-depth-v1 \
        --update-registry
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import re
import struct
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent.parent / "configs" / "checkpoint_registry.yaml"

MS_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
# ModelScope SDK tokens look like ms-<uuid>; never echo them in errors or files.
MS_TOKEN_RE = re.compile(r"ms-[0-9a-fA-F][0-9a-fA-F-]{20,}")
HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")
MS_IGNORED_REMOTE = {".gitattributes", ".gitignore"}


def _build_model_card(
    task: str,
    description: str,
    repo_id: str,
    base_model: str = "black-forest-labs/FLUX.1-schnell",
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
) -> str:
    """Build a Hugging Face Model Card in markdown format."""
    metrics = metrics or {}
    training_config = training_config or {}

    task_title = "Shared Multi-Task" if task == "multitask" else task.title()
    task_description = (
        "semantic, instance, referring/reasoning, metric depth, and surface normal tasks"
        if task == "multitask"
        else f"{task} estimation"
    )
    usage_task = "depth" if task == "multitask" else task
    card = f"""# Open-Vision-Banana LoRA - {task_title}

## Model Description

{description}

This is a LoRA checkpoint for the Open-Vision-Banana project,
instruction-tuned on top of FLUX.1-schnell for {task_description} via
RGB parameterization.

This project is a method-aligned open implementation. The paper uses the closed
Nano Banana Pro base, so results from this checkpoint are not numerical
reproduction of the closed model unless an attached evaluation explicitly says so.

## Base Model

- **Base**: [{base_model}](https://huggingface.co/{base_model})
- **Architecture**: Rectified-flow transformer with a LoRA/rsLoRA adapter
- **Adapter size**: See `adapter_config.json` and the artifact file sizes

## Training

"""
    if training_config:
        card += "| Parameter | Value |\n|-----------|-------|\n"
        for k, v in training_config.items():
            card += f"| {k} | {v} |\n"
        card += "\n"

    card += """## Metrics

"""
    if metrics:
        card += "| Metric | Value |\n|--------|-------|\n"
        for k, v in metrics.items():
            card += f"| {k} | {v:.4f} |\n"
        card += "\n"
    else:
        card += "Metrics not yet reported. See evaluation section below.\n\n"

    card += f"""## Usage

```python
from open_vision_banana import VisionBanana

model = VisionBanana.from_pretrained(
    "{base_model}",
    lora_path="hf:{repo_id}",
)

input_tensor = "input.jpg"  # replace with a PIL image or tensor in your pipeline
result = model.predict(input_tensor, task="{usage_task}")
```

## Limitations

- Paper metrics do not apply to this open checkpoint unless they are explicitly
  reported from the attached benchmark protocol; the paper's base model is
  closed and cannot be reproduced numerically here.
- Requires FLUX.1 [schnell] base model weights (~24GB download)
- Supported tasks are limited to those listed by this artifact's training manifest
- RGB parameterization introduces roundtrip error (see paper for analysis)

## Citation

```bibtex
@article{{vision-banana-2026,
  title={{Image Generators are Generalist Vision Learners}},
  journal={{arXiv:2604.20329}},
  year={{2026}}
}}
```

## License

MIT License — see LICENSE file in the repository.
"""
    return card


def _update_registry(
    alias: str,
    repo_id: str,
    task: str,
    status: str = "available",
) -> None:
    """Update the local checkpoint registry YAML."""
    if not REGISTRY_PATH.exists():
        logger.warning("Registry file not found: %s", REGISTRY_PATH)
        return

    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f) or {}

    checkpoints = data.setdefault("checkpoints", {})
    checkpoints[alias] = {
        "repo_id": repo_id,
        "revision": "main",
        "status": status,
        "description": f"Published {task} LoRA checkpoint",
        "tasks": [task],
    }

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    logger.info("Updated registry: %s → %s (%s)", alias, repo_id, status)


def publish_checkpoint(
    checkpoint_path: Path,
    repo_id: str,
    task: str,
    description: str,
    base_model: str = "black-forest-labs/FLUX.1-schnell",
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
    dry_run: bool = False,
    update_registry: bool = False,
) -> int:
    """Package and publish a checkpoint to Hugging Face Hub."""
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    run_root = checkpoint_path.parent
    supplemental_files = [
        path
        for path in (
            run_root / "training_config.yaml",
            run_root / "experiment_manifest.json",
        )
        if path.is_file()
    ]

    # Build model card
    model_card = _build_model_card(
        task=task,
        description=description,
        repo_id=repo_id,
        base_model=base_model,
        metrics=metrics,
        training_config=training_config,
    )

    if dry_run:
        print("=" * 60)
        print("DRY RUN — Would publish to Hugging Face Hub")
        print("=" * 60)
        print(f"  Repo:     {repo_id}")
        print(f"  File:     {checkpoint_path}")
        print(f"  Task:     {task}")
        print(f"  Metadata: {[path.name for path in supplemental_files]}")
        print(f"  Card:     {len(model_card)} chars")
        print("=" * 60)
        print(model_card[:2000])
        print("... [truncated]" if len(model_card) > 2000 else "")
        return 0

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    # Create repo (if not exists)
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        logger.info("Repository ready: %s", repo_id)
    except Exception as exc:
        logger.error("Failed to create repo: %s", exc)
        return 1

    # Upload checkpoint or complete PEFT adapter directory.
    try:
        if checkpoint_path.is_dir():
            api.upload_folder(folder_path=str(checkpoint_path), repo_id=repo_id)
            logger.info("Uploaded adapter directory to %s", repo_id)
        else:
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=checkpoint_path.name,
                repo_id=repo_id,
            )
            logger.info("Uploaded checkpoint to %s/%s", repo_id, checkpoint_path.name)
    except Exception as exc:
        logger.error("Upload failed: %s", exc)
        return 1


    for supplemental_path in supplemental_files:
        try:
            api.upload_file(
                path_or_fileobj=str(supplemental_path),
                path_in_repo=supplemental_path.name,
                repo_id=repo_id,
            )
        except Exception as exc:
            logger.error("Failed to upload %s: %s", supplemental_path.name, exc)
            return 1

    # Upload model card
    try:
        api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        logger.info("Uploaded model card to %s/README.md", repo_id)
    except Exception as exc:
        logger.error("Model card upload failed: %s", exc)
        return 1

    logger.info("✅ Successfully published to https://huggingface.co/%s", repo_id)

    if update_registry:
        alias = repo_id.split("/")[-1].replace("-", "_")
        _update_registry(alias, repo_id, task, status="available")

    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _safetensors_keys(path: Path) -> set[str]:
    """Check header and data length without loading multi-GB tensors into RAM."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= min(size - 8, 128 * 1024 * 1024):
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(length))
    data_length = size - 8 - length
    entries = {key: value for key, value in header.items() if key != "__metadata__"}
    if not entries or any("lora_" in key or "base_layer" in key for key in entries):
        raise ValueError(f"Empty model or unmerged adapter keys in {path}")
    intervals = []
    for key, record in entries.items():
        start, end = record["data_offsets"]
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start <= end <= data_length):
            raise ValueError(f"Invalid tensor offsets in {path}: {key}")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"Overlapping or missing tensor data in {path}")
        cursor = end
    if cursor != data_length:
        raise ValueError(f"Truncated or extra tensor data in {path}")
    # The official parser also validates dtype/shape byte sizes. Only inspect
    # keys; no arrays are materialized (including BF16 tensors).
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework="numpy") as stream:
            if set(stream.keys()) != set(entries):
                raise ValueError(f"Safetensors key mismatch in {path}")
    except Exception as error:
        raise ValueError(f"Invalid safetensors structure: {path}") from error
    return set(entries)


def _merged_upload_manifest(folder: Path) -> dict[str, tuple[int, str]]:
    """Validate the requested pipeline and whitelist files, excluding caches/secrets."""
    if not folder.is_dir() or (folder / "MERGE_INCOMPLETE").exists():
        raise ValueError("Missing merged directory or MERGE_INCOMPLETE is present")
    if not (folder / "MERGE_VALIDATED").is_file():
        raise ValueError("MERGE_VALIDATED is missing; do not publish an unverified merge")
    with (folder / "merge_metadata.json").open() as stream:
        metadata = json.load(stream)
    required_checks = ("download_sha256", "adapter_loaded_exactly", "all_lora_merge_formulas",
                       "all_exported_tensor_hashes", "copied_component_hashes",
                       "offline_component_loading")
    if metadata.get("base_repo") != "Qwen/Qwen-Image-2.1" or any(
            metadata.get("validation", {}).get(check) is not True for check in required_checks):
        raise ValueError("Merge provenance or recorded component verification is incomplete")
    with (folder / "model_index.json").open() as stream:
        if json.load(stream).get("_class_name") != "QwenImage21Pipeline":
            raise ValueError("Expected QwenImage21Pipeline")
    root_files = {"README.md", "LICENSE", "NOTICE.txt", "model_index.json",
                  "merge_metadata.json", "MERGE_VALIDATED"}
    components = {"transformer", "text_encoder", "vae", "processor", "scheduler"}
    selected = {}
    for path in sorted(folder.rglob("*")):
        relative = path.relative_to(folder)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        name = relative.as_posix()
        if name not in root_files and not (relative.parts[0] in components
                and path.suffix in {".json", ".safetensors", ".txt", ".jinja"}):
            continue
        if path.resolve() != path.absolute():
            raise ValueError(f"Refusing symlink in standalone pipeline: {name}")
        if not name.endswith(".safetensors"):
            if path.stat().st_size > 128 * 1024 * 1024:
                raise ValueError(f"Unexpected oversized metadata: {name}")
            text = path.read_text(encoding="utf-8")
            if HF_TOKEN_RE.search(text) or MS_TOKEN_RE.search(text):
                raise ValueError(f"Possible credential in upload file: {name}")
        selected[name] = path
    if not root_files.issubset(selected):
        raise ValueError(f"Missing publication files: {sorted(root_files - selected.keys())}")
    if "Qwen RESEARCH LICENSE AGREEMENT" not in (folder / "LICENSE").read_text():
        raise ValueError("The upstream research license must be retained")
    if "Built with Qwen" not in (folder / "README.md").read_text():
        raise ValueError("Missing derivative attribution in README.md")
    expected_weights = set()
    for component, index_name in (
            ("transformer", "diffusion_pytorch_model.safetensors.index.json"),
            ("text_encoder", "model.safetensors.index.json")):
        index_path = folder / component / index_name
        with index_path.open() as stream:
            weight_map = json.load(stream)["weight_map"]
        if not weight_map:
            raise ValueError(f"Empty tensor index: {index_path}")
        for shard in set(weight_map.values()):
            if Path(shard).name != shard or not shard.endswith(".safetensors"):
                raise ValueError(f"Invalid shard name in {index_path}")
            name = f"{component}/{shard}"
            if name not in selected:
                raise ValueError(f"Missing weight shard: {name}")
            if _safetensors_keys(selected[name]) != {key for key, value in weight_map.items() if value == shard}:
                raise ValueError(f"Weight keys disagree with index: {name}")
            if component == "transformer" and name + ".NOTICE.txt" not in selected:
                raise ValueError(f"Missing modified-shard notice: {name}")
            expected_weights.add(name)
    vae = "vae/diffusion_pytorch_model.safetensors"
    if vae not in selected:
        raise ValueError("Missing VAE weights")
    _safetensors_keys(selected[vae])
    expected_weights.add(vae)
    actual_weights = {name for name in selected if name.endswith(".safetensors")}
    if actual_weights != expected_weights:
        raise ValueError("Unexpected extra model weights; inspect the merged directory")
    for component in components:
        required = "scheduler_config.json" if component == "scheduler" else (
            "preprocessor_config.json" if component == "processor" else "config.json")
        if f"{component}/{required}" not in selected:
            raise ValueError(f"Missing {component} configuration")
    print(f"Validated headers for {len(expected_weights)} weight shards. Hashing upload files...", flush=True)
    return {name: (path.stat().st_size, _sha256(path)) for name, path in selected.items()}


def publish_merged_pipeline(folder: Path, repo_id: str | None = None,
                            public: bool = False, resume: bool = False,
                            dry_run: bool = False) -> int:
    """Upload a complete merged pipeline; never embed or persist a credential.

    Hub 2.x upload_folder uses hf_xet for streamed, deduplicated, resumable
    multi-commit uploads. Re-run with --resume after an interruption.
    """
    folder = folder.expanduser().resolve()
    manifest = _merged_upload_manifest(folder)
    total = sum(size for size, _ in manifest.values())
    print(f"Upload plan: {len(manifest)} files, {total / 1e9:.2f} GB; "
          f"requested visibility={'public' if public else 'private'}", flush=True)
    for name in manifest:
        print(f"  {name}", flush=True)
    if dry_run:
        print("DRY RUN: no authentication, remote repository or upload was performed.", flush=True)
        return 0
    if not sys.stdin.isatty():
        raise ValueError("Run in an interactive terminal for hidden token input")
    # Uploads must go to the official Hub, never to a download mirror.
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"
    os.environ["HF_DEBUG"] = "0"
    from huggingface_hub import HfApi, hf_hub_download

    version = importlib.metadata.version("huggingface_hub")
    if int(version.split(".")[0]) < 2:
        raise ValueError("This resumable upload path requires huggingface_hub >=2.0")
    # Abort instead of falling back to echoed input if the TTY cannot hide it.
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        token = getpass.getpass("Hugging Face write token (hidden; not saved): ").strip()
    if not token:
        raise ValueError("Empty token; no upload performed")
    api = HfApi(endpoint="https://huggingface.co", token=token)
    account = api.whoami(token=token)["name"]
    repo_id = repo_id or f"{account}/{folder.name}"
    if re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id) is None:
        raise ValueError("Invalid HF repository ID")
    print(f"Destination: https://huggingface.co/{repo_id}", flush=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True, token=token)
    info = api.model_info(repo_id, files_metadata=True, token=token, timeout=60)
    if bool(info.private) != (not public):
        raise ValueError("Existing repository visibility differs; it was NOT changed. "
                         "Choose another repo or explicitly use --public for a public repo")
    remote = {entry.rfilename: entry for entry in info.siblings
              if entry.rfilename != ".gitattributes"}
    if remote and not resume:
        raise ValueError("Repository already contains files. Use --resume only to continue "
                         "this same upload; existing remote files were not changed")
    for name, entry in remote.items():
        if name not in manifest or entry.size != manifest[name][0]:
            raise ValueError(f"Existing repo is not the same upload: {name}")
        if entry.lfs is not None:
            if entry.lfs.sha256 != manifest[name][1]:
                raise ValueError(f"Existing remote weight differs: {name}")
        else:
            downloaded = hf_hub_download(repo_id, name, revision=info.sha, token=token,
                                         endpoint="https://huggingface.co", etag_timeout=120)
            if _sha256(Path(downloaded)) != manifest[name][1]:
                raise ValueError(f"Existing remote metadata differs: {name}")
    try:
        import hf_xet  # noqa: F401
        xet_enabled = os.environ.get("HF_HUB_DISABLE_XET", "").lower() not in {"1", "true", "yes", "on"}
    except ImportError:
        xet_enabled = False
    if not xet_enabled:
        raise ValueError("Resumable large upload requires hf_xet and enabled Xet. "
                         "Install hf-xet and unset HF_HUB_DISABLE_XET in this terminal")
    print("Uploading to official HF. Re-run the same command with --resume if interrupted.", flush=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(folder),
        allow_patterns=list(manifest), token=token,
        commit_message="Upload validated Open-Vision-Banana E2b merged pipeline",
    )
    info = api.model_info(repo_id, files_metadata=True, token=token, timeout=60)
    entries = {entry.rfilename: entry for entry in info.siblings}
    for name, (size, digest) in manifest.items():
        entry = entries.get(name)
        if entry is None or entry.size != size:
            raise RuntimeError(f"Remote verification failed (missing or wrong size): {name}")
        if entry.lfs is not None:
            if entry.lfs.sha256 != digest:
                raise RuntimeError(f"Remote SHA256 verification failed: {name}")
        else:
            downloaded = hf_hub_download(repo_id, name, revision=info.sha, token=token,
                                         endpoint="https://huggingface.co", etag_timeout=120)
            if _sha256(Path(downloaded)) != digest:
                raise RuntimeError(f"Remote metadata verification failed: {name}")
    del api, token
    print(f"SUCCESS: https://huggingface.co/{repo_id}\nCommit: {info.sha}\n"
          f"Verified: {len(manifest)} files, {total / 1e9:.2f} GB, remote SHA256 matches.\n"
          "Image inference was not tested.", flush=True)
    return 0


def _ms_repo_files_or_none(api, repo_id: str) -> dict[str, dict] | None:
    """Return remote files, or None only when the Hub clearly reports 'no such repo'."""
    try:
        return _ms_remote_files(api, repo_id)
    except Exception as error:  # noqa: BLE001 - classified by message/type on purpose
        text = f"{type(error).__name__} {error}".lower()
        if "notexist" in text or "not exist" in text or "404" in text:
            return None
        raise


def _ms_create_model(api, repo_id: str, public: bool, license_name: str) -> None:
    """Create a model repo; only pass kwargs this SDK version actually accepts."""
    try:
        params = inspect.signature(api.create_model).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = {"model_id": repo_id}
    if "visibility" in params:
        # ModelScope constants: 1 = private, 5 = public.
        kwargs["visibility"] = 5 if public else 1
    if "license" in params and license_name:
        kwargs["license"] = license_name
    api.create_model(**kwargs)


def _ms_staging(folder: Path, names: list[str]) -> Path:
    """Hardlink the whitelisted files so only they are uploaded, at zero extra space."""
    staging = folder.parent / f".modelscope_upload_{folder.name}"
    staging.mkdir(exist_ok=True)
    for name in names:
        source = folder / name
        target = staging / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if (target.stat().st_size != source.stat().st_size
                    or target.stat().st_ino != source.stat().st_ino):
                raise ValueError(
                    f"Staging file conflicts with an earlier run: {target}. "
                    "Inspect and remove that staging directory yourself, then retry")
            continue
        os.link(source, target)
    return staging


def _ms_upload(api, repo_id: str, staging: Path, message: str) -> None:
    try:
        params = inspect.signature(api.upload_folder).parameters
    except (TypeError, ValueError):
        params = {}
    kwargs = {"repo_id": repo_id, "folder_path": str(staging)}
    if "commit_message" in params:
        kwargs["commit_message"] = message
    if "repo_type" in params:
        kwargs["repo_type"] = "model"
    api.upload_folder(**kwargs)


def _ms_remote_files(api, repo_id: str) -> dict[str, dict]:
    entries = api.get_model_files(repo_id, recursive=True)
    remote: dict[str, dict] = {}
    for entry in entries:
        name = str(entry.get("Path") or entry.get("Name") or "").strip("/")
        if not name or entry.get("Type") in {"tree", "dir"}:
            continue
        remote[name] = entry
    return remote


def _ms_compare(remote: dict[str, dict], manifest: dict[str, tuple[int, str]]
                ) -> tuple[int, int, list[str]]:
    """Compare remote sizes always, and remote SHA256 wherever the Hub reports it."""
    matched = 0
    sha_checked = 0
    problems: list[str] = []
    for name, (size, digest) in manifest.items():
        entry = remote.get(name)
        if entry is None:
            problems.append(f"missing: {name}")
            continue
        remote_size = entry.get("Size")
        if isinstance(remote_size, str) and remote_size.isdigit():
            remote_size = int(remote_size)
        if remote_size != size:
            problems.append(f"size differs: {name} ({remote_size} != {size})")
            continue
        remote_sha = entry.get("Sha256") or entry.get("SHA256")
        if isinstance(remote_sha, str) and remote_sha:
            sha_checked += 1
            if remote_sha.lower() != digest:
                problems.append(f"sha256 differs: {name}")
                continue
        matched += 1
    return matched, sha_checked, problems


def _ms_verify(api, repo_id: str, manifest: dict[str, tuple[int, str]]) -> tuple[int, int]:
    remote = _ms_remote_files(api, repo_id)
    extra = sorted(set(remote) - set(manifest) - MS_IGNORED_REMOTE)
    if extra:
        raise RuntimeError(f"Repository contains files outside this upload: {extra[:5]}")
    _, sha_checked, problems = _ms_compare(remote, manifest)
    if problems:
        raise RuntimeError("Remote verification failed: " + "; ".join(problems[:8]))
    return len(remote), sha_checked


def publish_merged_pipeline_modelscope(folder: Path, repo_id: str, public: bool = False,
                                       dry_run: bool = False, create: bool = True,
                                       license_name: str = "other") -> int:
    """Upload a validated merged pipeline to ModelScope without persisting the token."""
    folder = folder.expanduser().resolve()
    if not repo_id or MS_REPO_RE.fullmatch(repo_id) is None:
        raise ValueError("ModelScope needs an explicit --repo-id <owner>/<name>")
    manifest = _merged_upload_manifest(folder)
    total = sum(size for size, _ in manifest.values())
    print(f"Upload plan: {len(manifest)} files, {total / 1e9:.2f} GB; "
          f"destination=modelscope:{repo_id}; "
          f"visibility={'public' if public else 'private'}", flush=True)
    for name in manifest:
        print(f"  {name}", flush=True)
    print(f"Staging (hardlinks, no extra disk): "
          f"{folder.parent / f'.modelscope_upload_{folder.name}'}", flush=True)
    if dry_run:
        print("DRY RUN: no authentication, repository or upload was performed.", flush=True)
        return 0
    if not sys.stdin.isatty():
        raise ValueError("Run in an interactive terminal for hidden token input")
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        raise ValueError("ModelScope SDK is not installed in this interpreter. "
                         "Run: python -m pip install -U modelscope") from None

    # HubApi(token=...) authenticates per request; only login() writes credentials
    # to ~/.modelscope/credentials, so this path deliberately never calls login().
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        token = getpass.getpass("ModelScope SDK token (hidden; not saved): ").strip()
    if not token:
        raise ValueError("Empty token; no upload performed")
    api = HubApi(token=token)
    existing = _ms_repo_files_or_none(api, repo_id)
    if existing is None:
        if not create:
            raise ValueError(f"Repository {repo_id} does not exist and --no-create was given")
        print(f"Creating model repo {repo_id}...", flush=True)
        try:
            _ms_create_model(api, repo_id, public, license_name)
        except Exception as error:  # noqa: BLE001 - actionable fallback below
            raise ValueError(
                "Could not create the repository automatically. Create it on "
                "modelscope.cn with visibility 'Private' and license 'Other', then "
                f"re-run with --no-create. SDK error type: {type(error).__name__}"
            ) from None
    elif existing:
        extra = sorted(set(existing) - set(manifest) - MS_IGNORED_REMOTE)
        if extra:
            raise ValueError("Repository contains files outside this upload; "
                             f"refusing to mix them: {extra[:5]}")
        matched, _, problems = _ms_compare(existing, manifest)
        conflicts = [item for item in problems if not item.startswith("missing: ")]
        if conflicts:
            raise ValueError("Repository already contains different content; nothing was "
                             "overwritten. Inspect it first: " + "; ".join(conflicts[:5]))
        print(f"{matched} of {len(manifest)} remote files already match; "
              "uploading the remainder.", flush=True)
        if matched == len(manifest):
            print(f"SUCCESS (no changes needed): "
                  f"https://www.modelscope.cn/models/{repo_id}", flush=True)
            return 0
    staging = _ms_staging(folder, list(manifest))
    print("Uploading to ModelScope. Re-run the same command if it is interrupted.",
          flush=True)
    _ms_upload(api, repo_id, staging,
               "Upload validated Open-Vision-Banana E2b merged pipeline")
    file_count, sha_checked = _ms_verify(api, repo_id, manifest)
    del api, token
    print(f"SUCCESS: https://www.modelscope.cn/models/{repo_id}\n"
          f"Verified: {file_count} remote files, sizes for all {len(manifest)} upload files, "
          f"SHA256 for {sha_checked} of them.\n"
          f"Staging directory kept for retries: {staging}\n"
          "Image inference was not tested.", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a Vision Banana checkpoint or merged pipeline to HF")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint or merged pipeline")
    parser.add_argument("--repo-id", type=str,
                        help="HF repo ID; merged mode defaults to authenticated account/directory-name")
    parser.add_argument("--merged-pipeline", action="store_true",
                        help="Publish a validated complete Qwen21 pipeline, not an adapter")
    parser.add_argument("--public", action="store_true",
                        help="Explicitly select a public repo in merged mode; default is private")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the same merged upload; verify existing remote files first")
    parser.add_argument("--target", choices=["hf", "modelscope"], default="hf",
                        help="Upload destination for merged mode; default hf")
    parser.add_argument("--no-create", action="store_true",
                        help="ModelScope: require the repo to already exist; never create it")
    parser.add_argument("--license", type=str, default="other",
                        help="ModelScope repo license label; default 'other' keeps the "
                             "upstream Qwen research license from being overridden")
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        choices=["multitask", "depth", "normal", "semantic", "instance", "referring"],
    )
    parser.add_argument("--description", type=str, default="", help="Model description")
    parser.add_argument("--base-model", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be published")
    parser.add_argument("--update-registry", action="store_true", help="Update local checkpoint registry")
    args = parser.parse_args()
    if not args.merged_pipeline and (not args.repo_id or not args.task):
        parser.error("LoRA publishing requires --repo-id and --task")
    if not args.merged_pipeline and (args.public or args.resume):
        parser.error("--public and --resume are only supported in merged-pipeline mode")
    if args.merged_pipeline and args.update_registry:
        parser.error("Merged pipelines are not registered as LoRA adapters")
    if args.target == "modelscope":
        if not args.merged_pipeline:
            parser.error("--target modelscope is only supported with --merged-pipeline")
        if not args.repo_id:
            parser.error("ModelScope requires --repo-id <owner>/<name>; the namespace "
                         "is not inferred from the token")
        if args.resume:
            parser.error("--resume applies to Hugging Face only; just re-run the same "
                         "ModelScope command, unchanged files are skipped")
    return args


def _redact(message: str) -> str:
    return MS_TOKEN_RE.sub("[REDACTED]", HF_TOKEN_RE.sub("[REDACTED]", message))


def main() -> int:
    args = parse_args()
    if args.merged_pipeline:
        try:
            if args.target == "modelscope":
                return publish_merged_pipeline_modelscope(
                    Path(args.checkpoint), repo_id=args.repo_id, public=args.public,
                    dry_run=args.dry_run, create=not args.no_create,
                    license_name=args.license,
                )
            return publish_merged_pipeline(
                Path(args.checkpoint), repo_id=args.repo_id, public=args.public,
                resume=args.resume, dry_run=args.dry_run,
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"Upload stopped: {_redact(str(error))}", file=sys.stderr)
            return 1
        except Exception as error:
            # Never dump request objects, auth responses or exception locals.
            hint = ("Check the SDK token, storage quota and connectivity to modelscope.cn; "
                    "re-run the same command after fixing it. Token was not saved."
                    if args.target == "modelscope" else
                    "Check the write token, storage quota and terminal connectivity to "
                    "official HF. If an upload was interrupted, re-run with --resume. "
                    "Token was not saved.")
            print(f"Upload/verification stopped ({type(error).__name__}). "
                  f"No successful upload is claimed. {hint}", file=sys.stderr)
            return 1
    return publish_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        repo_id=args.repo_id,
        task=args.task,
        description=args.description or f"Vision Banana {args.task} LoRA checkpoint",
        base_model=args.base_model,
        dry_run=args.dry_run,
        update_registry=args.update_registry,
    )


if __name__ == "__main__":
    sys.exit(main())
