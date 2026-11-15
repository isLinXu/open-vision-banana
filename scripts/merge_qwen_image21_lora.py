#!/usr/bin/env python3
"""Download pinned Qwen-Image-2.1 + OVB rsLoRA and export an offline pipeline.

Run with Python >= 3.10. Dependencies matching the published training environment:
  torch>=2.4, diffusers==0.41.0, transformers==5.19.0, peft==0.21.0,
  accelerate>=1.15.0, huggingface_hub>=1.33.0, safetensors>=0.8.0, pillow.

Only the transformer is held in memory during merging. CPU merging needs no GPU;
32 GiB RAM is recommended. No pickle checkpoints or repository code are loaded.
All output components are real copies (not links to the download cache).
The training artifact did not record its base revision; the default below is
an explicitly pinned official snapshot, not a claimed training-revision match.
This script has not been executed by the authoring session.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from urllib.parse import urlsplit

BASE_REPO = "Qwen/Qwen-Image-2.1"
BASE_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
DEFAULT_LORA_REPO = "khazic/ovb-e2b_cap10k_s42_20261026_1016"
DEFAULT_LORA_REVISION = "036bc7285371e24b9e52ab1b0ad9aea9ea337f27"
DEFAULT_CHECKPOINT = "ckpt/checkpoint-10000_final"
COMPONENTS = ("processor", "scheduler", "text_encoder", "transformer", "vae")
ROOT_FILES = ("model_index.json", "LICENSE", "README.md")
TARGETS = {"to_q", "to_k", "to_v", "to_out.0", "img_mlp.proj",
           "img_mlp.gate_layer", "img_mlp.out"}


def normalize_endpoint(value: str) -> str:
    """Reject insecure endpoints and credentials embedded in logged URLs."""
    parts = urlsplit(value)
    if (parts.scheme != "https" or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment
            or parts.path not in ("", "/")):
        raise ValueError("--endpoint must be an HTTPS origin without credentials or a path")
    return value.rstrip("/")


def retryable_network_error(error: Exception) -> bool:
    """Recognize httpx/httpx2 errors, including wrapped download failures.

    Do not retry authentication, missing revisions, SSL certificates, bad
    configuration, disk errors or tensor validation failures.
    """
    visited = set()
    current = error
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None:
            return status in (408, 429, 500, 502, 503, 504)
        if isinstance(current, (TimeoutError, ConnectionError, socket.gaierror)):
            return True
        if isinstance(current, OSError):
            # A disk failure is permanent even when an old network exception
            # remains in its context. Do not follow that misleading chain.
            return current.errno in (54, 60, 61, 104, 110, 111)
        module = type(current).__module__.split(".")[0]
        names = {cls.__name__ for cls in type(current).__mro__}
        if module in {"httpx", "httpx2", "httpcore", "httpcore2", "requests"}:
            if names & {"ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
                        "TimeoutException", "RemoteProtocolError", "ReadError", "WriteError"}:
                return True
            # Certificate verification failures must never be retried as a fix.
            if "ConnectError" in names:
                message = str(current).lower()
                return not any(marker in message for marker in
                               ("certificate_verify_failed", "certificate verify failed"))
        current = current.__cause__ or current.__context__
    return False


def network_call(label: str, operation, attempts: int, endpoint: str):
    """Retry transient network failures without clearing partial downloads."""
    for attempt in range(1, attempts + 1):
        print(f"[network {attempt}/{attempts}] {label} @ {endpoint}", flush=True)
        try:
            return operation()
        except Exception as error:
            if not retryable_network_error(error):
                raise
            if attempt == attempts:
                raise RuntimeError(
                    f"{label}: network failed after {attempts} attempts at {endpoint}. "
                    "Check connectivity and terminal proxy settings, or explicitly choose "
                    "an accessible HTTPS mirror with --endpoint (use --disable-xet for "
                    "ordinary HTTP downloads). Retry with --check-network first. "
                    "Partial downloads are retained; no model merge is claimed."
                ) from error
            delay = min(2 ** attempt, 16)
            print(f"  {type(error).__name__}; retrying in {delay}s. "
                  "Keeping existing downloads.", flush=True)
            time.sleep(delay)
    raise AssertionError("--network-attempts must be positive")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(tensor) -> str:
    import torch
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def selected_base_file(name: str) -> bool:
    return name in ROOT_FILES or any(name.startswith(f"{c}/") for c in COMPONENTS)


def verify_download(info, directory: Path, selected) -> None:
    for entry in info.siblings:
        if not selected(entry.rfilename):
            continue
        path = directory / entry.rfilename
        if not path.is_file() or path.stat().st_size != entry.size:
            raise RuntimeError(f"Missing or wrong-size download: {path}")
        if entry.lfs:
            print(f"  SHA256: {entry.rfilename}", flush=True)
            if file_hash(path) != entry.lfs.sha256:
                raise RuntimeError(f"Download checksum mismatch: {path}")


def verify_serialization(directory: Path, expected: dict[str, str]) -> None:
    from safetensors import safe_open
    seen = set()
    for shard in sorted(directory.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                if key in seen or key not in expected:
                    raise RuntimeError(f"Unexpected or duplicate exported tensor: {key}")
                if tensor_hash(stream.get_tensor(key)) != expected[key]:
                    raise RuntimeError(f"Serialized tensor changed: {key}")
                seen.add(key)
    if seen != set(expected):
        raise RuntimeError(f"Export missing {len(set(expected) - seen)} tensors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--work-dir", type=Path,
                        default=root / "models" / "qwen21_merge_sources")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output" / "ovb-e2b_cap10k_s42_10000_merged")
    parser.add_argument("--base-repo", default=BASE_REPO,
                        help="Diffusers-layout base pipeline repository")
    parser.add_argument("--base-revision", default=BASE_REVISION,
                        help="Override only with a known compatible official base commit")
    parser.add_argument("--lora-repo", default=DEFAULT_LORA_REPO)
    parser.add_argument("--lora-revision", default=DEFAULT_LORA_REVISION)
    parser.add_argument("--adapter-subdir", default=f"{DEFAULT_CHECKPOINT}/lora_adapter",
                        help="Directory in --lora-repo containing adapter_config.json "
                             "and adapter_model.safetensors")
    parser.add_argument("--metadata-path", default=f"{DEFAULT_CHECKPOINT}/metadata.json",
                        help="Optional metadata.json path in --lora-repo; "
                             "pass an empty string to skip")
    parser.add_argument("--adapter-sha256", default=None,
                        help="Optional pinned SHA256 of adapter_model.safetensors")
    parser.add_argument("--expected-r", type=int, default=32)
    parser.add_argument("--expected-alpha", type=int, default=64)
    parser.add_argument("--rslora", action=argparse.BooleanOptionalAction, default=True,
                        help="Expected adapter scaling: rsLoRA alpha/sqrt(r), "
                             "or standard alpha/r with --no-rslora")
    parser.add_argument("--num-blocks", type=int, default=32,
                        help="Transformer blocks; LoRA must hit every target on each")
    parser.add_argument("--validate-pipeline", action="store_true",
                        help="Also load the entire offline pipeline; recommend >=48 GiB RAM")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                        help="Hub HTTPS origin; a third-party mirror must be selected explicitly")
    parser.add_argument("--api-timeout", type=int, default=60,
                        help="Timeout in seconds per model_info request (default: 60)")
    parser.add_argument("--etag-timeout", type=int, default=120,
                        help="Timeout in seconds per download metadata request (default: 120)")
    parser.add_argument("--download-timeout", type=int, default=300,
                        help="HTTP file request timeout in seconds, not total download time")
    parser.add_argument("--network-attempts", type=int, default=3,
                        help="Maximum attempts per API or snapshot operation (default: 3)")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Concurrent file downloads (default: 2)")
    parser.add_argument("--disable-xet", action="store_true",
                        help="Disable Xet and use HTTP downloads; useful with mirrors")
    parser.add_argument("--check-network", action="store_true",
                        help="Check both pinned repositories only; no weights downloaded or merged")
    args = parser.parse_args()
    if any(getattr(args, key) < 1 for key in
           ("api_timeout", "etag_timeout", "download_timeout", "network_attempts", "max_workers")):
        parser.error("Timeouts, --network-attempts and --max-workers must be positive")
    try:
        endpoint = normalize_endpoint(args.endpoint)
    except ValueError as error:
        parser.error(str(error))
    # Hub constants are evaluated on import. Configure before importing torch,
    # diffusers or transformers, which may themselves import huggingface_hub.
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(args.etag_timeout)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(args.download_timeout)
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    output = args.output_dir.expanduser().resolve()
    work = args.work_dir.expanduser().resolve()
    base = work / f"base_{args.base_revision[:12]}"
    lora = work / f"lora_{args.lora_revision[:12]}"
    if output.exists() and not args.check_network:
        raise RuntimeError(f"Refusing to overwrite existing output: {output}")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >=3.10 is required")

    # Check actual class availability before downloading tens of gigabytes.
    import torch
    from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler,
                           QwenImage21Pipeline, QwenImage21Transformer2DModel)
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel, get_peft_model_state_dict
    from peft.tuners.lora import LoraLayer
    from safetensors import safe_open
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "diffusers", "transformers", "peft", "accelerate",
                 "huggingface_hub", "safetensors")}
    print(f"Environment: {versions}", flush=True)
    print("CPU merge, BF16 output. Training base commit was NOT recorded.", flush=True)
    print(f"Network: endpoint={endpoint}; API timeout={args.api_timeout}s; "
          f"ETag timeout={args.etag_timeout}s; file timeout={args.download_timeout}s; "
          f"attempts={args.network_attempts}; workers={args.max_workers}", flush=True)
    # Both audited repos are public. Never send a locally stored HF token to
    # a third-party mirror (or to the public Hub for this task).
    api = HfApi(endpoint=endpoint, token=False)
    base_info = network_call(
        "Read base model metadata", lambda: api.model_info(
            args.base_repo, revision=args.base_revision, files_metadata=True,
            timeout=args.api_timeout, token=False,
        ), args.network_attempts, endpoint,
    )
    lora_info = network_call(
        "Read LoRA metadata", lambda: api.model_info(
            args.lora_repo, revision=args.lora_revision, files_metadata=True,
            timeout=args.api_timeout, token=False,
        ), args.network_attempts, endpoint,
    )
    if base_info.sha != args.base_revision or lora_info.sha != args.lora_revision:
        raise RuntimeError("Hub/mirror returned a different commit than requested")
    base_names = [f.rfilename for f in base_info.siblings if selected_base_file(f.rfilename)]
    lora_names = [f"{args.adapter_subdir}/adapter_config.json",
                  f"{args.adapter_subdir}/adapter_model.safetensors"]
    if args.metadata_path:
        lora_names.append(args.metadata_path)
    if not set(lora_names).issubset({f.rfilename for f in lora_info.siblings}):
        raise RuntimeError("Hub/mirror is missing pinned LoRA files")
    if not set(ROOT_FILES).issubset(base_names) or any(
            not any(name.startswith(f"{c}/") for name in base_names) for c in COMPONENTS):
        raise RuntimeError("Hub/mirror is missing required base model components")
    entries = [(base, f) for f in base_info.siblings if f.rfilename in base_names]
    entries += [(lora, f) for f in lora_info.siblings if f.rfilename in lora_names]
    if any(f.size is None for _, f in entries):
        raise RuntimeError("Hub API did not supply file sizes; cannot preflight disk usage")
    if args.check_network:
        print(f"Network check OK: both pinned repositories expose required metadata. "
              f"Selected files: {sum(f.size for _, f in entries) / 1e9:.2f} GB. "
              "No weights downloaded, no file CDN connectivity or model loading tested.", flush=True)
        return
    missing_bytes = sum(f.size for directory, f in entries
                        if not (directory / f.rfilename).is_file()
                        or (directory / f.rfilename).stat().st_size != f.size)
    output_bytes = sum(f.size for f in base_info.siblings if f.rfilename in base_names)
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Also refuse a nested destination, which could accidentally copy itself.
    if output == work or work in output.parents or output in work.parents:
        raise RuntimeError("Output and download directories must not contain one another")
    reserve = 4 * 1024**3
    source_disk = shutil.disk_usage(work)
    output_disk = shutil.disk_usage(output.parent)
    if work.stat().st_dev == output.parent.stat().st_dev:
        required = missing_bytes + output_bytes + reserve
        if source_disk.free < required:
            raise RuntimeError(f"Need {required / 1024**3:.1f} GiB free; "
                               f"have {source_disk.free / 1024**3:.1f} GiB")
    elif source_disk.free < missing_bytes + reserve or output_disk.free < output_bytes + reserve:
        raise RuntimeError("Insufficient disk space on download or output filesystem")
    print(f"[1/6] Download up to {missing_bytes / 1e9:.2f} GB; "
          f"standalone output about {output_bytes / 1e9:.2f} GB", flush=True)
    network_call(
        "Download base model", lambda: snapshot_download(
            args.base_repo, revision=base_info.sha, local_dir=base,
            allow_patterns=base_names, max_workers=args.max_workers,
            endpoint=endpoint, token=False, etag_timeout=args.etag_timeout,
        ), args.network_attempts, endpoint,
    )
    network_call(
        "Download LoRA", lambda: snapshot_download(
            args.lora_repo, revision=lora_info.sha, local_dir=lora,
            allow_patterns=lora_names, max_workers=args.max_workers,
            endpoint=endpoint, token=False, etag_timeout=args.etag_timeout,
        ), args.network_attempts, endpoint,
    )
    verify_download(base_info, base, selected_base_file)
    verify_download(lora_info, lora, lambda name: name in lora_names)
    adapter = lora / args.adapter_subdir
    if args.adapter_sha256 and file_hash(adapter / "adapter_model.safetensors") != args.adapter_sha256:
        raise RuntimeError("Pinned adapter SHA256 mismatch")
    config = read_json(adapter / "adapter_config.json")
    metadata = read_json(lora / args.metadata_path) if args.metadata_path else {}
    if (config.get("peft_type") != "LORA" or config.get("r") != args.expected_r
            or config.get("lora_alpha") != args.expected_alpha
            or bool(config.get("use_rslora")) != args.rslora
            or config.get("use_dora") or config.get("bias") != "none"
            or config.get("lora_bias") or config.get("modules_to_save")
            or config.get("rank_pattern") or config.get("alpha_pattern")
            or set(config.get("target_modules", [])) != TARGETS):
        raise RuntimeError("Adapter does not match the expected OVB LoRA configuration")
    if metadata and metadata.get("model_name") not in (None, "qwen-image-2.1"):
        raise RuntimeError(f"Adapter metadata names a different base: "
                           f"{metadata.get('model_name')}")
    import math
    expected_scaling = args.expected_alpha / (math.sqrt(args.expected_r) if args.rslora
                                              else args.expected_r)
    if read_json(base / "model_index.json").get("_class_name") != "QwenImage21Pipeline":
        raise RuntimeError("Wrong base pipeline class")

    print("[2/6] Load transformer only on CPU and validate every adapter tensor", flush=True)
    transformer = QwenImage21Transformer2DModel.from_pretrained(
        base, subfolder="transformer", torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    )
    transformer.eval()
    pair_modules = set()
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as stream:
        keys = set(stream.keys())
        for key in keys:
            prefix = "base_model.model."
            if not key.startswith(prefix) or not key.endswith((".lora_A.weight", ".lora_B.weight")):
                raise RuntimeError(f"Unsupported adapter key: {key}")
            module_name = key[len(prefix):].rsplit(".lora_", 1)[0]
            module = transformer.get_submodule(module_name)
            if not isinstance(module, torch.nn.Linear):
                raise RuntimeError(f"Adapter target is not a Linear: {module_name}")
            suffix = "A" if key.endswith(".lora_A.weight") else "B"
            expected_shape = ((args.expected_r, module.in_features) if suffix == "A"
                              else (module.out_features, args.expected_r))
            if tuple(stream.get_slice(key).get_shape()) != expected_shape:
                raise RuntimeError(f"Adapter/base shape mismatch: {key}")
            other = key.replace(f".lora_{suffix}.weight",
                                f".lora_{'B' if suffix == 'A' else 'A'}.weight")
            if other not in keys:
                raise RuntimeError(f"Missing LoRA matrix pair: {key}")
            pair_modules.add(module_name)
        if len(pair_modules) != args.num_blocks * len(TARGETS):
            raise RuntimeError(f"Expected 224 LoRA targets; got {len(pair_modules)}")
    del module

    name = "ovb"
    wrapped = PeftModel.from_pretrained(transformer, adapter, adapter_name=name,
                                        is_trainable=False, local_files_only=True)
    loaded = get_peft_model_state_dict(wrapped, adapter_name=name)
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as stream:
        if set(loaded) != set(stream.keys()):
            raise RuntimeError("PEFT loaded a different adapter key set")
        for key, value in loaded.items():
            if not torch.equal(value.detach().cpu().float(), stream.get_tensor(key).float()):
                raise RuntimeError(f"PEFT failed to load adapter tensor exactly: {key}")
    del loaded, value, transformer
    model = wrapped.base_model.model
    expected_merged = {}
    for module_name in sorted(pair_modules):
        layer = model.get_submodule(module_name)
        if not isinstance(layer, LoraLayer):
            raise RuntimeError(f"Missing PEFT layer: {module_name}")
        if abs(layer.scaling[name] - expected_scaling) > 1e-6:
            raise RuntimeError(f"Wrong LoRA scaling in {module_name}: "
                               f"{layer.scaling[name]} != {expected_scaling}")
        with torch.no_grad():
            expected_weight = layer.get_base_layer().weight.detach().clone()
            # Match PEFT safe_merge's BF16 rounding order exactly.
            expected_weight += layer.get_delta_weight(name).to(expected_weight.dtype)
            if not torch.isfinite(expected_weight).all():
                raise RuntimeError(f"Nonfinite merged weight: {module_name}")
            expected_merged[f"{module_name}.weight"] = tensor_hash(expected_weight)
        del expected_weight, layer

    print("[3/6] PEFT safe_merge, unload adapters, verify all 224 updated matrices", flush=True)
    merged = wrapped.merge_and_unload(safe_merge=True, adapter_names=[name])
    del wrapped, model
    merged.eval()
    for key, expected_digest in expected_merged.items():
        if tensor_hash(merged.get_parameter(key)) != expected_digest:
            raise RuntimeError(f"Merge formula mismatch: {key}")
    state = merged.state_dict()
    if any("lora_" in key or "base_layer" in key for key in state):
        raise RuntimeError("Export still contains adapter wrapper keys")
    expected_export = {}
    for key, value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"Nonfinite model tensor: {key}")
        expected_export[key] = tensor_hash(value)
    del state, value
    output.mkdir(exist_ok=False)
    (output / "MERGE_INCOMPLETE").write_text(
        "No validated model yet. Inspect the failing command; do not use this directory.\n",
        encoding="utf-8",
    )
    print("[4/6] Export BF16 safetensors and copy all other pipeline components", flush=True)
    merged.save_pretrained(output / "transformer", safe_serialization=True,
                           max_shard_size="5GB")
    del merged
    gc.collect()
    for component in COMPONENTS:
        if component != "transformer":
            shutil.copytree(base / component, output / component)
    for filename in ROOT_FILES:
        if (base / filename).is_file():
            shutil.copy2(base / filename, output / filename)
    verify_serialization(output / "transformer", expected_export)
    for entry in base_info.siblings:
        if selected_base_file(entry.rfilename) and not entry.rfilename.startswith("transformer/"):
            if file_hash(base / entry.rfilename) != file_hash(output / entry.rfilename):
                raise RuntimeError(f"Copied component checksum mismatch: {entry.rfilename}")

    print("[5/6] Reload each component offline, one at a time (no inference)", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for cls, component in ((QwenImage21Transformer2DModel, "transformer"),
                           (Qwen3VLForConditionalGeneration, "text_encoder"),
                           (AutoencoderKLQwenImage21, "vae")):
        loaded_component = cls.from_pretrained(
            output / component, torch_dtype=torch.bfloat16,
            local_files_only=True, low_cpu_mem_usage=True,
        )
        del loaded_component
        gc.collect()
    processor = Qwen3VLProcessor.from_pretrained(output / "processor", local_files_only=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        output / "scheduler", local_files_only=True,
    )
    del processor, scheduler
    if args.validate_pipeline:
        print("Loading complete offline pipeline; this needs more RAM", flush=True)
        pipeline = QwenImage21Pipeline.from_pretrained(
            output, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        del pipeline
        gc.collect()
    print("[6/6] Write provenance and successful validation status", flush=True)
    provenance = {
        "base_repo": args.base_repo, "base_revision": base_info.sha,
        "training_base_revision_known": False,
        "lora_repo": args.lora_repo, "lora_revision": lora_info.sha,
        "adapter_subdir": args.adapter_subdir,
        "adapter_sha256": file_hash(adapter / "adapter_model.safetensors"),
        "training_metadata": metadata, "adapter_config": config,
        "dtype": "bfloat16", "merge_method": "PEFT merge_and_unload(safe_merge=True)",
        "merged_modules": len(pair_modules), "software_versions": versions,
        "download_endpoint": endpoint,
        "validation": {"download_sha256": True, "adapter_loaded_exactly": True,
                       "all_lora_merge_formulas": True, "all_exported_tensor_hashes": True,
                       "copied_component_hashes": True, "offline_component_loading": True,
                       "offline_full_pipeline_loading": args.validate_pipeline,
                       "image_inference": False},
    }
    (output / "merge_metadata.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (output / "MERGE_INCOMPLETE").unlink()
    (output / "MERGE_VALIDATED").write_text(
        "Weight merge and offline component loading verified. Image inference not tested.\n",
        encoding="utf-8",
    )
    total = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    print(f"Done: {output}\nStandalone size: {total / 1e9:.2f} GB. "
          "No separate LoRA is needed. Image inference was NOT tested.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}\nNo successful merge is claimed. Existing files are retained.",
              file=sys.stderr, flush=True)
        raise
