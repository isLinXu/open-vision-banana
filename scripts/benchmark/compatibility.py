"""Explicit, limited compatibility for the laptop component benchmark.

This does not change the project's Python >= 3.10 / PyTorch >= 2.0 policy.
Only standalone Tiny and torchvision can opt into the older runtime.
"""

import builtins
import hashlib
import inspect
import itertools
import re
import sys
from pathlib import Path

import torch


def runtime_status():
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "project_runtime_supported": sys.version_info >= (3, 10)
        and int(re.match(r"\d+", torch.__version__).group()) >= 2,
    }


def require_runtime(allow_legacy=False, full_pipeline=False):
    status = runtime_status()
    if not status["project_runtime_supported"]:
        if full_pipeline or not allow_legacy:
            raise RuntimeError(
                "Project runtime requires Python >= 3.10 and PyTorch >= 2.0. "
                "--allow-legacy only enables standalone Tiny/torchvision component tests; "
                "it does not enable VisionBanana diffusion pipelines."
            )
        status["compatibility_scope"] = "explicit standalone component compatibility"
    else:
        status["compatibility_scope"] = "supported project runtime"
    return status


def strict_zip(*iterables, **kwargs):
    """Python 3.7-compatible zip(strict=True), without changing builtins."""
    strict = kwargs.pop("strict", False)
    if kwargs:
        raise TypeError("Unexpected zip keywords: %s" % sorted(kwargs))
    if not strict:
        return builtins.zip(*iterables)
    sentinel = object()

    def checked():
        for row in itertools.zip_longest(*iterables, fillvalue=sentinel):
            if any(item is sentinel for item in row):
                raise ValueError("zip arguments have different lengths")
            yield row

    return checked()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tensor_state(path, allow_legacy=False, trusted_checkpoint=False):
    """Load tensor-only weights, recording any explicitly allowed unsafe fallback.

    Tiny.save historically writes torch.save bytes to a .safetensors filename;
    inspect the container rather than trusting its suffix. Modern pickle loads
    always use weights_only=True. Older torch cannot restrict its unpickler, so
    a separate trust declaration is required in addition to allow_legacy.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Checkpoint file does not exist: %s" % path)
    with path.open("rb") as handle:
        first = handle.read(9)
    tensor_container = (
        len(first) == 9
        and 0 < int.from_bytes(first[:8], "little") < 100 * 1024 * 1024
        and first[8:] == b"{"
    )
    if tensor_container:
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
        method = "safetensors"
    elif "weights_only" in inspect.signature(torch.load).parameters:
        # New torch may dispatch .safetensors paths by suffix. Pass a file object
        # after container detection so historical Tiny.save files remain usable.
        with path.open("rb") as handle:
            state = torch.load(handle, map_location="cpu", weights_only=True)
        method = "torch.load(weights_only=True)"
    elif allow_legacy and trusted_checkpoint:
        with path.open("rb") as handle:
            state = torch.load(handle, map_location="cpu")
        method = "explicitly trusted legacy torch.load (unrestricted pickle)"
    else:
        raise RuntimeError(
            "This PyTorch cannot safely restrict pickle loading. Use a genuine "
            "safetensors checkpoint or modern torch. A known trusted local file "
            "requires BOTH --allow-legacy and options.trusted_checkpoint=true."
        )
    # Permit conventional wrappers, but never silently select EMA versus online.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("Expected a non-empty tensor state_dict, not a training checkpoint")
    return state, {"path": str(path.resolve()), "sha256": sha256_file(path), "load_method": method}
