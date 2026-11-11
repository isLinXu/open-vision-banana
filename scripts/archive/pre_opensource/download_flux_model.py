#!/usr/bin/env python3
"""Pre-download FLUX.1-schnell model weights to HF cache.

This avoids download timeouts during training startup.

Usage:
    # use the logged-in HuggingFace CLI token (recommended)
    python scripts/download_flux_model.py

    # pass the token through an environment variable
    HF_TOKEN=hf_xxx python scripts/download_flux_model.py
"""
import os
import sys
from pathlib import Path

# read the token from the environment first, otherwise try the HuggingFace CLI login cache
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not token:
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
if token:
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

from huggingface_hub import snapshot_download  # noqa: E402

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

def main():
    print(f"Pre-downloading {MODEL_ID}...")
    print(f"Cache dir: {CACHE_DIR}")
    print("This may take 10-30 minutes depending on network speed.\n")

    try:
        path = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(CACHE_DIR),
            resume_download=True,
            local_files_only=False,
        )
        print(f"\n✅ Download complete: {path}")
        print(f"Total cache size: {sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file()) / 1e9:.1f} GB")
        return 0
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
