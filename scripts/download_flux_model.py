#!/usr/bin/env python3
"""Pre-download FLUX.1-schnell model weights to HF cache.

This avoids download timeouts during training startup.
"""
import os
import sys
from pathlib import Path

os.environ["HF_TOKEN"] = "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"
os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

from huggingface_hub import snapshot_download

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
