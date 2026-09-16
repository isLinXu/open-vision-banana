#!/usr/bin/env python3
"""Smoke test for TarTargetDataset - verify data loading from tar volumes."""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.data.tar_dataset import TarTargetDataset

TAR_ROOT = Path("data/snvc50m")
INDEX_DIR = PROJECT_ROOT / "output" / "snvc50m_manifest" / "task_indexes"

def test_task(task: str, num_samples: int = 3):
    idx_path = INDEX_DIR / f"tar_index_{task}.json"
    if not idx_path.is_file():
        print(f"  SKIP: {idx_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"Testing task: {task}")
    print(f"Index: {idx_path}")
    print(f"Tar root: {TAR_ROOT}")
    print(f"{'='*60}")

    ds = TarTargetDataset(
        tar_index_path=idx_path,
        tar_root=TAR_ROOT,
        task=task,
        resolution=512,
        max_samples=100,  # Only first 100 (sorted by offset = beginning of stream)
    )
    print(f"Dataset size: {len(ds)}")
    print(f"First file offset: {ds._file_list[0][1] / 1e9:.1f} GB")

    # Test loading a few samples (first ones in stream = fast access)
    for i in range(num_samples):
        try:
            import time
            t0 = time.time()
            sample = ds[i]
            elapsed = time.time() - t0
            print(f"  Sample {i} ({elapsed:.1f}s):")
            print(f"    prompt: {sample['prompt'][:60]}...")
            print(f"    target_rgb: shape={sample['target_rgb'].shape}, "
                  f"min={sample['target_rgb'].min():.3f}, max={sample['target_rgb'].max():.3f}")
            print(f"    task: {sample['task']}")
            print(f"    flow_source: {sample['flow_source']}")
            print(f"    has 'image' key: {'image' in sample}")
        except Exception as e:
            print(f"  Sample {i}: FAILED - {e}")
            import traceback
            traceback.print_exc()

    ds.close()
    print(f"  OK!")

if __name__ == "__main__":
    for task in ["depth", "normal", "seg"]:
        test_task(task)
    print("\nSmoke test complete!")
