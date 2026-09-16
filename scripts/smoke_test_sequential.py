#!/usr/bin/env python3
"""Smoke test for sequential tar reading (no indexed_gzip needed)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from open_vision_banana.data.tar_dataset import TarTargetDataset, SequentialMultiTaskDataset

# Test with a small number of samples per task
MAX_SAMPLES = 5
TAR_ROOT = Path("data/snvc50m")
TASK_INDEX_DIR = Path("output/snvc50m_manifest/task_indexes")

print("=" * 60)
print("Smoke test: Sequential tar reading")
print("=" * 60)

# Create task datasets
task_datasets = {}
for task in ["depth", "normal", "seg"]:
    idx_path = TASK_INDEX_DIR / f"tar_index_{task}.json"
    if not idx_path.exists():
        print(f"  Skipping {task}: index not found")
        continue
    ds = TarTargetDataset(
        tar_index_path=idx_path,
        tar_root=TAR_ROOT,
        task=task,
        resolution=512,
        max_samples=MAX_SAMPLES,
    )
    task_datasets[task] = ds
    print(f"  {task}: {len(ds)} samples")

# Create sequential multi-task dataset
multi_ds = SequentialMultiTaskDataset(
    task_datasets=task_datasets,
    task_order=list(task_datasets.keys()),
)
print(f"\nTotal samples: {len(multi_ds)}")

# Read samples sequentially
print("\nReading samples sequentially...")
t0 = time.time()
for i in range(min(10, len(multi_ds))):
    t1 = time.time()
    sample = multi_ds[i]
    t2 = time.time()
    print(f"  [{i}] task={sample['task']}, target={sample['target_rgb'].shape}, "
          f"flow={sample['flow_source']}, time={t2-t1:.2f}s")

total_time = time.time() - t0
print(f"\nRead 10 samples in {total_time:.1f}s ({10/total_time:.2f} samples/s)")
print("SUCCESS: Sequential reading works!")

# Clean up
multi_ds.close()
