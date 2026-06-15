"""Data sub-package — dataset loaders for vision task instruction tuning.

Supports the datasets referenced in the paper:
- Hypersim (depth + normal)
- Virtual KITTI 2 (depth)
- ScanNet (normal)
- Cityscapes (semantic segmentation)
- SA-Co / Gold (instance segmentation)
- RefCOCOg (referring expression segmentation)
"""

from .datasets import (
    BaseVisionDataset,
    CityscapesSegDataset,
    HypersimDepthDataset,
    RefCOCOgDataset,
    SACoInstanceDataset,
    ScanNetNormalDataset,
    build_dataset,
)
from .mixed import MixedVisionDataset

__all__ = [
    "BaseVisionDataset",
    "HypersimDepthDataset",
    "ScanNetNormalDataset",
    "CityscapesSegDataset",
    "SACoInstanceDataset",
    "RefCOCOgDataset",
    "MixedVisionDataset",
    "build_dataset",
]
