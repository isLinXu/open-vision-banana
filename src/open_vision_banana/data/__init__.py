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
from .generation import GenerationDataset, build_generation_dataset
from .generation_builder import (
    build_flux_generation_dataset,
    build_synthetic_generation_dataset,
    load_prompts_file,
)
from .mixed import MixedVisionDataset
from .split import split_dataset, split_vision_datasets
from .validation import ValidationReport, validate_vision_dataset

__all__ = [
    "BaseVisionDataset",
    "HypersimDepthDataset",
    "ScanNetNormalDataset",
    "CityscapesSegDataset",
    "SACoInstanceDataset",
    "RefCOCOgDataset",
    "MixedVisionDataset",
    "GenerationDataset",
    "build_dataset",
    "build_generation_dataset",
    "build_synthetic_generation_dataset",
    "build_flux_generation_dataset",
    "load_prompts_file",
    "split_dataset",
    "split_vision_datasets",
    "ValidationReport",
    "validate_vision_dataset",
]
