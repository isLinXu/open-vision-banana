"""Data sub-package — dataset loaders for vision task instruction tuning.

Supports the datasets referenced in the paper:
- Hypersim (depth + normal)
- Virtual KITTI 2 (depth)
- ScanNet (normal)
- Cityscapes (semantic segmentation)
- SA-Co / Gold (instance segmentation)
- RefCOCOg (referring expression segmentation)

Extended benchmarks (15+ datasets):
- NYU Depth v2, DIODE, KITTI, ETH3D, TUM RGB-D, Virtual KITTI (depth)
- OASIS (normal)
- ADE20K, COCO-Stuff, Mapillary Vistas (semantic)
- COCO (detection/instance)
- Sintel, FlyingThings3D (flow)
- RefCOCO / RefCOCO+ (referring)
- nuScenes (multi-task autonomous driving)
"""

from .datasets import (
    BaseVisionDataset,
    COCODetectionDataset,
    COCOStuffDataset,
    CityscapesSegDataset,
    DIODEDataset,
    DIODENormalDataset,
    ETH3DDataset,
    FlyingThings3DDataset,
    HypersimDepthDataset,
    IBimsDepthDataset,
    KITTIDataset,
    MapillaryVistasDataset,
    NYUDepthDataset,
    NYUNormalDataset,
    OASISNormalDataset,
    RefCOCODataset,
    RefCOCOgDataset,
    SACoInstanceDataset,
    SintelFlowDataset,
    ScanNetNormalDataset,
    TUMRGBDDataset,
    VirtualKITTIDataset,
    VirtualKITTINormalDataset,
    nuScenesDataset,
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
    "RefCOCODataset",
    "NYUDepthDataset",
    "NYUNormalDataset",
    "IBimsDepthDataset",
    "DIODEDataset",
    "DIODENormalDataset",
    "KITTIDataset",
    "ETH3DDataset",
    "OASISNormalDataset",
    "COCODetectionDataset",
    "COCOStuffDataset",
    "VirtualKITTIDataset",
    "VirtualKITTINormalDataset",
    "TUMRGBDDataset",
    "FlyingThings3DDataset",
    "SintelFlowDataset",
    "MapillaryVistasDataset",
    "nuScenesDataset",
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
