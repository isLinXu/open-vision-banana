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
    CityscapesSegDataset,
    COCODetectionDataset,
    COCOStuffDataset,
    DIODEDataset,
    DIODENormalDataset,
    ETH3DDataset,
    FlyingThings3DDataset,
    HypersimDepthDataset,
    IBimsDepthDataset,
    KITTIDataset,
    MapillaryVistasDataset,
    NuScenesDepthDataset,
    NYUDepthDataset,
    NYUNormalDataset,
    OASISNormalDataset,
    RefCOCODataset,
    RefCOCOgDataset,
    SACoInstanceDataset,
    ScanNetNormalDataset,
    SintelFlowDataset,
    TUMRGBDDataset,
    V3FlatDataset,
    VirtualKITTIDataset,
    VirtualKITTINormalDataset,
    build_dataset,
    build_v3_datasets,
    load_v3_train_val_split,
    nuScenesDataset,
)
from .generation import GenerationDataset, build_generation_dataset
from .generation_builder import (
    build_flux_generation_dataset,
    build_synthetic_generation_dataset,
    load_prompts_file,
)
from .manifest import fingerprint_records, read_jsonl, resolve_manifest_path
from .mixed import HomogeneousMixedBatchSampler, MixedVisionDataset
from .paper_eval_datasets import GenAIBenchDataset, ImgEditDataset, ReasonSegDataset
from .res_dataset import RESManifestDataset
from .split import split_dataset, split_vision_datasets
from .tar_dataset import SequentialMultiTaskDataset, TarArchiveDataset, TarTargetDataset
from .validation import ValidationReport, validate_vision_dataset

__all__ = [
    "BaseVisionDataset",
    "HypersimDepthDataset",
    "ScanNetNormalDataset",
    "CityscapesSegDataset",
    "SACoInstanceDataset",
    "RefCOCOgDataset",
    "RESManifestDataset",
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
    "NuScenesDepthDataset",
    "nuScenesDataset",
    "ReasonSegDataset",
    "GenAIBenchDataset",
    "ImgEditDataset",
    "MixedVisionDataset",
    "HomogeneousMixedBatchSampler",
    "fingerprint_records",
    "read_jsonl",
    "resolve_manifest_path",
    "GenerationDataset",
    "build_dataset",
    "V3FlatDataset",
    "build_v3_datasets",
    "load_v3_train_val_split",
    "build_generation_dataset",
    "build_synthetic_generation_dataset",
    "build_flux_generation_dataset",
    "load_prompts_file",
    "split_dataset",
    "split_vision_datasets",
    "TarArchiveDataset",
    "TarTargetDataset",
    "SequentialMultiTaskDataset",
    "StreamOrderedMultiTaskDataset",
    "ValidationReport",
    "validate_vision_dataset",
]
