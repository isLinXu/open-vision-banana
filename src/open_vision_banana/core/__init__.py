"""Core algorithms — no external dependencies beyond PyTorch/NumPy.

Provides reversible RGB mapping for all vision tasks:
- DepthRGBMapper: power transform + spectral colormap
- NormalRGBMapper: direct linear mapping
- SegmentationDecoder: 5-stage instance clustering
- RGBMapper: unified dispatcher
"""

from .depth_mapper import DepthRGBMapper
from .normal_mapper import NormalRGBMapper
from .rgb_mapper import RGBMapper
from .segmentation import SegmentationDecoder

__all__ = [
    "DepthRGBMapper",
    "NormalRGBMapper",
    "SegmentationDecoder",
    "RGBMapper",
]
