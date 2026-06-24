"""Core algorithms — no external dependencies beyond PyTorch/NumPy.

Provides reversible RGB mapping for all vision tasks:
- DepthRGBMapper: power transform + cube-edge colormap
- NormalRGBMapper: direct linear mapping
- SegmentationDecoder: 5-stage instance clustering
- UnionFind: disjoint set union with path compression + union by rank
- RGBMapper: unified dispatcher
- AdaptiveDepthRGBMapper: adaptive precision depth mapping
"""

from .adaptive_depth_mapper import AdaptiveDepthRGBMapper
from .depth_mapper import DepthRGBMapper
from .normal_mapper import NormalRGBMapper
from .rgb_mapper import RGBMapper
from .segmentation import SegmentationDecoder
from .union_find import UnionFind

__all__ = [
    "DepthRGBMapper",
    "NormalRGBMapper",
    "SegmentationDecoder",
    "UnionFind",
    "RGBMapper",
    "AdaptiveDepthRGBMapper",
]
