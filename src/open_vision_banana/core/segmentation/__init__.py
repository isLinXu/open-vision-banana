"""Instance/semantic segmentation decoding (P2-1 package split)."""

from .decoder import (
    PAPER_INSTANCE_DECODING_CONFIG,
    InstanceDecodingConfig,
    SegmentationDecoder,
)

__all__ = [
    "InstanceDecodingConfig",
    "PAPER_INSTANCE_DECODING_CONFIG",
    "SegmentationDecoder",
]
