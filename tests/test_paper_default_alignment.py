"""Paper-default alignment regression tests (P0-5).

The paper's instance decoding protocol (Appendix A) — seed-based grouping
with a fixed color tolerance τ=14 — must be the *default* on every public
construction surface. Adjacent grouping and adaptive tolerance remain
available only as explicit opt-ins.
"""

import inspect

from open_vision_banana.config import SegmentationConfig
from open_vision_banana.core.rgb_mapper import RGBMapper
from open_vision_banana.core.segmentation import (
    PAPER_INSTANCE_DECODING_CONFIG,
    InstanceDecodingConfig,
    SegmentationDecoder,
)
from open_vision_banana.inference.decoder import VisualizationDecoder
from open_vision_banana.inference.pipeline import VisionBanana


def _signature_defaults(func, names):
    sig = inspect.signature(func)
    return {name: sig.parameters[name].default for name in names}


def test_instance_decoding_config_defaults_match_paper():
    default = InstanceDecodingConfig()
    assert default.grouping_mode == PAPER_INSTANCE_DECODING_CONFIG.grouping_mode == "seed"
    assert (
        default.adaptive_color_tolerance
        is PAPER_INSTANCE_DECODING_CONFIG.adaptive_color_tolerance
        is False
    )
    # All scalar hyperparameters must match Appendix A exactly.
    assert default.color_tolerance == 14.0
    assert default.min_area_fraction == 2e-4
    assert default.erosion_threshold == 0.1
    assert default.bbox_expansion_factor == 5.0


def test_segmentation_decoder_default_is_paper_protocol():
    decoder = SegmentationDecoder()
    assert decoder.instance_config == PAPER_INSTANCE_DECODING_CONFIG


def test_segmentation_config_defaults_match_paper():
    config = SegmentationConfig()
    assert config.grouping_mode == "seed"
    assert config.adaptive_color_tolerance is False


def test_rgb_mapper_default_is_paper_protocol():
    mapper = RGBMapper()
    assert mapper.seg_decoder.instance_config == PAPER_INSTANCE_DECODING_CONFIG


def test_pipeline_and_decoder_signature_defaults_match_paper():
    for func in (
        VisionBanana.__init__,
        VisionBanana.from_pretrained,
        VisualizationDecoder.__init__,
    ):
        defaults = _signature_defaults(
            func, ("instance_grouping_mode", "instance_adaptive_color_tolerance")
        )
        assert defaults["instance_grouping_mode"] == "seed", func
        assert defaults["instance_adaptive_color_tolerance"] is False, func


def test_paper_reproduction_classmethod_still_matches():
    decoder = SegmentationDecoder.paper_reproduction()
    assert decoder.instance_config == PAPER_INSTANCE_DECODING_CONFIG
