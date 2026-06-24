"""v0.4.0 SegmentationDecoder optimization regression tests.

Covers two performance/correctness improvements:
- S4 tiny-component pre-filter: components with < 9 pixels cannot survive a
  3x3 erosion and are dropped up front (exact, prevents conv-cost blowup on
  noisy / low-resolution inputs).
- S5 vectorized connected-component merging: the Python greedy-merge loop is
  replaced by a tensor Union-Find that computes the same final partition.
"""

from __future__ import annotations

import time

import torch

from open_vision_banana.core.segmentation import SegmentationDecoder


def _two_block_image(size: int = 64) -> torch.Tensor:
    """Synthetic image with two well-separated solid color blocks."""
    img = torch.zeros(1, 3, size, size)
    h = size // 2
    img[0, 0, :h, :h] = 1.0          # red block top-left
    img[0, 1, h:, h:] = 1.0          # green block bottom-right
    return img


class TestS4TinyComponentPrefilter:
    def test_structured_image_yields_correct_instances(self):
        decoder = SegmentationDecoder()
        instances = decoder.decode_instance(_two_block_image(64))
        assert len(instances) == 1
        # Two distinct color blocks => two instances.
        assert len(instances[0]) == 2

    def test_noisy_input_is_fast(self):
        """Random 64x64 used to take >6s due to thousands of singleton
        components hitting the S4 multi-hot conv. With the pre-filter it
        should be well under a generous bound even under load."""
        decoder = SegmentationDecoder()
        rgb = torch.rand(1, 3, 64, 64)
        decoder.decode_instance(rgb)  # warmup
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            decoder.decode_instance(rgb)
            timings.append(time.perf_counter() - start)
        timings.sort()
        median = timings[len(timings) // 2]
        assert median < 2.0, f"S4 prefilter regressed: {median:.3f}s median"

    def test_tiny_components_removed(self):
        """A lone single-pixel speck must not survive S4 erosion."""
        decoder = SegmentationDecoder()
        img = torch.zeros(1, 3, 32, 32)
        # One large block + one isolated single pixel of a different color.
        img[0, 0, :16, :16] = 1.0
        img[0, 2, 30, 30] = 1.0
        instances = decoder.decode_instance(img)
        # The single-pixel speck is eroded away; only the block remains.
        assert len(instances[0]) == 1


class TestS5VectorizedMerging:
    def test_merging_preserves_batch_shape(self):
        decoder = SegmentationDecoder()
        img = _two_block_image(64).repeat(2, 1, 1, 1)
        instances = decoder.decode_instance(img)
        assert len(instances) == 2

    def test_close_same_color_blocks_merge(self):
        """Two same-color blocks within the bbox-expansion radius should be
        merged into a single instance by S5."""
        decoder = SegmentationDecoder(bbox_expansion_factor=50.0)
        img = torch.zeros(1, 3, 64, 64)
        # Two same-color (red) blocks close together.
        img[0, 0, 10:20, 10:20] = 1.0
        img[0, 0, 10:20, 24:34] = 1.0
        instances = decoder.decode_instance(img, adaptive_color_tolerance=False)
        # With a large expansion factor the two same-color blocks merge.
        assert len(instances[0]) == 1

    def test_distinct_colors_not_merged(self):
        decoder = SegmentationDecoder()
        img = _two_block_image(64)
        instances = decoder.decode_instance(img, adaptive_color_tolerance=False)
        assert len(instances[0]) == 2
