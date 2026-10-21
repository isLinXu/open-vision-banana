"""Dependency-free RES encoding names and expression-conditioned prompt."""

from __future__ import annotations

import re

BINARY_REFERRING_ENCODING = "binary_white_black_v1"
LEGACY_REFERRING_ENCODING = "legacy_color_distance_v1"

# Paper (arXiv:2604.20329) Fig. 4 (c)/(d) template, with the white/black colours
# the training manifests use. Training data, inference and evaluation all build
# the referring prompt from this one string so the model never sees a different
# instruction at test time than it did during instruction tuning.
BINARY_REFERRING_TEMPLATE = (
    "This image shows segmentation masks from the given image. "
    "The background is black color. "
    "The {expression} is represented by a solid white."
)

# Older RES manifests were built with this wording; ``is_binary_referring_prompt``
# keeps them loadable without claiming they match the paper template.
_LEGACY_BINARY_REFERRING_TEMPLATE = (
    "Segment {expression} in solid white. Render all other pixels black."
)

_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def normalize_referring_expression(expression: str) -> str:
    """Strip whitespace and a leading article so templates can prefix "The"."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("A nonempty expression is required for referring segmentation")
    return _LEADING_ARTICLE_RE.sub("", expression.strip(), count=1)


def build_binary_referring_prompt(expression: str) -> str:
    """Keep data preparation and inference on the same white/black contract."""
    return BINARY_REFERRING_TEMPLATE.format(expression=normalize_referring_expression(expression))


def is_binary_referring_prompt(prompt: str, expression: str) -> bool:
    """True for the canonical prompt or the pre-template wording of ``expression``."""
    if prompt == build_binary_referring_prompt(expression):
        return True
    return prompt == _LEGACY_BINARY_REFERRING_TEMPLATE.format(expression=expression.strip())
