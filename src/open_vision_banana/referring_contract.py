"""Dependency-free RES encoding names and expression-conditioned prompt."""

from __future__ import annotations

import re

BINARY_REFERRING_ENCODING = "binary_white_black_v1"
LEGACY_REFERRING_ENCODING = "legacy_color_distance_v1"

# Historical Fig. 4 wording retained for explicit checkpoint comparisons.
FIG4_REFERRING_TEMPLATE = (
    "This image shows segmentation masks from the given image. "
    "The background is black color. "
    "The {expression} is represented by a solid white."
)

# Older RES manifests were built with this wording; ``is_binary_referring_prompt``
# keeps them loadable without claiming they match the paper template.
_LEGACY_BINARY_REFERRING_TEMPLATE = (
    "Segment {expression} in solid white. Render all other pixels black."
)

DEFAULT_REFERRING_PROMPT_VERSION = "query_v2"
REFERRING_PROMPT_VERSIONS = ("query_v2", "fig4_v1", "segment_v0")
BINARY_REFERRING_TEMPLATE = (
    "Identify the object or region referred to by the following query in the image.\n"
    "Query: {query}\n"
    "Render its segmentation mask in solid white and all other pixels in black."
)

_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def normalize_referring_expression(expression: str) -> str:
    """Strip whitespace and a leading article so templates can prefix "The"."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("A nonempty expression is required for referring segmentation")
    return _LEADING_ARTICLE_RE.sub("", expression.strip(), count=1)


def build_binary_referring_prompt(
    expression: str,
    prompt_version: str = DEFAULT_REFERRING_PROMPT_VERSION,
) -> str:
    """Version text independently of the unchanged binary mask encoding."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("A nonempty expression is required for referring segmentation")
    if prompt_version == "query_v2":
        return BINARY_REFERRING_TEMPLATE.format(query=expression.strip())
    if prompt_version == "fig4_v1":
        return FIG4_REFERRING_TEMPLATE.format(expression=normalize_referring_expression(expression))
    if prompt_version == "segment_v0":
        return _LEGACY_BINARY_REFERRING_TEMPLATE.format(expression=expression.strip())
    raise ValueError(f"Unknown referring prompt version: {prompt_version!r}")


def is_binary_referring_prompt(prompt: str, expression: str) -> bool:
    """Accept historical serialized prompts without changing their pixel encoding."""
    return any(
        prompt == build_binary_referring_prompt(expression, version)
        for version in REFERRING_PROMPT_VERSIONS
    )
