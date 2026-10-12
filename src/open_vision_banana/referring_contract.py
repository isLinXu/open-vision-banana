"""Dependency-free RES encoding names and expression-conditioned prompt."""

BINARY_REFERRING_ENCODING = "binary_white_black_v1"
LEGACY_REFERRING_ENCODING = "legacy_color_distance_v1"


def build_binary_referring_prompt(expression: str) -> str:
    """Keep data preparation and inference on the same white/black contract."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("A nonempty expression is required for referring segmentation")
    return f"Segment {expression.strip()} in solid white. Render all other pixels black."
