"""Validation of expression-conditioned RES manifest rows.

Legacy prompt-only manifests remain readable. Versioned binary rows require
the expression explicitly so data preparation can audit the region/query link.
"""

from __future__ import annotations

from typing import Any

from ..referring_contract import (
    BINARY_REFERRING_ENCODING,
    build_binary_referring_prompt,
    is_binary_referring_prompt,
)

BINARY_RES_ENCODING = BINARY_REFERRING_ENCODING


def resolve_referring_prompt(row: dict[str, Any]) -> str:
    """Return a conditioned prompt, rejecting missing or conflicting text."""
    encoding = row.get("encoding")
    if encoding not in (None, "", BINARY_RES_ENCODING):
        raise ValueError(f"Unsupported RES manifest encoding: {encoding!r}")
    expression = row.get("expression")
    prompt = row.get("prompt")
    if expression is not None and (not isinstance(expression, str) or not expression.strip()):
        raise ValueError("RES expression must be a non-empty string")
    if prompt is not None and not isinstance(prompt, str):
        raise ValueError("RES prompt must be a string")
    expression = expression.strip() if expression is not None else None
    prompt = prompt.strip() if prompt else ""
    if encoding == BINARY_RES_ENCODING and expression is None:
        raise ValueError("Versioned RES rows require an explicit expression")
    if expression is not None:
        if not prompt:
            return build_binary_referring_prompt(expression)
        if encoding == BINARY_RES_ENCODING and not is_binary_referring_prompt(prompt, expression):
            raise ValueError("Versioned binary RES rows require the canonical white/black prompt")

        def normalized(text: str) -> str:
            return " ".join(text.casefold().split())

        if normalized(expression) not in normalized(prompt):
            raise ValueError("RES prompt does not contain its expression")
    if not prompt:
        raise ValueError("RES requires an expression or a complete per-sample prompt")
    return prompt
