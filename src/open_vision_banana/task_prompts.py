"""Shared task prompt policy for training and inference.

Semantic segmentation intentionally uses one compact instruction.  Class-by-
class colour legends routinely exceed CLIP's 77-token context and make the
effective conditioning depend on where truncation happens.
"""

from __future__ import annotations

SEMANTIC_SEGMENTATION_PROMPT = (
    "Generate a semantic segmentation map of this image. "
    "Each class is represented by a distinct RGB color."
)

SEMANTIC_TASK_NAMES = frozenset({"semantic", "segmentation"})


def normalize_task_prompt(task: str | None, prompt: str) -> str:
    """Return the canonical prompt for tasks with a fixed conditioning contract."""
    if task in SEMANTIC_TASK_NAMES:
        return SEMANTIC_SEGMENTATION_PROMPT
    return prompt


__all__ = [
    "SEMANTIC_SEGMENTATION_PROMPT",
    "SEMANTIC_TASK_NAMES",
    "normalize_task_prompt",
]
