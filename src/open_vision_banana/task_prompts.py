"""Shared task prompt policy for training, inference, and evaluation."""

from __future__ import annotations

CITYSCAPES_RGB_LEGEND: dict[str, tuple[int, int, int]] = {
    "road": (128, 64, 128),
    "sidewalk": (244, 35, 232),
    "building": (70, 70, 70),
    "wall": (102, 102, 156),
    "fence": (190, 153, 153),
    "pole": (153, 153, 153),
    "traffic_light": (250, 170, 30),
    "traffic_sign": (220, 220, 0),
    "vegetation": (107, 142, 35),
    "terrain": (152, 251, 152),
    "sky": (70, 130, 180),
    "person": (220, 20, 60),
    "rider": (255, 0, 0),
    "car": (0, 0, 142),
    "truck": (0, 0, 70),
    "bus": (0, 60, 100),
    "train": (0, 80, 100),
    "motorcycle": (0, 0, 230),
    "bicycle": (119, 11, 32),
}

_CITYSCAPES_LEGEND_TEXT = ", ".join(
    f'"{name}": <{red}, {green}, {blue}>'
    for name, (red, green, blue) in CITYSCAPES_RGB_LEGEND.items()
)
SEMANTIC_SEGMENTATION_PROMPT = (
    "Generate a semantic segmentation visualization image, using this color mapping: "
    "{" + _CITYSCAPES_LEGEND_TEXT + "}."
)
SEMANTIC_GENERIC_PROMPT = (
    "Generate a semantic segmentation map of this image. "
    "Each class is represented by a distinct RGB color."
)

SEMANTIC_TASK_NAMES = frozenset({"semantic", "segmentation"})


def normalize_task_prompt(task: str | None, prompt: str, *, keep_custom: bool = False) -> str:
    """Return the canonical prompt for tasks with a fixed conditioning contract.

    By default every semantic prompt is replaced by the fixed Cityscapes legend so
    training, inference and evaluation share one contract. Datasets whose targets
    are painted with a per-sample color mapping (open-vocabulary data from
    ``scripts/build_openvocab_semantic_manifest.py``) must keep their own prompt,
    otherwise prompt and target contradict each other; they pass ``keep_custom=True``.
    An empty prompt always falls back to the canonical one.
    """
    if task in SEMANTIC_TASK_NAMES:
        if keep_custom and prompt and prompt.strip():
            return prompt
        return SEMANTIC_SEGMENTATION_PROMPT
    return prompt


__all__ = [
    "SEMANTIC_SEGMENTATION_PROMPT",
    "SEMANTIC_GENERIC_PROMPT",
    "CITYSCAPES_RGB_LEGEND",
    "SEMANTIC_TASK_NAMES",
    "normalize_task_prompt",
]
