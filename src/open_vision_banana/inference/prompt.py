"""Task-specific prompt construction for Vision Banana.

The paper uses various prompting styles:
- Natural language: "the macaron cakes are represented by yellow"
- Structured JSON: {"cat": "red", "background": "yellow"}
- RGB tuples: "cat ears: <255, 165, 0>"
- Hex codes: "The menu is #80C000"
"""

from __future__ import annotations

from ..task_prompts import (
    CITYSCAPES_RGB_LEGEND,
    SEMANTIC_GENERIC_PROMPT,
    SEMANTIC_SEGMENTATION_PROMPT,
)


class PromptBuilder:
    """Build instruction prompts for Vision Banana."""

    SEMANTIC_PROMPT = SEMANTIC_SEGMENTATION_PROMPT

    SEMANTIC_TEMPLATES = [
        'Generate a semantic segmentation visualization image, '
        'using this color mapping: {color_map}.',
        'This image is a per-pixel class labeling of the input. {descriptions}.',
        'Generate a semantic segmentation visualization of the input. {hex_descriptions}.',
    ]

    INSTANCE_TEMPLATES = [
        'Generate an instance segmentation visualization of this image. '
        'Each {object_name} is colored differently.',
        'This image shows segmentation masks for the {object_name} from the '
        'input image. The background is set to {bg_color}. Each {object_name} '
        'instance is represented by a solid circular mask, and a different '
        'color is used for each mask.',
    ]

    REFERRING_TEMPLATES = [
        'A segmentation map image. The area that corresponds to '
        '{expression} is rendered solid {color}; the other objects are '
        'rendered in {other_color}.',
        'This image shows segmentation masks from the given image. '
        'The background is {bg_color} color. The {expression} is '
        'represented by a solid {color}.',
    ]

    DEPTH_PROMPT = 'Generate a metric depth visualization of this image.'
    NORMAL_PROMPT = 'Generate a surface normal estimation visualization of this image.'
    EDGE_PROMPT = 'This image is an edge detection map. The edges are rendered in white, the background in black.'
    FLOW_PROMPT = 'This image is an optical flow visualization. Direction is encoded by hue, magnitude by saturation.'
    KEYPOINT_PROMPT = 'This image shows 2D keypoint detections. Each keypoint is represented by a colored dot on a black background.'
    OBJECT_DETECTION_PROMPT = 'This image shows object detection results. Each detected object is outlined by a colored bounding box on a black background.'
    DEPTH_EDGE_PROMPT = 'This image is a depth edge map. Depth discontinuities are rendered in white, the background in black.'

    @classmethod
    def build_semantic_prompt(
        cls,
        class_color_map: dict[str, tuple[int, int, int]],
        template_idx: int | None = None,
    ) -> str:
        """Build the semantic prompt used by the RGB decoder.

        Cityscapes has a fixed 19-class output alphabet.  Its complete RGB
        legend is part of the model contract and therefore is included by
        default.  Larger open-vocabulary maps keep the compact prompt unless
        a template is explicitly selected, since their legends cannot fit in
        the short-text encoders used by some legacy tasks.
        """
        # Callers often pass only the classes present in one image.  If those
        # colors are an exact Cityscapes subset, expand back to the canonical
        # alphabet so training, inference, and evaluation share one contract.
        if template_idx is None and class_color_map and all(
            name in CITYSCAPES_RGB_LEGEND
            and tuple(color) == CITYSCAPES_RGB_LEGEND[name]
            for name, color in class_color_map.items()
        ):
            class_color_map = CITYSCAPES_RGB_LEGEND
        if template_idx is None:
            template_idx = 0 if len(class_color_map) <= 19 else None
            if template_idx is None:
                return SEMANTIC_GENERIC_PROMPT
        if template_idx == 0:
            color_parts = []
            for cls_name, (r, g, b) in class_color_map.items():
                color_parts.append(f'"{cls_name}": <{r}, {g}, {b}>')
            color_map_str = '{' + ', '.join(color_parts) + '}'
            return cls.SEMANTIC_TEMPLATES[0].format(color_map=color_map_str)

        elif template_idx == 1:
            descriptions = []
            for cls_name, (r, g, b) in class_color_map.items():
                descriptions.append(
                    f'The {cls_name} are represented by ({r}, {g}, {b})'
                )
            return cls.SEMANTIC_TEMPLATES[1].format(
                descriptions='. '.join(descriptions)
            )

        elif template_idx == 2:
            hex_parts = []
            for cls_name, (r, g, b) in class_color_map.items():
                hex_code = f'#{r:02X}{g:02X}{b:02X}'
                hex_parts.append(f'The {cls_name} is {hex_code}')
            return cls.SEMANTIC_TEMPLATES[2].format(
                hex_descriptions='. '.join(hex_parts)
            )

        raise ValueError(f"Unknown template index: {template_idx}")

    @classmethod
    def build_instance_prompt(
        cls,
        object_name: str,
        background_color: str = "black",
        template_idx: int = 0,
    ) -> str:
        """Build prompt for instance segmentation."""
        template = cls.INSTANCE_TEMPLATES[template_idx]
        return template.format(
            object_name=object_name,
            bg_color=background_color,
        )

    @classmethod
    def build_referring_prompt(
        cls,
        expression: str,
        color: str = "white",
        other_color: str = "green",
        bg_color: str = "black",
        template_idx: int = 0,
    ) -> str:
        """Build prompt for referring expression segmentation."""
        template = cls.REFERRING_TEMPLATES[template_idx]
        return template.format(
            expression=expression,
            color=color,
            other_color=other_color,
            bg_color=bg_color,
        )

    @classmethod
    def build_depth_prompt(cls) -> str:
        """Build prompt for metric depth estimation."""
        return cls.DEPTH_PROMPT

    @classmethod
    def build_normal_prompt(cls) -> str:
        """Build prompt for surface normal estimation."""
        return cls.NORMAL_PROMPT

    @classmethod
    def build_edge_prompt(cls) -> str:
        """Build prompt for edge detection."""
        return cls.EDGE_PROMPT

    @classmethod
    def build_flow_prompt(cls) -> str:
        """Build prompt for optical flow."""
        return cls.FLOW_PROMPT

    @classmethod
    def build_keypoint_prompt(cls) -> str:
        """Build prompt for 2D keypoint detection."""
        return cls.KEYPOINT_PROMPT

    @classmethod
    def build_object_detection_prompt(cls) -> str:
        """Build prompt for object detection."""
        return cls.OBJECT_DETECTION_PROMPT

    @classmethod
    def build_depth_edge_prompt(cls) -> str:
        """Build prompt for depth edge detection."""
        return cls.DEPTH_EDGE_PROMPT


__all__ = ["PromptBuilder"]
