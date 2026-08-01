#!/usr/bin/env python3
"""Dynamic prompt generator for per-sample instructions.

Generates task-specific prompts with:
- A fixed CLIP-safe instruction for semantic segmentation
- Class-color maps for instance segmentation
- Real referring expressions for referring segmentation
- Task-specific instructions for depth/normal/specular

Per-sample details remain enabled where they fit the model context window.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

from .task_prompts import SEMANTIC_SEGMENTATION_PROMPT

# ─── Color Palette (must match build_v3_dataset.py) ──────────────────────────

CITYSCAPES_PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100),
    (0, 0, 230), (119, 11, 32),
]

CITYSCAPES_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

ADE_EXT_PALETTE = [
    (0, 0, 0), (120, 120, 120), (180, 180, 180), (200, 180, 180),
    (60, 180, 75), (255, 225, 25), (0, 130, 200), (210, 245, 60),
    (250, 190, 190), (245, 100, 100), (150, 80, 0), (128, 128, 0),
    (255, 200, 0), (170, 110, 40), (128, 128, 128), (238, 130, 238),
    (255, 160, 74), (106, 60, 154), (160, 160, 40), (60, 40, 200),
    (0, 250, 170), (255, 0, 255), (200, 0, 0), (0, 128, 128),
    (0, 200, 255), (200, 200, 0), (255, 128, 0), (128, 0, 255),
    (255, 255, 0), (0, 255, 128), (128, 255, 255),
]

ADE_EXT_NAMES = [
    "background", "floor", "ceiling", "cabinet", "grass", "sand",
    "water", "rug", "table", "chair", "wood", "door", "lamp",
    "sofa", "shelf", "plant", "bed", "curtain", "box", "mirror",
    "counter", "desk", "picture frame", "window", "refrigerator",
    "television", "oven", "sink", "bottle", "bag", "book"
]

FULL_PALETTE = CITYSCAPES_PALETTE + ADE_EXT_PALETTE
FULL_NAMES = CITYSCAPES_NAMES + ADE_EXT_NAMES

INSTANCE_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]

INSTANCE_NAMES = [
    "object 1", "object 2", "object 3", "object 4", "object 5",
    "object 6", "object 7", "object 8", "object 9", "object 10",
    "object 11", "object 12", "object 13", "object 14", "object 15",
    "object 16", "object 17", "object 18", "object 19", "object 20",
]


# ─── Prompt Templates ─────────────────────────────────────────────────────────

DEPTH_TEMPLATES = [
    "Estimate the depth map for this image. Closer objects should appear in warm colors (red/yellow), farther objects in cool colors (blue/purple). Output a smooth depth visualization preserving spatial structure.",
    "Generate a depth estimation visualization. Use the turbo colormap: red for near, blue for far. Maintain the spatial layout of the input.",
    "Create a monocular depth map. Near surfaces in warm tones, distant surfaces in cool tones. Preserve all spatial boundaries.",
]

NORMAL_TEMPLATES = [
    "Estimate the surface normal map for this image. Encode normals as RGB: R=right, G=up, B=toward viewer. Flat surfaces facing the camera should be light purple (128,128,255).",
    "Generate a surface normal visualization. Map normals to colors: right=red, up=green, forward=blue. Preserve geometric detail.",
    "Create a normal map estimation. Use the standard RGB encoding for surface normals. Curved surfaces should show gradient transitions.",
]

SPECULAR_TEMPLATES = [
    "Remove all specular highlights and glossy reflections from this image. Preserve the underlying surface texture and color. Output should look evenly lit without bright hotspots.",
    "Eliminate glare and specular reflections while maintaining the original material appearance. The result should look diffuse and evenly lit.",
    "Strip away bright reflections and hotspots. Keep colors and textures intact. Produce a matte, diffuse version of the surface.",
]


# ─── Dynamic Prompt Builders ──────────────────────────────────────────────────

def detect_semantic_classes(target_img: Image.Image) -> list[tuple[int, str, tuple[int, int, int]]]:
    """Detect which palette classes are present in the semantic target."""
    arr = np.array(target_img.convert("RGB"))
    flat = arr.reshape(-1, 3)
    present = []
    for idx, color in enumerate(FULL_PALETTE):
        matches = np.all(flat == np.array(color), axis=1)
        if matches.any():
            name = FULL_NAMES[idx] if idx < len(FULL_NAMES) else f"class_{idx}"
            present.append((idx, name, color))
    return present


def build_semantic_prompt(target_img: Image.Image | None = None, meta: dict | None = None) -> str:
    """Build the canonical CLIP-safe semantic segmentation prompt."""
    del target_img, meta
    return SEMANTIC_SEGMENTATION_PROMPT


def build_instance_prompt(meta: dict | None = None) -> str:
    """Build dynamic instance segmentation prompt."""
    instances = meta.get("instances", []) if meta else []

    if not instances:
        return (
            "Generate an instance segmentation mask. "
            "Each separate object instance must have a distinct color. "
            "Background should be black (0,0,0)."
        )

    legend_parts = []
    for i, (cls_name, color) in enumerate(instances[:20]):
        r, g, b = color
        legend_parts.append(f"{cls_name} instance {i+1}=RGB({r},{g},{b})")
    legend = ", ".join(legend_parts)

    return (
        f"Generate an instance segmentation mask. "
        f"Color legend: {legend}. "
        f"Each separate object instance must have a distinct color. "
        f"Background should be black (0,0,0)."
    )


def build_referring_prompt(meta: dict | None = None) -> str:
    """Build dynamic referring expression segmentation prompt."""
    expr = meta.get("expression", "the main object") if meta else "the main object"

    return (
        f"Segment the object described as: '{expr}'. "
        f"Highlight only the described region in white (255,255,255), "
        f"everything else in black (0,0,0)."
    )


def build_depth_prompt(meta: dict | None = None) -> str:
    return random.choice(DEPTH_TEMPLATES)


def build_normal_prompt(meta: dict | None = None) -> str:
    return random.choice(NORMAL_TEMPLATES)


def build_specular_prompt(meta: dict | None = None) -> str:
    return random.choice(SPECULAR_TEMPLATES)


def build_generation_prompt(meta: dict | None = None) -> str:
    """Build a text-to-image generation prompt (protection branch)."""
    prompts = [
        "A photorealistic image of a modern city street at sunset, with cars and pedestrians",
        "A cozy living room interior with warm lighting, wooden furniture, and a fireplace",
        "A lush green forest path with sunlight filtering through the trees",
        "An aerial view of a coastal town with colorful houses and a harbor",
        "A close-up portrait of a cat sitting on a windowsill with soft natural light",
        "A minimalist studio product shot of a ceramic vase on a white background",
        "A vibrant autumn landscape with red and orange leaves reflected in a calm lake",
        "A futuristic cityscape at night with neon lights and flying vehicles",
        "A rustic kitchen with copper pots, herbs, and a wooden cutting board",
        "A dramatic mountain vista with clouds rolling over snow-capped peaks",
    ]
    if meta and "prompt" in meta:
        return meta["prompt"]
    return random.choice(prompts)


# ─── Prompt Registry ──────────────────────────────────────────────────────────

PROMPT_BUILDERS = {
    "depth": build_depth_prompt,
    "normal": build_normal_prompt,
    "semantic": build_semantic_prompt,
    "instance": build_instance_prompt,
    "referring": build_referring_prompt,
    "specular": build_specular_prompt,
    "generation": build_generation_prompt,
}


def build_prompt(task: str, meta: dict | None = None, target_img: Image.Image | None = None) -> str:
    """Build a dynamic prompt for a given task and sample metadata."""
    builder = PROMPT_BUILDERS.get(task)
    if builder is None:
        raise ValueError(f"Unknown task: {task}. Available: {list(PROMPT_BUILDERS.keys())}")

    if task == "semantic" and target_img is not None:
        return builder(target_img=target_img, meta=meta)
    return builder(meta=meta)


# ─── Referring Expression Generator ───────────────────────────────────────────

REFERRING_TEMPLATES = {
    "person": [
        "the person on the {position}",
        "the {attribute} person",
        "the person wearing {clothing}",
    ],
    "car": [
        "the {color} car",
        "the car on the {position}",
        "the {size} vehicle",
    ],
    "animal": [
        "the {animal_type} in the {location}",
        "the {color} {animal_type}",
    ],
    "object": [
        "the {object_name} on the {surface}",
        "the {color} {object_name}",
        "the {size} {object_name} in the {location}",
    ],
}


def generate_referring_expression(meta: dict) -> str:
    """Generate a natural referring expression from metadata."""
    category = meta.get("category", "object")
    templates = REFERRING_TEMPLATES.get(category, REFERRING_TEMPLATES["object"])
    template = random.choice(templates)

    replacements = {
        "position": meta.get("position", "left"),
        "attribute": meta.get("attribute", "standing"),
        "clothing": meta.get("clothing", "a jacket"),
        "color": meta.get("color", "red"),
        "size": meta.get("size", "large"),
        "animal_type": meta.get("animal_type", "dog"),
        "location": meta.get("location", "background"),
        "object_name": meta.get("object_name", "vase"),
        "surface": meta.get("surface", "table"),
        "vehicle": meta.get("vehicle", "car"),
    }

    return template.format(**replacements)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test prompt generation
    print("=== Depth ===")
    print(build_prompt("depth"))
    print("\n=== Normal ===")
    print(build_prompt("normal"))
    print("\n=== Specular ===")
    print(build_prompt("specular"))
    print("\n=== Semantic (with meta) ===")
    print(build_prompt("semantic", meta={"classes": [0, 8, 10, 13]}))
    print("\n=== Instance ===")
    print(build_prompt("instance", meta={"instances": [("car", (31, 119, 180)), ("person", (255, 127, 14))]}))
    print("\n=== Referring ===")
    print(build_prompt("referring", meta={"expression": "the red car on the left"}))
    print("\n=== Generation ===")
    print(build_prompt("generation"))
