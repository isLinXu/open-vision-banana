"""Conservative, category-conditioned COCONut-B RGB training targets.

Source black is background relative to the selected categories, not a universal
negative label. Unrecognized classes are never used as absent-class negatives.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

REVISION = "ab56be89855bd1414a4be2aa98fef35602adc892"
POLICY_VERSION = "coconut_b_conditioned_v1"
# Explicit, auditable vocabulary. Deliberately omit food, watch, container,
# plant, table and seat: the source's scope/aliases can be ambiguous.
THINGS = frozenset(
    [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "skateboard",
        "surfboard",
        "bottle",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "bed",
        "toilet",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
    ]
) | frozenset(
    {
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "sports ball",
        "baseball bat",
        "baseball glove",
        "tennis racket",
        "wine glass",
        "dining table",
        "potted plant",
        "cell phone",
        "teddy bear",
        "hair drier",
        "toothbrush",
    }
)
STUFF = frozenset(
    {
        "sky",
        "grass",
        "tree",
        "snow",
        "sand",
        "road",
        "sidewalk",
        "river",
        "sea",
        "ceiling",
        "floor",
        "wall",
        "fence",
        "bridge",
        "mountain",
    }
)
ALIASES = {
    "people": "person",
    "pedestrian": "person",
    "human": "person",
    "automobile": "car",
    "auto": "car",
    "bike": "bicycle",
    "cycle": "bicycle",
    "motorbike": "motorcycle",
    "plane": "airplane",
    "aircraft": "airplane",
    "lorry": "truck",
    "cargo truck": "truck",
    "avian": "bird",
    "feline": "cat",
    "kitten": "cat",
    "puppy": "dog",
    "stallion": "horse",
    "cattle": "cow",
    "ox": "cow",
    "lamb": "sheep",
    "ursine": "bear",
    "elephantidae": "elephant",
    "fridge": "refrigerator",
    "computer mouse": "mouse",
    "remote control": "remote",
    "smartphone": "cell phone",
    "mobile phone": "cell phone",
    "doughnut": "donut",
    "necktie": "tie",
    "sofa": "couch",
    "hydrant": "fire hydrant",
    "traffic signal": "traffic light",
    "signal light": "traffic light",
}
LEGEND = re.compile(r"<p>([^<>]+)<color>\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)</color></p>")


def parse_legend(row):
    answers = [x["value"] for x in row["conversations"] if x["from"] == "gpt"]
    if len(answers) != 1:
        raise ValueError("expected one source answer")
    matches = LEGEND.findall(answers[0])
    if not matches or len(matches) != answers[0].count("<color>"):
        raise ValueError("malformed source legend")
    result = []
    for label, r, g, b in matches:
        color = tuple(map(int, (r, g, b)))
        if min(color) < 0 or max(color) > 255 or color == (0, 0, 0):
            raise ValueError("invalid or background legend color")
        match = re.fullmatch(r"(.+)-(\d+)", label)
        name = match[1] if match else label
        canonical = ALIASES.get(name, name)
        result.append({"label": label, "category": canonical, "indexed": bool(match), "rgb": color})
    if len({x["label"] for x in result}) != len(result) or len({x["rgb"] for x in result}) != len(
        result
    ):
        raise ValueError("duplicate label/color")
    if row.get("num_instances") != len(result):
        raise ValueError("source region count mismatch")
    return result


def eligible_groups(rgb, entries, min_pixels=64, min_fraction=2e-4):
    """Reject an entire category if any of its source instances is empty/tiny."""
    packed = (
        (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    )
    values, counts = np.unique(packed, return_counts=True)
    observed = dict(zip(map(int, values), map(int, counts), strict=True))
    keys = {(e["rgb"][0] << 16) | (e["rgb"][1] << 8) | e["rgb"][2] for e in entries}
    if set(observed) - keys - {0}:
        raise ValueError("mask contains nonblack colors absent from legend")
    groups = defaultdict(list)
    reasons = []
    for e in entries:
        groups[e["category"]].append(e)
    accepted = {}
    limit = max(min_pixels, int(np.ceil(rgb.shape[0] * rgb.shape[1] * min_fraction)))
    for category, group in sorted(groups.items()):
        if category not in THINGS | STUFF:
            reasons.append({"category": category, "reason": "unreviewed_category"})
            continue
        if category in THINGS and not all(e["indexed"] for e in group):
            reasons.append({"category": category, "reason": "thing_without_instance_identity"})
            continue
        if category in STUFF and any(e["indexed"] for e in group):
            reasons.append({"category": category, "reason": "stuff_with_instance_identity"})
            continue
        areas = [
            observed.get((e["rgb"][0] << 16) | (e["rgb"][1] << 8) | e["rgb"][2], 0) for e in group
        ]
        if min(areas) < limit:
            reasons.append({"category": category, "reason": "empty_or_tiny_region", "areas": areas})
            continue
        accepted[category] = group
    return accepted, reasons, observed.get(0, 0) / packed.size


def render_targets(rgb, groups, seed, max_semantic=12, max_instance_rows=3):
    """Yield canonical prompts and exact RGB unions, without inventing negatives."""
    import random

    from .data.instance_palette import PALETTE
    from .data.openvocab_semantic import sample_openvocab_semantic
    from .inference.prompt import PromptBuilder

    if not groups:
        return
    categories = sorted(groups)
    labels = np.full(rgb.shape[:2], -1, np.int32)
    for i, name in enumerate(categories):
        for e in groups[name]:
            labels[np.all(rgb == e["rgb"], axis=2)] = i
    sample = sample_openvocab_semantic(
        labels,
        categories,
        random.Random(seed),
        ignore_index=-1,
        min_area_fraction=0,
        max_present=max_semantic,
        max_absent=0,
        style_weights={"json_tuple": 1},
    )
    yield (
        {
            "task": "semantic",
            "prompt": sample.prompt,
            "mapping": sample.mapping,
            "categories": list(sample.present_classes),
            "absent_classes": [],
        },
        sample.target_rgb,
    )
    choices = [name for name in categories if name in THINGS and len(groups[name]) <= len(PALETTE)]
    # Deterministic image-specific choice, not alphabetical dominance.
    random.Random(seed).shuffle(choices)
    for name in choices[:max_instance_rows]:
        colors = list(PALETTE)
        random.Random(seed + categories.index(name)).shuffle(colors)
        target = np.zeros_like(rgb)
        for entry, color in zip(groups[name], colors, strict=False):
            target[np.all(rgb == entry["rgb"], axis=2)] = color
        yield (
            {
                "task": "instance",
                "category": name,
                "num_instances": len(groups[name]),
                "background_color": [0, 0, 0],
                "prompt": PromptBuilder.build_instance_prompt(name),
            },
            target,
        )


def image_split(pixel_hash, image_id, prior_splits, val_fraction=0.02):
    if image_id in prior_splits:
        return prior_splits[image_id]
    return (
        "val"
        if int(hashlib.sha256(pixel_hash.encode()).hexdigest()[:16], 16) / 2**64 < val_fraction
        else "train"
    )
