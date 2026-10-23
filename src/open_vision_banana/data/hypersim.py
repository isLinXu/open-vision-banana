"""Hypersim geometry channels -> paper RGB targets (normal, semantic, instance).

Hypersim ships, per frame and camera, ``*.normal_cam.hdf5`` (camera-space unit
normals), ``*.semantic.hdf5`` (NYU40 ids, -1 = unlabelled) and
``*.semantic_instance.hdf5`` (instance ids, -1 = none). This module turns them
into the paper's RGB contracts so one Hypersim scene feeds four tasks.

Coordinate frame: Hypersim's camera space is OpenGL-style right-handed (+x right,
+y up, +z towards the viewer / out of the image plane), which is exactly the
frame the paper states for its normal encoding, so ``normal_cam`` maps through
``NormalRGBMapper`` without any axis flip. ``normal_backfacing_fraction`` measures
how many valid normals point clearly away from the camera along their own view ray
(n . -ray < -0.1). Visible surfaces cannot face away, so for a correctly oriented
frame this is ~0; the preparer warns when it exceeds 0.1, the signature of a flipped
axis. The optical axis alone is not enough: off-axis pixels of the 60 degree camera
see oblique walls with n_z down to about -0.5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from ..core.normal_mapper import NormalRGBMapper
from ..inference.prompt import PromptBuilder
from .depth_geometry import hypersim_focal_px
from .openvocab_semantic import OpenVocabSample, sample_openvocab_semantic

# NYU40 label set used by Hypersim's semantic.hdf5 (id 1..40; 0 / -1 unlabelled).
NYU40_CLASSES: tuple[str, ...] = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
    "floor mat", "clothes", "ceiling", "books", "refrigerator", "television", "paper", "towel",
    "shower curtain", "box", "whiteboard", "person", "night stand", "toilet", "sink", "lamp",
    "bathtub", "bag", "other structure", "other furniture", "other prop",
)
# Catch-all NYU40 classes are not meaningful instance prompts ("Each other prop is colored differently").
INSTANCE_EXCLUDED_CLASSES = frozenset({"other structure", "other furniture", "other prop"})
INVALID_NORMAL_RGB = (128, 128, 128)  # the encoding of the zero vector; evaluators mask invalid GT anyway


BACKFACING_COS = -0.1


def normal_backfacing_fraction(normal: np.ndarray, focal_px: float | None = None) -> float:
    """Fraction of valid normals facing away from the camera: n . -ray < -0.1.

    Rays use pixel centres in this camera frame (looking along -z, +y up, image rows down);
    ``focal_px`` defaults to the Hypersim 60 degree rule for the frame width.
    """
    normal = np.asarray(normal, dtype=np.float32)
    norm = np.linalg.norm(normal, axis=-1)
    valid = np.isfinite(normal).all(axis=-1) & (norm > 1e-6)
    if not valid.any():
        return 0.0
    height, width = normal.shape[:2]
    focal = hypersim_focal_px(width) if focal_px is None else float(focal_px)
    # pixel-centre ray offsets, as in ray_distance_to_planar_depth; towards the camera is (-u, v, 1)
    us = ((np.arange(width) - (width - 1) / 2.0) / focal).astype(np.float32)[None, :]
    vs = ((np.arange(height) - (height - 1) / 2.0) / focal).astype(np.float32)[:, None]
    dot = -us * normal[..., 0] + vs * normal[..., 1] + normal[..., 2]
    facing = dot / (norm * np.sqrt(us**2 + vs**2 + 1.0))
    return float((facing[valid] < BACKFACING_COS).mean())


def flip_normals(normal: np.ndarray, flip: str = "none") -> np.ndarray:
    """Float32 copy of ``[H, W, 3]`` normals in the paper frame.

    ``flip`` is an escape hatch for exports in another frame: ``"z"`` negates n_z,
    ``"yz"`` negates n_y and n_z (OpenCV-style camera to the paper frame).
    """
    n = np.array(normal, dtype=np.float32, copy=True)
    if n.ndim != 3 or n.shape[-1] != 3:
        raise ValueError(f"normal map must be [H, W, 3], got {n.shape}")
    if flip == "z":
        n[..., 2] *= -1
    elif flip == "yz":
        n[..., 1:] *= -1
    elif flip != "none":
        raise ValueError("flip must be none, z or yz")
    return n


def normal_to_target(normal: np.ndarray, *, flip: str = "none") -> tuple[np.ndarray, float]:
    """``[H, W, 3]`` camera-space normals -> uint8 RGB target and valid fraction.

    ``flip`` as in ``flip_normals``.
    """
    n = flip_normals(normal, flip)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-6)
    n = np.where(valid[..., None], n / np.clip(norm, 1e-6, None), 0.0)
    rgb = NormalRGBMapper().normal_to_rgb(torch.from_numpy(n).permute(2, 0, 1).unsqueeze(0))[0]
    target = (rgb.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    target[~valid] = INVALID_NORMAL_RGB
    return target, float(valid.mean())


def nyu40_to_label_index(semantic: np.ndarray) -> np.ndarray:
    """NYU40 ids (1..40, <=0 unlabelled) -> 0-based class index with -1 for ignore."""
    sem = np.asarray(semantic).astype(np.int64)
    index = sem - 1
    index[(sem <= 0) | (sem > len(NYU40_CLASSES))] = -1
    return index


def semantic_sample(
    semantic: np.ndarray, rng: random.Random, *, max_present: int = 12, max_absent: int = 6
) -> OpenVocabSample:
    """One randomized open-vocabulary (target, prompt) pair from Hypersim NYU40 labels."""
    return sample_openvocab_semantic(
        nyu40_to_label_index(semantic), NYU40_CLASSES, rng,
        ignore_index=-1, max_present=max_present, max_absent=max_absent,
    )


@dataclass(frozen=True)
class InstanceTarget:
    class_name: str
    target: np.ndarray  # [H, W, 3] uint8, black background
    num_instances: int
    prompt: str


def instance_targets(
    semantic: np.ndarray,
    instance: np.ndarray,
    rng: random.Random,
    *,
    palette: list[tuple[int, int, int]],
    min_instances: int = 1,
    min_pixels: int = 64,
    background: tuple[int, int, int] = (0, 0, 0),
) -> list[InstanceTarget]:
    """One paper-style instance target per NYU40 class present in the frame.

    Instances of a class are the distinct ``semantic_instance`` ids inside that
    class's pixels; ids covering fewer than ``min_pixels`` are dropped as render
    noise. Colours come from ``palette`` (>= 60 apart, far from ``background``).
    """
    sem = np.asarray(semantic).astype(np.int64)
    inst = np.asarray(instance).astype(np.int64)
    if sem.shape != inst.shape:
        raise ValueError("semantic and instance maps must have the same shape")
    results: list[InstanceTarget] = []
    for class_id in np.unique(sem):
        if class_id <= 0 or class_id > len(NYU40_CLASSES):
            continue
        name = NYU40_CLASSES[class_id - 1]
        if name in INSTANCE_EXCLUDED_CLASSES:
            continue
        class_mask = sem == class_id
        ids, counts = np.unique(inst[class_mask], return_counts=True)
        keep = [(i, c) for i, c in zip(ids, counts, strict=True) if i >= 0 and c >= min_pixels]
        if len(keep) < min_instances or len(keep) > len(palette):
            continue
        colors = palette[: len(keep)]
        rng.shuffle(colors)
        target = np.empty((*sem.shape, 3), dtype=np.uint8)
        target[...] = np.asarray(background, np.uint8)
        # largest first so small instances stay on top of any overlap
        for (instance_id, _), color in zip(sorted(keep, key=lambda kc: -kc[1]), colors, strict=True):
            target[class_mask & (inst == instance_id)] = color
        results.append(
            InstanceTarget(
                class_name=name,
                target=target,
                num_instances=len(keep),
                prompt=PromptBuilder.build_instance_prompt(name),
            )
        )
    return results


__all__ = [
    "INSTANCE_EXCLUDED_CLASSES",
    "INVALID_NORMAL_RGB",
    "NYU40_CLASSES",
    "InstanceTarget",
    "instance_targets",
    "flip_normals",
    "normal_backfacing_fraction",
    "normal_to_target",
    "nyu40_to_label_index",
    "semantic_sample",
]
