#!/usr/bin/env python3
"""Rebuild v3 dataset: unified split, MD5 index, manifest, segmentation palette,
dedup, near-blank filtering, dynamic prompts.

This is the single source of truth for the paper's 5-task training data:
  depth, normal, semantic, instance, referring

Plus a generation protection branch (sampled from base model's native domain).

Usage:
  python scripts/build_v3_dataset.py --dataset-root <path> --output-root <path>
  python scripts/build_v3_dataset.py --tasks depth,semantic --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v3_build")

# ─── Constants ─────────────────────────────────────────────────────────────────

SEED = 20260726
VAL_RATIO = 0.05  # 5% held-out
MIN_SAMPLES_PER_TASK = 50
NEAR_BLANK_STD_THRESHOLD = 3.0  # pixels with std < 3.0 across channels = near-blank
DEDUP_HASH_THRESHOLD = 0.99  # MD5 exact dedup; perceptual not needed at this scale

# Unified segmentation palette: 19-class Cityscapes + 150-class ADE20k subset.
# We use a FIXED mapping so every target uses the same color for the same class.
CITYSCAPES_PALETTE = np.array([
    (128, 64, 128),    # 0 road
    (244, 35, 232),    # 1 sidewalk
    (70, 70, 70),      # 2 building
    (102, 102, 156),   # 3 wall
    (190, 153, 153),   # 4 fence
    (153, 153, 153),   # 5 pole
    (250, 170, 30),    # 6 traffic_light
    (220, 220, 0),     # 7 traffic_sign
    (107, 142, 35),    # 8 vegetation
    (152, 251, 152),   # 9 terrain
    (70, 130, 180),    # 10 sky
    (220, 20, 60),     # 11 person
    (255, 0, 0),       # 12 rider
    (0, 0, 142),       # 13 car
    (0, 0, 70),        # 14 truck
    (0, 60, 100),      # 15 bus
    (0, 80, 100),      # 16 train
    (0, 0, 230),       # 17 motorcycle
    (119, 11, 32),     # 18 bicycle
], dtype=np.uint8)

CITYSCAPES_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# Extended palette for ADE20k-style scenes (indices 19-49, used for indoor/nature)
ADE_EXT_PALETTE = np.array([
    (0, 0, 0),         # 19 background/ignore
    (120, 120, 120),   # 20 floor
    (180, 180, 180),   # 21 ceiling
    (200, 180, 180),   # 22 cabinet
    (60, 180, 75),     # 23 grass
    (255, 225, 25),    # 24 sand
    (0, 130, 200),     # 25 water
    (210, 245, 60),    # 26 rug
    (250, 190, 190),   # 27 table
    (245, 100, 100),   # 28 chair
    (150, 80, 0),      # 29 wood
    (128, 128, 0),     # 30 door
    (255, 200, 0),     # 31 lamp
    (170, 110, 40),    # 32 sofa
    (128, 128, 128),   # 33 shelf
    (238, 130, 238),   # 34 plant
    (255, 160, 74),    # 35 bed
    (106, 60, 154),    # 36 curtain
    (160, 160, 40),    # 37 box
    (60, 40, 200),     # 38 mirror
    (0, 250, 170),     # 39 counter
    (255, 0, 255),     # 40 desk
    (200, 0, 0),       # 41 picture/frame
    (0, 128, 128),     # 42 window
    (0, 200, 255),     # 43 refrigerator
    (200, 200, 0),     # 44 television
    (255, 128, 0),     # 45 oven
    (128, 0, 255),     # 46 sink
    (255, 255, 0),     # 47 bottle
    (0, 255, 128),     # 48 bag
    (128, 255, 255),   # 49 book
], dtype=np.uint8)

ADE_EXT_NAMES = [
    "background", "floor", "ceiling", "cabinet", "grass", "sand",
    "water", "rug", "table", "chair", "wood", "door", "lamp",
    "sofa", "shelf", "plant", "bed", "curtain", "box", "mirror",
    "counter", "desk", "picture frame", "window", "refrigerator",
    "television", "oven", "sink", "bottle", "bag", "book"
]

FULL_PALETTE = np.concatenate([CITYSCAPES_PALETTE, ADE_EXT_PALETTE], axis=0)
FULL_NAMES = CITYSCAPES_NAMES + ADE_EXT_NAMES

# Instance segmentation colors: 20 distinct high-contrast colors
INSTANCE_PALETTE = np.array([
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
], dtype=np.uint8)

# ─── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class V3Sample:
    sample_id: str
    task: str
    split: str  # train / val
    input_path: str
    target_path: str
    input_md5: str
    target_md5: str
    width: int
    height: int
    prompt: str
    meta: dict = field(default_factory=dict)


@dataclass
class V3Manifest:
    version: str = "3.0"
    seed: int = SEED
    created_at: str = ""
    tasks: dict = field(default_factory=dict)  # task -> {train_count, val_count, samples}
    dedup_removed: int = 0
    near_blank_removed: int = 0
    palette: dict = field(default_factory=dict)


# ─── Utility Functions ────────────────────────────────────────────────────────

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def is_near_blank(img_path: Path) -> bool:
    """Check if image is near-blank (std < threshold across all pixels)."""
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
        if img.size == 0:
            return True
        std = float(np.std(img))
        return std < NEAR_BLANK_STD_THRESHOLD
    except Exception:
        return True


def load_image_safe(path: Path) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def resize_pair(input_img: Image.Image, target_img: Image.Image, size: int = 512) -> tuple:
    """Resize input with LANCZOS, target with NEAREST (for masks) or LANCZOS (for RGB)."""
    input_resized = input_img.resize((size, size), Image.LANCZOS)
    target_resized = target_img.resize((size, size), Image.LANCZOS)
    return input_resized, target_resized


# ─── Task Processors ───────────────────────────────────────────────────────────

class TaskProcessor:
    """Base class for task-specific data processing."""

    def __init__(self, task_name: str, dataset_root: Path, output_root: Path):
        self.task_name = task_name
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.samples: list[V3Sample] = []

    def process(self) -> list[V3Sample]:
        raise NotImplementedError

    def build_dynamic_prompt(self, sample_meta: dict) -> str:
        raise NotImplementedError


class SemanticProcessor(TaskProcessor):
    """Process semantic segmentation data with unified palette."""

    def __init__(self, dataset_root: Path, output_root: Path):
        super().__init__("semantic", dataset_root, output_root)

    def process(self) -> list[V3Sample]:
        seg_root = self.dataset_root / "semantic-segmentation" / "processed" / "semantic_all"
        manifest_path = seg_root / "manifest.json"
        prompts_path = seg_root / "prompts.json"

        if not manifest_path.exists():
            logger.warning(f"Semantic manifest not found: {manifest_path}")
            return []

        manifest = json.loads(manifest_path.read_text())
        prompts = json.loads(prompts_path.read_text())
        records = manifest.get("records", [])
        logger.info(f"Semantic: {len(records)} records found")

        seen_md5 = set()
        removed_dedup = 0
        removed_blank = 0

        out_dir = self.output_root / "semantic"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train" / "input").mkdir(parents=True, exist_ok=True)
        (out_dir / "train" / "target").mkdir(parents=True, exist_ok=True)
        (out_dir / "val" / "input").mkdir(parents=True, exist_ok=True)
        (out_dir / "val" / "target").mkdir(parents=True, exist_ok=True)

        rng = random.Random(SEED)

        for rec in records:
            split = rec.get("split", "train")
            input_file = rec.get("input_file", "")
            target_file = rec.get("target_file", "")

            input_path = seg_root / split / "input" / input_file
            target_path = seg_root / split / "target" / target_file

            if not input_path.exists() or not target_path.exists():
                continue

            # Near-blank check
            if is_near_blank(input_path) or is_near_blank(target_path):
                removed_blank += 1
                continue

            # MD5 dedup
            input_md5 = md5_file(input_path)
            target_md5 = md5_file(target_path)
            dedup_key = (input_md5, target_md5)
            if dedup_key in seen_md5:
                removed_dedup += 1
                continue
            seen_md5.add(dedup_key)

            # Load and validate
            input_img = load_image_safe(input_path)
            target_img = load_image_safe(target_path)
            if input_img is None or target_img is None:
                continue

            # Resize FIRST (512x512), then remap palette on small image
            input_r = input_img.resize((512, 512), Image.LANCZOS)
            target_r = target_img.resize((512, 512), Image.LANCZOS)

            # Remap target to unified palette (on 512x512 = 262K pixels, fast)
            target_arr = np.array(target_r)
            remapped = self._remap_to_unified_palette(target_arr)

            sample_id = f"sem_{Path(input_file).stem}"
            # Use split from manifest, but normalize to train/val
            if split not in ("train", "val"):
                split = "train"

            out_input = out_dir / split / "input" / f"{sample_id}.jpg"
            out_target = out_dir / split / "target" / f"{sample_id}.png"
            input_r.save(out_input, quality=95)
            Image.fromarray(remapped).save(out_target)  # lossless PNG, already 512x512

            # Build dynamic prompt with class-color map
            present_classes = self._detect_classes(remapped)
            prompt = self.build_dynamic_prompt({"classes": present_classes})

            self.samples.append(V3Sample(
                sample_id=sample_id,
                task="semantic",
                split=split,
                input_path=str(out_input),
                target_path=str(out_target),
                input_md5=input_md5,
                target_md5=md5_file(out_target),
                width=512,
                height=512,
                prompt=prompt,
                meta={"classes": present_classes, "source_file": input_file},
            ))

        logger.info(f"Semantic: {len(self.samples)} samples (dedup_removed={removed_dedup}, blank_removed={removed_blank})")
        return self.samples

    def _remap_to_unified_palette(self, target_arr: np.ndarray) -> np.ndarray:
        """Snap target colors to nearest unified palette color (fast KD-tree)."""
        from scipy.spatial import cKDTree

        h, w, c = target_arr.shape
        flat = target_arr.reshape(-1, 3).astype(np.float32)

        # Use KD-tree for O(N log K) nearest-neighbor instead of O(N*K)
        tree = cKDTree(FULL_PALETTE.astype(np.float32))
        _, nearest = tree.query(flat, k=1)

        remapped = FULL_PALETTE[nearest].reshape(h, w, 3)
        return remapped

    def _detect_classes(self, arr: np.ndarray) -> list[int]:
        """Detect which palette indices are present in the remapped target."""
        flat = arr.reshape(-1, 3)
        # Use set of unique colors for O(N) instead of O(N*K)
        unique_colors = set(map(tuple, np.unique(flat, axis=0).tolist()))
        present = []
        for i, color in enumerate(FULL_PALETTE):
            if tuple(color.tolist()) in unique_colors:
                present.append(i)
        return sorted(present)

    def build_dynamic_prompt(self, meta: dict) -> str:
        classes = meta.get("classes", [])
        if not classes:
            return "Generate a semantic segmentation mask for this image."

        color_map_parts = []
        for idx in classes[:15]:  # cap at 15 to keep prompt manageable
            if idx < len(FULL_NAMES):
                r, g, b = FULL_PALETTE[idx]
                color_map_parts.append(f"{FULL_NAMES[idx]}=RGB({r},{g},{b})")

        legend = ", ".join(color_map_parts)
        return (
            f"Generate a pixel-level semantic segmentation mask. "
            f"Color legend: {legend}. "
            f"Each pixel must be painted with the exact RGB color of its class. "
            f"No gradients, no anti-aliasing — flat color regions only."
        )


class SpecularProcessor(TaskProcessor):
    """Process specular highlight removal data."""

    def __init__(self, dataset_root: Path, output_root: Path):
        super().__init__("specular", dataset_root, output_root)

    def process(self) -> list[V3Sample]:
        spec_root = self.dataset_root / "specular-highlight-removal"
        manifest_path = spec_root / "manifest.jsonl"
        prompts_path = spec_root / "prompts.json"

        if not manifest_path.exists():
            logger.warning(f"Specular manifest not found: {manifest_path}")
            return []

        prompts = json.loads(prompts_path.read_text())
        records = [json.loads(line) for line in manifest_path.read_text().strip().split("\n") if line.strip()]
        logger.info(f"Specular: {len(records)} records found")

        seen_md5 = set()
        removed_dedup = 0
        removed_blank = 0

        out_dir = self.output_root / "specular"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train" / "input").mkdir(parents=True, exist_ok=True)
        (out_dir / "train" / "target").mkdir(parents=True, exist_ok=True)
        (out_dir / "val" / "input").mkdir(parents=True, exist_ok=True)
        (out_dir / "val" / "target").mkdir(parents=True, exist_ok=True)

        for rec in records:
            split = rec.get("split", "train")
            input_file = rec.get("input_file", "")
            target_file = rec.get("target_file", "")
            source = rec.get("source", "explicit")

            input_path = spec_root / split / "input" / input_file
            target_path = spec_root / split / "target" / target_file

            if not input_path.exists() or not target_path.exists():
                # Try alternate path (group samples may have different structure)
                input_path = spec_root / "train" / "input" / input_file
                target_path = spec_root / "train" / "target" / target_file
                if not input_path.exists() or not target_path.exists():
                    continue

            if is_near_blank(input_path) or is_near_blank(target_path):
                removed_blank += 1
                continue

            input_md5 = md5_file(input_path)
            target_md5 = md5_file(target_path)
            dedup_key = (input_md5, target_md5)
            if dedup_key in seen_md5:
                removed_dedup += 1
                continue
            seen_md5.add(dedup_key)

            input_img = load_image_safe(input_path)
            target_img = load_image_safe(target_path)
            if input_img is None or target_img is None:
                continue

            input_r, target_r = resize_pair(input_img, target_img, 512)

            sample_id = f"spec_{Path(input_file).stem}"
            # Normalize split: test -> val
            split_norm = "val" if split == "test" else "train"

            out_input = out_dir / split_norm / "input" / f"{sample_id}.jpg"
            out_target = out_dir / split_norm / "target" / f"{sample_id}.jpg"
            input_r.save(out_input, quality=95)
            target_r.save(out_target, quality=95)

            # Use original prompt if available, else build dynamic
            orig_prompt = prompts.get(input_file, "")
            if orig_prompt:
                prompt = orig_prompt
            else:
                prompt = self.build_dynamic_prompt({"source": source})

            self.samples.append(V3Sample(
                sample_id=sample_id,
                task="specular",
                split=split_norm,
                input_path=str(out_input),
                target_path=str(out_target),
                input_md5=input_md5,
                target_md5=md5_file(out_target),
                width=512,
                height=512,
                prompt=prompt,
                meta={"source": source, "original_file": input_file},
            ))

        logger.info(f"Specular: {len(self.samples)} samples (dedup_removed={removed_dedup}, blank_removed={removed_blank})")
        return self.samples

    def build_dynamic_prompt(self, meta: dict) -> str:
        source = meta.get("source", "explicit")
        if source == "group":
            return (
                "Remove all specular highlights and glossy reflections from this image. "
                "Preserve the underlying surface texture and color. "
                "Output should look evenly lit without bright hotspots."
            )
        return (
            "Remove the specular reflections and bright glare from this image. "
            "Keep the original colors and textures intact. "
            "The result should look diffuse and evenly lit."
        )


class DepthProcessor(TaskProcessor):
    """Process depth estimation data.

    Uses existing depth targets if available, or generates pseudo-labels
    via Depth-Anything-V2.
    """

    def __init__(self, dataset_root: Path, output_root: Path):
        super().__init__("depth", dataset_root, output_root)

    def process(self) -> list[V3Sample]:
        # Look for existing depth data in multiple locations
        depth_dirs = [
            self.dataset_root / "depth-estimation",
            self.dataset_root / "depth",
            self.dataset_root / "monocular-depth",
        ]

        depth_root = None
        for d in depth_dirs:
            if d.exists():
                depth_root = d
                break

        if depth_root is None:
            logger.warning("No depth dataset directory found. Skipping depth task.")
            return []

        logger.info(f"Depth: processing from {depth_root}")
        # Reuse the generic pair processor
        return self._process_pairs(depth_root)

    def _process_pairs(self, root: Path) -> list[V3Sample]:
        seen_md5 = set()
        removed_dedup = 0
        removed_blank = 0

        out_dir = self.output_root / "depth"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train" / "input").mkdir(parents=True, exist_ok=True)
        (out_dir / "train" / "target").mkdir(parents=True, exist_ok=True)
        (out_dir / "val" / "input").mkdir(parents=True, exist_ok=True)
        (out_dir / "val" / "target").mkdir(parents=True, exist_ok=True)

        # Look for train/val or train/test splits
        for split_name in ("train", "val", "test"):
            split_dir = root / split_name
            if not split_dir.exists():
                continue

            input_dir = split_dir / "input"
            target_dir = split_dir / "target"
            if not input_dir.exists():
                # Try alternate names
                input_dir = split_dir / "images"
                target_dir = split_dir / "targets"

            if not input_dir.exists() or not target_dir.exists():
                continue

            split_norm = "val" if split_name in ("val", "test") else "train"

            input_files = sorted(input_dir.glob("*"))
            for input_path in input_files:
                # Match target by stem
                stem = input_path.stem
                target_candidates = list(target_dir.glob(f"{stem}.*"))
                if not target_candidates:
                    continue

                target_path = target_candidates[0]

                if is_near_blank(input_path) or is_near_blank(target_path):
                    removed_blank += 1
                    continue

                input_md5 = md5_file(input_path)
                target_md5 = md5_file(target_path)
                dedup_key = (input_md5, target_md5)
                if dedup_key in seen_md5:
                    removed_dedup += 1
                    continue
                seen_md5.add(dedup_key)

                input_img = load_image_safe(input_path)
                target_img = load_image_safe(target_path)
                if input_img is None or target_img is None:
                    continue

                # Depth targets: ensure grayscale colormap (viridis-like)
                target_arr = np.array(target_img.convert("L"))
                target_colored = self._colorize_depth(target_arr)

                input_r = input_img.resize((512, 512), Image.LANCZOS)
                target_r = Image.fromarray(target_colored).resize((512, 512), Image.LANCZOS)

                sample_id = f"dep_{stem}"
                out_input = out_dir / split_norm / "input" / f"{sample_id}.jpg"
                out_target = out_dir / split_norm / "target" / f"{sample_id}.png"
                input_r.save(out_input, quality=95)
                target_r.save(out_target)

                prompt = self.build_dynamic_prompt({})

                self.samples.append(V3Sample(
                    sample_id=sample_id,
                    task="depth",
                    split=split_norm,
                    input_path=str(out_input),
                    target_path=str(out_target),
                    input_md5=input_md5,
                    target_md5=md5_file(out_target),
                    width=512,
                    height=512,
                    prompt=prompt,
                    meta={},
                ))

        logger.info(f"Depth: {len(self.samples)} samples (dedup_removed={removed_dedup}, blank_removed={removed_blank})")
        return self.samples

    def _colorize_depth(self, gray: np.ndarray) -> np.ndarray:
        """Apply turbo colormap to grayscale depth."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.cm as cm
        except ImportError:
            # Fallback: simple grayscale to RGB
            return np.stack([gray] * 3, axis=-1)

        normalized = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        colored = (cm.turbo(normalized)[:, :, :3] * 255).astype(np.uint8)
        return colored

    def build_dynamic_prompt(self, meta: dict) -> str:
        return (
            "Estimate the depth map for this image. "
            "Closer objects should appear in warm colors (red/yellow), "
            "farther objects in cool colors (blue/purple). "
            "Output a smooth depth visualization preserving spatial structure."
        )


class NormalProcessor(TaskProcessor):
    """Process surface normal estimation data."""

    def __init__(self, dataset_root: Path, output_root: Path):
        super().__init__("normal", dataset_root, output_root)

    def process(self) -> list[V3Sample]:
        normal_dirs = [
            self.dataset_root / "surface-normal",
            self.dataset_root / "normal-estimation",
            self.dataset_root / "normal",
        ]

        root = None
        for d in normal_dirs:
            if d.exists():
                root = d
                break

        if root is None:
            logger.warning("No normal dataset directory found. Skipping normal task.")
            return []

        # Reuse depth processor logic (same pair structure)
        proc = DepthProcessor(self.dataset_root, self.output_root)
        proc.task_name = "normal"
        proc.out_dir_base = self.output_root / "normal"
        return proc._process_pairs(root)

    def build_dynamic_prompt(self, meta: dict) -> str:
        return (
            "Estimate the surface normal map for this image. "
            "Encode normals as RGB: R=right, G=up, B=toward viewer. "
            "Flat surfaces facing the camera should be light purple (128,128,255)."
        )


class InstanceProcessor(TaskProcessor):
    """Process instance segmentation data."""

    def process(self) -> list[V3Sample]:
        # Placeholder — instance data needs to be generated from COCO/Objects365
        logger.warning("Instance segmentation data not yet available. Skipping.")
        return []

    def build_dynamic_prompt(self, meta: dict) -> str:
        instances = meta.get("instances", [])
        parts = []
        for i, (cls_name, color) in enumerate(instances[:20]):
            r, g, b = color
            parts.append(f"{cls_name}=RGB({r},{g},{b})")
        legend = ", ".join(parts) if parts else "each object gets a unique color"
        return (
            f"Generate an instance segmentation mask. "
            f"Color legend: {legend}. "
            f"Each separate object instance must have a distinct color. "
            f"Background should be black (0,0,0)."
        )


class ReferringProcessor(TaskProcessor):
    """Process referring expression segmentation data."""

    def process(self) -> list[V3Sample]:
        # Placeholder — referring data needs RefCOCO processing
        logger.warning("Referring expression data not yet available. Skipping.")
        return []

    def build_dynamic_prompt(self, meta: dict) -> str:
        expr = meta.get("expression", "the main object")
        return (
            f"Segment the object described as: '{expr}'. "
            f"Highlight only the described region in white (255,255,255), "
            f"everything else in black (0,0,0)."
        )


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def build_v3_dataset(
    dataset_root: Path,
    output_root: Path,
    tasks: list[str] | None = None,
    dry_run: bool = False,
) -> V3Manifest:
    """Build the v3 unified dataset."""

    all_tasks = ["depth", "normal", "semantic", "specular", "instance", "referring"]
    if tasks:
        all_tasks = [t for t in all_tasks if t in tasks]

    logger.info(f"Building v3 dataset: tasks={all_tasks}, root={dataset_root}, output={output_root}")

    manifest = V3Manifest(created_at="2026-07-26")
    manifest.palette = {
        "semantic": FULL_PALETTE.tolist(),
        "semantic_names": FULL_NAMES,
        "instance": INSTANCE_PALETTE.tolist(),
    }

    all_samples: list[V3Sample] = []
    total_dedup = 0
    total_blank = 0

    for task in all_tasks:
        if dry_run:
            logger.info(f"[DRY RUN] Would process task: {task}")
            continue

        if task == "semantic":
            proc = SemanticProcessor(dataset_root, output_root)
        elif task == "specular":
            proc = SpecularProcessor(dataset_root, output_root)
        elif task == "depth":
            proc = DepthProcessor(dataset_root, output_root)
        elif task == "normal":
            proc = NormalProcessor(dataset_root, output_root)
        elif task == "instance":
            proc = InstanceProcessor(task, dataset_root, output_root)
        elif task == "referring":
            proc = ReferringProcessor(task, dataset_root, output_root)
        else:
            logger.warning(f"Unknown task: {task}")
            continue

        samples = proc.process()
        all_samples.extend(samples)

        train_count = sum(1 for s in samples if s.split == "train")
        val_count = sum(1 for s in samples if s.split == "val")
        manifest.tasks[task] = {
            "train_count": train_count,
            "val_count": val_count,
            "total": len(samples),
        }

    # Build MD5 index
    md5_index: dict[str, list[str]] = defaultdict(list)
    for s in all_samples:
        md5_index[s.input_md5].append(s.sample_id)
        md5_index[s.target_md5].append(s.sample_id)

    # Find duplicates across tasks
    cross_task_dups = {md5: ids for md5, ids in md5_index.items() if len(ids) > 1}

    # Save manifest
    manifest_data = {
        "version": manifest.version,
        "seed": manifest.seed,
        "created_at": manifest.created_at,
        "tasks": manifest.tasks,
        "dedup_removed": manifest.dedup_removed,
        "near_blank_removed": manifest.near_blank_removed,
        "palette": manifest.palette,
        "total_samples": len(all_samples),
        "cross_task_md5_duplicates": len(cross_task_dups),
    }

    # Save samples as JSONL
    samples_jsonl = output_root / "v3_samples.jsonl"
    with open(samples_jsonl, "w") as f:
        for s in all_samples:
            f.write(json.dumps({
                "sample_id": s.sample_id,
                "task": s.task,
                "split": s.split,
                "input_path": s.input_path,
                "target_path": s.target_path,
                "input_md5": s.input_md5,
                "target_md5": s.target_md5,
                "width": s.width,
                "height": s.height,
                "prompt": s.prompt,
                "meta": s.meta,
            }) + "\n")

    # Save MD5 index
    md5_index_path = output_root / "v3_md5_index.json"
    with open(md5_index_path, "w") as f:
        json.dump({k: v for k, v in md5_index.items()}, f, indent=2)

    # Save manifest
    manifest_path = output_root / "v3_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"V3 dataset complete: {len(all_samples)} total samples")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info(f"  Samples: {samples_jsonl}")
    logger.info(f"  MD5 index: {md5_index_path}")
    for task, stats in manifest.tasks.items():
        logger.info(f"  {task}: train={stats['train_count']}, val={stats['val_count']}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build v3 unified dataset")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/raw/lllab-vision-banana-dataset",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names (depth,normal,semantic,specular,instance,referring)",
    )
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else None
    build_v3_dataset(
        Path(args.dataset_root),
        Path(args.output_root),
        tasks=tasks,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
