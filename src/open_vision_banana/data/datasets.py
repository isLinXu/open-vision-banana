"""Dataset loaders for Vision Banana instruction tuning.

Each loader produces (prompt, image, target_rgb) triples where:
- prompt: task-specific instruction text
- image: input RGB image tensor [3, H, W] in [-1, 1]
- target_rgb: ground-truth encoded RGB visualization [3, H, W] in [0, 1]
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..core.depth_mapper import DepthRGBMapper
from ..core.normal_mapper import NormalRGBMapper
from ..inference.prompt import PromptBuilder
from ..task_prompts import (
    CITYSCAPES_RGB_LEGEND,
    SEMANTIC_SEGMENTATION_PROMPT,
    normalize_task_prompt,
)
from .benchmark_io import (
    find_matching_file,
    load_eth3d_depth,
    load_normal_map,
    load_scalar,
    resize_normal,
    resize_scalar,
)

logger = logging.getLogger(__name__)


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor [C, H, W] without torchvision dependency."""
    import numpy as np

    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return torch.from_numpy(arr.transpose(2, 0, 1)).clone()


def _fill_polygon(mask: np.ndarray, pts: np.ndarray) -> None:
    """Scanline polygon fill — pure NumPy, no cv2 required.

    Args:
        mask: uint8 2-D array [H, W] to fill in-place.
        pts:  float32 array [N, 2] of (x, y) polygon vertices.
    """
    import numpy as np

    if len(pts) < 3:
        return

    h, w = mask.shape
    xs, ys = pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32)
    y_min = max(int(np.floor(ys.min())), 0)
    y_max = min(int(np.ceil(ys.max())), h - 1)

    n = len(pts)
    for y in range(y_min, y_max + 1):
        intersections = []
        for i in range(n):
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[(i + 1) % n], ys[(i + 1) % n]
            if ((y0 <= y < y1) or (y1 <= y < y0)) and y1 != y0:
                xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                intersections.append(xi)
        intersections.sort()
        for k in range(0, len(intersections) - 1, 2):
            x_start = max(int(np.ceil(intersections[k])), 0)
            x_end = min(int(np.floor(intersections[k + 1])), w - 1)
            if x_start <= x_end:
                mask[y, x_start : x_end + 1] = 1


def _init_native_normal(dataset: object, native_gt: bool) -> None:
    dataset.native_gt = bool(native_gt)


def _pack_normal_sample(dataset, normal: torch.Tensor, valid: torch.Tensor) -> dict:
    """Resize a native normal map to the model canvas, keeping native GT on request."""
    native: dict = {}
    if getattr(dataset, "native_gt", False):
        native = {
            "gt_normal_native": torch.where(valid.unsqueeze(0), normal, torch.zeros_like(normal)),
            "valid_mask_native": valid,
        }
    resized = resize_normal(normal, dataset.resolution)
    valid_resized = resize_scalar(valid.float().unsqueeze(0), dataset.resolution).squeeze(0) > 0.5
    resized = torch.where(valid_resized.unsqueeze(0), resized, torch.zeros_like(resized))
    target_rgb = dataset.normal_mapper.normal_to_rgb(resized.unsqueeze(0)).squeeze(0)
    return {
        "prompt": PromptBuilder.build_normal_prompt(),
        "target_rgb": target_rgb,
        "gt_normal": resized,
        "valid_mask": valid_resized,
        **native,
    }


def _color_to_normal_path(path: Path) -> Path:
    """Map a ScanNet color frame to its normal frame.

    Directory components are rewritten only when the component itself is
    ``color`` / ``colors``. The filename may contain ``color`` as a substring
    (``scene_0_color.png``). Parent paths that merely contain those letters
    are left alone.
    """
    parts: list[str] = []
    for index, part in enumerate(path.parts):
        if index == len(path.parts) - 1:
            parts.append(part.replace("color", "normal"))
        elif part == "color":
            parts.append("normal")
        elif part == "colors":
            parts.append("normals")
        else:
            parts.append(part)
    return Path(*parts)


def _nyu_depth_dir(parent: Path) -> Path | None:
    """Accept ``depths/`` and the ``depth/`` name used by the NYU test export."""
    for name in ("depths", "depth"):
        candidate = parent / name
        if candidate.is_dir():
            return candidate
    return None


def _load_normal_mask(mask_path: Path | None, height: int, width: int) -> torch.Tensor | None:
    if mask_path is None:
        return None
    mask = torch.from_numpy(np.load(mask_path)).reshape(height, width)
    return mask.bool()


class BaseVisionDataset(Dataset):
    """Abstract base class for vision task datasets."""

    TASK_TYPE: str = "unknown"

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        resolution: int = 512,
        task_mix_weight: float = 1.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.resolution = resolution
        self.task_mix_weight = task_mix_weight
        self.samples: list[dict] = []
        self._scan()

    def _scan(self) -> None:
        """Scan for data samples. Override in subclasses."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str | Path) -> torch.Tensor:
        """Load image as tensor [3, H, W] in [-1, 1]."""
        with Image.open(path) as source:
            img = source.convert("RGB")
        if img.size[0] != self.resolution or img.size[1] != self.resolution:
            img = img.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        tensor = _pil_to_tensor(img).float() / 255.0
        return tensor * 2.0 - 1.0

    def _tag_task(self, sample: dict) -> dict:
        """Attach the TASK_TYPE to the sample dict for downstream use."""
        sample["task"] = self.TASK_TYPE
        return sample


class HypersimDepthDataset(BaseVisionDataset):
    """Hypersim dataset for metric depth estimation."""

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        self.prepared_manifest: dict | None = None
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"Hypersim directory not found: {self.root_dir}")
            return
        prepared_images = self.root_dir / "images"
        prepared_metric = self.root_dir / "metric_depth"
        prepared_targets = self.root_dir / "targets"
        if prepared_images.is_dir() and prepared_metric.is_dir() and prepared_targets.is_dir():
            manifest_path = self.root_dir / "manifest.json"
            if manifest_path.is_file():
                try:
                    import json

                    self.prepared_manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    logger.warning("Could not read Hypersim manifest %s: %s", manifest_path, exc)
            for img_path in sorted(prepared_images.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                metric_path = prepared_metric / f"{img_path.stem}.npy"
                target_path = prepared_targets / f"{img_path.stem}.png"
                if metric_path.is_file() and target_path.is_file():
                    self.samples.append(
                        {
                            "image": img_path,
                            "target": target_path,
                            "metric_depth": metric_path,
                        }
                    )
            if self.samples:
                return
        for img_path in sorted(self.root_dir.rglob("*color*")):
            depth_path = Path(str(img_path).replace("color", "depth"))
            if depth_path.exists():
                self.samples.append({"image": img_path, "target": depth_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        if "metric_depth" in sample:
            depth_arr = np.load(sample["metric_depth"], allow_pickle=False)
            if depth_arr.ndim != 2:
                raise ValueError(
                    f"Prepared metric depth must be 2-D, got {depth_arr.shape} "
                    f"for {sample['metric_depth']}"
                )
            depth = torch.from_numpy(np.ascontiguousarray(depth_arr)).float().unsqueeze(0)
            with Image.open(sample["target"]) as source:
                target_img = source.convert("RGB")
            if target_img.size != (self.resolution, self.resolution):
                target_img = target_img.resize(
                    (self.resolution, self.resolution), Image.Resampling.LANCZOS
                )
            target_rgb = _pil_to_tensor(target_img).float() / 255.0
        else:
            depth_img = Image.open(sample["target"])
            depth = _pil_to_tensor(depth_img).float()
            if depth.max() > 1.0:
                depth = depth / 1000.0
            target_rgb = None

        # depth is [1, H, W] → add batch dim for interpolate
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        valid_mask = torch.isfinite(depth) & (depth > 0)
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))
        if target_rgb is None:
            target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "gt_depth": depth,
            "valid_mask": valid_mask.squeeze(0),
        }


class ScanNetNormalDataset(BaseVisionDataset):
    """ScanNet surface-normal adapter.

    The official ScanNet release is reconstructed meshes, not per-frame normal
    maps. This adapter reads an already-rendered camera-space normal directory:

    * ``images/`` + ``normals/`` (optionally under ``<split>/``), or
    * a ``color`` path whose sibling is obtained by replacing ``color`` with
      ``normal`` (``.png`` / ``.npy``).

    PNG maps must already use the paper frame. Normals are not derived from depth.
    """

    TASK_TYPE = "normal"

    def __init__(self, root_dir, split="train", resolution=512, native_gt=False, **kwargs) -> None:
        self.normal_mapper = NormalRGBMapper()
        _init_native_normal(self, native_gt)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"ScanNet directory not found: {self.root_dir}")
            return
        paired = self._scan_image_normal_dirs()
        if paired:
            self.samples.extend(paired)
            return
        for img_path in sorted(self.root_dir.rglob("*color*")):
            if not img_path.is_file() or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            normal_path = _color_to_normal_path(img_path)
            if not normal_path.is_file():
                for suffix in (".png", ".npy"):
                    candidate = normal_path.with_suffix(suffix)
                    if candidate.is_file():
                        normal_path = candidate
                        break
            if normal_path.is_file() and normal_path.resolve() != img_path.resolve():
                self.samples.append({"image": img_path, "target": normal_path})

    def _scan_image_normal_dirs(self) -> list[dict]:
        layouts = (
            (self.root_dir / self.split / "images", self.root_dir / self.split / "normals"),
            (self.root_dir / "images" / self.split, self.root_dir / "normals" / self.split),
            (self.root_dir / "images", self.root_dir / "normals"),
        )
        for image_dir, normal_dir in layouts:
            if not image_dir.is_dir() or not normal_dir.is_dir():
                continue
            pairs = []
            for image_path in sorted(image_dir.rglob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                normal_path = find_matching_file(normal_dir, image_path.stem, (".npy", ".png"))
                if normal_path is not None:
                    pairs.append({"image": image_path, "target": normal_path})
            if pairs:
                return pairs
        return []

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        normal, valid = load_normal_map(sample["target"])
        return {
            "image": self._load_image(sample["image"]),
            **_pack_normal_sample(self, normal, valid),
        }


class CityscapesSegDataset(BaseVisionDataset):
    """Cityscapes dataset for semantic segmentation."""

    TASK_TYPE = "semantic"

    CITYSCAPES_COLORS = dict(CITYSCAPES_RGB_LEGEND)

    # Official Cityscapes labelIds -> trainIds mapping. Labels not listed are
    # ignored during evaluation, including license plate and void regions.
    CITYSCAPES_LABEL_TO_TRAIN = {
        7: 0,
        8: 1,
        11: 2,
        12: 3,
        13: 4,
        17: 5,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        31: 16,
        32: 17,
        33: 18,
    }

    def __init__(self, root_dir, split="train", resolution=512, native_gt=False, **kwargs) -> None:
        # ``native_gt=True`` additionally returns ``gt_labels_native`` (unresized trainIds)
        # so mIoU can be scored at the 2048x1024 GT resolution instead of the model canvas.
        self.native_gt = bool(native_gt)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        img_dir = self.root_dir / "leftImg8bit" / self.split
        if not img_dir.exists():
            logger.warning(f"Cityscapes directory not found: {img_dir}")
            return
        for img_path in sorted(img_dir.rglob("*.png")):
            gt_name = img_path.name.replace("leftImg8bit", "gtFine_labelIds")
            gt_path = (
                self.root_dir
                / "gtFine"
                / self.split
                / img_path.relative_to(img_dir).parent
                / gt_name
            )
            if gt_path.exists():
                self.samples.append({"image": img_path, "target": gt_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        with Image.open(sample["target"]) as label_img:
            label = _pil_to_tensor(label_img).long().squeeze(0)
        mapped = torch.full_like(label, 255)
        for label_id, train_id in self.CITYSCAPES_LABEL_TO_TRAIN.items():
            mapped[label == label_id] = train_id
        label = mapped
        native_label = label.clone()
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = (
            torch.nn.functional.interpolate(
                label, size=(self.resolution, self.resolution), mode="nearest"
            )
            .long()
            .squeeze(0)
            .squeeze(0)
        )

        target_rgb = self._label_to_rgb(label)
        prompt = PromptBuilder.build_semantic_prompt(self.CITYSCAPES_COLORS)
        item = {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "gt_labels": label,
        }
        if self.native_gt:
            item["gt_labels_native"] = native_label
        return item

    # Build a [num_classes, 3] color palette tensor at class level (lazy, cached).
    _COLOR_PALETTE: torch.Tensor | None = None

    @classmethod
    def _get_palette(cls) -> torch.Tensor:
        """Return cached [C, 3] float32 palette tensor (values in [0, 1])."""
        if cls._COLOR_PALETTE is None:
            colors = list(cls.CITYSCAPES_COLORS.values())
            cls._COLOR_PALETTE = torch.tensor(colors, dtype=torch.float32) / 255.0
        return cls._COLOR_PALETTE

    def _label_to_rgb(self, label: torch.Tensor) -> torch.Tensor:
        """Vectorized label → RGB using palette gather.

        Official ignore / void (trainId 255, and any index outside 0–18) stays
        black. Clamping those pixels onto bicycle (class 18) was wrong: it
        painted unlabeled regions as a real class.
        """
        palette = self._get_palette().to(label.device)  # [C, 3]
        n_cls = palette.shape[0]
        flat = label.reshape(-1).long()
        valid = (flat >= 0) & (flat < n_cls)
        rgb_flat = torch.zeros(flat.shape[0], 3, device=label.device, dtype=palette.dtype)
        rgb_flat[valid] = palette[flat[valid]]
        H, W = label.shape
        return rgb_flat.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]


class SACoInstanceDataset(BaseVisionDataset):
    """SA-Co / SA-Co-Gold instance queries.

    Accepts either the training-style ``*color*`` / ``*instance*`` pairs or a
    Gold query JSONL with noun phrases and optional presence labels::

        root/queries.jsonl
        root/val/queries.jsonl
        root/gold.jsonl

    Each JSONL row is ``{"image": "...", "query": "a red car", "present": true,
    "mask": "optional.png"}``. The paper's cgF1 uses Gemini 3.1 Flash-Lite to
    decide presence; this adapter only loads the official query set. Passing
    ``gt_present`` lets the evaluator compute IL_MCC without inventing a filter.
    """

    TASK_TYPE = "instance"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"SA-Co directory not found: {self.root_dir}")
            return
        for name in ("queries.jsonl", "gold.jsonl", f"{self.split}.jsonl"):
            for path in (self.root_dir / self.split / name, self.root_dir / name):
                if path.is_file():
                    self._scan_gold_queries(path)
                    if self.samples:
                        return
        for img_path in sorted(self.root_dir.rglob("*color*")):
            inst_path = Path(str(img_path).replace("color", "instance"))
            if inst_path.exists():
                self.samples.append({"image": img_path, "target": inst_path, "query": "objects"})

    def _scan_gold_queries(self, path: Path) -> None:
        import json

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            query = str(row.get("query") or row.get("noun_phrase") or row.get("text") or "").strip()
            image_name = row.get("image") or row.get("file_name")
            if not query or not image_name:
                continue
            image = Path(image_name)
            if not image.is_file():
                image = self.root_dir / image_name
            if not image.is_file():
                matches = sorted(self.root_dir.rglob(Path(image_name).name))
                image = matches[0] if matches else None
            if image is None:
                continue
            mask_name = row.get("mask") or row.get("target")
            mask = None
            if mask_name:
                mask = Path(mask_name)
                if not mask.is_file():
                    mask = self.root_dir / mask_name
                if not mask.is_file():
                    matches = sorted(self.root_dir.rglob(Path(mask_name).name))
                    mask = matches[0] if matches else None
            present = row.get("present")
            if present is None:
                present = row.get("gt_present")
            self.samples.append(
                {
                    "image": image,
                    "target": mask,
                    "query": query,
                    "present": None if present is None else bool(present),
                }
            )

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        query = str(sample.get("query") or "objects")
        if sample.get("target") is None:
            target_rgb = torch.zeros(3, self.resolution, self.resolution)
        else:
            inst_img = Image.open(sample["target"]).convert("RGB")
            if inst_img.size != (self.resolution, self.resolution):
                inst_img = inst_img.resize(
                    (self.resolution, self.resolution), Image.Resampling.LANCZOS
                )
            target_rgb = _pil_to_tensor(inst_img).float() / 255.0
        present = sample.get("present")
        if present is None:
            present = bool(target_rgb.sum() > 0)
        item = {
            "prompt": PromptBuilder.build_instance_prompt(query),
            "image": image,
            "target_rgb": target_rgb,
            "object_name": query,
            "gt_present": present,
        }
        return item


class RefCOCOgDataset(BaseVisionDataset):
    """RefCOCOg dataset for referring expression segmentation.

    Expected directory layout::

        root_dir/
          images/                 ← COCO images, or images/train2014, images/val2014
          refs(umd).p             ← paper split (preferred when protocol="umd")
          refs(g).p               ← Google split (legacy fallback only)
          instances.json          ← COCO instances; required for a real cIoU

    Each referring sentence is its own sample (standard cIoU counts every
    expression). Refs that record a ``split`` are kept only when it matches
    ``split``; by default refs with no split field are kept for small fixtures.
    ``strict=True`` is required by the official benchmark command: it requires
    the requested protocol, explicit splits, readable images, and nonempty COCO
    polygon/RLE masks rasterized with pycocotools at native image dimensions.
    Invalid selected records raise instead of silently changing the benchmark.
    The default keeps legacy fixture behavior (protocol fallback, omitted
    missing masks, and black masks without instances). Bounding boxes are never
    used as the ground-truth region.
    """

    TASK_TYPE = "referring"

    def __init__(
        self,
        root_dir,
        split="train",
        resolution=512,
        native_gt=False,
        protocol="umd",
        strict=False,
        **kwargs,
    ) -> None:
        if protocol not in {"umd", "google"}:
            raise ValueError(f"protocol must be 'umd' or 'google', got {protocol!r}")
        self.native_gt = bool(native_gt)
        self.protocol = protocol
        self.strict = bool(strict)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _annotation_file(self) -> Path | None:
        if self.strict:
            names = ("refs(umd).p",) if self.protocol == "umd" else ("refs(g).p", "refs(google).p")
            for name in names:
                path = self.root_dir / name
                if path.is_file():
                    return path
            raise FileNotFoundError(
                f"Strict RefCOCOg protocol={self.protocol!r} requires {' or '.join(names)} "
                f"under {self.root_dir}; fallback to another protocol is forbidden"
            )
        names = (
            ("refs(umd).p", "refs(g).p", "refs(google).p")
            if self.protocol == "umd"
            else ("refs(g).p", "refs(google).p", "refs(umd).p")
        )
        for name in names:
            path = self.root_dir / name
            if path.is_file():
                return path
        return None

    def _scan(self) -> None:
        if self.strict:
            self._scan_strict()
            return
        if not self.root_dir.exists():
            logger.warning(f"RefCOCOg directory not found: {self.root_dir}")
            return

        ann_file = self._annotation_file()
        if ann_file is None:
            logger.warning(f"RefCOCOg annotation not found under {self.root_dir}")
            return
        self._ann_file = ann_file

        instances_file = self.root_dir / "instances.json"
        self._image_hw: dict[int, tuple[int, int]] = {}
        self._ann_lookup: dict[int, dict] = {}
        if instances_file.exists():
            try:
                import json

                with open(instances_file) as f:
                    coco_data = json.load(f)
                self._ann_lookup = {ann["id"]: ann for ann in coco_data.get("annotations", [])}
                for image in coco_data.get("images", []):
                    if "height" in image and "width" in image:
                        self._image_hw[int(image["id"])] = (int(image["height"]), int(image["width"]))
            except Exception as e:
                logger.warning(f"Failed to load COCO instances: {e}")
                self._ann_lookup = {}
        require_mask = bool(self._ann_lookup)

        try:
            import pickle

            with open(ann_file, "rb") as f:
                refs = pickle.load(f)

            for ref in refs:
                ref_split = ref.get("split")
                if ref_split is not None and str(ref_split) != self.split:
                    continue
                image_id = ref.get("image_id", "")
                ann_id = ref.get("ann_id")
                if require_mask and (ann_id not in self._ann_lookup or not self._ann_lookup[ann_id].get("segmentation")):
                    continue
                expressions = self._expressions(ref)
                if not expressions:
                    continue
                img_path = self._find_image(image_id, ref.get("file_name"))
                if img_path is None:
                    continue
                for expr in expressions:
                    self.samples.append(
                        {
                            "image": img_path,
                            "expression": expr,
                            "ann_id": ann_id,
                            "image_id": image_id,
                        }
                    )

        except Exception as e:
            logger.warning(f"Failed to load RefCOCOg annotations: {e}")

    def _scan_strict(self) -> None:
        """Validate the full selected split before running an official benchmark."""
        import json
        import pickle

        ann_file = self._annotation_file()
        self._ann_file = ann_file
        instances_file = self.root_dir / "instances.json"
        if not instances_file.is_file():
            raise FileNotFoundError(f"Strict RefCOCOg requires {instances_file}")
        with instances_file.open(encoding="utf-8") as handle:
            coco_data = json.load(handle)
        if not isinstance(coco_data, dict):
            raise ValueError(f"Invalid COCO instances object in {instances_file}")
        self._ann_lookup = {}
        image_lookup: dict[int, dict] = {}
        for key, lookup in (("annotations", self._ann_lookup), ("images", image_lookup)):
            records = coco_data.get(key)
            if not isinstance(records, list) or not records:
                raise ValueError(f"Strict RefCOCOg requires nonempty COCO {key} in {instances_file}")
            for record in records:
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("id"), int)
                    or isinstance(record["id"], bool)
                ):
                    raise ValueError(f"Invalid COCO {key} record in {instances_file}")
                if record["id"] in lookup:
                    raise ValueError(f"Duplicate COCO {key} id={record['id']} in {instances_file}")
                lookup[record["id"]] = record
        self._image_hw = {}
        with ann_file.open("rb") as handle:
            refs = pickle.load(handle)
        if not isinstance(refs, list):
            raise ValueError(f"Expected a list of RefCOCOg refs in {ann_file}")
        validated_images: dict[int, Path] = {}
        validated_annotations: set[int] = set()
        for index, ref in enumerate(refs):
            if not isinstance(ref, dict) or not isinstance(ref.get("split"), str) or not ref["split"]:
                raise ValueError(f"RefCOCOg ref {index} requires an explicit nonempty split in {ann_file}")
            if ref["split"] != self.split:
                continue
            image_id, ann_id = ref.get("image_id"), ref.get("ann_id")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in (image_id, ann_id)):
                raise ValueError(f"RefCOCOg ref {index}: image_id and ann_id must be integers")
            if image_id not in image_lookup:
                raise ValueError(f"RefCOCOg ref {index}: missing COCO image_id={image_id}")
            ann = self._ann_lookup.get(ann_id)
            if ann is None:
                raise ValueError(f"RefCOCOg ref {index}: missing COCO ann_id={ann_id}")
            if ann.get("image_id") != image_id:
                raise ValueError(f"RefCOCOg ann_id={ann_id}: annotation image_id does not match ref")
            expressions = self._strict_expressions(ref, index)
            if image_id not in validated_images:
                info = image_lookup[image_id]
                height, width = info.get("height"), info.get("width")
                if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (height, width)):
                    raise ValueError(f"RefCOCOg image_id={image_id}: invalid native image dimensions")
                img_path = self._find_image(image_id, info.get("file_name") or ref.get("file_name"))
                if img_path is None:
                    raise FileNotFoundError(f"RefCOCOg image_id={image_id}: image not found under {self.root_dir}")
                try:
                    with Image.open(img_path) as source:
                        source.load()
                        if source.size != (width, height):
                            raise ValueError(f"image size {source.size} differs from COCO size {(width, height)}")
                except (OSError, ValueError) as exc:
                    raise ValueError(f"RefCOCOg image_id={image_id}: invalid image {img_path}: {exc}") from exc
                self._image_hw[image_id] = (height, width)
                validated_images[image_id] = img_path
            if ann_id not in validated_annotations:
                self._decode_mask(ann_id, image_id)
                validated_annotations.add(ann_id)
            for expression in expressions:
                self.samples.append({
                    "image": validated_images[image_id],
                    "expression": expression,
                    "ann_id": ann_id,
                    "image_id": image_id,
                })
        if not self.samples:
            raise ValueError(f"No RefCOCOg samples for split={self.split!r} in {ann_file}")

    @staticmethod
    def _strict_expressions(ref: dict, index: int) -> list[str]:
        raw = ref.get("sentences", [ref.get("sent")])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"RefCOCOg ref {index}: missing expressions")
        texts = []
        for value in raw:
            if isinstance(value, dict):
                value = value.get("raw") or value.get("sent")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RefCOCOg ref {index}: empty or invalid expression")
            texts.append(value.strip())
        return texts

    @staticmethod
    def _expressions(ref: dict) -> list[str]:
        raw = ref.get("sentences", [ref.get("sent", "")])
        if isinstance(raw, str):
            raw = [raw]
        texts: list[str] = []
        for expr in raw:
            if isinstance(expr, dict):
                expr = expr.get("raw") or expr.get("sent") or ""
            text = str(expr).strip()
            if text:
                texts.append(text)
        return texts

    def _find_image(self, image_id, file_name: str | None) -> Path | None:
        names: list[str] = []
        if file_name:
            names.append(Path(file_name).name)
        names.append(f"{image_id}.jpg")
        try:
            number = int(image_id)
        except (TypeError, ValueError):
            number = None
        if number is not None:
            names.extend(
                [
                    f"COCO_train2014_{number:012d}.jpg",
                    f"COCO_val2014_{number:012d}.jpg",
                    f"{number:012d}.jpg",
                ]
            )
        folders = (
            self.root_dir / "images",
            self.root_dir / "images" / "train2014",
            self.root_dir / "images" / "val2014",
            self.root_dir,
        )
        for folder in folders:
            for name in names:
                path = folder / name
                if path.is_file():
                    return path
        return None

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        expression = sample["expression"]
        native = self._decode_mask(sample.get("ann_id"), sample.get("image_id"))
        if native is None:
            target_rgb = torch.zeros(3, self.resolution, self.resolution)
            gt_mask = torch.zeros(self.resolution, self.resolution, dtype=torch.bool)
        else:
            mask = torch.from_numpy(np.ascontiguousarray(native)).float().view(1, 1, *native.shape)
            resized = (
                torch.nn.functional.interpolate(mask, size=(self.resolution, self.resolution), mode="nearest")
                .squeeze(0)
                .squeeze(0)
            )
            gt_mask = resized > 0.5
            target_rgb = gt_mask.float().unsqueeze(0).expand(3, -1, -1).clone()
        item = {
            "prompt": PromptBuilder.build_referring_prompt(expression),
            "image": image,
            "target_rgb": target_rgb,
            "gt_mask": gt_mask,
            "expression": expression,
        }
        if self.strict:
            # Match the portable RES adapter's explicit input range. Bright
            # signed tensors otherwise look like [0,1] to generator heuristics.
            item["image_range"] = "minus_one_one"
        if self.native_gt and native is not None:
            item["gt_mask_native"] = torch.from_numpy(np.ascontiguousarray(native)).bool()
        elif self.native_gt:
            item["gt_mask_native"] = gt_mask
        return item

    def _load_mask(self, ann_id: int | None) -> torch.Tensor:
        """Decode segmentation mask for a COCO annotation.

        Returns a [3, H, W] tensor in [0, 1] where white is the referred
        region. An unavailable annotation is all black. Bounding boxes are
        not painted as the region.
        """
        native = self._decode_mask(ann_id, None)
        if native is None:
            return torch.zeros(3, self.resolution, self.resolution)
        mask = torch.from_numpy(np.ascontiguousarray(native)).float().view(1, 1, *native.shape)
        resized = (
            torch.nn.functional.interpolate(mask, size=(self.resolution, self.resolution), mode="nearest")
            .squeeze(0)
            .squeeze(0)
        )
        return resized.unsqueeze(0).expand(3, -1, -1).clone()

    def _decode_mask(self, ann_id: int | None, image_id) -> np.ndarray | None:
        if self.strict:
            return self._decode_mask_strict(ann_id, image_id)
        if ann_id is None or ann_id not in getattr(self, "_ann_lookup", {}):
            return None
        ann = self._ann_lookup[ann_id]
        seg = ann.get("segmentation")
        if not seg:
            return None
        try:
            height, width = self._mask_hw(ann, image_id, seg)
            if isinstance(seg, list):
                mask = np.zeros((height, width), dtype=np.uint8)
                try:
                    import cv2

                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                except ImportError:
                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        _fill_polygon(mask, pts)
                return mask
            if isinstance(seg, dict):
                try:
                    from pycocotools import mask as coco_mask

                    decoded = coco_mask.decode(seg)
                    if decoded.ndim == 3:
                        decoded = decoded[..., 0]
                    return np.ascontiguousarray(decoded)
                except ImportError:
                    counts = seg.get("counts", [])
                    size = seg.get("size", [height, width])
                    if not isinstance(counts, list):
                        return None
                    flat = np.zeros(int(size[0]) * int(size[1]), dtype=np.uint8)
                    idx = 0
                    val = 0
                    for count in counts:
                        flat[idx : idx + int(count)] = val
                        idx += int(count)
                        val = 1 - val
                    # COCO uncompressed RLE is column-major; size is [height, width].
                    return flat.reshape((int(size[0]), int(size[1])), order="F")
        except Exception as e:
            logger.debug(f"Mask decode failed for ann_id={ann_id}: {e}")
        return None

    def _decode_mask_strict(self, ann_id: int | None, image_id) -> np.ndarray:
        from ..referring_masks import decode_coco_segmentation

        ann = self._ann_lookup.get(ann_id)
        if ann is None:
            raise ValueError(f"RefCOCOg: missing COCO ann_id={ann_id}")
        lookup_id = image_id if image_id is not None else ann.get("image_id")
        if lookup_id not in self._image_hw:
            raise ValueError(f"RefCOCOg ann_id={ann_id}: missing verified native image dimensions")
        height, width = self._image_hw[lookup_id]
        try:
            return decode_coco_segmentation(ann.get("segmentation"), height, width)
        except ValueError as exc:
            raise ValueError(f"RefCOCOg ann_id={ann_id}: {exc}") from exc

    def _mask_hw(self, ann: dict, image_id, seg) -> tuple[int, int]:
        if isinstance(seg, dict) and isinstance(seg.get("size"), (list, tuple)) and len(seg["size"]) == 2:
            return int(seg["size"][0]), int(seg["size"][1])
        lookup_id = image_id if image_id is not None else ann.get("image_id")
        try:
            keyed = int(lookup_id)
        except (TypeError, ValueError):
            keyed = None
        if keyed is not None and keyed in getattr(self, "_image_hw", {}):
            return self._image_hw[keyed]
        if "height" in ann and "width" in ann:
            return int(ann["height"]), int(ann["width"])
        bbox = ann.get("bbox", [0, 0, self.resolution, self.resolution])
        return int(bbox[1] + bbox[3]) + 1, int(bbox[0] + bbox[2]) + 1


class _NativeGTMixin:
    """Optional unresized ground truth for native-resolution benchmark scoring.

    With ``native_gt=True`` a depth dataset additionally returns
    ``gt_depth_native`` ``[1, H, W]`` and ``valid_mask_native`` ``[H, W]`` at the
    original resolution, using the same validity rule as the resized tensors.
    ``VisionEvaluator.evaluate(native_resolution=True)`` consumes them.
    """

    native_gt: bool = False

    def _init_native(self, native_gt: bool) -> None:
        self.native_gt = bool(native_gt)

    def _native_fields(self, depth: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
        if not self.native_gt:
            return {}
        if valid is None:
            valid = torch.isfinite(depth) & (depth > 0)
        return {
            "gt_depth_native": torch.where(valid, depth, torch.zeros_like(depth)),
            "valid_mask_native": valid.squeeze(0),
        }


class NYUDepthDataset(_NativeGTMixin, BaseVisionDataset):
    """NYU Depth v2 dataset for indoor metric depth estimation.

    The official NYU Depth v2 dataset contains 1449 RGB-D pairs with dense
    depth labels. The standard split is 795 train / 654 test images.

    Expected directory structure::

        root_dir/
          images/          ← RGB images (*.jpg)
          depths/          ← Depth maps (*.png or *.npy)

    Depth values are in meters (uint16 PNG: depth_mm / 1000.0).
    """

    TASK_TYPE = "depth"
    OFFICIAL_TRAIN_COUNT = 795
    OFFICIAL_TEST_COUNT = 654

    def __init__(self, root_dir, split="train", resolution=512, native_gt=False, **kwargs) -> None:
        """``native_gt=True`` additionally returns the unresized depth and
        validity mask (``gt_depth_native`` / ``valid_mask_native``) so the
        evaluator can score at the original resolution."""
        self.depth_mapper = DepthRGBMapper()
        self._init_native(native_gt)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        # Official exports often retain one HDF5 file per frame under
        # ``train/official`` and ``test/official``.  Keep the split boundary
        # in the directory path and decode those files lazily in __getitem__.
        split_root = self.root_dir / self.split
        h5_paths = sorted(split_root.rglob("*.h5")) + sorted(split_root.rglob("*.hdf5"))
        if h5_paths:
            self.samples.extend({"image": path, "target": path, "format": "nyu_h5"} for path in h5_paths)
            return
        # Prefer an explicit official split layout.  The extracted NYU test
        # archive is flat (images/ + depths/ or depth/) and must never be reused as train.
        split_root = self.root_dir / self.split
        split_depth = _nyu_depth_dir(split_root)
        nested_depth = None
        for _name in ("depths", "depth"):
            _candidate = self.root_dir / _name / self.split
            if _candidate.is_dir():
                nested_depth = _candidate
                break
        flat_depth = _nyu_depth_dir(self.root_dir)
        if (split_root / "images").is_dir() and split_depth is not None:
            img_dir = split_root / "images"
            depth_dir = split_depth
            allowed_names: set[str] | None = None
        elif (self.root_dir / "images" / self.split).is_dir() and nested_depth is not None:
            img_dir = self.root_dir / "images" / self.split
            depth_dir = nested_depth
            allowed_names = None
        elif (self.root_dir / "images").is_dir() and flat_depth is not None:
            if self.split not in {"test", "val"}:
                logger.warning(
                    "NYU flat layout has no official train split; refusing to use "
                    "the extracted test set for split=%s: %s",
                    self.split,
                    self.root_dir,
                )
                return
            img_dir = self.root_dir / "images"
            depth_dir = flat_depth
            allowed_names = self._read_split_index()
            if allowed_names is None:
                image_count = sum(
                    1
                    for path in img_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if image_count > self.OFFICIAL_TEST_COUNT:
                    logger.warning(
                        "NYU flat layout contains %d frames without an official split index; "
                        "refusing ambiguous split=%s (expected %d test frames)",
                        image_count,
                        self.split,
                        self.OFFICIAL_TEST_COUNT,
                    )
                    return
        else:
            img_dir = self.root_dir / "images"
            depth_dir = self.root_dir / "depths"
            allowed_names = None
        if not img_dir.exists():
            logger.warning(f"NYU images directory not found: {img_dir}")
            return
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            if allowed_names is not None and img_path.name not in allowed_names:
                continue
            # Match depth file
            depth_name = img_path.stem + ".png"
            depth_path = depth_dir / depth_name
            if not depth_path.exists():
                depth_path = depth_dir / (img_path.stem + ".npy")
            if depth_path.exists():
                self.samples.append({"image": img_path, "target": depth_path})

    def _read_split_index(self) -> set[str] | None:
        """Read an optional split index for flat benchmark exports."""
        candidates = (
            self.root_dir / "splits" / f"{self.split}.txt",
            self.root_dir / f"{self.split}.txt",
            self.root_dir / f"{self.split}_files.txt",
        )
        for index_path in candidates:
            if not index_path.is_file():
                continue
            names: set[str] = set()
            for line in index_path.read_text(encoding="utf-8").splitlines():
                value = line.strip()
                if value and not value.startswith("#"):
                    names.add(Path(value).name)
            return names
        return None

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if sample.get("format") == "nyu_h5":
            try:
                import h5py
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("NYU .h5 loading requires h5py") from exc
            with h5py.File(sample["image"], "r") as handle:
                rgb = np.asarray(handle["rgb"])
                depth_array = np.asarray(handle["depth"])
            if rgb.ndim != 3 or rgb.shape[0] != 3:
                raise ValueError(f"NYU rgb must have shape [3,H,W], got {rgb.shape}")
            image_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
            image = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) * 2.0 - 1.0
            depth = torch.from_numpy(np.asarray(depth_array, dtype=np.float32)).unsqueeze(0)
            native = self._native_fields(depth)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0),
                size=(self.resolution, self.resolution),
                mode="nearest",
            ).squeeze(0)
            valid_mask = torch.isfinite(depth) & (depth > 0)
            depth = torch.where(valid_mask, depth, torch.zeros_like(depth))
            target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
            return {
                "prompt": PromptBuilder.build_depth_prompt(),
                "image": image,
                "target_rgb": target_rgb,
                "gt_depth": depth,
                "valid_mask": valid_mask.squeeze(0),
                **native,
            }
        image = self._load_image(sample["image"])

        depth_path = sample["target"]
        if depth_path.suffix == ".npy":
            depth = np.load(depth_path)
            depth = torch.from_numpy(depth).float().unsqueeze(0)
        else:
            depth_img = Image.open(depth_path)
            depth = _pil_to_tensor(depth_img).float()
            if depth.max() > 1.0:
                depth = depth / 1000.0  # uint16 mm to meters

        native = self._native_fields(depth)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        valid_mask = torch.isfinite(depth) & (depth > 0)
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))
        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "gt_depth": depth,
            "valid_mask": valid_mask.squeeze(0),
            **native,
        }



class DIODEDataset(_NativeGTMixin, BaseVisionDataset):
    """DIODE dataset for indoor/outdoor metric depth estimation.

    DIODE is the first public dataset with dense indoor and outdoor depth
    ground truth. Train split: 16,886 images. Val split: 3,325 images.

    Official layout (what the DIODE download produces)::

        root_dir/
          val/
            indoors/   (or outdoor/)
              scene_00019/scan_00183/
                00019_00183_indoors_000_010.png        ← RGB
                00019_00183_indoors_000_010_depth.npy  ← float32 [H, W, 1], metres
                00019_00183_indoors_000_010_depth_mask.npy ← [H, W], 1 = valid

    The earlier ``*_rgb.png`` / ``*_depth.npy`` naming is still accepted.

    ``scene_type`` restricts to ``"indoors"`` or ``"outdoor"`` scenes. The paper
    evaluates **DIODE-Indoor**, so benchmark runs pass ``scene_type="indoors"``.
    """

    TASK_TYPE = "depth"
    _SCENE_ALIASES = {
        "indoors": {"indoors", "indoor"},
        "indoor": {"indoors", "indoor"},
        "outdoor": {"outdoor", "outdoors"},
        "outdoors": {"outdoor", "outdoors"},
    }

    def __init__(
        self,
        root_dir,
        split="train",
        resolution=512,
        native_gt=False,
        scene_type: str | None = None,
        **kwargs,
    ) -> None:
        if scene_type is not None and scene_type.lower() not in self._SCENE_ALIASES:
            raise ValueError(f"scene_type must be one of {sorted(self._SCENE_ALIASES)}, got {scene_type!r}")
        self.depth_mapper = DepthRGBMapper()
        self._init_native(native_gt)
        self.scene_type = scene_type.lower() if scene_type else None
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            logger.warning(f"DIODE split directory not found: {split_dir}")
            return
        wanted = self._SCENE_ALIASES[self.scene_type] if self.scene_type else None
        for rgb_path in sorted(split_dir.rglob("*.png")):
            stem = rgb_path.name[: -len(".png")]
            base = stem[: -len("_rgb")] if stem.endswith("_rgb") else stem
            depth_path = rgb_path.with_name(f"{base}_depth.npy")
            if not depth_path.exists():
                continue
            if wanted is not None and not (wanted & {part.lower() for part in rgb_path.relative_to(split_dir).parts}):
                continue
            mask_path = rgb_path.with_name(f"{base}_depth_mask.npy")
            self.samples.append(
                {
                    "image": rgb_path,
                    "target": depth_path,
                    "mask": mask_path if mask_path.exists() else None,
                }
            )

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        import numpy as np

        depth = load_scalar(sample["target"])  # [1, H, W]; DIODE stores [H, W, 1]
        native_valid = (depth >= 0) & torch.isfinite(depth)
        native_mask = None
        if sample.get("mask") is not None:
            native_mask = torch.from_numpy(np.load(sample["mask"])).bool().reshape(1, *depth.shape[-2:])
            native_valid &= native_mask
        native = self._native_fields(depth, native_valid)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        # Mask invalid depth (DIODE uses 0 / mask for invalid)
        valid_mask = (depth >= 0) & torch.isfinite(depth)
        if native_mask is not None:
            mask = (
                torch.nn.functional.interpolate(
                    native_mask.unsqueeze(0).float(),
                    size=(self.resolution, self.resolution),
                    mode="nearest",
                )
                .squeeze(0)
                .bool()
            )
            valid_mask &= mask
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "gt_depth": depth,
            "valid_mask": valid_mask.squeeze(0),
            **native,
        }


class IBimsDepthDataset(NYUDepthDataset):
    """iBims-1 depth benchmark with flexible RGB/depth folder matching."""

    def _scan(self) -> None:
        image_dir = self.root_dir / "images"
        if not image_dir.exists():
            image_dir = self.root_dir / "rgb"
        depth_dir = self.root_dir / "depths"
        if not depth_dir.exists():
            depth_dir = self.root_dir / "depth"
        if not image_dir.exists() or not depth_dir.exists():
            logger.warning("iBims image/depth directories not found under %s", self.root_dir)
            return
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            depth_path = find_matching_file(
                depth_dir,
                image_path.stem,
                (".png", ".npy", ".pfm"),
            )
            if depth_path is not None:
                self.samples.append({"image": image_path, "target": depth_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        native_depth = load_scalar(
            sample["target"], scale=0.001 if sample["target"].suffix == ".png" else 1.0
        )
        native = self._native_fields(
            native_depth, torch.isfinite(native_depth) & (native_depth > 0) & (native_depth < 100)
        )
        depth = resize_scalar(native_depth, self.resolution)
        valid_mask = torch.isfinite(depth) & (depth > 0) & (depth < 100)
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))
        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        return {
            "prompt": PromptBuilder.build_depth_prompt(),
            "image": image,
            "target_rgb": target_rgb,
            "gt_depth": depth,
            "valid_mask": valid_mask.squeeze(0),
            **native,
        }


class NYUNormalDataset(BaseVisionDataset):
    """NYU surface-normal benchmark adapter.

    Official NYU Depth v2 H5 files contain RGB and depth only. This adapter
    reads a community ``normals/`` directory of camera-space maps (``.npy``
    vectors or paper-encoded PNG). It does not derive normals from depth.
    """

    TASK_TYPE = "normal"
    OFFICIAL_TEST_COUNT = 654

    def __init__(self, root_dir, split="val", resolution=512, native_gt=False, **kwargs) -> None:
        self.normal_mapper = NormalRGBMapper()
        _init_native_normal(self, native_gt)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        split_root = self.root_dir / self.split
        if (split_root / "images").is_dir() and (split_root / "normals").is_dir():
            image_dir = split_root / "images"
            normal_dir = split_root / "normals"
            allowed_names = None
        elif (self.root_dir / "images" / self.split).is_dir() and (
            self.root_dir / "normals" / self.split
        ).is_dir():
            image_dir = self.root_dir / "images" / self.split
            normal_dir = self.root_dir / "normals" / self.split
            allowed_names = None
        elif (self.root_dir / "images").is_dir() and (self.root_dir / "normals").is_dir():
            if self.split not in {"test", "val"}:
                logger.warning(
                    "NYU normal flat layout has no official train split; refusing "
                    "to use it for split=%s: %s",
                    self.split,
                    self.root_dir,
                )
                return
            image_dir = self.root_dir / "images"
            normal_dir = self.root_dir / "normals"
            allowed_names = self._read_split_index()
            if allowed_names is None:
                image_count = sum(
                    1
                    for path in image_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                if image_count > self.OFFICIAL_TEST_COUNT:
                    logger.warning(
                        "NYU normal flat layout contains %d frames without an official "
                        "split index; refusing ambiguous split=%s",
                        image_count,
                        self.split,
                    )
                    return
        else:
            image_dir = self.root_dir / "images"
            normal_dir = self.root_dir / "normals"
            allowed_names = None
        if not image_dir.exists() or not normal_dir.exists():
            logger.warning("NYU normal image/normal directories not found under %s", self.root_dir)
            return
        for image_path in sorted(image_dir.rglob("*")):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if allowed_names is not None and image_path.name not in allowed_names:
                continue
            normal_path = find_matching_file(normal_dir, image_path.stem, (".npy", ".png"))
            if normal_path is not None:
                self.samples.append({"image": image_path, "target": normal_path})

    def _read_split_index(self) -> set[str] | None:
        candidates = (
            self.root_dir / "splits" / f"{self.split}.txt",
            self.root_dir / f"{self.split}.txt",
            self.root_dir / f"{self.split}_files.txt",
        )
        for index_path in candidates:
            if not index_path.is_file():
                continue
            return {
                Path(line.strip()).name
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
        return None

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        normal, valid = load_normal_map(sample["target"])
        return {
            "image": self._load_image(sample["image"]),
            **_pack_normal_sample(self, normal, valid),
        }


class DIODENormalDataset(BaseVisionDataset):
    """DIODE surface-normal benchmark adapter.

    Official files sit next to the RGB frame::

        <stem>.png
        <stem>_normal.npy          ← float32 [H, W, 3], camera space
        <stem>_depth_mask.npy      ← optional, 1 = valid (also accepts _normal_mask)

    ``scene_type="indoors"`` keeps DIODE-Indoor, matching the depth benchmark.
    Zero-length normal vectors are invalid. Normals are not derived from depth.
    """

    TASK_TYPE = "normal"
    _SCENE_ALIASES = {
        "indoors": {"indoors", "indoor"},
        "indoor": {"indoors", "indoor"},
        "outdoor": {"outdoor", "outdoors"},
        "outdoors": {"outdoor", "outdoors"},
    }
    _SKIP_SUFFIXES = ("_depth", "_normal", "_mask", "_depth_mask", "_normal_mask")

    def __init__(
        self,
        root_dir,
        split="val",
        resolution=512,
        native_gt=False,
        scene_type: str | None = None,
        **kwargs,
    ) -> None:
        if scene_type is not None and scene_type.lower() not in self._SCENE_ALIASES:
            raise ValueError(f"scene_type must be one of {sorted(self._SCENE_ALIASES)}, got {scene_type!r}")
        self.normal_mapper = NormalRGBMapper()
        _init_native_normal(self, native_gt)
        self.scene_type = scene_type.lower() if scene_type else None
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        split_dir = self.root_dir / self.split
        if not split_dir.is_dir():
            logger.warning("DIODE normal split directory not found: %s", split_dir)
            return
        wanted = self._SCENE_ALIASES[self.scene_type] if self.scene_type else None
        for image_path in sorted(split_dir.rglob("*")):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = image_path.stem
            if stem.endswith(self._SKIP_SUFFIXES):
                continue
            base = stem[:-4] if stem.endswith("_rgb") else stem
            normal_path = image_path.with_name(f"{base}_normal.npy")
            if not normal_path.is_file():
                normal_path = image_path.with_name(f"{base}_normal.png")
            if not normal_path.is_file():
                alt = find_matching_file(image_path.parent, base, (".npy",))
                if alt is None or alt.resolve() == image_path.resolve():
                    continue
                normal_path = alt
            if wanted is not None and not (wanted & {part.lower() for part in image_path.relative_to(split_dir).parts}):
                continue
            mask_path = None
            for name in (f"{base}_normal_mask.npy", f"{base}_depth_mask.npy"):
                candidate = image_path.with_name(name)
                if candidate.is_file():
                    mask_path = candidate
                    break
            self.samples.append({"image": image_path, "target": normal_path, "mask": mask_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        normal, valid = load_normal_map(sample["target"])
        mask = _load_normal_mask(sample.get("mask"), normal.shape[-2], normal.shape[-1])
        if mask is not None:
            valid = valid & mask
            normal = torch.where(valid.unsqueeze(0), normal, torch.zeros_like(normal))
        return {
            "image": self._load_image(sample["image"]),
            **_pack_normal_sample(self, normal, valid),
        }


class VirtualKITTINormalDataset(DIODENormalDataset):
    """Virtual KITTI 2 surface-normal adapter (paper Table 7 "VKitti").

    VKITTI2 ships depth, not normals. Two layouts are accepted:

    * ``<scene>/frames/rgb/Camera_X/rgb_NNNNN.jpg`` with a sibling ``normal/`` or
      ``normals/`` directory of pre-computed maps in the paper frame (``.npy`` /
      paper-encoded PNG) — used as-is;
    * the stock layout with ``<scene>/frames/depth/Camera_X/depth_NNNNN.png``
      (uint16 **centimetres**, 65535 = sky/far) — normals are **derived from depth**
      with the per-frame intrinsics from ``<scene>/intrinsic.txt`` (falling back to
      the VKITTI2 default pinhole) via ``depth_geometry.depth_to_normals``; sky and
      depth-discontinuity pixels are invalid.
    """

    _FRAME_RE = re.compile(r"(\d+)$")

    def _scan(self) -> None:

        self._intrinsics_cache: dict[Path, dict] = {}
        for rgb_dir in sorted(self.root_dir.rglob("rgb")):
            normal_dir = rgb_dir.parent / "normal"
            if not normal_dir.exists():
                normal_dir = rgb_dir.parent / "normals"
            depth_dir = rgb_dir.parent / "depth"
            for image_path in sorted(rgb_dir.rglob("*")):
                if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if normal_dir.exists():
                    normal_path = find_matching_file(normal_dir, image_path.stem, (".png", ".npy"))
                    if normal_path is not None:
                        self.samples.append({"image": image_path, "target": normal_path})
                        continue
                if not depth_dir.exists():
                    continue
                camera = image_path.parent.name if image_path.parent != rgb_dir else ""
                depth_name = image_path.stem.replace("rgb", "depth", 1) + ".png"
                depth_path = depth_dir / camera / depth_name if camera else depth_dir / depth_name
                if not depth_path.is_file():
                    continue
                scene_dir = rgb_dir.parent.parent  # <scene>/<variation>/frames/rgb -> <scene>/<variation>
                intrinsic_file = scene_dir / "intrinsic.txt"
                frame_match = self._FRAME_RE.search(image_path.stem)
                cam_match = self._FRAME_RE.search(camera)
                self.samples.append(
                    {
                        "image": image_path,
                        "target": None,
                        "depth": depth_path,
                        "intrinsic_file": intrinsic_file if intrinsic_file.is_file() else None,
                        "frame": int(frame_match.group(1)) if frame_match else 0,
                        "camera": int(cam_match.group(1)) if cam_match else 0,
                    }
                )
        if self.samples and all(sample.get("target") is None for sample in self.samples):
            logger.info("VKITTI2: deriving normals from depth for %d frames (no normal/ directory found)", len(self.samples))

    def _intrinsics_for(self, sample: dict) -> tuple[float, float, float, float]:
        from .depth_geometry import VKITTI2_DEFAULT_INTRINSICS, read_vkitti_intrinsics

        path = sample.get("intrinsic_file")
        if path is None:
            return VKITTI2_DEFAULT_INTRINSICS
        table = self._intrinsics_cache.get(path)
        if table is None:
            table = read_vkitti_intrinsics(path)
            self._intrinsics_cache[path] = table
        return table.get((sample["frame"], sample["camera"]), VKITTI2_DEFAULT_INTRINSICS)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if sample.get("target") is not None:
            return super().__getitem__(idx)
        from .depth_geometry import depth_to_normals, load_vkitti_depth_png

        depth_m, depth_valid = load_vkitti_depth_png(sample["depth"])
        normal_np, valid_np = depth_to_normals(depth_m, *self._intrinsics_for(sample), valid=depth_valid)
        normal = torch.from_numpy(normal_np).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid_np)
        return {
            "image": self._load_image(sample["image"]),
            **_pack_normal_sample(self, normal, valid),
        }


class BlendedMVSDepthDataset(BaseVisionDataset):
    """BlendedMVS dataset for synthetic metric depth estimation.

    BlendedMVS provides **entirely synthetic depth data from Blender simulation
    engines** — zero real-world depth data. This makes it ideal for training
    depth estimation models when the paper mandates simulation-only depth.

    The dataset contains 502 scenes with rendered RGB images and dense
    ground-truth depth maps in PFM format. Some depth maps may be empty
    (incomplete textured mesh); these are automatically filtered.

    Expected directory structure (raw BlendedMVS layout)::

        root_dir/
          <PID>/
            blended_images/
              00000000.jpg
              00000001.jpg
              ...
            rendered_depth_maps/
              00000000.pfm
              00000001.pfm
              ...

    Or pre-processed layout (after prepare_blendedmvs.py)::

        root_dir/
          images/    ← RGB input images (*.png)
          targets/   ← Depth encoded as RGB via DepthRGBMapper

    Depth values are in metric meters (float32 PFM from simulation).
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    @staticmethod
    def _read_pfm(path: Path) -> np.ndarray | None:
        """Read a PFM depth file and return the depth array.

        PFM format (grayscale, little-endian):
            Pf
            <width> <height>
            <scale>   (negative = little-endian)
            <raw float32 data>
        """
        try:
            with open(path, "rb") as f:
                header = f.readline().strip()
                if header not in (b"Pf", b"PF"):
                    return None
                is_color = header == b"PF"

                dims = f.readline().strip()
                while dims.startswith(b"#") or dims == b"":
                    dims = f.readline().strip()
                width, height = map(int, dims.split())

                scale_line = f.readline().strip()
                scale = float(scale_line)
                endian = "<" if scale < 0 else ">"

                num_channels = 3 if is_color else 1
                num_floats = width * height * num_channels
                expected_bytes = num_floats * 4
                raw = f.read()
                if len(raw) < expected_bytes:
                    return None

                arr = np.frombuffer(raw, dtype=f"{endian}f4")
                if len(arr) != num_floats:
                    return None

                if is_color:
                    arr = arr.reshape((height, width, 3))[..., 0]
                else:
                    arr = arr.reshape((height, width))

                arr = np.flipud(arr).copy()  # PFM stores bottom-to-top
                return arr
        except Exception:
            return None

    def _scan_raw(self) -> None:
        """Scan raw BlendedMVS PID directory structure."""
        for pid_dir in sorted(self.root_dir.rglob("*")):
            if not pid_dir.is_dir():
                continue
            img_dir = pid_dir / "blended_images"
            depth_dir = pid_dir / "rendered_depth_maps"
            if not img_dir.exists() or not depth_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                if "_masked" in img_path.stem:
                    continue
                depth_path = depth_dir / (img_path.stem + ".pfm")
                if depth_path.exists():
                    self.samples.append({"image": img_path, "target": depth_path})

    def _scan_preprocessed(self) -> None:
        """Scan pre-processed images/ + targets/ layout."""
        img_dir = self.root_dir / "images"
        tgt_dir = self.root_dir / "targets"
        if not img_dir.exists() or not tgt_dir.exists():
            return
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            tgt_path = tgt_dir / (img_path.stem + ".png")
            if not tgt_path.exists():
                tgt_path = tgt_dir / (img_path.stem + ".jpg")
            if tgt_path.exists():
                self.samples.append({"image": img_path, "target": tgt_path})

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"BlendedMVS directory not found: {self.root_dir}")
            return
        # Prefer pre-processed layout if available
        if (self.root_dir / "images").exists() and (self.root_dir / "targets").exists():
            self._scan_preprocessed()
        else:
            self._scan_raw()
        if not self.samples:
            logger.warning(f"No BlendedMVS samples found in {self.root_dir}")

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        target_path = sample["target"]
        if target_path.suffix == ".pfm":
            # Raw PFM depth
            depth_arr = self._read_pfm(target_path)
            if depth_arr is None:
                # Fallback to zero depth on read failure
                depth = torch.zeros(1, self.resolution, self.resolution)
            else:
                depth = torch.from_numpy(depth_arr).float().unsqueeze(0)
                depth = torch.clamp(depth, min=0.0)
                # Resize to target resolution
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(0),
                    size=(self.resolution, self.resolution),
                    mode="nearest",
                ).squeeze(0)
            target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        else:
            # Pre-encoded RGB target
            target_img = Image.open(target_path).convert("RGB")
            if target_img.size != (self.resolution, self.resolution):
                target_img = target_img.resize(
                    (self.resolution, self.resolution), Image.Resampling.LANCZOS
                )
            target_rgb = _pil_to_tensor(target_img).float() / 255.0

        prompt = PromptBuilder.build_depth_prompt()
        return self._tag_task({"prompt": prompt, "image": image, "target_rgb": target_rgb})


class KITTIDataset(_NativeGTMixin, BaseVisionDataset):
    """KITTI dataset for autonomous driving depth estimation.

    KITTI provides 23,488 training and 1,000 test images with sparse LiDAR
    depth annotations. Standard evaluation uses the Eigen split.

    Expected directory structure (Eigen split preprocessed)::

        root_dir/
          train/
            rgb/      ← RGB images (*.jpg or *.png)
            depth/    ← Depth maps (*.npy or *.png, sparse LiDAR)
          val/
            rgb/
            depth/

    Depth values are in meters.
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, native_gt=False, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        self._init_native(native_gt)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        rgb_dir = self.root_dir / self.split / "rgb"
        depth_dir = self.root_dir / self.split / "depth"
        if rgb_dir.exists():
            for img_path in sorted(rgb_dir.rglob("*")):
                if img_path.suffix.lower() not in (".jpg", ".png", ".jpeg"):
                    continue
                depth_name = img_path.stem + ".npy"
                depth_path = depth_dir / depth_name
                if not depth_path.exists():
                    depth_path = depth_dir / (img_path.stem + ".png")
                if depth_path.exists():
                    self.samples.append({"image": img_path, "target": depth_path})
        if self.samples:
            return
        self._scan_official_annotated()
        if not self.samples:
            logger.warning(
                "KITTI RGB/depth pairs not found under %s (need val/rgb+depth "
                "or drive/image_02/data + proj_depth/groundtruth/image_02)",
                self.root_dir,
            )

    def _scan_official_annotated(self) -> None:
        """Pair KITTI raw RGB with the depth-completion annotated maps."""
        split_dir = self.root_dir / self.split
        if not split_dir.is_dir():
            split_dir = self.root_dir
        for drive in sorted(path for path in split_dir.glob("*_sync") if path.is_dir()):
            depth_dir = drive / "proj_depth" / "groundtruth" / "image_02"
            rgb_dir = drive / "image_02" / "data"
            if not depth_dir.is_dir() or not rgb_dir.is_dir():
                continue
            for depth_path in sorted(depth_dir.glob("*.png")):
                rgb_path = rgb_dir / depth_path.name
                if rgb_path.is_file():
                    self.samples.append({"image": rgb_path, "target": depth_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        depth_path = sample["target"]
        if depth_path.suffix == ".npy":
            import numpy as np

            depth = np.load(depth_path)
            depth = torch.from_numpy(depth).float().unsqueeze(0)
        else:
            depth_img = Image.open(depth_path)
            depth = _pil_to_tensor(depth_img).float()
            if depth.max() > 1.0:
                depth = depth / 256.0  # KITTI depth PNG scaling

        native = self._native_fields(depth, depth > 0.0)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        # KITTI has sparse depth (0 = invalid)
        valid_mask = depth > 0.0
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "gt_depth": depth,
            "valid_mask": valid_mask.squeeze(0),
            **native,
        }


class ADE20KDataset(BaseVisionDataset):
    """ADE20K dataset for semantic segmentation (scene parsing).

    ADE20K contains 20,210 training images with 150 semantic categories
    covering indoor and outdoor scenes.

    Expected directory structure::

        root_dir/
          images/
            training/  ← RGB images (*.jpg)
            validation/
          annotations/
            training/  ← Annotation PNGs (*.png, label indices)
            validation/

    Label indices range from 0-149 (0 = unlabeled/background).
    """

    TASK_TYPE = "semantic"

    ADE20K_COLORS: dict[int, tuple[int, int, int]] = {
        # Core 20 classes (full palette would be 150 colors)
        # Using a deterministic hash-based color assignment for all 150 classes
    }

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        # Build deterministic 150-class palette
        self._build_palette()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _build_palette(self):
        """Build deterministic 150-class color palette."""
        import random

        rng = random.Random(42)  # Deterministic seed
        colors = []
        for _ in range(150):
            colors.append((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        self._color_palette = torch.tensor(colors, dtype=torch.float32) / 255.0
        # Build class names
        self._class_names = [f"class_{i}" for i in range(150)]
        self._color_map = {f"class_{i}": colors[i] for i in range(150)}

    def _scan(self) -> None:
        img_dir = self.root_dir / "images" / ("training" if self.split == "train" else "validation")
        ann_dir = (
            self.root_dir / "annotations" / ("training" if self.split == "train" else "validation")
        )
        if not img_dir.exists():
            logger.warning(f"ADE20K images directory not found: {img_dir}")
            return
        for img_path in sorted(img_dir.rglob("*.jpg")):
            ann_name = img_path.name.replace(".jpg", ".png")
            ann_path = ann_dir / ann_name
            if ann_path.exists():
                self.samples.append({"image": img_path, "target": ann_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        label_img = Image.open(sample["target"])
        label = _pil_to_tensor(label_img).long().squeeze(0)
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = (
            torch.nn.functional.interpolate(
                label, size=(self.resolution, self.resolution), mode="nearest"
            )
            .long()
            .squeeze(0)
            .squeeze(0)
        )

        target_rgb = self._label_to_rgb(label)
        prompt = PromptBuilder.build_semantic_prompt(self._color_map)
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "class_color_map": self._color_map,
        }

    def _label_to_rgb(self, label: torch.Tensor) -> torch.Tensor:
        """Vectorized label → RGB using palette gather."""
        palette = self._color_palette.to(label.device)  # [150, 3]
        n_cls = palette.shape[0]
        flat = label.reshape(-1).clamp(0, n_cls - 1).long()  # [H*W]
        rgb_flat = palette[flat]  # [H*W, 3]
        H, W = label.shape
        return rgb_flat.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]


class SintelFlowDataset(BaseVisionDataset):
    """Sintel dataset for optical flow estimation.

    Sintel provides 1,064 training and 564 test frames with dense optical
    flow ground truth from animated scenes. Two versions: clean and final.

    Expected directory structure::

        root_dir/
          clean/  or  final/
            frames/          ← RGB frames (*.png)
            flow/            ← Flow files (*.flo or *.npy)

    Flow values are in pixels/frame (u, v) as float32.
    """

    TASK_TYPE = "flow"

    def __init__(self, root_dir, split="train", resolution=512, variant="clean", **kwargs) -> None:
        self.variant = variant  # "clean" or "final"
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        frames_dir = self.root_dir / self.variant / "frames"
        flow_dir = self.root_dir / self.variant / "flow"
        if not frames_dir.exists():
            logger.warning(f"Sintel frames directory not found: {frames_dir}")
            return
        for img_path in sorted(frames_dir.rglob("*.png")):
            # Match flow file
            flow_name = img_path.stem + ".flo"
            flow_path = flow_dir / flow_name
            if not flow_path.exists():
                flow_path = flow_dir / (img_path.stem + ".npy")
            if flow_path.exists():
                self.samples.append({"image": img_path, "target": flow_path})

    def _load_flo(self, path: Path) -> torch.Tensor:
        """Load .flo format optical flow file.

        .flo format:
        - Header: "PIEH" (4 bytes) + width (4 bytes) + height (4 bytes)
        - Data: float32 (u, v) pairs, width * height * 2 values
        """
        import numpy as np

        with open(path, "rb") as f:
            tag = np.fromfile(f, np.float32, count=1)[0]
            # Check tag = 202021.25 (PIEH in float32)
            if abs(tag - 202021.25) > 1e-6:
                # Try .npy fallback
                flow = np.load(path)
                return torch.from_numpy(flow).float().permute(2, 0, 1)  # [2, H, W]
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            flow = np.fromfile(f, np.float32, count=w * h * 2)
            flow = flow.reshape(h, w, 2)
        return torch.from_numpy(flow).float().permute(2, 0, 1)  # [2, H, W]

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        flow_path = sample["target"]
        if flow_path.suffix == ".npy":
            import numpy as np

            flow = np.load(flow_path)
            flow = torch.from_numpy(flow).float().permute(2, 0, 1)
        else:
            flow = self._load_flo(flow_path)

        # Resize flow to target resolution
        flow = torch.nn.functional.interpolate(
            flow.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Encode flow as RGB visualization (HSV-style)
        # For training target: encode flow as HSV and convert to RGB
        u, v = flow[0], flow[1]
        magnitude = torch.sqrt(u**2 + v**2 + 1e-10)
        direction = torch.atan2(v, u)  # [-π, π]

        # Normalize for visualization
        max_mag = magnitude.max() + 1e-10
        hue = (direction + math.pi) / (2 * math.pi)  # [0, 1]
        sat = magnitude / max_mag
        val = torch.ones_like(hue)

        # Simple HSV → RGB conversion (lightweight)
        target_rgb = torch.stack([hue, sat, val], dim=0)  # [3, H, W]
        target_rgb = target_rgb.clamp(0, 1)

        prompt = PromptBuilder.build_flow_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class ETH3DDataset(_NativeGTMixin, BaseVisionDataset):
    """ETH3D dataset for high-accuracy depth estimation.

    ETH3D provides 10 indoor/outdoor scenes with high-precision laser
    scanned depth ground truth. Standard benchmark for multi-view stereo
    and depth estimation.

    Expected directory structure (flat adapter layout)::

        root_dir/
          images/          ← RGB images (*.png, *.jpg)
          depths/          ← Depth maps (*.npy or *.png, in meters)

    Official ETH3D high-res DSLR layout is also accepted::

        root_dir/
          <scene>/
            images/dslr_images_undistorted/*.JPG
            ground_truth_depth/dslr_images/*.JPG   ← raw float32 metres

    Depth values are in meters (float32). Official dumps use inf for invalid
    pixels and have no image header.
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, native_gt=False, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        self._init_native(native_gt)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        self._scan_flat()
        if self.samples:
            return
        self._scan_official_scenes()
        if not self.samples:
            logger.warning("ETH3D images/depths not found under %s", self.root_dir)

    def _scan_flat(self) -> None:
        img_dir = self.root_dir / "images"
        depth_dir = self.root_dir / "depths"
        if not img_dir.exists():
            return
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            depth_name = img_path.stem + ".npy"
            depth_path = depth_dir / depth_name
            if not depth_path.exists():
                depth_path = depth_dir / (img_path.stem + ".png")
            if depth_path.exists():
                self.samples.append({"image": img_path, "target": depth_path})

    def _scan_official_scenes(self) -> None:
        if not self.root_dir.is_dir():
            return
        for scene in sorted(path for path in self.root_dir.iterdir() if path.is_dir()):
            depth_dir = scene / "ground_truth_depth" / "dslr_images"
            img_dir = scene / "images" / "dslr_images_undistorted"
            if not img_dir.is_dir():
                img_dir = scene / "images" / "dslr_images"
            if not depth_dir.is_dir() or not img_dir.is_dir():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                depth_path = find_matching_file(
                    depth_dir,
                    img_path.stem,
                    (img_path.suffix, ".JPG", ".jpg", ".png", ".npy"),
                )
                if depth_path is not None:
                    self.samples.append({"image": img_path, "target": depth_path})

    def _load_depth(self, depth_path: Path) -> torch.Tensor:
        if depth_path.suffix.lower() == ".npy":
            return load_scalar(depth_path)
        try:
            return load_scalar(depth_path)
        except (ValueError, OSError, Image.UnidentifiedImageError):
            return load_eth3d_depth(depth_path)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        depth = self._load_depth(Path(sample["target"]))

        native = self._native_fields(depth, (depth > 0.0) & (depth < 100.0))
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        valid_mask = (depth > 0.0) & (depth < 100.0)
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "gt_depth": depth,
            "valid_mask": valid_mask.squeeze(0),
            **native,
        }


class TartanAirDepthDataset(BaseVisionDataset):
    """TartanAir dataset for metric depth estimation from synthetic UE4 scenes.

    TartanAir provides entirely synthetic data from Unreal Engine 4 environments,
    including RGB images, metric depth maps, semantic segmentation, optical flow,
    and camera poses. This loader uses only the RGB and depth modalities,
    ensuring zero real-world depth data — satisfying the paper requirement for
    purely synthetic depth training data.

    Expected directory structure (after extraction from official zip files)::

        root_dir/
          ENV_NAME/
            Easy/
              P000/
                image_left/      ← 000000_left.png, 000001_left.png, ...
                depth_left/      ← 000000_left_depth.npy, 000001_left_depth.npy, ...
              P001/
                ...
            Hard/
              ...
          ENV_NAME_2/
            ...

    Depth values are metric meters (float32 .npy).
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"TartanAir directory not found: {self.root_dir}")
            return

        # Scan canonical TartanAir layout: image_left/ + depth_left/
        for img_dir in sorted(self.root_dir.rglob("image_left")):
            if not img_dir.is_dir():
                continue
            depth_dir = img_dir.parent / "depth_left"
            if not depth_dir.is_dir():
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                depth_name = img_path.stem + "_depth.npy"
                depth_path = depth_dir / depth_name
                if depth_path.exists():
                    self.samples.append({"image": img_path, "target": depth_path})

        logger.info("TartanAir: found %d samples under %s", len(self.samples), self.root_dir)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        # Load metric depth from .npy (float32, meters)
        import numpy as np

        depth_arr = np.load(sample["target"]).astype(np.float32)
        if depth_arr.ndim > 2:
            depth_arr = depth_arr.squeeze()
        depth = torch.from_numpy(depth_arr).float().unsqueeze(0)

        # Replace inf/NaN with 0
        depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
        depth = depth.clamp(min=0.0)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class OASISNormalDataset(BaseVisionDataset):
    """OASIS dataset for indoor surface normal estimation.

    OASIS provides 1.2M images with dense surface normal ground truth
    from indoor scenes. One of the largest indoor normal datasets.

    Expected directory structure::

        root_dir/
          images/          ← RGB images (*.jpg)
          normals/         ← Normal maps (*.npy or *.png, [-1,1]×3)

    Normal values are in [-1, 1]³ (unit vectors).
    """

    TASK_TYPE = "normal"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.normal_mapper = NormalRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        img_dir = self.root_dir / "images"
        normal_dir = self.root_dir / "normals"
        if not img_dir.exists():
            logger.warning(f"OASIS images directory not found: {img_dir}")
            return
        for img_path in sorted(img_dir.rglob("*.jpg")):
            normal_name = img_path.stem + ".npy"
            normal_path = normal_dir / normal_name
            if not normal_path.exists():
                normal_path = normal_dir / (img_path.stem + ".png")
            if normal_path.exists():
                self.samples.append({"image": img_path, "target": normal_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        normal_path = sample["target"]
        if normal_path.suffix == ".npy":
            import numpy as np

            normal = np.load(normal_path)
            normal = torch.from_numpy(normal).float().permute(2, 0, 1)
        else:
            normal_img = Image.open(normal_path)
            normal = _pil_to_tensor(normal_img).float()
            if normal.max() > 1.0:
                normal = normal / 127.5 - 1.0

        normal = torch.nn.functional.interpolate(
            normal.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        normal = normal / (normal.norm(dim=0, keepdim=True) + 1e-10)
        target_rgb = self.normal_mapper.normal_to_rgb(normal.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_normal_prompt()
        valid_mask = torch.isfinite(normal).all(dim=0) & (normal.norm(dim=0) > 1e-6)
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "gt_normal": normal,
            "valid_mask": valid_mask,
        }


class COCODetectionDataset(BaseVisionDataset):
    """COCO dataset for object detection and instance segmentation.

    COCO provides 118K training images with 80 object categories,
    bounding boxes, and instance segmentation masks.

    Expected directory structure (COCO 2017 format)::

        root_dir/
          images/
            train2017/     ← RGB images (*.jpg)
            val2017/
          annotations/
            instances_train2017.json
            instances_val2017.json

    Labels are 0-79 (80 classes, 0 = person, etc.).
    """

    TASK_TYPE = "object_detection"
    COCO_CLASSES = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
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
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
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
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self._build_palette()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _build_palette(self):
        import random

        rng = random.Random(2023)
        colors = []
        for _ in range(80):
            colors.append((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        self._color_palette = torch.tensor(colors, dtype=torch.float32) / 255.0

    def _scan(self) -> None:
        import json as json_mod

        ann_file = self.root_dir / "annotations" / f"instances_{self.split}2017.json"
        img_dir = self.root_dir / "images" / f"{self.split}2017"
        if not ann_file.exists():
            logger.warning(f"COCO annotations not found: {ann_file}")
            return
        if not img_dir.exists():
            logger.warning(f"COCO images directory not found: {img_dir}")
            return

        with open(ann_file) as f:
            coco = json_mod.load(f)

        img_to_anns: dict[str, list] = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        img_info = {img["id"]: img for img in coco["images"]}

        for img_id, anns in img_to_anns.items():
            if img_id not in img_info:
                continue
            img_path = img_dir / img_info[img_id]["file_name"]
            if img_path.exists():
                self.samples.append(
                    {
                        "image": img_path,
                        "annotations": anns,
                        "width": img_info[img_id]["width"],
                        "height": img_info[img_id]["height"],
                    }
                )

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        W, H = sample["width"], sample["height"]
        label_map = torch.zeros((H, W), dtype=torch.long)
        for ann in sample["annotations"]:
            if "segmentation" in ann and ann.get("iscrowd", 0) == 0:
                try:
                    from pycocotools import mask as mask_utils

                    if isinstance(ann["segmentation"], dict):
                        m = mask_utils.decode(ann["segmentation"])
                    else:
                        m = mask_utils.decode(mask_utils.frPyObjects(ann["segmentation"], H, W))
                    if m.ndim == 3:
                        m = m.max(axis=2)
                    label_map[m > 0] = ann["category_id"]
                except ImportError:
                    logger.warning("pycocotools not available, skipping COCO mask decode")
                    break

        label_map = label_map.unsqueeze(0).unsqueeze(0).float()
        label_map = (
            torch.nn.functional.interpolate(
                label_map, size=(self.resolution, self.resolution), mode="nearest"
            )
            .long()
            .squeeze(0)
            .squeeze(0)
        )

        unique_labels = label_map.unique()
        palette = self._color_palette.to(label_map.device)
        rgb = palette[label_map.clamp(0, 79).long()]
        target_rgb = rgb.permute(2, 0, 1)

        class_color_map = {}
        for cat_id in unique_labels.tolist():
            if 0 <= cat_id < 80:
                class_color_map[self.COCO_CLASSES[cat_id]] = tuple(
                    int(palette[cat_id][i].item() * 255) for i in range(3)
                )

        prompt = PromptBuilder.build_instance_prompt("objects")
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "class_color_map": class_color_map,
            "annotations": sample["annotations"],
        }


class COCOStuffDataset(BaseVisionDataset):
    """COCO-Stuff dataset for semantic segmentation (thing + stuff).

    COCO-Stuff extends COCO with 91 thing classes + 80 stuff classes
    (171 total). 118K training images with full pixel-wise annotations.

    Expected directory structure::

        root_dir/
          images/
            train2017/     ← RGB images (*.jpg)
          annotations/
            stuff_train2017.json

    Label indices: 0 = background, 1-90 = thing, 91-170 = stuff.
    """

    TASK_TYPE = "semantic"
    NUM_CLASSES = 171

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self._build_palette()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _build_palette(self):
        import random

        rng = random.Random(171)
        colors = []
        for _ in range(171):
            colors.append((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        self._color_palette = torch.tensor(colors, dtype=torch.float32) / 255.0

    def _scan(self) -> None:
        import json as json_mod

        ann_file = self.root_dir / "annotations" / f"stuff_{self.split}2017.json"
        img_dir = self.root_dir / "images" / f"{self.split}2017"
        if not ann_file.exists():
            logger.warning(f"COCO-Stuff annotations not found: {ann_file}")
            return
        if not img_dir.exists():
            logger.warning(f"COCO-Stuff images directory not found: {img_dir}")
            return

        with open(ann_file) as f:
            stuff = json_mod.load(f)

        img_to_anns: dict[str, list] = {}
        for ann in stuff["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        img_info = {img["id"]: img for img in stuff["images"]}

        for img_id, anns in img_to_anns.items():
            if img_id not in img_info:
                continue
            img_path = img_dir / img_info[img_id]["file_name"]
            if img_path.exists():
                self.samples.append(
                    {
                        "image": img_path,
                        "annotations": anns,
                        "width": img_info[img_id]["width"],
                        "height": img_info[img_id]["height"],
                    }
                )

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        W, H = sample["width"], sample["height"]
        label_map = torch.zeros((H, W), dtype=torch.long)
        for ann in sample["annotations"]:
            try:
                from pycocotools import mask as mask_utils

                if isinstance(ann["segmentation"], dict):
                    m = mask_utils.decode(ann["segmentation"])
                else:
                    m = mask_utils.decode(mask_utils.frPyObjects(ann["segmentation"], H, W))
                if m.ndim == 3:
                    m = m.max(axis=2)
                label_map[m > 0] = ann["category_id"] + 1
            except ImportError:
                logger.warning("pycocotools not available, skipping COCO-Stuff mask decode")
                break

        label_map = label_map.unsqueeze(0).unsqueeze(0).float()
        label_map = (
            torch.nn.functional.interpolate(
                label_map, size=(self.resolution, self.resolution), mode="nearest"
            )
            .long()
            .squeeze(0)
            .squeeze(0)
        )

        unique_labels = label_map.unique()
        palette = self._color_palette.to(label_map.device)
        rgb = palette[label_map.clamp(0, 170).long()]
        target_rgb = rgb.permute(2, 0, 1)

        class_color_map = {}
        for cat_id in unique_labels.tolist():
            if 0 <= cat_id < 171:
                class_color_map[f"class_{cat_id}"] = tuple(
                    int(palette[cat_id][i].item() * 255) for i in range(3)
                )

        prompt = PromptBuilder.build_semantic_prompt(class_color_map)
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "class_color_map": class_color_map,
        }


class RefCOCODataset(BaseVisionDataset):
    """RefCOCO / RefCOCO+ / RefCOCOg dataset for referring expression segmentation.

    RefCOCO: 142,209 expressions for 50,000 objects in 19,994 images.
    RefCOCO+: 141,564 expressions with harder, more complex descriptions.
    RefCOCOg: 104,560 expressions with longer, more natural language.

    Expected directory structure (processed format)::

        root_dir/
          images/          ← COCO images (*.jpg)
          annotations.json ← RefCOCO annotations

    Args:
        root_dir: Dataset root directory.
        split: One of "train", "val", "test".
        variant: One of "refcoco", "refcoco+", "refcocog".
    """

    TASK_TYPE = "referring"

    def __init__(
        self, root_dir, split="train", variant="refcoco", resolution=512, **kwargs
    ) -> None:
        self.variant = variant
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        import json as json_mod

        ann_file = self.root_dir / f"{self.variant}_{self.split}.json"
        if not ann_file.exists():
            ann_file = self.root_dir / "annotations.json"
        if not ann_file.exists():
            logger.warning(f"RefCOCO annotations not found: {ann_file}")
            return

        img_dir = self.root_dir / "images"
        with open(ann_file) as f:
            data = json_mod.load(f)

        for item in data:
            img_path = img_dir / item.get("image_file", item.get("file_name", ""))
            if not img_path.exists() and "image_id" in item:
                img_path = img_dir / f"{item['image_id']:012d}.jpg"
            if img_path.exists():
                self.samples.append(
                    {
                        "image": img_path,
                        "expression": item.get("sentence", item.get("ref", "the object")),
                        "mask": item.get("mask", None),
                        "bbox": item.get("bbox", None),
                    }
                )

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        W, H = image.shape[2], image.shape[1]
        mask = torch.zeros((H, W), dtype=torch.float32)

        if sample.get("mask") is not None:
            m_data = sample["mask"]
            if isinstance(m_data, dict):
                try:
                    from pycocotools import mask as mask_utils

                    m = mask_utils.decode(m_data)
                    mask = torch.from_numpy(m).float()
                except ImportError:
                    pass
            elif isinstance(m_data, list):
                from PIL import ImageDraw

                m_img = Image.new("L", (W, H), 0)
                draw = ImageDraw.Draw(m_img)
                for poly in m_data:
                    draw.polygon(poly, fill=255)
                mask = _pil_to_tensor(m_img).float().squeeze(0) / 255.0

        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = (
            torch.nn.functional.interpolate(
                mask, size=(self.resolution, self.resolution), mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
        )

        target_rgb = mask.unsqueeze(0).repeat(3, 1, 1)
        prompt = PromptBuilder.build_referring_prompt(sample["expression"])
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "expression": sample["expression"],
        }


class VirtualKITTIDataset(BaseVisionDataset):
    """Virtual KITTI 2 dataset for synthetic depth estimation.

    Virtual KITTI provides 50,000 synthetic frames with perfect depth
    ground truth, cloned from KITTI with weather/time variations.

    Expected directory structure::

        root_dir/
          Scene01/  ... Scene06/
            frames/
              rgb/         ← RGB images (*.jpg)
              depth/       ← Depth maps (*.png, uint16 centimetres, 65535 = sky)

    Depth values are in meters (uint16 PNG: depth_cm / 100.0); sky pixels are NaN
    before encoding and zero in the RGB target.
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        for scene_dir in sorted(self.root_dir.rglob("frames")):
            rgb_dir = scene_dir / "rgb"
            depth_dir = scene_dir / "depth"
            if not rgb_dir.exists() or not depth_dir.exists():
                continue
            for img_path in sorted(rgb_dir.rglob("*.jpg")):
                # stock layout: rgb/Camera_0/rgb_00000.jpg <-> depth/Camera_0/depth_00000.png
                relative = img_path.relative_to(rgb_dir)
                depth_name = img_path.stem.replace("rgb", "depth", 1) + ".png"
                depth_path = depth_dir / relative.parent / depth_name
                if not depth_path.exists():
                    depth_path = depth_dir / relative.parent / (img_path.stem + ".png")
                if depth_path.exists():
                    self.samples.append({"image": img_path, "target": depth_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        # VKITTI2 depth PNGs are uint16 *centimetres*; 65535 marks sky / beyond the far plane.
        raw = torch.from_numpy(np.asarray(Image.open(sample["target"])).astype(np.float32))
        depth = (raw / 100.0).unsqueeze(0)
        depth = torch.where(raw.unsqueeze(0) >= 65535, torch.full_like(depth, float("nan")), depth)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        target_rgb = self.depth_mapper.depth_to_rgb(torch.nan_to_num(depth, nan=0.0).unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class TUMRGBDDataset(BaseVisionDataset):
    """TUM RGB-D dataset for indoor depth estimation with camera trajectories.

    TUM RGB-D provides 39 sequences with RGB-D frames and ground truth
    trajectories, commonly used for SLAM and depth estimation evaluation.

    Expected directory structure (per sequence)::

        root_dir/
          sequence_name/
            rgb/           ← RGB images (*.png)
            depth/         ← Depth maps (*.png, uint16 mm)

    Depth values are in meters (uint16 PNG: depth_mm / 5000.0).
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"TUM RGB-D root directory not found: {self.root_dir}")
            return
        for seq_dir in sorted(self.root_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            rgb_dir = seq_dir / "rgb"
            depth_dir = seq_dir / "depth"
            if not rgb_dir.exists() or not depth_dir.exists():
                continue
            for img_path in sorted(rgb_dir.rglob("*.png")):
                depth_name = img_path.stem + ".png"
                depth_path = depth_dir / depth_name
                if depth_path.exists():
                    self.samples.append({"image": img_path, "target": depth_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        depth_img = Image.open(sample["target"])
        depth = _pil_to_tensor(depth_img).float()
        if depth.max() > 1.0:
            depth = depth / 5000.0

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        valid_mask = depth > 0.0
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "valid_mask": valid_mask.squeeze(0),
        }


class FlyingThings3DDataset(BaseVisionDataset):
    """FlyingThings3D dataset for optical flow estimation.

    FlyingThings3D provides 25,000 synthetic stereo frames with dense
    optical flow ground truth from 3D animated scenes.

    Expected directory structure::

        root_dir/
          train/
            image_clean/
              left/        ← RGB images (*.png)
            flow/
              left/        ← Flow files (*.flo)

    Flow values are in pixels/frame (u, v) as float32.
    """

    TASK_TYPE = "flow"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        img_dir = self.root_dir / self.split / "image_clean" / "left"
        flow_dir = self.root_dir / self.split / "flow" / "left"
        if not img_dir.exists():
            logger.warning(f"FlyingThings3D images directory not found: {img_dir}")
            return
        for img_path in sorted(img_dir.rglob("*.png")):
            flow_name = img_path.stem + ".flo"
            flow_path = flow_dir / flow_name
            if flow_path.exists():
                self.samples.append({"image": img_path, "target": flow_path})

    def _load_flo(self, path: Path) -> torch.Tensor:
        import numpy as np

        with open(path, "rb") as f:
            tag = np.fromfile(f, np.float32, count=1)[0]
            if abs(tag - 202021.25) > 1e-6:
                flow = np.load(path)
                return torch.from_numpy(flow).float().permute(2, 0, 1)
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            flow = np.fromfile(f, np.float32, count=w * h * 2)
            flow = flow.reshape(h, w, 2)
        return torch.from_numpy(flow).float().permute(2, 0, 1)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        flow = self._load_flo(sample["target"])
        flow = torch.nn.functional.interpolate(
            flow.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        u, v = flow[0], flow[1]
        magnitude = torch.sqrt(u**2 + v**2 + 1e-10)
        direction = torch.atan2(v, u)
        max_mag = magnitude.max() + 1e-10
        hue = (direction + math.pi) / (2 * math.pi)
        sat = magnitude / max_mag
        val = torch.ones_like(hue)
        target_rgb = torch.stack([hue, sat, val], dim=0).clamp(0, 1)

        prompt = PromptBuilder.build_flow_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class MapillaryVistasDataset(BaseVisionDataset):
    """Mapillary Vistas dataset for street-level semantic segmentation.

    Mapillary Vistas provides 25,000 high-resolution street-level images
    with 66 semantic classes, covering diverse global locations.

    Expected directory structure::

        root_dir/
          images/          ← RGB images (*.jpg)
          labels/          ← Label PNGs (*.png, 65 classes + background)

    Label indices: 0-64 (65 classes), 255 = ignore.
    """

    TASK_TYPE = "semantic"
    NUM_CLASSES = 65

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self._build_palette()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _build_palette(self):
        import random

        rng = random.Random(65)
        colors = []
        for _ in range(65):
            colors.append((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        self._color_palette = torch.tensor(colors, dtype=torch.float32) / 255.0

    def _scan(self) -> None:
        img_dir = self.root_dir / "images" / self.split
        label_dir = self.root_dir / "labels" / self.split
        if not img_dir.exists():
            logger.warning(f"Mapillary images directory not found: {img_dir}")
            return
        for img_path in sorted(img_dir.rglob("*.jpg")):
            label_name = img_path.stem + ".png"
            label_path = label_dir / label_name
            if label_path.exists():
                self.samples.append({"image": img_path, "target": label_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        label_img = Image.open(sample["target"])
        label = _pil_to_tensor(label_img).long().squeeze(0)
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = (
            torch.nn.functional.interpolate(
                label, size=(self.resolution, self.resolution), mode="nearest"
            )
            .long()
            .squeeze(0)
            .squeeze(0)
        )

        label = torch.where(label == 255, torch.zeros_like(label), label)
        label = label.clamp(0, 64)

        palette = self._color_palette.to(label.device)
        rgb = palette[label.long()]
        target_rgb = rgb.permute(2, 0, 1)

        class_color_map = {
            f"class_{i}": tuple(int(c * 255) for c in palette[i].tolist()) for i in range(65)
        }
        prompt = PromptBuilder.build_semantic_prompt(class_color_map)
        return {
            "prompt": prompt,
            "image": image,
            "target_rgb": target_rgb,
            "class_color_map": class_color_map,
        }


class nuScenesDataset(_NativeGTMixin, BaseVisionDataset):  # noqa: N801
    """nuScenes dataset for multi-task autonomous driving perception.

    nuScenes provides 1,000 scenes with 360° camera coverage, LiDAR,
    radar, and semantic annotations. Supports depth, semantic segmentation,
    object detection, and instance segmentation tasks.

    Expected directory structure (simplified)::

        root_dir/
          samples/
            CAM_FRONT/       ← Front camera images (*.jpg)
          depth/             ← LiDAR depth projections (*.npy or *.png)
          labels/            ← Semantic labels (*.png)

        root_dir/<split>/...  is also accepted.

    Depth evaluation returns metric ``gt_depth`` so the paper 80 m protocol
    can score decoded RGB. The class ``TASK_TYPE`` stays ``semantic`` for the
    default constructor; pass ``task="depth"`` for the paper depth bench, or use
    the ``nuscenes_depth`` key of ``build_dataset``.
    """

    TASK_TYPE = "semantic"

    def __init__(
        self,
        root_dir,
        split="train",
        resolution=512,
        task="semantic",
        native_gt=False,
        **kwargs,
    ) -> None:
        self.task = task
        self._build_palette()
        self.depth_mapper = DepthRGBMapper()
        self._init_native(native_gt)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _build_palette(self):
        import random

        rng = random.Random(23)
        colors = []
        for _ in range(23):
            colors.append((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        self._color_palette = torch.tensor(colors, dtype=torch.float32) / 255.0

    def _scan_roots(self) -> list[Path]:
        roots = []
        split_root = self.root_dir / self.split
        if split_root.is_dir():
            roots.append(split_root)
        roots.append(self.root_dir)
        return roots

    def _scan(self) -> None:
        if self.task == "depth":
            for root in self._scan_roots():
                img_dir = root / "samples" / "CAM_FRONT"
                depth_dir = root / "depth"
                if not img_dir.exists() or not depth_dir.exists():
                    continue
                for img_path in sorted(img_dir.rglob("*.jpg")):
                    depth_path = depth_dir / (img_path.stem + ".npy")
                    if not depth_path.exists():
                        depth_path = depth_dir / (img_path.stem + ".png")
                    if depth_path.exists():
                        self.samples.append(
                            {"image": img_path, "target": depth_path, "task": "depth"}
                        )
                if self.samples:
                    return
            return
        if self.task in ("semantic", "instance"):
            for root in self._scan_roots():
                img_dir = root / "samples" / "CAM_FRONT"
                label_dir = root / "labels"
                if not img_dir.exists():
                    continue
                for img_path in sorted(img_dir.rglob("*.jpg")):
                    label_path = label_dir / (img_path.stem + ".png")
                    if label_path.exists():
                        self.samples.append(
                            {"image": img_path, "target": label_path, "task": self.task}
                        )
                if self.samples:
                    return

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        task = sample.get("task", self.task)

        if task == "depth":
            depth_path = Path(sample["target"])
            if depth_path.suffix.lower() == ".npy":
                depth = torch.from_numpy(np.load(depth_path)).float()
                if depth.dim() == 2:
                    depth = depth.unsqueeze(0)
            else:
                depth = load_scalar(depth_path)
                if depth.max() > 200:
                    depth = depth / 256.0
            native = self._native_fields(depth, torch.isfinite(depth) & (depth > 0))
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(self.resolution, self.resolution), mode="nearest"
            ).squeeze(0)
            valid_mask = torch.isfinite(depth) & (depth > 0)
            depth = torch.where(valid_mask, depth, torch.zeros_like(depth))
            target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
            prompt = PromptBuilder.build_depth_prompt()
            return {
                "prompt": prompt,
                "image": image,
                "target_rgb": target_rgb,
                "gt_depth": depth,
                "valid_mask": valid_mask.squeeze(0),
                **native,
            }

        label_img = Image.open(sample["target"])
        label = _pil_to_tensor(label_img).long().squeeze(0)
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = (
            torch.nn.functional.interpolate(
                label, size=(self.resolution, self.resolution), mode="nearest"
            )
            .long()
            .squeeze(0)
            .squeeze(0)
        )
        label = label.clamp(0, 22)
        palette = self._color_palette.to(label.device)
        rgb = palette[label.long()]
        target_rgb = rgb.permute(2, 0, 1)
        class_color_map = {
            f"class_{i}": tuple(int(c * 255) for c in palette[i].tolist()) for i in range(23)
        }
        prompt = PromptBuilder.build_semantic_prompt(class_color_map)

        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class NuScenesDepthDataset(nuScenesDataset):
    """nuScenes front-camera metric depth (the paper's 80 m benchmark) as its own key.

    ``build_dataset`` names its first argument ``task``, so the depth task cannot be
    passed through its keyword arguments.
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        if kwargs.pop("task", "depth") != "depth":
            raise ValueError("nuscenes_depth is the depth task; use the nuscenes key for others")
        super().__init__(root_dir, split, resolution, task="depth", **kwargs)


class PreEncodedDepthDataset(BaseVisionDataset):
    """Dataset for pre-encoded depth RGB targets.

    Handles the common ``images/`` + ``targets/`` directory layout where
    target images are already RGB-encoded depth visualizations (not raw
    depth maps). This is the format produced by the v4 data pipeline and
    used by the synthetic data generator.

    Expected directory structure::

        root_dir/
          images/   ← RGB input images (*.jpg, *.png)
          targets/  ← Pre-encoded RGB depth targets (*.png)

    The target RGB is loaded directly as [3, H, W] in [0, 1] without
    any depth-to-RGB conversion.
    """

    TASK_TYPE = "depth"

    def _scan(self) -> None:
        img_dir = self.root_dir / "images"
        target_dir = self.root_dir / "targets"
        if not img_dir.exists():
            logger.warning(f"Images directory not found: {img_dir}")
            return
        if not target_dir.exists():
            logger.warning(f"Targets directory not found: {target_dir}")
            return
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix not in (".jpg", ".jpeg", ".png"):
                continue
            target_path = target_dir / (img_path.stem + ".png")
            if not target_path.exists():
                target_path = target_dir / (img_path.stem + ".jpg")
            if target_path.exists():
                self.samples.append({"image": img_path, "target": target_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        target_img = Image.open(sample["target"]).convert("RGB")
        if target_img.size != (self.resolution, self.resolution):
            target_img = target_img.resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )
        target_rgb = _pil_to_tensor(target_img).float() / 255.0

        prompt = PromptBuilder.build_depth_prompt()
        return self._tag_task({"prompt": prompt, "image": image, "target_rgb": target_rgb})


def build_dataset(
    task: str,
    root_dir: str,
    split: str = "train",
    resolution: int = 512,
    **kwargs,
) -> BaseVisionDataset:
    """Factory function to create task-specific datasets."""
    registry = {
        "depth": HypersimDepthDataset,
        "normal": ScanNetNormalDataset,
        "semantic": CityscapesSegDataset,
        "instance": SACoInstanceDataset,
        "referring": RefCOCOgDataset,
        # Extended datasets (Phase 1: 5 datasets)
        "nyu_depth": NYUDepthDataset,
        "ibims": IBimsDepthDataset,
        "nyu_normal": NYUNormalDataset,
        "diode": DIODEDataset,
        "blendedmvs": BlendedMVSDepthDataset,
        "kitti": KITTIDataset,
        "ade20k": ADE20KDataset,
        "sintel": SintelFlowDataset,
        # Extended datasets (Phase 2: 8 more datasets)
        "eth3d": ETH3DDataset,
        "tartanair": TartanAirDepthDataset,
        "oasis": OASISNormalDataset,
        "diode_normal": DIODENormalDataset,
        "vkitti_normal": VirtualKITTINormalDataset,
        "coco_detection": COCODetectionDataset,
        "coco_stuff": COCOStuffDataset,
        "refcoco": RefCOCODataset,
        "virtual_kitti": VirtualKITTIDataset,
        "tum_rgbd": TUMRGBDDataset,
        "flyingthings3d": FlyingThings3DDataset,
        "mapillary": MapillaryVistasDataset,
        "nuscenes": nuScenesDataset,
        "nuscenes_depth": NuScenesDepthDataset,
        "preencoded_depth": PreEncodedDepthDataset,
    }
    if task in {"reasonseg", "genai_bench", "imgedit"}:
        from .paper_eval_datasets import GenAIBenchDataset, ImgEditDataset, ReasonSegDataset

        registry = {
            **registry,
            "reasonseg": ReasonSegDataset,
            "genai_bench": GenAIBenchDataset,
            "imgedit": ImgEditDataset,
        }
    if task not in registry:
        raise ValueError(f"Unknown task: {task}. Choose from: {list(registry.keys())}")
    return registry[task](root_dir=root_dir, split=split, resolution=resolution, **kwargs)


class V3FlatDataset(Dataset):
    """Generic loader for v3-format flat datasets (images/targets/prompts).

    Each task directory follows the same layout::

        <root>/<task_dir>/
            images/    *.jpg / *.png
            targets/   matching filenames (RGB PNG)
            prompts/   <stem>.txt  (optional per-sample prompt)

    This class replaces the original-data-specific Dataset subclasses
    (CityscapesSegDataset, HypersimDepthDataset, etc.) when loading
    pre-packaged v3 data, avoiding the directory-structure mismatch.

    Supports train_val_split.json filtering: when a split file is provided,
    only samples whose image filenames appear in the selected split
    (train or val) are included. This preserves the cross-task MD5 grouping
    that prevents data leakage.
    """

    # Directory name -> task label used by downstream training code.
    # Keys are v3 directory names; values are the labels attached to each
    # sample's "task" field. When they differ, the mapping ensures
    # MixedVisionDataset and InstructionTuner see the expected task name.
    DIR_TO_TASK: dict[str, str] = {
        "depth": "depth",
        "normal": "normal",
        "segmentation": "semantic",  # dir "segmentation" -> label "semantic"
        "instance": "instance",
        "referring": "referring",
        "keypoint": "keypoint",
        "object_detection": "object_detection",
        "depth_edge": "depth_edge",
        "multiview": "multiview",
        "structure": "structure",
        "specular_removal": "specular_removal",
    }

    # Default prompts for tasks that don't have per-sample prompt files.
    TASK_PROMPTS: dict[str, str] = {
        "depth": "Generate a metric depth visualization of this image.",
        "normal": "Generate a surface normal estimation visualization of this image.",
        "semantic": SEMANTIC_SEGMENTATION_PROMPT,
        "instance": "Generate an instance segmentation visualization of this image.",
        "referring": "Generate a referring expression segmentation visualization of this image.",
        "keypoint": "Generate a keypoint detection visualization of this image.",
        "object_detection": "Generate an object detection visualization of this image.",
        "depth_edge": "Generate a depth edge detection visualization of this image.",
        "multiview": "Generate a multi-view visualization of this image.",
        "structure": "Generate a structure-from-motion visualization of this image.",
        "specular_removal": "Remove specular highlights from this image.",
    }

    DEFAULT_PROMPT = "Generate a visual understanding output for this image."

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(
        self,
        task_dir: str | Path,
        task_label: str | None = None,
        resolution: int = 512,
        split_filter: tuple[set[str], str] | None = None,
        keep_semantic_prompts: bool = False,
    ) -> None:
        """Args:
        task_dir: Path to the task directory (e.g. ``.../vb_snvc_train_v3/depth``).
        task_label: Override the task label. If ``None``, derived from the
            directory name via ``DIR_TO_TASK``.
        resolution: Target resolution for image resizing.
        split_filter: Optional ``(filenames_set, split_name)`` tuple.
            When provided, only samples whose image filename appears in
            ``filenames_set`` are loaded. ``split_name`` is used for logging.
        keep_semantic_prompts: Keep per-sample semantic prompts from
            ``prompts/*.txt`` instead of forcing the fixed Cityscapes legend
            (for open-vocabulary data with per-sample color mappings).
        """
        self.keep_semantic_prompts = keep_semantic_prompts
        self.task_dir = Path(task_dir)
        self.dir_name = self.task_dir.name
        self.task = task_label or self.DIR_TO_TASK.get(self.dir_name, self.dir_name)
        self.resolution = resolution
        self.samples: list[dict] = []
        self._load(split_filter)

    def _load(self, split_filter: tuple[set[str], str] | None) -> None:
        import os

        root = self.task_dir
        img_dir = root / "images"
        tgt_dir = root / "targets"
        prompt_dir = root / "prompts"

        if not img_dir.exists():
            logger.warning("V3FlatDataset: images/ not found at %s", img_dir)
            return

        # Build target index keyed by stem for O(1) lookup
        target_index: dict[str, Path] = {}
        if tgt_dir.exists():
            with os.scandir(str(tgt_dir)) as entries:
                for entry in entries:
                    if entry.is_file():
                        p = Path(entry.path)
                        if p.suffix.lower() in self.IMG_EXTS:
                            target_index[p.stem] = p

        # Load split filter
        allowed_filenames: set[str] | None = None
        split_name = "all"
        if split_filter is not None:
            allowed_filenames, split_name = split_filter

        default_prompt = self.TASK_PROMPTS.get(self.task, self.DEFAULT_PROMPT)

        image_paths: list[Path] = []
        with os.scandir(str(img_dir)) as entries:
            for entry in entries:
                if entry.is_file():
                    p = Path(entry.path)
                    if p.suffix.lower() in self.IMG_EXTS:
                        image_paths.append(p)
        image_paths.sort()

        skipped = 0
        for img_path in image_paths:
            tgt_path = target_index.get(img_path.stem)
            if tgt_path is None:
                continue

            # Apply split filter
            if allowed_filenames is not None and img_path.name not in allowed_filenames:
                skipped += 1
                continue

            prompt = default_prompt
            prompt_path = prompt_dir / f"{img_path.stem}.txt"
            if prompt_path.is_file():
                text = prompt_path.read_text(encoding="utf-8").strip()
                if text:
                    prompt = text
            prompt = normalize_task_prompt(self.task, prompt, keep_custom=self.keep_semantic_prompts)

            self.samples.append(
                {
                    "image_path": str(img_path),
                    "target_path": str(tgt_path),
                    "prompt": prompt,
                }
            )

        logger.info(
            "V3FlatDataset[%s/%s]: loaded %d samples%s",
            self.dir_name,
            self.task,
            len(self.samples),
            f" (split={split_name}, skipped {skipped})" if skipped else "",
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        import numpy as np
        from PIL import Image

        def _load(p: str, normalise_range: tuple) -> torch.Tensor:
            img = Image.open(p).convert("RGB")
            if hasattr(self, "resolution"):
                img = img.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            lo, hi = normalise_range
            return arr * (hi - lo) + lo

        image = _load(sample["image_path"], (-1.0, 1.0))
        target_rgb = _load(sample["target_path"], (0.0, 1.0))

        return {
            "prompt": sample["prompt"],
            "image": image,
            "image_range": "-1_1",
            "target_rgb": target_rgb,
            "target_rgb_range": "0_1",
            "task": self.task,
        }


def load_v3_train_val_split(
    dataset_root: str | Path,
) -> dict[str, dict[str, list[str]]] | None:
    """Load train_val_split.json from a v3 dataset root.

    Returns a dict with keys ``"train"`` and ``"val"``, each mapping
    task_name -> list of image filenames. Returns ``None`` if the file
    does not exist or is malformed.
    """
    import json

    split_path = Path(dataset_root) / "train_val_split.json"
    if not split_path.exists():
        return None
    try:
        data = json.loads(split_path.read_text(encoding="utf-8"))
        if "train" in data and "val" in data:
            return data
        logger.warning("train_val_split.json exists but missing train/val keys")
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load train_val_split.json: %s", e)
        return None


def build_v3_datasets(
    dataset_root: str | Path,
    resolution: int = 512,
    split: str = "train",
    use_split_file: bool = True,
) -> list[V3FlatDataset]:
    """Build V3FlatDataset instances for all v3 task directories.

    Scans ``dataset_root`` for subdirectories matching known v3 task names,
    creates a :class:`V3FlatDataset` for each, and optionally applies
    train/val split filtering from ``train_val_split.json``.

    Args:
        dataset_root: Path to the v3 dataset root (e.g.
            ``.../vb_snvc_train_v3``).
        resolution: Target image resolution.
        split: ``"train"`` or ``"val"`` — which split to load.
        use_split_file: If ``True`` and ``train_val_split.json`` exists,
            filter samples by the split. If ``False``, load all samples.

    Returns:
        List of :class:`V3FlatDataset` with ``len > 0``. Empty datasets
        are silently skipped.
    """
    root = Path(dataset_root)
    if not root.is_dir():
        logger.error("V3 dataset root not found: %s", root)
        return []

    # Load split file if requested
    split_data: dict[str, dict[str, list[str]]] | None = None
    if use_split_file:
        split_data = load_v3_train_val_split(root)
        if split_data is None:
            logger.info("No train_val_split.json found — loading all samples (no split filter)")

    datasets: list[V3FlatDataset] = []
    for dir_name, task_label in V3FlatDataset.DIR_TO_TASK.items():
        task_dir = root / dir_name
        if not task_dir.is_dir():
            continue

        split_filter: tuple[set[str], str] | None = None
        if split_data is not None:
            split_map = split_data.get(split, {})
            # train_val_split.json keys are directory names (e.g. "segmentation")
            filenames = split_map.get(dir_name)
            if filenames is not None:
                split_filter = (set(filenames), split)
            else:
                logger.debug(
                    "V3: task %s not in split file %s — loading all",
                    dir_name,
                    split,
                )

        ds = V3FlatDataset(
            task_dir=task_dir,
            task_label=task_label,
            resolution=resolution,
            split_filter=split_filter,
        )
        if len(ds) > 0:
            datasets.append(ds)

    return datasets


__all__ = [
    "BaseVisionDataset",
    "HypersimDepthDataset",
    "ScanNetNormalDataset",
    "CityscapesSegDataset",
    "SACoInstanceDataset",
    "RefCOCOgDataset",
    "NYUDepthDataset",
    "IBimsDepthDataset",
    "NYUNormalDataset",
    "DIODEDataset",
    "BlendedMVSDepthDataset",
    "KITTIDataset",
    "ADE20KDataset",
    "SintelFlowDataset",
    "ETH3DDataset",
    "TartanAirDepthDataset",
    "OASISNormalDataset",
    "DIODENormalDataset",
    "VirtualKITTINormalDataset",
    "COCODetectionDataset",
    "COCOStuffDataset",
    "RefCOCODataset",
    "VirtualKITTIDataset",
    "TUMRGBDDataset",
    "FlyingThings3DDataset",
    "MapillaryVistasDataset",
    "nuScenesDataset",
    "PreEncodedDepthDataset",
    "V3FlatDataset",
    "load_v3_train_val_split",
    "build_dataset",
    "build_v3_datasets",
]
