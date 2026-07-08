"""Dataset loaders for Vision Banana instruction tuning.

Each loader produces (prompt, image, target_rgb) triples where:
- prompt: task-specific instruction text
- image: input RGB image tensor [3, H, W] in [-1, 1]
- target_rgb: ground-truth encoded RGB visualization [3, H, W] in [0, 1]
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..core.depth_mapper import DepthRGBMapper
from ..core.normal_mapper import NormalRGBMapper
from ..inference.prompt import PromptBuilder

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
        img = Image.open(path).convert("RGB")
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
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"Hypersim directory not found: {self.root_dir}")
            return
        for img_path in sorted(self.root_dir.rglob("*color*")):
            depth_path = Path(str(img_path).replace("color", "depth"))
            if depth_path.exists():
                self.samples.append({"image": img_path, "target": depth_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        depth_img = Image.open(sample["target"])
        depth = _pil_to_tensor(depth_img).float()
        if depth.max() > 1.0:
            depth = depth / 1000.0

        # depth is [1, H, W] → add batch dim for interpolate
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class ScanNetNormalDataset(BaseVisionDataset):
    """ScanNet dataset for surface normal estimation."""

    TASK_TYPE = "normal"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.normal_mapper = NormalRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"ScanNet directory not found: {self.root_dir}")
            return
        for img_path in sorted(self.root_dir.rglob("*color*")):
            normal_path = Path(str(img_path).replace("color", "normal"))
            if normal_path.exists():
                self.samples.append({"image": img_path, "target": normal_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        normal_img = Image.open(sample["target"]).convert("RGB")
        if normal_img.size != (self.resolution, self.resolution):
            normal_img = normal_img.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        target_rgb = _pil_to_tensor(normal_img).float() / 255.0

        prompt = PromptBuilder.build_normal_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class CityscapesSegDataset(BaseVisionDataset):
    """Cityscapes dataset for semantic segmentation."""

    TASK_TYPE = "semantic"

    CITYSCAPES_COLORS = {
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

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
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
        prompt = PromptBuilder.build_semantic_prompt(self.CITYSCAPES_COLORS)
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}

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

        Replaces the per-class Python loop with a single index_select operation.
        Unknown label indices are clamped to the last palette entry (harmless).
        """
        palette = self._get_palette().to(label.device)  # [C, 3]
        n_cls = palette.shape[0]
        flat = label.reshape(-1).clamp(0, n_cls - 1).long()  # [H*W]
        rgb_flat = palette[flat]  # [H*W, 3]
        H, W = label.shape
        return rgb_flat.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]


class SACoInstanceDataset(BaseVisionDataset):
    """SA-Co dataset for instance segmentation.

    Expects a directory structure with images and instance masks.
    Each instance is encoded as a unique color in the mask image.
    """

    TASK_TYPE = "instance"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"SA-Co directory not found: {self.root_dir}")
            return
        for img_path in sorted(self.root_dir.rglob("*color*")):
            inst_path = Path(str(img_path).replace("color", "instance"))
            if inst_path.exists():
                self.samples.append({"image": img_path, "target": inst_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        inst_img = Image.open(sample["target"]).convert("RGB")
        if inst_img.size != (self.resolution, self.resolution):
            inst_img = inst_img.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        target_rgb = _pil_to_tensor(inst_img).float() / 255.0

        prompt = PromptBuilder.build_instance_prompt("objects")
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class RefCOCOgDataset(BaseVisionDataset):
    """RefCOCOg dataset for referring expression segmentation.

    Expected directory layout::

        root_dir/
          images/          ← COCO images (*.jpg)
          refs(g).p        ← RefCOCOg annotation pickle
          instances.json   ← COCO instances annotation (optional, for RLE masks)

    Each sample returns a binary white-on-black mask: the referred region is
    rendered solid white (1.0) and the rest is black (0.0).  If a segmentation
    annotation is available (via ``instances.json``) it is used directly;
    otherwise a filled bounding-box is used as a coarse substitute.
    """

    TASK_TYPE = "referring"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"RefCOCOg directory not found: {self.root_dir}")
            return

        ann_file = self.root_dir / "refs(g).p"
        if not ann_file.exists():
            logger.warning(f"RefCOCOg annotation not found: {ann_file}")
            return

        # Load COCO instances for mask decoding (optional)
        instances_file = self.root_dir / "instances.json"
        self._coco = None
        if instances_file.exists():
            try:
                import json

                with open(instances_file) as f:
                    coco_data = json.load(f)
                # Build ann_id → annotation lookup
                self._ann_lookup: dict[int, dict] = {
                    ann["id"]: ann for ann in coco_data.get("annotations", [])
                }
            except Exception as e:
                logger.warning(f"Failed to load COCO instances: {e}")
                self._ann_lookup = {}
        else:
            self._ann_lookup = {}

        try:
            import pickle

            with open(ann_file, "rb") as f:
                refs = pickle.load(f)

            for ref in refs:
                image_id = ref.get("image_id", "")
                ann_id = ref.get("ann_id")
                expressions = ref.get("sentences", [ref.get("sent", "")])
                if not expressions:
                    continue

                # Pick the first expression
                expr = expressions[0] if isinstance(expressions, list) else expressions
                if isinstance(expr, dict):
                    expr = expr.get("sent", "")

                # Locate image file
                img_path = self.root_dir / "images" / f"{image_id}.jpg"
                if not img_path.exists():
                    # Try zero-padded COCO naming
                    img_path = self.root_dir / "images" / f"COCO_train2014_{image_id:012d}.jpg"
                    if not img_path.exists():
                        continue

                self.samples.append(
                    {
                        "image": img_path,
                        "expression": str(expr),
                        "ann_id": ann_id,
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to load RefCOCOg annotations: {e}")

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        expression = sample["expression"]

        # Try to decode the segmentation mask from COCO annotation
        target_rgb = self._load_mask(sample.get("ann_id"))

        prompt = PromptBuilder.build_referring_prompt(expression)
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}

    def _load_mask(self, ann_id: int | None) -> torch.Tensor:
        """Decode segmentation mask for a COCO annotation.

        Returns a [3, H, W] tensor in [0, 1] where:
        - White (1, 1, 1) = referred region
        - Black (0, 0, 0) = background

        Falls back to an all-black mask if the annotation is unavailable.
        """
        target_rgb = torch.zeros(3, self.resolution, self.resolution)

        if ann_id is None or ann_id not in getattr(self, "_ann_lookup", {}):
            return target_rgb

        ann = self._ann_lookup[ann_id]
        seg = ann.get("segmentation")
        if seg is None:
            return target_rgb

        try:
            import numpy as np

            # Determine original image size from the annotation bbox
            bbox = ann.get("bbox", [0, 0, self.resolution, self.resolution])
            img_w = ann.get("width", int(bbox[0] + bbox[2]) + 1)
            img_h = ann.get("height", int(bbox[1] + bbox[3]) + 1)

            mask = np.zeros((img_h, img_w), dtype=np.uint8)

            if isinstance(seg, list):
                # Polygon format
                # Prefer cv2.fillPoly (fast); fall back to pure-numpy scanline
                # fill so that cv2 remains an optional dependency.
                try:
                    import cv2  # optional

                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                except ImportError:
                    # Pure-numpy scanline polygon fill (no extra dependencies)
                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        _fill_polygon(mask, pts)
            elif isinstance(seg, dict):
                # RLE format (requires pycocotools or manual decode)
                try:
                    from pycocotools import mask as coco_mask

                    mask = coco_mask.decode(seg)
                except ImportError:
                    # Manual RLE decode (uncompressed)
                    counts = seg.get("counts", [])
                    size = seg.get("size", [img_h, img_w])
                    if isinstance(counts, list):
                        flat = np.zeros(size[0] * size[1], dtype=np.uint8)
                        idx = 0
                        val = 0
                        for c in counts:
                            flat[idx : idx + c] = val
                            idx += c
                            val = 1 - val
                        mask = flat.reshape(size[0], size[1])

            # Resize to model resolution
            mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
            mask_resized = (
                torch.nn.functional.interpolate(
                    mask_tensor, size=(self.resolution, self.resolution), mode="nearest"
                )
                .squeeze(0)
                .squeeze(0)
            )

            # Binary white/black encoding
            target_rgb = mask_resized.unsqueeze(0).expand(3, -1, -1).clone()

        except Exception as e:
            logger.debug(f"Mask decode failed for ann_id={ann_id}: {e}")

        return target_rgb


class NYUDepthDataset(BaseVisionDataset):
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

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        img_dir = self.root_dir / "images"
        depth_dir = self.root_dir / "depths"
        if not img_dir.exists():
            logger.warning(f"NYU images directory not found: {img_dir}")
            return
        for img_path in sorted(img_dir.rglob("*.jpg")):
            # Match depth file
            depth_name = img_path.stem + ".png"
            depth_path = depth_dir / depth_name
            if not depth_path.exists():
                depth_path = depth_dir / (img_path.stem + ".npy")
            if depth_path.exists():
                self.samples.append({"image": img_path, "target": depth_path})

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
                depth = depth / 1000.0  # uint16 mm to meters

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class DIODEDataset(BaseVisionDataset):
    """DIODE dataset for indoor/outdoor metric depth estimation.

    DIODE is the first public dataset with dense indoor and outdoor depth
    ground truth. Train split: 16,886 images. Val split: 3,325 images.

    Expected directory structure::

        root_dir/
          train/
            indoor/ or outdoor/
              scenes/
                */  (scene folders)
                  *_rgb.png
                  *_depth.npy
                  *_depth_mask.npy  (optional, valid mask)
          val/
            ... same structure

    Depth values are in meters (float32 .npy).
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            logger.warning(f"DIODE split directory not found: {split_dir}")
            return
        for rgb_path in sorted(split_dir.rglob("*_rgb.png")):
            depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
            if depth_path.exists():
                self.samples.append({"image": rgb_path, "target": depth_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        import numpy as np
        depth = np.load(sample["target"])
        depth = torch.from_numpy(depth).float().unsqueeze(0)

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        # Mask invalid depth (DIODE uses -1 for invalid)
        depth = torch.where(depth < 0, torch.zeros_like(depth), depth)

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


class KITTIDataset(BaseVisionDataset):
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

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        rgb_dir = self.root_dir / self.split / "rgb"
        depth_dir = self.root_dir / self.split / "depth"
        if not rgb_dir.exists():
            logger.warning(f"KITTI RGB directory not found: {rgb_dir}")
            return
        for img_path in sorted(rgb_dir.rglob("*")):
            if img_path.suffix.lower() not in (".jpg", ".png", ".jpeg"):
                continue
            depth_name = img_path.stem + ".npy"
            depth_path = depth_dir / depth_name
            if not depth_path.exists():
                depth_path = depth_dir / (img_path.stem + ".png")
            if depth_path.exists():
                self.samples.append({"image": img_path, "target": depth_path})

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
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb, "valid_mask": valid_mask.squeeze(0)}


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
        ann_dir = self.root_dir / "annotations" / ("training" if self.split == "train" else "validation")
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
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb, "class_color_map": self._color_map}

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
        magnitude = torch.sqrt(u ** 2 + v ** 2 + 1e-10)
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


class ETH3DDataset(BaseVisionDataset):
    """ETH3D dataset for high-accuracy depth estimation.

    ETH3D provides 10 indoor/outdoor scenes with high-precision laser
    scanned depth ground truth. Standard benchmark for multi-view stereo
    and depth estimation.

    Expected directory structure::

        root_dir/
          images/          ← RGB images (*.png, *.jpg)
          depths/          ← Depth maps (*.npy or *.png, in meters)

    Depth values are in meters (float32).
    """

    TASK_TYPE = "depth"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        img_dir = self.root_dir / "images"
        depth_dir = self.root_dir / "depths"
        if not img_dir.exists():
            logger.warning(f"ETH3D images directory not found: {img_dir}")
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
            if depth.max() > 10.0:
                depth = depth / 1000.0

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        valid_mask = (depth > 0.0) & (depth < 100.0)
        depth = torch.where(valid_mask, depth, torch.zeros_like(depth))

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb, "valid_mask": valid_mask.squeeze(0)}


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
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


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
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
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
                self.samples.append({
                    "image": img_path,
                    "annotations": anns,
                    "width": img_info[img_id]["width"],
                    "height": img_info[img_id]["height"],
                })

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
        label_map = torch.nn.functional.interpolate(
            label_map, size=(self.resolution, self.resolution), mode="nearest"
        ).long().squeeze(0).squeeze(0)

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
                self.samples.append({
                    "image": img_path,
                    "annotations": anns,
                    "width": img_info[img_id]["width"],
                    "height": img_info[img_id]["height"],
                })

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
        label_map = torch.nn.functional.interpolate(
            label_map, size=(self.resolution, self.resolution), mode="nearest"
        ).long().squeeze(0).squeeze(0)

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

    def __init__(self, root_dir, split="train", variant="refcoco", resolution=512, **kwargs) -> None:
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
                self.samples.append({
                    "image": img_path,
                    "expression": item.get("sentence", item.get("ref", "the object")),
                    "mask": item.get("mask", None),
                    "bbox": item.get("bbox", None),
                })

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
        mask = torch.nn.functional.interpolate(
            mask, size=(self.resolution, self.resolution), mode="nearest"
        ).squeeze(0).squeeze(0)

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
              depth/       ← Depth maps (*.png, uint16 mm)

    Depth values are in meters (uint16 PNG: depth_mm / 1000.0).
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
            depth = depth / 1000.0

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode="nearest",
        ).squeeze(0)

        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
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
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb, "valid_mask": valid_mask.squeeze(0)}


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
        magnitude = torch.sqrt(u ** 2 + v ** 2 + 1e-10)
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
        label = torch.nn.functional.interpolate(
            label, size=(self.resolution, self.resolution), mode="nearest"
        ).long().squeeze(0).squeeze(0)

        label = torch.where(label == 255, torch.zeros_like(label), label)
        label = label.clamp(0, 64)

        palette = self._color_palette.to(label.device)
        rgb = palette[label.long()]
        target_rgb = rgb.permute(2, 0, 1)

        class_color_map = {f"class_{i}": tuple(int(c * 255) for c in palette[i].tolist()) for i in range(65)}
        prompt = PromptBuilder.build_semantic_prompt(class_color_map)
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb, "class_color_map": class_color_map}


class nuScenesDataset(BaseVisionDataset):
    """nuScenes dataset for multi-task autonomous driving perception.

    nuScenes provides 1,000 scenes with 360° camera coverage, LiDAR,
    radar, and semantic annotations. Supports depth, semantic segmentation,
    object detection, and instance segmentation tasks.

    Expected directory structure (simplified)::

        root_dir/
          samples/
            CAM_FRONT/       ← Front camera images (*.jpg)
          depth/             ← LiDAR depth projections (*.npy)
          labels/            ← Semantic labels (*.png)

    This is a lightweight wrapper; full nuScenes SDK integration would
    provide richer annotations (3D boxes, trajectories, etc.).
    """

    TASK_TYPE = "semantic"

    def __init__(self, root_dir, split="train", resolution=512, task="semantic", **kwargs) -> None:
        self.task = task
        self._build_palette()
        super().__init__(root_dir, split, resolution, **kwargs)

    def _build_palette(self):
        import random
        rng = random.Random(23)
        colors = []
        for _ in range(23):
            colors.append((rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
        self._color_palette = torch.tensor(colors, dtype=torch.float32) / 255.0

    def _scan(self) -> None:
        if self.task == "depth":
            img_dir = self.root_dir / "samples" / "CAM_FRONT"
            depth_dir = self.root_dir / "depth"
            if not img_dir.exists():
                return
            for img_path in sorted(img_dir.rglob("*.jpg")):
                depth_path = depth_dir / (img_path.stem + ".npy")
                if depth_path.exists():
                    self.samples.append({"image": img_path, "target": depth_path, "task": "depth"})
        elif self.task in ("semantic", "instance"):
            img_dir = self.root_dir / "samples" / "CAM_FRONT"
            label_dir = self.root_dir / "labels"
            if not img_dir.exists():
                return
            for img_path in sorted(img_dir.rglob("*.jpg")):
                label_path = label_dir / (img_path.stem + ".png")
                if label_path.exists():
                    self.samples.append({"image": img_path, "target": label_path, "task": self.task})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        task = sample.get("task", self.task)

        if task == "depth":
            import numpy as np
            depth = np.load(sample["target"])
            depth = torch.from_numpy(depth).float().unsqueeze(0)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0), size=(self.resolution, self.resolution), mode="nearest"
            ).squeeze(0)
            mapper = DepthRGBMapper()
            target_rgb = mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
            prompt = PromptBuilder.build_depth_prompt()
        else:
            label_img = Image.open(sample["target"])
            label = _pil_to_tensor(label_img).long().squeeze(0)
            label = label.unsqueeze(0).unsqueeze(0).float()
            label = torch.nn.functional.interpolate(
                label, size=(self.resolution, self.resolution), mode="nearest"
            ).long().squeeze(0).squeeze(0)
            label = label.clamp(0, 22)
            palette = self._color_palette.to(label.device)
            rgb = palette[label.long()]
            target_rgb = rgb.permute(2, 0, 1)
            class_color_map = {f"class_{i}": tuple(int(c * 255) for c in palette[i].tolist()) for i in range(23)}
            prompt = PromptBuilder.build_semantic_prompt(class_color_map)

        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


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
        "diode": DIODEDataset,
        "kitti": KITTIDataset,
        "ade20k": ADE20KDataset,
        "sintel": SintelFlowDataset,
        # Extended datasets (Phase 2: 8 more datasets)
        "eth3d": ETH3DDataset,
        "oasis": OASISNormalDataset,
        "coco_detection": COCODetectionDataset,
        "coco_stuff": COCOStuffDataset,
        "refcoco": RefCOCODataset,
        "virtual_kitti": VirtualKITTIDataset,
        "tum_rgbd": TUMRGBDDataset,
        "flyingthings3d": FlyingThings3DDataset,
        "mapillary": MapillaryVistasDataset,
        "nuscenes": nuScenesDataset,
        "preencoded_depth": PreEncodedDepthDataset,
    }
    if task not in registry:
        raise ValueError(f"Unknown task: {task}. Choose from: {list(registry.keys())}")
    return registry[task](root_dir=root_dir, split=split, resolution=resolution, **kwargs)


__all__ = [
    "BaseVisionDataset",
    "HypersimDepthDataset",
    "ScanNetNormalDataset",
    "CityscapesSegDataset",
    "SACoInstanceDataset",
    "RefCOCOgDataset",
    "NYUDepthDataset",
    "DIODEDataset",
    "KITTIDataset",
    "ADE20KDataset",
    "SintelFlowDataset",
    "ETH3DDataset",
    "OASISNormalDataset",
    "COCODetectionDataset",
    "COCOStuffDataset",
    "RefCOCODataset",
    "VirtualKITTIDataset",
    "TUMRGBDDataset",
    "FlyingThings3DDataset",
    "MapillaryVistasDataset",
    "nuScenesDataset",
    "PreEncodedDepthDataset",
    "build_dataset",
]
