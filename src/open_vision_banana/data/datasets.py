"""Dataset loaders for Vision Banana instruction tuning.

Each loader produces (prompt, image, target_rgb) triples where:
- prompt: task-specific instruction text
- image: input RGB image tensor [3, H, W] in [-1, 1]
- target_rgb: ground-truth encoded RGB visualization [3, H, W] in [0, 1]
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    return torch.from_numpy(arr.transpose(2, 0, 1)).copy()


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
        self.samples = []
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
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        tensor = _pil_to_tensor(img).float() / 255.0
        return tensor * 2.0 - 1.0


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
        depth = _pil_to_tensor(depth_img).float().unsqueeze(0)
        if depth.max() > 1.0:
            depth = depth / 1000.0

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(self.resolution, self.resolution),
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
            normal_img = normal_img.resize(
                (self.resolution, self.resolution), Image.LANCZOS
            )
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
            gt_path = self.root_dir / "gtFine" / self.split / img_path.relative_to(img_dir).parent / gt_name
            if gt_path.exists():
                self.samples.append({"image": img_path, "target": gt_path})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])

        label_img = Image.open(sample["target"])
        label = _pil_to_tensor(label_img).long().squeeze(0)
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = torch.nn.functional.interpolate(
            label, size=(self.resolution, self.resolution), mode="nearest"
        ).long().squeeze(0).squeeze(0)

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
        palette = self._get_palette().to(label.device)           # [C, 3]
        n_cls = palette.shape[0]
        flat = label.reshape(-1).clamp(0, n_cls - 1).long()     # [H*W]
        rgb_flat = palette[flat]                                  # [H*W, 3]
        H, W = label.shape
        return rgb_flat.reshape(H, W, 3).permute(2, 0, 1)       # [3, H, W]


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
            inst_img = inst_img.resize(
                (self.resolution, self.resolution), Image.LANCZOS
            )
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

                self.samples.append({
                    "image": img_path,
                    "expression": str(expr),
                    "ann_id": ann_id,
                })

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
                import cv2  # optional dependency
                for poly in seg:
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
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
            mask_resized = torch.nn.functional.interpolate(
                mask_tensor, size=(self.resolution, self.resolution), mode="nearest"
            ).squeeze(0).squeeze(0)

            # Binary white/black encoding
            target_rgb = mask_resized.unsqueeze(0).expand(3, -1, -1).clone()

        except Exception as e:
            logger.debug(f"Mask decode failed for ann_id={ann_id}: {e}")

        return target_rgb


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
    "build_dataset",
]
