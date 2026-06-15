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

    def _label_to_rgb(self, label: torch.Tensor) -> torch.Tensor:
        H, W = label.shape
        rgb = torch.zeros(3, H, W, device=label.device)
        class_names = list(self.CITYSCAPES_COLORS.keys())
        for cls_idx, cls_name in enumerate(class_names):
            color = self.CITYSCAPES_COLORS[cls_name]
            mask = label == cls_idx
            for ch, c_val in enumerate(color):
                rgb[ch][mask] = c_val / 255.0
        return rgb


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
    """RefCOCOg dataset for referring expression segmentation."""

    TASK_TYPE = "referring"

    def __init__(self, root_dir, split="train", resolution=512, **kwargs) -> None:
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning(f"RefCOCOg directory not found: {self.root_dir}")
            return
        ann_file = self.root_dir / "refs(g).p"
        if ann_file.exists():
            try:
                import pickle
                with open(ann_file, "rb") as f:
                    refs = pickle.load(f)
                for ref in refs:
                    img_id = ref.get("image_id", ref.get("file_name", ""))
                    expressions = ref.get("sentences", [ref.get("sent", "")])
                    if expressions:
                        expr = expressions[0] if isinstance(expressions, list) else expressions
                        expr = expr.get("sent", expr) if isinstance(expr, dict) else expr
                        img_path = self.root_dir / "images" / f"{img_id}.jpg"
                        if img_path.exists():
                            self.samples.append({"image": img_path, "expression": expr})
            except Exception as e:
                logger.warning(f"Failed to load RefCOCOg annotations: {e}")

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        expression = sample["expression"]
        target_rgb = torch.zeros(3, self.resolution, self.resolution)
        target_rgb[0] = 1.0
        prompt = PromptBuilder.build_referring_prompt(expression)
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}


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
