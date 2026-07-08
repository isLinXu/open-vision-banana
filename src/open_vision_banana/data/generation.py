"""Generation dataset for preserving FLUX text-to-image capability during tuning.

Used with ``MixedVisionDataset`` at the paper's 9:1 generation-to-vision ratio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_GENERATION_PROMPT = "Generate a high quality, photorealistic image."


class GenerationDataset(Dataset):
    """Text-to-image generation samples for instruction-tuning mix.

    Expected layout (pick one prompt source)::

        <root>/
          images/              ← RGB images (*.jpg / *.png)
          prompts.json         ← {"stem": "prompt text", ...}  (optional)
          metadata.jsonl       ← {"image": "file.jpg", "prompt": "..."}  (optional)

    Each sample returns ``prompt``, ``target_rgb`` in [0, 1], and optionally
    ``image`` in [-1, 1] (same pixels as target, for img2img-style batches).
    """

    def __init__(
        self,
        root_dir: str | Path,
        resolution: int = 512,
        default_prompt: str = DEFAULT_GENERATION_PROMPT,
        include_input_image: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.resolution = resolution
        self.default_prompt = default_prompt
        self.include_input_image = include_input_image
        self.samples: list[dict] = []
        self._load()

    def _load_prompts_json(self) -> dict[str, str]:
        path = self.root_dir / "prompts.json"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): str(v) for k, v in data.items()}

    def _load_metadata_jsonl(self) -> list[dict]:
        path = self.root_dir / "metadata.jsonl"
        if not path.exists():
            return []
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load(self) -> None:
        img_dir = self.root_dir / "images"
        if not img_dir.is_dir():
            logger.warning("GenerationDataset: images/ not found at %s", img_dir)
            return

        prompt_map = self._load_prompts_json()
        jsonl_rows = self._load_metadata_jsonl()

        if jsonl_rows:
            for row in jsonl_rows:
                image_name = row.get("image") or row.get("image_path")
                if not image_name:
                    continue
                img_path = img_dir / image_name
                if not img_path.exists():
                    img_path = self.root_dir / image_name
                if not img_path.exists():
                    continue
                self.samples.append({
                    "image_path": str(img_path),
                    "prompt": row.get("prompt", self.default_prompt),
                })
            if self.samples:
                logger.info("GenerationDataset: loaded %d samples from metadata.jsonl", len(self.samples))
                return

        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
                continue
            prompt = prompt_map.get(img_path.name) or prompt_map.get(img_path.stem, self.default_prompt)
            self.samples.append({
                "image_path": str(img_path),
                "prompt": prompt,
            })

        logger.info("GenerationDataset: loaded %d samples from %s", len(self.samples), img_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        import numpy as np

        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)

        target_rgb = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)

        item = {
            "prompt": sample["prompt"],
            "target_rgb": target_rgb,
        }
        if self.include_input_image:
            item["image"] = target_rgb * 2.0 - 1.0
        return item


def build_generation_dataset(
    root_dir: str | Path,
    resolution: int = 512,
    **kwargs,
) -> GenerationDataset:
    """Factory for generation preservation datasets."""
    return GenerationDataset(root_dir=root_dir, resolution=resolution, **kwargs)


__all__ = ["GenerationDataset", "build_generation_dataset", "DEFAULT_GENERATION_PROMPT"]
