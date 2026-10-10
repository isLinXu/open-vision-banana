"""Paper-protocol adapters that are not part of the training-data loaders.

ReasonSeg val, GenAI-Bench prompts, and ImgEdit instruction pairs. SA-Co/Gold
query JSONL lives on ``SACoInstanceDataset`` so the instance factory stays one
class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..inference.prompt import PromptBuilder
from .datasets import BaseVisionDataset, _fill_polygon, _pil_to_tensor

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_image(root: Path, name: str | None) -> Path | None:
    if not name:
        return None
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    direct = root / name
    if direct.is_file():
        return direct
    basename = Path(name).name
    for folder in (root, root / "images", root / "val", root / "test"):
        path = folder / basename
        if path.is_file():
            return path
    matches = sorted(root.rglob(basename))
    return matches[0] if matches else None


class ReasonSegDataset(BaseVisionDataset):
    """ReasonSeg val/test referring queries (LISA JSON-next-to-image layout).

    Expected layouts::

        root/val/*.jpg + root/val/*.json
        root/*.jpg + root/*.json

    Each JSON uses the LISA fields ``text`` / ``is_sentence`` / ``shapes``.
    The paper's gIoU uses Gemini 2.5 Pro to rewrite reasoning queries first;
    this adapter scores the raw query unless ``rewritten_text`` is present.
    """

    TASK_TYPE = "referring"

    def __init__(
        self,
        root_dir,
        split="val",
        resolution=512,
        native_gt=False,
        **kwargs,
    ) -> None:
        self.native_gt = bool(native_gt)
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            logger.warning("ReasonSeg directory not found: %s", self.root_dir)
            return
        roots = []
        split_dir = self.root_dir / self.split
        if split_dir.is_dir():
            roots.append(split_dir)
        roots.append(self.root_dir)
        seen: set[Path] = set()
        for root in roots:
            for img_path in sorted(root.iterdir() if root.is_dir() else []):
                if img_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                json_path = img_path.with_suffix(".json")
                if not json_path.is_file():
                    continue
                if img_path.resolve() in seen:
                    continue
                seen.add(img_path.resolve())
                self.samples.append(
                    {"image": img_path, "annotation": json_path, "expression": None}
                )
            if self.samples:
                return
        manifest = self.root_dir / "annotations.json"
        if manifest.is_file():
            try:
                rows = _read_json(manifest)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read ReasonSeg annotations.json: %s", exc)
                return
            if isinstance(rows, dict):
                rows = rows.get("annotations") or rows.get("images") or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                image = _find_image(self.root_dir, row.get("image") or row.get("file_name"))
                text = str(row.get("rewritten_text") or row.get("text") or row.get("query") or "").strip()
                if image is None or not text:
                    continue
                self.samples.append({"image": image, "annotation": None, "expression": text, "row": row})

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        expression, native = self._expression_and_mask(sample)
        if native is None:
            gt_mask = torch.zeros(self.resolution, self.resolution, dtype=torch.bool)
            target_rgb = torch.zeros(3, self.resolution, self.resolution)
        else:
            mask = torch.from_numpy(np.ascontiguousarray(native)).float().view(1, 1, *native.shape)
            resized = (
                torch.nn.functional.interpolate(
                    mask, size=(self.resolution, self.resolution), mode="nearest"
                )
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
        if self.native_gt:
            item["gt_mask_native"] = (
                torch.from_numpy(np.ascontiguousarray(native)).bool()
                if native is not None
                else gt_mask
            )
        return item

    def _expression_and_mask(self, sample: dict) -> tuple[str, np.ndarray | None]:
        if sample.get("annotation") is not None:
            payload = _read_json(sample["annotation"])
            if not isinstance(payload, dict):
                payload = {}
            expression = str(
                payload.get("rewritten_text") or payload.get("text") or payload.get("query") or ""
            ).strip()
            native = self._mask_from_shapes(payload.get("shapes") or [], sample["image"])
            return expression, native
        row = sample.get("row") or {}
        return str(sample.get("expression") or ""), self._mask_from_row(row, sample["image"])

    def _mask_from_shapes(self, shapes: list, image_path: Path) -> np.ndarray | None:
        if not shapes:
            return None
        with Image.open(image_path) as source:
            width, height = source.size
        mask = np.zeros((height, width), dtype=np.uint8)
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            points = shape.get("points") or []
            if len(points) < 3:
                continue
            pts = np.asarray(points, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2:
                continue
            _fill_polygon(mask, pts)
        return mask if mask.any() else None

    def _mask_from_row(self, row: dict, image_path: Path) -> np.ndarray | None:
        mask_name = row.get("mask") or row.get("target")
        mask_path = _find_image(self.root_dir, mask_name) if mask_name else None
        if mask_path is not None:
            array = np.asarray(Image.open(mask_path).convert("L"))
            return (array > 127).astype(np.uint8)
        return self._mask_from_shapes(row.get("shapes") or [], image_path)


class GenAIBenchDataset(BaseVisionDataset):
    """GenAI-Bench prompt set. Images are optional.

    Layouts::

        root/prompts.json          {"id": "prompt", ...} or [{"prompt": "..."}]
        root/prompts.jsonl         {"prompt": "...", "image": "optional.jpg"}
        root/metadata.jsonl

    The paper metric is a human pairwise win rate against the untuned base.
    This loader only prepares prompts (and optional references) for a proxy.
    """

    TASK_TYPE = "generation"

    def __init__(self, root_dir, split="val", resolution=512, **kwargs) -> None:
        self.has_images = False
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            return
        jsonl_names = ("prompts.jsonl", "metadata.jsonl", f"{self.split}.jsonl")
        for name in jsonl_names:
            path = self.root_dir / name
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = str(row.get("prompt") or row.get("caption") or row.get("text") or "").strip()
                if not prompt:
                    continue
                image = _find_image(self.root_dir, row.get("image") or row.get("file_name"))
                self.samples.append({"prompt": prompt, "image": image})
            if self.samples:
                self.has_images = any(sample["image"] is not None for sample in self.samples)
                return
        for name in ("prompts.json", "genai_bench.json"):
            path = self.root_dir / name
            if not path.is_file():
                continue
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
                items = payload["prompts"]
            elif isinstance(payload, dict):
                items = [
                    {"prompt": value}
                    for value in payload.values()
                    if isinstance(value, str) and value.strip()
                ]
            else:
                continue
            for item in items:
                if isinstance(item, str):
                    prompt = item.strip()
                    image = None
                elif isinstance(item, dict):
                    prompt = str(item.get("prompt") or item.get("caption") or item.get("text") or "").strip()
                    image = _find_image(self.root_dir, item.get("image") or item.get("file_name"))
                else:
                    continue
                if prompt:
                    self.samples.append({"prompt": prompt, "image": image})
            if self.samples:
                self.has_images = any(sample["image"] is not None for sample in self.samples)
                return

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if sample["image"] is None:
            raise RuntimeError(
                "GenAI-Bench sample is prompt-only. The paper metric is a human "
                "pairwise win rate; generate images first or use "
                "scripts/evaluate_generation_preservation.py for a CLIP/FID proxy."
            )
        image = self._load_image(sample["image"])
        target_rgb = (image + 1.0) * 0.5
        return {
            "prompt": sample["prompt"],
            "image": image,
            "target_rgb": target_rgb,
        }


class ImgEditDataset(BaseVisionDataset):
    """ImgEdit instruction pairs: source image + edit instruction.

    Layouts::

        root/metadata.jsonl   {"image": "src.jpg", "instruction": "...", "target": "opt.png"}
        root/images + instructions.json
    """

    TASK_TYPE = "generation"

    def __init__(self, root_dir, split="val", resolution=512, **kwargs) -> None:
        self.has_images = False
        super().__init__(root_dir, split, resolution, **kwargs)

    def _scan(self) -> None:
        if not self.root_dir.exists():
            return
        for name in ("metadata.jsonl", "edits.jsonl", f"{self.split}.jsonl"):
            path = self.root_dir / name
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = str(
                    row.get("instruction") or row.get("prompt") or row.get("edit") or ""
                ).strip()
                image = _find_image(self.root_dir, row.get("image") or row.get("source"))
                target = _find_image(self.root_dir, row.get("target") or row.get("edited"))
                if image is None or not instruction:
                    continue
                self.samples.append(
                    {"image": image, "target": target, "prompt": instruction}
                )
            if self.samples:
                self.has_images = True
                return
        instructions = self.root_dir / "instructions.json"
        images = self.root_dir / "images"
        if instructions.is_file() and images.is_dir():
            try:
                mapping = _read_json(instructions)
            except (OSError, json.JSONDecodeError):
                mapping = {}
            if isinstance(mapping, dict):
                for name, prompt in mapping.items():
                    image = _find_image(images, str(name))
                    if image is None or not str(prompt).strip():
                        continue
                    self.samples.append({"image": image, "target": None, "prompt": str(prompt).strip()})
            self.has_images = bool(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        if sample["target"] is not None:
            target = self._load_image(sample["target"])
            target_rgb = (target + 1.0) * 0.5
        else:
            target_rgb = (image + 1.0) * 0.5
        return {
            "prompt": sample["prompt"],
            "image": image,
            "target_rgb": target_rgb,
        }
