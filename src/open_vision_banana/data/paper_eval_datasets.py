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


def _reasonseg_queries(payload: dict) -> list[str]:
    """Read all expressions from the selected text field without joining them."""
    if not isinstance(payload, dict):
        raise ValueError("ReasonSeg annotation must be an object")
    value = payload.get("rewritten_text") or payload.get("text") or payload.get("query")
    if isinstance(value, list):
        if not value or any(not isinstance(x, str) or not x.strip() for x in value):
            raise ValueError("ReasonSeg query list must contain nonempty strings")
        return [query.strip() for query in value]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ReasonSeg requires a nonempty query")
    return [value.strip()]


class ReasonSegDataset(BaseVisionDataset):
    """ReasonSeg val/test referring queries (LISA JSON-next-to-image layout).

    Expected layouts::

        root/val/*.jpg + root/val/*.json
        root/*.jpg + root/*.json

    Each JSON uses the LISA fields ``text`` / ``is_sentence`` / ``shapes``.
    The paper's gIoU uses Gemini 2.5 Pro to rewrite reasoning queries first;
    this adapter scores the raw query unless ``rewritten_text`` is present.
    Each expression becomes a separate evaluation item in source order;
    ``query_mode='first'`` opts into the historical first-expression subset.
    """

    TASK_TYPE = "referring"

    def __init__(
        self,
        root_dir,
        split="val",
        resolution=512,
        native_gt=False,
        query_mode="all",
        annotation_mode="legacy_union",
        **kwargs,
    ) -> None:
        self.native_gt = bool(native_gt)
        if query_mode not in {"all", "first"}:
            raise ValueError("ReasonSeg query_mode must be 'all' or 'first'")
        self.query_mode = query_mode
        if annotation_mode not in {"legacy_union", "lisa_v1"}:
            raise ValueError("Unknown ReasonSeg annotation_mode")
        self.annotation_mode = annotation_mode
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
                self._append_queries(img_path, _read_json(json_path), annotation=json_path)
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
                if image is None:
                    continue
                self._append_queries(image, row)

    def _append_queries(self, image: Path, payload: dict, annotation: Path | None = None) -> None:
        queries = _reasonseg_queries(payload)
        if self.query_mode == "first":
            queries = queries[:1]
        try:
            image_id = image.resolve().relative_to(self.root_dir.resolve()).as_posix()
        except ValueError:
            image_id = image.resolve().as_posix()
        for query_index, expression in enumerate(queries):
            self.samples.append({
                "image": image, "annotation": annotation, "row": payload,
                "expression": expression, "image_id": image_id, "query_index": query_index,
            })

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        expression, native = self._expression_and_mask(sample)
        valid_native = native != 255 if native is not None else None
        if native is not None:
            native = (native == 1).astype(np.uint8)
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
            "image_id": sample["image_id"],
            "query_index": sample["query_index"],
        }
        if self.native_gt:
            item["gt_mask_native"] = (
                torch.from_numpy(np.ascontiguousarray(native)).bool()
                if native is not None
                else gt_mask
            )
        if self.annotation_mode == "lisa_v1" and valid_native is not None:
            valid_tensor = torch.from_numpy(np.ascontiguousarray(valid_native))
            item["valid_mask"] = torch.nn.functional.interpolate(
                valid_tensor.float()[None, None],
                size=(self.resolution, self.resolution), mode="nearest",
            )[0, 0].bool()
            if self.native_gt:
                item["valid_mask_native"] = valid_tensor
        return item

    def _expression_and_mask(self, sample: dict) -> tuple[str, np.ndarray | None]:
        if sample.get("annotation") is not None:
            payload = sample["row"]
            expression = sample["expression"]
            native = self._mask_from_shapes(payload.get("shapes") or [], sample["image"])
            return expression, native
        row = sample.get("row") or {}
        return str(sample.get("expression") or ""), self._mask_from_row(row, sample["image"])

    def _mask_from_shapes(self, shapes: list, image_path: Path) -> np.ndarray | None:
        if not shapes:
            return None
        with Image.open(image_path) as source:
            width, height = source.size
        if self.annotation_mode == "lisa_v1":
            return reasonseg_lisa_labels(shapes, height, width)
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


def reasonseg_lisa_labels(shapes: list, height: int, width: int) -> np.ndarray:
    """LISA-compatible polygon rasterization: background=0, target=1, ignore=255.

    Protocol reference: JIA-Lab-research/LISA, utils/data_processing.py
    (Apache-2.0). Uses integer truncation and descending rasterized polygon area;
    small polygons override large ones, including target/ignore overlaps.
    """
    import cv2

    polygons = []
    areas = []
    for shape in shapes:
        if str(shape["label"]).lower() == "flag":
            continue
        points = np.asarray([shape["points"]], dtype=np.int32)
        if points.ndim != 3 or points.shape[1] < 3 or points.shape[2] != 2:
            raise ValueError("ReasonSeg polygon requires at least three 2D points")
        region = np.zeros((height, width), dtype=np.uint8)
        cv2.polylines(region, points, True, 1, 1)
        cv2.fillPoly(region, points, 1)
        polygons.append((points, 255 if "ignore" in shape["label"].lower() else 1))
        areas.append(region.sum())
    labels = np.zeros((height, width), dtype=np.uint8)
    for index in np.argsort(areas)[::-1]:
        points, value = polygons[index]
        cv2.polylines(labels, points, True, value, 1)
        cv2.fillPoly(labels, points, value)
    return labels


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
