"""Evaluate portable RES manifests without the original RefCOCO pickles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..referring_contract import BINARY_REFERRING_ENCODING, build_binary_referring_prompt
from ..res_manifest import portable_path, sha256_file


class RESManifestDataset(Dataset):
    """One held-out expression and its native binary mask per sample.

    ``split`` must explicitly select ``val`` or ``test``. ``source_split``
    retains official ``testA``/``testB`` subsets within the normalized test
    split. Assets must be portable paths under dataset_root; their dimensions
    and pixels are checked on access or for the whole subset by validate_assets.
    This adapter intentionally requires the versioned binary manifest contract.
    """

    TASK_TYPE = "referring"

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        manifest: str = "pairs.jsonl",
        source_dataset: str | None = None,
        source_protocol: str | None = None,
        source_split: str | None = None,
        resolution: int = 512,
    ) -> None:
        if split not in {"val", "test"}:
            raise ValueError("RES evaluation split must explicitly be 'val' or 'test'")
        if not isinstance(resolution, int) or isinstance(resolution, bool) or resolution <= 0:
            raise ValueError("resolution must be a positive integer")
        if source_split is not None and source_split not in (
            {"val"} if split == "val" else {"test", "testA", "testB"}
        ):
            raise ValueError("source_split is incompatible with the evaluation split")
        self.root_dir = Path(dataset_root).resolve()
        if (self.root_dir / "BUILD_INCOMPLETE").exists():
            raise ValueError("RES dataset build is incomplete")
        self.manifest_path = portable_path(self.root_dir, manifest)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.resolution = resolution
        self.filters = {
            "split": split,
            "source_dataset": source_dataset,
            "source_protocol": source_protocol,
            "source_split": source_split,
        }
        self.samples: list[dict] = []
        identifiers: set[str] = set()
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"RES manifest line {line_number} must be an object")
                if any(
                    value is not None and row.get(key) != value
                    for key, value in self.filters.items()
                ):
                    continue
                if (
                    row.get("task") != "referring"
                    or row.get("encoding") != BINARY_REFERRING_ENCODING
                ):
                    raise ValueError(
                        f"RES manifest line {line_number} requires referring/{BINARY_REFERRING_ENCODING}"
                    )
                for key in ("sample_id", "source_dataset", "source_protocol", "source_split"):
                    if not isinstance(row.get(key), str) or not row[key].strip():
                        raise ValueError(f"RES manifest line {line_number}: missing {key}")
                if row["source_split"] not in (
                    {"val"} if split == "val" else {"test", "testA", "testB"}
                ):
                    raise ValueError(f"RES manifest line {line_number}: inconsistent source_split")
                prompt = build_binary_referring_prompt(row.get("expression"))
                if row.get("prompt") != prompt:
                    raise ValueError(
                        f"RES manifest line {line_number}: prompt differs from binary expression contract"
                    )
                if row["sample_id"] in identifiers:
                    raise ValueError(f"Duplicate RES sample_id={row['sample_id']}")
                identifiers.add(row["sample_id"])
                for key in ("width", "height"):
                    if (
                        not isinstance(row.get(key), int)
                        or isinstance(row[key], bool)
                        or row[key] <= 0
                    ):
                        raise ValueError(f"RES manifest line {line_number}: invalid {key}")
                for key in ("input", "target"):
                    path = portable_path(self.root_dir, row.get(key))
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"RES sample {row['sample_id']}: missing {key}: {path}"
                        )
                self.samples.append(row)
        if not self.samples:
            raise ValueError(f"No RES evaluation samples match {self.filters}")

    def __len__(self) -> int:
        return len(self.samples)

    def _native_assets(self, row: dict) -> tuple[Image.Image, np.ndarray]:
        with Image.open(portable_path(self.root_dir, row["input"])) as source:
            source.load()
            image = source.convert("RGB")
        with Image.open(portable_path(self.root_dir, row["target"])) as source:
            if source.format != "PNG" or source.mode not in {"1", "L", "RGB"}:
                raise ValueError(
                    f"RES sample {row['sample_id']}: target must be a binary RGB/grayscale PNG"
                )
            source.load()
            if source.size != image.size or source.size != (row["width"], row["height"]):
                raise ValueError(
                    f"RES sample {row['sample_id']}: native image/mask/metadata dimensions differ"
                )
            rgb = np.asarray(source.convert("RGB"))
        if not np.isin(rgb, [0, 255]).all() or not (rgb == rgb[..., :1]).all():
            raise ValueError(
                f"RES sample {row['sample_id']}: target must contain only black and white pixels"
            )
        native = np.ascontiguousarray(rgb[..., 0] == 255)
        if not native.any():
            raise ValueError(f"RES sample {row['sample_id']}: empty foreground mask")
        return image, native

    def __getitem__(self, index: int) -> dict:
        row = self.samples[index]
        image, native = self._native_assets(row)
        resized = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        image_tensor = (
            torch.from_numpy(np.array(resized).transpose(2, 0, 1).copy()).float() / 127.5 - 1
        )
        native_tensor = torch.from_numpy(native)
        small = torch.nn.functional.interpolate(
            native_tensor.float()[None, None],
            size=(self.resolution, self.resolution),
            mode="nearest",
        )[0, 0].bool()
        return {
            **row,
            "image": image_tensor,
            "image_range": "minus_one_one",
            "target_rgb": small.float().unsqueeze(0).expand(3, -1, -1).clone(),
            "gt_mask": small,
            "gt_mask_native": native_tensor,
        }

    def validate_assets(self) -> dict:
        """Read all selected native assets and report preserved source counts."""
        checked: set[tuple[str, str, int, int]] = set()
        sources: Counter[str] = Counter()
        for row in self.samples:
            pair = (row["input"], row["target"], row["width"], row["height"])
            if pair not in checked:
                self._native_assets(row)
                checked.add(pair)
            sources[f"{row['source_dataset']}/{row['source_protocol']}/{row['source_split']}"] += 1
        return {
            "samples": len(self),
            "unique_images": len({row["input"] for row in self.samples}),
            "unique_masks": len({row["target"] for row in self.samples}),
            "source_counts": dict(sorted(sources.items())),
        }
