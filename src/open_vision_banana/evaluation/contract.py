"""Training-Evaluation Encoding Contract.

Loads and validates the encoding contract that specifies how training targets
are encoded (colormap, palette, prompts).  All evaluation scripts MUST use
these parameters for decoding to ensure consistency with training.

The contract file is at ``configs/encoding_contract.yaml``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_CONTRACT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "encoding_contract.yaml"


@dataclass(frozen=True)
class DepthContract:
    """Depth encoding specification."""

    encoding: str  # "grayscale" or colormap name
    colormap: str | None  # None for grayscale
    power_lam: float = -3.0
    power_c: float = 10.0 / 3.0
    decode_hint: str = ""

    def decode_rgb_to_depth(
        self,
        rgb: torch.Tensor,
        *,
        return_metric: bool = False,
    ) -> torch.Tensor:
        """Decode RGB tensor to depth using the contract-specified encoding.

        Args:
            rgb: [B, 3, H, W] in [0, 1] — model output or target RGB.
            return_metric: If True, apply inverse power transform for metric depth.
                If False (default), return normalized depth in [0, 1].

        Returns:
            Depth tensor [B, 1, H, W].
        """
        if self.encoding == "grayscale":
            # Grayscale: R == G == B, take mean for robustness
            normalized = rgb.mean(dim=1, keepdim=True)
        else:
            # Colormap-based decoding
            from ..core.depth_mapper import DepthRGBMapper

            mapper = DepthRGBMapper(colormap=self.colormap, lam=self.power_lam, c=self.power_c)
            normalized = mapper.rgb_to_normalized(rgb)

        if return_metric:
            from ..core.depth_mapper import DepthRGBMapper

            mapper = DepthRGBMapper(lam=self.power_lam, c=self.power_c)
            return mapper.normalized_to_depth(normalized)
        return normalized


@dataclass(frozen=True)
class SemanticContract:
    """Semantic segmentation encoding specification."""

    palette_source: str
    n_classes: int
    palette_path: str
    prompt: str

    def load_palette(self) -> dict[str, tuple[int, int, int]]:
        """Load the color palette from the contract-specified path.

        Returns:
            Dict mapping class name → (R, G, B) tuple with values in [0, 255].
        """
        palette_file = Path(__file__).resolve().parent.parent.parent.parent / self.palette_path
        if not palette_file.is_file():
            raise FileNotFoundError(f"Semantic palette not found: {palette_file}")
        data = json.loads(palette_file.read_text(encoding="utf-8"))
        raw_palette = data.get("palette", data)
        return {
            name: tuple(color)
            for name, color in raw_palette.items()
        }


@dataclass(frozen=True)
class NormalContract:
    """Normal encoding specification."""

    encoding: str = "standard_xyz_to_rgb"
    prompt: str = ""


@dataclass(frozen=True)
class EncodingContract:
    """Complete training-evaluation encoding contract."""

    depth: DepthContract
    semantic: SemanticContract
    normal: NormalContract
    prompts: dict[str, str] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    def get_prompt(self, task: str) -> str:
        """Get the canonical prompt for a task.

        Falls back to the task-specific prompt in the semantic/normal contracts
        if not found in the top-level prompts section.
        """
        if task in self.prompts:
            return self.prompts[task]
        # Fallback to task-specific contract prompt
        if task == "depth" and self.depth.decode_hint:
            return self.prompts.get("depth", "")
        if task == "semantic":
            return self.semantic.prompt
        if task == "normal":
            return self.normal.prompt
        return ""


def load_encoding_contract(
    contract_path: str | Path | None = None,
) -> EncodingContract:
    """Load the encoding contract from YAML.

    Args:
        contract_path: Path to the contract YAML file.
            If None, uses the default ``configs/encoding_contract.yaml``.

    Returns:
        Parsed EncodingContract dataclass.
    """
    try:
        import yaml
    except ImportError:
        # Fallback: parse YAML manually for simple key-value structure
        return _load_contract_fallback(contract_path)

    path = Path(contract_path) if contract_path else _CONTRACT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Encoding contract not found: {path}. "
            f"Run the audit pipeline first or create the file manually."
        )

    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    depth_data = data.get("depth", {})
    semantic_data = data.get("semantic", {})
    normal_data = data.get("normal", {})

    return EncodingContract(
        depth=DepthContract(
            encoding=depth_data.get("encoding", "grayscale"),
            colormap=depth_data.get("colormap"),
            power_lam=depth_data.get("power_lam", -3.0),
            power_c=depth_data.get("power_c", 10.0 / 3.0),
            decode_hint=depth_data.get("decode_hint", ""),
        ),
        semantic=SemanticContract(
            palette_source=semantic_data.get("palette_source", "ade20k_150_seed42"),
            n_classes=semantic_data.get("n_classes", 150),
            palette_path=semantic_data.get("palette_path", "configs/semantic_palette.json"),
            prompt=semantic_data.get("prompt", ""),
        ),
        normal=NormalContract(
            encoding=normal_data.get("encoding", "standard_xyz_to_rgb"),
            prompt=normal_data.get("prompt", ""),
        ),
        prompts=data.get("prompts", {}),
        model=data.get("model", {}),
    )


def _load_contract_fallback(contract_path: str | Path | None) -> EncodingContract:
    """Load contract without PyYAML — parse the known structure manually."""
    path = Path(contract_path) if contract_path else _CONTRACT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Encoding contract not found: {path}")

    # Simple line-by-line parse for key fields
    text = path.read_text(encoding="utf-8")
    depth_encoding = "grayscale"
    depth_colormap = None
    semantic_palette_path = "configs/semantic_palette.json"
    semantic_n_classes = 150

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("encoding:"):
            val = stripped.split(":", 1)[1].strip().split("#")[0].strip()
            if val and val != "null":
                depth_encoding = val
        elif stripped.startswith("colormap:") and "depth" not in stripped:
            val = stripped.split(":", 1)[1].strip().split("#")[0].strip()
            if val and val != "null":
                depth_colormap = val
        elif stripped.startswith("palette_path:"):
            semantic_palette_path = stripped.split(":", 1)[1].strip().split("#")[0].strip()
        elif stripped.startswith("n_classes:"):
            try:
                semantic_n_classes = int(stripped.split(":", 1)[1].strip().split("#")[0].strip())
            except ValueError:
                pass

    # Extract prompts (simple parse)
    prompts: dict[str, str] = {}
    in_prompts = False
    for line in text.splitlines():
        if line.strip() == "prompts:":
            in_prompts = True
            continue
        if in_prompts:
            if line and not line[0].isspace():
                break
            stripped = line.strip()
            if ":" in stripped and not stripped.startswith("#"):
                key, val = stripped.split(":", 1)
                val = val.strip().strip('"').strip("'")
                prompts[key.strip()] = val

    # Extract semantic prompt
    sem_prompt = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("prompt:") and "semantic" in text[: text.index(stripped)]:
            sem_prompt = stripped.split(":", 1)[1].strip().strip('"').strip("'")

    return EncodingContract(
        depth=DepthContract(
            encoding=depth_encoding,
            colormap=depth_colormap,
            decode_hint="",
        ),
        semantic=SemanticContract(
            palette_source="ade20k_150_seed42",
            n_classes=semantic_n_classes,
            palette_path=semantic_palette_path,
            prompt=sem_prompt,
        ),
        normal=NormalContract(),
        prompts=prompts,
    )


__all__ = [
    "EncodingContract",
    "DepthContract",
    "SemanticContract",
    "NormalContract",
    "load_encoding_contract",
]
