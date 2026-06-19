"""Utilities for building generation preservation datasets.

Supports two modes:

1. **Synthetic** — fast, no GPU; coloured placeholder images + curated prompts.
2. **FLUX** — generate real images from prompts via FLUX.1 (requires GPU + weights).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

DEFAULT_GENERATION_PROMPTS = [
    "A photorealistic landscape with mountains and a lake at sunset.",
    "A cozy coffee shop interior with warm lighting and wooden furniture.",
    "A futuristic city skyline with flying vehicles and neon lights.",
    "A close-up portrait of a smiling person in natural daylight.",
    "A bowl of fresh fruit on a marble kitchen counter.",
    "An astronaut walking on a rocky alien planet under two moons.",
    "A medieval castle on a hill surrounded by fog.",
    "A golden retriever playing in a green meadow.",
    "An abstract painting with bold red, blue, and yellow strokes.",
    "A rainy street in Tokyo at night with reflections on the pavement.",
    "A minimalist Scandinavian living room with plants and sunlight.",
    "A vintage red sports car parked beside the ocean.",
    "A snow-covered forest path with soft morning light.",
    "A chef preparing sushi in a Japanese restaurant kitchen.",
    "A hot air balloon festival over rolling countryside hills.",
    "A coral reef underwater scene with colourful tropical fish.",
    "A bookshelf in a quiet library with afternoon sunbeams.",
    "A bouquet of wildflowers in a ceramic vase on a windowsill.",
    "A campfire under the Milky Way in a desert landscape.",
    "A bustling farmers market with fresh vegetables and flowers.",
]


def load_prompts_file(path: str | Path) -> list[str]:
    """Load prompts from a text or JSON file.

    Supported formats:
    - ``.txt``: one prompt per line (blank lines skipped)
    - ``.json``: JSON array of strings, or ``{"prompts": [...]}``
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(p).strip() for p in data if str(p).strip()]
        if isinstance(data, dict) and "prompts" in data:
            return [str(p).strip() for p in data["prompts"] if str(p).strip()]
        raise ValueError(f"Unsupported JSON prompts format in {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _synthetic_image(index: int, size: int, seed: int) -> Image.Image:
    """Create a deterministic colourful synthetic RGB image."""
    rng = random.Random(seed + index * 9973)
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)

    c1 = tuple(rng.randint(30, 220) for _ in range(3))
    c2 = tuple(rng.randint(30, 220) for _ in range(3))
    for y in range(size):
        t = y / max(size - 1, 1)
        row_color = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
        draw.line([(0, y), (size, y)], fill=row_color)

    for _ in range(rng.randint(2, 6)):
        x0 = rng.randint(0, size - 1)
        y0 = rng.randint(0, size - 1)
        x1 = rng.randint(0, size - 1)
        y1 = rng.randint(0, size - 1)
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        if x0 == x1:
            x1 = min(x0 + 1, size - 1)
        if y0 == y1:
            y1 = min(y0 + 1, size - 1)
        color = tuple(rng.randint(0, 255) for _ in range(3))
        draw.ellipse([x0, y0, x1, y1], fill=color)

    return img


def build_synthetic_generation_dataset(
    output_dir: str | Path,
    num_samples: int = 100,
    resolution: int = 512,
    seed: int = 42,
    prompts: list[str] | None = None,
) -> Path:
    """Write a synthetic generation dataset to disk.

    Creates::

        <output_dir>/
          images/00000.jpg …
          prompts.json
          metadata.jsonl

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    prompt_pool = prompts or DEFAULT_GENERATION_PROMPTS
    if not prompt_pool:
        raise ValueError("At least one prompt is required")

    prompts_map: dict[str, str] = {}
    metadata_rows: list[dict] = []

    for i in range(num_samples):
        name = f"{i:05d}.jpg"
        img_path = img_dir / name
        img = _synthetic_image(i, resolution, seed)
        img.save(img_path, quality=92)

        prompt = prompt_pool[i % len(prompt_pool)]
        prompts_map[name] = prompt
        metadata_rows.append({"image": name, "prompt": prompt})

    with (output_dir / "prompts.json").open("w", encoding="utf-8") as f:
        json.dump(prompts_map, f, indent=2, ensure_ascii=False)

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Built synthetic generation dataset: %d samples → %s", num_samples, output_dir)
    return output_dir


def build_flux_generation_dataset(
    output_dir: str | Path,
    prompts: list[str],
    *,
    model_name: str = "black-forest-labs/FLUX.1-schnell",
    resolution: int = 512,
    device: str = "auto",
    seed: int = 42,
    num_inference_steps: int = 1,
) -> Path:
    """Generate images with FLUX.1 and write a generation dataset.

    Requires GPU, diffusers, and downloaded FLUX weights.
    """
    if not prompts:
        raise ValueError("At least one prompt is required for FLUX generation")

    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from open_vision_banana.inference.generator import BaseGenerator

    generator = BaseGenerator(
        model_name=model_name,
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    prompts_map: dict[str, str] = {}
    metadata_rows: list[dict] = []

    for i, prompt in enumerate(prompts):
        name = f"{i:05d}.jpg"
        logger.info("Generating [%d/%d]: %s", i + 1, len(prompts), prompt[:80])

        result = generator.generate(
            prompt=prompt,
            height=resolution,
            width=resolution,
            num_inference_steps=num_inference_steps,
            seed=seed + i,
        )

        # result: [B, 3, H, W] in [0, 1]
        tensor = result[0].clamp(0, 1)
        arr = (tensor.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        Image.fromarray(arr).save(img_dir / name, quality=95)

        prompts_map[name] = prompt
        metadata_rows.append({"image": name, "prompt": prompt})

    with (output_dir / "prompts.json").open("w", encoding="utf-8") as f:
        json.dump(prompts_map, f, indent=2, ensure_ascii=False)

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Built FLUX generation dataset: %d samples → %s", len(prompts), output_dir)
    return output_dir


__all__ = [
    "DEFAULT_GENERATION_PROMPTS",
    "build_flux_generation_dataset",
    "build_synthetic_generation_dataset",
    "load_prompts_file",
]
