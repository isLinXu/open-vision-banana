#!/usr/bin/env python3
"""Hugging Face Hub publishing script for Vision Banana LoRA checkpoints.

After training a LoRA checkpoint, use this script to package and publish it
to Hugging Face Hub with an auto-generated model card.

Usage:
    # Publish a trained checkpoint
    python scripts/publish_to_hub.py \
        --checkpoint output/checkpoints/best_model.pt \
        --repo-id open-vision-banana/vision-banana-depth-v1 \
        --task depth \
        --description "Depth estimation LoRA trained on Hypersim"

    # Dry run (show what would be published without uploading)
    python scripts/publish_to_hub.py \
        --checkpoint output/checkpoints/best_model.pt \
        --repo-id open-vision-banana/vision-banana-depth-v1 \
        --dry-run

    # Update registry after successful publish
    python scripts/publish_to_hub.py \
        --checkpoint output/checkpoints/best_model.pt \
        --repo-id open-vision-banana/vision-banana-depth-v1 \
        --update-registry
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent.parent / "configs" / "checkpoint_registry.yaml"

PAPER_CITATION = """@article{gabeur2026image,
  title={Image Generators are Generalist Vision Learners},
  author={Gabeur, Valentin and Long, Shangbang and Peng, Songyou and Voigtlaender, Paul and Sun, Shuyang and Bao, Yanan and Truong, Karen and Wang, Zhicheng and Zhou, Wenlei and Barron, Jonathan T. and Genova, Kyle and Kannen, Nithish and Ben, Sherry and Li, Yandong and Guo, Mandy and Yogin, Suhas and Gu, Yiming and Chen, Huizhong and Wang, Oliver and Xie, Saining and Zhou, Howard and He, Kaiming and Funkhouser, Thomas and Alayrac, Jean-Baptiste and Soricut, Radu},
  journal={arXiv preprint arXiv:2604.20329},
  year={2026}
}"""


def _build_model_card(
    task: str,
    description: str,
    repo_id: str,
    base_model: str = "black-forest-labs/FLUX.1-schnell",
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
) -> str:
    """Build a Hugging Face Model Card in markdown format."""
    metrics = metrics or {}
    training_config = training_config or {}

    card = f"""# Vision Banana LoRA — {task.title()} Estimation

## Model Description

{description}

This is a LoRA (Low-Rank Adaptation) checkpoint for the Open Vision Banana
project, instruction-tuned on top of `{base_model}` for {task} estimation via
RGB parameterization.

This open-backbone checkpoint is independent from the original paper's closed
Nano Banana Pro model. Paper metrics do not apply to this checkpoint unless
they are reproduced and reported in the checkpoint-specific Metrics section.

## Base Model

- **Base**: [{base_model}](https://huggingface.co/{base_model})
- **Architecture**: Base-model architecture with LoRA adapters
- **Parameters**: See the checkpoint training metadata below

## Training

"""
    if training_config:
        card += "| Parameter | Value |\n|-----------|-------|\n"
        for k, v in training_config.items():
            card += f"| {k} | {v} |\n"
        card += "\n"

    card += """## Metrics

"""
    if metrics:
        card += "| Metric | Value |\n|--------|-------|\n"
        for k, v in metrics.items():
            card += f"| {k} | {v:.4f} |\n"
        card += "\n"
    else:
        card += "Metrics not yet reported. See evaluation section below.\n\n"

    card += f"""## Usage

```python
from open_vision_banana import VisionBanana

model = VisionBanana.from_pretrained(
    "{base_model}",
    lora_path="hf:{repo_id}",
)

# The API expects a normalized torch.Tensor, not a file path.
result = model.predict(input_tensor, task="{task}")
```

## Limitations

- Requires the exact `{base_model}` base model and matching adapter configuration
- LoRA weights are task-specific; multi-task requires loading multiple LoRAs
- RGB parameterization and quantization introduce checkpoint-specific roundtrip error
- Results from the original Nano Banana Pro paper must not be attributed to this checkpoint

## Citation

```bibtex
{PAPER_CITATION}
```

## License

The project code is MIT licensed. Checkpoint distribution and use also remain
subject to the selected base model and training-dataset licenses.
"""
    return card


def _update_registry(
    alias: str,
    repo_id: str,
    task: str,
    status: str = "available",
) -> None:
    """Update the local checkpoint registry YAML."""
    if not REGISTRY_PATH.exists():
        logger.warning("Registry file not found: %s", REGISTRY_PATH)
        return

    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f) or {}

    checkpoints = data.setdefault("checkpoints", {})
    checkpoints[alias] = {
        "repo_id": repo_id,
        "revision": "main",
        "status": status,
        "description": f"Published {task} LoRA checkpoint",
        "tasks": [task],
    }

    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    logger.info("Updated registry: %s → %s (%s)", alias, repo_id, status)


def publish_checkpoint(
    checkpoint_path: Path,
    repo_id: str,
    task: str,
    description: str,
    base_model: str = "black-forest-labs/FLUX.1-schnell",
    metrics: dict[str, float] | None = None,
    training_config: dict[str, Any] | None = None,
    dry_run: bool = False,
    update_registry: bool = False,
) -> int:
    """Package and publish a checkpoint to Hugging Face Hub."""
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        return 1

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    # Build model card
    model_card = _build_model_card(
        task=task,
        description=description,
        repo_id=repo_id,
        base_model=base_model,
        metrics=metrics,
        training_config=training_config,
    )

    if dry_run:
        print("=" * 60)
        print("DRY RUN — Would publish to Hugging Face Hub")
        print("=" * 60)
        print(f"  Repo:     {repo_id}")
        print(f"  File:     {checkpoint_path}")
        print(f"  Task:     {task}")
        print(f"  Card:     {len(model_card)} chars")
        print("=" * 60)
        print(model_card[:2000])
        print("... [truncated]" if len(model_card) > 2000 else "")
        return 0

    # Create repo (if not exists)
    api = HfApi()
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        logger.info("Repository ready: %s", repo_id)
    except Exception as exc:
        logger.error("Failed to create repo: %s", exc)
        return 1

    # Upload checkpoint
    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo="model.safetensors",
            repo_id=repo_id,
        )
        logger.info("Uploaded checkpoint to %s/model.safetensors", repo_id)
    except Exception as exc:
        logger.error("Upload failed: %s", exc)
        return 1

    # Upload model card
    try:
        api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        logger.info("Uploaded model card to %s/README.md", repo_id)
    except Exception as exc:
        logger.error("Model card upload failed: %s", exc)
        return 1

    logger.info("✅ Successfully published to https://huggingface.co/%s", repo_id)

    if update_registry:
        alias = repo_id.split("/")[-1].replace("-", "_")
        _update_registry(alias, repo_id, task, status="available")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish Vision Banana LoRA to Hugging Face Hub")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Hub repo ID (org/model)")
    parser.add_argument("--task", type=str, required=True, choices=["depth", "normal", "semantic", "instance", "referring"])
    parser.add_argument("--description", type=str, default="", help="Model description")
    parser.add_argument("--base-model", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be published")
    parser.add_argument("--update-registry", action="store_true", help="Update local checkpoint registry")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return publish_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        repo_id=args.repo_id,
        task=args.task,
        description=args.description or f"Vision Banana {args.task} LoRA checkpoint",
        base_model=args.base_model,
        dry_run=args.dry_run,
        update_registry=args.update_registry,
    )


if __name__ == "__main__":
    sys.exit(main())
