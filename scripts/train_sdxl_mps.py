#!/usr/bin/env python3
"""SDXL/SSD-1B LoRA fine-tuning on Apple MPS — full multi-task training.

Trains SSD-1B (SDXL-distilled, 1.3B UNet) or full SDXL (2.6B UNet) with LoRA on:
  - Depth estimation (COCO pseudo + SNVC)
  - Normal estimation (SNVC)
  - Semantic segmentation (ADE20K-style)
  - Edge detection (BSDS500-style)
  - Text-to-image generation (COCO captions)

Uses the unified InstructionTuner + BaseGenerator infrastructure, which handles
SDXL-specific concerns:
  - Dual text encoder (CLIP-L + OpenCLIP-bigG) → concatenated hidden states + pooled embeds
  - added_cond_kwargs (pooled text_embeds + time_ids) required by SDXL UNet
  - scheduler-aware SDXL epsilon/v-prediction objective
  - img2img vision tasks use SAMPLE (x0) prediction (stable target norm)
  - text-to-image uses standard epsilon prediction

Usage:
  python scripts/train_sdxl_mps.py                              # full training, defaults
  python scripts/train_sdxl_mps.py --model sdxl                 # full SDXL (2.6B)
  python scripts/train_sdxl_mps.py --max-steps 50 --dry-run    # smoke test
  python scripts/train_sdxl_mps.py --resume                     # auto-resume
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.data import MixedVisionDataset  # noqa: E402
from open_vision_banana.training.tuner import InstructionTuner, VisionTaskDataset  # noqa: E402
from open_vision_banana.training.utils import enable_gradient_checkpointing  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_sdxl")

PAPER_LORA_RANK = 64
PAPER_LORA_ALPHA = 128
PAPER_VISION_RATIO = 0.1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed all RNGs used by the training and sampling pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dirs(data_dir: str | None) -> list[Path]:
    """Resolve an explicit dataset root or preserve the legacy merged sources."""
    if data_dir is None:
        return [
            PROJECT_ROOT / "data" / "vb_zimage_train",
            PROJECT_ROOT / "data" / "vb_snvc_train",
        ]

    resolved = Path(data_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"Training data directory not found: {resolved}")
    return [resolved]


def combine_training_datasets(
    vision_datasets,
    generation_datasets,
    *,
    balance_data: bool,
    vision_ratio: float,
    seed: int,
):
    """Build the paper-aligned outer generation/vision mixture.

    Vision tasks are balanced within the 10% vision branch. Generation remains
    a separate 90% branch instead of being treated as one more vision task by a
    ``WeightedRandomSampler``.
    """
    from torch.utils.data import ConcatDataset, WeightedRandomSampler

    if not vision_datasets and not generation_datasets:
        raise ValueError("At least one training dataset is required")

    if generation_datasets and vision_datasets:
        generation_dataset = (
            ConcatDataset(generation_datasets)
            if len(generation_datasets) > 1
            else generation_datasets[0]
        )
        vision_sources = vision_datasets
        if not balance_data and len(vision_datasets) > 1:
            vision_sources = [ConcatDataset(vision_datasets)]
        combined = MixedVisionDataset(
            vision_datasets=vision_sources,
            generation_dataset=generation_dataset,
            vision_ratio=vision_ratio,
            seed=seed,
        )
        return combined, None

    datasets = vision_datasets or generation_datasets
    if len(datasets) == 1:
        return datasets[0], None

    combined = ConcatDataset(datasets)
    if not balance_data or not vision_datasets:
        return combined, None

    dataset_sizes = [len(dataset) for dataset in datasets]
    max_size = max(dataset_sizes)
    weights = []
    for size in dataset_sizes:
        weights.extend([max_size / size] * size)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(combined),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return combined, sampler


def main():
    parser = argparse.ArgumentParser(description="SDXL/SSD-1B LoRA training on MPS")
    parser.add_argument("--model", type=str, default="ssd-1b",
                        choices=["ssd-1b", "sdxl"],
                        help="Base model (default: ssd-1b, use sdxl for full 2.6B)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max steps")
    parser.add_argument("--save-every", type=int, default=500, help="Save every N steps")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Checkpoint directory (default: output/<model>_checkpoints)")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--rank", type=int, default=PAPER_LORA_RANK, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=PAPER_LORA_ALPHA, help="LoRA alpha")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Training resolution (must match inference resolution)")
    parser.add_argument("--max-per-task", type=int, default=300,
                        help="Max samples per vision task (balanced sampling)")
    parser.add_argument("--max-generation", type=int, default=3000,
                        help="Max generation samples loaded per source")
    parser.add_argument(
        "--vision-ratio",
        type=float,
        default=PAPER_VISION_RATIO,
        help="Probability of sampling a vision task when generation data is available",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=("Use only this dataset root. If omitted, preserve the legacy behavior "
              "of merging data/vb_zimage_train and data/vb_snvc_train."),
    )
    parser.add_argument("--balance-data", action="store_true", default=True,
                        help="Balance task sampling (oversample minority tasks)")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA")
    parser.add_argument("--resume", action="store_true", help="Auto-resume from latest checkpoint")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from specific checkpoint dir")
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip text-to-image generation task (vision-only training)")
    parser.add_argument("--dry-run", action="store_true", help="Load model and exit")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "core", "depth", "normal", "segmentation", "edge", "generation",
                        "depth_edge", "keypoint", "object_detection", "referring",
                        "instance", "structure", "multiview", "specular_removal"],
                        help="Which task to train on ('core' = depth+normal+segmentation+structure)")
    parser.add_argument("--no-logit-normal", action="store_true",
                        help="Disable logit-normal timestep sampling (use uniform)")
    args = parser.parse_args()

    if not 0.0 < args.vision_ratio < 1.0:
        parser.error("--vision-ratio must be between 0 and 1")

    set_seed(args.seed)

    device = get_device()
    logger.info(f"Device: {device}")

    # ─── Load SDXL/SSD-1B model ───────────────────────────────────────────
    # Use registered model name → resolve to HF repo ID for BaseGenerator.
    # SSD-1B is preferred on MPS (5.5GB vs 7GB for full SDXL).
    # Map short names to HF repo IDs so diffusers can use the local cache.
    _MODEL_ID_MAP = {
        "ssd-1b": "segmind/SSD-1B",
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    }
    model_name = _MODEL_ID_MAP.get(args.model, args.model)
    logger.info(f"Loading {args.model} → {model_name} pipeline (bfloat16)...")
    t0 = time.time()
    generator = BaseGenerator(
        model_name=model_name,
        device=str(device),
        dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Verify model config
    mc = generator.model_config
    logger.info(f"Model family: {mc.family}")
    logger.info(f"Pipeline class: {mc.pipeline_class}")
    logger.info(f"Cross attention dim: {mc.cross_attention_dim}")
    logger.info(f"VAE scaling: {mc.vae_scaling_factor}")
    logger.info(f"VAE latent channels: {mc.vae_latent_channels}")
    logger.info(f"LoRA target modules: {mc.lora_target_modules}")
    logger.info(f"Text encoder max length: {mc.text_encoder_max_length}")
    logger.info(f"Needs added_cond_kwargs: {mc.needs_added_cond_kwargs}")
    logger.info(f"Needs pooled projections: {mc.needs_pooled_projections}")
    logger.info(f"Memory budget: {mc.memory_budget_gb}GB")

    if args.dry_run:
        logger.info("Dry run complete — model loaded successfully")
        return

    # ─── Setup LoRA training ─────────────────────────────────────────────
    logger.info(f"Setting up LoRA: rank={args.rank}, alpha={args.alpha}, lr={args.lr}")

    tuner = InstructionTuner(
        model=generator,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
        use_rslora=True,  # Enable rsLoRA for better high-rank training
        lr=args.lr,
        use_ema=not args.no_ema,
        ema_decay=0.9999,
        use_min_snr=True,
        use_logit_normal_timesteps=not args.no_logit_normal,
        use_img2img_flow=False,
        cfg_dropout_rate=0.1,
        warmup_ratio=0.05,
    )

    # Enable gradient checkpointing for UNet on MPS.
    # diffusers UNet uses `_gradient_checkpointing_func` (set by
    # `enable_gradient_checkpointing()`), not the raw `gradient_checkpointing`
    # flag that utils.enable_gradient_checkpointing toggles.
    unet = getattr(generator.pipe, "unet", None)
    if unet is not None and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on UNet (diffusers native)")
    else:
        enable_gradient_checkpointing(tuner.peft_model)
        logger.info("Gradient checkpointing enabled (generic fallback)")

    # ─── Load training data ──────────────────────────────────────────────
    res = args.resolution
    data_dirs = resolve_data_dirs(args.data_dir)
    if args.data_dir is not None:
        logger.info(f"Using explicit training data directory only: {data_dirs[0]}")
    else:
        logger.info("Using legacy merged training data directories: "
                    + ", ".join(str(path) for path in data_dirs))

    # ─── Load train/val split for proper validation ──────────────────────
    split_data = None
    split_file = data_dirs[0] / "train_val_split.json" if data_dirs else None
    if split_file and split_file.exists():
        import json as _json_split
        with open(split_file) as f:
            split_data = _json_split.load(f)
        logger.info(f"Loaded train_val_split.json: "
                    f"{sum(len(v) for v in split_data['train'].values())} train, "
                    f"{sum(len(v) for v in split_data['val'].values())} val")

    def _filter_by_split(samples, task_name, split_key="train"):
        """Filter dataset samples by train_val_split.json filenames."""
        if split_data is None:
            return samples
        split_files = set()
        # Map vision_task names to split keys
        split_task_map = {
            "depth": "depth", "normal": "normal", "semantic": "segmentation",
            "segmentation": "segmentation", "edge": "depth_edge",
            "structure": "structure", "depth_edge": "depth_edge",
            "keypoint": "keypoint", "object_detection": "object_detection",
            "referring": "referring", "instance": "instance",
            "specular_removal": "specular_removal", "multiview": "multiview",
        }
        split_task_key = split_task_map.get(task_name, task_name)
        if split_task_key in split_data.get(split_key, {}):
            split_files = set(split_data[split_key][split_task_key])
        if not split_files:
            return samples if split_key == "train" else []
        # Filter: keep samples whose image filename is in the split
        filtered = []
        for s in samples:
            img_name = Path(s["image_path"]).name
            if img_name in split_files:
                filtered.append(s)
        return filtered

    # Generation dataset class (defined at function scope for reuse)
    import json as _json_mod
    from PIL import Image as _PIL_Image  # noqa: I001

    class GenerationDataset(VisionTaskDataset):
        """Text-to-image dataset using COCO captions."""
        def __init__(self, data_path, resolution=256, max_samples=None):
            self.samples = []
            self.resolution = resolution
            self.task = None
            prompts_file = Path(data_path) / "prompts.jsonl"
            prompts_map = {}
            if prompts_file.exists():
                with open(prompts_file) as f:
                    for line in f:
                        item = _json_mod.loads(line)
                        prompts_map[item["image"]] = item["prompt"]
            img_dir = Path(data_path) / "images"
            for img_path in sorted(img_dir.glob("*.jpg")):
                prompt = prompts_map.get(img_path.name, "a photograph")
                self.samples.append({
                    "prompt": prompt,
                    "image_path": str(img_path),
                })
            if max_samples is not None and len(self.samples) > max_samples:
                self.samples = self.samples[:max_samples]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            from torchvision import transforms
            s = self.samples[idx]
            img = _PIL_Image.open(s["image_path"]).convert("RGB").resize(
                (self.resolution, self.resolution), _PIL_Image.LANCZOS
            )
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            img_tensor = transform(img)
            # P0-2 fix: text-to-image generation must NOT pass the input image
            # as conditioning. Previously image == target_rgb, which made the
            # img2img flow velocity_target = target - input = 0, teaching the
            # model to output zero velocity (i.e. "generate nothing") for ~22%
            # of all training steps. Returning image=None routes generation
            # through the pure t2i branch: noisy = (1-t)*target + t*noise,
            # velocity_target = target - noise (standard flow matching).
            return {
                "prompt": s["prompt"], "image": None, "target_rgb": img_tensor,
                "target_rgb_range": "-1_1",
            }

    datasets = []
    generation_datasets = []

    def _load_task_dirs(task_name, subdir, vision_task, max_per_task=None, split_key="train"):
        """Load and merge a task from all data directories, filtered by split."""
        merged = []
        for d in data_dirs:
            task_dir = d / subdir
            if (task_dir / "images").exists():
                ds = VisionTaskDataset(data_path=str(task_dir), resolution=res, task=vision_task)
                if ds.samples:
                    # Apply split filtering
                    ds.samples = _filter_by_split(ds.samples, vision_task, split_key)
                    if max_per_task is not None and len(ds.samples) > max_per_task:
                        ds.samples = ds.samples[:max_per_task]
                    if ds.samples:
                        merged.append(ds)
                        logger.info(f"  {task_name} ({split_key}) from {d.name}: {len(ds)} samples")
        return merged

    def _load_generation_dirs(split_key="train"):
        """Load and merge generation data from all directories, filtered by split."""
        merged = []
        for d in data_dirs:
            gen_dir = d / "generation"
            if (gen_dir / "images").exists():
                ds = GenerationDataset(str(gen_dir), resolution=res, max_samples=args.max_generation)
                if ds.samples:
                    # Apply split filtering
                    ds.samples = _filter_by_split(ds.samples, "generation", split_key)
                    if ds.samples:
                        merged.append(ds)
                        logger.info(f"  generation ({split_key}) from {d.name}: {len(ds)} samples")
        return merged

    # Depth task
    if args.task in ("all", "core", "depth"):
        datasets.extend(_load_task_dirs("depth", "depth", "depth", max_per_task=args.max_per_task, split_key="train"))

    # Normal task
    if args.task in ("all", "core", "normal"):
        datasets.extend(_load_task_dirs("normal", "normal", "normal", max_per_task=args.max_per_task, split_key="train"))

    # Segmentation task
    if args.task in ("all", "core", "segmentation"):
        segmentation_dirs = (
            data_dirs
            if args.data_dir is not None
            else [PROJECT_ROOT / "data" / "vb_snvc_train"]
        )
        for data_dir in segmentation_dirs:
            seg_dir = data_dir / "segmentation"
            if (seg_dir / "images").exists():
                ds = VisionTaskDataset(data_path=str(seg_dir), resolution=res, task="semantic")
                if ds.samples:
                    ds.samples = _filter_by_split(ds.samples, "semantic", "train")
                    if args.max_per_task and len(ds.samples) > args.max_per_task:
                        ds.samples = ds.samples[:args.max_per_task]
                    if ds.samples:
                        datasets.append(ds)
                        logger.info(f"  segmentation (train) from {data_dir.name}: {len(ds)} samples")

    # Edge task (edge subdirectory only — structure is loaded separately below
    # as its own task to avoid duplicate loading with conflicting prompts)
    if args.task in ("all", "edge"):
        datasets.extend(_load_task_dirs("edge", "edge", "edge", max_per_task=args.max_per_task, split_key="train"))

    # Depth edge task
    if args.task in ("all", "depth_edge"):
        datasets.extend(_load_task_dirs("depth_edge", "depth_edge", "depth_edge", max_per_task=args.max_per_task, split_key="train"))

    # Keypoint task
    if args.task in ("all", "keypoint"):
        datasets.extend(_load_task_dirs("keypoint", "keypoint", "keypoint", max_per_task=args.max_per_task, split_key="train"))

    # Object detection task
    if args.task in ("all", "object_detection"):
        datasets.extend(_load_task_dirs("object_detection", "object_detection", "object_detection", max_per_task=args.max_per_task, split_key="train"))

    # Referring task
    if args.task in ("all", "referring"):
        datasets.extend(_load_task_dirs("referring", "referring", "referring", max_per_task=args.max_per_task, split_key="train"))

    # Instance segmentation task
    if args.task in ("all", "instance"):
        datasets.extend(_load_task_dirs("instance", "instance", "instance", max_per_task=args.max_per_task, split_key="train"))

    # Structure task
    if args.task in ("all", "core", "structure"):
        datasets.extend(_load_task_dirs("structure", "structure", "structure", max_per_task=args.max_per_task, split_key="train"))

    # Multiview task
    if args.task in ("all", "multiview"):
        datasets.extend(_load_task_dirs("multiview", "multiview", "multiview", max_per_task=args.max_per_task, split_key="train"))

    # Specular removal task
    if args.task in ("all", "specular_removal"):
        datasets.extend(_load_task_dirs("specular_removal", "specular_removal", "specular_removal", max_per_task=args.max_per_task, split_key="train"))

    # Generation task
    if args.task in ("all", "generation") and not args.skip_generation:
        generation_datasets.extend(_load_generation_dirs(split_key="train"))

    if not datasets and not generation_datasets:
        logger.error("No training data found! Run data preparation scripts first.")
        sys.exit(1)

    combined, sampler = combine_training_datasets(
        datasets,
        generation_datasets,
        balance_data=args.balance_data,
        vision_ratio=args.vision_ratio,
        seed=args.seed,
    )
    total_samples = len(combined)
    if datasets and generation_datasets:
        logger.info(
            "Paper-aligned sampling: %.0f%% vision / %.0f%% generation",
            args.vision_ratio * 100,
            (1.0 - args.vision_ratio) * 100,
        )
    logger.info(f"Total training samples: {total_samples}")

    # ─── Build validation dataset from split ─────────────────────────────
    val_combined = None
    if split_data is not None:
        val_datasets = []
        if args.task in ("all", "core", "depth"):
            val_datasets.extend(_load_task_dirs("depth", "depth", "depth", split_key="val"))
        if args.task in ("all", "core", "normal"):
            val_datasets.extend(_load_task_dirs("normal", "normal", "normal", split_key="val"))
        if args.task in ("all", "core", "segmentation"):
            for data_dir in (data_dirs if args.data_dir is not None else [PROJECT_ROOT / "data" / "vb_snvc_train"]):
                seg_dir = data_dir / "segmentation"
                if (seg_dir / "images").exists():
                    ds = VisionTaskDataset(data_path=str(seg_dir), resolution=res, task="semantic")
                    ds.samples = _filter_by_split(ds.samples, "semantic", "val")
                    if ds.samples:
                        val_datasets.append(ds)
                        logger.info(f"  segmentation (val) from {data_dir.name}: {len(ds)} samples")
        if args.task in ("all", "edge"):
            val_datasets.extend(_load_task_dirs("edge", "edge", "edge", split_key="val"))
        if args.task in ("all", "core", "structure"):
            val_datasets.extend(_load_task_dirs("structure", "structure", "structure", split_key="val"))
        if args.task in ("all", "generation") and not args.skip_generation:
            val_datasets.extend(_load_generation_dirs(split_key="val"))

        if val_datasets:
            from torch.utils.data import ConcatDataset as _CD
            val_combined = _CD(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
            logger.info(f"Total validation samples: {len(val_combined)}")

    # ─── Training ────────────────────────────────────────────────────────
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else PROJECT_ROOT / "output" / f"{args.model}_checkpoints"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = len(combined)
    steps_per_epoch = max(
        1,
        (total_samples + args.batch_size * args.grad_accum - 1)
        // (args.batch_size * args.grad_accum),
    )
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(200, max(1, total_steps // 10))

    # NOTE: Previous code used Subset to limit dataset to max_steps * grad_accum
    # samples, which also set sampler=None — destroying the WeightedRandomSampler.
    # Now we keep the full dataset + sampler and pass max_steps to tuner.train()
    # for a clean early exit.

    logger.info(f"Starting training: {args.epochs} epochs, {total_samples} samples, "
                f"batch_size={args.batch_size}, grad_accum={args.grad_accum}, "
                f"resolution={args.resolution}, total_steps={total_steps}")

    # ─── Resume from checkpoint ──────────────────────────────────────────
    start_step = 0
    start_epoch = 0
    resume_checkpoint = None
    if args.resume or args.resume_from:
        import glob
        if args.resume_from:
            resume_dir = Path(args.resume_from)
        else:
            ckpt_dirs = sorted(glob.glob(str(output_dir / "checkpoint-*")),
                               key=lambda x: int(x.split("-")[-1]))
            resume_dir = Path(ckpt_dirs[-1]) if ckpt_dirs else None

        if resume_dir and resume_dir.exists():
            logger.info(f"Resuming from {resume_dir}")
            meta_file = resume_dir / "metadata.json"
            if meta_file.exists():
                import json as _json
                meta = _json.loads(meta_file.read_text())
                start_step = meta.get("step", 0)
                start_epoch = meta.get("epoch", 0)
                logger.info(f"Resumed at step={start_step}, epoch={start_epoch}")
            resume_checkpoint = resume_dir
        else:
            raise FileNotFoundError(
                f"Resume requested but no checkpoint found: {resume_dir}"
            )

    result = tuner.train(
        train_dataset=combined,
        val_dataset=val_combined,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        scheduler_type="cosine",
        warmup_steps=warmup_steps,
        output_dir=str(output_dir),
        save_every=args.save_every,
        eval_every=max(1, total_steps // 4),  # Evaluate ~4 times during training
        use_amp=True,
        num_workers=0,  # MPS must be 0
        log_every=args.log_every,
        start_step=start_step,
        start_epoch=start_epoch,
        train_sampler=sampler,
        resume_checkpoint=resume_checkpoint,
        restore_rng_on_resume=False,  # Old checkpoints may lack rng_state.pt
        max_steps=args.max_steps,  # P0 fix: actually limit training steps
    )

    logger.info(f"Training complete! Result: {result}")
    logger.info(f"Checkpoints saved to: {output_dir}")

    # Save final LoRA weights as .pt for evaluation
    final_lora_path = output_dir / "lora_final.pt"
    lora_state = tuner.peft_model.state_dict()
    lora_only = {k: v for k, v in lora_state.items() if "lora_" in k}
    torch.save(lora_only, final_lora_path)
    logger.info(f"Final LoRA weights saved to: {final_lora_path} ({len(lora_only)} keys)")


if __name__ == "__main__":
    main()
