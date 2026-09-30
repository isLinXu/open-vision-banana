"""Unified multi-task, multi-model Trainer class.

Assembles ModelBackend, TrainingLosses, and TrainingCallbacks into a
complete training loop that replaces both train_sana_512.py and train_flux.py.

Usage:
    config = load_config(args)
    trainer = Trainer(config)
    trainer.train()
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler

from .callbacks import TrainingCallbacks
from .config import TrainingConfig
from .distributed import (
    all_reduce_gradients_,
    barrier,
    broadcast_parameters_,
    cleanup,
    init_distributed,
)
from .losses import TrainingLosses
from .model_backend import ModelBackend
from .optimization import SchedulerFactory
from .utils import EMAModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


# ── Task aliases and default prompts ──────────────────────────────────────

TASK_ALIASES = {"segmentation": "semantic", "seg": "semantic"}

TASK_PROMPTS = {
    "depth": "Generate a relative depth map. Preserve object boundaries and encode continuous relative depth with the training color map.",
    "normal": "Generate an RGB surface-normal map. Encode XYZ surface orientation in the RGB channels and preserve fine geometry.",
    "semantic": "Generate a 150-class semantic segmentation mask. Use only the fixed training palette, with one solid class color per pixel and sharp object boundaries.",
    "structure": "Generate a binary structural edge map. Use a black background and thin white edges aligned to visible scene boundaries.",
}


# ── Shared dataset ────────────────────────────────────────────────────────


class ManifestVisionDataset(Dataset):
    """Load input/target pairs from a JSONL manifest for multi-task training.

    Works for both SANA and FLUX training — the dataset is model-agnostic.
    """

    def __init__(
        self,
        manifest_path: Path,
        resolution: int,
        split: str,
        tasks: set[str],
        max_samples: int | None = None,
        data_root: Path | None = None,
    ):
        self.resolution = resolution
        self.samples: list[dict[str, str]] = []
        path_anchor = data_root.resolve() if data_root else manifest_path

        from open_vision_banana.data.manifest import resolve_manifest_path

        with Path(manifest_path).open(encoding="utf-8") as f:
            for _line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                raw_task = row.get("task")
                mapped = TASK_ALIASES.get(raw_task, raw_task)
                if mapped not in tasks:
                    continue
                if row.get("split") != split:
                    continue
                img_path = resolve_manifest_path(row.get("input", ""), path_anchor)
                tgt_path = resolve_manifest_path(row.get("target", ""), path_anchor)
                if not img_path.is_file() or not tgt_path.is_file():
                    continue
                prompt = row.get("prompt") or TASK_PROMPTS.get(
                    mapped, TASK_PROMPTS.get(raw_task, "")
                )
                self.samples.append(
                    {
                        "image_path": str(img_path),
                        "target_path": str(tgt_path),
                        "task": mapped,
                        "prompt": prompt,
                    }
                )

        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        last_err = None
        for attempt, cur in enumerate(self._candidates(idx)):
            try:
                return self._load(cur)
            except Exception as e:
                last_err = e
                if attempt == 0:
                    logger.warning("Skipping unreadable sample idx=%s (%s)", cur, e)
        raise RuntimeError(f"All candidates failed (last: {last_err})")

    def _candidates(self, idx: int):
        yield idx
        rng = random.Random(idx)
        n = len(self.samples)
        for _ in range(3):
            yield rng.randrange(n) if n else idx

    def _load(self, idx: int) -> dict:
        s = self.samples[idx]
        with Image.open(s["image_path"]) as img:
            img.load()
            img = img.convert("RGB").resize(
                (self.resolution, self.resolution), Image.Resampling.LANCZOS
            )
            inp = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        resample = Image.Resampling.NEAREST if s["task"] == "semantic" else Image.Resampling.LANCZOS
        with Image.open(s["target_path"]) as img:
            img.load()
            img = img.convert("RGB").resize((self.resolution, self.resolution), resample)
            tgt = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        return {
            "prompt": s["prompt"],
            "image": inp * 2.0 - 1.0,
            "target_rgb": tgt,
            "task": s["task"],
        }


def _collate_fn(batch: list[dict]) -> dict:
    return {
        "prompt": [b["prompt"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "target_rgb": torch.stack([b["target_rgb"] for b in batch]),
        "task": [b["task"] for b in batch],
    }


# ── Trainer ───────────────────────────────────────────────────────────────


class Trainer:
    """Unified multi-task, multi-model training trainer.

    Replaces both train_sana_512.py and train_flux.py main() functions.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.dist = init_distributed()
        self.device = config.get_device()
        if self.dist.enabled and self.device.type == "cuda":
            self.device = torch.device("cuda", self.dist.local_rank)
        self.dtype = config.get_dtype()

        # Seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        logger.info("=" * 70)
        logger.info("Vision Banana — Unified Training")
        logger.info("=" * 70)
        logger.info(
            "Model: %s | Device: %s | dtype: %s", config.model_name, self.device, config.dtype
        )
        logger.info(
            "LoRA: rank=%d, alpha=%d, rsLoRA=%s", config.lora_rank, config.lora_alpha, config.rslora
        )
        logger.info(
            "Steps: %d | LR: %s | eff_batch: %d (%d rank(s))",
            config.steps,
            config.lr,
            config.batch_size * config.grad_accum * self.dist.world_size,
            self.dist.world_size,
        )
        logger.info("Output: %s", config.output_dir)

        # ── Model backend ──────────────────────────────────────────────
        self.backend = ModelBackend(
            model_name=config.model_name,
            device=self.device,
            dtype=self.dtype,
            quantize=config.quantize,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            use_rslora=config.rslora,
            lora_targets=config.lora_targets,
            enable_gradient_checkpointing=config.gradient_checkpointing,
        )
        self.trainable_params = [p for p in self.backend.peft_model.parameters() if p.requires_grad]
        broadcast_parameters_(self.trainable_params, self.dist)
        if self.dist.enabled:
            # Same seed above keeps LoRA init identical; from here on each rank needs its
            # own noise / timestep / CFG-dropout draws, or every rank samples the same ones.
            torch.manual_seed(config.seed + self.dist.rank)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed + self.dist.rank)

        # ── Losses ─────────────────────────────────────────────────────
        task_weights = config.parse_task_weights()
        self.losses = TrainingLosses(
            task_weights=task_weights,
            depth_edge_aux_weight=config.depth_edge_aux_weight,
            normal_angular_loss_weight=config.normal_angular_loss_weight,
            depth_edge_aux_gradient_weight=config.depth_edge_aux_gradient_weight,
            depth_edge_aux_edge_weight=config.depth_edge_aux_edge_weight,
            depth_edge_aux_edge_threshold=config.depth_edge_aux_edge_threshold,
            depth_edge_aux_sigma_floor=config.depth_edge_aux_sigma_floor,
            min_snr=config.min_snr,
            min_snr_gamma=config.min_snr_gamma,
            family=self.backend.family,
        )

        # ── Callbacks ──────────────────────────────────────────────────
        self.callbacks = TrainingCallbacks(
            output_dir=config.output_dir,
            wandb_enabled=config.wandb and self.dist.is_main,
            wandb_project=config.wandb_project,
            wandb_run_name=config.wandb_run_name,
            wandb_config={k: str(v) for k, v in config.__dict__.items() if not k.startswith("_")},
            log_every=config.log_every,
            save_every=config.save_every,
            eval_every=config.eval_every,
        )

        # ── EMA ────────────────────────────────────────────────────────
        self.ema = EMAModel(
            self.backend.peft_model,
            decay=config.ema_decay,
            warmup_steps=config.ema_warmup,
        )

        # ── Datasets ───────────────────────────────────────────────────
        self.train_loader, self.val_loader = self._build_datasets()

        # ── Optimizer & Scheduler ──────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        total_steps = config.steps
        warmup_steps = config.warmup_steps or int(total_steps * config.warmup_ratio)
        self.lr_scheduler = SchedulerFactory.create(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            scheduler_type=config.scheduler,
        )

        logger.info("Setup complete. Starting training...")

    def _build_datasets(self) -> tuple[DataLoader, DataLoader | None]:
        """Build train and val data loaders from manifest."""
        config = self.config
        data_root = Path(config.data_dir) if config.data_dir else None
        manifest_path = Path(config.pair_manifest) if config.pair_manifest else None

        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError(f"Pair manifest not found: {manifest_path}")

        # Resolve tasks
        all_tasks = {"depth", "normal", "semantic"}
        tasks_to_load = set()
        if config.tasks:
            for t in config.tasks:
                tasks_to_load.add(TASK_ALIASES.get(t, t))
        else:
            tasks_to_load = all_tasks

        # Build per-task datasets
        train_datasets = [
            ManifestVisionDataset(
                manifest_path,
                config.resolution,
                "train",
                {t},
                config.max_samples_per_task,
                data_root=data_root,
            )
            for t in sorted(tasks_to_load)
        ]
        train_datasets = [ds for ds in train_datasets if len(ds) > 0]

        val_datasets = [
            ManifestVisionDataset(manifest_path, config.resolution, "val", {t}, data_root=data_root)
            for t in sorted(tasks_to_load)
        ]
        val_datasets = [ds for ds in val_datasets if len(ds) > 0]

        train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        val_ds = ConcatDataset(val_datasets) if val_datasets else None

        # Under torchrun each rank reads its own shard; set_epoch() in train() reshuffles it.
        train_sampler = (
            DistributedSampler(
                train_ds,
                num_replicas=self.dist.world_size,
                rank=self.dist.rank,
                shuffle=True,
                seed=config.seed,
                drop_last=True,
            )
            if self.dist.enabled
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.num_workers,
            collate_fn=_collate_fn,
            drop_last=True,
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=_collate_fn,
            )

        for ds in train_datasets:
            task_name = ds.samples[0]["task"] if ds.samples else "?"
            logger.info("  train/%s: %d samples", task_name, len(ds))
        logger.info("  Total train: %d | Val: %d", len(train_ds), len(val_ds) if val_ds else 0)

        return train_loader, val_loader

    def train(self) -> None:
        """Main training loop."""
        config = self.config
        peft_model = self.backend.peft_model
        total_steps = config.steps

        train_metrics = {
            "loss": 0.0,
            "flow_loss": 0.0,
            "depth_edge_aux_loss": 0.0,
            "normal_angular_aux_loss": 0.0,
            "steps": 0,
        }
        task_loss_acc: dict[str, list[float]] = {}
        global_step = 0
        micro_step = 0
        data_pass = 0  # full passes over the train loader; seeds DistributedSampler shuffles
        self.optimizer.zero_grad(set_to_none=True)

        while global_step < total_steps:
            updates_per_epoch = max(
                1, (len(self.train_loader) + config.grad_accum - 1) // config.grad_accum
            )
            epoch = global_step // updates_per_epoch
            if isinstance(self.train_loader.sampler, DistributedSampler):
                # not `epoch`: that lags the real pass count when grad_accum does not divide
                # len(train_loader), which would replay the same shuffle.
                self.train_loader.sampler.set_epoch(data_pass)
            data_pass += 1
            peft_model.train()

            for batch in self.train_loader:
                if global_step >= total_steps:
                    break

                # ── Forward ────────────────────────────────────────────
                loss_dict = self._train_step(batch, global_step)

                # ── Accumulate metrics ─────────────────────────────────
                train_metrics["loss"] += loss_dict["total_loss"]
                train_metrics["flow_loss"] += loss_dict["flow_loss"]
                train_metrics["depth_edge_aux_loss"] += loss_dict["depth_aux"]
                train_metrics["normal_angular_aux_loss"] += loss_dict["normal_aux"]
                train_metrics["steps"] += 1

                for t, vals in loss_dict.get("task_loss_acc", {}).items():
                    task_loss_acc.setdefault(t, []).extend(vals)

                # ── Backward ───────────────────────────────────────────
                loss = loss_dict["loss"] / config.grad_accum
                loss.backward()

                micro_step += 1
                if micro_step % config.grad_accum != 0:
                    continue

                # ── Optimizer step ─────────────────────────────────────
                all_reduce_gradients_(self.trainable_params, self.dist)
                torch.nn.utils.clip_grad_norm_(self.trainable_params, config.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.ema.update(peft_model)
                global_step += 1

                # ── Logging ────────────────────────────────────────────
                if self.callbacks.should_log(global_step):
                    n = max(1, train_metrics["steps"])
                    metrics = {
                        k: train_metrics[k] / n
                        for k in (
                            "loss",
                            "flow_loss",
                            "depth_edge_aux_loss",
                            "normal_angular_aux_loss",
                        )
                    }
                    lr = self.lr_scheduler.get_last_lr()[0]
                    if self.dist.is_main:
                        self.callbacks.on_log(
                            global_step, total_steps, metrics, lr, epoch, task_loss_acc
                        )
                    train_metrics = {
                        "loss": 0.0,
                        "flow_loss": 0.0,
                        "depth_edge_aux_loss": 0.0,
                        "normal_angular_aux_loss": 0.0,
                        "steps": 0,
                    }
                    task_loss_acc = {}

                # Ranks hold identical weights, so rank 0 alone saves and validates;
                # the others wait for it in the next step's gradient all-reduce.
                # ── Checkpoint ─────────────────────────────────────────
                if self.dist.is_main and self.callbacks.should_checkpoint(global_step):
                    self.callbacks.on_checkpoint(
                        global_step,
                        peft_model,
                        self.optimizer,
                        self.lr_scheduler,
                        ema=self.ema,
                        model_name=config.model_name,
                        lora_rank=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                    )

                # ── Validation ─────────────────────────────────────────
                if (
                    self.dist.is_main
                    and self.val_loader is not None
                    and self.callbacks.should_validate(global_step)
                ):
                    ema_backup = self.ema.apply_shadow(peft_model)
                    try:
                        val_loss = self.validate()
                    finally:
                        self.ema.restore_shadow(peft_model, ema_backup)
                    self.callbacks.on_validate(
                        global_step,
                        val_loss,
                        peft_model,
                        self.optimizer,
                        self.lr_scheduler,
                        ema=self.ema,
                        model_name=config.model_name,
                        lora_rank=config.lora_rank,
                        lora_alpha=config.lora_alpha,
                    )

        # ── Final save ─────────────────────────────────────────────────
        if self.dist.is_main:
            self.callbacks.on_checkpoint(
                global_step,
                peft_model,
                self.optimizer,
                self.lr_scheduler,
                ema=self.ema,
                model_name=config.model_name,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                suffix="_final",
            )
            self.callbacks.on_finish(global_step)
        barrier(self.dist)  # keep the process group alive until rank 0 has saved
        cleanup(self.dist)

    def _train_step(self, batch: dict, global_step: int) -> dict:
        """Execute a single training step (forward + loss)."""
        config = self.config
        prompts = batch["prompt"]
        batch_size = len(prompts)
        target_rgb = batch["target_rgb"].to(device=self.device, dtype=self.dtype)
        input_images = batch["image"].to(device=self.device, dtype=self.dtype)

        # ── VAE encode ─────────────────────────────────────────────────
        with torch.no_grad():
            target_pixels = target_rgb * 2.0 - 1.0  # [0,1] → [-1,1]
            target_latents = self.backend.encode_image(target_pixels)
            input_pixels = input_images  # already [-1,1]
            input_latents = self.backend.encode_image(input_pixels)
            if input_latents.shape != target_latents.shape:
                input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])

        # ── Flow matching ──────────────────────────────────────────────
        flow_batch = self.backend.prepare_flow_batch(
            target_latents,
            input_latents,
            batch_size,
            img2img_flow=config.img2img_flow,
            flow_sigma=config.flow_sigma,
            logit_normal=config.logit_normal,
            flow_shift=config.flow_shift,
        )
        timesteps = flow_batch["timesteps"]
        noisy_latents = flow_batch["noisy_latents"]
        velocity_target = flow_batch["velocity_target"]

        # ── Text encoding ──────────────────────────────────────────────
        with torch.no_grad():
            text_cond = self.backend.encode_text(prompts, self.device, self.dtype)
            text_cond = self.backend.apply_cfg_dropout(
                text_cond, config.cfg_dropout_rate, batch_size
            )

        # ── Forward pass ───────────────────────────────────────────────
        predicted = self.backend.forward(
            noisy_latents,
            timesteps,
            text_cond,
            batch_size,
        )

        # ── Loss computation ───────────────────────────────────────────
        batch_tasks = batch.get("task", ["depth"] * batch_size)
        flow_loss, per_sample_loss, task_loss_acc = self.losses.compute_flow_loss(
            predicted,
            velocity_target,
            timesteps,
            batch_tasks,
            self.device,
        )

        depth_aux_loss, normal_aux_loss = self.losses.compute_aux_losses(
            predicted,
            noisy_latents,
            target_latents,
            target_rgb,
            timesteps,
            batch_tasks,
            vae_decode_fn=self.backend.decode_latents,
            device=self.device,
        )

        total = self.losses.total_loss(flow_loss, depth_aux_loss, normal_aux_loss)

        if not torch.isfinite(total).all():
            raise FloatingPointError(f"Non-finite loss at step {global_step}")

        return {
            "loss": total,
            "total_loss": total.item(),
            "flow_loss": flow_loss.item(),
            "depth_aux": depth_aux_loss.item(),
            "normal_aux": normal_aux_loss.item(),
            "task_loss_acc": task_loss_acc,
        }

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss.

        Every call draws the same noise and timesteps, so losses are comparable across
        evaluations and best-checkpoint selection does not depend on the draw. The training
        RNG stream is left as it was.
        """
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.manual_seed(self.config.seed)
            return self._validate()

    def _validate(self) -> float:
        peft_model = self.backend.peft_model
        peft_model.eval()

        total_loss = 0.0
        n_batches = 0
        max_val_batches = 20

        for i, batch in enumerate(self.val_loader):
            if i >= max_val_batches:
                break

            prompts = batch["prompt"]
            batch_size = len(prompts)
            target_rgb = batch["target_rgb"].to(device=self.device, dtype=self.dtype)
            input_images = batch["image"].to(device=self.device, dtype=self.dtype)

            # VAE encode
            target_pixels = target_rgb * 2.0 - 1.0
            target_latents = self.backend.encode_image(target_pixels)
            input_latents = self.backend.encode_image(input_images)
            if input_latents.shape != target_latents.shape:
                input_latents = F.adaptive_avg_pool2d(input_latents, target_latents.shape[-2:])

            # Flow matching
            flow_batch = self.backend.prepare_flow_batch(
                target_latents,
                input_latents,
                batch_size,
                img2img_flow=self.config.img2img_flow,
                flow_sigma=self.config.flow_sigma,
                logit_normal=self.config.logit_normal,
                flow_shift=self.config.flow_shift,
            )

            # Text encoding
            text_cond = self.backend.encode_text(prompts, self.device, self.dtype)

            # Forward
            predicted = self.backend.forward(
                flow_batch["noisy_latents"],
                flow_batch["timesteps"],
                text_cond,
                batch_size,
            )

            # Loss
            batch_tasks = batch.get("task", ["depth"] * batch_size)
            flow_loss, _, _ = self.losses.compute_flow_loss(
                predicted,
                flow_batch["velocity_target"],
                flow_batch["timesteps"],
                batch_tasks,
                self.device,
            )
            total_loss += flow_loss.item()
            n_batches += 1

        peft_model.train()
        return total_loss / max(1, n_batches)
