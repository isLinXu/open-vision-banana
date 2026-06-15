"""Instruction alignment via LoRA fine-tuning.

Implements the lightweight instruction-tuning strategy from the paper:
- Mix vision task data into the base model's training mixture at a 9:1 ratio
- Use rsLoRA scaling (rank=64, alpha=128) to prevent learning rate decay
- Train only 0.5%-1% of total parameters
- Preserve 90%+ generation capability while unlocking understanding
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .checkpoint import CheckpointManager
from .utils import EMAModel, VisionAugmentation, enable_gradient_checkpointing

logger = logging.getLogger(__name__)


class VisionTaskDataset(Dataset):
    """Dataset for vision task instruction tuning.

    Each sample contains:
    - prompt: Instruction text
    - image: Input image tensor
    - target_rgb: Target RGB visualization image tensor
    """

    def __init__(self, data_path: str | None = None) -> None:
        self.samples = []
        if data_path is not None:
            self._load_data(data_path)

    def _load_data(self, path: str) -> None:
        """Load vision task data from directory."""
        pass  # Override in subclasses or use data/ datasets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class InstructionTuner:
    """LoRA-based instruction alignment for Vision Banana.

    Applies rsLoRA scaling (rank=64, alpha=128) to prevent learning rate
    decay at high ranks. Trains only 0.5%-1% of parameters while preserving
    90%+ generation capability.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lr: float = 5e-5,
        target_modules: list[str] | None = None,
        use_rslora: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_warmup_steps: int = 2000,
        use_gradient_checkpointing: bool = True,
        augmentation: VisionAugmentation | None = None,
    ) -> None:
        self.base_model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lr = lr
        self.target_modules = target_modules or ["to_k", "to_v"]
        self.use_rslora = use_rslora
        self.peft_model = None
        self.optimizer = None
        self.augmentation = augmentation
        self._setup_lora()

        self.use_ema = use_ema
        self.ema = None
        if use_ema:
            self.ema = EMAModel(
                self.peft_model,
                decay=ema_decay,
                warmup_steps=ema_warmup_steps,
            )

        if use_gradient_checkpointing:
            enable_gradient_checkpointing(self.peft_model)

    @staticmethod
    def _detect_device_type() -> str:
        """Detect the best available device type: cuda, mps, or cpu."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _setup_lora(self) -> None:
        """Set up LoRA with rsLoRA scaling on the pipeline's transformer."""
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            use_rslora=self.use_rslora,
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        transformer = self.base_model.transformer
        self.peft_model = get_peft_model(transformer, lora_config)
        self.peft_model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            self.peft_model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
        )

    def train(
        self,
        train_dataset: VisionTaskDataset,
        val_dataset: VisionTaskDataset | None = None,
        epochs: int = 10,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        scheduler_type: str = "cosine",
        warmup_steps: int = 500,
        output_dir: str | Path | None = None,
        save_every: int = 1000,
        eval_every: int = 500,
        use_amp: bool = True,
        callback: callable | None = None,
    ) -> None:
        """Train the LoRA instruction alignment."""
        self.peft_model.train()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=_collate_fn,
        )

        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = self._create_scheduler(
            scheduler_type, total_steps, warmup_steps
        )

        device_type = self._detect_device_type()
        scaler = torch.amp.GradScaler(device_type, enabled=use_amp and device_type == "cuda")
        ckpt_manager = CheckpointManager(output_dir) if output_dir else None

        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(train_loader):
                with torch.autocast(device_type, enabled=use_amp, dtype=torch.bfloat16):
                    loss = self._training_step(batch)

                scaled_loss = scaler.scale(loss / gradient_accumulation_steps)
                scaled_loss.backward()

                epoch_loss += loss.item()
                num_steps += 1

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.peft_model.parameters(), max_grad_norm
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    if self.ema is not None:
                        self.ema.update(self.peft_model)

                if callback and global_step > 0 and global_step % 10 == 0:
                    callback(
                        step=global_step,
                        loss=epoch_loss / max(num_steps, 1),
                        lr=scheduler.get_last_lr()[0],
                    )

                if ckpt_manager and global_step > 0 and global_step % save_every == 0:
                    ckpt_manager.save(
                        peft_model=self.peft_model,
                        optimizer=self.optimizer,
                        step=global_step,
                        epoch=epoch,
                        loss=epoch_loss / max(num_steps, 1),
                        lr=scheduler.get_last_lr()[0],
                        ema=self.ema,
                        scheduler=scheduler,
                    )

                if val_dataset and global_step > 0 and global_step % eval_every == 0:
                    self._evaluate(val_dataset)

            avg_loss = epoch_loss / max(num_steps, 1)
            logger.info(
                f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, "
                f"LR: {scheduler.get_last_lr()[0]:.2e}, Steps: {global_step}"
            )

        if ckpt_manager:
            ckpt_manager.save(
                peft_model=self.peft_model,
                optimizer=self.optimizer,
                step=global_step,
                epoch=epochs,
                loss=epoch_loss / max(num_steps, 1),
                lr=scheduler.get_last_lr()[0],
                ema=self.ema,
                scheduler=scheduler,
            )

    def _training_step(self, batch: dict) -> torch.Tensor:
        """Single training step — rectified flow velocity prediction loss."""
        target_rgb = batch["target_rgb"]
        input_images = batch.get("image")

        # Move data to model device
        device = next(self.peft_model.parameters()).device
        target_rgb = target_rgb.to(device)
        if input_images is not None:
            input_images = input_images.to(device)

        batch_size = target_rgb.shape[0]

        if self.augmentation is not None and input_images is not None:
            aug_images, aug_targets = [], []
            for i in range(batch_size):
                aug_img, aug_tgt = self.augmentation(input_images[i], target_rgb[i])
                aug_images.append(aug_img)
                aug_targets.append(aug_tgt)
            target_rgb = torch.stack(aug_targets)
            input_images = torch.stack(aug_images)

        target = target_rgb * 2.0 - 1.0
        timesteps = torch.rand(batch_size, device=device)
        noise = torch.randn_like(target)

        t_expand = timesteps.view(-1, 1, 1, 1)
        noisy_target = (1.0 - t_expand) * target + t_expand * noise
        velocity_target = target - noise

        predicted_velocity = self.peft_model(noisy_target)
        loss = nn.functional.mse_loss(predicted_velocity, velocity_target)
        return loss

    def _evaluate(self, val_dataset: VisionTaskDataset) -> None:
        """Evaluate on validation set."""
        self.peft_model.eval()
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_fn,
        )
        total_loss = 0.0
        num_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = self._training_step(batch)
                total_loss += loss.item()
                num_steps += 1
        avg_loss = total_loss / max(num_steps, 1)
        logger.info(f"Validation loss: {avg_loss:.6f}")
        self.peft_model.train()

    def _create_scheduler(self, scheduler_type: str, total_steps: int,
                          warmup_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler with linear warmup."""
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )

        if warmup_steps > 0:
            warmup = LinearLR(
                self.optimizer, start_factor=0.01, total_iters=warmup_steps
            )
        else:
            warmup = None

        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps - warmup_steps, 1)
            )
        else:
            main_scheduler = LinearLR(
                self.optimizer, total_iters=max(total_steps - warmup_steps, 1)
            )

        if warmup is not None:
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, main_scheduler],
                milestones=[warmup_steps],
            )
        return main_scheduler


def _collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for vision task batches."""
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], str):
            result[key] = [b[key] for b in batch]
        else:
            result[key] = [b[key] for b in batch]
    return result


__all__ = ["InstructionTuner", "VisionTaskDataset"]
