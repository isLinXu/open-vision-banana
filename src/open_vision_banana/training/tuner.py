"""Instruction alignment via LoRA fine-tuning.

Implements the lightweight instruction-tuning strategy from the paper:
- Mix vision task data into the base model's training mixture at a 9:1 ratio
- Use rsLoRA scaling (rank=64, alpha=128) to prevent learning rate decay
- Train only 0.5%-1% of total parameters
- Preserve 90%+ generation capability while unlocking understanding
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _Dataset

from .checkpoint import CheckpointManager
from .early_stopping import EarlyStopping
from .logging import TrainingLogger
from .optimization import (
    SchedulerFactory,
    compute_min_snr_weight,
    sample_timesteps_logit_normal,
)
from .trainer_interface import NativeTrainer, TrainerInterface
from .utils import EMAModel, VisionAugmentation, enable_gradient_checkpointing

logger = logging.getLogger(__name__)


class VisionTaskDataset(_Dataset):
    """Dataset for vision task instruction tuning.

    Each sample contains:
    - prompt: Instruction text
    - image: Input image tensor [3, H, W] in [-1, 1]
    - target_rgb: Target RGB visualization image tensor [3, H, W] in [0, 1]

    This base class supports loading from a flat directory of paired files::

        data_path/
          images/           ← input images  (*.jpg / *.png)
          targets/          ← target visualisations  (*.jpg / *.png)

    Subclasses can override ``_load_data`` for custom layouts.
    """

    # Task-specific default prompts — more descriptive than a generic
    # template, these align with the PromptBuilder used during inference.
    TASK_PROMPTS = {
        "depth": "Generate a metric depth visualization of this image.",
        "normal": "Generate a surface normal estimation visualization of this image.",
        "semantic": "Generate a semantic segmentation visualization of this image.",
        "instance": "Generate an instance segmentation visualization of this image.",
        "referring": "Generate a referring expression segmentation visualization of this image.",
        "edge": "Generate an edge detection visualization of this image.",
        "flow": "Generate an optical flow visualization of this image.",
        "keypoint": "Generate a keypoint detection visualization of this image.",
        "object_detection": "Generate an object detection visualization of this image.",
        "depth_edge": "Generate a depth edge detection visualization of this image.",
    }

    DEFAULT_PROMPT = "Generate a visual understanding output for this image."

    def __init__(
        self,
        data_path: str | None = None,
        resolution: int = 512,
        task: str | None = None,
    ) -> None:
        self.samples: list[dict] = []
        self.resolution = resolution
        self.task = task
        if data_path is not None:
            self._load_data(data_path)

    def _load_data(self, path: str) -> None:
        """Load paired (image, target_rgb) samples from a flat directory.

        Expected layout::

            <path>/images/    ← *.jpg / *.png input images
            <path>/targets/   ← matching target visualisations (same filenames)

        Files without a matching target are silently skipped.

        Uses ``os.scandir`` for efficient directory scanning and builds a
        target-file index for O(1) lookup, avoiding the O(N×K) extension
        probing of the previous implementation.
        """
        import os
        from pathlib import Path

        root = Path(path)
        img_dir = root / "images"
        tgt_dir = root / "targets"

        if not img_dir.exists():
            import logging

            logging.getLogger(__name__).warning(
                f"VisionTaskDataset: images dir not found at {img_dir}"
            )
            return

        IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

        # Build a target-file index keyed by stem for O(1) lookup
        # (avoids iterating extensions per image)
        target_index: dict[str, Path] = {}
        if tgt_dir.exists():
            with os.scandir(str(tgt_dir)) as entries:
                for entry in entries:
                    if entry.is_file():
                        p = Path(entry.path)
                        if p.suffix.lower() in IMG_EXTS:
                            target_index[p.stem] = p

        # Scan images using os.scandir (faster than iterdir for large dirs)
        image_paths: list[Path] = []
        with os.scandir(str(img_dir)) as entries:
            for entry in entries:
                if entry.is_file():
                    p = Path(entry.path)
                    if p.suffix.lower() in IMG_EXTS:
                        image_paths.append(p)

        # Sort for deterministic ordering
        image_paths.sort()

        prompt = self.TASK_PROMPTS.get(self.task, self.DEFAULT_PROMPT)
        for img_path in image_paths:
            tgt_path = target_index.get(img_path.stem)
            if tgt_path is None:
                continue
            self.samples.append(
                {
                    "image_path": str(img_path),
                    "target_path": str(tgt_path),
                    "prompt": prompt,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        import numpy as np
        from PIL import Image

        def _load(p: str, normalise_range: tuple) -> torch.Tensor:
            img = Image.open(p).convert("RGB")
            if hasattr(self, "resolution"):
                img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
            arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            lo, hi = normalise_range
            return arr * (hi - lo) + lo

        image = _load(sample["image_path"], (-1.0, 1.0))
        target_rgb = _load(sample["target_path"], (0.0, 1.0))

        return {
            "prompt": sample["prompt"],
            "image": image,
            "target_rgb": target_rgb,
        }


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
        task: str = "depth",
        channel_loss_weights: list[float] | torch.Tensor | None = None,
        use_logit_normal_timesteps: bool = True,
        use_min_snr: bool = True,
        min_snr_gamma: float = 5.0,
        cfg_dropout_rate: float = 0.1,
        use_img2img_flow: bool = True,
        flow_sigma: float = 0.0,
        warmup_ratio: float | None = None,
        trainer_backend: TrainerInterface | None = None,
    ) -> None:
        self.base_model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lr = lr
        self.target_modules = target_modules or ["to_k", "to_v"]
        self.channel_loss_weights = channel_loss_weights
        self.use_logit_normal_timesteps = use_logit_normal_timesteps
        self.use_min_snr = use_min_snr
        self.min_snr_gamma = min_snr_gamma
        self.cfg_dropout_rate = cfg_dropout_rate
        self.use_img2img_flow = use_img2img_flow
        self.flow_sigma = flow_sigma
        self.warmup_ratio = warmup_ratio
        self.use_rslora = use_rslora
        self.peft_model = None
        self.optimizer = None
        self.deepspeed_engine = None
        self.task = task
        self.augmentation = augmentation
        # ── Trainer backend (TrainerInterface) ────────────────────────
        # When provided, train() delegates backward / optimizer_step / grad
        # scaling to this backend instead of manually managing them inline.
        # This enables DDP / FSDP / DeepSpeed / Accelerate via the same code path.
        self.trainer_backend = trainer_backend
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

        # ── Interface probing: detect forward() signature at init time ────
        # Instead of a try/except fallback chain at every training step,
        # probe the model's forward() signature once and store the result.
        self._forward_mode = self._probe_forward_signature()

    @staticmethod
    def _detect_device_type() -> str:
        """Detect the best available device type: cuda, mps, or cpu."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int, height: int, width: int, device, dtype
    ) -> torch.Tensor:
        """Build FLUX latent position IDs [H*W, 3] for the transformer.

        Mirrors FluxPipeline._prepare_latent_image_ids: creates a 3-channel
        ID grid where channel 0 is zeros, channel 1 is row indices, and
        channel 2 is column indices, then flattens to [H*W, 3].
        """
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels, height, width):
        """Pack VAE latents for FLUX transformer.

        [B, C, H, W] → [B, (H/2)*(W/2), C*4]

        Mirrors FluxPipeline._pack_latents — the 2×2 spatial unpacking
        increases channel count from latent_channels (16) to in_channels (64).
        """
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width):
        """Unpack FLUX transformer output back to VAE latent format.

        [B, (H/2)*(W/2), C*4] → [B, C, H, W]
        """
        batch_size, num_patches, channels = latents.shape
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, height, width)
        return latents

    def _probe_forward_signature(self) -> str:
        """Probe the model's forward() signature at init time.

        Returns one of:
        - "full": accepts hidden_states, timestep, encoder_hidden_states,
                  pooled_projections (full FLUX signature)
        - "partial": accepts hidden_states, timestep, encoder_hidden_states
        - "minimal": accepts hidden_states, timestep only
        - "positional": accepts positional args only (smoke-test mocks)

        Uses ``inspect.signature`` to statically analyse the forward()
        parameter list, avoiding exception-based control flow.  Falls back
        to a single trial call only when static analysis is inconclusive
        (e.g. ``**kwargs`` present but specific named params absent).
        """
        import inspect

        # ── P0-5: Fast-path for known FLUX transformer ──────────────────
        # PEFT wrappers hide the real forward signature behind (*args, **kwargs).
        # Probe the underlying model chain directly to avoid shape-mismatch
        # dummy-call failures on MPS / bfloat16.
        underlying = self.peft_model
        for attr in ("base_model", "model"):
            if hasattr(underlying, attr):
                underlying = getattr(underlying, attr)
        model_name = type(underlying).__name__
        if model_name in ("FluxTransformer2DModel", "FluxTransformer2DModelWithLora"):
            return "full"

        try:
            sig = inspect.signature(self.peft_model.forward)
        except (ValueError, TypeError):
            # Cannot introspect (e.g. a compiled/capturing object) —
            # fall back to positional mode.
            return "positional"

        params = sig.parameters
        # Collect named parameters (exclude *args / **kwargs)
        named = {
            name
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params.values())

        # Check for the full FLUX signature
        required_full = {"hidden_states", "timestep", "encoder_hidden_states", "pooled_projections"}
        if required_full.issubset(named):
            return "full"

        # Check for partial signature (3 named params)
        required_partial = {"hidden_states", "timestep", "encoder_hidden_states"}
        if required_partial.issubset(named):
            return "partial"

        # Check for minimal signature (2 named params)
        required_minimal = {"hidden_states", "timestep"}
        if required_minimal.issubset(named):
            return "minimal"

        # If **kwargs is present but no specific named params were found
        # (e.g. PEFT wrapper with (*args, **kwargs)), the model might still
        # accept these params dynamically.  Do a single trial call with the
        # most complete signature to detect acceptance.
        if has_kwargs:
            dummy_hidden = torch.zeros(1, 16, 2, 2)  # FLUX latent shape
            dummy_timestep = torch.tensor([0.5])
            dummy_encoder = torch.zeros(1, 1, 4096)
            dummy_pooled = torch.zeros(1, 768)

            # Try full first, then partial, then minimal
            for mode, kwargs in [
                (
                    "full",
                    {
                        "hidden_states": dummy_hidden,
                        "timestep": dummy_timestep,
                        "encoder_hidden_states": dummy_encoder,
                        "pooled_projections": dummy_pooled,
                    },
                ),
                (
                    "partial",
                    {
                        "hidden_states": dummy_hidden,
                        "timestep": dummy_timestep,
                        "encoder_hidden_states": dummy_encoder,
                    },
                ),
                (
                    "minimal",
                    {
                        "hidden_states": dummy_hidden,
                        "timestep": dummy_timestep,
                    },
                ),
            ]:
                try:
                    self.peft_model(**kwargs)
                    return mode
                except TypeError:
                    continue

        return "positional"

    def enable_deepspeed(
        self,
        *,
        config_path: str | Path | None = None,
        micro_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ) -> None:
        """Wrap the LoRA model with DeepSpeed ZeRO-3."""
        from .deepspeed_utils import initialize_deepspeed

        engine, optimizer = initialize_deepspeed(
            self.peft_model,
            self.optimizer,
            config_path=config_path,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            lr=self.lr,
            max_grad_norm=max_grad_norm,
        )
        self.peft_model = engine
        self.optimizer = optimizer
        self.deepspeed_engine = engine
        logger.info("DeepSpeed ZeRO-3 enabled")

    def _ema_module(self) -> nn.Module:
        """Return the underlying module for EMA updates (handles DeepSpeed/DDP)."""
        model = self.peft_model
        if self.deepspeed_engine is not None:
            return getattr(model, "module", model)
        return getattr(model, "module", model)

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

        # base_model is a BaseGenerator whose transformer lives inside the
        # diffusers pipeline (FluxPipeline._t2i_pipe.transformer).
        pipeline = getattr(self.base_model, "_t2i_pipe", None) or getattr(
            self.base_model, "pipe", None
        )
        if pipeline is None:
            # Fallback: base_model may itself be a diffusers pipeline
            pipeline = self.base_model
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise AttributeError(
                f"Could not resolve 'transformer' from base_model. "
                f"base_model type: {type(self.base_model)}"
            )
        self.peft_model = get_peft_model(transformer, lora_config)
        self.peft_model.print_trainable_parameters()

        # Cast LoRA parameters to the base model's dtype.
        # PEFT initializes LoRA adapters in float32 by default, but on MPS
        # mixed float32/bfloat16 operations cause "mps.multiply op requires
        # the same element type" errors. Casting to bfloat16 resolves this.
        base_dtype = next(transformer.parameters()).dtype
        if base_dtype != torch.float32:
            for param in self.peft_model.parameters():
                if param.dtype != base_dtype:
                    param.data = param.data.to(base_dtype)

        self.optimizer = torch.optim.AdamW(
            self.peft_model.parameters(),
            lr=self.lr,
            weight_decay=0.01,
        )

    def train(
        self,
        train_dataset: _Dataset,
        val_dataset: _Dataset | None = None,
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
        num_workers: int = 0,
        callback: Callable[[int, float, float], Any] | None = None,
        training_logger: TrainingLogger | None = None,
        log_every: int = 10,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.0,
        train_sampler=None,
        is_main_process: bool = True,
    ) -> dict:
        """Train the LoRA instruction alignment.

        Returns:
            Summary dict with ``epochs_run``, ``global_step``, ``best_val_loss``,
            and ``stopped_early`` flag.
        """
        self.peft_model.train()

        if val_dataset is None:
            logger.info("No val_dataset provided — per-step evaluation is disabled.")
            if early_stopping_patience is not None:
                logger.warning(
                    "early_stopping_patience is set but no val_dataset — ignoring early stopping"
                )
                early_stopping_patience = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=_collate_fn,
        )

        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = self._create_scheduler(scheduler_type, total_steps, warmup_steps)

        device_type = self._detect_device_type()
        use_deepspeed = self.deepspeed_engine is not None

        # ── Determine whether to use TrainerInterface or legacy path ──
        use_trainer_interface = self.trainer_backend is not None

        if use_trainer_interface:
            # Delegate to TrainerInterface — it handles backward, grad scaling,
            # optimizer step, gradient clipping, and accumulation internally.
            trainer = self.trainer_backend
            trainer.setup_optimizer(self.optimizer)
            if trainer.gradient_accumulation_steps != gradient_accumulation_steps:
                logger.warning(
                    "trainer_backend.gradient_accumulation_steps (%d) != "
                    "train() gradient_accumulation_steps (%d); using backend value.",
                    trainer.gradient_accumulation_steps,
                    gradient_accumulation_steps,
                )
                gradient_accumulation_steps = trainer.gradient_accumulation_steps
            use_amp = getattr(trainer, "use_autocast", False)
            scaler = trainer.scaler
            device_type = trainer.device.type
        else:
            if use_deepspeed:
                use_amp = False
            scaler = torch.amp.GradScaler(
                device_type, enabled=use_amp and device_type == "cuda" and not use_deepspeed
            )
        ckpt_manager = CheckpointManager(output_dir) if output_dir else None
        early_stopper = (
            EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
            if early_stopping_patience is not None
            else None
        )

        global_step = 0
        best_val_loss: float | None = None
        stopped_early = False

        for epoch in range(epochs):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            num_steps = 0

            for step, batch in enumerate(train_loader):
                if use_trainer_interface:
                    # ── TrainerInterface path ──
                    with torch.autocast(
                        device_type, enabled=use_amp, dtype=torch.bfloat16
                    ):
                        loss = self._training_step(batch)
                    stepped = trainer.train_step(self.peft_model, loss, step)
                elif use_deepspeed:
                    loss = self._training_step(batch)
                    self.deepspeed_engine.backward(loss / gradient_accumulation_steps)
                    stepped = (step + 1) % gradient_accumulation_steps == 0
                    if stepped:
                        self.deepspeed_engine.step()
                        scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                else:
                    # ── Legacy native path (backward compat) ──
                    with torch.autocast(device_type, enabled=use_amp, dtype=torch.bfloat16):
                        loss = self._training_step(batch)
                    scaled_loss = scaler.scale(loss / gradient_accumulation_steps)
                    scaled_loss.backward()

                    epoch_loss += loss.item()
                    num_steps += 1

                    if (step + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.peft_model.parameters(), max_grad_norm)
                        scaler.step(self.optimizer)
                        scaler.update()
                        scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                    if self.ema is not None:
                        self.ema.update(self._ema_module())

                    running_loss = epoch_loss / max(num_steps, 1)
                    current_lr = scheduler.get_last_lr()[0]

                    if global_step > 0 and global_step % log_every == 0 and not getattr(self, "_last_logged_step", -1) == global_step:
                        self._last_logged_step = global_step
                        logger.info(
                            f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                            f"Loss: {running_loss:.4f} | LR: {current_lr:.2e}"
                        )

                    if training_logger and global_step % log_every == 0:
                        training_logger.log_metrics(
                            global_step,
                            {
                                "train/loss": running_loss,
                                "train/lr": current_lr,
                                "train/epoch": float(epoch + 1),
                            },
                        )

                    if callback and global_step % log_every == 0:
                        callback(global_step, running_loss, current_lr)

                if use_trainer_interface:
                    epoch_loss += loss.item()
                    num_steps += 1

                    if stepped:
                        scheduler.step()
                        global_step += 1

                        if self.ema is not None:
                            self.ema.update(self._ema_module())

                        running_loss = epoch_loss / max(num_steps, 1)
                        current_lr = scheduler.get_last_lr()[0]

                        if global_step % log_every == 0:
                            logger.info(
                                f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                                f"Loss: {running_loss:.4f} | LR: {current_lr:.2e}"
                            )

                        if training_logger and global_step % log_every == 0:
                            training_logger.log_metrics(
                                global_step,
                                {
                                    "train/loss": running_loss,
                                    "train/lr": current_lr,
                                    "train/epoch": float(epoch + 1),
                                },
                            )

                        if callback and global_step % log_every == 0:
                            callback(global_step, running_loss, current_lr)

                if (
                    ckpt_manager
                    and is_main_process
                    and global_step > 0
                    and global_step % save_every == 0
                ):
                    ckpt_path = ckpt_manager.save(
                        peft_model=self.peft_model,
                        optimizer=self.optimizer,
                        step=global_step,
                        epoch=epoch,
                        loss=epoch_loss / max(num_steps, 1),
                        lr=scheduler.get_last_lr()[0],
                        ema=self.ema,
                        scheduler=scheduler,
                    )
                    if training_logger:
                        training_logger.log_checkpoint(
                            ckpt_path,
                            alias=f"step-{global_step}",
                            metadata={
                                "step": global_step,
                                "epoch": epoch,
                                "train_loss": epoch_loss / max(num_steps, 1),
                            },
                            step=global_step,
                        )

                if val_dataset and global_step > 0 and global_step % eval_every == 0:
                    val_loss = self._evaluate(val_dataset)
                    if training_logger:
                        training_logger.log_metrics(global_step, {"val/loss": val_loss})

                    if early_stopper is not None:
                        is_best = early_stopper.step(val_loss)
                        if is_best:
                            best_val_loss = val_loss
                            if ckpt_manager is not None and is_main_process:
                                best_path = ckpt_manager.save_best(
                                    peft_model=self.peft_model,
                                    optimizer=self.optimizer,
                                    step=global_step,
                                    epoch=epoch,
                                    train_loss=epoch_loss / max(num_steps, 1),
                                    val_loss=val_loss,
                                    lr=scheduler.get_last_lr()[0],
                                    ema=self.ema,
                                    scheduler=scheduler,
                                )
                                if training_logger:
                                    training_logger.log_checkpoint(
                                        best_path,
                                        alias="best",
                                        metadata={
                                            "step": global_step,
                                            "val_loss": val_loss,
                                            "best": True,
                                        },
                                        step=global_step,
                                    )
                        if early_stopper.should_stop:
                            stopped_early = True
                            break

            avg_loss = epoch_loss / max(num_steps, 1)
            logger.info(
                f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, "
                f"LR: {scheduler.get_last_lr()[0]:.2e}, Steps: {global_step}"
            )

            if training_logger:
                training_logger.log_metrics(
                    global_step,
                    {
                        "epoch/train_loss": avg_loss,
                        "epoch/number": float(epoch + 1),
                    },
                )

            if stopped_early:
                logger.info("Training stopped early at epoch %d", epoch + 1)
                break

        if ckpt_manager and is_main_process:
            final_path = ckpt_manager.save(
                peft_model=self.peft_model,
                optimizer=self.optimizer,
                step=global_step,
                epoch=epoch + 1 if stopped_early else epochs,
                loss=epoch_loss / max(num_steps, 1),
                lr=scheduler.get_last_lr()[0],
                ema=self.ema,
                scheduler=scheduler,
            )
            if training_logger:
                training_logger.log_checkpoint(
                    final_path,
                    alias="final",
                    metadata={"step": global_step, "stopped_early": stopped_early},
                    step=global_step,
                )

        if training_logger:
            training_logger.finish()

        return {
            "epochs_run": epoch + 1,
            "global_step": global_step,
            "best_val_loss": best_val_loss
            if best_val_loss is not None
            else early_stopper.best_loss
            if early_stopper
            else None,
            "stopped_early": stopped_early,
        }

    def _encode_text(self, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        """Encode text prompts to encoder_hidden_states using FLUX's text encoders.

        FLUX.1 uses a dual-encoder setup:
        - CLIP (text_encoder): produces pooled_projections [B, D_clip]
        - T5 (text_encoder_2): produces prompt_embeds [B, seq_len, D_t5]

        Returns a dict with keys ``prompt_embeds`` and ``pooled_prompt_embeds``.
        Falls back to zero tensors when the pipeline's text encoders are not
        accessible (e.g. unit tests with a mock model).
        """
        pipeline = getattr(self.base_model, "_t2i_pipe", None) or getattr(
            self.base_model, "pipe", None
        )
        if pipeline is None:
            # base_model may itself be a diffusers pipeline
            pipeline = self.base_model

        try:
            # MPS + bfloat16 can trigger "Placeholder storage has not been
            # allocated on MPS device!" for T5.  Temporarily cast text
            # encoders to float32 for the encode_prompt call, then restore.
            encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
            original_dtypes = []
            for enc in encoders:
                if enc is not None:
                    original_dtypes.append(enc.dtype)
                    enc.to(torch.float32)
                else:
                    original_dtypes.append(None)

            try:
                # diffusers ≥ 0.30 exposes encode_prompt on FluxPipeline
                # Use no_grad to prevent MPS graph caching of float32
                # intermediate tensors from the text encoders, which can
                # cause dtype mismatch errors in subsequent transformer
                # forward passes.
                with torch.no_grad():
                    text_outputs = pipeline.encode_prompt(
                        prompt=prompts,
                        prompt_2=prompts,
                        device=device,
                        num_images_per_prompt=1,
                        max_sequence_length=512,
                    )
            finally:
                # Restore original dtypes regardless of success/failure
                for enc, dtype in zip(encoders, original_dtypes):
                    if enc is not None and dtype is not None:
                        enc.to(dtype)

            # encode_prompt return structure varies across diffusers versions:
            #   < 0.30: (prompt_embeds, negative_prompt_embeds,
            #           pooled_prompt_embeds, negative_pooled_prompt_embeds)  — 4 items
            #   ≥ 0.30: (prompt_embeds, pooled_prompt_embeds, text_ids)  — 3 items
            #   ≥ 0.38: (prompt_embeds, pooled_prompt_embeds)  — 2 items (sometimes)
            # Use shape-based detection: prompt_embeds has 3 dims [B, seq, D],
            # pooled_prompt_embeds has 2 dims [B, D_clip],
            # text_ids has 2 dims [seq, 3] (no batch dim).
            prompt_embeds = text_outputs[0]  # always [B, seq_len, D_t5]
            pooled_prompt_embeds = None
            text_ids = None
            for item in text_outputs[1:]:
                if item is None:
                    continue
                if item.dim() == 2 and item.shape[-1] == 3 and item.shape[0] != 768:
                    # text_ids: [seq_len, 3] — distinguish from pooled [B, 768]
                    text_ids = item
                elif item.dim() == 2:
                    pooled_prompt_embeds = item  # [B, D_clip]

            # Cast embeddings to the model's dtype (encoders were temporarily
            # cast to float32 for MPS stability, so outputs are float32).
            model_dtype = next(self.peft_model.parameters()).dtype

            return {
                "prompt_embeds": prompt_embeds.to(device=device, dtype=model_dtype),
                "pooled_prompt_embeds": (
                    pooled_prompt_embeds.to(device=device, dtype=model_dtype)
                    if pooled_prompt_embeds is not None
                    else None
                ),
                "text_ids": text_ids.to(device=device, dtype=model_dtype) if text_ids is not None else None,
            }

        except Exception as e:
            logger.warning(
                f"Text encoding failed ({e}); falling back to zero conditioning. "
                "The model will train without text guidance for this batch."
            )
            # Graceful fallback: zero embeddings — model still trains the flow
            # objective but without text conditioning.  Useful for smoke tests.
            B = len(prompts)
            return {
                "prompt_embeds": torch.zeros(B, 1, 4096, device=device),
                "pooled_prompt_embeds": torch.zeros(B, 768, device=device),
                "text_ids": torch.zeros(1, 3, device=device),
            }

    def _encode_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to VAE latents.

        FLUX uses a 16-channel VAE with 8× spatial compression:
            [B, 3, H, W] → [B, 16, H/8, W/8]

        The encoded latents are scaled by the VAE's scaling_factor before
        being passed to the transformer (standard diffusers convention).

        Falls back to a lightweight 1×1 conv proxy when the VAE is not
        accessible (e.g. unit tests or CPU-only smoke runs).
        """
        pipeline = getattr(self.base_model, "_t2i_pipe", None) or getattr(
            self.base_model, "pipe", None
        )
        if pipeline is None:
            pipeline = self.base_model

        vae = getattr(pipeline, "vae", None)
        if vae is None:
            logger.warning(
                "VAE not found on pipeline; using identity proxy for latents. "
                "Training will proceed but latent statistics will be incorrect."
            )
            # Identity proxy: spatial downsample only, keep original channels.
            # We intentionally do NOT expand to 16 ch here — the downstream
            # transformer mock / shape-mismatch handler deals with channel
            # differences.  Keeping 3 ch preserves compatibility with unit-test
            # mocks whose Conv2d layers expect 3-channel input.
            B, C, H, W = images.shape
            latent_h, latent_w = max(1, H // 8), max(1, W // 8)
            latents = torch.nn.functional.adaptive_avg_pool2d(images, (latent_h, latent_w))
            return latents

        try:
            with torch.no_grad():
                # images expected in [-1, 1] by the VAE
                # Ensure input dtype and device match VAE weights to avoid
                # "Input type and bias type should be the same" and
                # "input and weight must be on the same device" on MPS.
                vae_device = next(vae.parameters()).device
                vae_dtype = next(vae.parameters()).dtype
                images = images.to(device=vae_device, dtype=vae_dtype)
                latent_dist = vae.encode(images).latent_dist
                latents = latent_dist.sample()
                # Apply VAE scaling factor (FLUX default: 0.3611)
                scaling_factor = getattr(vae.config, "scaling_factor", 0.3611)
                latents = latents * scaling_factor
                # Cast to model dtype to avoid MPS dtype mismatches
                model_dtype = next(self.peft_model.parameters()).dtype
                latents = latents.to(dtype=model_dtype)
            return latents
        except Exception as e:
            logger.warning(f"VAE encoding failed ({e}); using zero latents.")
            B, _, H, W = images.shape
            return torch.zeros(B, 16, H // 8, W // 8, device=images.device, dtype=images.dtype)

    def _training_step(self, batch: dict) -> torch.Tensor:
        """Single training step — rectified flow velocity prediction in latent space.

        Implements the FLUX flow-matching objective:
            loss = MSE(v_θ(z_t, t, c), z_0 - ε)
        where:
            z_0 = VAE.encode(target_rgb)     (target latent)
            z_t = (1-t)*z_0 + t*ε           (noisy latent, linear interpolation)
            v*  = z_0 - ε                    (target velocity)
            t   ~ Uniform[0, 1]              (per-sample timestep)
            c   = (prompt_embeds,            (text conditioning via CLIP + T5)
                   pooled_prompt_embeds)

        Key fixes over the original implementation:
        1. target_rgb is encoded to VAE latents before flow matching
           (FLUX trains in latent space, not pixel space).
        2. prompt strings are encoded to encoder_hidden_states and
           pooled_projections and passed to the transformer, enabling
           text-conditional generation.
        """
        target_rgb = batch["target_rgb"]
        input_images = batch.get("image")
        prompts: list[str] = batch.get("prompt", [""])
        if isinstance(prompts, str):
            prompts = [prompts]

        # Infer device from peft_model parameters (MPS/CUDA) rather than batch
        # tensor device, because _encode_latents moves latents to the VAE's
        # device (which may differ from DataLoader's CPU device).
        device = next(
            (p.device for p in self.peft_model.parameters()), torch.device("cpu")
        )

        target_rgb = target_rgb.to(device)
        if input_images is not None:
            input_images = input_images.to(device)

        batch_size = target_rgb.shape[0]

        # ── Data augmentation ────────────────────────────────────────────
        if self.augmentation is not None and input_images is not None:
            aug_images, aug_targets = [], []
            for i in range(batch_size):
                aug_img, aug_tgt = self.augmentation(input_images[i], target_rgb[i])
                aug_images.append(aug_img)
                aug_targets.append(aug_tgt)
            target_rgb = torch.stack(aug_targets)
            input_images = torch.stack(aug_images)

        # ── P0-2: VAE encoding — work in latent space ───────────────────
        # Convert target_rgb [0,1] → [-1,1] before VAE encoding.
        target_pixels = target_rgb * 2.0 - 1.0  # [B, 3, H, W]
        target_latents = self._encode_latents(target_pixels)  # [B, 16, H/8, W/8]

        input_latents = None
        if input_images is not None:
            input_latents = self._encode_latents(input_images)

        # ── Flow matching in latent space ────────────────────────────────
        model_dtype = next(self.peft_model.parameters()).dtype
        if self.use_logit_normal_timesteps:
            timesteps = sample_timesteps_logit_normal(batch_size, device).to(model_dtype)
        else:
            timesteps = torch.rand(batch_size, device=device).to(model_dtype)

        noise = torch.randn_like(target_latents)
        t_b = timesteps.view(batch_size, *([1] * (target_latents.dim() - 1)))

        if self.use_img2img_flow and input_latents is not None:
            if input_latents.shape != target_latents.shape:
                input_latents = torch.nn.functional.adaptive_avg_pool2d(
                    input_latents, target_latents.shape[-2:]
                )
            if self.flow_sigma > 0:
                noise_term = self.flow_sigma * (t_b * (1.0 - t_b)).sqrt() * noise
            else:
                noise_term = 0.0
            noisy_latents = (1.0 - t_b) * input_latents + t_b * target_latents + noise_term
            velocity_target = target_latents - input_latents
        else:
            noisy_latents = (1.0 - t_b) * target_latents + t_b * noise
            velocity_target = target_latents - noise

        # ── P0-1: Text conditioning ───────────────────────────────────────
        text_cond = self._encode_text(prompts, device)
        prompt_embeds = text_cond["prompt_embeds"]  # [B, seq_len, D_t5]
        pooled_prompt_embeds = text_cond["pooled_prompt_embeds"]  # [B, D_clip] | None
        text_ids = text_cond.get("text_ids")  # [seq_len, 3] | None (FLUX-specific)

        if self.cfg_dropout_rate > 0:
            uncond = self._encode_text([""] * batch_size, device)
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_dropout_rate
            if drop_mask.any():
                drop = drop_mask.view(-1, *([1] * (prompt_embeds.dim() - 1)))
                prompt_embeds = torch.where(drop, uncond["prompt_embeds"], prompt_embeds)
                if (
                    pooled_prompt_embeds is not None
                    and uncond.get("pooled_prompt_embeds") is not None
                ):
                    drop_p = drop_mask.view(-1, *([1] * (pooled_prompt_embeds.dim() - 1)))
                    pooled_prompt_embeds = torch.where(
                        drop_p, uncond["pooled_prompt_embeds"], pooled_prompt_embeds
                    )

        # ── Prepare hidden_states for FLUX transformer ──────────────────
        # FLUX uses a 2×2 latent packing: [B, 16, H, W] → [B, (H/2)*(W/2), 64]
        # The transformer's x_embedder is nn.Linear(64, inner_dim), so it
        # expects the packed sequence form with 64 channels.
        latent_h, latent_w = noisy_latents.shape[-2], noisy_latents.shape[-1]
        latent_channels = noisy_latents.shape[1]
        # Pack: [B, C, H, W] → [B, (H/2)*(W/2), C*4]
        # Detach from VAE computation graph — MPS graph caching conflicts
        # between VAE (float32 internal) and transformer (bfloat16) cause
        # dtype mismatch errors. The latents don't need gradients.
        noisy_latents_packed = self._pack_latents(
            noisy_latents.detach(), batch_size, latent_channels, latent_h, latent_w
        )
        # img_ids correspond to packed spatial dims [H/2, W/2]
        packed_h, packed_w = latent_h // 2, latent_w // 2
        img_ids = self._prepare_latent_image_ids(
            batch_size, packed_h, packed_w, device, noisy_latents.dtype
        )

        # ── Forward pass — use probed signature ────────────────────────
        # The forward mode was detected at init time via _probe_forward_signature,
        # eliminating the per-step try/except fallback chain.
        # Cast all inputs to the model's dtype to avoid MPS dtype mismatches.
        model_dtype = next(self.peft_model.parameters()).dtype
        forward_kwargs: dict = {
            "hidden_states": noisy_latents_packed.to(model_dtype),
            "timestep": timesteps.to(model_dtype),
            "encoder_hidden_states": prompt_embeds.to(model_dtype),
        }
        if self._forward_mode == "full":
            forward_kwargs["pooled_projections"] = pooled_prompt_embeds.to(model_dtype)
            forward_kwargs["txt_ids"] = text_ids.to(model_dtype)
            forward_kwargs["img_ids"] = img_ids.to(model_dtype)

        # ── Run FLUX transformer forward ──────────────────────────────

        # P0-6: Workaround for diffusers PEFT wrapper mis-routing.
        # The diffusers peft_utils.py wrapper occasionally captures the wrong
        # forward_fn (e.g. T5EncoderModel.forward instead of FluxTransformer2DModel.
        # forward).  We unwrap via functools.__wrapped__ and call the original
        # forward directly, then manually apply LoRA scaling/unscaling.
        try:
            predicted_velocity = self.peft_model(**forward_kwargs)
        except TypeError as e:
            if "input_ids" in str(e):
                if not getattr(self, "_peft_misroute_warned", False):
                    logger.warning(
                        "PEFT wrapper mis-routed forward call (%s); "
                        "falling back to unwrapped transformer forward. "
                        "(This warning is shown only once.)", e
                    )
                    self._peft_misroute_warned = True
                transformer = self.peft_model.base_model.model
                # Unwrap the diffusers wrapper to get the real forward
                raw_forward = transformer.forward
                while hasattr(raw_forward, "__wrapped__"):
                    raw_forward = raw_forward.__wrapped__
                # Apply LoRA scaling before the raw forward
                # (USE_PEFT_BACKEND removed in diffusers ≥ 0.31; scale/unscale
                #  are safe no-ops when no LoRA layers are present.)
                from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
                scale_lora_layers(transformer, 1.0)
                try:
                    predicted_velocity = raw_forward(transformer, **forward_kwargs)
                finally:
                    unscale_lora_layers(transformer, 1.0)
            else:
                raise

        # Extract tensor from Transformer2DModelOutput if needed
        if hasattr(predicted_velocity, "sample"):
            predicted_velocity = predicted_velocity.sample

        # Handle FLUX output: transformer returns [B, (H/2)*(W/2), C*4] (packed sequence).
        # Unpack back to [B, C, H, W] to match velocity_target.
        if predicted_velocity.dim() == 3 and velocity_target.dim() == 4:
            H, W = velocity_target.shape[-2], velocity_target.shape[-1]
            predicted_velocity = self._unpack_latents(predicted_velocity, H, W)

        # Ensure shape matches velocity_target before computing loss.
        # (Some transformers return a different channel/spatial layout.)
        if predicted_velocity.shape != velocity_target.shape:
            logger.warning(
                f"Shape mismatch: predicted {predicted_velocity.shape} vs "
                f"target {velocity_target.shape}. Attempting adaptive pool."
            )
            predicted_velocity = torch.nn.functional.adaptive_avg_pool2d(
                predicted_velocity, velocity_target.shape[-2:]
            )
            if predicted_velocity.shape[1] != velocity_target.shape[1]:
                predicted_velocity = predicted_velocity[:, : velocity_target.shape[1]]

        return self._compute_flow_loss(predicted_velocity, velocity_target, timesteps)

    def _compute_flow_loss(
        self,
        predicted_velocity: torch.Tensor,
        velocity_target: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Per-channel variance-normalised MSE with optional Min-SNR weighting."""
        sq_err = (predicted_velocity - velocity_target) ** 2

        if self.channel_loss_weights is not None:
            weights = torch.as_tensor(
                self.channel_loss_weights,
                dtype=sq_err.dtype,
                device=sq_err.device,
            ).view(1, -1, 1, 1)
            per_sample = (sq_err * weights).sum(dim=(1, 2, 3)) / (
                weights.sum() * sq_err.shape[2] * sq_err.shape[3]
            )
        else:
            channel_var = (
                (velocity_target - velocity_target.mean(dim=(0, 2, 3), keepdim=True)) ** 2
            ).mean(dim=(0, 2, 3)).clamp(min=1e-8)
            per_sample = (sq_err / channel_var.view(1, -1, 1, 1)).mean(dim=(1, 2, 3))

        if self.use_min_snr:
            snr_weights = compute_min_snr_weight(timesteps, gamma=self.min_snr_gamma)
            return (per_sample * snr_weights).mean()
        return per_sample.mean()

    def _evaluate(self, val_dataset: _Dataset) -> float:
        """Evaluate on validation set and return average loss."""
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
        return avg_loss

    def _create_scheduler(
        self,
        scheduler_type: str,
        total_steps: int,
        warmup_steps: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Create LR scheduler via SchedulerFactory (P2-19)."""
        return SchedulerFactory.create(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps if warmup_steps > 0 else None,
            warmup_ratio=self.warmup_ratio if self.warmup_ratio is not None else 0.1,
            scheduler_type=scheduler_type,
        )


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


__all__ = ["InstructionTuner", "VisionTaskDataset", "EarlyStopping", "TrainingLogger"]
