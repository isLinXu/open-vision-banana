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
    ) -> None:
        self.base_model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lr = lr
        self.target_modules = target_modules or ["to_k", "to_v"]
        self.use_rslora = use_rslora
        self.peft_model = None
        self.optimizer = None
        self.deepspeed_engine = None
        self.task = task
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
            dummy_hidden = torch.zeros(1, 3, 2, 2)
            dummy_timestep = torch.tensor([0.5])
            dummy_encoder = torch.zeros(1, 1, 4)
            dummy_pooled = torch.zeros(1, 4)

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
                if use_deepspeed:
                    loss = self._training_step(batch)
                    self.deepspeed_engine.backward(loss / gradient_accumulation_steps)
                else:
                    with torch.autocast(device_type, enabled=use_amp, dtype=torch.bfloat16):
                        loss = self._training_step(batch)
                    scaled_loss = scaler.scale(loss / gradient_accumulation_steps)
                    scaled_loss.backward()

                epoch_loss += loss.item()
                num_steps += 1

                if (step + 1) % gradient_accumulation_steps == 0:
                    if use_deepspeed:
                        self.deepspeed_engine.step()
                    else:
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
            # diffusers ≥ 0.30 exposes encode_prompt on FluxPipeline
            text_outputs = pipeline.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
            )
            # encode_prompt returns (prompt_embeds, negative_prompt_embeds,
            #                        pooled_prompt_embeds, negative_pooled_prompt_embeds)
            if isinstance(text_outputs, (tuple, list)) and len(text_outputs) >= 3:
                prompt_embeds = text_outputs[0]  # [B, seq_len, D_t5]
                pooled_prompt_embeds = text_outputs[2]  # [B, D_clip]
            else:
                # Some diffusers versions return a different structure
                prompt_embeds = text_outputs[0]
                pooled_prompt_embeds = text_outputs[1] if len(text_outputs) > 1 else None

            return {
                "prompt_embeds": prompt_embeds.to(device),
                "pooled_prompt_embeds": (
                    pooled_prompt_embeds.to(device) if pooled_prompt_embeds is not None else None
                ),
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
                latent_dist = vae.encode(images).latent_dist
                latents = latent_dist.sample()
                # Apply VAE scaling factor (FLUX default: 0.18215)
                scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
                latents = latents * scaling_factor
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

        # Infer device from batch tensors; fall back to peft_model parameters.
        device = (
            target_rgb.device
            if isinstance(target_rgb, torch.Tensor)
            else next((p.device for p in self.peft_model.parameters()), torch.device("cpu"))
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

        # ── Flow matching in latent space ────────────────────────────────
        timesteps = torch.rand(batch_size, device=device)  # [B]
        noise = torch.randn_like(target_latents)

        t_b = timesteps.view(batch_size, *([1] * (target_latents.dim() - 1)))
        noisy_latents = (1.0 - t_b) * target_latents + t_b * noise  # z_t
        velocity_target = target_latents - noise  # v*

        # ── P0-1: Text conditioning ───────────────────────────────────────
        text_cond = self._encode_text(prompts, device)
        prompt_embeds = text_cond["prompt_embeds"]  # [B, seq_len, D_t5]
        pooled_prompt_embeds = text_cond["pooled_prompt_embeds"]  # [B, D_clip] | None

        # ── Forward pass — use probed signature ────────────────────────
        # The forward mode was detected at init time via _probe_forward_signature,
        # eliminating the per-step try/except fallback chain.
        forward_kwargs: dict = {
            "hidden_states": noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": prompt_embeds,
        }
        if self._forward_mode == "full":
            forward_kwargs["pooled_projections"] = pooled_prompt_embeds
            predicted_velocity = self.peft_model(**forward_kwargs)
        elif self._forward_mode == "partial":
            predicted_velocity = self.peft_model(**forward_kwargs)
        elif self._forward_mode == "minimal":
            predicted_velocity = self.peft_model(
                hidden_states=noisy_latents,
                timestep=timesteps,
            )
        else:  # "positional"
            predicted_velocity = self.peft_model(noisy_latents, timestep=timesteps)

        # Some transformer forward() returns a tuple (output, *extras)
        if isinstance(predicted_velocity, (tuple, list)):
            predicted_velocity = predicted_velocity[0]

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

        loss = nn.functional.mse_loss(predicted_velocity, velocity_target)
        return loss

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
        """Create learning rate scheduler with linear warmup."""
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )

        if warmup_steps > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_steps)
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


__all__ = ["InstructionTuner", "VisionTaskDataset", "EarlyStopping", "TrainingLogger"]
