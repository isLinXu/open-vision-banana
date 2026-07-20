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

from ..model_registry import ModelConfig, resolve_model_config
from ..model_registry import vae_encode as registry_vae_encode
from .checkpoint import CheckpointManager
from .early_stopping import EarlyStopping
from .logging import TrainingLogger
from .optimization import (
    SchedulerFactory,
    compute_min_snr_weight,
    sample_timesteps_logit_normal,
)
from .trainer_interface import TrainerInterface
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
                img = img.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            arr = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            lo, hi = normalise_range
            return arr * (hi - lo) + lo

        image = _load(sample["image_path"], (-1.0, 1.0))
        target_rgb = _load(sample["target_path"], (0.0, 1.0))

        return {
            "prompt": sample["prompt"],
            "image": image,
            "target_rgb": target_rgb,
            "target_rgb_range": "0_1",
            "task": self.task,
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

        # ── Resolve model config from the base model ───────────────────
        # This enables multi-model support: the tuner adapts its text
        # encoding, VAE encoding, and forward pass to the detected family.
        model_name = getattr(model, "model_name", None) or getattr(model, "DEFAULT_MODEL", None)
        if model_name is not None:
            self.model_config: ModelConfig = resolve_model_config(model_name)
        else:
            # Fallback for raw pipelines without a BaseGenerator wrapper
            self.model_config = resolve_model_config(
                getattr(model, "name", "black-forest-labs/FLUX.1-schnell")
            )

        family = self.model_config.family
        logger.info(f"InstructionTuner: resolved model family='{family}', "
                     f"pipeline='{self.model_config.pipeline_class}'")

        # Use model-appropriate LoRA target modules if not explicitly set
        if target_modules is not None:
            self.target_modules = target_modules
        else:
            self.target_modules = self.model_config.lora_target_modules or [
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out", "ff.net.0.proj", "ff.net.2",
            ]
            logger.info(f"Using model-specific LoRA targets: {self.target_modules}")

        self.channel_loss_weights = channel_loss_weights
        self.use_logit_normal_timesteps = use_logit_normal_timesteps
        self.use_min_snr = use_min_snr
        self.min_snr_gamma = min_snr_gamma
        self.cfg_dropout_rate = cfg_dropout_rate
        self.use_img2img_flow = use_img2img_flow
        self.flow_sigma = flow_sigma
        self.warmup_ratio = warmup_ratio
        self.use_rslora = use_rslora
        self.peft_model: nn.Module | Any = None
        self.optimizer: torch.optim.Optimizer | Any = None
        self.deepspeed_engine: Any = None
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

        # ── P0-7: Detect PEFT mis-route at init time ────────────────────
        # diffusers' PEFT integration may mis-route forward() calls to the
        # wrong module (e.g. T5EncoderModel.forward which expects input_ids).
        # When detected, we bypass the PEFT wrapper entirely and call the
        # underlying transformer's forward directly with manual LoRA scaling.
        self._use_direct_forward = self._detect_peft_misroute()

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

            # Try full first, then partial, then minimal.
            # Catch ALL exceptions: TypeError means wrong calling convention,
            # RuntimeError means wrong input shape (e.g. mock Conv2d expects 3
            # channels but dummy uses 16).  Either way, this mode doesn't work
            # — fall through to the next.
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
                except Exception:
                    continue

        return "positional"

    def _detect_peft_misroute(self) -> bool:
        """Detect whether the PEFT wrapper mis-routes forward() calls.

        diffusers' PEFT integration (peft_utils.py) sometimes captures the
        wrong ``forward_fn`` — e.g. T5EncoderModel.forward instead of
        FluxTransformer2DModel.forward.  When this happens, calling the
        PEFT-wrapped model with FLUX kwargs (hidden_states, timestep, etc.)
        raises a TypeError mentioning 'input_ids'.

        We detect this once at init time by inspecting the PEFT wrapper's
        forward signature.  If the wrapper's forward accepts 'input_ids'
        but NOT 'hidden_states', it's mis-routed.

        Returns:
            True if the PEFT wrapper is mis-routed and should be bypassed.
        """
        import inspect

        try:
            sig = inspect.signature(self.peft_model.forward)
        except (ValueError, TypeError):
            return False

        params = {
            name
            for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }

        # Mis-route: wrapper accepts input_ids (T5 signature) but not
        # hidden_states (FLUX transformer signature)
        has_input_ids = "input_ids" in params
        has_hidden_states = "hidden_states" in params
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

        if has_input_ids and not has_hidden_states:
            logger.warning(
                "PEFT wrapper mis-route detected at init: forward() accepts "
                "'input_ids' but not 'hidden_states'. Bypassing PEFT wrapper "
                "and calling underlying transformer forward directly."
            )
            return True

        # If the wrapper has **kwargs (opaque), do a single trial call to
        # check whether it actually works with FLUX kwargs.
        if has_kwargs and not has_hidden_states:
            try:
                dummy = torch.zeros(1, 4, 64, device="cpu", dtype=torch.float32)
                self.peft_model(
                    hidden_states=dummy,
                    timestep=torch.tensor([0.5]),
                    encoder_hidden_states=torch.zeros(1, 1, 4096),
                    pooled_projections=torch.zeros(1, 768),
                )
            except TypeError as e:
                if "input_ids" in str(e):
                    logger.warning(
                        "PEFT wrapper mis-route detected (trial call): %s. "
                        "Bypassing PEFT wrapper for all training steps.", e
                    )
                    return True
            except Exception:
                pass  # Other errors (shape mismatch etc.) are expected; not a mis-route

        return False

    def _direct_forward(self, forward_kwargs: dict) -> torch.Tensor:
        """Call the underlying transformer's forward directly, bypassing PEFT wrapper.

        Used when _detect_peft_misroute() returns True.  Manually applies
        LoRA scale/unscale around the raw forward call.

        Important: we call transformer.forward WITHOUT unwrapping __wrapped__
        because the PEFT wrapper patches forward with LoRA-adapted weights.
        Unwrapping would bypass the LoRA layers entirely, breaking the
        computation graph (loss has no grad_fn).
        """
        transformer = self.peft_model.base_model.model  # type: ignore[union-attr]

        from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
        scale_lora_layers(transformer, 1.0)
        try:
            output = transformer(**forward_kwargs)
        finally:
            unscale_lora_layers(transformer, 1.0)
        return output

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
        # SDXL/SD1.5/SD2 use pipeline.unet; FLUX/SD3/PixArt use pipeline.transformer
        backbone = getattr(pipeline, "transformer", None) or getattr(pipeline, "unet", None)
        if backbone is None:
            raise AttributeError(
                f"Could not resolve 'transformer' or 'unet' from base_model. "
                f"base_model type: {type(self.base_model)}"
            )
        transformer = backbone  # Keep variable name for dtype logic below
        self.peft_model = get_peft_model(backbone, lora_config)
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
        start_step: int = 0,
        start_epoch: int = 0,
        resume_checkpoint: str | Path | None = None,
        restore_rng_on_resume: bool = False,
    ) -> dict:
        """Train the LoRA instruction alignment.

        Args:
            start_step: Resume global step (for LR scheduler offset).
            start_epoch: Resume epoch index.

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

        total_steps = (
            len(train_loader) * epochs + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        scheduler = self._create_scheduler(scheduler_type, total_steps, warmup_steps)

        device_type = self._detect_device_type()
        use_deepspeed = self.deepspeed_engine is not None

        # ── Determine whether to use TrainerInterface or legacy path ──
        use_trainer_interface = self.trainer_backend is not None

        if use_trainer_interface:
            # Delegate to TrainerInterface — it handles backward, grad scaling,
            # optimizer step, gradient clipping, and accumulation internally.
            trainer: Any = self.trainer_backend
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

        global_step = start_step
        best_val_loss: float | None = None
        stopped_early = False

        scheduler_state_loaded = False
        if resume_checkpoint is not None:
            resume_path = Path(resume_checkpoint)
            resume_root = resume_path.parent
            resume_step = start_step
            if resume_step <= 0 and resume_path.name.startswith("checkpoint-"):
                try:
                    resume_step = int(resume_path.name.split("-")[-1])
                except ValueError:
                    resume_step = None
            resume_manager = CheckpointManager(str(resume_root))
            resume_manager.load(
                self.peft_model,
                optimizer=self.optimizer,
                ema=self.ema,
                scheduler=scheduler,
                step=resume_step,
                restore_rng=restore_rng_on_resume,
            )
            scheduler_state_loaded = (resume_path / "scheduler.pt").exists()
            logger.info("Loaded full training state from %s", resume_path)

        # Only fast-forward when no scheduler state was restored.
        if start_step > 0 and not scheduler_state_loaded:
            for _ in range(start_step):
                scheduler.step()
            logger.info(f"Resumed LR scheduler to step {start_step}, lr={scheduler.get_last_lr()[0]:.2e}")

        for epoch in range(start_epoch, epochs):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            epoch_loss = 0.0
            num_steps = 0
            accumulated_steps = 0

            for step, batch in enumerate(train_loader):
                if use_trainer_interface:
                    # ── TrainerInterface path ──
                    with torch.autocast(
                        device_type, enabled=use_amp, dtype=torch.bfloat16
                    ):
                        loss = self._training_step(batch)
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {epoch}, batch {step}"
                        )
                    stepped = trainer.train_step(self.peft_model, loss, step)
                    accumulated_steps += 1
                elif use_deepspeed:
                    loss = self._training_step(batch)
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {epoch}, batch {step}"
                        )
                    self.deepspeed_engine.backward(loss / gradient_accumulation_steps)
                    accumulated_steps += 1
                    stepped = (step + 1) % gradient_accumulation_steps == 0
                    if stepped:
                        self.deepspeed_engine.step()
                        scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        accumulated_steps = 0
                else:
                    # ── Legacy native path (backward compat) ──
                    with torch.autocast(device_type, enabled=use_amp, dtype=torch.bfloat16):
                        loss = self._training_step(batch)
                    if not torch.isfinite(loss).all():
                        self.optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {epoch}, batch {step}"
                        )
                    scaled_loss = scaler.scale(loss / gradient_accumulation_steps)
                    scaled_loss.backward()

                    epoch_loss += loss.item()
                    num_steps += 1
                    accumulated_steps += 1

                    if (step + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.peft_model.parameters(),
                            max_grad_norm,
                            error_if_nonfinite=True,
                        )
                        scaler.step(self.optimizer)
                        scaler.update()
                        scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        accumulated_steps = 0
                        if self.ema is not None:
                            self.ema.update(self._ema_module())

                    running_loss: float = float(epoch_loss) / max(num_steps, 1)
                    current_lr = float(scheduler.get_last_lr()[0])

                    if global_step > 0 and global_step % log_every == 0 and getattr(self, "_last_logged_step", -1) != global_step:
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
                        accumulated_steps = 0

                        if self.ema is not None:
                            self.ema.update(self._ema_module())

                        running_loss = float(epoch_loss) / max(num_steps, 1)  # noqa: F811
                        current_lr = float(scheduler.get_last_lr()[0])

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
                    and global_step > start_step
                    and getattr(self, "_last_saved_step", -1) != global_step
                ):
                    self._last_saved_step = global_step
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

            # Do not carry an incomplete gradient window into the next epoch.
            if accumulated_steps > 0:
                if use_trainer_interface:
                    if trainer.flush_accumulation(accumulated_steps):
                        scheduler.step()
                        global_step += 1
                        if self.ema is not None:
                            self.ema.update(self._ema_module())
                elif use_deepspeed:
                    scale = gradient_accumulation_steps / accumulated_steps
                    for parameter in self.deepspeed_engine.module.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                    self.deepspeed_engine.step()
                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                else:
                    scaler.unscale_(self.optimizer)
                    scale = gradient_accumulation_steps / accumulated_steps
                    for parameter in self.peft_model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                    torch.nn.utils.clip_grad_norm_(
                        self.peft_model.parameters(),
                        max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if self.ema is not None:
                        self.ema.update(self._ema_module())

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

    def _encode_text(self, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor | None]:
        """Encode text prompts using the model's text encoders (multi-model).

        Routes to the appropriate encoding strategy based on model family:
        - FLUX: pipeline.encode_prompt (dual CLIP+T5)
        - Hunyuan-Image / SDXL / SD3: pipeline.encode_prompt (dual-encoder)
        - Qwen-Image / Z-Image / SANA / etc.: registry encode_text

        Returns dict with prompt_embeds, pooled_prompt_embeds, text_ids, attention_mask.
        """
        pipeline = getattr(self.base_model, "_t2i_pipe", None) or getattr(
            self.base_model, "pipe", None
        )
        if pipeline is None:
            pipeline = self.base_model

        family = self.model_config.family

        # FLUX family: use pipeline.encode_prompt (dual CLIP+T5)
        if family == "flux":
            return self._encode_text_flux(prompts, device, pipeline)

        # SDXL, SD3: dual-encoder with pipeline.encode_prompt + pooled embeds
        if family in ("sdxl", "sd3"):
            max_len = getattr(self.model_config, "text_encoder_max_length", 512)
            return self._encode_text_dual_encoder(prompts, device, pipeline, max_len)

        # All other families (hunyuan-image, qwen-image, z-image, sana, etc.):
        # use registry encode_text which handles attention masks, guidance, etc.
        return self._encode_text_registry(prompts, device, pipeline)

    def _encode_text_flux(self, prompts, device, pipeline):
        """FLUX-specific text encoding via pipeline.encode_prompt."""
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
                for enc, dtype in zip(encoders, original_dtypes, strict=False):
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
                "attention_mask": None,
            }

        except Exception as e:
            logger.warning(
                f"Text encoding failed ({e}); falling back to zero conditioning. "
                "The model will train without text guidance for this batch."
            )
            B = len(prompts)
            return {
                "prompt_embeds": torch.zeros(B, 1, 4096, device=device),
                "pooled_prompt_embeds": torch.zeros(B, 768, device=device),
                "text_ids": torch.zeros(1, 3, device=device),
                "attention_mask": None,
            }

    def _encode_text_dual_encoder(self, prompts, device, pipeline, max_length=512):
        """Dual-encoder text encoding for HunyuanImage, SDXL, SD3, etc."""
        try:
            encoders = [
                getattr(pipeline, "text_encoder", None),
                getattr(pipeline, "text_encoder_2", None),
                getattr(pipeline, "text_encoder_3", None),  # SD3 T5-XXL
            ]
            original_dtypes = []
            for enc in encoders:
                if enc is not None:
                    original_dtypes.append(enc.dtype)
                    enc.to(torch.float32)
                else:
                    original_dtypes.append(None)

            try:
                with torch.no_grad():
                    encode_kwargs = dict(
                        prompt=prompts,
                        prompt_2=prompts,
                        device=device,
                        num_images_per_prompt=1,
                    )
                    # SD3 uses a third T5-XXL encoder and accepts max_sequence_length
                    if getattr(pipeline, "text_encoder_3", None) is not None:
                        encode_kwargs["prompt_3"] = prompts
                        encode_kwargs["max_sequence_length"] = max_length
                    # SDXL (no text_encoder_3): StableDiffusionXLPipeline.encode_prompt
                    # does NOT accept max_sequence_length — tokenizers are hard-capped
                    # at 77. Passing it raises TypeError.
                    text_outputs = pipeline.encode_prompt(**encode_kwargs)
            finally:
                for enc, dtype in zip(encoders, original_dtypes, strict=False):
                    if enc is not None and dtype is not None:
                        enc.to(dtype)

            prompt_embeds = text_outputs[0]
            pooled_prompt_embeds = None
            for item in text_outputs[1:]:
                if item is None:
                    continue
                if item.dim() == 2:
                    pooled_prompt_embeds = item

            model_dtype = next(self.peft_model.parameters()).dtype
            return {
                "prompt_embeds": prompt_embeds.to(device=device, dtype=model_dtype),
                "pooled_prompt_embeds": (
                    pooled_prompt_embeds.to(device=device, dtype=model_dtype)
                    if pooled_prompt_embeds is not None else None
                ),
                "text_ids": None,
                "attention_mask": None,
            }
        except Exception as e:
            logger.warning(f"Dual-encoder text encoding failed ({e}); using zero conditioning.")
            B = len(prompts)
            return {
                "prompt_embeds": torch.zeros(B, 1, 4096, device=device),
                "pooled_prompt_embeds": None,
                "text_ids": None,
                "attention_mask": None,
            }

    def _encode_text_registry(self, prompts, device, pipeline):
        """Registry-based text encoding for Qwen-Image, Z-Image, SANA, etc."""
        try:
            from ..model_registry import PipelineComponents, encode_text

            components = PipelineComponents(
                unet=getattr(pipeline, "transformer", getattr(pipeline, "unet", None)),
                vae=pipeline.vae,
                text_encoder=getattr(pipeline, "text_encoder", None),
                text_encoder_2=getattr(pipeline, "text_encoder_2", None),
                text_encoder_3=getattr(pipeline, "text_encoder_3", None),
                tokenizer=getattr(pipeline, "tokenizer", None),
                tokenizer_2=getattr(pipeline, "tokenizer_2", None),
                tokenizer_3=getattr(pipeline, "tokenizer_3", None),
                scheduler=getattr(pipeline, "scheduler", None),
                config=self.model_config,
                text_encoder_4=getattr(pipeline, "text_encoder_4", None),
                tokenizer_4=getattr(pipeline, "tokenizer_4", None),
                processor=getattr(pipeline, "processor", None) or getattr(pipeline, "text_processor", None),
            )

            model_dtype = next(self.peft_model.parameters()).dtype

            encoders = [components.text_encoder, components.text_encoder_2]
            original_dtypes = []
            for enc in encoders:
                if enc is not None:
                    original_dtypes.append(enc.dtype)
                    enc.to(torch.float32)
                else:
                    original_dtypes.append(None)

            try:
                with torch.no_grad():
                    encoder_hidden_states, pooled_or_mask = encode_text(
                        components, prompts, device, torch.float32
                    )
            finally:
                for enc, dtype in zip(encoders, original_dtypes, strict=False):
                    if enc is not None and dtype is not None:
                        enc.to(dtype)

            encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=model_dtype)

            result: dict[str, Any] = {
                "prompt_embeds": encoder_hidden_states,
                "pooled_prompt_embeds": None,
                "text_ids": None,
                "attention_mask": None,
            }

            if self.model_config.needs_attention_mask and pooled_or_mask is not None:
                if isinstance(pooled_or_mask, torch.Tensor):
                    result["attention_mask"] = pooled_or_mask.to(device=device, dtype=model_dtype)
                elif isinstance(pooled_or_mask, tuple):
                    # HunyuanImage: (clip_mask, t5_hidden, t5_mask) — cast each element
                    casted = []
                    for m in pooled_or_mask:
                        if isinstance(m, torch.Tensor):
                            casted.append(m.to(device=device, dtype=model_dtype))
                        else:
                            casted.append(m)
                    result["attention_mask"] = tuple(casted)
            elif pooled_or_mask is not None and isinstance(pooled_or_mask, torch.Tensor):
                result["pooled_prompt_embeds"] = pooled_or_mask.to(device=device, dtype=model_dtype)

            return result

        except Exception as e:
            logger.warning(f"Text encoding failed ({e}); using zero conditioning.")
            B = len(prompts)
            return {
                "prompt_embeds": torch.zeros(B, 1, 4096, device=device),
                "pooled_prompt_embeds": None,
                "text_ids": None,
                "attention_mask": None,
            }

    def _encode_latents(self, images: torch.Tensor) -> torch.Tensor:
        """Encode pixel-space images to VAE latents (multi-model).

        Uses the registry's vae_encode helper which handles model-specific
        scaling factors and encoding conventions. Falls back to a lightweight
        proxy when the VAE is not accessible (e.g. unit tests).
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
            B, C, H, W = images.shape
            latent_h, latent_w = max(1, H // 8), max(1, W // 8)
            latents = torch.nn.functional.adaptive_avg_pool2d(images, (latent_h, latent_w))
            return latents

        try:
            with torch.no_grad():
                vae_device = next(vae.parameters()).device
                vae_dtype = next(vae.parameters()).dtype
                images = images.to(device=vae_device, dtype=vae_dtype)

                # Use registry vae_encode for model-appropriate scaling
                latents = registry_vae_encode(vae, images, self.model_config)

                model_dtype = next(self.peft_model.parameters()).dtype
                latents = latents.to(dtype=model_dtype)
            return latents
        except Exception as e:
            if self.model_config.family == "sdxl":
                raise RuntimeError("SDXL VAE encoding failed; refusing silent fallback") from e
            logger.warning(f"VAE encoding failed ({e}); using zero latents.")
            B, _, H, W = images.shape
            # Use model's expected latent channels (FLUX/Z-Image=16, others=4)
            latent_ch = self.model_config.vae_latent_channels
            vae_sf = self.model_config.vae_downsample_factor
            return torch.zeros(
                B, latent_ch, H // vae_sf, W // vae_sf,
                device=images.device, dtype=images.dtype,
            )

    def _training_step(self, batch: dict) -> torch.Tensor:
        """Dispatch to the objective that belongs to the registered model family."""
        if self.model_config.family == "sdxl":
            return self._training_step_sdxl(batch)
        return self._training_step_flow(batch)

    def _training_step_flow(self, batch: dict) -> torch.Tensor:
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
        target_rgb = batch.get("target_rgb")
        # Handle None (generation task) or list format
        if target_rgb is None:
            # Generation task: use input image as target
            target_rgb = batch.get("image")
        if target_rgb is None:
            raise ValueError("Both target_rgb and image are None in batch")
        if isinstance(target_rgb, list):
            target_rgb = torch.stack(target_rgb) if all(isinstance(t, torch.Tensor) for t in target_rgb) else target_rgb[0]

        input_images = batch.get("image")
        # P0-2: generation samples carry image=None so they route through the
        # pure t2i branch (no img2img conditioning). The collate fn wraps
        # per-sample Nones into a list ([None]), so unwrap it here; a batch
        # whose samples all lack an input image must yield input_images=None.
        if isinstance(input_images, list):
            input_images = next(
                (t for t in input_images if isinstance(t, torch.Tensor)), None
            )
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

        # ── Colormap augmentation logging (paper Section 3.2) ─────────────
        colormaps = batch.get("colormap")
        if colormaps is not None:
            unique_cmaps = set(colormaps)
            if len(unique_cmaps) > 1 or "cube_edge" not in unique_cmaps:
                logger.debug(
                    f"Colormap augmentation active: {dict((c, colormaps.count(c)) for c in unique_cmaps)}"
                )

        # ── P0-2: VAE encoding — work in latent space ───────────────────
        # Convert target_rgb [0,1] → [-1,1] before VAE encoding.
        # VisionTaskDataset historically emits [0, 1], while generation emits
        # [-1, 1].  New datasets may declare the range explicitly; retain the
        # legacy default for existing callers and tests.
        pixel_range = batch.get("target_rgb_range", "0_1")
        if isinstance(pixel_range, list):
            pixel_range = pixel_range[0]
        target_pixels = target_rgb if pixel_range == "-1_1" else target_rgb * 2.0 - 1.0
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
                noise_term = torch.zeros_like(input_latents)
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

        # ── Prepare hidden_states for the model's transformer/UNet ──────
        latent_h, latent_w = noisy_latents.shape[-2], noisy_latents.shape[-1]
        latent_channels = noisy_latents.shape[1]
        model_dtype = next(self.peft_model.parameters()).dtype
        family = self.model_config.family

        attention_mask = text_cond.get("attention_mask")

        # FLUX uses 2x2 latent packing: [B, 16, H, W] -> [B, (H/2)*(W/2), 64]
        if family == "flux" and latent_channels == 16:
            noisy_latents_input = self._pack_latents(
                noisy_latents.detach(), batch_size, latent_channels, latent_h, latent_w
            )
            packed_h, packed_w = latent_h // 2, latent_w // 2
            img_ids = self._prepare_latent_image_ids(
                batch_size, packed_h, packed_w, device, noisy_latents.dtype
            )
        else:
            # Non-FLUX models: use latents directly without packing
            noisy_latents_input = noisy_latents.detach()
            img_ids = None

        # ── Forward pass ────────────────────────────────────────────────
        # Route to the appropriate forward strategy based on model family.
        if family == "flux":
            predicted_velocity = self._forward_flux(
                noisy_latents_input, timesteps, prompt_embeds,
                pooled_prompt_embeds, text_ids, img_ids,
                model_dtype, device, batch_size,
            )
        else:
            # Non-FLUX: use registry backbone_forward for unified calling convention
            predicted_velocity = self._forward_registry(
                noisy_latents_input, timesteps, prompt_embeds,
                pooled_prompt_embeds, attention_mask,
                model_dtype, device,
            )

        # Extract tensor from Transformer2DModelOutput if needed
        if hasattr(predicted_velocity, "sample"):
            predicted_velocity = predicted_velocity.sample

        # Ensure shape matches velocity_target before computing loss.
        # (Some transformers return a different channel/spatial layout.)
        if predicted_velocity.shape != velocity_target.shape:
            raise RuntimeError(
                f"Model output shape {tuple(predicted_velocity.shape)} does not match "
                f"target shape {tuple(velocity_target.shape)}; refusing silent repair."
            )

        return self._compute_flow_loss(predicted_velocity, velocity_target, timesteps)

    def _training_step_sdxl(self, batch: dict) -> torch.Tensor:
        """Scheduler-aware SDXL epsilon/v-prediction objective.

        SDXL uses the discrete training scheduler; it must not consume the
        continuous FLUX interpolation or input-target velocity target.
        """
        target = batch.get("target_rgb")
        if target is None:
            raise ValueError("SDXL batch requires target_rgb")
        if isinstance(target, list):
            target = torch.stack(target)
        value_range = batch.get("target_rgb_range", "0_1")
        if isinstance(value_range, list):
            value_range = value_range[0]
        pixels = target if value_range == "-1_1" else target * 2.0 - 1.0
        device = next((p.device for p in self.peft_model.parameters()), pixels.device)
        pixels = pixels.to(device)
        b = pixels.shape[0]
        z0 = self._encode_latents(pixels)

        pipeline = getattr(self.base_model, "_t2i_pipe", None) or getattr(self.base_model, "pipe", None) or self.base_model
        scheduler = getattr(pipeline, "scheduler", None)
        if scheduler is None or not hasattr(scheduler, "add_noise"):
            raise RuntimeError("SDXL training requires a pipeline scheduler with add_noise()")
        cfg = getattr(scheduler, "config", scheduler)
        n_train = int(getattr(cfg, "num_train_timesteps", 1000))
        timesteps = torch.randint(0, n_train, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        noisy = scheduler.add_noise(z0, noise, timesteps)

        prompts = batch.get("prompt", [""] * b)
        if isinstance(prompts, str):
            prompts = [prompts]
        text = self._encode_text(prompts, device)
        prompt_embeds = text["prompt_embeds"]
        pooled = text.get("pooled_prompt_embeds")
        if pooled is None:
            raise RuntimeError("SDXL requires pooled_prompt_embeds from both text encoders")
        pred = self._forward_registry(
            noisy, timesteps, prompt_embeds, pooled, text.get("attention_mask"),
            next(self.peft_model.parameters()).dtype, device,
            micro_conditioning=batch,
        )
        if hasattr(pred, "sample"):
            pred = pred.sample
        if pred.shape != z0.shape:
            raise RuntimeError(f"SDXL output shape {tuple(pred.shape)} != latent shape {tuple(z0.shape)}")

        prediction_type = str(getattr(cfg, "prediction_type", "epsilon"))
        if prediction_type == "epsilon":
            target_value = noise
        elif prediction_type == "v_prediction":
            if not hasattr(scheduler, "get_velocity"):
                raise RuntimeError("v_prediction scheduler must implement get_velocity()")
            target_value = scheduler.get_velocity(z0, noise, timesteps)
        else:
            raise ValueError(f"Unsupported SDXL scheduler prediction_type: {prediction_type}")
        per_sample = (pred.float() - target_value.float()).square().mean(dim=(1, 2, 3))
        if self.use_min_snr:
            alpha = getattr(scheduler, "alphas_cumprod", None)
            if alpha is None:
                raise RuntimeError("Min-SNR requires scheduler.alphas_cumprod")
            alpha = alpha.to(device=device, dtype=torch.float32)[timesteps]
            snr = alpha / (1.0 - alpha).clamp_min(1e-12)
            cap = torch.minimum(snr, torch.full_like(snr, self.min_snr_gamma))
            weight = cap / (snr + 1.0 if prediction_type == "v_prediction" else snr.clamp_min(1e-12))
            per_sample = per_sample * weight
        return per_sample.mean()

    def _forward_flux(
        self, noisy_latents_input, timesteps, prompt_embeds,
        pooled_prompt_embeds, text_ids, img_ids,
        model_dtype, device, batch_size,
    ) -> torch.Tensor:
        """FLUX-specific forward pass with PEFT wrapper handling."""
        forward_kwargs: dict = {
            "hidden_states": noisy_latents_input.to(model_dtype),
            "timestep": timesteps.to(model_dtype),
            "encoder_hidden_states": prompt_embeds.to(model_dtype),
        }
        if self._forward_mode == "full":
            forward_kwargs["pooled_projections"] = (
                pooled_prompt_embeds.to(model_dtype)
                if pooled_prompt_embeds is not None
                else torch.zeros(batch_size, 768, device=device, dtype=model_dtype)
            )
            if text_ids is not None:
                forward_kwargs["txt_ids"] = text_ids.to(model_dtype)
            if img_ids is not None:
                forward_kwargs["img_ids"] = img_ids.to(model_dtype)

        if self._use_direct_forward:
            predicted_velocity = self._direct_forward(forward_kwargs)
        else:
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
                    self._use_direct_forward = True
                    predicted_velocity = self._direct_forward(forward_kwargs)
                else:
                    raise

        if hasattr(predicted_velocity, "sample"):
            predicted_velocity = predicted_velocity.sample

        # Handle FLUX packed output: [B, (H/2)*(W/2), C*4] -> [B, C, H, W]
        if predicted_velocity.dim() == 3:
            # Unpack using the original latent spatial dims
            # Infer H, W from the packed sequence length
            seq_len = predicted_velocity.shape[1]
            # packed_h * packed_w = seq_len, and H = 2*packed_h
            import math
            side = int(math.isqrt(seq_len))
            if side * side == seq_len:
                predicted_velocity = self._unpack_latents(predicted_velocity, side * 2, side * 2)

        return predicted_velocity

    def _forward_registry(
        self, noisy_latents_input, timesteps, prompt_embeds,
        pooled_prompt_embeds, attention_mask,
        model_dtype, device, micro_conditioning=None,
    ) -> torch.Tensor:
        """Registry-based forward pass for non-FLUX models.

        Uses backbone_forward() from the model registry which handles
        architecture-specific calling conventions for Qwen-Image, Z-Image,
        SANA, Lumina2, HunyuanDiT, PixArt, SD1.5/2.1, SDXL, etc.
        """
        from ..model_registry import backbone_forward, build_added_cond_kwargs

        # Cast inputs to model dtype
        noisy_latents_input = noisy_latents_input.to(model_dtype)
        timesteps = timesteps if self.model_config.family == "sdxl" else timesteps.to(model_dtype)
        prompt_embeds = prompt_embeds.to(model_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(model_dtype)
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.to(model_dtype)

        # Get guidance scale for models that need it
        guidance = None
        if self.model_config.needs_guidance_in_forward:
            guidance = self.model_config.default_guidance_scale

        # Build added_cond_kwargs for SDXL/Kolors-family UNet models.
        # backbone_forward's UNet path requires added_cond_kwargs (not raw
        # pooled_embeds); without this SDXL/SSD-1B training raises a shape
        # error because the UNet expects pooled text_embeds + time_ids.
        added_cond_kwargs = None
        if self.model_config.needs_added_cond_kwargs and pooled_prompt_embeds is not None:
            batch_size = noisy_latents_input.shape[0]
            latent_h, latent_w = noisy_latents_input.shape[-2], noisy_latents_input.shape[-1]
            # Prefer per-sample SDXL micro-conditioning metadata.  Falling
            # back to the final canvas is only for legacy batches.
            def _meta(name, default):
                value = (micro_conditioning or {}).get(name, default)
                if isinstance(value, list):
                    value = torch.as_tensor(value, device=device)
                if not isinstance(value, torch.Tensor):
                    value = torch.full((batch_size, 2), value, device=device)
                return value.to(device=device, dtype=model_dtype)

            original = _meta("original_size", [latent_h * 8, latent_w * 8])
            crop = _meta("crop_top_left", [0, 0])
            target_size = _meta("target_size", [latent_h * 8, latent_w * 8])
            if original.ndim == 1:
                original = original.unsqueeze(0).expand(batch_size, -1)
            if crop.ndim == 1:
                crop = crop.unsqueeze(0).expand(batch_size, -1)
            if target_size.ndim == 1:
                target_size = target_size.unsqueeze(0).expand(batch_size, -1)
            time_ids = torch.cat((original, crop, target_size), dim=1)
            added_cond_kwargs = build_added_cond_kwargs(
                config=self.model_config,
                pooled_embeds=pooled_prompt_embeds,
                resolution=(latent_h * 8, latent_w * 8),
                device=device,
                dtype=model_dtype,
                batch_size=batch_size,
            )
            if added_cond_kwargs is not None:
                added_cond_kwargs["time_ids"] = time_ids

        # Use PEFT-wrapped model as the backbone
        backbone = self.peft_model

        # Handle PEFT mis-route for non-FLUX models too
        if self._use_direct_forward:
            # Bypass PEFT wrapper, call underlying model directly
            transformer = self.peft_model.base_model.model  # type: ignore[union-attr]
            from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
            scale_lora_layers(transformer, 1.0)
            try:
                predicted_velocity = backbone_forward(
                    backbone=transformer,
                    latents=noisy_latents_input,
                    timesteps=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    config=self.model_config,
                    added_cond_kwargs=added_cond_kwargs,
                    guidance=guidance,
                    attention_mask=attention_mask,
                    pooled_embeds=pooled_prompt_embeds,
                )
            finally:
                unscale_lora_layers(transformer, 1.0)
        else:
            try:
                predicted_velocity = backbone_forward(
                    backbone=backbone,
                    latents=noisy_latents_input,
                    timesteps=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    config=self.model_config,
                    added_cond_kwargs=added_cond_kwargs,
                    guidance=guidance,
                    attention_mask=attention_mask,
                    pooled_embeds=pooled_prompt_embeds,
                )
            except TypeError as e:
                # PEFT mis-route: wrapper calls forward with wrong param names
                # (e.g. "hidden_states" for Z-Image/Qwen-Image, "input_ids" for text encoders)
                err_str = str(e)
                if "input_ids" in err_str or "hidden_states" in err_str or "got an unexpected" in err_str or "positional argument" in err_str:
                    logger.warning(
                        "PEFT mis-route detected for %s (%s); bypassing wrapper.",
                        self.model_config.family, err_str[:100],
                    )
                    self._use_direct_forward = True
                    transformer = self.peft_model.base_model.model  # type: ignore[union-attr]
                    from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
                    scale_lora_layers(transformer, 1.0)
                    try:
                        predicted_velocity = backbone_forward(
                            backbone=transformer,
                            latents=noisy_latents_input,
                            timesteps=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            config=self.model_config,
                            guidance=guidance,
                            attention_mask=attention_mask,
                            pooled_embeds=pooled_prompt_embeds,
                        )
                    finally:
                        unscale_lora_layers(transformer, 1.0)
                else:
                    raise

        if hasattr(predicted_velocity, "sample"):
            predicted_velocity = predicted_velocity.sample

        return predicted_velocity

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
    """Schema-aware collate that rejects heterogeneous tensor/None fields."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    keys = set().union(*(sample.keys() for sample in batch))
    result: dict = {}
    for key in keys:
        values = [sample.get(key) for sample in batch]
        if all(isinstance(v, torch.Tensor) for v in values):
            try:
                result[key] = torch.stack(values)
            except RuntimeError as exc:
                raise ValueError(f"Cannot stack field {key!r}: {[tuple(v.shape) for v in values]}") from exc
        elif all(isinstance(v, str) for v in values):
            result[key] = values
        elif key == "image" and all(v is None for v in values):
            # Generation has no visual condition; keep this explicit rather
            # than manufacturing a zero image that can be mistaken for one.
            result[key] = None
        elif all(v is None for v in values):
            result[key] = None
        elif all(isinstance(v, (int, float, bool)) for v in values):
            result[key] = values
        else:
            raise ValueError(f"Heterogeneous batch field {key!r}: {[type(v).__name__ for v in values]}")
    return result


__all__ = ["InstructionTuner", "VisionTaskDataset", "EarlyStopping", "TrainingLogger"]
