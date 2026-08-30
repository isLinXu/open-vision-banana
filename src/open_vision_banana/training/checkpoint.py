"""Checkpoint management for Vision Banana training.

Handles saving and loading of:
- LoRA weights (via PEFT)
- EMA shadow parameters
- Optimizer state
- Training step / epoch metadata
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path

import torch

from ..conditioning import (
    CONV_IN_STATE_NAME,
    SourceConditioningConfig,
    read_source_conditioning_config,
    write_source_conditioning_config,
)
from ..spatial_adapter import SPATIAL_ADAPTER_STATE_NAME

logger = logging.getLogger(__name__)

DIFFUSERS_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


class CheckpointManager:
    """Manages training checkpoint save/load with metadata.

    Each checkpoint directory contains:
    - adapter_model.safetensors / adapter_config.json (PEFT LoRA)
    - ema_weights.pt (EMA shadow parameters, if available)
    - optimizer.pt (optimizer state dict)
    - metadata.json (step, epoch, loss, lr)
    """

    def __init__(self, output_dir: str, max_keep: int = 5) -> None:
        self.output_dir = Path(output_dir)
        self.max_keep = max_keep
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        peft_model,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        loss: float,
        lr: float,
        ema=None,
        scheduler=None,
        conditioning_module=None,
        conditioning_config: SourceConditioningConfig | None = None,
        spatial_adapter=None,
        diffusers_component: str | None = None,
        extra_metadata: dict | None = None,
    ) -> Path:
        """Save a training checkpoint."""
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        peft_model.save_pretrained(ckpt_dir)
        if diffusers_component is not None:
            self._save_diffusers_lora_weights(
                peft_model,
                ckpt_dir,
                component=diffusers_component,
            )

        # Save EMA weights
        if ema is not None:
            ema_path = ckpt_dir / "ema_weights.pt"
            torch.save(ema.shadow_params, ema_path)

        # Save optimizer state
        opt_path = ckpt_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), opt_path)

        # Save scheduler state
        if scheduler is not None:
            sched_path = ckpt_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), sched_path)

        self._save_conditioning(
            ckpt_dir,
            conditioning_module=conditioning_module,
            conditioning_config=conditioning_config,
            spatial_adapter=spatial_adapter,
        )

        # Save metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Keep stochastic state with the checkpoint so a resumed run can
        # reproduce sampling and augmentation decisions.
        torch.save(self._capture_rng_state(), ckpt_dir / "rng_state.pt")

        logger.info(f"Checkpoint saved to {ckpt_dir}")

        # Prune old checkpoints
        self._prune_old_checkpoints()
        return ckpt_dir

    def save_best(
        self,
        peft_model,
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        ema=None,
        scheduler=None,
        conditioning_module=None,
        conditioning_config: SourceConditioningConfig | None = None,
        spatial_adapter=None,
        diffusers_component: str | None = None,
        extra_metadata: dict | None = None,
    ) -> Path:
        """Save the best validation checkpoint to ``checkpoint-best/``."""
        ckpt_dir = self.output_dir / "checkpoint-best"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        peft_model.save_pretrained(ckpt_dir)
        if diffusers_component is not None:
            self._save_diffusers_lora_weights(
                peft_model,
                ckpt_dir,
                component=diffusers_component,
            )

        if ema is not None:
            torch.save(ema.shadow_params, ckpt_dir / "ema_weights.pt")

        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        self._save_conditioning(
            ckpt_dir,
            conditioning_module=conditioning_module,
            conditioning_config=conditioning_config,
            spatial_adapter=spatial_adapter,
        )

        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "best": True,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        torch.save(self._capture_rng_state(), ckpt_dir / "rng_state.pt")

        logger.info("Best checkpoint saved to %s (val_loss=%.6f)", ckpt_dir, val_loss)
        return ckpt_dir

    def load(
        self,
        peft_model,
        optimizer: torch.optim.Optimizer | None = None,
        ema=None,
        scheduler=None,
        step: int | None = None,
        restore_rng: bool = False,
        allow_partial_resume: bool = False,
        conditioning_module=None,
        conditioning_config: SourceConditioningConfig | None = None,
        spatial_adapter=None,
    ) -> dict:
        """Load a training checkpoint.

        Args:
            peft_model: PEFT model to load LoRA weights into.
            optimizer: Optimizer to load state into.
            ema: EMA model to load shadow params into.
            scheduler: LR scheduler to load state into.
            step: Specific step to load. If None, loads latest.
            restore_rng: Restore Python, NumPy and PyTorch RNG state.
            allow_partial_resume: Continue when adapter/EMA state is incomplete.
            conditioning_module: Expanded ``conv_in`` module to restore.
            conditioning_config: Expected source-conditioning contract.

        Returns:
            Metadata dict with step, epoch, loss, lr.
        """
        if step is not None:
            ckpt_dir = self.output_dir / f"checkpoint-{step}"
        else:
            ckpt_dir = self._latest_checkpoint()

        if ckpt_dir is None or not ckpt_dir.exists():
            raise FileNotFoundError(f"No checkpoint found in {self.output_dir}")

        # Load LoRA weights directly from safetensors into the existing PEFT model.
        # Avoids PeftModel.from_pretrained which creates a second adapter and
        # causes double-wrapping that breaks the computation graph.
        try:
            is_peft_model = hasattr(peft_model, "peft_config")
            if is_peft_model:
                from peft import PeftModel
            if is_peft_model and isinstance(peft_model, PeftModel):
                from safetensors.torch import load_file
                weights_file = ckpt_dir / "adapter_model.safetensors"
                if weights_file.exists():
                    raw_state_dict = load_file(str(weights_file))
                    # Remap checkpoint keys to match the live PEFT model's key
                    # naming: checkpoint uses `.lora_A.weight` but the live model
                    # uses `.lora_A.default.weight` (adapter name "default").
                    state_dict = {}
                    for k, v in raw_state_dict.items():
                        new_k = k
                        if k.endswith(".lora_A.weight"):
                            new_k = k.replace(".lora_A.weight", ".lora_A.default.weight")
                        elif k.endswith(".lora_B.weight"):
                            new_k = k.replace(".lora_B.weight", ".lora_B.default.weight")
                        state_dict[new_k] = v
                    live_keys = set(peft_model.state_dict())
                    matched_adapter_keys = {
                        key for key in state_dict
                        if key in live_keys and (".lora_A." in key or ".lora_B." in key)
                    }
                    if not matched_adapter_keys:
                        raise RuntimeError("checkpoint contains no live LoRA adapter keys")
                    missing, unexpected = peft_model.load_state_dict(state_dict, strict=False)
                    missing_adapter = [
                        key for key in missing
                        if ".lora_A." in key or ".lora_B." in key
                    ]
                    if missing_adapter:
                        message = f"Missing LoRA keys during resume: {len(missing_adapter)}"
                        if not allow_partial_resume:
                            raise RuntimeError(message)
                        logger.warning(message)
                    if unexpected:
                        logger.warning(f"Unexpected keys during LoRA resume: {len(unexpected)} keys")
                    logger.info(f"LoRA weights loaded from {ckpt_dir}")
                else:
                    raise FileNotFoundError(f"adapter_model.safetensors not found in {ckpt_dir}")
        except ImportError as e:
            if hasattr(peft_model, "peft_config") and not allow_partial_resume:
                raise RuntimeError(f"PEFT load failed: {e}") from e
            logger.debug("PEFT is unavailable; using generic checkpoint loading")
        except Exception as e:
            if not allow_partial_resume:
                raise RuntimeError(f"PEFT load failed: {e}") from e
            logger.warning(f"PEFT load failed (partial resume allowed): {e}")

        self._load_conditioning(
            ckpt_dir,
            conditioning_module=conditioning_module,
            expected_config=conditioning_config,
            spatial_adapter=spatial_adapter,
            allow_partial_resume=allow_partial_resume,
        )

        # Load EMA shadow params with key/shape validation (P1-16)
        if ema is not None:
            ema_path = ckpt_dir / "ema_weights.pt"
            if ema_path.exists():
                loaded_ema = torch.load(ema_path, weights_only=True)
                self._apply_ema_shadow(ema, loaded_ema, ckpt_dir)
                logger.info(f"EMA weights loaded from {ema_path}")
            elif not allow_partial_resume:
                raise FileNotFoundError(f"ema_weights.pt not found in {ckpt_dir}")

        # Load optimizer state
        if optimizer is not None:
            opt_path = ckpt_dir / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(opt_path, weights_only=True)
                )
                logger.info(f"Optimizer state loaded from {opt_path}")

        # Load scheduler state
        if scheduler is not None:
            sched_path = ckpt_dir / "scheduler.pt"
            if sched_path.exists():
                scheduler.load_state_dict(
                    torch.load(sched_path, weights_only=True)
                )

        if restore_rng:
            rng_path = ckpt_dir / "rng_state.pt"
            if not rng_path.exists():
                logger.warning(f"rng_state.pt not found in {ckpt_dir} — skipping RNG restore")
            else:
                try:
                    rng_state = torch.load(rng_path, weights_only=True)
                except (pickle.UnpicklingError, RuntimeError) as exc:
                    # Legacy checkpoints contain Python/NumPy objects. Do not
                    # re-enable unsafe deserialization just to restore RNG.
                    logger.warning(
                        "RNG state in %s is not weights-only loadable; "
                        "skipping legacy RNG restore: %s",
                        rng_path,
                        exc,
                    )
                else:
                    self._restore_rng_state(rng_state)

        # Load metadata
        metadata_path = ckpt_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"step": 0, "epoch": 0, "loss": 0.0, "lr": 0.0}

        logger.info(f"Checkpoint loaded from {ckpt_dir}")
        return metadata

    @staticmethod
    def _save_diffusers_lora_weights(
        peft_model,
        ckpt_dir: Path,
        *,
        component: str,
    ) -> Path:
        """Write a Diffusers-loadable companion to the PEFT resume adapter.

        PEFT wraps keys in ``base_model.model`` while pipeline loaders expect
        component-prefixed keys such as ``unet.down_blocks...lora_A.weight``.
        The PEFT adapter remains the source of truth for resume; this companion
        enables ``pipe.load_lora_weights`` and dynamic adapter scaling.
        """
        if component not in {"unet", "transformer"}:
            raise ValueError(f"Unsupported Diffusers LoRA component: {component!r}")

        from peft import get_peft_model_state_dict
        from safetensors.torch import save_file

        model = getattr(peft_model, "module", peft_model)
        from ..inference.checkpoint import peft_adapter_state_to_diffusers

        packed = peft_adapter_state_to_diffusers(
            get_peft_model_state_dict(model),
            component,
        )
        packed = {
            key: value.detach().to(device="cpu").contiguous()
            for key, value in packed.items()
        }

        configs = getattr(model, "peft_config", {})
        adapter_config = next(iter(configs.values()), None)
        metadata_payload = {}
        if adapter_config is not None:
            metadata_payload = {
                f"{component}.{key}": (
                    sorted(value) if isinstance(value, set) else value
                )
                for key, value in adapter_config.to_dict().items()
            }
        metadata = {"format": "pt"}
        if metadata_payload:
            metadata["lora_adapter_metadata"] = json.dumps(
                metadata_payload,
                indent=2,
                sort_keys=True,
            )

        path = ckpt_dir / DIFFUSERS_LORA_WEIGHT_NAME
        save_file(packed, path, metadata=metadata)
        logger.info(
            "Diffusers LoRA companion saved to %s (%d tensors)",
            path,
            len(packed),
        )
        return path

    @staticmethod
    def _save_conditioning(
        ckpt_dir: Path,
        *,
        conditioning_module,
        conditioning_config: SourceConditioningConfig | None,
        spatial_adapter,
    ) -> None:
        """Persist the expanded input convolution and its inference contract."""
        if conditioning_module is None and conditioning_config is None:
            return
        if conditioning_module is None or conditioning_config is None:
            raise ValueError(
                "conditioning_module and conditioning_config must be provided together"
            )
        conditioning_config.validate()
        if getattr(conditioning_module, "in_channels", None) != conditioning_config.in_channels:
            raise ValueError("conditioning module channels do not match conditioning config")
        expects_adapter = conditioning_config.spatial_adapter_type != "none"
        if expects_adapter != (spatial_adapter is not None):
            raise ValueError(
                "spatial adapter module does not match conditioning config"
            )
        torch.save(conditioning_module.state_dict(), ckpt_dir / CONV_IN_STATE_NAME)
        if spatial_adapter is not None:
            torch.save(
                spatial_adapter.state_dict(),
                ckpt_dir / SPATIAL_ADAPTER_STATE_NAME,
            )
        write_source_conditioning_config(ckpt_dir, conditioning_config)

    @staticmethod
    def _load_conditioning(
        ckpt_dir: Path,
        *,
        conditioning_module,
        expected_config: SourceConditioningConfig | None,
        spatial_adapter,
        allow_partial_resume: bool,
    ) -> SourceConditioningConfig | None:
        """Restore and validate an optional source-conditioned input convolution."""
        loaded_config = read_source_conditioning_config(ckpt_dir)
        if loaded_config is None:
            if expected_config is not None and not allow_partial_resume:
                raise FileNotFoundError(
                    f"conditioning_config.json not found in source-conditioned checkpoint {ckpt_dir}"
                )
            return None
        if expected_config is not None and loaded_config != expected_config:
            if not loaded_config.is_architecture_compatible(expected_config):
                raise ValueError(
                    "Checkpoint conditioning architecture does not match the live model: "
                    f"checkpoint={loaded_config.to_dict()}, "
                    f"live={expected_config.to_dict()}"
                )
            logger.warning(
                "Resuming source-conditioned architecture with updated training options: "
                "checkpoint=%s, live=%s",
                loaded_config.to_dict(),
                expected_config.to_dict(),
            )
        if conditioning_module is None:
            if allow_partial_resume:
                logger.warning(
                    "Checkpoint %s is source-conditioned but no conditioning module was supplied",
                    ckpt_dir,
                )
                return loaded_config
            raise ValueError(
                "Checkpoint is source-conditioned; supply the expanded conditioning module"
            )
        state_path = ckpt_dir / CONV_IN_STATE_NAME
        if not state_path.is_file():
            if allow_partial_resume:
                logger.warning("%s not found; source conv was not restored", state_path)
                return loaded_config
            raise FileNotFoundError(f"{CONV_IN_STATE_NAME} not found in {ckpt_dir}")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        conditioning_module.load_state_dict(state, strict=True)
        logger.info("Source-conditioned conv_in loaded from %s", state_path)
        if loaded_config.spatial_adapter_type != "none":
            if spatial_adapter is None:
                if allow_partial_resume:
                    logger.warning(
                        "Checkpoint %s declares a spatial adapter but none was supplied",
                        ckpt_dir,
                    )
                    return loaded_config
                raise ValueError(
                    "Checkpoint declares a spatial adapter; supply the live adapter module"
                )
            adapter_path = ckpt_dir / SPATIAL_ADAPTER_STATE_NAME
            if not adapter_path.is_file():
                if allow_partial_resume:
                    logger.warning("%s not found; spatial adapter was not restored", adapter_path)
                    return loaded_config
                raise FileNotFoundError(
                    f"{SPATIAL_ADAPTER_STATE_NAME} not found in {ckpt_dir}"
                )
            adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
            spatial_adapter.load_state_dict(adapter_state, strict=True)
            logger.info("Source spatial adapter loaded from %s", adapter_path)
        elif spatial_adapter is not None:
            raise ValueError("Live spatial adapter supplied for a checkpoint without one")
        return loaded_config

    @staticmethod
    def _apply_ema_shadow(ema, loaded: dict, ckpt_dir: Path) -> None:
        """Merge EMA shadow params with validation warnings for mismatched keys."""
        if not isinstance(loaded, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in loaded.items()
        ):
            raise TypeError(f"Invalid EMA payload in {ckpt_dir}")
        if not ema.shadow_params:
            ema.shadow_params = loaded
            return

        expected = set(ema.shadow_params.keys())
        loaded_keys = set(loaded.keys())
        missing = expected - loaded_keys
        unexpected = loaded_keys - expected
        if missing:
            logger.warning(
                "EMA checkpoint %s missing %d keys (first 5: %s)",
                ckpt_dir,
                len(missing),
                sorted(missing)[:5],
            )
        if unexpected:
            logger.warning(
                "EMA checkpoint %s has %d unexpected keys (first 5: %s)",
                ckpt_dir,
                len(unexpected),
                sorted(unexpected)[:5],
            )
        shape_mismatch = []
        merged = dict(ema.shadow_params)
        for name, tensor in loaded.items():
            if name not in merged:
                continue
            if merged[name].shape != tensor.shape:
                shape_mismatch.append(name)
                continue
            merged[name] = tensor
        if shape_mismatch:
            logger.warning(
                "EMA checkpoint %s shape mismatch on %d keys (first 5: %s)",
                ckpt_dir,
                len(shape_mismatch),
                shape_mismatch[:5],
            )
        ema.shadow_params = merged

    @staticmethod
    def _capture_rng_state() -> dict:
        """Capture RNGs in a ``weights_only``-safe tensor/scalar payload."""
        python_state = random.getstate()
        state = {
            "python": {
                "version": int(python_state[0]),
                "state": torch.tensor(python_state[1], dtype=torch.int64),
                "gauss_next": python_state[2],
            },
            "torch": torch.get_rng_state(),
        }
        try:
            import numpy as np

            numpy_state = np.random.get_state()
            state["numpy"] = {
                "algorithm": str(numpy_state[0]),
                "state": torch.from_numpy(numpy_state[1].copy()),
                "position": int(numpy_state[2]),
                "has_gauss": int(numpy_state[3]),
                "cached_gaussian": float(numpy_state[4]),
            }
        except ImportError:
            pass
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: dict) -> None:
        """Restore a safe RNG payload produced by ``_capture_rng_state``."""
        python_state = state["python"]
        if isinstance(python_state, dict):
            random.setstate(
                (
                    int(python_state["version"]),
                    tuple(int(value) for value in python_state["state"].tolist()),
                    python_state.get("gauss_next"),
                )
            )
        else:
            # Keep direct compatibility for callers holding an in-memory
            # legacy state; checkpoint files use the safe dict format above.
            random.setstate(python_state)
        torch.set_rng_state(state["torch"])
        if "numpy" in state:
            import numpy as np

            numpy_state = state["numpy"]
            if isinstance(numpy_state, dict):
                np.random.set_state(
                    (
                        str(numpy_state["algorithm"]),
                        numpy_state["state"].cpu().numpy().astype("uint32", copy=False),
                        int(numpy_state["position"]),
                        int(numpy_state["has_gauss"]),
                        float(numpy_state["cached_gaussian"]),
                    )
                )
            else:
                np.random.set_state(numpy_state)
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])

    @staticmethod
    def _ckpt_step(path: Path) -> int:
        """Extract numeric step from checkpoint directory name (e.g. 'checkpoint-1000' → 1000)."""
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    def _latest_checkpoint(self) -> Path | None:
        """Find the latest numbered checkpoint directory."""
        checkpoints = self._numbered_checkpoints(self.output_dir)
        if not checkpoints:
            return None
        return checkpoints[-1]

    @staticmethod
    def _numbered_checkpoints(output_dir: Path | None = None) -> list[Path]:
        """List checkpoint-* dirs with numeric step suffix (excludes checkpoint-best)."""
        root = output_dir if output_dir is not None else Path(".")
        result = []
        for path in root.glob("checkpoint-*"):
            suffix = path.name.split("-")[-1]
            if suffix.isdigit():
                result.append(path)
        return sorted(result, key=CheckpointManager._ckpt_step)

    def _prune_old_checkpoints(self) -> None:
        """Remove oldest numbered checkpoints beyond max_keep."""
        checkpoints = self._numbered_checkpoints(self.output_dir)
        while len(checkpoints) > self.max_keep:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest, ignore_errors=True)
            logger.info(f"Pruned old checkpoint: {oldest}")


__all__ = ["CheckpointManager", "DIFFUSERS_LORA_WEIGHT_NAME"]
