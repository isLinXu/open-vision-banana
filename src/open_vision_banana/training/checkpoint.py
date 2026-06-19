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
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


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
    ) -> Path:
        """Save a training checkpoint."""
        ckpt_dir = self.output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        peft_model.save_pretrained(ckpt_dir)

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

        # Save metadata
        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

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
    ) -> Path:
        """Save the best validation checkpoint to ``checkpoint-best/``."""
        ckpt_dir = self.output_dir / "checkpoint-best"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        peft_model.save_pretrained(ckpt_dir)

        if ema is not None:
            torch.save(ema.shadow_params, ckpt_dir / "ema_weights.pt")

        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

        if scheduler is not None:
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        metadata = {
            "step": step,
            "epoch": epoch,
            "loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "best": True,
        }
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Best checkpoint saved to %s (val_loss=%.6f)", ckpt_dir, val_loss)
        return ckpt_dir

    def load(
        self,
        peft_model,
        optimizer: torch.optim.Optimizer | None = None,
        ema=None,
        scheduler=None,
        step: int | None = None,
    ) -> dict:
        """Load a training checkpoint.

        Args:
            peft_model: PEFT model to load LoRA weights into.
            optimizer: Optimizer to load state into.
            ema: EMA model to load shadow params into.
            scheduler: LR scheduler to load state into.
            step: Specific step to load. If None, loads latest.

        Returns:
            Metadata dict with step, epoch, loss, lr.
        """
        if step is not None:
            ckpt_dir = self.output_dir / f"checkpoint-{step}"
        else:
            ckpt_dir = self._latest_checkpoint()

        if ckpt_dir is None or not ckpt_dir.exists():
            raise FileNotFoundError(f"No checkpoint found in {self.output_dir}")

        # Load LoRA weights via PEFT
        # PeftModel.from_pretrained returns a *new* model; we copy the adapter
        # weights back into the existing peft_model to preserve its reference.
        try:
            from peft import PeftModel
            if isinstance(peft_model, PeftModel):
                loaded = PeftModel.from_pretrained(
                    peft_model.base_model.model,
                    str(ckpt_dir),
                )
                # Copy adapter state dict in-place
                peft_model.load_state_dict(loaded.state_dict(), strict=False)
                del loaded
                logger.info(f"LoRA weights loaded from {ckpt_dir}")
        except Exception as e:
            logger.warning(f"PEFT load failed (non-fatal): {e}")

        # Load EMA shadow params
        # Use weights_only=False for EMA because shadow_params is a plain dict
        # of Tensors, which is safe but not accepted by weights_only=True.
        if ema is not None:
            ema_path = ckpt_dir / "ema_weights.pt"
            if ema_path.exists():
                ema.shadow_params = torch.load(ema_path, weights_only=False)
                logger.info(f"EMA weights loaded from {ema_path}")

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


__all__ = ["CheckpointManager"]
