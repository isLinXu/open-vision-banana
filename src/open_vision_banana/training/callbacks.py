"""Training callbacks: W&B logging, checkpointing, and validation tracking.

Decouples logging/checkpointing from the training loop so that new
backends (TensorBoard, MLflow, etc.) can be added without modifying
the Trainer class.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class TrainingCallbacks:
    """Handles logging, checkpointing, and validation."""

    def __init__(
        self,
        output_dir: str | Path,
        wandb_enabled: bool = False,
        wandb_project: str = "open-vision-banana",
        wandb_run_name: str | None = None,
        wandb_config: dict | None = None,
        log_every: int = 25,
        save_every: int = 500,
        eval_every: int = 250,
        task_prompts: dict[str, str] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_enabled = wandb_enabled
        self.log_every = log_every
        self.save_every = save_every
        self.eval_every = eval_every
        self.best_val_loss = float("inf")
        # Default instruction per task, stored with every checkpoint so inference
        # can prompt the LoRA the way it was trained.
        self.task_prompts = dict(task_prompts or {})

        # W&B initialization
        if wandb_enabled:
            import random

            import wandb

            run_name = wandb_run_name or f"train-{random.randint(1000, 9999)}"
            wandb.init(
                project=wandb_project,
                name=run_name,
                config=wandb_config or {},
                dir=str(self.output_dir),
            )
            logger.info("W&B initialized: %s/%s", wandb_project, run_name)

    def should_log(self, step: int) -> bool:
        return step > 0 and step % self.log_every == 0

    def should_checkpoint(self, step: int) -> bool:
        return step > 0 and step % self.save_every == 0

    def should_validate(self, step: int) -> bool:
        return step > 0 and step % self.eval_every == 0

    def on_log(
        self,
        step: int,
        total_steps: int,
        metrics: dict[str, float],
        lr: float,
        epoch: int,
        task_loss_acc: dict[str, list[float]] | None = None,
    ) -> None:
        """Log training metrics."""
        parts = [f"[step {step}/{total_steps}]"]
        for key, value in sorted(metrics.items()):
            parts.append(f"{key}={value:.4f}")
        parts.append(f"lr={lr:.2e}")
        parts.append(f"epoch={epoch}")

        if task_loss_acc:
            task_str = " ".join(
                f"{t}={sum(vals) / len(vals):.4f}" for t, vals in sorted(task_loss_acc.items())
            )
            parts.append(task_str)

        logger.info("  " + " ".join(parts))

        if self.wandb_enabled:
            import wandb

            log_dict = {f"train/{k}": v for k, v in metrics.items()}
            log_dict["train/lr"] = lr
            log_dict["train/epoch"] = epoch
            wandb.log(log_dict, step=step)

            if task_loss_acc:
                for t, vals in sorted(task_loss_acc.items()):
                    wandb.log({f"train/task_{t}_loss": sum(vals) / len(vals)}, step=step)

    def on_checkpoint(
        self,
        step: int,
        peft_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        val_loss: float | None = None,
        ema: Any | None = None,
        model_name: str = "",
        lora_rank: int = 32,
        lora_alpha: int = 64,
        suffix: str = "",
    ) -> Path:
        """Save a training checkpoint."""
        ckpt_dir = self.output_dir / f"checkpoint-{step}{suffix}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        lora_state = {
            n: p.data.cpu().clone() for n, p in peft_model.named_parameters() if p.requires_grad
        }
        ckpt = {
            "lora_state_dict": lora_state,
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        if ema is not None and getattr(ema, "shadow_params", None):
            ckpt["ema_state_dict"] = {n: t.cpu().clone() for n, t in ema.shadow_params.items()}

        torch.save(ckpt, str(ckpt_dir / "lora.pt"))

        # Save LoRA adapter in PEFT format
        try:
            peft_model.save_pretrained(str(ckpt_dir / "lora_adapter"))
        except Exception as e:
            logger.warning("Could not save LoRA adapter: %s", e)

        # Save metadata
        meta = {
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "model_name": model_name,
        }
        if self.task_prompts:
            meta["task_prompts"] = self.task_prompts
        (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info("  Saved checkpoint at step %d → %s", step, ckpt_dir)
        return ckpt_dir

    def on_validate(
        self,
        step: int,
        val_loss: float,
        peft_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        ema: Any | None = None,
        model_name: str = "",
        lora_rank: int = 32,
        lora_alpha: int = 64,
    ) -> bool:
        """Handle validation results. Returns True if new best."""
        logger.info("  [val] step=%d val_loss=%.4f", step, val_loss)

        if self.wandb_enabled:
            import wandb

            wandb.log({"val/loss": val_loss}, step=step)

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.on_checkpoint(
                step,
                peft_model,
                optimizer,
                scheduler,
                val_loss=val_loss,
                ema=ema,
                model_name=model_name,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                suffix="_best",
            )
            logger.info("  New best val_loss=%.4f", val_loss)

        return is_best

    def on_finish(self, step: int) -> None:
        """Called when training completes."""
        if self.wandb_enabled:
            import wandb

            wandb.finish()
        logger.info("Training complete! Steps=%d, Best val_loss=%.4f", step, self.best_val_loss)
