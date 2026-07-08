"""Training logging integrations (Weights & Biases and TensorBoard)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Unified logger for training metrics.

    Supports optional Weights & Biases and TensorBoard backends.
    Both are no-ops when disabled or when dependencies are missing.
    """

    def __init__(
        self,
        *,
        wandb_enabled: bool = False,
        wandb_project: str = "open-vision-banana",
        wandb_run_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_log_checkpoints: bool = False,
        tensorboard_enabled: bool = False,
        tensorboard_dir: str = "output/runs",
        run_name: str | None = None,
    ) -> None:
        self.wandb_enabled = wandb_enabled
        self.wandb_log_checkpoints = wandb_log_checkpoints and wandb_enabled
        self.tensorboard_enabled = tensorboard_enabled
        self._wandb = None
        self._writer = None

        run_label = run_name or wandb_run_name or "vision-banana-run"

        if wandb_enabled:
            try:
                import wandb

                self._wandb = wandb.init(  # type: ignore[attr-defined]
                    project=wandb_project,
                    name=run_label,
                    entity=wandb_entity,
                    reinit=True,
                )
                logger.info("W&B run started: %s", run_label)
            except ImportError:
                logger.warning("wandb not installed — disabling W&B logging")
                self.wandb_enabled = False
            except Exception as exc:
                logger.warning("Failed to init wandb: %s", exc)
                self.wandb_enabled = False

        if tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_path = Path(tensorboard_dir) / run_label
                tb_path.mkdir(parents=True, exist_ok=True)
                self._writer = SummaryWriter(log_dir=str(tb_path))
                logger.info("TensorBoard logging to %s", tb_path)
            except ImportError:
                logger.warning("tensorboard not available — disabling TB logging")
                self.tensorboard_enabled = False
            except Exception as exc:
                logger.warning("Failed to init TensorBoard: %s", exc)
                self.tensorboard_enabled = False

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        """Log scalar metrics at a training step."""
        if self.wandb_enabled and self._wandb is not None:
            try:
                import wandb

                wandb.log(metrics, step=step)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Failed to log metrics to W&B", exc_info=True)

        if self.tensorboard_enabled and self._writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, step)

    def log_hyperparams(self, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
        """Log hyperparameters (TensorBoard hparams panel + W&B config)."""
        if self.wandb_enabled and self._wandb is not None:
            try:
                import wandb

                wandb.config.update(hparams)  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Failed to log hyperparams to W&B", exc_info=True)

        if self.tensorboard_enabled and self._writer is not None:
            flat_metrics = metrics or {}
            try:
                self._writer.add_hparams(
                    {k: str(v) for k, v in hparams.items()},
                    {k: float(v) for k, v in flat_metrics.items()},
                )
            except Exception:
                # add_hparams can fail if called after scalars on some versions
                logger.debug("Skipped TensorBoard add_hparams", exc_info=True)

    def log_checkpoint(
        self,
        checkpoint_dir: str | Path,
        alias: str = "checkpoint",
        metadata: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> None:
        """Upload a checkpoint directory as a W&B artifact (when enabled)."""
        if not self.wandb_log_checkpoints or self._wandb is None:
            return

        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.is_dir():
            logger.warning("Skipping W&B artifact upload — not a directory: %s", ckpt_path)
            return

        try:
            import wandb

            safe_alias = alias.replace("/", "-").replace(" ", "_")
            artifact = wandb.Artifact(  # type: ignore[attr-defined]
                name=f"vision-banana-lora-{safe_alias}",
                type="model",
                metadata=metadata or {},
            )
            artifact.add_dir(str(ckpt_path))
            logged = self._wandb.log_artifact(artifact)
            if step is not None:
                wandb.log({"checkpoint/uploaded": 1}, step=step)  # type: ignore[attr-defined]
            logger.info("Uploaded W&B checkpoint artifact: %s", logged.name)
        except Exception as exc:
            logger.warning("Failed to upload W&B checkpoint artifact: %s", exc)

    def finish(self) -> None:
        """Close all logging backends."""
        if self._wandb is not None:
            import wandb

            wandb.finish()  # type: ignore[attr-defined]
            self._wandb = None

        if self._writer is not None:
            self._writer.close()
            self._writer = None


def create_training_logger_from_config(cfg) -> TrainingLogger:
    """Build a TrainingLogger from VisionBananaConfig.logging."""
    logging_cfg = getattr(cfg, "logging", None)
    if logging_cfg is None:
        return TrainingLogger()

    return TrainingLogger(
        wandb_enabled=getattr(logging_cfg, "wandb_enabled", False),
        wandb_project=getattr(logging_cfg, "wandb_project", "open-vision-banana"),
        wandb_run_name=getattr(logging_cfg, "wandb_run_name", None),
        wandb_entity=getattr(logging_cfg, "wandb_entity", None),
        wandb_log_checkpoints=getattr(logging_cfg, "wandb_log_checkpoints", False),
        tensorboard_enabled=getattr(logging_cfg, "tensorboard_enabled", False),
        tensorboard_dir=getattr(logging_cfg, "tensorboard_dir", "output/runs"),
        run_name=getattr(logging_cfg, "run_name", None),
    )


__all__ = ["TrainingLogger", "create_training_logger_from_config"]
