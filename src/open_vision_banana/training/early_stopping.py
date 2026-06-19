"""Early stopping utility for validation loss monitoring."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of evaluations with no improvement before stopping.
        min_delta: Minimum decrease in val loss to count as improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError(f"patience must be >= 1; got {patience}")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Record a validation loss and update stop state.

        Returns:
            True if this step is a new best validation loss.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            logger.info("EarlyStopping: new best val_loss=%.6f", val_loss)
            return True

        self.counter += 1
        logger.info(
            "EarlyStopping: no improvement (%d/%d), best=%.6f, current=%.6f",
            self.counter,
            self.patience,
            self.best_loss,
            val_loss,
        )
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info("EarlyStopping: stopping training (patience exhausted)")
        return False

    def reset(self) -> None:
        """Reset internal state."""
        self.best_loss = None
        self.counter = 0
        self.should_stop = False


__all__ = ["EarlyStopping"]
