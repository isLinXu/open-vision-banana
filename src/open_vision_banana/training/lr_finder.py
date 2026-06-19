"""Learning rate finder utility for Vision Banana training.

Implements the "LR range test" from Smith (2015): gradually increase the
learning rate from a tiny value and log the loss, then plot loss-vs-LR to
identify the range where loss decreases most rapidly.

Usage::

    from open_vision_banana.training.lr_finder import LRFinder

    finder = LRFinder(model, optimizer, criterion, device="cpu")
    finder.range_test(dataloader, start_lr=1e-7, end_lr=1.0, num_iter=100)
    finder.plot()  # or finder.suggest_lr()
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LRFinder:
    """Learning rate range test (Smith, 2015).

    Smoothly increases the learning rate over a fixed number of iterations
    and records the loss at each step. The resulting loss-vs-LR curve helps
    identify a good learning rate range before starting full training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[..., torch.Tensor],
        device: str = "cpu",
        smooth_factor: float = 0.05,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.smooth_factor = smooth_factor

        self.lrs: list[float] = []
        self.losses: list[float] = []

        # Save original state for restore
        self._model_state = None
        self._optimizer_state = None

    def range_test(
        self,
        dataloader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 1.0,
        num_iter: int = 100,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """Run the LR range test.

        Gradually increases LR from ``start_lr`` to ``end_lr`` over
        ``num_iter`` iterations, recording loss at each step.

        Args:
            dataloader: Training data loader.
            start_lr: Starting learning rate (typically very small).
            end_lr: Ending learning rate (typically large).
            num_iter: Number of iterations to run.
            gradient_accumulation_steps: Gradient accumulation steps.
        """
        # Save state for restoration
        self._model_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        self._optimizer_state = self.optimizer.state_dict()

        # Compute LR multiplier (exponential schedule)
        lr_multiplier = (end_lr / start_lr) ** (1.0 / max(num_iter - 1, 1))

        # Set initial LR
        for group in self.optimizer.param_groups:
            group["lr"] = start_lr

        self.model.train()
        self.lrs = []
        self.losses = []

        iter_count = 0
        data_iter = iter(dataloader)
        best_loss = float("inf")

        while iter_count < num_iter:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
            elif isinstance(batch, (list, tuple)):
                batch = [v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for v in batch]

            # Forward + backward
            self.optimizer.zero_grad()

            if isinstance(batch, dict):
                # Dict-style batch: extract image as model input, target_rgb as target
                image_key = "image" if "image" in batch else "input"
                input_tensor = batch.get(image_key, next(
                    v for v in batch.values() if isinstance(v, torch.Tensor)
                ))
                output = self.model(input_tensor)
                target = batch.get("target_rgb", batch.get("target", output))
            else:
                output = self.model(batch[0])
                target = batch[1] if len(batch) > 1 else output

            loss = self.criterion(output, target)
            loss.backward()

            if (iter_count + 1) % gradient_accumulation_steps == 0:
                self.optimizer.step()

            # Record current LR and loss
            current_lr = self.optimizer.param_groups[0]["lr"]
            loss_val = loss.item()

            # Smooth loss (exponential moving average)
            if len(self.losses) == 0:
                smoothed = loss_val
            else:
                smoothed = self.smooth_factor * loss_val + \
                           (1 - self.smooth_factor) * self.losses[-1]

            self.lrs.append(current_lr)
            self.losses.append(smoothed)

            # Stop if loss diverges
            if smoothed < best_loss:
                best_loss = smoothed
            if smoothed > 4 * best_loss:
                logger.info(f"Loss diverged at LR={current_lr:.2e}, stopping early.")
                break

            # Increase LR
            for group in self.optimizer.param_groups:
                group["lr"] *= lr_multiplier

            iter_count += 1

        logger.info(f"LR range test completed: {len(self.lrs)} iterations, "
                     f"LR range [{self.lrs[0]:.2e}, {self.lrs[-1]:.2e}]")

    def suggest_lr(self) -> float:
        """Suggest a learning rate based on the steepest descent point.

        The suggested LR is the point where the loss decreases most rapidly,
        which is typically 1-2 orders of magnitude below the divergence point.

        Returns:
            Suggested learning rate.
        """
        if len(self.losses) < 3:
            return 1e-4  # Default fallback

        # Compute loss gradients (differences)
        loss_grads = []
        for i in range(1, len(self.losses)):
            grad = (self.losses[i] - self.losses[i - 1]) / \
                   (self.lrs[i] - self.lrs[i - 1] + 1e-12)
            loss_grads.append(grad)

        # Find the steepest negative gradient (max decrease)
        min_grad_idx = 0
        min_grad = 0.0
        for i, g in enumerate(loss_grads):
            if g < min_grad:
                min_grad = g
                min_grad_idx = i

        # Use the LR at the steepest descent point
        suggested_lr = self.lrs[min_grad_idx + 1]

        # Clamp to a reasonable range
        suggested_lr = max(1e-7, min(suggested_lr, 1e-1))

        logger.info(f"Suggested LR: {suggested_lr:.2e} "
                     f"(steepest descent at iteration {min_grad_idx + 1})")
        return suggested_lr

    def plot(self, save_path: str | None = None) -> None:
        """Plot the loss-vs-LR curve.

        Args:
            save_path: If provided, save the plot to this path instead of showing.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; cannot plot. "
                           "Use suggest_lr() for a programmatic suggestion.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(self.lrs, self.losses)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title("LR Range Test")

        # Mark suggested LR
        suggested = self.suggest_lr()
        ax.axvline(x=suggested, color="r", linestyle="--",
                   label=f"Suggested LR: {suggested:.2e}")
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"LR finder plot saved to {save_path}")
        else:
            plt.show()

    def restore(self) -> None:
        """Restore model and optimizer to their pre-test state."""
        if self._model_state is not None:
            self.model.load_state_dict(self._model_state)
        if self._optimizer_state is not None:
            self.optimizer.load_state_dict(self._optimizer_state)
        logger.info("Model and optimizer restored to pre-LR-test state.")


__all__ = ["LRFinder"]
