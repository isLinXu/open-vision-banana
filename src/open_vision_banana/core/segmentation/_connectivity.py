"""Shared connectivity constants for instance segmentation grouping."""

from __future__ import annotations

# 16-connectivity: orthogonal + diagonal + knight's move (paper Appendix A)
CONNECTIVITY_SHIFTS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
    (-2, -1), (-2, 1), (2, -1), (2, 1),
    (-1, -2), (-1, 2), (1, -2), (1, 2),
]


def build_valid_mask(h: int, w: int, dy: int, dx: int, device) -> "torch.Tensor":
    """Border mask for a shift offset on an H×W grid."""
    import torch

    valid = torch.ones(h, w, dtype=torch.bool, device=device)
    if dy == -1:
        valid[0, :] = False
    elif dy == -2:
        valid[0:2, :] = False
    elif dy == 1:
        valid[-1, :] = False
    elif dy == 2:
        valid[-2:, :] = False
    if dx == -1:
        valid[:, 0] = False
    elif dx == -2:
        valid[:, 0:2] = False
    elif dx == 1:
        valid[:, -1] = False
    elif dx == 2:
        valid[:, -2:] = False
    return valid


def relabel_consecutive(labels: "torch.Tensor") -> "torch.Tensor":
    """Relabel component IDs to consecutive integers (background → 0)."""
    unique_labels, inverse = labels.reshape(-1).unique(return_inverse=True)
    is_bg = unique_labels < 0
    non_bg_cumsum = (~is_bg).cumsum(dim=0)
    new_id = non_bg_cumsum.long()
    new_id = torch.where(is_bg, torch.zeros_like(new_id), new_id)
    return new_id[inverse].reshape(labels.shape)


__all__ = ["CONNECTIVITY_SHIFTS", "build_valid_mask", "relabel_consecutive"]
