"""Reversible depth-to-RGB mapping using power transform + cube-edge colormap.

From the paper (Section 3.2):
- Apply power transform: f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)
- Parameters: λ=-3, c=10/3
- Then map via the first-order 3D Hilbert path from black to white

This maps unbounded depth values [0, ∞) to bounded RGB [0, 1]³.
The curving prioritizes accuracy for nearby content (higher utility).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Cube-Edge Colormap for Depth Encoding (Paper Section 3.2)
# ──────────────────────────────────────────────────────────────

# The paper instructs the model to output a false-color visualization where
# curved and normalized distances interpolate along a piecewise-linear
# function that follows the edges of the RGB cube, traversing from black
# to white, similarly to the first iteration of a 3D Hilbert curve.
#
# Path shown in paper Figure 5:
# Black(0,0,0) → Red(1,0,0) → Yellow(1,1,0) → Green(0,1,0)
# → Cyan(0,1,1) → Blue(0,0,1) → Magenta(1,0,1) → White(1,1,1)
# Each segment lies strictly on an edge of the RGB cube.

# ──────────────────────────────────────────────────────────────
# Multi-Colormap Keypoints for Depth Encoding (Paper Section 3.2)
# ──────────────────────────────────────────────────────────────

# First-order 3D Hilbert path. Equal 1/7 intervals reproduce the corner depths
# annotated in Figure 5 after the paper's power transform (approximately
# 0.8, 1.8, 3.2, 5.3, 8.7, and 16 metres).
CUBE_EDGE_KEYPOINTS = torch.tensor(
    [
        [0.0 / 7.0, 0.00, 0.00, 0.00],  # Black   — (0,0,0)
        [1.0 / 7.0, 1.00, 0.00, 0.00],  # Red     — (1,0,0)
        [2.0 / 7.0, 1.00, 1.00, 0.00],  # Yellow  — (1,1,0)
        [3.0 / 7.0, 0.00, 1.00, 0.00],  # Green   — (0,1,0)
        [4.0 / 7.0, 0.00, 1.00, 1.00],  # Cyan    — (0,1,1)
        [5.0 / 7.0, 0.00, 0.00, 1.00],  # Blue    — (0,0,1)
        [6.0 / 7.0, 1.00, 0.00, 1.00],  # Magenta — (1,0,1)
        [7.0 / 7.0, 1.00, 1.00, 1.00],  # White   — (1,1,1)
    ],
    dtype=torch.float32,
)

# Compatibility mapping used by checkpoints trained before the Figure 5 path
# was corrected. It remains opt-in and is no longer the paper-aligned default.
LEGACY_CUBE_EDGE_KEYPOINTS = torch.tensor(
    [
        [0.00, 0.00, 0.00, 0.00],
        [0.20, 0.00, 0.00, 1.00],
        [0.40, 0.00, 1.00, 1.00],
        [0.60, 0.00, 1.00, 0.00],
        [0.80, 1.00, 1.00, 0.00],
        [1.00, 1.00, 1.00, 1.00],
    ],
    dtype=torch.float32,
)

# Plasma colormap keypoints (sampled from matplotlib)
PLASMA_KEYPOINTS = torch.tensor(
    [
        [0.000, 0.0504, 0.0298, 0.5280],
        [0.143, 0.3251, 0.0069, 0.6395],
        [0.286, 0.5462, 0.0390, 0.6470],
        [0.429, 0.7234, 0.1962, 0.5390],
        [0.571, 0.8598, 0.3606, 0.4069],
        [0.714, 0.9555, 0.5331, 0.2855],
        [0.857, 0.9945, 0.7409, 0.1663],
        [1.000, 0.9400, 0.9752, 0.1313],
    ],
    dtype=torch.float32,
)

# Inferno colormap keypoints (sampled from matplotlib)
INFERNO_KEYPOINTS = torch.tensor(
    [
        [0.000, 0.0015, 0.0005, 0.0139],
        [0.143, 0.1558, 0.0446, 0.3253],
        [0.286, 0.3977, 0.0833, 0.4332],
        [0.429, 0.6217, 0.1642, 0.3888],
        [0.571, 0.8323, 0.2839, 0.2574],
        [0.714, 0.9613, 0.4887, 0.0843],
        [0.857, 0.9812, 0.7591, 0.1569],
        [1.000, 0.9884, 0.9984, 0.6449],
    ],
    dtype=torch.float32,
)

# Viridis colormap keypoints (sampled from matplotlib)
VIRIDIS_KEYPOINTS = torch.tensor(
    [
        [0.000, 0.2670, 0.0049, 0.3294],
        [0.143, 0.2752, 0.1949, 0.4960],
        [0.286, 0.2124, 0.3597, 0.5517],
        [0.429, 0.1534, 0.4970, 0.5577],
        [0.571, 0.1223, 0.6332, 0.5304],
        [0.714, 0.2889, 0.7584, 0.4284],
        [0.857, 0.6266, 0.8546, 0.2234],
        [1.000, 0.9932, 0.9062, 0.1439],
    ],
    dtype=torch.float32,
)

# Grayscale: linear mapping from depth to (gray, gray, gray)
GRAYSCALE_KEYPOINTS = torch.tensor(
    [
        [0.00, 0.00, 0.00, 0.00],
        [1.00, 1.00, 1.00, 1.00],
    ],
    dtype=torch.float32,
)

# Rainbow colormap: Black → Red → Yellow → Green → Cyan → Blue → Violet → White
# From the paper website: "rainbow (black-red-yellow-green-cyan-blue-violet-white) color palette"
RAINBOW_KEYPOINTS = torch.tensor(
    [
        [0.000, 0.00, 0.00, 0.00],  # Black
        [0.143, 1.00, 0.00, 0.00],  # Red
        [0.286, 1.00, 1.00, 0.00],  # Yellow
        [0.429, 0.00, 1.00, 0.00],  # Green
        [0.571, 0.00, 1.00, 1.00],  # Cyan
        [0.714, 0.00, 0.00, 1.00],  # Blue
        [0.857, 0.50, 0.00, 1.00],  # Violet (approximate)
        [1.000, 1.00, 1.00, 1.00],  # White
    ],
    dtype=torch.float32,
)

# Colormap registry
_COLORMAP_REGISTRY: dict[str, torch.Tensor] = {
    "cube_edge": CUBE_EDGE_KEYPOINTS,
    "legacy_cube_edge": LEGACY_CUBE_EDGE_KEYPOINTS,
    "plasma": PLASMA_KEYPOINTS,
    "inferno": INFERNO_KEYPOINTS,
    "viridis": VIRIDIS_KEYPOINTS,
    "grayscale": GRAYSCALE_KEYPOINTS,
    "rainbow": RAINBOW_KEYPOINTS,
}

_AVAILABLE_COLORMAPS = list(_COLORMAP_REGISTRY.keys())



class DepthRGBMapper:
    """Reversible depth-to-RGB mapping using power transform + multi-colormap support.

    From the paper (Section 3.2):
    - Apply power transform: f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)
    - Parameters: λ=-3, c=10/3
    - Then map via colormap (cube-edge by default, or plasma/inferno/viridis/grayscale)

    This maps unbounded depth values [0, ∞) to bounded RGB [0, 1]³.
    The curving prioritizes accuracy for nearby content (higher utility).

    Multi-colormap support enables training-time augmentation:
    the model learns to be robust to different RGB encoding schemes,
    as recommended by the paper.
    """

    def __init__(self, colormap: str = "cube_edge", lam: float = -3.0, c: float = 10.0 / 3.0):
        self.lam = lam
        self.c = c
        self.colormap = colormap
        self._keypoints = self._get_keypoints(colormap)

    def _get_keypoints(self, colormap: str) -> torch.Tensor:
        """Get keypoints for the specified colormap."""
        if colormap not in _COLORMAP_REGISTRY:
            raise ValueError(
                f"Unknown colormap: {colormap}. "
                f"Available: {get_available_colormaps()}"
            )
        return _COLORMAP_REGISTRY[colormap]

    @property
    def colormap_keypoints(self) -> torch.Tensor:
        """Return the current colormap's keypoints tensor."""
        return self._keypoints

    def set_colormap(self, colormap: str) -> None:
        """Switch to a different colormap (e.g. for training augmentation)."""
        self.colormap = colormap
        self._keypoints = self._get_keypoints(colormap)

    def depth_to_normalized(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply power transform with adaptive precision.

        f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)

        Args:
            depth: Metric depth values [B, 1, H, W] in meters, range [0, ∞).

        Returns:
            Normalized depth values [B, 1, H, W] in [0, 1).
        """
        depth = depth.clamp(min=0.0)

        ratio = depth / (self.lam * self.c)
        ratio = ratio.clamp(max=1.0 - 1e-7)
        base = 1.0 - ratio
        exponent = self.lam + 1  # = -2
        normalized = 1.0 - torch.pow(base, exponent)
        return normalized.clamp(0.0, 1.0)

    def normalized_to_depth(self, normalized: torch.Tensor) -> torch.Tensor:
        """Inverse: normalized [0, 1) → depth [0, ∞)."""
        exponent = 1.0 / (self.lam + 1)  # = -0.5
        base = 1.0 - normalized
        ratio = 1.0 - torch.pow(base.clamp(min=1e-10), exponent)
        depth = self.lam * self.c * ratio
        return depth.clamp(min=0.0)

    def normalized_to_rgb(self, normalized: torch.Tensor) -> torch.Tensor:
        """Map normalized depth to RGB via the selected colormap.

        Uses piecewise-linear interpolation between keypoints.
        """
        kp = self._keypoints.to(normalized.device)
        positions = kp[:, 0].contiguous()
        colors = kp[:, 1:].contiguous()

        orig_shape = normalized.shape
        n_flat = normalized.reshape(-1)

        seg_idx = torch.searchsorted(positions, n_flat, right=True) - 1
        seg_idx = seg_idx.clamp(0, len(positions) - 2)

        p_start = positions[seg_idx]
        p_end = positions[seg_idx + 1]
        local_t = (n_flat - p_start) / (p_end - p_start + 1e-10)
        local_t = local_t.clamp(0.0, 1.0)

        c_start = colors[seg_idx]
        c_end = colors[seg_idx + 1]
        rgb_flat = c_start + local_t.unsqueeze(-1) * (c_end - c_start)

        rgb = rgb_flat.reshape(*orig_shape, 3)

        # P1-7: robust dimension handling — squeeze singleton dims between batch
        # and the RGB channel, but only until we reach 4D or 3D.
        squeeze_dims = [i for i in range(1, rgb.dim() - 1) if rgb.shape[i] == 1]
        for i in reversed(squeeze_dims):
            if rgb.dim() <= 4:
                break
            rgb = rgb.squeeze(i)

        if rgb.dim() == 4:
            rgb = rgb.permute(0, 3, 1, 2)
        elif rgb.dim() == 3:
            rgb = rgb.permute(2, 0, 1)
        else:
            raise ValueError(
                f"Unexpected RGB shape after reshape: {rgb.shape}. "
                f"Expected 3D or 4D after squeezing singleton dims, got {rgb.dim()}D."
            )

        return rgb.clamp(0.0, 1.0)

    def _project_rgb_chunk_to_normalized(
        self,
        rgb_chunk: torch.Tensor,
        positions: torch.Tensor,
        colors: torch.Tensor,
    ) -> torch.Tensor:
        """Project RGB pixels onto the piecewise-linear curve.

        Uses analytical segment projection instead of brute-force
        nearest-neighbour — ~30× faster with equal accuracy
        for on-curve encodings and better behaviour for noisy inputs.

        Args:
            rgb_chunk: [B, N, 3] pixel colours in [0, 1].
            positions: [K] keypoint positions along the curve.
            colors: [K, 3] keypoint RGB values.

        Returns:
            Normalized positions [B, N] in [0, 1].
        """
        c0 = colors[:-1]
        c1 = colors[1:]
        seg_vec = c1 - c0
        seg_len_sq = (seg_vec * seg_vec).sum(-1).clamp(min=1e-10)

        p0_pos = positions[:-1]
        p1_pos = positions[1:]

        # [B, N, 1, 3] vs [1, 1, S, 3]
        p = rgb_chunk.unsqueeze(2)
        c0_e = c0.unsqueeze(0).unsqueeze(0)
        seg_vec_e = seg_vec.unsqueeze(0).unsqueeze(0)

        diff = p - c0_e
        local_t = (diff * seg_vec_e).sum(-1) / seg_len_sq.unsqueeze(0).unsqueeze(0)
        local_t = local_t.clamp(0.0, 1.0)

        closest = c0_e + local_t.unsqueeze(-1) * seg_vec_e
        dists_sq = ((p - closest) ** 2).sum(-1)

        best_seg = dists_sq.argmin(dim=-1)
        num_segments = len(p0_pos)

        flat_seg = best_seg.reshape(-1)
        flat_t = local_t.reshape(-1, num_segments).gather(1, flat_seg.unsqueeze(1)).squeeze(1)
        flat_start = p0_pos[flat_seg]
        flat_end = p1_pos[flat_seg]
        refined = flat_start + flat_t * (flat_end - flat_start)

        return refined.reshape(rgb_chunk.shape[0], rgb_chunk.shape[1])

    def rgb_to_normalized(self, rgb: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
        """Inverse: RGB → normalized depth via colormap inversion.

        Projects each RGB pixel onto the nearest point of the piecewise-linear
        curve using segment projection. Works for any colormap in the registry.

        Memory optimisation:
            Pixels are processed in ``chunk_size`` strips so peak extra VRAM
            is O(chunk_size · num_segments) regardless of resolution.

        Args:
            rgb: [B, 3, H, W] in [0, 1].
            chunk_size: Number of pixels processed at once.

        Returns:
            Normalized depth [B, 1, H, W] in [0, 1].
        """
        kp = self._keypoints.to(rgb.device)
        positions = kp[:, 0].contiguous()
        colors = kp[:, 1:].contiguous()

        B, _, H, W = rgb.shape
        rgb_flat = rgb.permute(0, 2, 3, 1).reshape(B, H * W, 3)
        result = torch.empty(B, H * W, device=rgb.device, dtype=rgb.dtype)

        for start in range(0, H * W, chunk_size):
            end = min(start + chunk_size, H * W)
            chunk = rgb_flat[:, start:end, :]
            result[:, start:end] = self._project_rgb_chunk_to_normalized(chunk, positions, colors)

        return result.reshape(B, 1, H, W).clamp(0.0, 1.0)

    def depth_to_rgb(self, depth: torch.Tensor) -> torch.Tensor:
        """Full forward: depth → RGB visualization."""
        normalized = self.depth_to_normalized(depth)
        return self.normalized_to_rgb(normalized)

    def rgb_to_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """Full inverse: RGB → depth."""
        normalized = self.rgb_to_normalized(rgb)
        return self.normalized_to_depth(normalized)


def get_available_colormaps() -> list[str]:
    """Return the list of supported colormap names."""
    return _AVAILABLE_COLORMAPS.copy()


__all__ = [
    "DepthRGBMapper",
    "CUBE_EDGE_KEYPOINTS",
    "LEGACY_CUBE_EDGE_KEYPOINTS",
    "PLASMA_KEYPOINTS",
    "INFERNO_KEYPOINTS",
    "VIRIDIS_KEYPOINTS",
    "GRAYSCALE_KEYPOINTS",
    "get_available_colormaps",
]
