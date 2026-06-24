"""Optical flow RGB encoder/decoder using HSV encoding.

Encoding: flow (u, v) -> HSV -> RGB
    - hue = atan2(v, u) / (2*pi) + 0.5       # [0, 1]
    - saturation = magnitude / max_flow      # [0, 1], clipped
    - value = 0.5                              # fixed brightness

Decoding: RGB -> HSV -> flow
    - direction = (hue - 0.5) * 2*pi
    - magnitude = saturation * max_flow
    - u = magnitude * cos(direction)
    - v = magnitude * sin(direction)
"""

from __future__ import annotations

import math

import torch


class FlowRGBMapper:
    """Optical flow RGB encoder/decoder using HSV encoding.

    Encoding: flow (u, v) -> HSV -> RGB.
    Decoding: RGB -> HSV -> flow.
    """

    def __init__(self, max_flow: float = 20.0) -> None:
        self.max_flow = max_flow

    def _hsv_to_rgb(self, h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Convert HSV to RGB.  All tensors in [0, 1].

        Manual implementation using PyTorch ops only (no cv2/torchvision).
        """
        # Standard HSV -> RGB conversion
        h6 = h * 6.0  # [0, 6)
        i = h6.floor().long()  # integer part [0, 5]
        f = h6 - h6.floor()  # fractional part

        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        # For each sector, pick (r, g, b) from (v, t, p, q)
        i = i % 6
        i = i.clamp(0, 5)

        r = torch.where(
            i == 0, v,
            torch.where(
                i == 1, q,
                torch.where(
                    i == 2, p,
                    torch.where(
                        i == 3, p,
                        torch.where(i == 4, t, v)
                    )
                )
            )
        )
        g = torch.where(
            i == 0, t,
            torch.where(
                i == 1, v,
                torch.where(
                    i == 2, v,
                    torch.where(
                        i == 3, q,
                        torch.where(i == 4, p, p)
                    )
                )
            )
        )
        b = torch.where(
            i == 0, p,
            torch.where(
                i == 1, p,
                torch.where(
                    i == 2, t,
                    torch.where(
                        i == 3, v,
                        torch.where(i == 4, v, q)
                    )
                )
            )
        )

        # When saturation is 0, all channels should equal value (gray)
        gray = s < 1e-6
        r = torch.where(gray, v, r)
        g = torch.where(gray, v, g)
        b = torch.where(gray, v, b)

        return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

    def _rgb_to_hsv(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert RGB to HSV.  Input in [0, 1], outputs in [0, 1]."""
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]

        mx = torch.max(torch.max(r, g), b)
        mn = torch.min(torch.min(r, g), b)
        diff = mx - mn + 1e-10

        # Hue
        h = torch.zeros_like(r)
        # r is max
        mask_r = (mx == r) & (diff > 1e-6)
        h = torch.where(mask_r, ((g - b) / diff) % 6.0, h)
        # g is max
        mask_g = (mx == g) & (diff > 1e-6)
        h = torch.where(mask_g, ((b - r) / diff + 2.0) % 6.0, h)
        # b is max
        mask_b = (mx == b) & (diff > 1e-6)
        h = torch.where(mask_b, ((r - g) / diff + 4.0) % 6.0, h)
        h = h / 6.0  # normalize to [0, 1]

        # Saturation
        s = torch.where(mx > 1e-6, diff / (mx + 1e-10), torch.zeros_like(r))

        # Value
        v = mx

        return h, s, v

    def flow_to_rgb(self, flow: torch.Tensor) -> torch.Tensor:
        """Encode optical flow to RGB visualization.

        Args:
            flow: Optical flow [B, 2, H, W] where channel 0 = u, channel 1 = v.

        Returns:
            RGB tensor [B, 3, H, W] in [0, 1].
        """
        u = flow[:, 0:1]
        v = flow[:, 1:2]

        magnitude = torch.sqrt(u**2 + v**2 + 1e-10)
        direction = torch.atan2(v, u)  # [-pi, pi]

        # Hue: direction -> [0, 1]
        hue = (direction / (2.0 * math.pi) + 0.5).clamp(0.0, 1.0)
        # Saturation: magnitude normalized to [0, 1]
        sat = (magnitude / self.max_flow).clamp(0.0, 1.0)
        # Value: fixed at 0.5 for brightness
        val = torch.full_like(hue, 0.5)

        return self._hsv_to_rgb(hue, sat, val)

    def rgb_to_flow(self, rgb: torch.Tensor) -> torch.Tensor:
        """Decode RGB visualization back to optical flow.

        Args:
            rgb: RGB tensor [B, 3, H, W] in [0, 1].

        Returns:
            Flow tensor [B, 2, H, W] (u, v).
        """
        hue, sat, val = self._rgb_to_hsv(rgb)

        # Direction: hue -> [-pi, pi]
        direction = (hue - 0.5) * 2.0 * math.pi
        # Magnitude: saturation scaled back
        magnitude = sat * self.max_flow

        u = magnitude * torch.cos(direction)
        v = magnitude * torch.sin(direction)

        return torch.cat([u, v], dim=1)


__all__ = ["FlowRGBMapper"]
