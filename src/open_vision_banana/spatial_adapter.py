"""Optional multi-scale source-image adapter for SSD/SDXL UNets."""

from __future__ import annotations

import torch
from torch import nn

SPATIAL_ADAPTER_STATE_NAME = "spatial_adapter.pt"
T2I_ADAPTER_XL = "t2i_adapter_xl"


class SourceSpatialAdapter(nn.Module):
    """T2I-Adapter-XL with ControlNet-style zero output projections."""

    def __init__(
        self,
        channels: tuple[int, ...] = (320, 640, 1280, 1280),
        *,
        downscale_factor: int = 16,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) < 2 or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain at least two positive values")
        if downscale_factor <= 0:
            raise ValueError("downscale_factor must be positive")
        from diffusers import T2IAdapter

        self.channels = channels
        self.downscale_factor = downscale_factor
        self.adapter = T2IAdapter(
            in_channels=3,
            channels=list(channels),
            num_res_blocks=num_res_blocks,
            downscale_factor=downscale_factor,
            adapter_type="full_adapter_xl",
        )
        self.zero_convs = nn.ModuleList(
            nn.Conv2d(channel, channel, kernel_size=1) for channel in channels
        )
        for projection in self.zero_convs:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, source_rgb: torch.Tensor) -> list[torch.Tensor]:
        if source_rgb.ndim != 4 or source_rgb.shape[1] != 3:
            raise ValueError("source_rgb must be shaped [B, 3, H, W]")
        height, width = source_rgb.shape[-2:]
        if height % self.downscale_factor or width % self.downscale_factor:
            raise ValueError(
                "source image dimensions must be divisible by adapter downscale factor"
            )
        features = self.adapter(source_rgb)
        if len(features) != len(self.zero_convs):
            raise RuntimeError(
                f"Adapter returned {len(features)} features; expected {len(self.zero_convs)}"
            )
        return [
            projection(feature)
            for projection, feature in zip(self.zero_convs, features, strict=True)
        ]


def create_source_spatial_adapter(
    unet: nn.Module,
    *,
    downscale_factor: int = 16,
) -> SourceSpatialAdapter:
    """Build an adapter matching an SDXL UNet's down-block channels."""
    config = getattr(unet, "config", None)
    block_channels = tuple(int(value) for value in getattr(config, "block_out_channels", ()))
    if not block_channels:
        raise ValueError("UNet config must declare block_out_channels")
    channels = (*block_channels, block_channels[-1])
    return SourceSpatialAdapter(channels, downscale_factor=downscale_factor)


__all__ = [
    "SPATIAL_ADAPTER_STATE_NAME",
    "T2I_ADAPTER_XL",
    "SourceSpatialAdapter",
    "create_source_spatial_adapter",
]
