"""Native observed-depth filtering and metric RGB target preparation."""

from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

POLICY = "indoor_metric_observed98_v3"
ROTATIONS = {
    "Up": None,
    "Left": Image.Transpose.ROTATE_270,
    "Right": Image.Transpose.ROTATE_90,
    "Down": Image.Transpose.ROTATE_180,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def orient(array, direction):
    if direction not in ROTATIONS:
        raise ValueError("Unknown sky direction")
    image = Image.fromarray(array)
    return np.asarray(image.transpose(ROTATIONS[direction]) if ROTATIONS[direction] else image)


def clean(rgb, depth, valid):
    """Keep >=98% observed frames; retain missing measurements without filling."""
    depth = np.asarray(depth, np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected native uint8 RGB")
    if rgb.shape[:2] != depth.shape or valid.shape != depth.shape:
        raise ValueError("Unaligned RGB/depth/mask")
    valid = valid & np.isfinite(depth) & (depth >= 0.25) & (depth <= 20)
    coverage = float(valid.mean())
    if coverage < 0.98:
        raise ValueError("insufficient_observed_depth")
    values = depth[valid]
    p5, p50, p95 = np.quantile(values, [0.05, 0.5, 0.95])
    if p95 - p5 < 0.3 or p50 > 10:
        raise ValueError("flat_or_distant_depth")
    gray = np.asarray(Image.fromarray(rgb).convert("L"), np.float32)
    brightness, sharpness = float(gray.mean()), float(laplace(gray).var())
    if not 8 <= brightness <= 247 or sharpness < 15:
        raise ValueError("dark_bright_or_blurred_rgb")
    observed_depth = np.where(valid, depth, 0).astype(np.float32)
    return (
        observed_depth,
        valid,
        dict(
            valid_fraction=coverage,
            missing_fraction=1 - coverage,
            depth_p5=float(p5),
            depth_median=float(p50),
            depth_p95=float(p95),
            brightness=brightness,
            sharpness=sharpness,
        ),
    )


def save_pair(root, identity, rgb, depth, valid, provenance):
    """Preserve observed millimetres, mask, lossless RGB input and cube-edge target."""
    import torch

    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    from open_vision_banana.data.hypersim_supplement import decode_quantized_rgb

    _, valid, stats = clean(rgb, depth, valid)
    observed = np.where(valid, np.rint(depth * 1000), 0).astype(np.uint16)
    observed_depth = observed.astype(np.float32) / 1000
    mapper = DepthRGBMapper()
    encoded = mapper.depth_to_rgb(torch.from_numpy(observed_depth.copy())[None, None])
    target = (encoded[0].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
    decoded = decode_quantized_rgb(target, mapper)
    error = np.abs(decoded[valid] - observed_depth[valid])
    if error.max() > 0.15:
        raise ValueError("metric_encoding_roundtrip")
    pixels = sha(f"{rgb.shape[1]}x{rgb.shape[0]}:".encode() + rgb.tobytes())
    paths = {
        "input": f"images/{identity}.png",
        "target": f"targets/{identity}.png",
        "metric_depth": f"metric_depth/{identity}.png",
        "valid_mask": f"valid_masks/{identity}.png",
    }
    arrays = {
        "input": rgb,
        "target": target,
        "metric_depth": observed,
        "valid_mask": valid.astype(np.uint8) * 255,
    }
    assets = []
    for key, name in paths.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arrays[key]).save(path, compress_level=3)
        data = path.read_bytes()
        assets.append(dict(path=name, bytes=len(data), sha256=sha(data)))
    return dict(
        row=dict(
            task="depth",
            sample_id=identity,
            **paths,
            prompt="Generate a metric depth visualization of this image.",
            target_kind="metric_depth",
            encoding="cube_edge_metric_power",
            depth_convention="planar_z_depth",
            depth_units="metres",
            metric_depth_scale=1000,
            target_hole_policy="zero_missing_no_interpolation",
            rgb_pixels_sha256=pixels,
            **provenance,
        ),
        assets=assets,
        statistics=dict(
            **stats,
            encoding_max_error=float(error.max()),
            source_depth_quantization_max_error=float(
                np.abs(observed_depth[valid] - depth[valid]).max()
            ),
        ),
        rgb_pixels_sha256=pixels,
    )


def read_png(data):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return np.array(image)
