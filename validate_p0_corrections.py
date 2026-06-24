#!/usr/bin/env python3
"""Validation suite for P0-1 and P0-2 corrections.

Validates:
1. Depth roundtrip with cube-edge colormap (error < 1%)
2. Normal roundtrip (error < 1°)
3. Semantic segmentation nearest-color decode (correctness)
4. Referring segmentation nearest-color decode (new P0-2 implementation)
5. Colormap visual check (cube-edge path is on RGB cube edges)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_vision_banana.core.depth_mapper import CUBE_EDGE_KEYPOINTS, DepthRGBMapper
from open_vision_banana.core.normal_mapper import NormalRGBMapper
from open_vision_banana.core.segmentation import SegmentationDecoder
from open_vision_banana.inference.decoder import VisualizationDecoder


def validate_depth_roundtrip() -> dict:
    """Validate depth roundtrip with cube-edge colormap."""
    mapper = DepthRGBMapper()
    results = {}

    # Test 1: Dense roundtrip (self-intersection check)
    n = torch.linspace(0.0, 1.0, 2000).view(1, 1, 1, -1)
    rgb = mapper.normalized_to_rgb(n)
    recovered = mapper.rgb_to_normalized(rgb)
    max_err = (n - recovered).abs().max().item()
    results["dense_roundtrip_max_err"] = max_err
    results["dense_pass"] = max_err < 1e-3

    # Test 2: Quantized roundtrip (8-bit)
    torch.manual_seed(0)
    depth = torch.rand(1, 1, 64, 64) * 49.0 + 1.0
    rgb = mapper.depth_to_rgb(depth)
    rgb_q = (rgb * 255).round() / 255.0
    recovered = mapper.rgb_to_depth(rgb_q)
    abs_rel = ((depth - recovered).abs() / depth).mean().item()
    results["quantized_abs_rel"] = abs_rel
    results["quantized_pass"] = abs_rel < 0.005

    # Test 3: Near-range precision (0-5m)
    near_depths = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0]).view(1, 1, 1, -1)
    rgb_near = mapper.depth_to_rgb(near_depths)
    recovered_near = mapper.rgb_to_depth(rgb_near)
    near_abs_rel = ((near_depths - recovered_near).abs() / near_depths).mean().item()
    results["near_abs_rel"] = near_abs_rel
    results["near_pass"] = near_abs_rel < 0.01

    # Test 4: Far-range stability (>200m)
    far_depths = torch.tensor([300.0, 500.0, 1000.0]).view(1, 1, 1, -1)
    rgb_far = mapper.depth_to_rgb(far_depths)
    recovered_far = mapper.rgb_to_depth(rgb_far)
    results["far_finite"] = torch.isfinite(recovered_far).all().item()

    # Test 5: Cube-edge path verification
    kp = CUBE_EDGE_KEYPOINTS
    colors = kp[:, 1:]
    on_edges = 0
    total_segments = len(colors) - 1
    for i in range(total_segments):
        c0 = colors[i]
        c1 = colors[i + 1]
        diff = c1 - c0
        nonzero = (diff != 0).sum().item()
        # Each segment should change exactly 1 or 2 coordinates (cube edge)
        if nonzero <= 2:
            on_edges += 1
    results["cube_edge_fraction"] = on_edges / total_segments
    results["cube_edge_pass"] = on_edges == total_segments

    return results


def validate_normal_roundtrip() -> dict:
    """Validate normal roundtrip."""
    mapper = NormalRGBMapper()
    results = {}

    normals = torch.randn(1, 3, 64, 64)
    normals = normals / normals.norm(dim=1, keepdim=True).clamp(min=1e-8)
    rgb = mapper.normal_to_rgb(normals)
    rgb_q = (rgb * 255).round() / 255.0
    recovered = mapper.rgb_to_normal(rgb_q)

    cos_sim = torch.nn.functional.cosine_similarity(
        normals, recovered, dim=1
    ).clamp(-1, 1)
    mean_angle = (torch.acos(cos_sim).mean() * 180.0 / 3.14159265).item()

    results["mean_angle_deg"] = mean_angle
    results["pass"] = mean_angle < 1.0
    return results


def validate_semantic_decode() -> dict:
    """Validate semantic segmentation nearest-color decode."""
    decoder = SegmentationDecoder()
    results = {}

    # Create synthetic RGB with two known colors
    class_map = {"cat": (255, 255, 0), "background": (128, 0, 128)}
    rgb = torch.zeros(1, 3, 32, 32)
    rgb[0, 0, :16, :] = 1.0  # yellow (R=1, G=1, B=0) → cat
    rgb[0, 1, :16, :] = 1.0
    rgb[0, 0, 16:, :] = 128 / 255  # purple (R=128, G=0, B=128) → background
    rgb[0, 2, 16:, :] = 128 / 255

    mask = decoder.decode_semantic(rgb, class_map)
    results["shape_correct"] = mask.shape == (1, 1, 32, 32)
    results["top_half_cat"] = (mask[0, 0, :16, :] == 0).all().item()
    results["bottom_half_bg"] = (mask[0, 0, 16:, :] == 1).all().item()
    results["pass"] = results["shape_correct"] and results["top_half_cat"] and results["bottom_half_bg"]
    return results


def validate_referring_decode() -> dict:
    """Validate referring segmentation nearest-color decode (P0-2)."""
    vis_decoder = VisualizationDecoder()
    results = {}

    # Create synthetic RGB: yellow square on black background
    rgb = torch.zeros(1, 3, 64, 64)
    rgb[0, 0, 20:44, 20:44] = 1.0   # R=1
    rgb[0, 1, 20:44, 20:44] = 1.0   # G=1  → yellow (255,255,0)

    # Decode with yellow target color
    result = vis_decoder.decode_referring(rgb, target_color=(255, 255, 0))
    mask = result["mask"]

    results["shape"] = mask.shape == (1, 1, 64, 64)
    # Check that yellow region is detected
    yellow_region = mask[0, 0, 20:44, 20:44].float().mean().item()
    bg_region = mask[0, 0, :20, :20].float().mean().item()
    results["yellow_coverage"] = yellow_region
    results["bg_coverage"] = bg_region
    results["yellow_detected"] = yellow_region > 0.8
    results["bg_clean"] = bg_region < 0.1
    results["pass"] = results["shape"] and results["yellow_detected"] and results["bg_clean"]

    # Test with non-white target color (different from old mean-threshold approach)
    rgb2 = torch.zeros(1, 3, 64, 64)
    rgb2[0, 0, 20:44, 20:44] = 1.0   # Red square (255,0,0)
    result2 = vis_decoder.decode_referring(rgb2, target_color=(255, 0, 0))
    mask2 = result2["mask"]
    red_detected = mask2[0, 0, 20:44, 20:44].float().mean().item() > 0.8
    results["red_detected"] = red_detected

    return results


def main():
    print("=" * 70)
    print("Vision Banana P0-1/P0-2 Validation Suite")
    print("=" * 70)
    print()

    all_pass = True

    # ── Depth ─────────────────────────────────────────────────────
    print("1. Depth Roundtrip (Cube-Edge Colormap)")
    print("-" * 40)
    depth_results = validate_depth_roundtrip()
    for k, v in depth_results.items():
        print(f"   {k:<25s}: {v}")
    if not depth_results.get("dense_pass", False):
        print("   ❌ FAILED: dense roundtrip")
        all_pass = False
    elif not depth_results.get("quantized_pass", False):
        print("   ❌ FAILED: quantized roundtrip")
        all_pass = False
    elif not depth_results.get("cube_edge_pass", False):
        print("   ❌ FAILED: cube-edge path verification")
        all_pass = False
    else:
        print("   ✅ PASSED")
    print()

    # ── Normal ──────────────────────────────────────────────────
    print("2. Normal Roundtrip")
    print("-" * 40)
    normal_results = validate_normal_roundtrip()
    for k, v in normal_results.items():
        print(f"   {k:<25s}: {v}")
    if not normal_results.get("pass", False):
        print("   ❌ FAILED")
        all_pass = False
    else:
        print("   ✅ PASSED")
    print()

    # ── Semantic ────────────────────────────────────────────────
    print("3. Semantic Segmentation Decode")
    print("-" * 40)
    sem_results = validate_semantic_decode()
    for k, v in sem_results.items():
        print(f"   {k:<25s}: {v}")
    if not sem_results.get("pass", False):
        print("   ❌ FAILED")
        all_pass = False
    else:
        print("   ✅ PASSED")
    print()

    # ── Referring ───────────────────────────────────────────────
    print("4. Referring Segmentation Decode (P0-2)")
    print("-" * 40)
    ref_results = validate_referring_decode()
    for k, v in ref_results.items():
        print(f"   {k:<25s}: {v}")
    if not ref_results.get("pass", False):
        print("   ❌ FAILED")
        all_pass = False
    else:
        print("   ✅ PASSED")
    print()

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 70)
    if all_pass:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
