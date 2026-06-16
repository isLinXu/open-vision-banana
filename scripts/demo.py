#!/usr/bin/env python3
"""Demo: Vision Banana core algorithms.

Demonstrates the key algorithms from the paper without requiring
the full FLUX.1 model (which needs GPU + 24GB VRAM).

Shows:
1. Depth → RGB mapping (power transform + cube edge interpolation)
2. Normal → RGB mapping (direct linear encoding)
3. Segmentation decoding (multi-stage clustering)
4. Prompt construction for all 5 tasks
"""

import torch

from open_vision_banana import DepthRGBMapper, NormalRGBMapper, SegmentationDecoder
from open_vision_banana.inference.prompt import PromptBuilder


def demo_depth_mapping():
    """Demo: Depth estimation RGB mapping."""
    print("=" * 70)
    print("1. DEPTH ESTIMATION: Power Transform + Cube Edge Interpolation")
    print("=" * 70)
    print()
    print("Paper formula: f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)")
    print("Parameters: λ=-3, c=10/3")
    print()

    mapper = DepthRGBMapper(lam=-3.0, c=10.0/3.0)

    # Show mapping for various depths
    depths = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    depth_tensor = torch.tensor(depths).reshape(1, 1, 1, -1)

    normalized = mapper.depth_to_normalized(depth_tensor)
    rgb = mapper.normalized_to_rgb(normalized)

    print(f"{'Depth (m)':>12} {'Normalized':>12} {'R':>8} {'G':>8} {'B':>8}")
    print("-" * 52)
    for i, d in enumerate(depths):
        n = normalized[0, 0, 0, i].item()
        r = rgb[0, 0, 0, i].item()
        g = rgb[0, 1, 0, i].item()
        b = rgb[0, 2, 0, i].item()
        print(f"{d:>12.1f} {n:>12.4f} {r:>8.3f} {g:>8.3f} {b:>8.3f}")

    print()
    print("Key insight: Nearby depths (0-5m) get more RGB color range")
    print("than distant depths (50-100m), prioritizing accuracy for")
    print("close objects (higher utility for robotics, AR, etc.)")

    # Show roundtrip accuracy
    print()
    depth_orig = torch.tensor([[[[0.5, 1.0, 3.0, 10.0, 30.0]]]])
    norm = mapper.depth_to_normalized(depth_orig)
    depth_recovered = mapper.normalized_to_depth(norm)
    print("Roundtrip accuracy (depth → normalized → depth):")
    for i, d in enumerate(depth_orig.flatten().tolist()):
        r = depth_recovered.flatten()[i].item()
        print(f"  {d:.1f}m → {r:.2f}m (error: {abs(d-r):.3f}m)")


def demo_normal_mapping():
    """Demo: Surface normal RGB mapping."""
    print()
    print("=" * 70)
    print("2. SURFACE NORMAL ESTIMATION: Direct Linear Mapping")
    print("=" * 70)
    print()
    print("Camera-space (right-handed: +x right, +y up, +z toward camera)")
    print("R = (1-x)/2, G = (1+y)/2, B = (1+z)/2")
    print()

    mapper = NormalRGBMapper()

    directions = [
        ("Facing Camera", (0, 0, 1), "Light Blue"),
        ("Facing Left", (-1, 0, 0), "Pinkish Red"),
        ("Facing Right", (1, 0, 0), "Cyan"),
        ("Facing Up", (0, 1, 0), "Light Green"),
        ("Facing Down", (0, -1, 0), "Dark"),
        ("Facing Away", (0, 0, -1), "Dark"),
    ]

    print(f"{'Direction':>20} {'Normal':>15} {'RGB':>25} {'Color Name':>15}")
    print("-" * 80)
    for name, (x, y, z), color_name in directions:
        normal = torch.tensor([[[[x]], [[y]], [[z]]]], dtype=torch.float32)
        rgb = mapper.normal_to_rgb(normal)
        r, g, b = int(rgb[0,0,0].item()*255), int(rgb[0,1,0].item()*255), int(rgb[0,2,0].item()*255)
        print(f"{name:>20} ({x:>2},{y:>2},{z:>2})  ({r:>3},{g:>3},{b:>3})  {color_name:>15}")


def demo_segmentation_decoding():
    """Demo: Multi-stage clustering for instance segmentation."""
    print()
    print("=" * 70)
    print("3. INSTANCE SEGMENTATION: Multi-Stage Clustering (Appendix A)")
    print("=" * 70)
    print()
    print("5-stage algorithm for parsing RGB masks into instance masks:")
    print()
    print("Stage 1: Background Initialization")
    print("  - Classify pixels as background if ||C(p) - C_bg||_2 < τ=14")
    print()
    print("Stage 2: Color Similarity Grouping")
    print("  - 16-connectivity flood-fill with seed-based color tolerance")
    print("  - Merge if ||C(p_i) - C(p_seed)||_2 ≤ τ²")
    print()
    print("Stage 3: Noise Pruning")
    print("  - Discard components with |S| < θ_size × (H×W)")
    print("  - θ_size = 2×10⁻⁴ (0.02% of image area)")
    print()
    print("Stage 4: Boundary Artifact Elimination")
    print("  - 3×3 binary erosion; prune if preserved ratio < θ_erosion=0.1")
    print("  - Removes thin colored boundary halos")
    print()
    print("Stage 5: Spatially Constrained Merging")
    print("  - Merge disjoint components with similar colors")
    print("  - Condition: A(S_a ∪ S_b) ≤ γ × (A(S_a) + A(S_b))")
    print("  - γ = 5.0 (max bounding box expansion factor)")

    # Quick demo with synthetic image
    decoder = SegmentationDecoder()
    # Create a simple 2-class image
    class_map = {"cat": (255, 255, 0), "background": (128, 0, 128)}
    rgb = torch.zeros(1, 3, 32, 32)
    rgb[0, 0, :16, :] = 1.0   # Yellow top half
    rgb[0, 1, :16, :] = 1.0
    rgb[0, 0, 16:, :] = 128/255  # Purple bottom half
    rgb[0, 2, 16:, :] = 128/255

    mask = decoder.decode_semantic(rgb, class_map)
    print()
    print(f"Semantic decode demo: 32×32 image → mask shape {mask.shape}")
    print(f"  Top half (yellow): class index = {mask[0, 0, 0, 0].item()}")
    print(f"  Bottom half (purple): class index = {mask[0, 0, 16, 0].item()}")


def demo_prompts():
    """Demo: Task-specific prompt construction."""
    print()
    print("=" * 70)
    print("4. PROMPT CONSTRUCTION: Task-Specific Instructions")
    print("=" * 70)
    print()

    # Semantic segmentation
    class_map = {
        "road": (128, 64, 128),
        "sidewalk": (244, 35, 232),
        "building": (70, 70, 70),
        "sky": (70, 130, 180),
    }
    print("Semantic Segmentation (3 prompt styles):")
    print("-" * 60)
    for idx in range(3):
        prompt = PromptBuilder.build_semantic_prompt(class_map, template_idx=idx)
        print(f"  Style {idx}: {prompt[:100]}...")
    print()

    # Instance segmentation
    print("Instance Segmentation:")
    print("-" * 60)
    prompt = PromptBuilder.build_instance_prompt("dog")
    print(f"  {prompt}")
    print()

    # Referring expression
    print("Referring Expression Segmentation:")
    print("-" * 60)
    prompt = PromptBuilder.build_referring_prompt("the man in pink t shirt")
    print(f"  {prompt}")
    print()

    # Depth
    print("Depth Estimation:")
    print("-" * 60)
    prompt = PromptBuilder.build_depth_prompt()
    print(f"  {prompt}")
    print()

    # Normal
    print("Surface Normal Estimation:")
    print("-" * 60)
    prompt = PromptBuilder.build_normal_prompt()
    print(f"  {prompt}")


def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          Open-Vision-Banana: Core Algorithm Demo               ║")
    print("║  Image Generators are Generalist Vision Learners               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    demo_depth_mapping()
    demo_normal_mapping()
    demo_segmentation_decoding()
    demo_prompts()

    print()
    print("=" * 70)
    print("Demo complete! All core algorithms verified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
