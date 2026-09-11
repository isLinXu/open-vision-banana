#!/usr/bin/env python3
"""Run validated task-specialist inference paths.

This replaces the current MAR/SANA RGB diffusion outputs for tasks whose v5
targets have a directly auditable construction.  The DINOv2 semantic and DIODE
normal heads are only enabled through explicit checkpoints, so their use stays
traceable to the held-out gates that validated them.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_sana_structure_binary_targets import build_binary_structure_map
from infer_depth_normal import angular_variation_degrees, normal_from_relative_depth, normal_to_rgb
from infer_depth_pseudo import (
    depth_to_rgb,
    estimate_depth,
    load_depth_estimator,
    normalize_depth_to_uint8,
    select_device,
)
from normal_dinov2_probe import NormalTokenHead, expanded_normal_grid as expand_normal_grid
from segmentation_dinov2_probe import DINO_FEATURE_DIM, LocalDINOv2ViTS14
from segmentation_resnet18_probe import TokenClassifier, load_palette


AVAILABLE_TASKS = ("depth", "normal", "structure", "semantic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="depth,normal,structure", help="Comma-separated: depth, normal, structure, semantic.")
    parser.add_argument("--resolution", type=int, default=0, help="Square output size; 0 preserves input resolution.")
    parser.add_argument("--normal-strength", type=float, default=1.0)
    parser.add_argument("--min-line-fraction", type=float, default=0.08)
    parser.add_argument("--edge-width", type=int, default=3)
    parser.add_argument(
        "--structure-mode", choices=("rule", "sana"), default="rule",
        help="Structure source. Sana mode requires --structure-checkpoint.",
    )
    parser.add_argument("--structure-checkpoint", type=Path, help="Validated binary-structure SANA LoRA adapter directory.")
    parser.add_argument("--structure-threshold", type=float, default=0.30, help="SANA structure luminance threshold; checkpoint-250 is validated at 0.30.")
    parser.add_argument("--structure-inference-steps", type=int, default=20)
    parser.add_argument("--structure-guidance-scale", type=float, default=1.0)
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument(
        "--normal-mode", choices=("depth_derived", "diode_dinov2"), default="depth_derived",
        help="Normal source. DIODE mode requires --normal-checkpoint.",
    )
    parser.add_argument("--normal-checkpoint", type=Path, help="Validated DIODE DINOv2 normal-head checkpoint.")
    parser.add_argument("--semantic-checkpoint", type=Path, help="Validated DINOv2 semantic-head checkpoint.")
    parser.add_argument("--semantic-palette", type=Path, help="Fixed 150-class semantic palette JSON.")
    parser.add_argument(
        "--dino-repository", type=Path,
        default=Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main",
    )
    parser.add_argument(
        "--dino-weights", type=Path,
        default=Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth",
    )
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    return parser.parse_args()


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(dict.fromkeys(task.strip() for task in value.split(",") if task.strip()))
    unknown = sorted(set(tasks) - set(AVAILABLE_TASKS))
    if unknown:
        raise ValueError(
            f"unsupported specialist task(s): {', '.join(unknown)}. "
            "Supported tasks are depth, normal, structure, semantic."
        )
    if not tasks:
        raise ValueError("at least one task is required")
    return tasks


def resize_depth(depth: np.ndarray, resolution: int) -> np.ndarray:
    if not resolution:
        return depth
    tensor = torch.from_numpy(depth)[None, None]
    return torch.nn.functional.interpolate(
        tensor, size=(resolution, resolution), mode="bilinear", align_corners=False
    ).squeeze().numpy()


def load_checkpoint(path: Path, label: str) -> dict:
    checkpoint_path = path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"unable to load {label} checkpoint: {checkpoint_path}") from error
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("head"), dict):
        raise ValueError(f"{label} checkpoint has no serialised head: {checkpoint_path}")
    return checkpoint


def checkpoint_image_size(checkpoint: dict, label: str) -> int:
    value = checkpoint.get("image_size")
    if not isinstance(value, int) or value < 14 or value % 14:
        raise ValueError(f"{label} checkpoint contains an invalid DINO image size: {value!r}")
    return value


def checkpoint_hidden_dim(checkpoint: dict, label: str) -> int:
    weight = checkpoint["head"].get("layers.1.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[1] != DINO_FEATURE_DIM:
        raise ValueError(f"{label} checkpoint has an incompatible DINOv2 head")
    return int(weight.shape[0])


def load_semantic_head(checkpoint: dict, device: torch.device) -> TokenClassifier:
    head = TokenClassifier(DINO_FEATURE_DIM, checkpoint_hidden_dim(checkpoint, "semantic"))
    head.load_state_dict(checkpoint["head"], strict=True)
    return head.to(device).eval()


def load_normal_head(checkpoint: dict, device: torch.device) -> NormalTokenHead:
    head = NormalTokenHead(DINO_FEATURE_DIM, checkpoint_hidden_dim(checkpoint, "normal"))
    head.load_state_dict(checkpoint["head"], strict=True)
    return head.to(device).eval()


@torch.no_grad()
def dino_features(source: Image.Image, image_size: int, backbone: LocalDINOv2ViTS14,
                  device: torch.device) -> np.ndarray:
    """Extract local DINOv2 patch tokens for one RGB image."""
    # Reuse the exact ImageNet preprocessing used by the held-out gates.
    array = np.asarray(
        source.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    return backbone(tensor).cpu().numpy()


def semantic_to_rgb(labels: np.ndarray, palette: np.ndarray) -> Image.Image:
    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError(f"expected [H, W] semantic labels, got {values.shape}")
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    valid = (values >= 0) & (values < len(palette))
    output[valid] = palette[values[valid]]
    return Image.fromarray(output, mode="RGB")


def normal_to_diode_rgb(normal: np.ndarray) -> Image.Image:
    """Encode the DIODE head convention, normal = normalize(2*RGB/255 - 1)."""
    values = np.asarray(normal, dtype=np.float32)
    return Image.fromarray(np.rint(np.clip((values + 1.0) * 0.5, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")


def infer_sana_structure(
    source: Image.Image,
    checkpoint: Path,
    device: torch.device,
    output_resolution: int,
    threshold: float,
    inference_steps: int,
    guidance_scale: float,
) -> tuple[Image.Image, Image.Image]:
    """Generate a 512px SANA structure map, then apply the calibrated binary threshold."""
    if not checkpoint.is_dir() or not (checkpoint / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"SANA structure adapter not found: {checkpoint}")
    if not 0.0 < threshold < 1.0 or inference_steps < 1 or guidance_scale <= 0:
        raise ValueError("invalid SANA structure inference parameters")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from infer_sana_512 import flow_matching_inference, get_dtype, load_model_and_lora

    dtype = get_dtype("float16")
    model_args = argparse.Namespace(
        model_name="sana-600m-512px",
        checkpoint=str(checkpoint),
        lora_rank=32,
        lora_alpha=64,
        no_rslora=False,
        dtype="float16",
    )
    peft_model, vae, components, model_cfg = load_model_and_lora(model_args, device, dtype)
    model_dtype = next(peft_model.parameters()).dtype
    source_512 = source.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.asarray(source_512, dtype=np.float32) / 255.0).permute(2, 0, 1)
    prompt = "Generate a binary structural edge map. Use a black background and thin white edges aligned to visible scene boundaries."
    try:
        prediction = flow_matching_inference(
            peft_model,
            vae,
            components,
            (tensor * 2.0 - 1.0).unsqueeze(0).to(device=device, dtype=model_dtype),
            [prompt],
            device,
            model_dtype,
            num_steps=inference_steps,
            use_img2img=True,
            max_length=model_cfg.text_encoder_max_length,
            guidance_scale=guidance_scale,
        )[0].cpu()
    finally:
        del peft_model, vae, components
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    raw = Image.fromarray(
        np.rint(prediction.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    )
    binary = np.where(np.asarray(raw.convert("L"), dtype=np.float32) / 255.0 >= threshold, 255, 0).astype(np.uint8)
    binary_image = Image.fromarray(np.repeat(binary[..., None], 3, axis=2), mode="RGB")
    if output_resolution != 512:
        raw = raw.resize((output_resolution, output_resolution), Image.Resampling.NEAREST)
        binary_image = binary_image.resize((output_resolution, output_resolution), Image.Resampling.NEAREST)
    return raw, binary_image


def write_panel(path: Path, source: Image.Image, outputs: list[tuple[str, Image.Image]]) -> None:
    preview_size, label_height, gap = 256, 28, 8
    images = [("Input", source.convert("RGB").resize((preview_size, preview_size), Image.Resampling.BICUBIC))]
    images.extend((label, image.convert("RGB").resize((preview_size, preview_size), Image.Resampling.NEAREST)) for label, image in outputs)
    canvas = Image.new("RGB", (len(images) * preview_size + (len(images) + 1) * gap, preview_size + label_height + 2 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        x = gap + index * (preview_size + gap)
        draw.text((x, 6), label, fill="black")
        canvas.paste(image, (x, label_height + gap))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    try:
        tasks = parse_tasks(args.tasks)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.resolution < 0 or args.normal_strength <= 0:
        raise SystemExit("--resolution must be non-negative and --normal-strength must be positive")
    if not 0.0 < args.min_line_fraction <= 1.0 or args.edge_width < 1 or args.edge_width % 2 == 0:
        raise SystemExit("invalid structure parameters")
    if args.structure_mode == "sana" and args.structure_checkpoint is None:
        raise SystemExit("--structure-mode sana requires --structure-checkpoint")
    if "normal" in tasks and args.normal_mode == "diode_dinov2" and args.normal_checkpoint is None:
        raise SystemExit("--normal-mode diode_dinov2 requires --normal-checkpoint")
    if "semantic" in tasks and (args.semantic_checkpoint is None or args.semantic_palette is None):
        raise SystemExit("semantic inference requires --semantic-checkpoint and --semantic-palette")
    input_path = args.input_image.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input image not found: {input_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        source = image.convert("RGB")
    output_resolution = args.resolution or max(source.size)
    outputs: list[tuple[str, Image.Image]] = []
    report: dict[str, object] = {
        "input": str(input_path),
        "output_dir": str(args.output_dir.resolve()),
        "tasks": list(tasks),
        "resolution": output_resolution,
        "semantic": {"status": "not_requested"},
    }

    depth = None
    if "depth" in tasks or ("normal" in tasks and args.normal_mode == "depth_derived"):
        device = select_device(args.device)
        model, processor = load_depth_estimator(args.depth_model, device)
        depth = resize_depth(estimate_depth(source, model, processor, device), output_resolution)
        report["depth_model"] = args.depth_model
        report["depth_device"] = str(device)

    if "depth" in tasks:
        depth_rgb = depth_to_rgb(normalize_depth_to_uint8(depth))
        depth_path = args.output_dir / f"{input_path.stem}_depth_pseudo.png"
        depth_rgb.save(depth_path)
        outputs.append(("Depth pseudo", depth_rgb))
        report["depth"] = {
            "output": str(depth_path.resolve()),
            "source": "depth_anything_v2_small",
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
        }

    if "normal" in tasks:
        if args.normal_mode == "depth_derived":
            normal = normal_from_relative_depth(depth, args.normal_strength)
            normal_rgb = normal_to_rgb(normal)
            normal_path = args.output_dir / f"{input_path.stem}_normal_from_depth.png"
            normal_rgb.save(normal_path)
            outputs.append(("Depth-derived normal", normal_rgb))
            report["normal"] = {
                "output": str(normal_path.resolve()),
                "source": "relative_depth_derived",
                "normal_strength": args.normal_strength,
                **angular_variation_degrees(normal),
            }

    # Release Depth-Anything before the substantially larger SANA structure path.
    if depth is not None and args.structure_mode == "sana":
        del model, processor, depth
        gc.collect()
        if select_device(args.device).type == "mps":
            torch.mps.empty_cache()

    if "structure" in tasks:
        if args.structure_mode == "sana":
            structure_raw, structure_rgb = infer_sana_structure(
                source,
                args.structure_checkpoint.expanduser().resolve(),
                select_device(args.device),
                output_resolution,
                args.structure_threshold,
                args.structure_inference_steps,
                args.structure_guidance_scale,
            )
            raw_path = args.output_dir / f"{input_path.stem}_structure_sana_raw.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            structure_raw.save(raw_path)
            structure_path = args.output_dir / f"{input_path.stem}_structure_sana.png"
            report["structure"] = {
                "output": str(structure_path.resolve()),
                "raw_output": str(raw_path.resolve()),
                "source": "sana_600m_binary_structure_lora",
                "checkpoint": str(args.structure_checkpoint.expanduser().resolve()),
                "resolution": 512,
                "inference_steps": args.structure_inference_steps,
                "guidance_scale": args.structure_guidance_scale,
                "threshold": args.structure_threshold,
                "validation": "32 val samples, 2px tolerance F1=0.7133 at threshold=0.30",
            }
        else:
            structure = build_binary_structure_map(input_path, output_resolution, args.min_line_fraction, args.edge_width)
            structure_rgb = Image.fromarray(structure, mode="RGB")
            structure_path = args.output_dir / f"{input_path.stem}_structure.png"
            report["structure"] = {
                "output": str(structure_path.resolve()),
                "source": "canny_lsd_rule",
                "positive_fraction": float((structure[..., 0] > 0).mean()),
                "min_line_fraction": args.min_line_fraction,
                "edge_width": args.edge_width,
            }
        structure_rgb.save(structure_path)
        outputs.append(("Structure", structure_rgb))

    dino_tasks = "semantic" in tasks or ("normal" in tasks and args.normal_mode == "diode_dinov2")
    if dino_tasks:
        device = select_device(args.device)
        if not args.dino_repository.is_dir() or not args.dino_weights.is_file():
            raise SystemExit("local DINOv2 repository or weights are missing")
        semantic_checkpoint = load_checkpoint(args.semantic_checkpoint, "semantic") if "semantic" in tasks else None
        normal_checkpoint = load_checkpoint(args.normal_checkpoint, "normal") if args.normal_mode == "diode_dinov2" and "normal" in tasks else None
        image_sizes = {
            checkpoint_image_size(checkpoint, label)
            for checkpoint, label in ((semantic_checkpoint, "semantic"), (normal_checkpoint, "normal"))
            if checkpoint is not None
        }
        if len(image_sizes) != 1:
            raise SystemExit("DINO semantic and normal checkpoints must use the same image size")
        dino_image_size = image_sizes.pop()
        backbone = LocalDINOv2ViTS14(args.dino_repository.resolve(), args.dino_weights.resolve()).to(device).eval()
        tokens = dino_features(source, dino_image_size, backbone, device)
        report["dino"] = {
            "backbone": "dinov2_vits14",
            "image_size": dino_image_size,
            "grid_size": dino_image_size // 14,
            "device": str(device),
            "local_files_only": True,
        }
        if semantic_checkpoint is not None:
            palette, _ = load_palette(args.semantic_palette)
            semantic_head = load_semantic_head(semantic_checkpoint, device)
            with torch.no_grad():
                labels = semantic_head(torch.from_numpy(tokens).to(device)).argmax(dim=-1).cpu().numpy()
            grid_size = dino_image_size // 14
            labels = labels.reshape(grid_size, grid_size)
            semantic_rgb = semantic_to_rgb(labels, palette).resize(
                (output_resolution, output_resolution), Image.Resampling.NEAREST
            )
            semantic_path = args.output_dir / f"{input_path.stem}_semantic_dinov2.png"
            semantic_rgb.save(semantic_path)
            outputs.append(("Semantic DINOv2", semantic_rgb))
            report["semantic"] = {
                "status": "enabled",
                "output": str(semantic_path.resolve()),
                "source": "frozen_dinov2_vits14_discrete_150_class_head",
                "checkpoint": str(args.semantic_checkpoint.expanduser().resolve()),
                "palette": str(args.semantic_palette.expanduser().resolve()),
            }
        if normal_checkpoint is not None:
            normal_head = load_normal_head(normal_checkpoint, device)
            with torch.no_grad():
                normal_tokens = normal_head(torch.from_numpy(tokens).to(device)).cpu().numpy()
            normal = expand_normal_grid(normal_tokens, dino_image_size)[0]
            normal_rgb = normal_to_diode_rgb(normal).resize((output_resolution, output_resolution), Image.Resampling.NEAREST)
            normal_path = args.output_dir / f"{input_path.stem}_normal_diode_dinov2.png"
            normal_rgb.save(normal_path)
            outputs.append(("DIODE DINOv2 normal", normal_rgb))
            report["normal"] = {
                "output": str(normal_path.resolve()),
                "source": "frozen_dinov2_vits14_diode_camera_normal_head",
                "checkpoint": str(args.normal_checkpoint.expanduser().resolve()),
                "normal_encoding": "normal = normalize(2 * rgb / 255 - 1)",
                **angular_variation_degrees(normal),
            }

    panel_path = args.output_dir / f"{input_path.stem}_specialists.png"
    write_panel(panel_path, source, outputs)
    report["panel"] = str(panel_path.resolve())
    (args.output_dir / "specialist_inference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
