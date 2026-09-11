#!/usr/bin/env python3
"""Visualize a SANA paper-gate LoRA checkpoint.

Saves input | prediction | target panels for metric depth and Cityscapes
19-class segmentation, plus one text-to-image sample. SANA uses 20 steps.
Do not run this while another MPS job is using the GPU.

    python scripts/viz_sana_paper_gate.py \
      --checkpoint output/sana_paper_gate_from_hypersim500_20260828/checkpoint-250
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.rgb_mapper import RGBMapper  # noqa: E402
from open_vision_banana.data.datasets import CityscapesSegDataset  # noqa: E402
from open_vision_banana.inference.pipeline import VisionBanana  # noqa: E402
from open_vision_banana.training.tuner import VisionTaskDataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viz_sana_paper_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/sana_paper_gate_from_hypersim500_20260828/checkpoint-250"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/flux_paper_gate_512"),
        help="v3-style depth/segmentation/generation tree for panel sources",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-per-task", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.dim() == 4:
        tensor = tensor[0]
    tensor = tensor.detach().float().cpu()
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if float(tensor.min()) < 0:
        tensor = (tensor + 1.0) / 2.0
    array = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8))


def load_rgb(path: Path, resolution: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)


def pil_to_model_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0


def amplify_cube_edge_rb(image: Image.Image) -> Image.Image:
    """Stretch R/B on cube-edge targets so green-face geometry is visible.

    Indoor metric depths often land on the G=255 cube edge; structure lives in
    R/B and looks like a flat green slab to the naked eye.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32)

    def _stretch(channel: np.ndarray) -> np.ndarray:
        lo = float(channel.min())
        hi = float(channel.max())
        if hi - lo < 1.0:
            return np.full_like(channel, 128.0)
        return (channel - lo) / (hi - lo) * 255.0

    amplified = np.stack(
        [_stretch(array[:, :, 0]), np.full_like(array[:, :, 1], 160.0), _stretch(array[:, :, 2])],
        axis=-1,
    )
    return Image.fromarray(amplified.astype(np.uint8))


def _as_hw_depth(depth: torch.Tensor | np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(depth).detach().float().cpu()
    while tensor.dim() > 2:
        tensor = tensor[0]
    return tensor


def metric_turbo_pil(depth: torch.Tensor | np.ndarray, *, valid_max: float | None = None) -> Image.Image:
    """Diagnostic metre-map viz (turbo). Not the training cube-edge target."""
    from open_vision_banana.core.depth_mapper import DepthRGBMapper

    metres = _as_hw_depth(depth)
    finite = torch.isfinite(metres) & (metres > 0)
    if not bool(finite.any()):
        return Image.new("RGB", (metres.shape[-1], metres.shape[-2]), (40, 40, 40))
    capped = metres.clone()
    hi = float(valid_max) if valid_max is not None else float(metres[finite].quantile(0.98))
    hi = max(hi, 1e-3)
    capped = capped.clamp(0.0, hi)
    capped[~finite] = 0.0
    # DepthRGBMapper still applies the paper power transform; for display we want
    # a linear metre → colormap path, so normalize to [0,1] first and encode via
    # normalized_to_rgb with turbo keypoints.
    mapper = DepthRGBMapper(colormap="turbo")
    normalized = (capped / hi).clamp(0.0, 1.0)
    rgb = mapper.normalized_to_rgb(normalized)
    return tensor_to_pil(rgb)


def decode_cube_edge_metres(rgb_pil: Image.Image, mapper: RGBMapper) -> torch.Tensor:
    array = np.asarray(rgb_pil.convert("RGB"), dtype=np.float32) / 255.0
    rgb = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    depth = mapper.decode(rgb, task="depth")
    return _as_hw_depth(depth)


def make_panel(images: list[Image.Image], labels: list[str], title: str) -> Image.Image:
    width, height = images[0].size
    label_h = 28
    title_h = 32
    canvas = Image.new("RGB", (width * len(images), height + label_h + title_h), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill=(240, 240, 240))
    for index, (image, label) in enumerate(zip(images, labels)):
        left = index * width
        draw.text((left + 8, title_h + 6), label, fill=(220, 220, 220))
        canvas.paste(image, (left, title_h + label_h))
    return canvas


def write_index(output_dir: Path, panels: list[Path]) -> None:
    rows = "\n".join(
        f'<section><h2>{path.stem}</h2><img src="{path.name}" alt="{path.stem}" /></section>'
        for path in panels
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>SANA paper-gate visualization</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #111; color: #eee; }}
    img {{ max-width: 100%; height: auto; }}
    h1, h2 {{ font-weight: 600; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>SANA-600M shared LoRA · checkpoint visualization</h1>
  <p>Depth panels: input | prediction RGB | prediction cube-edge re-encode | prediction metric (turbo, diagnostic) | target cube-edge | target R/B amplified | target metric (turbo, diagnostic). Indoor cube-edge targets often sit on the green face; R/B amp and turbo metric are for human inspection only. 20-step SANA img2img. Not Table 1.</p>
  {rows}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def resolve_lora_path(checkpoint: Path) -> Path:
    adapter = checkpoint / "lora_adapter"
    if adapter.is_dir():
        return adapter
    return checkpoint


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (args.checkpoint.parent / f"viz_{args.checkpoint.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_path = resolve_lora_path(args.checkpoint)
    logger.info("Loading sana-600m-512px + LoRA from %s", lora_path)
    model = VisionBanana.from_pretrained(
        model_name="sana-600m-512px",
        lora_path=str(lora_path),
        device=args.device,
        dtype=_dtype(args.dtype),
    )

    jobs: list[tuple[str, str, Path, Path | None]] = []
    depth = VisionTaskDataset(str(args.data_dir / "depth"), resolution=args.resolution, task="depth")
    semantic = VisionTaskDataset(
        str(args.data_dir / "segmentation"),
        resolution=args.resolution,
        task="semantic",
    )
    for sample in depth.samples[: args.n_per_task]:
        jobs.append(
            (
                "depth",
                sample["prompt"],
                Path(sample["image_path"]),
                Path(sample["target_path"]) if sample.get("target_path") else None,
            )
        )
    for sample in semantic.samples[: args.n_per_task]:
        jobs.append(
            (
                "semantic",
                sample["prompt"],
                Path(sample["image_path"]),
                Path(sample["target_path"]) if sample.get("target_path") else None,
            )
        )

    depth_mapper = RGBMapper.paper_reproduction()
    saved: list[Path] = []
    for task, prompt, image_path, target_path in jobs:
        input_pil = load_rgb(image_path, args.resolution)
        target_pil = (
            load_rgb(target_path, args.resolution)
            if target_path is not None and target_path.is_file()
            else None
        )
        logger.info("Infer %s %s", task, image_path.name)
        kwargs = {
            "image": pil_to_model_tensor(input_pil).to(args.device),
            "task": task,
            "prompt": prompt,
            "num_inference_steps": args.num_inference_steps,
            "strength": args.strength,
            "height": args.resolution,
            "width": args.resolution,
            "seed": args.seed,
        }
        if task == "semantic":
            kwargs["class_color_map"] = CityscapesSegDataset.CITYSCAPES_COLORS
        result = model.predict(**kwargs)
        pred_pil = tensor_to_pil(result["rgb_output"])
        labels = ["input", "prediction"]
        images = [input_pil, pred_pil]
        if task == "depth":
            pred_metres = result.get("depth")
            if pred_metres is None:
                pred_metres = decode_cube_edge_metres(pred_pil, depth_mapper)
            else:
                pred_metres = _as_hw_depth(pred_metres)
            depth_rgb = depth_mapper.encode(pred_metres, task="depth")
            images.append(tensor_to_pil(depth_rgb))
            labels.append("prediction (cube-edge)")
            target_metres = None
            if target_pil is not None:
                target_metres = decode_cube_edge_metres(target_pil, depth_mapper)
                metric_hi = float(target_metres[torch.isfinite(target_metres)].quantile(0.98))
            else:
                finite = pred_metres[torch.isfinite(pred_metres) & (pred_metres > 0)]
                metric_hi = float(finite.quantile(0.98)) if finite.numel() else 10.0
            images.append(metric_turbo_pil(pred_metres, valid_max=metric_hi))
            labels.append("prediction (metric turbo)")
            if target_pil is not None:
                images.append(target_pil)
                labels.append("target (cube-edge)")
                images.append(amplify_cube_edge_rb(target_pil))
                labels.append("target (R/B amp)")
                images.append(metric_turbo_pil(target_metres, valid_max=metric_hi))
                labels.append("target (metric turbo)")
                amplify_cube_edge_rb(target_pil).save(
                    output_dir / f"{task}_{image_path.stem}_target_rb_amp.png"
                )
                metric_turbo_pil(target_metres, valid_max=metric_hi).save(
                    output_dir / f"{task}_{image_path.stem}_target_metric_turbo.png"
                )
            metric_turbo_pil(pred_metres, valid_max=metric_hi).save(
                output_dir / f"{task}_{image_path.stem}_pred_metric_turbo.png"
            )
        elif target_pil is not None:
            images.append(target_pil)
            labels.append("target")
        panel = make_panel(images, labels, f"{task} · {image_path.name} · {args.checkpoint.name}")
        out_path = output_dir / f"{task}_{image_path.stem}_panel.png"
        panel.save(out_path)
        pred_pil.save(output_dir / f"{task}_{image_path.stem}_pred.png")
        saved.append(out_path)
        logger.info("Wrote %s", out_path)

    prompts_file = args.data_dir / "generation" / "prompts.jsonl"
    if prompts_file.is_file():
        first = json.loads(prompts_file.read_text().splitlines()[0])
        logger.info("Infer generation: %s", first["prompt"][:80])
        rgb = model.generator.generate(
            prompt=first["prompt"],
            image=None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.resolution,
            width=args.resolution,
            seed=args.seed,
        )
        pred_pil = tensor_to_pil(rgb)
        ref = args.data_dir / "generation" / "images" / first["image"]
        images = [pred_pil]
        labels = ["prediction"]
        if ref.is_file():
            images.append(load_rgb(ref, args.resolution))
            labels.append("reference caption image")
        panel = make_panel(
            images,
            labels,
            f"generation · {args.checkpoint.name} · {first['prompt'][:80]}",
        )
        out_path = output_dir / "generation_panel.png"
        panel.save(out_path)
        pred_pil.save(output_dir / "generation_pred.png")
        saved.append(out_path)

    write_index(output_dir, saved)
    ckpt_meta_path = args.checkpoint / "metadata.json"
    ckpt_meta = {}
    if ckpt_meta_path.is_file():
        ckpt_meta = json.loads(ckpt_meta_path.read_text(encoding="utf-8"))
    step = ckpt_meta.get("step", args.checkpoint.name)
    loss = ckpt_meta.get("loss", ckpt_meta.get("val_loss"))
    loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "unknown"
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "lora_path": str(lora_path.resolve()),
                "step": step,
                "train_loss": loss,
                "data_dir": str(args.data_dir.resolve()),
                "resolution": args.resolution,
                "num_inference_steps": args.num_inference_steps,
                "strength": args.strength,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "n_per_task": args.n_per_task,
                "panels": [path.name for path in saved],
                "note": (
                    f"SANA paper-method gate visualization of {args.checkpoint.name} "
                    f"(train loss {loss_text}). Depth panels include cube-edge target, "
                    "R/B-amplified target, and turbo metric diagnostics. Not Table 1."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Gallery: %s", output_dir / "index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
