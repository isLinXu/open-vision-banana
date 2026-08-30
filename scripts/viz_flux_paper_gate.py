#!/usr/bin/env python3
"""Visualize a FLUX.1-schnell paper-gate LoRA checkpoint.

Saves input | prediction | target panels for metric depth and Cityscapes
19-class segmentation, plus one text-to-image sample. Schnell inference is
4 steps at guidance 0. Do not run this while another MPS job is using the GPU.

    python scripts/viz_flux_paper_gate.py \
      --checkpoint output/flux_schnell_paper_gate_lr1e5_20260824/checkpoint-10
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
logger = logging.getLogger("viz_flux_paper_gate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("output/flux_schnell_paper_gate_lr1e5_20260824/checkpoint-10"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/flux_paper_gate_512"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-per-task", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
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
  <title>FLUX schnell paper-gate visualization</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #111; color: #eee; }}
    img {{ max-width: 100%; height: auto; }}
    h1, h2 {{ font-weight: 600; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>FLUX.1-schnell shared LoRA · checkpoint visualization</h1>
  <p>Input | prediction RGB | target RGB. Depth adds a cube-edge re-encode of the decoded metric map so it uses the same palette as the target. Schnell 4-step Euler flow, guidance 0. Not Table 1.</p>
  {rows}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (
        args.checkpoint.parent / f"viz_{args.checkpoint.name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading flux-schnell + LoRA from %s", args.checkpoint)
    model = VisionBanana.from_pretrained(
        model_name="flux-schnell",
        lora_path=str(args.checkpoint),
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
        if task == "depth" and result.get("depth") is not None:
            depth_rgb = RGBMapper(depth_colormap="cube_edge").encode(
                result["depth"], task="depth"
            )
            images.append(tensor_to_pil(depth_rgb))
            labels.append("prediction (cube-edge)")
        if target_pil is not None:
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
            guidance_scale=0.0,
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
    loss = ckpt_meta.get("loss")
    loss_text = f"{loss:.4f}" if isinstance(loss, (int, float)) else "unknown"
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "step": step,
                "train_loss": loss,
                "data_dir": str(args.data_dir.resolve()),
                "resolution": args.resolution,
                "num_inference_steps": args.num_inference_steps,
                "strength": args.strength,
                "guidance_scale": 0.0,
                "seed": args.seed,
                "n_per_task": args.n_per_task,
                "panels": [path.name for path in saved],
                "note": (
                    f"Paper-method gate visualization of {args.checkpoint.name} "
                    f"(train loss {loss_text}). Not Table 1."
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
