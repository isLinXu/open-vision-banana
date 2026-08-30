#!/usr/bin/env python3
"""Qualitative NYU Depth viz for a SANA LoRA checkpoint.

Panels: input | pred cube-edge RGB | pred R/B amp | pred turbo | GT turbo.
Also writes per-sample Eigen+10m AbsRel / δ1 next to the aggregate.

Do not run while another MPS job is using the GPU.

    PYTHONPATH=src python scripts/viz_sana_nyu_depth.py \
      --checkpoint output/sana_hypersim_metric_depth_lr1e5_20260822/checkpoint-500 \
      --nyu-root data/benchmarks/nyu/extracted \
      --max-samples 8
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
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from open_vision_banana.core.rgb_mapper import RGBMapper  # noqa: E402
from open_vision_banana.data.datasets import NYUDepthDataset  # noqa: E402
from open_vision_banana.evaluation.benchmark import (  # noqa: E402
    DepthAccumulator,
    apply_depth_benchmark_protocol,
)
from open_vision_banana.inference.pipeline import VisionBanana  # noqa: E402
from viz_sana_paper_gate import (  # noqa: E402
    _as_hw_depth,
    amplify_cube_edge_rb,
    decode_cube_edge_metres,
    make_panel,
    metric_turbo_pil,
    resolve_lora_path,
    tensor_to_pil,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("viz_sana_nyu_depth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nyu-root", type=Path, default=Path("data/benchmarks/nyu/extracted"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def write_index(output_dir: Path, panels: list[Path], summary: dict) -> None:
    metrics = summary.get("aggregate", {})
    rows = "\n".join(
        f'<section><h2>{path.stem}</h2><img src="{path.name}" alt="{path.stem}" /></section>'
        for path in panels
    )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>SANA NYU depth visualization</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #111; color: #eee; }}
    img {{ max-width: 100%; height: auto; }}
    h1, h2 {{ font-weight: 600; }}
    section {{ margin-bottom: 32px; }}
    code {{ color: #9cf; }}
  </style>
</head>
<body>
  <h1>Hypersim-500 · NYU qualitative depth gate</h1>
  <p>Panels: input | pred cube-edge | pred R/B amp | pred metric turbo | GT metric turbo.
  Protocol: Eigen crop + 10m cap. Aggregate AbsRel={metrics.get("AbsRel")} δ1={metrics.get("delta_1.25")}
  (n={summary.get("n_samples")}). Not Table 1.</p>
  {rows}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (
        args.checkpoint.parent / f"viz_{args.checkpoint.name}_nyu_n{args.max_samples}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lora_path = resolve_lora_path(args.checkpoint)
    logger.info("Loading sana-600m-512px + LoRA from %s", lora_path)
    model = VisionBanana.from_pretrained(
        model_name="sana-600m-512px",
        lora_path=str(lora_path),
        device=args.device,
        dtype=_dtype(args.dtype),
    )
    depth_mapper = RGBMapper.paper_reproduction()
    dataset = NYUDepthDataset(
        root_dir=str(args.nyu_root),
        split="val",
        resolution=args.resolution,
    )
    n_total = len(dataset)
    if n_total == 0:
        raise SystemExit(f"No NYU samples under {args.nyu_root}")
    indices = list(range(args.start_index, min(args.start_index + args.max_samples, n_total)))

    panels: list[Path] = []
    per_sample: list[dict] = []
    accumulator = DepthAccumulator()

    for rank, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        name = Path(dataset.samples[idx]["image"]).stem
        logger.info("[%d/%d] %s", rank + 1, len(indices), name)

        result = model.predict(
            image=image.to(args.device),
            task="depth",
            num_inference_steps=args.num_inference_steps,
            strength=args.strength,
            height=args.resolution,
            width=args.resolution,
            seed=args.seed + rank,
        )
        pred_pil = tensor_to_pil(result["rgb_output"])
        pred_metres = result.get("depth")
        if pred_metres is None:
            pred_metres = decode_cube_edge_metres(pred_pil, depth_mapper)
        else:
            pred_metres = _as_hw_depth(pred_metres)

        gt_hw = _as_hw_depth(sample["gt_depth"])
        mask_hw = sample["valid_mask"]
        if isinstance(mask_hw, torch.Tensor):
            mask_hw = mask_hw.detach().bool().cpu()
            while mask_hw.dim() > 2:
                mask_hw = mask_hw[0]
        else:
            mask_hw = torch.as_tensor(mask_hw).bool()

        pred_p, gt_p, mask_p = apply_depth_benchmark_protocol(
            "nyu", pred_metres, gt_hw, mask_hw
        )
        sample_acc = DepthAccumulator()
        sample_acc.update(pred_p.unsqueeze(0), gt_p.unsqueeze(0), mask_p.unsqueeze(0))
        sample_metrics = sample_acc.compute()
        accumulator.update(pred_p.unsqueeze(0), gt_p.unsqueeze(0), mask_p.unsqueeze(0))
        per_sample.append(
            {
                "index": idx,
                "name": name,
                "AbsRel": sample_metrics.get("AbsRel"),
                "delta_1.25": sample_metrics.get("delta_1.25"),
                "valid_pixels": sample_metrics.get("valid_pixels"),
            }
        )

        input_pil = tensor_to_pil(image)
        d1 = float(sample_metrics.get("delta_1.25", float("nan")))
        ar = float(sample_metrics.get("AbsRel", float("nan")))
        panel = make_panel(
            [
                input_pil,
                pred_pil,
                amplify_cube_edge_rb(pred_pil),
                metric_turbo_pil(pred_metres, valid_max=10.0),
                metric_turbo_pil(gt_hw, valid_max=10.0),
            ],
            [
                "input",
                "pred cube-edge",
                "pred R/B amp",
                f"pred turbo δ1={d1:.3f}",
                "GT turbo",
            ],
            title=f"NYU {name} · AbsRel={ar:.3f}",
        )
        out = output_dir / f"nyu_{idx:04d}_{name}_panel.png"
        panel.save(out)
        panels.append(out)
        pred_pil.save(output_dir / f"nyu_{idx:04d}_{name}_pred.png")
        metric_turbo_pil(pred_metres, valid_max=10.0).save(
            output_dir / f"nyu_{idx:04d}_{name}_pred_turbo.png"
        )
        metric_turbo_pil(gt_hw, valid_max=10.0).save(
            output_dir / f"nyu_{idx:04d}_{name}_gt_turbo.png"
        )

    aggregate = accumulator.compute()
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "nyu_root": str(args.nyu_root.resolve()),
        "n_samples": len(indices),
        "start_index": args.start_index,
        "protocol": "Eigen crop + 10m cap",
        "aggregate": {
            k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
            for k, v in aggregate.items()
        },
        "per_sample": per_sample,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_index(output_dir, panels, summary)
    logger.info(
        "Aggregate n=%d AbsRel=%.4f δ1=%.4f → %s",
        len(indices),
        float(aggregate.get("AbsRel", float("nan"))),
        float(aggregate.get("delta_1.25", float("nan"))),
        output_dir / "index.html",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
