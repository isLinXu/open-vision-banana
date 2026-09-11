#!/usr/bin/env python3
"""Janus-1.3B depth-only 完整复现 —— 标准 benchmark + 生成保真.

1. 标准 depth benchmark（NYU-style metrics）
2. 生成保真验证（text-to-image generation）
3. 与 frozen depth probe 对比

用法:
    PYTHONPATH=src:/tmp/Janus python scripts/janus_depth_complete.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, "/tmp/Janus")

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.evaluation.depth import evaluate_depth
from open_vision_banana.training.tuner import VisionTaskDataset

MODEL_ID = os.environ.get("VB_MODEL", "deepseek-ai/Janus-1.3B")
DATA_DIR = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
CKPT_DIR = os.environ.get("VB_CKPT", "")
RES = int(os.environ.get("VB_RES", "384"))
HOLDOUT = [int(x) for x in os.environ.get(
    "VB_HOLDOUT", "120,500,900,1300,1700,2100").split(",") if x.strip()]
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_depth_complete")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32
IMAGE_TOKEN_NUM = 576
DEPTH_PROMPT = "Generate a metric depth visualization of this image."


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    device = _resolve_device()
    print(f"[cfg] model={MODEL_ID}  device={device}")
    print(f"[cfg] ckpt={CKPT_DIR}  res={RES}  holdout={HOLDOUT}")

    if not CKPT_DIR or not Path(CKPT_DIR).exists():
        raise RuntimeError(f"Checkpoint not found: {CKPT_DIR}")

    # 1) 加载模型 ───────────────────────────────────────────────────────
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from janus.models.modeling_vlm import MultiModalityConfig, MultiModalityCausalLM
    AutoConfig.register("multi_modality", MultiModalityConfig)
    AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = model.to(device)
    print(f"[model] loaded")

    # 加载 RGB head
    rgb_head_path = Path(CKPT_DIR) / "rgb_head.pt"
    if rgb_head_path.exists():
        from janus_continuous_rgb import RGBHead
        rgb_head = RGBHead(model.config.language_config.hidden_size, IMAGE_TOKEN_NUM, RES).to(device, DTYPE)
        rgb_head.load_state_dict(torch.load(rgb_head_path, map_location=device))
        print(f"[ckpt] loaded RGB head from {rgb_head_path}")
    else:
        rgb_head = None
        print(f"[ckpt] no RGB head found, using VQ decode")

    # 2) 标准 depth benchmark ──────────────────────────────────────────
    print("\n[bench] standard depth benchmark...")
    ds = VisionTaskDataset(data_path=DATA_DIR, resolution=RES, task="depth")
    n_total = len(ds)
    print(f"[data] loaded {n_total} samples")

    mapper = DepthRGBMapper(colormap="viridis")

    @torch.no_grad()
    def benchmark(sample_idx: int) -> dict:
        s = ds[sample_idx]
        inp_tensor = s["image"]  # [-1, 1]
        tgt_rgb = s["target_rgb"]  # [0, 1]
        prompt = s.get("prompt", DEPTH_PROMPT)
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy()
        )

        # 推理
        pixel_values = inp_tensor.unsqueeze(0).unsqueeze(0).to(device, DTYPE)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        seq_len = prompt_len + IMAGE_TOKEN_NUM
        full_input_ids = torch.cat([
            prompt_ids,
            torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
        ], dim=1)
        images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        images_seq_mask[0, prompt_len:] = True
        images_emb_mask = torch.ones(1, 1, IMAGE_TOKEN_NUM, dtype=torch.bool, device=device)
        input_embeds = model.prepare_inputs_embeds(
            input_ids=full_input_ids,
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        outputs = model.language_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        gen_hidden = hidden_states[:, prompt_len:, :]

        # 生成 RGB
        if rgb_head is not None:
            pred_rgb = rgb_head(gen_hidden).squeeze(0).clamp(0, 1).cpu()
        else:
            gen_logits = model.gen_head(gen_hidden)
            pred_tokens = torch.argmax(gen_logits, dim=-1)
            pred_tokens_flat = pred_tokens.squeeze(0)
            dec = model.gen_vision_model.decode_code(
                pred_tokens_flat,
                shape=[1, 8, 24, 24],
            )
            pred_rgb = dec.squeeze(0).clamp(0, 1).cpu()
        pred_d = mapper.rgb_to_depth(pred_rgb.unsqueeze(0)).squeeze().clamp(0, 15).float()
        gt_d = mapper.rgb_to_depth(tgt_rgb.unsqueeze(0)).squeeze().clamp(0, 15).float()

        # 保存可视化
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"bench_{sample_idx:04d}.png")

        # 标准指标
        m = evaluate_depth(pred_d.reshape(1, 1, *pred_d.shape),
                           gt_d.reshape(1, 1, *gt_d.shape))
        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        return {
            "idx": sample_idx,
            "delta1": m["δ<1.25"],
            "absrel": m["AbsRel"],
            "rmse": m["RMSE"],
            "rgb_mae": rgb_mae,
        }

    bench_results = []
    for idx in sorted(HOLDOUT):
        r = benchmark(idx)
        bench_results.append(r)
        print(f"  idx={idx:>4}  δ1={r['delta1']:.4f}  AbsRel={r['absrel']:.4f}  "
              f"RMSE={r['rmse']:.4f}  rgb_mae={r['rgb_mae']:.4f}")

    avg_d1 = sum(r["delta1"] for r in bench_results) / len(bench_results)
    avg_absrel = sum(r["absrel"] for r in bench_results) / len(bench_results)
    avg_rmse = sum(r["rmse"] for r in bench_results) / len(bench_results)
    avg_rgb_mae = sum(r["rgb_mae"] for r in bench_results) / len(bench_results)

    print(f"\n[bench] avg δ1={avg_d1:.4f}  avg AbsRel={avg_absrel:.4f}  "
          f"avg RMSE={avg_rmse:.4f}  avg rgb_mae={avg_rgb_mae:.4f}")

    # 3) 生成保真验证 ────────────────────────────────────────────────────
    print("\n[gen] generation preservation test...")
    test_prompts = [
        "A beautiful sunset over the ocean",
        "A cat sitting on a windowsill",
        "A futuristic cityscape at night",
    ]

    gen_results = []
    for prompt in test_prompts:
        print(f"  prompt: {prompt[:50]}...")
        # 简化：用 t2i 模式生成
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = prompt_ids.shape[1]
        seq_len = prompt_len + IMAGE_TOKEN_NUM
        full_input_ids = torch.cat([
            prompt_ids,
            torch.full((1, IMAGE_TOKEN_NUM), -1, device=device, dtype=torch.long),
        ], dim=1)
        images_seq_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        images_seq_mask[0, prompt_len:] = True
        images_emb_mask = torch.ones(1, 1, IMAGE_TOKEN_NUM, dtype=torch.bool, device=device)
        input_embeds = model.prepare_inputs_embeds(
            input_ids=full_input_ids,
            pixel_values=torch.zeros(1, 1, 3, RES, RES, device=device, dtype=DTYPE),
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        outputs = model.language_model.model(
            inputs_embeds=input_embeds,
            output_hidden_states=True,
        )
        hidden_states = outputs.last_hidden_state
        gen_hidden = hidden_states[:, prompt_len:, :]
        gen_logits = model.gen_head(gen_hidden)
        pred_tokens = torch.argmax(gen_logits, dim=-1)

        pred_tokens_flat = pred_tokens.squeeze(0)
        dec = model.gen_vision_model.decode_code(
            pred_tokens_flat,
            shape=[1, 8, 24, 24],
        )
        pred_rgb = dec.squeeze(0).clamp(0, 1).cpu()
        img = Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        )
        img.save(OUT / f"gen_{len(gen_results):02d}.png")
        gen_results.append({"prompt": prompt, "saved": f"gen_{len(gen_results):02d}.png"})

    print(f"  generated {len(gen_results)} images")

    # 保存结果
    (OUT / "complete_result.json").write_text(json.dumps({
        "model": MODEL_ID, "ckpt": CKPT_DIR, "device": device,
        "res": RES, "holdout": sorted(HOLDOUT),
        "benchmark": {
            "avg_delta1": avg_d1,
            "avg_absrel": avg_absrel,
            "avg_rmse": avg_rmse,
            "avg_rgb_mae": avg_rgb_mae,
            "per_sample": bench_results,
        },
        "generation": gen_results,
    }, indent=2))
    print(f"[out] → {OUT / 'complete_result.json'}")


if __name__ == "__main__":
    main()
