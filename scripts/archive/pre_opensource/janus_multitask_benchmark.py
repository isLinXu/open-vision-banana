#!/usr/bin/env python3
"""Janus-1.3B multi-task benchmark: evaluate every task."""
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
from open_vision_banana.training.tuner import VisionTaskDataset

MODEL_ID = os.environ.get("VB_MODEL", "deepseek-ai/Janus-1.3B")
DATA_ROOT = os.environ.get(
    "VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3",
)
CKPT_DIR = os.environ.get("VB_CKPT", "")
TASKS = os.environ.get("VB_TASKS", "depth,normal,instance,referring,keypoint").split(",")
RES = int(os.environ.get("VB_RES", "384"))
SAMPLES_PER_TASK = int(os.environ.get("VB_SAMPLES", "2"))
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "janus_multitask_benchmark")))
OUT.mkdir(parents=True, exist_ok=True)

DTYPE = torch.float32
IMAGE_TOKEN_NUM = 576

TASK_DEFAULT_PROMPTS = {
    "depth": "Generate a metric depth visualization of this image.",
    "normal": "Generate a surface normal estimation visualization of this image.",
    "instance": "Generate an instance segmentation visualization of this image.",
    "referring": "Generate a referring expression segmentation visualization of this image.",
    "keypoint": "Generate a keypoint detection visualization of this image.",
}


def _resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    device = _resolve_device()
    print(f"[cfg] model={MODEL_ID}  device={device}")
    print(f"[cfg] ckpt={CKPT_DIR}  tasks={TASKS}  samples={SAMPLES_PER_TASK}")

    if not CKPT_DIR or not Path(CKPT_DIR).exists():
        raise RuntimeError(f"Checkpoint not found: {CKPT_DIR}")

    # load the model
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

    # load the LoRA
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    ckpt = Path(CKPT_DIR)
    adapter_dir = ckpt / "adapter"
    if adapter_dir.exists():
        adapter_state = load_peft_weights(str(adapter_dir), device=device)
        set_peft_model_state_dict(model.language_model, adapter_state)
        print(f"[ckpt] loaded adapter from {adapter_dir}")

    # load the datasets
    mapper = DepthRGBMapper(colormap="viridis")
    all_results = {}

    @torch.no_grad()
    def infer_task(task: str, sample_idx: int) -> dict:
        ds = VisionTaskDataset(data_path=str(Path(DATA_ROOT) / task), resolution=RES, task=task)
        s = ds[sample_idx]
        inp_tensor = s["image"]  # [-1, 1]
        tgt_rgb = s["target_rgb"]  # [0, 1]
        prompt = s.get("prompt", TASK_DEFAULT_PROMPTS.get(task, ""))
        pil_in = Image.fromarray(
            (inp_tensor.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5)
            .mul(255).byte().numpy()
        )

        # inference
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
        gen_logits = model.gen_head(gen_hidden)
        pred_tokens = torch.argmax(gen_logits, dim=-1)

        # decode
        pred_tokens_flat = pred_tokens.squeeze(0)
        dec = model.gen_vision_model.decode_code(
            pred_tokens_flat,
            shape=[1, 8, 24, 24],
        )
        pred_rgb = dec.squeeze(0).clamp(0, 1).cpu()

        # triptych
        panel = Image.new("RGB", (RES * 3, RES))
        panel.paste(pil_in.resize((RES, RES)), (0, 0))
        panel.paste(Image.fromarray(
            (pred_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES, 0))
        panel.paste(Image.fromarray(
            (tgt_rgb.permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy())
        ).resize((RES, RES)), (RES * 2, 0))
        panel.save(OUT / f"panel_{task}_{sample_idx:02d}.png")

        rgb_mae = (pred_rgb.float() - tgt_rgb.float()).abs().mean().item()
        return {"idx": sample_idx, "rgb_mae": rgb_mae}

    for task in TASKS:
        print(f"\n[eval] task={task}:")
        task_results = []
        for idx in range(SAMPLES_PER_TASK):
            r = infer_task(task, idx)
            task_results.append(r)
            print(f"  idx={idx:>2}  rgb_mae={r['rgb_mae']:.4f}")
        avg_mae = sum(r["rgb_mae"] for r in task_results) / len(task_results)
        all_results[task] = {"avg_rgb_mae": avg_mae, "samples": task_results}
        print(f"  → avg rgb_mae={avg_mae:.4f}")

    overall_mae = sum(r["avg_rgb_mae"] for r in all_results.values()) / len(all_results)
    print(f"\n[bench] overall avg rgb_mae={overall_mae:.4f}")

    (OUT / "benchmark_result.json").write_text(json.dumps({
        "model": MODEL_ID, "ckpt": CKPT_DIR, "device": device,
        "tasks": TASKS, "samples_per_task": SAMPLES_PER_TASK,
        "per_task": all_results, "overall_rgb_mae": overall_mae,
    }, indent=2))
    print(f"[out] → {OUT / 'benchmark_result.json'}")


if __name__ == "__main__":
    main()
