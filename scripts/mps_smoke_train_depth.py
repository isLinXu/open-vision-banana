"""MPS (Apple Silicon) 真实训练冒烟验证。

用本地 FLUX.1-schnell 权重 + vb_snvc_train_v3/depth（真实数据集）在
Apple Silicon 的 Metal (mps) 后端、float16 精度、关闭 AMP（绕开 MPS 不支持
bf16 autocast）的条件下，跑真实的 img2img LoRA flow 训练步，验证：
  1) 模型/VAE/T5 能在 mps 上加载与前向；
  2) 真实训练步能产出有限（finite）且下降的 loss；
  3) （best-effort）合并 LoRA 后做 img2img 推理，输出 depth 可视化三联图。

用法:
    PYTHONPATH=src python scripts/mps_smoke_train_depth.py
可调环境变量:
    VB_MODEL   FLUX 权重目录
    VB_DATA    depth 数据目录 (含 images/targets/prompts)
    VB_STEPS   训练步数 (默认 20)
    VB_RES     分辨率 (默认 256，mps 上更快)
"""
import os
import sys
import json
import traceback
from pathlib import Path

import torch

# ── 路径配置 ──────────────────────────────────────────────
MODEL = os.environ.get(
    "VB_MODEL",
    "/Users/gatilin/MyWork/models/ckpts/FLUX.1-schnell",
)
DEPTH = os.environ.get(
    "VB_DATA",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v3/depth",
)
N_STEPS = int(os.environ.get("VB_STEPS", "20"))
RES = int(os.environ.get("VB_RES", "256"))
OUT = Path("output/mps_smoke")
OUT.mkdir(parents=True, exist_ok=True)

# ── 设备选择：优先 mps，回退 cpu ─────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"[device] {DEVICE}  (mps_available={torch.backends.mps.is_available()})")
print(f"[model ] {MODEL}")
print(f"[data  ] {DEPTH}")
print(f"[steps ] {N_STEPS}  [res] {RES}")

# ── 导入项目模块 ────────────────────────────────────────
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import (
    InstructionTuner,
    VisionTaskDataset,
    _collate_fn,
)
from open_vision_banana.core.depth_mapper import DepthRGBMapper


def main():
    # 1) 数据集
    ds = VisionTaskDataset(data_path=DEPTH, resolution=RES, task="depth")
    print(f"[data  ] loaded {len(ds)} depth samples")
    if len(ds) == 0:
        raise RuntimeError("数据集为空，检查 images/targets/prompts 目录布局")
    s0 = ds[0]
    print(f"[data  ] sample keys={list(s0.keys())} "
          f"image={tuple(s0['image'].shape)} target={tuple(s0['target_rgb'].shape)}")

    # 2) 生成器 (mps + float16)
    #    MPS 不支持 bf16 autocast -> 用 float16（Metal 原生），并关闭 AMP。
    gen = BaseGenerator(MODEL, device=DEVICE, dtype="float16")
    print(f"[model ] family={gen.model_config.family} pipe={gen.model_config.pipeline_class}")

    # 3) Tuner (LoRA + img2img flow, flow_sigma=1.0 修复 R7)
    tuner = InstructionTuner(
        model=gen,
        task="depth",
        lora_rank=32,
        lora_alpha=32,
        use_rslora=True,
        flow_sigma=1.0,
        use_img2img_flow=True,
        use_ema=False,
        use_gradient_checkpointing=False,
        cfg_dropout_rate=0.1,
    )
    tuner.peft_model.train()
    n_params = sum(p.numel() for p in tuner.peft_model.parameters() if p.requires_grad)
    print(f"[lora  ] trainable LoRA params = {n_params:,}")

    # 4) 手动训练循环（直接调用 _flow_training_step，完全复现 train() 的核心计算，
    #    同时便于逐 step 记录 loss 与后续可视化）
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=0,
        pin_memory=False, collate_fn=_collate_fn,
    )
    opt = torch.optim.AdamW(tuner.peft_model.parameters(), lr=5e-5)

    losses = []
    it = iter(loader)
    print("\n[train ] starting real MPS training steps ...")
    for step in range(1, N_STEPS + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        opt.zero_grad(set_to_none=True)
        loss = tuner._training_step(batch)  # 内部含 VAE/t5/transformer(mps,fp16)，FLUX 走 _training_step_flow
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(tuner.peft_model.parameters(), 1.0,
                                           error_if_nonfinite=True)
        except FloatingPointError:
            print(f"[train ] step {step}: non-finite grad, skipping update (loss still recorded)")
            losses.append(loss.item())
            continue
        opt.step()
        losses.append(loss.item())
        print(f"[train ] step {step:>3}/{N_STEPS}  loss={loss.item():.6f}")
        sys.stdout.flush()

    avg_first = sum(losses[: max(1, N_STEPS // 3)]) / max(1, N_STEPS // 3)
    avg_last = sum(losses[-max(1, N_STEPS // 3):]) / max(1, N_STEPS // 3)
    trend = "DECREASING" if avg_last < avg_first else "NOT decreasing"
    print(f"\n[train ] avg first 1/3 = {avg_first:.6f} | avg last 1/3 = {avg_last:.6f} -> {trend}")

    # 保存 loss 曲线数据
    (OUT / "losses.json").write_text(json.dumps({
        "device": DEVICE, "model": MODEL, "steps": N_STEPS, "resolution": RES,
        "flow_sigma": 1.0, "losses": losses,
        "avg_first_third": avg_first, "avg_last_third": avg_last,
    }, indent=2))
    print(f"[out   ] losses -> {OUT / 'losses.json'}")

    # 保存 LoRA 权重（PEFT 标准格式），便于独立复用 / 后续评估
    lora_dir = OUT / "lora_adapter"
    lora_dir.mkdir(parents=True, exist_ok=True)
    tuner.peft_model.save_pretrained(lora_dir)
    print(f"[out   ] LoRA weights -> {lora_dir}")

    # 5) Best-effort 推理可视化（合并 LoRA -> img2img 推理 -> 解码 depth）
    try:
        do_inference_viz(tuner, gen, ds, OUT)
    except Exception as e:
        print(f"[viz   ] SKIPPED (推理可视化失败，不影响训练验证结论): {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n[done  ] MPS 训练验证完成。")


def do_inference_viz(tuner, gen, ds, out_dir: Path):
    """合并 LoRA 到 transformer，对单张样本做 img2img 推理，输出 depth 三联图。"""
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    mapper = DepthRGBMapper()  # 默认 cube_edge，与训练编码一致
    sample = ds[0]
    inp = sample["image"]  # [-1,1]
    tgt = sample["target_rgb"]  # [0,1]

    # 合并 LoRA
    merged = tuner.peft_model.merge_and_unload()
    merged = merged.to(DEVICE)
    gen._t2i_pipe.transformer = merged
    if getattr(gen, "_i2i_pipe", None) is not None:
        gen._i2i_pipe.transformer = merged

    # 输入图 -> PIL
    inp_pil = Image.fromarray(
        (inp.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy()
    )

    out = gen.generate(image=inp_pil, strength=0.85, num_inference_steps=4,
                       prompt=sample.get("prompt", "depth") or "depth")
    # generate 可能返回 tensor [B,3,H,W] 或 PIL.Image
    if isinstance(out, torch.Tensor):
        pred_rgb = out[0].clamp(0, 1)
    elif isinstance(out, (list, tuple)):
        img0 = out[0]
        if isinstance(img0, Image.Image):
            pred_rgb = torch.from_numpy(np.array(img0.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)
        else:
            pred_rgb = torch.as_tensor(img0)
    else:
        pred_rgb = torch.from_numpy(np.array(out.convert("RGB"), np.float32) / 255.0).permute(2, 0, 1)

    # 解码 depth 三联图
    inp_img = (inp.clamp(-1, 1) * 0.5 + 0.5)
    tgt_depth = mapper.rgb_to_depth(tgt.unsqueeze(0)).squeeze()
    pred_depth = mapper.rgb_to_depth(pred_rgb.unsqueeze(0)).squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(inp_img.permute(1, 2, 0).numpy())
    axes[0].set_title("Input (RGB)")
    axes[1].imshow(tgt_depth.cpu().numpy(), cmap="viridis")
    axes[1].set_title("Target depth (GT RGB->depth)")
    axes[2].imshow(pred_depth.cpu().numpy(), cmap="viridis")
    axes[2].set_title("Pred depth (LoRA img2img)")
    for a in axes:
        a.axis("off")
    fig.suptitle(f"MPS FLUX LoRA depth - step {N_STEPS} (device={DEVICE})")
    fig_path = out_dir / "mps_smoke_depth.png"
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    print(f"[viz   ] 三联图 -> {fig_path}")
    print(f"[viz   ] target depth finite={not torch.isnan(tgt_depth).any()} "
          f"| pred depth finite={not torch.isnan(pred_depth).any()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
