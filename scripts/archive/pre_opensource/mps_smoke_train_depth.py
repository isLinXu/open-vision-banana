"""Real-training smoke check on MPS (Apple Silicon).

Uses local FLUX.1-schnell weights + vb_snvc_train_v3/depth (a real dataset) on
the Apple Silicon Metal (mps) backend, in float16 with AMP off (MPS does not support
bf16 autocast), running real img2img LoRA flow training steps to check that:
  1) the model/VAE/T5 load and run forward on mps;
  2) real training steps give a finite, decreasing loss;
  3) (best effort) after merging the LoRA, img2img inference produces a depth visualization triptych.

Usage:
    PYTHONPATH=src python scripts/mps_smoke_train_depth.py
Environment variables:
    VB_MODEL   FLUX weights directory
    VB_DATA    depth data directory (with images/targets/prompts)
    VB_STEPS   training steps (default 20)
    VB_RES     resolution (default 256, faster on mps)
"""
import os
import sys
import json
import traceback
from pathlib import Path

import torch

# ── Paths ─────────────────────────────────────────────────
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

# ── Device: prefer mps, fall back to cpu ────────────────
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

# ── Project imports ─────────────────────────────────────
from open_vision_banana.inference.generator import BaseGenerator
from open_vision_banana.training.tuner import (
    InstructionTuner,
    VisionTaskDataset,
    _collate_fn,
)
from open_vision_banana.core.depth_mapper import DepthRGBMapper


def main():
    # 1) dataset
    ds = VisionTaskDataset(data_path=DEPTH, resolution=RES, task="depth")
    print(f"[data  ] loaded {len(ds)} depth samples")
    if len(ds) == 0:
        raise RuntimeError("The dataset is empty, check the images/targets/prompts layout")
    s0 = ds[0]
    print(f"[data  ] sample keys={list(s0.keys())} "
          f"image={tuple(s0['image'].shape)} target={tuple(s0['target_rgb'].shape)}")

    # 2) generator (mps + float16)
    #    MPS does not support bf16 autocast -> use float16 (native on Metal) with AMP off.
    gen = BaseGenerator(MODEL, device=DEVICE, dtype="float16")
    print(f"[model ] family={gen.model_config.family} pipe={gen.model_config.pipeline_class}")

    # 3) tuner (LoRA + img2img flow, flow_sigma=1.0 fixes R7)
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

    # 4) manual training loop (calls _flow_training_step directly, reproducing the core of train(),
    #    and makes per-step loss logging and later visualization easy)
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
        loss = tuner._training_step(batch)  # runs VAE/t5/transformer (mps, fp16); FLUX goes through _training_step_flow
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

    # save the loss curve data
    (OUT / "losses.json").write_text(json.dumps({
        "device": DEVICE, "model": MODEL, "steps": N_STEPS, "resolution": RES,
        "flow_sigma": 1.0, "losses": losses,
        "avg_first_third": avg_first, "avg_last_third": avg_last,
    }, indent=2))
    print(f"[out   ] losses -> {OUT / 'losses.json'}")

    # save the LoRA weights (standard PEFT format) for reuse / later evaluation
    lora_dir = OUT / "lora_adapter"
    lora_dir.mkdir(parents=True, exist_ok=True)
    tuner.peft_model.save_pretrained(lora_dir)
    print(f"[out   ] LoRA weights -> {lora_dir}")

    # 5) best-effort inference visualization (merge LoRA -> img2img inference -> decode depth)
    try:
        do_inference_viz(tuner, gen, ds, OUT)
    except Exception as e:
        print(f"[viz   ] SKIPPED (inference visualization failed; does not affect the training check): {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n[done  ] MPS training check complete.")


def do_inference_viz(tuner, gen, ds, out_dir: Path):
    """Merge the LoRA into the transformer, run img2img on one sample and output a depth triptych."""
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    mapper = DepthRGBMapper()  # default cube_edge, same as the training encoding
    sample = ds[0]
    inp = sample["image"]  # [-1,1]
    tgt = sample["target_rgb"]  # [0,1]

    # merge the LoRA
    merged = tuner.peft_model.merge_and_unload()
    merged = merged.to(DEVICE)
    gen._t2i_pipe.transformer = merged
    if getattr(gen, "_i2i_pipe", None) is not None:
        gen._i2i_pipe.transformer = merged

    # input image -> PIL
    inp_pil = Image.fromarray(
        (inp.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5).mul(255).byte().numpy()
    )

    out = gen.generate(image=inp_pil, strength=0.85, num_inference_steps=4,
                       prompt=sample.get("prompt", "depth") or "depth")
    # generate may return a tensor [B,3,H,W] or a PIL.Image
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

    # decode the depth triptych
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
    print(f"[viz   ] triptych -> {fig_path}")
    print(f"[viz   ] target depth finite={not torch.isnan(tgt_depth).any()} "
          f"| pred depth finite={not torch.isnan(pred_depth).any()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
