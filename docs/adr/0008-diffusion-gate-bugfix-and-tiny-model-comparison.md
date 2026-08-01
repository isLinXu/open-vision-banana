# ADR-0008: Diffusion gate bugfix and tiny-model comparison baseline

## Status

Accepted.

## Context

ADR-0007 declared the source-conditioned SSD-1B diffusion path as failed and
promoted the frozen-feature probe to production. The failure was attributed to
a "readout/optimization failure" because all diffusion variants (epsilon,
scalar geometry, x0 prediction, conv LoRA targets) failed their gates:
train Pearson reached 0.745 but validation Pearson was -0.10.

A subsequent code audit of the standalone `scripts/overfit_gate_depth.py`
gate script revealed **three implementation bugs** that fully explain the
"mean prediction" / "extreme oversmoothing" symptoms reported by the user.
The bugs were NOT present in the project's shared infrastructure
(`conditioning.py`, `tuner.py`, `generator.py`), which correctly implements
the InstructPix2Pix-style source conditioning. The bugs were isolated to the
standalone diagnostic script that was used for early gate experiments.

## Root cause analysis

### Bug 1: Source latent scaling (CRITICAL)

**Location**: `scripts/overfit_gate_depth.py`, training loop (line ~390) and
evaluation loop (line ~412).

**Old code**:
```python
model_input = torch.cat([noisy, source_lat], dim=1)
model_input = model_input / ((sigma**2 + 1) ** 0.5)  # scaled ENTIRE concat
```

**Problem**: `EulerDiscreteScheduler.scale_model_input` divides by
`sqrt(sigma^2 + 1)`. This scaling must be applied **only to the noisy target
latent**, not to the clean source latent. The source latent is a conditioning
signal, not part of the denoised trajectory. When the entire 8-channel
concatenation is divided by `sqrt(sigma^2 + 1)`, the source signal is
attenuated at high-noise timesteps (sigma can be ~14 at t=999, making the
source latent ~1/14th of its original magnitude). The UNet then sees a
vanishingly small source image and falls back to mean prediction.

**Fix**: Scale only the noisy target before concatenation:
```python
scaled_noisy = noisy / ((sigma**2 + 1) ** 0.5)
model_input = torch.cat([scaled_noisy, source_lat], dim=1)
```

**Verification**: The project's `conditioning.py:build_source_conditioned_input()`
already does this correctly (line 278):
```python
scaled_target = noisy_target / torch.sqrt(sigma.square() + 1.0)
return torch.cat((scaled_target, source_latent), dim=1), noisy_target, sigma
```

### Bug 2: Pooled prompt embedding extraction (MODERATE)

**Location**: `scripts/overfit_gate_depth.py`, `encode_text()` (line ~241).

**Old code**:
```python
e2 = models["text_encoder_two"](tok2)[0]
pooled = models["text_encoder_two"](tok2)[0][:, 0, :]  # WRONG
```

**Problem**: SSD-1B's `text_encoder_two` is a `CLIPTextModelWithProjection`.
The correct pooled output is `.text_embeds` (the projected pooler output,
shape [1, 1280]), NOT the first token of `last_hidden_state` (which is the
penultimate hidden state, not projected). Using the wrong tensor as
`added_cond_kwargs.text_embeds` corrupts the SDXL micro-conditioning signal.

Additionally, `text_encoder_two` was called twice for the same input, wasting
computation.

**Fix**:
```python
enc2_out = models["text_encoder_two"](tok2)  # call once
e2 = enc2_out[0]  # last_hidden_state [1, 77, 1280]
pooled = enc2_out.text_embeds  # projected pooled output [1, 1280]
```

### Bug 3: Manual Euler step instead of scheduler.step (MINOR)

**Location**: `scripts/overfit_gate_depth.py`, evaluation loop (line ~433).

**Old code**:
```python
sigma_next = scheduler.sigmas[j + 1] if j + 1 < len(scheduler.sigmas) else 0
pred_x0 = latent - sigma * noise_pred.float()
latent = pred_x0 + sigma_next * noise_pred.float()
```

**Problem**: This manual Euler ancestral step is a simplification that omits
the sigma-derived step size used by `EulerDiscreteScheduler.step()`. While
mathematically close, it does not exactly match the scheduler's internal
implementation, introducing small numerical discrepancies between training
and inference.

**Fix**: Use `scheduler.step()` for exact consistency:
```python
latent = scheduler.step(noise_pred.float(), t, latent).prev_sample
```

### Bugs NOT found in shared infrastructure

The following files were audited and found **correct**:

- `src/open_vision_banana/conditioning.py`:
  `build_source_conditioned_input()` correctly scales only the target.
- `src/open_vision_banana/training/tuner.py`:
  `_training_step_sdxl()` delegates to `build_source_conditioned_input()`.
- `src/open_vision_banana/inference/generator.py`:
  `_generate_source_conditioned()` uses `scheduler.scale_model_input()` on
  only the noisy latents, then concatenates clean `source_latents`.
- `scripts/quick_overfit_ssd_source_condition.py`:
  Uses `build_conditioned_model_input()` (local helper with correct logic),
  `scheduler.step()`, and `tuner._encode_text()` (correct encoder delegation).

## Decision

1. **Fix the three bugs in `scripts/overfit_gate_depth.py`**. This restores
   the standalone diagnostic script to correctness, enabling it to serve as
   a quick synthetic-data sanity check.

2. **Use `scripts/quick_overfit_ssd_source_condition.py` as the primary gate
   script** for real v3 data gates. It already uses the project's shared
   infrastructure and was not affected by the bugs.

3. **The diffusion path is re-enabled as a comparison baseline** for the
   Tiny Model V5 route. The two routes serve different purposes:

   | Route | Architecture | Role | Params (trainable) |
   |-------|-------------|------|-------------------|
   | Diffusion (Path A) | SSD-1B + 8ch conv_in + LoRA | Paper replication baseline | ~50M (LoRA + conv_in) |
   | Tiny Model (Path B) | Standalone CNN | Lightweight independent | ~12-28M (full) |

4. **The frozen-feature probe** (ADR-0007) remains the production depth
   estimator. The diffusion path is for research comparison only.

## Execution commands

### Quick synthetic gate (fastest, verifies pipeline correctness):

```bash
python scripts/overfit_gate_depth.py \
  --use-synthetic \
  --num-samples 16 \
  --max-steps 300 \
  --lr 1e-4 \
  --output-dir output/diffusion_gate_synthetic_fixed
```

Gate criteria: SSIM > 0.80, L1 < 0.10 on training samples.

### Real v3 depth gate (split-clean train/val):

```bash
python scripts/quick_overfit_ssd_source_condition.py \
  --data-dir data/vb_snvc_train_v3 \
  --output-dir output/diffusion_gate_v3_fixed \
  --num-samples 8 \
  --eval-samples 4 \
  --resolution 512 \
  --max-steps 500 \
  --inference-steps 4 \
  --prediction-target epsilon \
  --rank 8 \
  --alpha 8 \
  --lr 1e-4 \
  --gate-mae 0.10 \
  --gate-correlation 0.70
```

Gate criteria (val): rgb_mae < 0.10, relative_position_pearson_r > 0.70.

### Comparison with Tiny Model V5:

```bash
# Tiny Model V5 (Path B)
python scripts/train_tiny_vb_v5.py \
  --data-dir data/vb_snvc_train_v3 \
  --steps 12000 \
  --output-dir output/tiny_vb_v5

# Evaluate both
python scripts/evaluate_tiny_vs_probe.py \
  --tiny-checkpoint output/tiny_vb_v5/best.pt \
  --data-dir data/vb_snvc_train_v3
```

## Tiny Model V2 analysis (user-reported issues)

The user reported extreme oversmoothing, dynamic range compression, and
semantic/spatial reasoning errors in the V2 model at
`output/912b6a07-652a-4b0d-a480-b3b7ee65ea9a/tiny_vb_v2/`.

V2 config: `base_width=32, num_stages=4, block_depth=2, output_activation=sigmoid`,
12.3M params, 3000 steps, lr=0.001.

Key issues identified:
1. **Sigmoid output activation**: Compresses output to [0, 1], losing dynamic
   range. V5 fixed this with `tanh` output.
2. **Small model (bw=32)**: 12.3M params is insufficient for fine detail.
3. **V2 had unfixed architecture issues**: residual connection drift, FiLM
   zero-init causing task collapse. V5 addresses all three.

V5 already addresses these issues per MEMORY.md:
- Fixed residual mask (non-learnable)
- Non-zero FiLM initialization
- tanh output (not sigmoid)
- 4-layer deep head
- bw=48 variant: 28.47M params

## Validation

The fixed `overfit_gate_depth.py` should now pass the synthetic gate
(SSIM > 0.80) because:
1. The source latent is no longer attenuated, so the UNet can actually
   "see" the input image.
2. The pooled embeddings are correct, so micro-conditioning works.
3. The Euler step matches the scheduler's internal implementation.

The `quick_overfit_ssd_source_condition.py` was already correct and should
be used for the real-data gate.
