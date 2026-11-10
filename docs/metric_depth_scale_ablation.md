# Metric depth scale loss ablation

The unified SFT trainer accepts `--depth-scale-loss-weight` (YAML
`loss.depth_scale_loss_weight`, default 0). Keep `metric_power` and
`cube_edge`. The auxiliary loss reconstructs the target endpoint from the
predicted flow, decodes it through the frozen VAE with gradients to the latent,
and projects the RGB onto the cube-edge curve. It minimizes Huber loss on each
image's 10%-trimmed mean log prediction/GT ratio. The Huber transition is
`log(1.25)`, the log ratio used by δ1. FP32 inverse arithmetic leaves
headroom at white and a 1 mm log floor at black.

A row qualifies only with explicit native `metric_depth` and `valid_mask`
assets and without a legacy depth exemption. The scalar plane is divided by
`metric_depth_scale` (units per metre, default 1). Use the same resize
dimensions and crop as the RGB pair, with nearest sampling of the native
depth and mask. Pixels must be valid in both native arrays. Missing supervision
leaves the normal flow loss active, without a VAE decode for that row. With the
option enabled, `depth_scale_samples` counts qualifying local samples in each
log window and `depth_scale_loss` is the unweighted auxiliary term averaged
over all samples, with zero contributions for unqualified rows. Validation
and best-checkpoint selection keep the existing EMA macro flow loss.

## Paired run

Use a separate job for each arm on allocated training nodes. Both arms use
fresh rsLoRA r64 from the same published
`Melo111/Fusion-C1-KQ0r128-60-40@9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d`
base, the same I108 manifest and admission record, and the same 16-process
layout. Verify the manifest SHA-256 against the original I108 run
(`d41071cf9f9486bff8a2ebf3aabfa7ca1afb4fa0b4fd6fd58f236bfb01a62c97`).
Use distinct empty output directories and rendezvous IDs. Keep the I108-A
schedule and compare the same saved steps and weight type in both arms.

| Arm | `--depth-scale-loss-weight` |
| --- | ---: |
| Flow MSE | 0 |
| Flow MSE + scale | 0.01 |

The proposed 0.01 is a starting coefficient for this ablation. On each node,
bind the following variables to the allocated job and exact inputs. Run the
same command on both nodes of an arm with their own `NODE_RANK`.

```bash
export OPEN_VISION_BANANA_MODEL_PATH="$FUSION_BASE"
export PYTHONPATH="$PWD/src"
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" scripts/train.py \
  --config configs/experiments/fusion_r64_large_scale.yaml \
  --data-dir "$TRAIN_DATA_ROOT" --pair-manifest "$TRAIN_MANIFEST" \
  --grad-accum 2 --depth-scale-loss-weight "$SCALE_WEIGHT" --output-dir "$ARM_OUTPUT"
```

Compare the two effective configs before launch. Their only intentional
differences are output directory and scale weight. Evaluate step 0 and the
same saved EMA steps, for example 2,500 and 5,000, on the same full NYUv2,
ETH3D and DIODE-Indoor splits. Keep the existing 512 BF16 protocol with
50 inference steps, guidance 4, strength 0.85, seed 42 and native GT. Pass
`--depth-scale-ratios` to `scripts/evaluate_paper_benchmarks.py` for each
depth evaluation, including every shard. Report:

- `decoded_metrics["delta_1.25"]` and `decoded_metrics["delta_1.25_aligned"]`.
- `depth_scale_ratios` in dataset order, merged in original shard order.
  Each value is median(prediction) / median(GT) over the same valid scoring
  pixels. A skipped image or one without valid positive pairs is `null`.
- On the matched images, median absolute log-ratio and the fraction of ratios
  within `[0.8, 1.25]`, together with the null count and aligned pixel coverage.
  Compare raw δ1 separately for each dataset. Use aligned δ1 to check for
  changes in relative geometry.

The aligned scores and scale ratios are diagnostics. The raw metric protocol
and default evaluation result schema stay unchanged.
