# SSD source-conditioned depth quality: stage 2 design

## Evidence

- The best one-sample source-conditioned gate has learned geometry
  (`Pearson r=0.8154`) but misses the strict RGB target (`MAE=0.1337`).
- A per-channel affine diagnostic reduces that same prediction to RGB
  `MAE=0.0409`, showing that most residual error is systematic colour and
  dynamic-range calibration rather than missing scene structure.
- The v3 manifest and prompts call the depth labels a power-transform RGB
  encoding. A 32-image target audit instead identifies the labels as viridis:
  mean nearest-curve error is `0.0075` for viridis versus `0.3631` for the
  paper-style RGB cube-edge curve.
- No raw metric-depth files or per-image scale metadata are present in v3.
  Therefore the RGB labels support relative-depth evaluation only; metric
  depth cannot be reconstructed from this dataset.
- After the latest dataset repair, `train_val_split.json` contains 2,182 depth
  train samples and 115 validation samples (2,297 total). The earlier
  2,162/135 count in this document is stale.
- The successful standalone gate trains on the four SSD inference timesteps;
  the reusable tuner currently samples uniformly from all 1,000 training
  timesteps.

## Selected approach

Keep v3 immutable and make its effective training contract truthful. Detect
the target colormap, use a matching relative-depth prompt, and filter the
training dataset through its declared split. Add an opt-in inference-schedule
timestep sampler to reproduce the successful gate. Add a frozen-VAE pixel L1
auxiliary loss on the implied x0 image so exact target colours receive direct
supervision in addition to latent epsilon/x0 losses.

The new options remain source-conditioning-only and legacy checkpoints retain
their four-channel path. Training-only hyperparameters may change on resume as
long as the source-conditioning architecture is compatible.

## Verification

1. Unit-test colormap detection, split filtering, timestep selection, pixel
   loss gradients, and checkpoint compatibility.
2. Run existing checkpoint/training/inference regression tests and static
   checks.
3. Resume the best one-sample gate with pixel loss for a short refinement and
   compare RGB MAE, Pearson correlation, colour moments, and viridis-relative
   depth error.
4. Run the reusable CLI for at least one production checkpoint round trip.

## Out of scope

- Claiming metric-depth accuracy from v3.
- Rewriting the user dataset in place.
- A dedicated discriminative depth head or full ControlNet migration.

## Verification outcome

- Pixel L1 is technically viable on MPS, but it is not stable enough to make
  the default: one refinement step improved RGB MAE from `0.13369` to
  `0.13240`, while eight more steps regressed to `0.13379`.
- Target-statistic moment calibration plus projection to the detected viridis
  curve improved the existing best result to RGB `MAE=0.05629` and
  `Pearson r=0.93513`. That early diagnostic used the first 16 declared train
  samples. Recomputing channel moments over all 2,162 declared train targets
  changed the same one-sample RGB MAE to `0.08828`, confirming that a sorted
  16-image prefix is not a representative calibration set. Production now
  uses all declared train targets for channel statistics and only the first 16
  for colormap identification.
- The current data-only production audit reports 2,297 total depth pairs,
  2,182 train, and 115 excluded validation samples. Dedicated
  source-conditioned training now uses all declared train samples by default instead of inheriting
  the legacy sorted-prefix cap of 300; an explicit `--max-per-task` still wins.
- A new relative-depth evaluator uses the official validation split and
  separately reports raw-model and configured-calibration RGB error,
  viridis-curve position error, correlation, and curve consistency from the
  same stochastic generation. All 135 validation input/target pairs resolve.
- The stage-2 production smoke completed on SSD-1B/MPS at 128 px for one
  optimizer step (`loss=1.07877`). Its checkpoint contains 360 complete LoRA
  A/B module pairs, an 8-channel input convolution, epsilon prediction,
  four-step inference-schedule sampling, full-train viridis statistics, and
  the truthful relative-depth prompt. A fresh-process reload produced
  pixel-identical repeated outputs for seed 42 and viridis projection error
  `4.7e-10`.
- On 30 official validation samples, the intentionally one-step smoke obtained
  raw RGB `MAE=0.27797`, relative-position `MAE=0.37658`, and Pearson
  `r=-0.05025`. Calibration changed these to RGB `MAE=0.21106`, position
  `MAE=0.34940`, and `r=-0.03737`. This validates the production/evaluation
  closure, not depth quality: meaningful generalization requires a longer
  split-clean training run.
- The focused regression suite finished with 109 passed and 1 skipped; Ruff,
  Python bytecode compilation, and `git diff --check` also passed.

## Stage 3 multi-sample outcome

- A split-clean SSD-1B run used 32 train samples, 16 fixed validation samples,
  128 px, rank/alpha 8/8, epsilon prediction, four inference timesteps, x0
  weight 1.0, and checkpoints at steps 20/60/100.
- From step 20 to 100, raw RGB MAE improved `0.2635 -> 0.2407` and relative
  position MAE improved `0.3611 -> 0.3399`, but Pearson remained negative
  (`-0.0567 -> -0.0593`) and final Spearman was `-0.0843`.
- The same step-100 checkpoint scored Pearson `0.0037` and Spearman `0.0015`
  on 16 train samples. An 8-sample/200-step capacity probe remained negative
  on its train samples (Pearson `-0.0873`, Spearman `-0.0830`).
- Raising LR from `3e-5` to the successful one-sample gate's `5e-4` improved
  train RGB MAE to `0.2139`, but train Pearson/Spearman remained only
  `0.0134/0.0141`. This excludes low optimizer magnitude as the sole cause.
- The added source-channel weights are updated and persist through checkpoint
  reload (source/target input-weight norm ratio reaches 3.77% in the high-LR
  probe), so the failure is not a disconnected conditioning branch.
- Stop rule: do not extend this configuration to 300 steps. The next gate must
  compare a structure-aware decoded-x0 loss against multi-scale spatial
  conditioning (T2I-Adapter/ControlNet) before another data-scale increase.
- The evaluator now supports explicit train/validation split selection and
  tie-aware Spearman correlation so colour-range progress cannot be mistaken
  for depth-order progress.

## Stage 4 geometry-gate outcome

Two opt-in alternatives were implemented and verified against the same
8-channel SSD-1B/LoRA baseline:

1. A decoded-x0 multi-scale Charbonnier loss with horizontal/vertical gradient
   matching.
2. A randomly initialized `T2IAdapter(full_adapter_xl)` with four
   zero-initialized 1x1 projections into the UNet's native
   `down_intrablock_additional_residuals` path.

Both variants completed one-step MPS smoke training, checkpoint round trips,
and fresh-process inference. The implementation regression finished with 799
passed and 12 skipped tests; Ruff, mypy, and `git diff --check` passed. The
adapter adds about 82.8M trainable parameters and its saved state is required
strictly at inference.

The formal matched gate used four train and four validation images, 128 px,
60 optimizer steps, four inference steps, rank/alpha 8/8, LR `5e-4`, and seed
42. Results below are means over each four-image split.

### Raw metrics

| Arm | Split | RGB MAE | Position MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | train | 0.2182 | 0.3254 | -0.0076 | 0.0247 |
| Baseline | val | 0.2124 | 0.2741 | 0.0705 | -0.0411 |
| Structure loss | train | 0.2222 | 0.3215 | -0.0277 | -0.0276 |
| Structure loss | val | 0.2171 | 0.2800 | 0.0816 | -0.0346 |
| Spatial adapter | train | 0.2158 | 0.3042 | 0.0004 | 0.0163 |
| Spatial adapter | val | 0.2199 | 0.3146 | -0.0040 | -0.1114 |

### Calibrated metrics

| Arm | Split | RGB MAE | Position MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | train | 0.2267 | 0.3526 | 0.0108 | 0.0293 |
| Baseline | val | 0.1965 | 0.3005 | 0.0528 | -0.0415 |
| Structure loss | train | 0.2239 | 0.3563 | -0.0232 | -0.0213 |
| Structure loss | val | 0.1875 | 0.2924 | 0.1036 | -0.0231 |
| Spatial adapter | train | 0.2240 | 0.3423 | 0.0208 | -0.0022 |
| Spatial adapter | val | 0.2006 | 0.3038 | 0.0287 | -0.0600 |

Neither arm passes the gate. Structure supervision improves calibrated
validation Pearson and error but regresses both train correlations, while
validation Spearman remains negative. The adapter worsens validation Pearson
and Spearman. Its zero-convolution weights are all non-zero after training, so
the negative result is not a disconnected-branch artifact.

## Stage 5 scalar-geometry outcome

The stage-4 recommendation was implemented as a differentiable soft projection
from decoded RGB to the audited colormap's scalar curve position. The loss
combines scalar Charbonnier reconstruction, scalar spatial gradients, and
deterministic long-range ordinal comparisons. It remains disabled by default
and its three weights are serialized in the source-conditioning config.

The same four-train/four-validation, 128 px, 60-step protocol tested overall
weights 0.2 and 1.0. The second arm is a bounded sensitivity probe: weight 0.2
changed the same-seed one-step total loss by only about 0.022 relative to
baseline, so one stronger setting was needed to exclude underweighting.

### Raw metrics

| Arm | Split | RGB MAE | Position MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | train | 0.2182 | 0.3254 | -0.0076 | 0.0247 |
| Baseline | val | 0.2124 | 0.2741 | 0.0705 | -0.0411 |
| Scalar 0.2 | train | 0.2133 | 0.3118 | 0.0260 | 0.0533 |
| Scalar 0.2 | val | 0.2111 | 0.2845 | 0.0478 | -0.0656 |
| Scalar 1.0 | train | 0.2123 | 0.3065 | 0.0272 | 0.0185 |
| Scalar 1.0 | val | 0.2122 | 0.2884 | 0.0520 | -0.0548 |

### Calibrated metrics

| Arm | Split | RGB MAE | Position MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Baseline | train | 0.2267 | 0.3526 | 0.0108 | 0.0293 |
| Baseline | val | 0.1965 | 0.3005 | 0.0528 | -0.0415 |
| Scalar 0.2 | train | 0.2198 | 0.3353 | 0.0700 | 0.0751 |
| Scalar 0.2 | val | 0.1984 | 0.3045 | 0.0354 | -0.0627 |
| Scalar 1.0 | train | 0.2204 | 0.3342 | 0.0741 | 0.0681 |
| Scalar 1.0 | val | 0.2014 | 0.3075 | 0.0580 | -0.0259 |

The 0.2 arm nudges both raw train correlations upward, but the gains are small
relative to standard deviations around 0.20--0.35 and reverse on validation.
Weight 1.0 does not improve raw train Spearman and also regresses raw
validation correlation. Stage 5 therefore fails the ordering-memory gate.

## Stage 6 frozen-feature architecture outcome

A deterministic `t=0` SSD-1B forward pass now caches VAE latent, three UNet
down-block outputs, the mid block, and final output. A 509,745-parameter
FPN-style scalar-depth head is the only trainable module. At 128 px the six
feature levels have channels `4/320/640/1280/1280/4` and spatial sizes
`16/8/4/4/4/16`.

The four-image gate used seed 42, full-batch AdamW, LR `3e-3`, and a 2,000-step
upper bound. It passed early at step 75:

| Step | Scalar MAE | Pearson | Spearman |
|---:|---:|---:|---:|
| 1 | 0.1649 | 0.5239 | 0.5399 |
| 25 | 0.0740 | 0.9438 | 0.9398 |
| 50 | 0.0575 | 0.9619 | 0.9608 |
| 75 | 0.0330 | 0.9736 | 0.9740 |

All four samples individually satisfy MAE <=0.05 and Pearson/Spearman >=0.95;
the weakest individual Pearson is 0.9566 and the largest MAE is 0.0421. The
entire run took 12.0 seconds including SSD load, feature extraction, head
training, and artifact writing. The saved head strictly reloads against its
six-level feature contract.

This gate passes. SSD/VAE frozen representations contain readily decodable
four-image geometry; the stage-3 through stage-5 failures are therefore more
specifically diffusion RGB readout/optimization failures. This is not held-out
evidence and does not rehabilitate the generative path by itself.

## Stage 7 recommendation

Continue only with the discriminative probe path. Run a split-clean VAE-only
versus VAE+SSD comparison on a larger train subset and fixed validation set,
keeping the head capacity and training budget matched. SSD intermediate
features earn promotion only if they improve held-out Pearson/Spearman over
the VAE-only control. Do not resume ControlNet, Adapter, or diffusion-loss
tuning from the four-image overfit result.

## Stage 7 frozen-feature ablation outcome

The split-clean comparison used 32 declared train images and 16 disjoint
official validation images at 128 px. The two arms had identical 509,745
parameter heads, initial state checksum, batch sequence, optimizer, loss, and
fixed 300-step budget. VAE-only activated only the post-projection VAE slot;
VAE+SSD activated all six slots.

| Arm | Split | Scalar MAE | Pearson | Spearman |
|---|---|---:|---:|---:|
| VAE-only | train | 0.1256 | 0.8694 | 0.8569 |
| VAE-only | val | 0.2591 | 0.1425 | 0.1629 |
| VAE+SSD | train | 0.0670 | 0.9602 | 0.9535 |
| VAE+SSD | val | 0.2402 | 0.3663 | 0.3682 |

At the predeclared endpoint, SSD features improve validation Pearson by
`+0.2238`, Spearman by `+0.2053`, MAE by `0.0190`, and RMSE by `0.0145`.
Both correlation deltas exceed the `+0.05` promotion threshold by more than
4x. The gate passes in 36.3 seconds total runtime.

This is the first split-clean evidence that SSD intermediate features add
held-out geometry beyond VAE reconstruction detail. It validates the frozen
discriminative path, not the failed generative RGB diffusion path.

## Stage 8 recommendation

Scale only the frozen discriminative probe. Use at least three seeds and a
larger declared train subset, reserve an internal development subset for
checkpoint selection, and evaluate the official validation subset only at the
chosen endpoint. Keep the matched VAE-only control. Do not resume LoRA,
ControlNet, Adapter, or diffusion-loss experiments from this result.

## Stage 8 frozen-feature multi-seed outcome

Stage 8 used a deterministic split of the declared manifest: 128 `train_fit`,
32 internal `dev`, and 32 untouched `official_val` images at 128 px. The split
checksum is `b674806977e5e44c37bcee81c2c23e296d7ffb2ee16d4846250a45240390a013`.
Seeds 42, 43, and 44 shared matched six-slot head capacity, per-seed initial
state and batch schedule checksums, optimizer/loss, and a 300-step budget. Each
arm selected its endpoint from dev only; official validation was evaluated only
after that selection.

| Seed | Selected step (VAE-only / VAE+SSD) | Official Pearson delta | Official Spearman delta | Official MAE delta |
|---:|---:|---:|---:|---:|
| 42 | 275 / 175 | +0.0744 | +0.0797 | -0.0062 |
| 43 | 300 / 125 | +0.0595 | +0.0633 | -0.0060 |
| 44 | 300 / 150 | +0.1194 | +0.1073 | -0.0060 |
| **Mean** | -- | **+0.0844** | **+0.0834** | **-0.0060** |

The dev deltas were positive for both correlations in all three seeds and had
the same sign on official validation. At least two seeds improved both
correlations (in fact all three did), and the aggregate MAE guardrail passed.
The Stage 8 multi-seed gate therefore **passes**. This strengthens the claim
that SSD intermediate features contribute held-out relative-depth geometry
beyond the matched VAE-only representation, while still saying nothing about
the failed RGB diffusion path or metric depth.

## Stage 9 full-dataset outcome

The full declared dataset was used under the same split-clean protocol: all
2,182 declared train names were consumed as 1,926 `train_fit` plus 256 `dev`,
and all 115 declared validation names remained untouched until endpoint
evaluation. At 128 px and 4,000 steps, seeds 42/43/44 completed both matched
arms in 1,822.7 seconds on MPS. The split checksum is
`f7c195ae0c382034ec859215ef46ec5b76bd868615ac7d7b5c640930ec36d4a6`.

| Seed | Selected step (VAE-only / VAE+SSD) | Official Pearson delta | Official Spearman delta | Official MAE delta |
|---:|---:|---:|---:|---:|
| 42 | 4000 / 3000 | +0.0762 | +0.0794 | -0.0154 |
| 43 | 4000 / 3200 | +0.0887 | +0.0931 | -0.0195 |
| 44 | 4000 / 2600 | +0.0787 | +0.0780 | -0.0159 |
| **Mean** | -- | **+0.0812** | **+0.0835** | **-0.0169** |

The aggregate official-val metrics for VAE+SSD were MAE `0.1733`, Pearson
`0.6527`, and Spearman `0.6345` (population standard deviations across seeds
were approximately `0.0012`, `0.0043`, and `0.0031`). The gate passes for all
three seeds. These checkpoints are valid full-dataset experiment endpoints;
for a final deployment checkpoint, the selected step should be rerun with a
VAE+SSD head fit on all 2,182 train names after the dev protocol is frozen.
