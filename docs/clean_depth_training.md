# Clean depth training admission

Depth training now uses a conservative maximum of **2% native invalid pixels**.
A frame with exactly 2% passes; a frame above 2% is excluded before sampling or
`max_samples` truncation. Missing, corrupt, or misaligned native evidence fails
closed. No source images or labels are deleted or inpainted.

## Prepare an auditable training manifest

Use a fresh output directory. The CPU-only command reads each native metric depth
and/or validity mask, intersects finite positive values, checks RGB/target geometry,
and hashes the source assets. It writes absolute paths into the new manifest so
moving the manifest alone does not change which images it refers to.

```bash
python scripts/prepare_clean_depth.py \
  --manifest data/hypersim-filtered-v1/pairs.jsonl \
  --output data/hypersim-observed98-v2 \
  --expected-depth-rejections 1066
```

The expected count is an assertion, not a fixed-size selection or a list of IDs.
The full native scan on 2026-11-07 measured 13,267 filtered-v1 frames and retained 12,201
frames, rejecting exactly 1,066. The [compact audit](data/clean_depth_hypersim_audit.json) and [sorted exclusion IDs](../configs/datasets/hypersim_filtered_v1_holes_gt2_exclusions.txt) record the result. Use `report.json` and `excluded.jsonl` for a new build.
`COMPLETE.json` binds the output manifests and reports by SHA-256. Partial or
failed scans do not publish a completed output directory. Existing outputs are
never overwritten. Keep the old component for provenance and reproducibility.

To combine clean Hypersim and independently validated DIODE components:

```bash
python scripts/prepare_clean_depth.py \
  --manifest data/hypersim-observed98-v2/pairs.jsonl \
  --manifest data/diode-existing-train/pairs.jsonl \
  --manifest data/diode-extra-train/pairs.jsonl \
  --manifest data/diode-indoor-train-additions-v1/pairs.jsonl \
  --output data/clean-depth-train
```

These paths are examples; use each component's actual manifest. The two existing
DIODE components contain 996 and 195 rows. The additions builder must exclude
both before conversion. If a supplied legacy component mixes ARKitScenes train
and validation, this composer excludes its explicit non-training rows instead
of relabelling them. Identical same-task duplicates are excluded; conflicting
identities and train/held-out overlap fail. Non-depth training tasks are preserved.
Unlabelled legacy rows are assumed train and counted in the audit; this is not
proof of an official split. Use official split evidence for new sources.

**NYU is not part of this training mixture or the DIODE addition.** Do not pass an
older combined NYU/DIODE manifest. Existing NYU files need not be deleted.
DIODE official validation remains held out. Using real DIODE training data changes
the protocol: its validation result is same-source held-out evaluation, not a
synthetic-only zero-shot result.

## Training behavior

The unified trainer and the independent legacy SANA and FLUX manifest loaders
apply native admission to training depth rows. Supply the new `pairs.jsonl` as
`--pair-manifest` with its parent as `--data-dir`, using the existing model and
training configuration. Validation/evaluation split behavior is unchanged.

Native `metric_depth` (scalar NPY or PNG) and/or `valid_mask` is required. The
legacy prepared-Hypersim `targets/X.png` layout can locate a sibling
`metric_depth/X.npy`. A black RGB pixel or a manifest's stated coverage fraction
is not sufficient evidence of validity. Raw, V3, and TAR depth training paths
without this contract must first be converted to a native-backed pair manifest;
they fail with an actionable message rather than bypass the gate.

For aspect-preserving depth crops, admission enumerates clean square offsets on
the nearest-resized validity mask and training samples only those offsets. If no
square crop passes, a full-frame square resize is used only when its transformed
mask also passes. If neither can pass at the requested resolution, the frame is excluded before
sampling and counted separately as a geometry rejection. Missing/corrupt evidence
still fails initialization rather than silently dropping data. Non-depth crop behavior is unchanged. Each manifest loader exposes
`depth_quality_rejections` with separate native and geometry counts. Depth training targets
use nearest-neighbor resampling to avoid mixing the black invalid placeholders
into neighboring valid RGB labels. This changes depth target interpolation from
legacy LANCZOS; evaluation interpolation is unchanged.

### Measured 512-pixel geometry

A second read-only scan checked all 12,201 native-clean Hypersim frames with the
512-pixel, aspect-preserving policy: 12,165 have an admissible square crop and 35
need the verified full-frame fallback. One additional frame,
`ai_015_005_cam_00_frame.0050`, has no passing transform. Its native hole fraction
is 1.99038%, but full-frame nearest resizing gives 5,245 / 262,144 invalid pixels
(2.00081%), above the permitted 5,242. It is separately excluded for geometry.
The full training-loader check confirms **12,200** admitted frames at 512 pixels,
not 12,201. It also loaded 16 deterministic samples with finite tensors and
verified transformed validity. See the [loader validation](data/hypersim_loader_validation.json).
See the [geometry audit](data/hypersim_512_geometry_audit.json). The native 1,066
exclusion count remains unchanged. Other resolutions require their own admission
check; these counts are not model-training or benchmark results.

This is filtering and geometry protection, **not masked loss**. The unchanged VAE
and flow loss still see the remaining invalid target pixels. The 98% coverage
threshold is a conservative policy, not a demonstrated optimum or a promise of
better model accuracy. No training run, benchmark gain, or retroactive modification
of historical checkpoints is implied.

## DIODE preparation and checks

See [DIODE training data](diode_training_data.md) for the official source, native
preservation, held-out verification, immutable resume, and independent validation.
The clean CLI defaults require at least 98% valid coverage and at most 0.05 m
maximum valid-pixel RGB encoding error. Final release acceptance also requires
pixel-weighted mean encoding error below 0.01 m. Download completion, conversion,
full independent verification, and model training are separate milestones.

Focused CPU regression checks:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest \
  tests/test_clean_depth.py tests/test_depth_training_quality.py \
  tests/test_diode_train.py tests/test_validate_diode_train.py
```