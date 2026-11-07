# Offline depth-data cleaning

This PR prepares and audits data, plus read-only safety checks at existing
training-data entry points. It does not modify training configuration, sampling,
interpolation, losses, evaluation, or running jobs.
It does not automatically replace any training input. NYU is excluded from this
work; existing NYU and other source files remain untouched.

## Hypersim native-depth filtering

A frame passes when its native invalid-pixel fraction is at most **2%**.
Exactly 2% passes; more than 2% fails. Validity is finite positive native metric
depth intersected with any supplied native validity mask. RGB black pixels or
reported coverage numbers are not substitutes for native evidence. Missing,
corrupt, or misaligned evidence fails the build.

```bash
python scripts/prepare_clean_depth.py \
  --manifest data/hypersim-filtered-v1/pairs.jsonl \
  --output data/hypersim-observed98-v2 \
  --expected-depth-rejections 1066
```

Use a new output directory. The CPU-only command checks native RGB/target geometry,
hashes all referenced assets, and writes an independent manifest with absolute
asset paths. Sources are not deleted, rewritten, resized, or inpainted.
The expected rejection count is an assertion, not a selection quota.

The complete native scan on 2026-11-07 checked **13,267** Hypersim frames, excluded
**1,066**, and retained **12,201**. See the [compact audit](data/clean_depth_hypersim_audit.json)
and [sorted exclusion IDs](data/hypersim_filtered_v1_holes_gt2_exclusions.txt).
Every run writes `report.json`, `admitted_audit.jsonl`, and `excluded.jsonl`;
`COMPLETE.json` binds outputs by SHA-256. Failed scans do not publish a completed
output directory, and existing outputs are never overwritten.

These are native-resolution data-quality counts. Later resizing or cropping can
change the hole fraction; this PR neither modifies nor validates those training
transforms. The read-time check described below only admits or rejects the
native data; it does not alter any subsequent data processing.

## Read-time safety checks

The unified, legacy SANA, and legacy FLUX manifest readers check selected training
depth rows before the existing sample limit. Any row above 2% native invalid
pixels rejects the input with a cleanup instruction; rows are not silently
filtered, rewritten, or reordered. Missing, corrupt, or misaligned native evidence
also rejects the input. The cleaned manifest is a separate explicit input.
The checks are not applied to validation rows or unrelated tasks.

Known raw, V3, and TAR depth routes without verifiable native evidence are
rejected with an instruction to prepare a native-backed pair manifest first.
Dataset-wrapper checks at the legacy tuner data boundary cover nested known
repository routes. No training arguments or configuration files are changed.

For admitted data, all previous image resizing, random cropping, interpolation,
augmentation, target encoding, sampling, and losses remain exactly as before.
In particular, this PR does not choose clean crop offsets, replace interpolation,
add a geometric fallback, or guarantee post-transform coverage.

## DIODE indoor additions

The builder preserves native RGB, float32 metric depth, and official validity
masks. Its default gates require at least **98% native valid pixels** and at most
**0.05 m maximum valid-pixel RGB encoding error**. Only stricter overrides are
accepted. There is no depth clipping, resizing, interpolation, or hole filling.
Invalid RGB target pixels remain black; no masked loss is introduced.

Official sources:
- Source and MIT notice: <https://diode-dataset.org/>.
- Archive: <https://diode-dataset.s3.amazonaws.com/train.tar.gz>,
  86,747,209,190 bytes, MD5 `3a94632398fe1d002d89f11743f748b1`.
- `diode_meta.json`: DIODE devkit revision
  `8b1765b7d801a5f5e2877c434ffe164e62ce8c90`.

Verify the complete archive checksum, then extract `train/indoors/`.
The builder accepts an extracted tree or uncompressed TAR, not a compressed TAR.

```bash
python scripts/prepare_diode_train.py \
  --source diode --input data/diode/train \
  --official-metadata data/diode/diode_meta.json \
  --exclude-manifest data/diode-existing-train/pairs.jsonl \
  --exclude-manifest data/diode-extra-train/pairs.jsonl \
  --output-dir data/diode-indoor-train-additions-v1

python scripts/validate_diode_train.py \
  data/diode-indoor-train-additions-v1 \
  --official-metadata data/diode/diode_meta.json
```

Paths above are examples. Both existing manifests, containing 996 and 195 DIODE
pairs, must be supplied. Official metadata must contain 8,574 unique indoor train
IDs disjoint from validation, and source triplets must match the official hierarchy.
After 1,191 existing exclusions, 7,383 is only an upper bound before quality gates,
not a final retained count. Incomplete sources fail unless `--allow-partial` is
explicitly requested; partial output is never a complete-source build.

Builds are immutable. `--resume` verifies unchanged configuration and artifact
bytes. Each retained row has four assets, provenance, source hashes, recorded
exclusions and quality rejections, and a completion marker. The independent
validator rereads every source triplet and output asset, verifies hashes, arrays,
split membership, all exclusions/rejections, exact accounting, and codec error.
It performs no training-loader checks, training, or publication.

Final data acceptance requires complete-source accounting, a passing independent
source/asset audit, and pixel-weighted mean encoding error below 0.01 m. Full
DIODE checksum verification, conversion, and acceptance are still pending; no final retained
count or published dataset is claimed. The 52-frame pilot retained 10 frames;
that ratio must not be extrapolated to the full dataset.

## Optional offline composition

After independent DIODE validation, the same cleaner can combine native-backed
manifests into a new output without changing existing data or training settings:

```bash
python scripts/prepare_clean_depth.py \
  --manifest data/hypersim-observed98-v2/pairs.jsonl \
  --manifest data/diode-existing-train/pairs.jsonl \
  --manifest data/diode-extra-train/pairs.jsonl \
  --manifest data/diode-indoor-train-additions-v1/pairs.jsonl \
  --output data/clean-depth-data
```

Explicit non-training rows are excluded rather than relabelled. Identical
same-task duplicates are excluded; conflicting identities and train/held-out
overlap fail. Non-depth training rows are preserved. Unlabelled legacy rows are
assumed train and counted in the audit, which is not proof of an official split.
Do not supply an older combined NYU/DIODE manifest. If real DIODE train is used
in a later approved training run, DIODE validation becomes same-source held-out
evaluation rather than synthetic-only zero-shot evaluation.

## Focused checks

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest \
  tests/test_clean_depth.py tests/test_diode_train.py \
  tests/test_validate_diode_train.py
```

The 98% threshold is a data-quality policy, not a proven optimum or a claim of
model accuracy gains. No training or benchmark result is claimed.
