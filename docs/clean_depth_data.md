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
The complete run accounts for all 8,574 IDs as 1,191 existing exclusions,
428 coverage rejections, 1,616 encoding-error rejections, and 5,339 retained pairs.
Incomplete sources fail unless `--allow-partial` is
explicitly requested; partial output is never a complete-source build.

Builds are immutable. `--resume` verifies unchanged configuration and artifact
bytes. Each retained row has four assets, provenance, source hashes, recorded
exclusions and quality rejections, and a completion marker. The independent
validator rereads every source triplet and output asset, verifies hashes, arrays,
split membership, all exclusions/rejections, exact accounting, and codec error.
It performs no training-loader checks, training, or publication.

Final acceptance passed both the original independent validator and this PR's
`validate_diode_train.py`: every retained, excluded, and rejected source row was
checked, together with all **21,356** output assets. Valid-pixel weighted mean
encoding error is **0.002854490 m** and maximum error is **0.049964905 m**.
Minimum per-frame valid coverage is **98.0018616%**; aggregate invalid pixels are
**0.1959770%**. These are native data/codec measurements, not training results.

## Published DIODE component

- Dataset: [Melo111/open-vision-banana-depth](https://huggingface.co/datasets/Melo111/open-vision-banana-depth).
- Component: `diode-indoor-train-additions-v1`, **5,339 train / 0 val / 0 test**.
- Immutable payload revision: `9ed5e65718db68a6cdd000754de681336ab18179`.
- [Payload files](https://huggingface.co/datasets/Melo111/open-vision-banana-depth/tree/9ed5e65718db68a6cdd000754de681336ab18179/diode-indoor-train-additions-v1):
  **45 TAR shards, 56 files, 23,961,280,573 bytes**. Allow at least 50 GB for
  both the downloaded package and extracted assets.
- Format: `diode_native_tar_v1`, native **1024 x 768**, float32 metric-depth NPY,
  original RGB, encoded depth target, and validity mask. This is a standalone
  native component, **not** `res_bundle`; do not use `download_res_dataset.py`.
- `pairs.jsonl` SHA-256:
  `021e3f028f7d785d064a4a2b2e46a97256e3f66af455c014d59863ac86fc9916`.
- `metadata.json` SHA-256:
  `a28ba11dc7bf559a40fe5308e6596b25e0201aadae1ca981f48f334b583432d4`.

All 56 files were downloaded anonymously to the training server and checked by
SHA-256. Safe extraction verified every member of all 45 TARs and all 21,356
asset hashes. Sixteen deterministic pairs were additionally decoded in all four
roles. This readback used `hf-mirror.com`, with bytes bound to the fixed HF
revision and trusted release hashes; it is not a second full-source pixel audit.
The two production-side full native-source audits are described above.
The historical producer label in the manifest is preserved for provenance;
all published rows are `source_dataset=diode_indoor`. No NYU is included.

### Download and verify on the consumer server

Run from the repository root with `huggingface_hub` available. Download only the
component at the pinned payload commit; this does not change training settings.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Melo111/open-vision-banana-depth",
    repo_type="dataset",
    revision="9ed5e65718db68a6cdd000754de681336ab18179",
    allow_patterns=["diode-indoor-train-additions-v1/**"],
    local_dir="data/diode-hf-release",
    token=False,
)
PY

COMPONENT=data/diode-hf-release/diode-indoor-train-additions-v1
(
  cd "$COMPONENT" &&
  printf '%s  SHA256SUMS\n' \
    d013046f95d0f8e7b277ac661b8d92ba3b9c6437b43e980f923bb2d3f787e869 \
    | sha256sum --check - &&
  sha256sum --check SHA256SUMS
) && python "$COMPONENT/diode_native_release.py" verify "$COMPONENT" --extract
```

The two checksum commands must both pass before executing the downloaded helper.
If direct HF access is unavailable, the verified server readback used
`HF_ENDPOINT=https://hf-mirror.com` for the Python download command, with the same
revision and checksum checks. The helper requires no original 86.75 GB archive.
It checks metadata, manifests, shards, every asset's size/hash, and safe paths.
Expected helper output includes `status=passed`, `pairs=5339`, `assets=21356`,
`shards=45`, and `extracted=true`.

`snapshot_download` preserves the component subdirectory. After extraction the
manifest is **`$COMPONENT/pairs.jsonl`**, with `images/`, `targets/`,
`metric_depth/`, and `valid_masks/` beneath the same component directory. Keep
these directories in place. The producer-only `validate_diode_train.py` command
requires the original source and exclusion manifests; use the shipped consumer
helper for downloaded-release verification instead.

## Optional offline composition

Combine only the clean Hypersim and new DIODE **training** components in a new
output directory. This command performs another full native-quality scan and
hashes references; it never modifies source assets or launches training.

```bash
python scripts/prepare_clean_depth.py \
  --manifest data/hypersim-observed98-v2/pairs.jsonl \
  --manifest "$COMPONENT/pairs.jsonl" \
  --output data/clean-depth-with-diode \
  --expected-depth-rejections 0
```

The output is **`data/clean-depth-with-diode/pairs.jsonl`**, containing
**12,201 + 5,339 = 17,540 train-only pairs**, plus audit files and `COMPLETE.json`.
The complete command was verified with the restored HF component and clean
Hypersim in a separate server directory: 17,540 rows scanned and retained, zero
exclusions, all train, no NYU, unchanged source manifests, and every completion
hash binding verified. Output hashes depend on local absolute asset paths.
Its asset references are absolute, so preserve both original component locations.
The new DIODE component does not include the existing 996- and 195-pair DIODE
components. If needed, pass their already verified native-backed training
manifests as additional `--manifest` inputs; they are not included in 17,540.

**Keep existing validation manifests separate and unchanged.** Explicit val/test
rows are excluded rather than preserved or relabelled by this cleaner. Do not
replace a mixed train/val input with this train-only output and assume validation
is still present. Identical same-task duplicates are excluded; conflicting IDs
and train/held-out overlap fail. Non-depth training rows are preserved. Unlabelled
legacy rows are assumed train and counted in the audit, not certified as official
train. Do not supply an older combined NYU/DIODE manifest.

Select the new manifest only in a later explicitly configured run; no training
configuration or processing changes are part of this workflow. The existing
strict paper-protocol builder intentionally excludes DIODE; this release does not
automatically switch that protocol. If real DIODE train is later used, its
validation set becomes same-source held-out evaluation, not synthetic-only
zero-shot evaluation.

## Focused checks

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m pytest \
  tests/test_clean_depth.py tests/test_diode_train.py \
  tests/test_validate_diode_train.py
```

The 98% threshold is a data-quality policy, not a proven optimum or a claim of
model accuracy gains. No training or benchmark result is claimed.
