# Clean DIODE indoor training data

NYU is not included. Existing source archives and builds are left unchanged.

Run `python scripts/prepare_diode_train.py --source diode --input /path/to/train.tar --official-metadata /path/to/diode_meta.json --output-dir /new/build --exclude-manifest /path/to/existing996.jsonl --exclude-manifest /path/to/existing195.jsonl`.

The default gates require at least 98% native valid pixels and at most 0.05 metres maximum valid-pixel RGB codec roundtrip error. Only stricter overrides are accepted. No resize, interpolation, inpainting or depth clipping is performed. Metric float32 arrays and native masks are retained. Invalid target pixels are black; this does not add masked training loss.

Official metadata must contain 8574 unique indoor train IDs disjoint from validation IDs. Native triplets must match the official scene/scan hierarchy. Both existing manifests must be supplied to deduplicate the previous 996 and 195 pairs. The builder does not infer their locations. Incomplete sources fail unless `--allow-partial` is explicitly selected; partial outputs must not be presented as complete-source builds.

Each build is immutable. `--resume` verifies identical configuration and existing artifact bytes; changed source/configuration requires a new output directory. Source hashes, exclusions, quality rejections, four assets per retained row, and a completion marker are recorded.

Run `python scripts/validate_diode_train.py /new/build --official-metadata /path/to/diode_meta.json`. This independently reads every source triplet and output asset, checks hashes and native arrays, official IDs, all exclusions and quality rejections, exact accounting and codec error, then runs CPU loader smoke checks. A passing partial audit is explicitly marked partial. It does not train or publish data.

## Official source and release acceptance

- Source and MIT notice: <https://diode-dataset.org/>.
- Official archive: <https://diode-dataset.s3.amazonaws.com/train.tar.gz>,
  86,747,209,190 bytes; official MD5 `3a94632398fe1d002d89f11743f748b1`.
- Official `diode_meta.json`: DIODE devkit revision
  `8b1765b7d801a5f5e2877c434ffe164e62ce8c90`.

Verify the whole archive checksum before extracting `train/indoors/`. The builder
accepts that extracted tree or an uncompressed TAR, not a compressed TAR. There
are 8,574 official indoor train frames; after the 996 + 195 existing exclusions,
at most 7,383 additions remain before quality filtering. Do not report that upper
bound as a retained count. The observed 52-frame quality pilot retained 10 frames;
its ratio must not be extrapolated to the full dataset.

Final acceptance requires full-source accounting, an independent passing asset
and source audit, and pixel-weighted mean RGB encoding error below 0.01 m. The
float32 metric array preserves the source measurements even though the training
RGB target is quantized. No dataset publication, model run or accuracy gain is
claimed by this preparation code. Validation is same-source held-out evaluation
once DIODE train is included; it is no longer a synthetic-only zero-shot protocol.