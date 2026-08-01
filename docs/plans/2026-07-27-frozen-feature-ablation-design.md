# VAE-only versus VAE+SSD frozen-feature ablation design

## Objective and success criterion

Stage 6 proved that a 509,745-parameter head can memorize four depth maps from
frozen VAE and SSD features. Stage 7 asks a narrower generalization question:
do SSD intermediate features improve held-out relative-depth ordering beyond
what the VAE latent already preserves?

Use the first 32 declared depth train samples and first 16 official validation
samples, with no overlap. Train two arms for the same fixed 300-step budget at
128 px, seed 42, batch 8, AdamW LR `1e-3`, and the same loss. The SSD arm earns
promotion only if endpoint validation Pearson and Spearman each exceed the
VAE-only arm by at least 0.05. Validation MAE may not regress by more than 0.01.
The endpoint, not the best validation checkpoint, is the primary comparison so
validation is not used to choose a different training duration per arm.

## Fairness design

Three alternatives were considered:

1. Build a smaller one-input VAE decoder. This is computationally efficient,
   but parameter count and fusion topology differ, confounding representation
   value with head capacity.
2. Replace SSD tensors with zero tensors before projection. This keeps the
   module graph but still allows learned projection biases to provide extra
   constants.
3. Use one identical six-slot head and apply a binary mask after each 1x1
   projection. The VAE-only arm keeps only `vae_latent`; all SSD slots become
   exact zero activations, including their biases. The VAE+SSD arm keeps every
   slot.

Select option 3. Both arms load the same initial state dict and consume the
same precomputed batch-index sequence. They share cached input tensors,
targets, optimizer type, LR schedule, loss weights, evaluation steps, and
checkpoint selection. Only the feature mask differs.

## Data flow

```text
32 declared train + 16 official val images
                  |
       frozen VAE/SSD extraction once
                  |
        shared six-level feature cache
             /                 \
 VAE mask [1,0,0,0,0,0]   full mask [1,1,1,1,1,1]
             \                 /
       identical FPN state + batch schedule
                  |
       endpoint train/val scalar metrics
                  |
  delta Pearson / delta Spearman / delta MAE
```

The shared extraction intentionally computes SSD features for both arms; the
VAE arm cannot read them after projection. This keeps input preprocessing and
feature numerical precision identical.

## Non-functional requirements

- **Performance:** one offline feature-extraction pass; both head runs should
  finish in minutes on MPS.
- **Memory:** features are cached to CPU one sample at a time; the base model is
  released before head optimization.
- **Reliability:** exact manifest split membership, duplicate/leak checks,
  identical initialization checksum, identical batch schedule checksum, and
  finite JSON output.
- **Compatibility:** no production inference or existing checkpoint contract
  changes; feature masks belong to research head checkpoints only.
- **Privacy/security:** local data and local cached weights; no network writes.

## Failure modes

| Failure | Mitigation |
|---|---|
| Validation sample leaks into train | assert disjoint manifest names before loading |
| Heads initialize differently | clone one template state and load it strictly into both arms |
| Batch order differs | precompute one CPU index schedule and reuse it |
| VAE-only sees projection bias from masked SSD slots | mask projected activation after normalization/activation |
| Best-step cherry-picking | gate on the fixed step-300 endpoint |
| SSD wins only on colour/range | gate on held-out Pearson and Spearman, with MAE guardrail |
| Single-seed result is marginal | do not promote unless both correlation deltas clear 0.05 |

## Expected artifacts

- `vae_only.pt` and `vae_ssd.pt`, each recording the mask and feature contract.
- `result.json` containing split names, matching checksums, full histories,
  endpoint train/val metrics, deltas, and the gate decision.
- Validation prediction montages for both arms.

## Outcome

Stage 7 passed at the fixed step-300 endpoint. VAE-only validation reached
Pearson/Spearman `0.1425/0.1629`; the exactly matched VAE+SSD arm reached
`0.3663/0.3682`, gains of `+0.2238/+0.2053`. Validation MAE also improved from
`0.2591` to `0.2402`. Both arms used 509,745 parameters, identical initial
state checksums, and one shared batch schedule checksum. Total runtime was
36.3 seconds including the single 48-image feature extraction pass.

SSD intermediate features therefore contribute held-out relative-depth
ordering beyond the VAE latent under this 32/16 gate. Absolute validation
correlations remain moderate and only one seed/subset was tested, so this is a
promotion to a larger discriminative validation stage, not a production or
generative result.
