# Referring query prompt versions

New binary referring training, inference and evaluation default to `query_v2`:

```text
Identify the object or region referred to by the following query in the image.
Query: {query}
Render its segmentation mask in solid white and all other pixels in black.
```

The query retains its words, punctuation and leading article (outer whitespace
is stripped). Questions, instructions and noun phrases share this template.
Answer/COT fields are not added. `binary_white_black_v1` still identifies the
unchanged **pixel encoding**; prompt versions are independent:

| Version | Text policy |
|---|---|
| `query_v2` | Separate query block above; default for future runs |
| `fig4_v1` | Historical `The {expression} is represented by a solid white.` template |
| `segment_v0` | Historical `Segment {expression} in solid white...` template |

New RES/reasoning manifests record `prompt_version: query_v2`. Existing prompts
are validated against their expression and any declared version. Recognized
historical prompts with an expression are reconstructed at load time for the
selected training version; files on disk are not rewritten. Legacy prompt-only
rows remain readable but cannot be safely migrated without an expression.

Set `data.referring_prompt_version` in training YAML (`effective_config.json`
records it). Training and validation use the same selection. For historical
inference or paper benchmark comparisons pass `--referring-prompt-version
fig4_v1` (or `segment_v0`) to `scripts/inference.py` or
`scripts/evaluate_paper_benchmarks.py` / `scripts/evaluate_res_manifest.py`. In Python set
`model.referring_prompt_version` before prediction, or supply `prompt_version`
to `PromptBuilder.build_referring_prompt`. An explicitly supplied complete
prompt remains the caller's override. This option concerns binary referring
queries; legacy colored/multi-expression protocols remain separate.

ReasonSeg accepts string or list text in both sidecar JSON and `annotations.json`.
Priority remains `rewritten_text`, then `text`, then `query`. A list is validated
as nonempty strings and its first expression is selected deterministically:
**one evaluation record per image**, without joining alternative expressions or
serializing Python brackets/quotes. Invalid text fails instead of becoming a
stringified object. Paper benchmark metadata records selection and prompt version.

Historical checkpoints are not automatically relabeled as trained with v2.
Changing their evaluation prompt is a prompt ablation; keep checkpoint, samples,
seed, decoding and mask metrics fixed when comparing versions.
