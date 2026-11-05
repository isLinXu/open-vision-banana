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
Priority remains `rewritten_text`, then `text`, then `query`: the highest-priority
nonempty field supplies the queries (rewrites replace raw text, not append to it).
A list is validated as nonempty strings and **all entries are expanded into
separate evaluation records**, in source order, sharing the image and target mask.
Repeated entries are retained. A string produces one record. Invalid text fails
instead of becoming a stringified object. Records carry `image_id` and `query_index`
for auditing; neither changes metric weights.

The default ReasonSeg protocol is `all_queries_v1`. Only query-level statistics
are reported: `gIoU` (also `IoU`) is the arithmetic mean of the IoU of every query;
`cIoU` is the sum of intersections divided by the sum of unions over those queries.
`Precision@0.5` is the fraction of queries whose IoU is at least 0.5. `samples`
counts evaluated queries, not unique images. There is no image-level averaging.
An image with three queries therefore contributes three equally weighted gIoU
observations. Shards merge raw sums, preserving the same query weights.

`--max-samples` limits expanded queries, and can stop within an image's list;
omit it for full evaluation. Referring evaluation requires `--batch-size 1`.
The paper benchmark evaluator fails on nonfinite generation by default rather
than silently reporting a complete result with missing queries.

For historical subset comparisons only, pass `--reasonseg-query-mode first`
(`first_queries_v1`). Preflight and execution use the same selection. Metadata
records the query mode, evaluation protocol and independent prompt version.
Scores from first-query and all-query runs are different protocols and must not
be presented as the same benchmark setting.

Historical checkpoints are not automatically relabeled as trained with v2.
Changing their evaluation prompt is a prompt ablation; keep checkpoint, samples,
seed, decoding and mask metrics fixed when comparing versions.
