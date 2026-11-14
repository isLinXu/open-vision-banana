#!/usr/bin/env python3
"""Export full diagnostics and a comparison table with distinct metric protocols.

Read a suite run directory, accepted dashboard records, or explicit full-result
JSON files. No inference and no reconstruction of missing legacy diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

CONTEXT = ["model", "repo", "revision", "step", "resolution", "dataset", "samples", "prompt", "query_mode",
           "reasonseg_annotation_mode", "dataset_fingerprint", "valid_pixels", "aligned_valid_pixels",
           "reasonseg_legacy_valid_pixels", "reasonseg_lisa_v1_valid_pixels", "source", "protocol"]
GROUPS = [
    ("原始／未对齐", ["AbsRel", "RMSE", "delta_1.25", "mIoU", "cIoU", "gIoU"]),
    ("逐图等权平均 _per_image", ["AbsRel_per_image", "RMSE_per_image", "delta_1.25_per_image"]),
    ("GT 中位数尺度对齐 _aligned", ["AbsRel_aligned", "RMSE_aligned", "delta_1.25_aligned"]),
    ("ReasonSeg lisa_v1（独立标注口径）", ["ReasonSeg_lisa_v1_gIoU", "ReasonSeg_lisa_v1_cIoU"]),
]
RATIOS = {"mIoU", "cIoU", "gIoU", "delta_1.25", "delta_1.25_per_image", "delta_1.25_aligned",
          "ReasonSeg_lisa_v1_gIoU", "ReasonSeg_lisa_v1_cIoU"}
SCORES = [key for _, keys in GROUPS for key in keys]
LEGACY_SOURCES = {"01e3d486630db36145a857273f318c2417d62f6b", "62c5b3de8c04e02a18946bf03076e675967d627a"}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten(child, f"{prefix}.{index}")
    elif value is None or isinstance(value, (int, float)):
        yield prefix, value if value is None or math.isfinite(value) else None


def comparison_row(record):
    metrics = record["metrics"]
    protocol = record.get("protocol", {})
    row = {key: record.get(key) for key in CONTEXT}
    row.update(dict(flatten(metrics)))
    row["prompt"] = protocol.get("referring_prompt_version", "未记录")
    row["query_mode"] = protocol.get("reasonseg_query_mode", "未记录")
    mode = protocol.get("reasonseg_annotation_mode")
    if record["dataset"] == "reasonseg" and not mode:
        revisions = {e.get("original_identity", {}).get("source_revision") for e in record.get("evidence", [])}
        if revisions & LEGACY_SOURCES:
            mode = "legacy_union"
    row["reasonseg_annotation_mode"] = mode if record["dataset"] == "reasonseg" else ""
    row["protocol"] = json.dumps(protocol, ensure_ascii=False, sort_keys=True)
    evidence = record.get("evidence", [])
    row["source"] = record.get("source") or (evidence[-1].get("path") if evidence else "未记录")
    if record["dataset"] == "reasonseg":
        variants = metrics.get("annotation_variants", {})
        row["reported_gIoU"] = metrics.get("gIoU")
        row["reported_cIoU"] = metrics.get("cIoU")
        # Explicit names prevent a corrected score from appearing in the old column.
        legacy = variants.get("legacy_union", metrics if mode == "legacy_union" else {})
        lisa = variants.get("lisa_v1", metrics if mode == "lisa_v1" else {})
        for key in ("gIoU", "cIoU"):
            row[key] = legacy.get(key)
            row["ReasonSeg_lisa_v1_" + key] = lisa.get(key)
        row["reasonseg_legacy_valid_pixels"] = legacy.get("valid_pixels")
        row["reasonseg_lisa_v1_valid_pixels"] = lisa.get("valid_pixels")
    # Missing diagnostics remain absent/blank; never derive them from raw scores.
    return row


def load_full_result(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    meta = data.get("metadata", {})
    identity = meta.get("external_run_identity", {})
    protocol = dict(meta.get("inference_protocol", {}))
    for key in ("referring_prompt_version", "reasonseg_query_mode", "reasonseg_annotation_mode", "reasonseg_annotation_variants"):
        if key in meta:
            protocol[key] = meta[key]
    for key, result in data["benchmarks"].items():
        if "shard" in result:
            raise ValueError(f"Merge shards before exporting: {path}")
        if not result.get("decoded_metrics"):
            continue
        yield dict(model=identity.get("model", path.parent.parent.name), step=identity.get("step"),
                   repo=identity.get("repo") or identity.get("adapter_repo") or identity.get("base_repo"),
                   revision=identity.get("revision") or identity.get("adapter_revision") or identity.get("base_revision"),
                   resolution=meta.get("resolution"), dataset=key.split("/", 1)[1],
                   samples=result["samples"], dataset_fingerprint=result.get("dataset_fingerprint"),
                   protocol=protocol, metrics=result["decoded_metrics"], source=str(path))


def fragment(rows):
    escape = html.escape
    labels = {"delta_1.25": "δ1 ↑", "delta_1.25_per_image": "δ1 ↑", "delta_1.25_aligned": "δ1 ↑",
              "ReasonSeg_lisa_v1_gIoU": "gIoU ↑", "ReasonSeg_lisa_v1_cIoU": "cIoU ↑"}
    headings = ['<th rowspan="2">模型 / 步数</th><th rowspan="2">数据集 / 画布</th><th rowspan="2">协议 / 样本</th>']
    headings += [f'<th colspan="{len(keys)}">{escape(label)}</th>' for label, keys in GROUPS]
    subhead = ''.join(f'<th>{escape(labels.get(key, key))}</th>' for key in SCORES)
    body = []
    for row in rows:
        context = [f'{row["model"]} / {row["step"] if row["step"] is not None else "未记录"}',
                   f'{row["dataset"]} / {row["resolution"]}',
                   f'{row["samples"]} 条 · {row["prompt"]} · {row["query_mode"]} · {row["reasonseg_annotation_mode"] or "—"}']
        cells = ['<td>' + escape(str(value)) + '</td>' for value in context]
        for key in SCORES:
            value = row.get(key)
            if value is None:
                shown = "待评测" if row["dataset"] == "reasonseg" and key.startswith("ReasonSeg_lisa_v1_") else "—"
            else:
                shown = f'{value * 100:.2f}%' if key in RATIOS else f'{value:.4f}'
            cells.append('<td>' + shown + '</td>')
        details = {key: row.get(key) for key in CONTEXT if key not in {"model", "dataset", "resolution", "step"}}
        cells[2] = cells[2].removesuffix('</td>') + '<details><summary>有效域与来源</summary><pre>' + escape(json.dumps(details, ensure_ascii=False, indent=2)) + '</pre></details></td>'
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return (
        '<section id="separate-protocol-metrics" style="margin:18px 0;padding:18px;background:#fff;border:1px solid #dce4df;border-radius:12px">'
        '<h2>独立口径指标：原始 / per_image / aligned / ReasonSeg lisa_v1</h2>'
        '<p>原始分数保留；ReasonSeg 的原始组 gIoU/cIoU 指 legacy_union，选择的主口径原值另存 reported_gIoU/cIoU。per_image 为逐图等权平均；aligned 使用 GT 中位数尺度对齐；'
        'lisa_v1 按官方标注处理 ignore/flag。新默认从同一次预测导出两种 ReasonSeg 评分；历史缺失结果显示待评测，不能由旧 IoU 推算。'
        '三种深度口径及两种 ReasonSeg 标注口径分别展示，不共用冠军排名；prompt、query、分辨率差异保留。</p>'
        '<p>主表仅展示重点列；完整指标、每类 IoU 和标注变体的 Precision/Recall/F-score 等均保留在配套 JSON/CSV。'
        'δ1、IoU 类指标显示百分数；RMSE 单位为米；未记录的诊断指标显示 —。</p>'
        '<details><summary>展开独立指标对照（' + str(len(rows)) + ' 条已记录结果）</summary>'
        '<div style="overflow:auto;max-height:720px"><table style="border-collapse:collapse;white-space:nowrap">'
        '<thead style="position:sticky;top:0;background:#eef4f1"><tr>' + ''.join(headings) + '</tr><tr>' + subhead + '</tr></thead>'
        '<tbody>' + ''.join(body) + '</tbody></table></div></details></section>'
    )


def export(records, output, fragment_path=None):
    rows = [comparison_row(r) for r in records]
    rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in ("model", "step", "resolution", "dataset", "prompt", "query_mode", "reasonseg_annotation_mode")))
    output.mkdir(parents=True, exist_ok=True)
    fields = CONTEXT + SCORES + sorted({k for row in rows for k in row} - set(CONTEXT) - set(SCORES))
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps(dict(rows=rows, records=records), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    content = fragment(rows)
    if fragment_path:
        fragment_path.write_text(content, encoding="utf-8")
    document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>独立评测指标</title><style>body{font:14px system-ui;background:#f4f7f5;margin:24px}td,th{padding:8px;border:1px solid #dce4df}pre{white-space:pre-wrap;max-width:520px}summary{cursor:pointer}</style></head><body>' + content + '</body></html>'
    (output / "comparison.html").write_text(document, encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records-json", type=Path)
    source.add_argument("--run-root", type=Path)
    source.add_argument("--result-json", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path)
    args = parser.parse_args()
    if args.records_json:
        records = json.loads(args.records_json.read_text(encoding="utf-8-sig"))["records"]
    else:
        paths = args.result_json or sorted(args.run_root.glob("*/*/full.json"))
        records = [record for path in paths for record in load_full_result(path)]
    rows = export(records, args.output, args.fragment)
    print(json.dumps(dict(records=len(rows),output=str(args.output))))


if __name__ == "__main__":
    main()
