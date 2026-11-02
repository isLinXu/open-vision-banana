#!/usr/bin/env python3
"""Fixed development-set metric evaluation for checkpoint selection (issue #97).

The dev set is drawn only from the frozen held-out ``val`` rows of the training pool (never from a
benchmark test split or the training release), so it can be used to pick LRs, ranks and checkpoints.

Subcommands:

* ``index``: freeze a dev index. Rows are ordered by a salted sha256 of their identity and allocated
  per source in proportion to the source's share of the task, so the index is a pure function of the
  manifest, the counts and the salt.
* ``eval``: run one shard of the index through a model (the row's own training prompt, one image per
  call, seed = base seed + index position) and write one JSON line per sample with additive sums.
* ``merge``: check that every index entry was scored exactly once and aggregate the sums.

Depth GT is the stored cube-edge metric PNG decoded to metres at its native resolution, then given
the dataset's own resize/centre-crop geometry with nearest sampling (resizing the RGB first would
blend palette colours into spurious depths). Valid GT is (0.001, 80] m and predictions are clamped
to the same range (KITTI / ETH3D convention); a 10 m cap (NYU convention) is reported alongside, as
are the share of predicted pixels outside the range and ``gt_floor``: the canvas-resized GT RGB
decoded directly, against the geometric GT (what a perfect RGB copy of the training target scores).
Raw metrics are primary; median-aligned metrics are structure diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_COUNTS = "depth=512,semantic=256,instance=256,referring=256,generation=64"
MAX_DEPTH, NYU_CAP = 80.0, 10.0
DEPTH_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, math.inf))
FORMAT = "ovb_dev_index_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_counts(spec: str) -> dict[str, int]:
    counts = {}
    for part in spec.split(","):
        task, _, value = part.partition("=")
        if not task or not value.isdigit() or int(value) <= 0:
            raise ValueError(f"Bad count spec: {part!r}")
        counts[task.strip()] = int(value)
    return counts


def row_source(row: dict) -> str:
    return str(
        row.get("target_kind")
        or row.get("source_dataset")
        or row.get("source")
        or row.get("style")
        or "-"
    )


def row_key(row: dict) -> str:
    return "\x1f".join(
        str(row.get(k, "")) for k in ("task", "input", "target", "prompt", "sample_id")
    )


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, capped by availability."""
    total = min(total, sum(sizes.values()))
    pool = sum(sizes.values())
    quota = {k: total * n / pool for k, n in sizes.items()}
    take = {k: min(sizes[k], int(q)) for k, q in quota.items()}
    order = sorted(sizes, key=lambda k: (-(quota[k] - int(quota[k])), k))
    while sum(take.values()) < total:
        for k in order:
            if sum(take.values()) < total and take[k] < sizes[k]:
                take[k] += 1
    return take


def build_index(manifest: Path, counts: dict[str, int], salt: str) -> dict:
    by_task: dict[str, dict[str, list[tuple[str, int, dict]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with manifest.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("split") != "val" or row.get("task") not in counts:
                continue
            order = hashlib.sha256(f"{salt}\x1f{row_key(row)}".encode()).hexdigest()
            by_task[row["task"]][row_source(row)].append((order, line_no, row))
    entries = []
    report = {}
    for task, total in counts.items():
        sources = by_task.get(task, {})
        if not sources:
            raise ValueError(f"No val rows for task {task}")
        take = allocate({s: len(v) for s, v in sources.items()}, total)
        picked = []
        for source, rows in sorted(sources.items()):
            picked += [
                (order, line_no, row, source)
                for order, line_no, row in sorted(rows)[: take[source]]
            ]
        picked.sort(key=lambda x: x[0])
        report[task] = {
            "requested": total,
            "selected": len(picked),
            "available": sum(map(len, sources.values())),
            "by_source": take,
        }
        for order, line_no, row, source in picked:
            entries.append(
                {
                    "task": task,
                    "line": line_no,
                    "source": source,
                    "order": order,
                    "input": row["input"],
                    "target": row["target"],
                    "prompt": row.get("prompt", ""),
                }
            )
    for position, entry in enumerate(entries):
        entry["position"] = position
    return {
        "format": FORMAT,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "salt": salt,
        "counts": counts,
        "tasks": report,
        "entries": entries,
    }


def load_index(path: Path, manifest: Path) -> dict:
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("format") != FORMAT:
        raise ValueError("Not a dev index")
    if sha256_file(manifest) != index["manifest_sha256"]:
        raise ValueError("Manifest differs from the one the dev index was frozen on")
    return index


def shard_entries(entries: list[dict], count: int, shard: int) -> list[dict]:
    if not 0 <= shard < count:
        raise ValueError("shard index out of range")
    return [e for e in entries if e["position"] % count == shard]


# ---------------------------------------------------------------- metrics (additive sums per sample)


def depth_geometry(depth, input_size: tuple[int, int], resolution: int, aspect_preserving: bool):
    """Apply ManifestVisionDataset's val resize/centre crop to a [1, 1, H, W] metric map (nearest)."""
    import torch.nn.functional as F

    w, h = input_size
    r = resolution
    if depth.shape[-1] != w or depth.shape[-2] != h:
        depth = F.interpolate(depth, size=(h, w), mode="nearest")
    if not aspect_preserving or (w, h) == (r, r) or min(w, h) <= 0:
        return F.interpolate(depth, size=(r, r), mode="nearest")
    scale = r / float(min(w, h))
    new_w, new_h = max(r, int(round(w * scale))), max(r, int(round(h * scale)))
    depth = F.interpolate(depth, size=(new_h, new_w), mode="nearest")
    left, top = (new_w - r) // 2, (new_h - r) // 2
    return depth[..., top : top + r, left : left + r]


def depth_sums(pred, gt, cap: float = MAX_DEPTH) -> dict:
    import torch

    finite = torch.isfinite(pred)
    valid = torch.isfinite(gt) & (gt > 1e-3) & (gt <= cap)
    out = {
        "pixels": int(valid.sum()),
        "pred_out_of_range": int(((pred > cap) | (pred < 1e-3) | ~finite)[valid].sum()),
    }
    p = torch.nan_to_num(pred, nan=cap, posinf=cap, neginf=1e-3)[valid].double().clamp(1e-3, cap)
    g = gt[valid].double()
    n = int(g.numel())
    if n == 0:
        return out
    err = p - g
    out.update(
        abs_rel=float((err.abs() / g).sum()),
        sq_err=float((err**2).sum()),
        delta1=int((torch.maximum(p / g, g / p) < 1.25).sum()),
        log_err=float((p.log() - g.log()).sum()),
        median_ratio=float(p.median() / g.median()),
    )
    scale = g.median() / p.median()
    pa = p * scale
    out.update(
        aligned_abs_rel=float(((pa - g).abs() / g).sum()),
        aligned_delta1=int((torch.maximum(pa / g, g / pa) < 1.25).sum()),
    )
    image = {
        "abs_rel": out["abs_rel"] / n,
        "rmse": math.sqrt(out["sq_err"] / n),
        "delta1": out["delta1"] / n,
    }
    bins = {}
    for low, high in DEPTH_BINS:
        m = (g >= low) & (g < high) if math.isfinite(high) else g >= low
        k = int(m.sum())
        if k:
            bins[f"{low:g}-{high:g}"] = {
                "pixels": k,
                "abs_rel": float((err[m].abs() / g[m]).sum()),
                "sq_err": float((err[m] ** 2).sum()),
            }
    out["bins"] = bins
    out["per_image"] = image
    return out


def referring_sums(pred_mask, gt_mask) -> dict:
    inter = int((pred_mask & gt_mask).sum())
    union = int((pred_mask | gt_mask).sum())
    return {"intersection": inter, "union": union, "iou": inter / union if union else 1.0}


def semantic_sums(pred_labels, gt_labels, names: list[str]) -> dict:
    per_class = {}
    for i, name in enumerate(names):
        p, g = pred_labels == i, gt_labels == i
        union = int((p | g).sum())
        if union:
            per_class[name] = [int((p & g).sum()), union]
    return {
        "per_class": per_class,
        "correct": int((pred_labels == gt_labels).sum()),
        "pixels": int(gt_labels.numel()),
    }


def aggregate(records: list[dict]) -> dict:
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    out = {}
    for task, rows in sorted(by_task.items()):
        ok = [r for r in rows if r.get("status") == "ok"]
        summary = {"samples": len(rows), "scored": len(ok), "failed": len(rows) - len(ok)}
        if task == "depth":
            s = [r["metrics"] for r in ok if r["metrics"].get("pixels")]
            n = sum(m["pixels"] for m in s)
            if n:
                summary["raw"] = {
                    "abs_rel": sum(m["abs_rel"] for m in s) / n,
                    "rmse": math.sqrt(sum(m["sq_err"] for m in s) / n),
                    "delta1": sum(m["delta1"] for m in s) / n,
                    "log_scale_bias": sum(m["log_err"] for m in s) / n,
                }
                summary["raw_per_image_mean"] = {
                    k: sum(m["per_image"][k] for m in s) / len(s)
                    for k in ("abs_rel", "rmse", "delta1")
                }
                summary["median_aligned_diagnostic"] = {
                    "abs_rel": sum(m["aligned_abs_rel"] for m in s) / n,
                    "delta1": sum(m["aligned_delta1"] for m in s) / n,
                }
                ratios = sorted(m["median_ratio"] for m in s)
                summary["median_pred_over_gt"] = ratios[len(ratios) // 2]
                bins = defaultdict(lambda: [0, 0.0, 0.0])
                for m in s:
                    for name, b in m["bins"].items():
                        bins[name][0] += b["pixels"]
                        bins[name][1] += b["abs_rel"]
                        bins[name][2] += b["sq_err"]
                summary["distance_bins_m"] = {
                    name: {"pixel_share": k / n, "abs_rel": a / k, "rmse": math.sqrt(q / k)}
                    for name, (k, a, q) in sorted(
                        bins.items(), key=lambda kv: float(kv[0].split("-")[0])
                    )
                }
                summary["pred_out_of_range_share"] = sum(m["pred_out_of_range"] for m in s) / n
                capped = [
                    r["metrics_cap10"] for r in ok if r.get("metrics_cap10", {}).get("pixels")
                ]
                n10 = sum(m["pixels"] for m in capped)
                if n10:
                    summary["raw_cap10m"] = {
                        "abs_rel": sum(m["abs_rel"] for m in capped) / n10,
                        "rmse": math.sqrt(sum(m["sq_err"] for m in capped) / n10),
                        "delta1": sum(m["delta1"] for m in capped) / n10,
                    }
                floor = [r["gt_floor"] for r in ok if r["gt_floor"].get("pixels")]
                nf = sum(m["pixels"] for m in floor)
                if nf:
                    summary["gt_floor"] = {
                        "abs_rel": sum(m["abs_rel"] for m in floor) / nf,
                        "rmse": math.sqrt(sum(m["sq_err"] for m in floor) / nf),
                        "delta1": sum(m["delta1"] for m in floor) / nf,
                    }
        elif task == "referring":
            i = sum(r["metrics"]["intersection"] for r in ok)
            u = sum(r["metrics"]["union"] for r in ok)
            summary.update(
                cIoU=i / u if u else None,
                gIoU=sum(r["metrics"]["iou"] for r in ok) / len(ok) if ok else None,
            )
        elif task == "semantic":
            classes = defaultdict(lambda: [0, 0])
            for r in ok:
                for name, (inter, union) in r["metrics"]["per_class"].items():
                    classes[name][0] += inter
                    classes[name][1] += union
            ious = {k: i / u for k, (i, u) in classes.items() if u}
            px = sum(r["metrics"]["pixels"] for r in ok)
            summary.update(
                mIoU=sum(ious.values()) / len(ious) if ious else None,
                classes=len(ious),
                pixel_accuracy=sum(r["metrics"]["correct"] for r in ok) / px if px else None,
            )
        elif task == "instance":
            keys = sorted({k for r in ok for k in r["metrics"]})
            summary.update(
                {k: sum(r["metrics"].get(k, 0.0) for r in ok) / len(ok) for k in keys} if ok else {}
            )
        elif task == "generation":
            keys = sorted({k for r in ok for k in r["metrics"]})
            summary["proxy"] = (
                {k: sum(r["metrics"][k] for r in ok) / len(ok) for k in keys} if ok else {}
            )
            summary["note"] = (
                "proxy only (CLIP image/text similarity); not a GenAI-Bench or human win rate"
            )
        out[task] = summary
    return out


# ---------------------------------------------------------------- inference


def to_pil(tensor):
    """[1, 3, H, W] in [0, 1] -> PIL RGB."""
    from PIL import Image

    array = tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    return Image.fromarray(array.round().astype("uint8"))


def run_eval(args) -> int:
    import numpy as np
    import torch
    from PIL import Image

    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    from open_vision_banana.core.referring import decode_referring_mask, resolve_referring_encoding
    from open_vision_banana.core.segmentation import SegmentationDecoder
    from open_vision_banana.evaluation.instance import evaluate_instance
    from open_vision_banana.inference.pipeline import VisionBanana
    from open_vision_banana.training.trainer import ManifestVisionDataset

    index = load_index(args.index, args.manifest)
    entries = shard_entries(index["entries"], args.num_shards, args.shard_index)
    done = set()
    if args.output_jsonl.exists():
        # Failed samples are retried; scored ones are not recomputed.
        records = [
            json.loads(line) for line in args.output_jsonl.read_text().splitlines() if line.strip()
        ]
        done = {r["position"] for r in records if r.get("status") == "ok"}
    entries = [e for e in entries if e["position"] not in done]
    tasks = {e["task"] for e in index["entries"]}
    dataset = ManifestVisionDataset(
        args.manifest,
        args.resolution,
        "val",
        tasks,
        aspect_preserving_crop=True,
        data_root=args.manifest.parent,
    )
    lookup = {}
    for i, s in enumerate(dataset.samples):
        lookup[(s["task"], str(s["image_path"]), str(s["target_path"]), s["prompt"])] = i
    rows = {}
    with args.manifest.open(encoding="utf-8") as stream:
        wanted = {e["line"] for e in entries}
        for line_no, line in enumerate(stream, 1):
            if line_no in wanted:
                rows[line_no] = json.loads(line)
    from open_vision_banana.data.manifest import resolve_manifest_path

    model = VisionBanana.from_pretrained(
        model_name=args.model_name, lora_path=args.lora_path, device="cuda", dtype=torch.bfloat16
    )
    generator = model.generator
    mapper = DepthRGBMapper(colormap="cube_edge")
    seg = SegmentationDecoder()
    clip = None
    if "generation" in tasks:
        from transformers import CLIPModel, CLIPProcessor

        clip = (
            CLIPModel.from_pretrained(args.clip_model).to("cuda").eval(),
            CLIPProcessor.from_pretrained(args.clip_model),
        )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("a", encoding="utf-8") as out:
        for entry in entries:
            row = rows[entry["line"]]
            image_path = resolve_manifest_path(row["input"], args.manifest)
            target_path = resolve_manifest_path(row["target"], args.manifest)
            key = (entry["task"], str(image_path), str(target_path), row.get("prompt", ""))
            record = {
                "position": entry["position"],
                "task": entry["task"],
                "line": entry["line"],
                "source": entry["source"],
            }
            started = time.perf_counter()
            try:
                if key not in lookup:
                    raise KeyError("index entry not found in the val dataset")
                item = dataset[lookup[key]]
                image = item["image"].unsqueeze(0).to("cuda")
                target = item["target_rgb"].unsqueeze(0).to("cuda")
                with torch.no_grad():
                    pred = generator.generate(
                        prompt=item["prompt"],
                        image=image,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        strength=args.strength,
                        height=args.resolution,
                        width=args.resolution,
                        seed=args.seed + entry["position"],
                    )
                pred = pred.float().to("cuda")
                if pred.shape[-2:] != target.shape[-2:]:
                    pred = torch.nn.functional.interpolate(
                        pred, size=target.shape[-2:], mode="bilinear", align_corners=False
                    )
                if not torch.isfinite(pred).all():
                    raise ValueError("non-finite generator output")
                task = entry["task"]
                if task == "depth":
                    with Image.open(image_path) as source:
                        input_size = source.size
                    with Image.open(target_path) as native_png:
                        native = np.asarray(native_png.convert("RGB"), np.float32) / 255.0
                    native_m = mapper.rgb_to_depth(
                        torch.from_numpy(native).permute(2, 0, 1).unsqueeze(0).to("cuda")
                    )
                    gt = depth_geometry(
                        native_m, input_size, args.resolution, dataset.aspect_preserving_crop
                    )
                    pred_m = mapper.rgb_to_depth(pred.clamp(0, 1))
                    record["metrics"] = depth_sums(pred_m, gt)
                    record["metrics_cap10"] = depth_sums(pred_m, gt, NYU_CAP)
                    record["gt_floor"] = depth_sums(mapper.rgb_to_depth(target), gt)
                elif task == "referring":
                    encoding = resolve_referring_encoding(item.get("encoding"), (255, 255, 255))
                    pm = (
                        decode_referring_mask(pred.clamp(0, 1), encoding=encoding).squeeze(1).bool()
                    )
                    gm = (
                        decode_referring_mask(
                            target, encoding=encoding, legacy_largest_component=False
                        )
                        .squeeze(1)
                        .bool()
                    )
                    record["metrics"] = referring_sums(pm, gm)
                elif task == "semantic":
                    mapping = row.get("mapping")
                    if isinstance(mapping, str):
                        import ast

                        mapping = ast.literal_eval(mapping)
                    cmap = {k: tuple(int(c) for c in v) for k, v in mapping.items()}
                    names = list(cmap)
                    record["metrics"] = semantic_sums(
                        seg.decode_semantic(pred.clamp(0, 1), cmap).squeeze(1),
                        seg.decode_semantic(target, cmap).squeeze(1),
                        names,
                    )
                elif task == "instance":
                    background = item.get("background_color")
                    if isinstance(background, str):
                        background = json.loads(background)
                    color = tuple(int(c) for c in background) if background else (0, 0, 0)
                    pi = seg.decode_instance(pred.clamp(0, 1), background_color=color)[0]
                    gi = seg.decode_instance(target, background_color=color)[0]
                    record["metrics"] = {k: float(v) for k, v in evaluate_instance(pi, gi).items()}
                elif task == "generation":
                    model_c, proc = clip
                    inputs = proc(
                        text=[item["prompt"]],
                        images=[to_pil(pred), to_pil(target)],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to("cuda")
                    with torch.no_grad():
                        img = torch.nn.functional.normalize(
                            model_c.get_image_features(pixel_values=inputs["pixel_values"]), dim=-1
                        )
                        txt = torch.nn.functional.normalize(
                            model_c.get_text_features(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                            ),
                            dim=-1,
                        )
                    record["metrics"] = {
                        "clip_image_to_reference": float(img[0] @ img[1]),
                        "clip_text_score": float(img[0] @ txt[0]),
                        "reference_clip_text_score": float(img[1] @ txt[0]),
                    }
                record["status"] = "ok"
            except Exception as exc:  # recorded, never silently dropped
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            record["seconds"] = round(time.perf_counter() - started, 2)
            out.write(json.dumps(record) + "\n")
            out.flush()
    return 0


def run_merge(args) -> int:
    index = json.loads(args.index.read_text(encoding="utf-8"))
    records = {}
    duplicates = 0
    for path in args.shards:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["position"] in records and records[r["position"]].get("status") == "ok":
                duplicates += 1
                continue
            records[r["position"]] = r
    expected = {e["position"] for e in index["entries"]}
    missing = sorted(expected - set(records))
    result = {
        "index_manifest_sha256": index["manifest_sha256"],
        "index_salt": index["salt"],
        "coverage": {
            "expected": len(expected),
            "scored_or_failed": len(records),
            "missing": len(missing),
            "duplicates_ignored": duplicates,
            "failed": Counter(r["task"] for r in records.values() if r.get("status") != "ok"),
        },
        "metrics": aggregate(list(records.values())),
        "complete": not missing and all(r.get("status") == "ok" for r in records.values()),
    }
    if args.meta:
        result["meta"] = json.loads(args.meta.read_text(encoding="utf-8"))
    args.output_json.write_text(json.dumps(result, indent=1, default=dict) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, default=dict))
    return 0 if result["complete"] else 3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("index")
    i.add_argument("--manifest", type=Path, required=True)
    i.add_argument("--counts", default=DEFAULT_COUNTS)
    i.add_argument("--salt", default="ovb-dev-v1")
    i.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("eval")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--index", type=Path, required=True)
    e.add_argument("--model-name", default="qwen-image-2.1")
    e.add_argument("--lora-path")
    e.add_argument("--resolution", type=int, default=512)
    e.add_argument("--num-inference-steps", type=int, default=50)
    e.add_argument("--guidance-scale", type=float, default=4.0)
    e.add_argument("--strength", type=float, default=0.85)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--shard-index", type=int, default=0)
    e.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    e.add_argument("--output-jsonl", type=Path, required=True)
    m = sub.add_parser("merge")
    m.add_argument("--index", type=Path, required=True)
    m.add_argument("--shards", type=Path, nargs="+", required=True)
    m.add_argument("--meta", type=Path)
    m.add_argument("--output-json", type=Path, required=True)
    a = p.parse_args(argv)
    if a.command == "index":
        index = build_index(a.manifest.resolve(), parse_counts(a.counts), a.salt)
        if a.output.exists():
            raise FileExistsError("A frozen dev index is never replaced")
        a.output.write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"sha256": sha256_file(a.output), "tasks": index["tasks"]}, indent=1))
        return 0
    if a.command == "eval":
        a.manifest = a.manifest.resolve()
        return run_eval(a)
    return run_merge(a)


if __name__ == "__main__":
    raise SystemExit(main())
