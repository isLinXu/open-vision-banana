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
the dataset's own resize/centre-crop geometry with PIL-style nearest sampling (resizing the RGB first would
blend palette colours into spurious depths). Valid GT is (0.001, 80] m and predictions are clamped
to the same range (KITTI / ETH3D convention); a 10 m cap (NYU convention) is reported alongside, as
are the share of predicted pixels outside the range and ``gt_floor``: the canvas-resized GT RGB
decoded directly, against the geometric GT (what a perfect RGB copy of the training target scores).
Raw metrics are primary; median-aligned metrics are structure diagnostics only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.data.manifest import read_jsonl, resolve_manifest_path  # noqa: E402
from open_vision_banana.evaluation.results_io import write_results  # noqa: E402
from open_vision_banana.res_bundle import sha256_file  # noqa: E402

DEFAULT_COUNTS = "depth=512,semantic=256,instance=256,referring=256,generation=64"
MAX_DEPTH, NYU_CAP = 80.0, 10.0
DEPTH_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, math.inf))
FORMAT = "ovb_dev_index_v1"


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
    pool = sum(sizes.values())
    total = min(total, pool)
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
            if row["task"] == "generation" and row.get("input") in (None, "", row.get("target")):
                # An edit of its own reference would score reconstruction, not generation.
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
    """Apply ManifestVisionDataset's val resize/centre crop to a [1, 1, H, W] metric map.

    ``nearest-exact`` samples pixel centres like PIL's NEAREST, so the map lines up with the canvas.
    """
    import torch.nn.functional as F

    def resize(t, size):
        return F.interpolate(t, size=size, mode="nearest-exact")

    w, h = input_size
    r = resolution
    if depth.shape[-1] != w or depth.shape[-2] != h:
        depth = resize(depth, (h, w))
    if not aspect_preserving or (w, h) == (r, r) or min(w, h) <= 0:
        return resize(depth, (r, r))
    scale = r / float(min(w, h))
    new_w, new_h = max(r, int(round(w * scale))), max(r, int(round(h * scale)))
    depth = resize(depth, (new_h, new_w))
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
    pa = p * (g.median() / p.median())
    out.update(
        aligned_abs_rel=float(((pa - g).abs() / g).sum()),
        aligned_delta1=int((torch.maximum(pa / g, g / pa) < 1.25).sum()),
    )
    out["per_image"] = {
        "abs_rel": out["abs_rel"] / n,
        "rmse": math.sqrt(out["sq_err"] / n),
        "delta1": out["delta1"] / n,
    }
    bins = {}
    for low, high in DEPTH_BINS:
        m = (g >= low) & (g < high)
        k = int(m.sum())
        if k:
            bins[f"{low:g}-{high:g}"] = {
                "pixels": k,
                "abs_rel": float((err[m].abs() / g[m]).sum()),
                "sq_err": float((err[m] ** 2).sum()),
            }
    out["bins"] = bins
    return out


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


def _pooled(sums: list[dict]) -> dict | None:
    """Pixel-weighted AbsRel / RMSE / delta1 over per-sample depth sums."""
    sums = [m for m in sums if m.get("pixels")]
    n = sum(m["pixels"] for m in sums)
    if not n:
        return None
    return {
        "abs_rel": sum(m["abs_rel"] for m in sums) / n,
        "rmse": math.sqrt(sum(m["sq_err"] for m in sums) / n),
        "delta1": sum(m["delta1"] for m in sums) / n,
    }


def _mean_keys(metrics: list[dict]) -> dict:
    keys = sorted({k for m in metrics for k in m})
    return {k: sum(m.get(k, 0.0) for m in metrics) / len(metrics) for k in keys} if metrics else {}


def summarize_depth(ok: list[dict]) -> dict:
    s = [r["metrics"] for r in ok if r["metrics"].get("pixels")]
    raw = _pooled(s)
    if raw is None:
        return {}
    n = sum(m["pixels"] for m in s)
    bins = defaultdict(lambda: [0, 0.0, 0.0])
    for m in s:
        for name, b in m["bins"].items():
            bins[name][0] += b["pixels"]
            bins[name][1] += b["abs_rel"]
            bins[name][2] += b["sq_err"]
    ratios = sorted(m["median_ratio"] for m in s)
    return {
        "raw": {**raw, "log_scale_bias": sum(m["log_err"] for m in s) / n},
        "raw_per_image_mean": _mean_keys([m["per_image"] for m in s]),
        "median_aligned_diagnostic": {
            "abs_rel": sum(m["aligned_abs_rel"] for m in s) / n,
            "delta1": sum(m["aligned_delta1"] for m in s) / n,
        },
        "median_pred_over_gt": ratios[len(ratios) // 2],
        "distance_bins_m": {
            name: {"pixel_share": k / n, "abs_rel": a / k, "rmse": math.sqrt(q / k)}
            for name, (k, a, q) in sorted(bins.items(), key=lambda kv: float(kv[0].split("-")[0]))
        },
        "pred_out_of_range_share": sum(m["pred_out_of_range"] for m in s) / n,
        "raw_cap10m": _pooled([r.get("metrics_cap10", {}) for r in ok]),
        "gt_floor": _pooled([r["gt_floor"] for r in ok]),
    }


def summarize_referring(ok: list[dict]) -> dict:
    from open_vision_banana.evaluation.shards import merge_accumulator_states

    return merge_accumulator_states([r["metrics"] for r in ok]).compute() if ok else {}


def summarize_semantic(ok: list[dict]) -> dict:
    classes = defaultdict(lambda: [0, 0])
    for r in ok:
        for name, (inter, union) in r["metrics"]["per_class"].items():
            classes[name][0] += inter
            classes[name][1] += union
    ious = [i / u for i, u in classes.values() if u]
    px = sum(r["metrics"]["pixels"] for r in ok)
    return {
        "mIoU": sum(ious) / len(ious) if ious else None,
        "classes": len(ious),
        "pixel_accuracy": sum(r["metrics"]["correct"] for r in ok) / px if px else None,
    }


SUMMARIES = {
    "depth": summarize_depth,
    "referring": summarize_referring,
    "semantic": summarize_semantic,
    "instance": lambda ok: _mean_keys([r["metrics"] for r in ok]),
    "generation": lambda ok: {
        "proxy": _mean_keys([r["metrics"] for r in ok]),
        "note": "proxy only (CLIP image/text similarity); not a GenAI-Bench or human win rate",
    },
}


def aggregate(records: list[dict]) -> dict:
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task"]].append(r)
    out = {}
    for task, rows in sorted(by_task.items()):
        ok = [r for r in rows if r.get("status") == "ok"]
        out[task] = {
            "samples": len(rows),
            "scored": len(ok),
            "failed": len(rows) - len(ok),
            **SUMMARIES[task](ok),
        }
    return out


# ---------------------------------------------------------------- inference


def to_pil(tensor):
    """[1, 3, H, W] in [0, 1] -> PIL RGB."""
    from PIL import Image

    array = tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    return Image.fromarray(array.round().astype("uint8"))


def score_depth(ctx, sample, pred, target) -> dict:
    import numpy as np
    import torch

    s = sample.spec  # ManifestVisionDataset sample: disk path or tar key, as the dataset reads it
    input_size = ctx.open_rgb(s["image_path"], s.get("input_tar_key", "")).size
    native_png = ctx.open_rgb(s["target_path"], s.get("target_tar_key", ""))
    native = np.asarray(native_png, np.float32) / 255.0
    native_m = ctx.mapper.rgb_to_depth(
        torch.from_numpy(native).permute(2, 0, 1).unsqueeze(0).to(pred.device)
    )
    gt = depth_geometry(native_m, input_size, ctx.resolution, ctx.aspect_preserving)
    pred_m = ctx.mapper.rgb_to_depth(pred)
    return {
        "metrics": depth_sums(pred_m, gt),
        "metrics_cap10": depth_sums(pred_m, gt, NYU_CAP),
        "gt_floor": depth_sums(ctx.mapper.rgb_to_depth(target), gt),
    }


def score_referring(ctx, sample, pred, target) -> dict:
    from open_vision_banana.core.referring import decode_referring_mask, resolve_referring_encoding
    from open_vision_banana.evaluation.referring import ReferringAccumulator
    from open_vision_banana.evaluation.shards import accumulator_state

    encoding = resolve_referring_encoding(sample.item.get("encoding"), (255, 255, 255))
    pred_mask = decode_referring_mask(pred, encoding=encoding).squeeze(1).bool()
    gt_mask = decode_referring_mask(target, encoding=encoding, legacy_largest_component=False)
    accumulator = ReferringAccumulator()
    accumulator.update(pred_mask, gt_mask.squeeze(1).bool())
    return {"metrics": accumulator_state(accumulator)}


def score_semantic(ctx, sample, pred, target) -> dict:
    from open_vision_banana.training.trainer import parse_semantic_legend_from_prompt

    mapping = sample.spec.get("mapping") or sample.row.get("mapping")
    if isinstance(mapping, str):
        mapping = ast.literal_eval(mapping)
    if not mapping:
        mapping = parse_semantic_legend_from_prompt(sample.item["prompt"])
    if not mapping:
        raise ValueError("semantic row has no legend")
    cmap = {k: tuple(int(c) for c in v) for k, v in mapping.items()}
    labels = [ctx.seg.decode_semantic(x, cmap).squeeze(1) for x in (pred, target)]
    return {"metrics": semantic_sums(*labels, list(cmap))}


def score_instance(ctx, sample, pred, target) -> dict:
    from open_vision_banana.evaluation.instance import evaluate_instance

    background = sample.item.get("background_color")
    if isinstance(background, str):
        background = json.loads(background)
    color = tuple(int(c) for c in background) if background else (0, 0, 0)
    pred_inst, gt_inst = (
        ctx.seg.decode_instance(x, background_color=color)[0] for x in (pred, target)
    )
    return {"metrics": {k: float(v) for k, v in evaluate_instance(pred_inst, gt_inst).items()}}


def score_generation(ctx, sample, pred, target) -> dict:
    import torch

    model, processor = ctx.clip
    inputs = processor(
        text=[sample.item["prompt"]],
        images=[to_pil(pred), to_pil(target)],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(pred.device)
    # Projected features by hand: get_image_features returns a model output, not a tensor, on
    # transformers 5.x.
    with torch.no_grad():
        vision = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        text = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).pooler_output
    image = torch.nn.functional.normalize(model.visual_projection(vision), dim=-1)
    caption = torch.nn.functional.normalize(model.text_projection(text), dim=-1)
    return {
        "metrics": {
            "clip_image_to_reference": float(image[0] @ image[1]),
            "clip_text_score": float(image[0] @ caption[0]),
            "reference_clip_text_score": float(image[1] @ caption[0]),
        }
    }


SCORERS = {
    "depth": score_depth,
    "referring": score_referring,
    "semantic": score_semantic,
    "instance": score_instance,
    "generation": score_generation,
}


def resolve_weights(model_path: Path, checkpoint: Path | None, view_dir: Path) -> dict:
    """Pipeline dir and LoRA path for a base snapshot plus an optional LoRA or full checkpoint.

    A full checkpoint (``transformer/`` only) is served through a symlinked pipeline view of the
    base with its transformer swapped in.
    """
    if not (model_path / "model_index.json").is_file():
        raise ValueError(f"{model_path} is not a diffusers pipeline snapshot")
    weights = {"model_path": str(model_path), "checkpoint": None, "kind": "base", "lora_path": None}
    if checkpoint is None:
        return weights
    weights["checkpoint"] = str(checkpoint)
    meta = checkpoint / "metadata.json"
    weights["checkpoint_metadata_sha256"] = sha256_file(meta) if meta.is_file() else None
    if (checkpoint / "lora_adapter").is_dir():
        adapter = checkpoint / "lora_adapter" / "adapter_model.safetensors"
        return {
            **weights,
            "kind": "lora",
            "lora_path": str(checkpoint),
            "adapter_sha256": sha256_file(adapter),
        }
    if (checkpoint / "transformer").is_dir():
        view_dir.mkdir(parents=True, exist_ok=True)
        for item in model_path.iterdir():
            if item.name != "transformer":
                _link(view_dir / item.name, item.resolve())
        _link(view_dir / "transformer", (checkpoint / "transformer").resolve())
        files = sorted((checkpoint / "transformer").iterdir())
        return {
            **weights,
            "kind": "full",
            "model_path": str(view_dir),
            "transformer_bytes": {f.name: f.stat().st_size for f in files},
        }
    raise ValueError(f"{checkpoint} has neither lora_adapter/ nor transformer/")


def _link(link: Path, target: Path) -> None:
    if link.is_symlink() and link.resolve() == target:
        return
    if link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def eval_identity(args, weights: dict) -> dict:
    """Everything that determines a sample's score; records carry its hash."""
    identity = {
        "index_sha256": sha256_file(args.index),
        "model_name": args.model_name,
        "base_model": str(args.model_path),
        **{k: v for k, v in weights.items() if k != "model_path"},
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "seed": args.seed,
    }
    blob = json.dumps(identity, sort_keys=True).encode()
    return {**identity, "sha256": hashlib.sha256(blob).hexdigest()}


def dataset_prompt(row: dict, task: str) -> str:
    """The prompt ManifestVisionDataset gives this row (RES prompts are resolved, others defaulted)."""
    from open_vision_banana.data.res_contract import resolve_referring_prompt
    from open_vision_banana.training.trainer import TASK_PROMPTS

    if task == "referring":
        return resolve_referring_prompt(row)
    return row.get("prompt") or TASK_PROMPTS.get(task, "")


def run_eval(args) -> int:
    import torch

    from open_vision_banana.core.depth_mapper import DepthRGBMapper
    from open_vision_banana.core.segmentation import SegmentationDecoder
    from open_vision_banana.inference.pipeline import VisionBanana
    from open_vision_banana.training.trainer import ManifestVisionDataset

    index = load_index(args.index, args.manifest)
    entries = shard_entries(index["entries"], args.num_shards, args.shard_index)
    view = args.output_jsonl.parent / f"model_view.{args.output_jsonl.stem}"
    weights = resolve_weights(args.model_path, args.checkpoint, view)
    identity = eval_identity(args, weights)
    if args.output_jsonl.exists():
        previous = read_jsonl(args.output_jsonl)
        foreign = {r.get("identity") for r in previous} - {identity["sha256"]}
        if foreign:
            raise ValueError(
                f"{args.output_jsonl} holds records of another model/index/protocol; use a new file"
            )
        # Failed samples are retried; scored ones are not recomputed.
        done = {r["position"] for r in previous if r.get("status") == "ok"}
        entries = [e for e in entries if e["position"] not in done]
    if not entries:
        return 0
    tasks = {e["task"] for e in entries}
    dataset = ManifestVisionDataset(
        args.manifest,
        args.resolution,
        "val",
        tasks,
        aspect_preserving_crop=True,
        data_root=args.manifest.parent,
    )
    lookup = {
        (s["task"], str(s["image_path"]), str(s["target_path"]), s["prompt"]): i
        for i, s in enumerate(dataset.samples)
    }
    wanted = {e["line"] for e in entries}
    with args.manifest.open(encoding="utf-8") as stream:
        rows = {n: json.loads(line) for n, line in enumerate(stream, 1) if n in wanted}

    os.environ["OPEN_VISION_BANANA_MODEL_PATH"] = weights["model_path"]
    model = VisionBanana.from_pretrained(
        model_name=args.model_name,
        lora_path=weights["lora_path"],
        device=args.device,
        dtype=torch.bfloat16,
    )
    ctx = SimpleNamespace(
        mapper=DepthRGBMapper(colormap="cube_edge"),
        seg=SegmentationDecoder(),
        resolution=args.resolution,
        aspect_preserving=dataset.aspect_preserving_crop,
        open_rgb=dataset._open_pil_rgb,
        clip=None,
    )
    if "generation" in tasks:
        from transformers import CLIPModel, CLIPProcessor

        ctx.clip = (
            CLIPModel.from_pretrained(args.clip_model).to(args.device).eval(),
            CLIPProcessor.from_pretrained(args.clip_model),
        )
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("a", encoding="utf-8") as out:
        for entry in entries:
            row = rows[entry["line"]]
            task = entry["task"]
            sample = SimpleNamespace(
                row=row,
                image_path=resolve_manifest_path(row["input"], args.manifest),
                target_path=resolve_manifest_path(row["target"], args.manifest),
            )
            key = (task, str(sample.image_path), str(sample.target_path), dataset_prompt(row, task))
            record = {
                "position": entry["position"],
                "task": task,
                "line": entry["line"],
                "source": entry["source"],
                "identity": identity["sha256"],
            }
            started = time.perf_counter()
            try:
                if key not in lookup:
                    raise KeyError("index entry not found in the val dataset")
                sample.spec = dataset.samples[lookup[key]]
                sample.item = dataset[lookup[key]]
                # [0, 1] on purpose: generator._tensor_to_pil reads a tensor without negative pixels
                # as [0, 1], so a bright image in the documented [-1, 1] range would be misread.
                image = ((sample.item["image"] + 1.0) / 2.0).unsqueeze(0).to(args.device)
                target = sample.item["target_rgb"].unsqueeze(0).to(args.device)
                with torch.no_grad():
                    pred = model.generator.generate(
                        prompt=sample.item["prompt"],
                        image=image,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        strength=args.strength,
                        height=args.resolution,
                        width=args.resolution,
                        seed=args.seed + entry["position"],
                    )
                pred = pred.float().to(args.device)
                if pred.shape[-2:] != target.shape[-2:]:
                    pred = torch.nn.functional.interpolate(
                        pred, size=target.shape[-2:], mode="bilinear", align_corners=False
                    )
                if not torch.isfinite(pred).all():
                    raise ValueError("non-finite generator output")
                pred = pred.clamp(0, 1)
                if args.save_dir is not None:
                    stem = f"{entry['position']:05d}_{task}"
                    to_pil(pred).save(args.save_dir / f"{stem}_pred.png")
                    to_pil(target).save(args.save_dir / f"{stem}_target.png")
                record.update(SCORERS[task](ctx, sample, pred, target))
                record["status"] = "ok"
            except Exception as exc:  # recorded, never silently dropped
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"[:500]
            record["seconds"] = round(time.perf_counter() - started, 2)
            if entry is entries[0]:
                record["identity_detail"] = identity
            out.write(json.dumps(record) + "\n")
            out.flush()
    return 0


def run_merge(args) -> int:
    index = json.loads(args.index.read_text(encoding="utf-8"))
    records = {}
    duplicates = 0
    for path in args.shards:
        for r in read_jsonl(path):
            if r["position"] in records and records[r["position"]].get("status") == "ok":
                duplicates += 1
                continue
            records[r["position"]] = r
    expected = {e["position"] for e in index["entries"]}
    foreign = sorted(set(records) - expected)
    for position in foreign:
        del records[position]
    missing = sorted(expected - set(records))
    identities = sorted({str(r.get("identity")) for r in records.values()})
    details = [r["identity_detail"] for r in records.values() if "identity_detail" in r]
    result = {
        "index_manifest_sha256": index["manifest_sha256"],
        "index_salt": index["salt"],
        "coverage": {
            "expected": len(expected),
            "scored_or_failed": len(records),
            "missing": len(missing),
            "duplicates_ignored": duplicates,
            "outside_index": len(foreign),
            "identities": len(identities),
            "failed": Counter(r["task"] for r in records.values() if r.get("status") != "ok"),
        },
        "identity": details[0] if len(identities) == 1 and details else None,
        "metrics": aggregate(list(records.values())),
        "complete": not missing
        and not foreign
        and len(identities) == 1
        and all(r.get("status") == "ok" for r in records.values()),
    }
    if args.meta:
        result["meta"] = json.loads(args.meta.read_text(encoding="utf-8"))
    write_results(args.output_json, result)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}))
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
    e.add_argument(
        "--model-path",
        type=Path,
        default=os.environ.get("OPEN_VISION_BANANA_MODEL_PATH"),
        help="base pipeline snapshot the run trained from (E: merged, Q: official)",
    )
    e.add_argument(
        "--checkpoint", type=Path, help="LoRA (lora_adapter/) or full (transformer/) checkpoint"
    )
    e.add_argument("--resolution", type=int, default=512)
    e.add_argument("--num-inference-steps", type=int, default=50)
    e.add_argument("--guidance-scale", type=float, default=4.0)
    e.add_argument("--strength", type=float, default=0.85)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--device", default="cuda")
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--shard-index", type=int, default=0)
    e.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    e.add_argument("--output-jsonl", type=Path, required=True)
    e.add_argument(
        "--save-dir", type=Path, help="also write each prediction and its canvas target as PNG"
    )
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
        if a.model_path is None:
            p.error("eval needs --model-path (or OPEN_VISION_BANANA_MODEL_PATH)")
        a.manifest, a.model_path = a.manifest.resolve(), Path(a.model_path).resolve()
        a.checkpoint = a.checkpoint.resolve() if a.checkpoint else None
        return run_eval(a)
    return run_merge(a)


if __name__ == "__main__":
    raise SystemExit(main())
