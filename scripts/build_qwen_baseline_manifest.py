#!/usr/bin/env python3
"""Build a paper-aligned, de-biased multi-task manifest for the Qwen-Image baseline.

Solves the two data bottlenecks identified in the SNVC shared-baseline ablation:
1. Depth supervision quality: replaces or filters out SNVC relative/grayscale
   pseudo-depth targets, keeping only valid ``metric_power`` + ``cube_edge``
   synthetic metric depth targets (e.g. Hypersim / Virtual KITTI 2) and unifying
   their prompt to ``PromptBuilder.DEPTH_PROMPT``.
2. Semantic class/domain balance: Cityscapes was only 311 / 4460 (6.97%) of
   semantic train rows in the raw SNVC pack. This builder either merges an
   official Cityscapes-19 manifest or deterministically upsamples Cityscapes
   train rows to reach ``--min-cityscapes-share`` (default 0.50), while keeping
   validation splits strictly un-duplicated.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.data.manifest import resolve_manifest_path  # noqa: E402
from open_vision_banana.inference.prompt import PromptBuilder  # noqa: E402
from open_vision_banana.task_prompts import SEMANTIC_SEGMENTATION_PROMPT  # noqa: E402

TASK_ALIASES = {
    "segmentation": "semantic",
    "seg": "semantic",
    "metric_depth": "depth",
}

# Paper zero-shot transfer protocol: no training data from any evaluated benchmark.
# Tables 2/4/5/6/7 evaluate Cityscapes val, RefCOCOg, ReasonSeg, NYU, iBims-1, ETH3D,
# DIODE, KITTI, nuScenes, ScanNet, and Virtual KITTI 2. Rows whose input/target
# paths come from these corpora are rejected unless the caller explicitly allows
# a benchmark (e.g. Cityscapes *train* when the run is not claimed as zero-shot).
ZERO_SHOT_BENCHMARK_PATTERNS: dict[str, tuple[str, ...]] = {
    "vkitti": ("vkitti", "virtual_kitti", "virtualkitti"),
    "nyu": ("nyu_depth", "nyuv2", "nyu-depth", "nyu/"),
    "ibims": ("ibims",),
    "eth3d": ("eth3d",),
    "diode": ("diode",),
    "kitti": ("kitti_raw", "kitti/", "kitti_depth", "data_depth_annotated"),
    "nuscenes": ("nuscenes",),
    "scannet": ("scannet/", "scannetv2", "scannet_v2"),
    "refcocog": ("refcocog", "refcoco"),
    "reasonseg": ("reasonseg",),
    "sa_co": ("saco", "sa-co", "sa_co"),
}


def benchmark_leak(row: dict, allowed: set[str]) -> str | None:
    """Return the benchmark name a manifest row leaks from, or None."""
    haystack = " ".join(
        str(row.get(k, "")) for k in ("input", "target", "image", "source", "dataset")
    ).lower()
    # An explicitly allowed corpus wins even if its path also contains a
    # shorter pattern (e.g. "virtual_kitti/" also contains "kitti/").
    for name in allowed:
        if any(p in haystack for p in ZERO_SHOT_BENCHMARK_PATTERNS[name]):
            return None
    for name, patterns in ZERO_SHOT_BENCHMARK_PATTERNS.items():
        if any(p in haystack for p in patterns):
            return name
    return None


# COCO file names only: COCO_train2014_000000581921.jpg or the 12-digit 000000581921.jpg.
# Short numeric names (Hypersim frame.0001.png, Cityscapes *_000019_leftImg8bit.png)
# deliberately do not parse, so an exclusion list cannot hit non-COCO sources.
_COCO_IMAGE_ID_RE = re.compile(
    r"^(?:COCO_(?:train|val|test)20\d{2}_)?(\d{12})\.(?:jpg|jpeg|png)$", re.IGNORECASE
)


def coco_image_id_from_path(path: str) -> int | None:
    """Return the COCO image id encoded in a COCO-style file name, or None."""
    match = _COCO_IMAGE_ID_RE.match(Path(str(path)).name)
    return int(match.group(1)) if match else None


def load_excluded_image_ids(sources: tuple[Path, ...] | list[Path]) -> set[int]:
    """Union of image ids from RefCOCO ``refs(*).p`` pickles, JSON lists, or text files.

    Pickled refs are filtered to val/test splits (the training split of a benchmark
    is what the zero-shot protocol forbids anyway via path patterns; its evaluation
    images are the ones that can hide inside COCO train2014). JSON files may be a
    list of ids or a ``{"images": [{"id": ...}]}`` COCO-style dict. Text files hold
    one id or file name per line.
    """
    ids: set[int] = set()
    for source in sources:
        source = Path(source)
        if source.suffix == ".p":
            import pickle

            with source.open("rb") as handle:
                refs = pickle.load(handle)
            for ref in refs:
                if str(ref.get("split", "val")) in {"val", "test", "testA", "testB"}:
                    ids.add(int(ref["image_id"]))
        elif source.suffix == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            entries = payload.get("images", []) if isinstance(payload, dict) else payload
            for entry in entries:
                if isinstance(entry, dict):
                    ids.add(int(entry["id"]))
                else:
                    parsed = coco_image_id_from_path(str(entry))
                    ids.add(int(entry) if parsed is None else parsed)
        else:
            for line in source.read_text(encoding="utf-8").splitlines():
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                parsed = coco_image_id_from_path(token)
                ids.add(int(token) if parsed is None else parsed)
    return ids


def excluded_image_leak(row: dict, excluded_ids: set[int]) -> bool:
    """True if the row's input image carries a COCO id from an evaluation split."""
    if not excluded_ids:
        return False
    for key in ("input", "image"):
        image_id = coco_image_id_from_path(str(row.get(key, "")))
        if image_id is not None and image_id in excluded_ids:
            return True
    return False


def is_valid_metric_cube_edge_target(
    target_path: Path,
    mapper: DepthRGBMapper,
    *,
    max_mae: float = 2.0 / 255.0,
    max_grayscale_fraction: float = 0.90,
) -> bool:
    """Return True if ``target_path`` is a valid non-grayscale cube-edge depth target."""
    if not target_path.is_file():
        return False
    try:
        with Image.open(target_path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except OSError:
        return False
    rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    channel_span = (rgb[:, 0] - rgb[:, 1]).abs() + (rgb[:, 1] - rgb[:, 2]).abs()
    gray_frac = float((channel_span < 1.0 / 255.0).float().mean().item())
    if gray_frac > max_grayscale_fraction:
        return False
    decoded = mapper.rgb_to_depth(rgb)
    if float(decoded.max().item()) <= 0.0:
        return False
    recon = mapper.depth_to_rgb(decoded)
    mae = float((recon - rgb).abs().mean().item())
    return mae <= max_mae


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_qwen_baseline_manifest(
    base_manifest: Path,
    output_manifest: Path,
    *,
    data_root: Path | None = None,
    metric_depth_manifest: Path | None = None,
    cityscapes_manifest: Path | None = None,
    min_cityscapes_share: float = 0.50,
    filter_non_metric_depth: bool = True,
    seed: int = 42,
    allow_benchmarks: tuple[str, ...] | list[str] = (),
    exclude_image_ids: tuple[Path, ...] | list[Path] = (),
) -> dict:
    """Build a cleaned, rebalanced manifest and return an audit summary.

    ``exclude_image_ids`` lists files of COCO image ids (see
    ``load_excluded_image_ids``) whose rows are dropped regardless of path. RefCOCOg
    UMD val images live inside COCO train2014, so a COCO-derived semantic/instance
    source passes the path audit yet still leaks evaluation images without this.
    """
    if not 0.0 <= min_cityscapes_share < 1.0:
        raise ValueError("min_cityscapes_share must be in [0, 1)")
    unknown = sorted(set(allow_benchmarks) - set(ZERO_SHOT_BENCHMARK_PATTERNS))
    if unknown:
        raise ValueError(
            f"allow_benchmarks contains unknown names {unknown}; "
            f"choose from {sorted(ZERO_SHOT_BENCHMARK_PATTERNS)}"
        )
    allowed = set(allow_benchmarks)

    rng = random.Random(seed)
    anchor = data_root.resolve() if data_root else base_manifest.resolve()
    mapper = DepthRGBMapper(colormap="cube_edge", curve="metric_power")

    base_rows = load_jsonl(base_manifest)
    kept_rows: list[dict] = []
    rejected_depth = 0
    rejected_benchmark: dict[str, int] = {}
    excluded_ids = load_excluded_image_ids(exclude_image_ids)
    rejected_image_ids = 0

    def leaks(row: dict) -> bool:
        nonlocal rejected_image_ids
        name = benchmark_leak(row, allowed)
        if name is not None:
            rejected_benchmark[name] = rejected_benchmark.get(name, 0) + 1
            return True
        if excluded_image_leak(row, excluded_ids):
            rejected_image_ids += 1
            return True
        return False

    for row in base_rows:
        raw_task = row.get("task", "")
        task = TASK_ALIASES.get(raw_task, raw_task)
        row = dict(row, task=task)
        if leaks(row):
            continue

        if task == "depth":
            if metric_depth_manifest is not None:
                continue
            if filter_non_metric_depth:
                tgt_path = resolve_manifest_path(row.get("target", ""), anchor)
                if not is_valid_metric_cube_edge_target(tgt_path, mapper):
                    rejected_depth += 1
                    continue
            row["prompt"] = PromptBuilder.DEPTH_PROMPT

        kept_rows.append(row)

    if metric_depth_manifest is not None:
        for row in load_jsonl(metric_depth_manifest):
            row = dict(row, task="depth", prompt=PromptBuilder.DEPTH_PROMPT)
            if leaks(row):
                continue
            kept_rows.append(row)

    if cityscapes_manifest is not None:
        for row in load_jsonl(cityscapes_manifest):
            row = dict(row, task="semantic", prompt=SEMANTIC_SEGMENTATION_PROMPT)
            if row.get("split") == "val" or leaks(row):
                continue
            kept_rows.append(row)

    # Rebalance Cityscapes train share within semantic
    cs_train = [
        r for r in kept_rows
        if r.get("task") == "semantic"
        and r.get("split") == "train"
        and r.get("prompt") == SEMANTIC_SEGMENTATION_PROMPT
    ]
    other_sem_train = [
        r for r in kept_rows
        if r.get("task") == "semantic"
        and r.get("split") == "train"
        and r.get("prompt") != SEMANTIC_SEGMENTATION_PROMPT
    ]
    upsampled_cs_added = 0
    if cs_train and other_sem_train and min_cityscapes_share > 0.0:
        target_cs_count = int(
            round(min_cityscapes_share * len(other_sem_train) / (1.0 - min_cityscapes_share))
        )
        if len(cs_train) < target_cs_count:
            needed = target_cs_count - len(cs_train)
            full_copies, remainder = divmod(needed, len(cs_train))
            extra = cs_train * full_copies + rng.sample(cs_train, remainder)
            kept_rows.extend(dict(r) for r in extra)
            upsampled_cs_added = len(extra)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in kept_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sem_train_total = len(cs_train) + upsampled_cs_added + len(other_sem_train)
    final_cs_train = len(cs_train) + upsampled_cs_added
    task_split_counts: dict[str, int] = {}
    for row in kept_rows:
        key = f"{row.get('task')}/{row.get('split')}"
        task_split_counts[key] = task_split_counts.get(key, 0) + 1

    audit = {
        "output_manifest": str(output_manifest.resolve()),
        "total_rows": len(kept_rows),
        "task_split_counts": task_split_counts,
        "rejected_non_metric_depth_rows": rejected_depth,
        "rejected_benchmark_leak_rows": rejected_benchmark,
        "rejected_excluded_image_id_rows": rejected_image_ids,
        "excluded_image_id_count": len(excluded_ids),
        "allowed_benchmarks": sorted(allowed),
        "cityscapes_train_initial": len(cs_train),
        "cityscapes_train_upsampled_added": upsampled_cs_added,
        "cityscapes_train_final": final_cs_train,
        "cityscapes_train_share": (
            final_cs_train / sem_train_total if sem_train_total > 0 else 0.0
        ),
    }
    output_manifest.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--metric-depth-manifest", type=Path, default=None)
    parser.add_argument("--cityscapes-manifest", type=Path, default=None)
    parser.add_argument("--min-cityscapes-share", type=float, default=0.50)
    parser.add_argument("--no-filter-non-metric-depth", action="store_true")
    parser.add_argument(
        "--allow-benchmark",
        action="append",
        default=[],
        choices=sorted(ZERO_SHOT_BENCHMARK_PATTERNS),
        help="Permit rows from this evaluated benchmark (breaks the paper zero-shot protocol).",
    )
    parser.add_argument(
        "--exclude-image-ids",
        action="append",
        type=Path,
        default=[],
        help=(
            "File of COCO image ids to drop regardless of path: a RefCOCO refs(umd).p "
            "pickle (val/test refs), a JSON list / COCO images dict, or one id per line. "
            "Use it when a COCO-derived source may contain RefCOCOg evaluation images."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_qwen_baseline_manifest(
        args.base_manifest,
        args.output,
        data_root=args.data_root,
        metric_depth_manifest=args.metric_depth_manifest,
        cityscapes_manifest=args.cityscapes_manifest,
        min_cityscapes_share=args.min_cityscapes_share,
        filter_non_metric_depth=not args.no_filter_non_metric_depth,
        seed=args.seed,
        allow_benchmarks=args.allow_benchmark,
        exclude_image_ids=args.exclude_image_ids,
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
