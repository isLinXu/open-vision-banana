#!/usr/bin/env python3
"""Turn downloaded evaluation archives into the layouts the eval datasets read.

    nyu-mat      nyu_depth_v2_labeled.mat (+ splits.mat) -> <out>/{train,test}/{images,depths}/
                 images as JPEG, depths as uint16 millimetre PNG (NYUDepthDataset layout).
                 The 654 Eigen test frames come from ``splits.mat`` (testNdxs, 1-based);
                 without it every frame goes to ``all/`` and the paper protocol is not met.
    genai-bench  GenAI-Bench parquet shards -> <out>/prompts.jsonl ({"id", "prompt", ...})
                 for GenAIBenchDataset; image columns are dropped (they are model outputs).

    python scripts/convert_eval_sets.py nyu-mat --mat data/raw/nyu_depth_v2/nyu_depth_v2_labeled.mat \\
        --splits data/raw/nyu_depth_v2/splits.mat --out data/eval/nyu_depth_v2
    python scripts/convert_eval_sets.py genai-bench --parquet data/raw/genai_bench/data --out data/eval/genai_bench
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _load_split_indices(splits_path: Path | None) -> dict[str, np.ndarray] | None:
    if splits_path is None:
        return None
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("reading splits.mat needs scipy: pip install scipy") from exc
    mat = loadmat(splits_path)
    return {
        "train": np.asarray(mat["trainNdxs"]).ravel().astype(int) - 1,  # MATLAB 1-based
        "test": np.asarray(mat["testNdxs"]).ravel().astype(int) - 1,
    }


def convert_nyu_mat(mat_path: Path, out: Path, *, splits_path: Path | None = None, jpeg_quality: int = 95) -> dict:
    """Write NYU labelled frames (metres -> mm PNG) into split/images + split/depths."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("reading the NYU .mat (HDF5 v7.3) needs h5py: pip install h5py") from exc
    splits = _load_split_indices(splits_path)
    counts: dict[str, int] = {}
    with h5py.File(mat_path, "r") as handle:
        images = handle["images"]  # [N, 3, W, H] uint8 in MATLAB column order
        depths = handle["depths"]  # [N, W, H] float32 metres (in-painted)
        n = images.shape[0]
        membership: dict[int, str] = {}
        if splits is None:
            membership = dict.fromkeys(range(n), "all")
        else:
            for split, indices in splits.items():
                for index in indices:
                    membership[int(index)] = split
        for index in range(n):
            split = membership.get(index)
            if split is None:
                continue
            img = np.asarray(images[index]).transpose(2, 1, 0)  # -> [H, W, 3]
            depth = np.asarray(depths[index]).T  # -> [H, W]
            img_dir = out / split / "images"
            depth_dir = out / split / "depths"
            img_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{index:04d}"
            Image.fromarray(img, "RGB").save(img_dir / f"{stem}.jpg", quality=jpeg_quality)
            depth_mm = np.clip(np.nan_to_num(depth, nan=0.0) * 1000.0, 0, 65535).round().astype(np.uint16)
            Image.fromarray(depth_mm, "I;16").save(depth_dir / f"{stem}.png")
            counts[split] = counts.get(split, 0) + 1
    summary = {
        "mat": str(mat_path),
        "splits": str(splits_path) if splits_path else None,
        "frames_written": counts,
        "depth_unit": "millimetres (uint16 PNG), from the in-painted `depths` field",
        "eigen_test_split": bool(splits_path),
    }
    (out / "convert_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def convert_genai_bench(parquet: Path, out: Path, *, prompt_columns: tuple[str, ...] = ("prompt", "source_prompt", "text", "caption")) -> dict:
    """Collect unique prompts from GenAI-Bench parquet shards into prompts.jsonl."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("reading parquet needs pyarrow: pip install pyarrow") from exc
    shards = sorted(parquet.rglob("*.parquet")) if parquet.is_dir() else [parquet]
    if not shards:
        raise FileNotFoundError(f"no parquet under {parquet}")
    seen: dict[str, dict] = {}
    for shard in shards:
        # GenAI-Bench-1600 capitalises its columns ("Prompt", "Tags"), the pairwise sets do not
        names = {name.lower(): name for name in pq.ParquetFile(shard).schema_arrow.names}
        column = next((names[c.lower()] for c in prompt_columns if c.lower() in names), None)
        if column is None:
            continue
        extra = [names[c] for c in ("basic_skills", "advanced_skills", "task", "instruct_prompt", "vote_type") if c in names]
        table = pq.read_table(shard, columns=[column, *extra]).to_pylist()
        for row in table:
            prompt = str(row.get(column) or "").strip()
            if prompt and prompt not in seen:
                seen[prompt] = {"id": f"genai_{len(seen):05d}", "prompt": prompt, "source_shard": shard.name,
                                **{k: row[k] for k in extra if row.get(k) is not None}}
    if not seen:
        raise ValueError(f"no prompts under {parquet}: no non-empty value in a column named one of {prompt_columns}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "prompts.jsonl").open("w", encoding="utf-8") as handle:
        for row in seen.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"shards": len(shards), "prompts": len(seen), "output": str(out / "prompts.jsonl")}
    (out / "convert_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    nyu = sub.add_parser("nyu-mat")
    nyu.add_argument("--mat", type=Path, required=True)
    nyu.add_argument("--splits", type=Path, default=None, help="splits.mat with trainNdxs / testNdxs")
    nyu.add_argument("--out", type=Path, required=True)
    gb = sub.add_parser("genai-bench")
    gb.add_argument("--parquet", type=Path, required=True, help="parquet file or directory of shards")
    gb.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "nyu-mat":
        summary = convert_nyu_mat(args.mat, args.out, splits_path=args.splits)
    else:
        summary = convert_genai_bench(args.parquet, args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
