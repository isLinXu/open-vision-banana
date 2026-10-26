#!/usr/bin/env python3
"""Merge ``evaluate_paper_benchmarks.py --num-shards N`` outputs into one evaluation JSON.

Every shard of a benchmark stores its accumulator's raw sums; the merge adds them and recomputes
the metrics, so the result is that of one unsharded run (same samples, same per-sample seeds), up
to the order in which the float sums are added.
The shards must be the complete set 0..N-1 of one command (same metadata, same benchmarks).

    python scripts/merge_eval_shards.py output/run/eval_refcocog.shard*.json \\
        --output-json output/run/eval_refcocog.json --output-csv output/run/eval_refcocog.csv

Shards of ``evaluate_generation_preservation.py --num-shards N`` merge the same way: each stores
the per-image features and distances of its block (``<shard>.state.pt`` next to its JSON), and
the merge recomputes the report of one unsharded run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.evaluation.generation_preservation import generation_metrics  # noqa: E402
from open_vision_banana.evaluation.results_io import write_metrics_csv, write_results  # noqa: E402
from open_vision_banana.evaluation.shards import merged_metrics  # noqa: E402

# metadata a generation shard reports about its own block, not about the run
GENERATION_PER_SHARD = frozenset(
    {
        "device",
        "requested_samples",
        "successful_samples",
        "failed_samples",
        "failure_jsonl",
        "elapsed_seconds",
        "shard",
    }
)


def common_metadata(metadatas: list[dict[str, Any]], per_shard: frozenset[str]) -> dict[str, Any]:
    """The metadata of the run: every shard's without its ``per_shard`` keys, which must agree."""
    common = [{k: v for k, v in m.items() if k not in per_shard} for m in metadatas]
    if len({json.dumps(c, sort_keys=True) for c in common}) != 1:
        raise ValueError("shards were produced with different metadata (model, LoRA or protocol)")
    return common[0]


def check_blocks(key: str, infos: list[dict[str, Any]]) -> dict[str, Any]:
    """``merged_shards`` record of shard ``infos`` (sorted by index): shards 0..N-1 tiling one set."""
    for field in ("count", "of_samples"):
        if len({info[field] for info in infos}) != 1:
            raise ValueError(f"{key}: shards disagree on {field}")
    indices = [info["index"] for info in infos]
    count = infos[0]["count"]
    if indices != list(range(count)):
        raise ValueError(f"{key}: expected shards 0..{count - 1}, got {indices}")
    blocks = [(info["start"], info["stop"]) for info in infos]
    total = infos[0]["of_samples"]
    starts, stops = [start for start, _ in blocks], [stop for _, stop in blocks]
    if [0, *stops] != [*starts, total]:
        raise ValueError(f"{key}: shard blocks {blocks} do not tile [0, {total})")
    return {"count": count, "blocks": blocks}


def merge_generation_shards(shards: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """One generation-preservation report from the (JSON, saved state) pair of every shard."""
    if not shards:
        raise ValueError("no generation shards given")
    shards = sorted(shards, key=lambda pair: pair[0]["metadata"]["shard"]["index"])
    metadata = [shard["metadata"] for shard, _ in shards]
    merged = common_metadata(metadata, GENERATION_PER_SHARD)
    for field in ("requested_samples", "successful_samples", "failed_samples", "elapsed_seconds"):
        merged[field] = sum(m[field] for m in metadata)
    merged["failure_jsonl"] = [m["failure_jsonl"] for m in metadata]
    merged["merged_shards"] = check_blocks("generation", [m["shard"] for m in metadata])
    return {"metadata": merged, "metrics": generation_metrics([state for _, state in shards])}


def merge_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """One evaluation result from the JSON of every shard of a sharded run."""
    if not shards:
        raise ValueError("no shard results given")
    keys = {tuple(sorted(shard["benchmarks"])) for shard in shards}
    if len(keys) != 1:
        raise ValueError(f"shards cover different benchmarks: {sorted(keys)}")
    # the device may differ per shard (one GPU each); everything else names the same run
    common_metadata([shard["metadata"] for shard in shards], frozenset({"device"}))
    merged = {key: value for key, value in shards[0].items() if key != "benchmarks"}
    merged["benchmarks"] = {}
    for key in shards[0]["benchmarks"]:
        records = sorted(
            (shard["benchmarks"][key] for shard in shards),
            key=lambda record: record["shard"]["index"],
        )
        merged_shards = check_blocks(key, [record["shard"] for record in records])
        if len({record.get("dataset_fingerprint") for record in records}) != 1:
            raise ValueError(f"{key}: shards scored different copies of the dataset")
        if any(record.get("accumulator_states") is None for record in records):
            raise ValueError(f"{key}: a shard has no accumulator states (evaluator too old)")
        states = [record["accumulator_states"] for record in records]
        merged_record = {
            k: v for k, v in records[0].items() if k not in {"shard", "accumulator_states"}
        }
        merged_record["samples"] = sum(record["samples"] for record in records)
        merged_record["decoded_metrics"] = merged_metrics(states) if states else {}
        merged_record["merged_shards"] = merged_shards
        merged["benchmarks"][key] = merged_record
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("shards", nargs="+", type=Path, help="evaluation JSON of every shard")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.shards]
    kinds = {"shard_state" in shard for shard in shards}  # generation shards save a state file
    if kinds == {True}:
        if args.output_csv is not None:
            raise SystemExit("--output-csv applies to benchmark shards, not generation shards")
        merged = merge_generation_shards(
            [
                (shard, torch.load(path.with_name(shard["shard_state"]), map_location="cpu"))
                for path, shard in zip(args.shards, shards, strict=True)
            ]
        )
        write_results(args.output_json, merged)
        print(f"generation: {merged['metadata']['successful_samples']} samples from {len(shards)} shards")
        return 0
    if kinds != {False}:
        raise SystemExit("cannot merge generation shards with benchmark shards")
    merged = merge_shards(shards)
    write_results(args.output_json, merged)
    if args.output_csv is not None:
        write_metrics_csv(args.output_csv, merged)
    for key, record in merged["benchmarks"].items():
        shards = record["merged_shards"]["count"]
        print(f"{key}: {record['samples']} samples from {shards} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
