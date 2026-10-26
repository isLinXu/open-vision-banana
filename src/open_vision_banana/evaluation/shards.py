"""Exact merging of benchmark shards.

``scripts/evaluate_paper_benchmarks.py --num-shards N --shard-index K`` scores one contiguous,
batch-aligned block of each benchmark (seeds stay those of the unsharded run) and stores the
accumulator's raw sums next to its metrics. ``scripts/merge_eval_shards.py`` adds the sums of
all N blocks and recomputes the metrics: those of a single run over the set, up to the order in
which the float sums are added.

Depth, semantic and referring accumulators are sums. Normal (per-pixel angle lists for the
median), instance and generation jobs are not shardable. ``evaluate_generation_preservation.py``
shards through per-image states instead (``generation_preservation.generation_shard_state``).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch

from .benchmark import DepthAccumulator, SemanticAccumulator, depth_report
from .referring import ReferringAccumulator

SHARDABLE_BENCHMARKS = frozenset({"depth", "semantic", "referring"})


def shard_range(total: int, num_shards: int, index: int, batch_size: int = 1) -> range:
    """Sample indices of block ``index`` of ``num_shards`` over ``total`` samples, batch-aligned."""
    if num_shards < 1 or not 0 <= index < num_shards:
        raise ValueError(f"shard index must be in [0, {num_shards}), got {index}")
    per_shard = math.ceil(math.ceil(total / batch_size) / num_shards) * batch_size
    return range(min(total, index * per_shard), min(total, (index + 1) * per_shard))


def _encode(value: Any) -> Any:
    """Floats that JSON cannot hold (an inf sum) travel as strings, so the merge keeps them."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return [_encode(item) for item in value] if isinstance(value, list) else value


def _decode(value: Any) -> Any:
    if isinstance(value, str) and value in {"inf", "-inf", "nan"}:
        return float(value)
    return [_decode(item) for item in value] if isinstance(value, list) else value


def accumulator_state(accumulator: Any) -> dict[str, Any]:
    """JSON-serialisable raw sums of a depth, semantic or referring accumulator."""
    if isinstance(accumulator, DepthAccumulator):
        fields = dataclasses.asdict(accumulator)
        return {"kind": "depth", **{key: _encode(value) for key, value in fields.items()}}
    if isinstance(accumulator, SemanticAccumulator):
        return {
            "kind": "semantic",
            "num_classes": accumulator.num_classes,
            "ignore_index": accumulator.ignore_index,
            "confusion": accumulator.confusion.tolist(),
        }
    if isinstance(accumulator, ReferringAccumulator):
        return {"kind": "referring", **{k: _encode(v) for k, v in vars(accumulator).items()}}
    raise TypeError(f"not a shardable accumulator: {type(accumulator).__name__}")


def _same(states: list[dict[str, Any]], key: str) -> Any:
    values = {repr(state[key]) for state in states}
    if len(values) != 1:
        raise ValueError(f"shards disagree on {key}: {sorted(values)}")
    return states[0][key]


def merge_accumulator_states(states: list[dict[str, Any]]) -> Any:
    """One accumulator holding the sums of all ``states`` (all of one kind)."""
    if not states:
        raise ValueError("no accumulator states to merge")
    kind = _same(states, "kind")
    if kind == "depth":
        merged = DepthAccumulator()
        for state in states:
            for item in dataclasses.fields(DepthAccumulator):
                current, value = getattr(merged, item.name), _decode(state[item.name])
                if isinstance(current, list):
                    setattr(merged, item.name, [a + b for a, b in zip(current, value, strict=True)])
                else:
                    setattr(merged, item.name, current + value)
        return merged
    if kind == "semantic":
        merged = SemanticAccumulator(_same(states, "num_classes"), _same(states, "ignore_index"))
        for state in states:
            merged.confusion += torch.tensor(state["confusion"], dtype=torch.int64)
        return merged
    if kind == "referring":
        merged = ReferringAccumulator(iou_threshold=_same(states, "iou_threshold"))
        for name in vars(merged).keys() - {"iou_threshold"}:
            setattr(merged, name, sum(_decode(state[name]) for state in states))
        return merged
    raise ValueError(f"unknown accumulator kind {kind!r}")


def merged_metrics(shard_states: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics of one benchmark from the ``last_accumulator_states`` of all its shards."""
    main = merge_accumulator_states([states["main"] for states in shard_states])
    if not isinstance(main, DepthAccumulator):
        return main.compute()
    relative_states = [states["relative"] for states in shard_states]
    if len({state is None for state in relative_states}) > 1:
        raise ValueError("shards disagree on the relative-luminance reading")
    relative = None if relative_states[0] is None else merge_accumulator_states(relative_states)
    return depth_report(
        main,
        relative,
        skipped_samples=sum(states["skipped_samples"] for states in shard_states),
        depth_curve=_same(shard_states, "depth_curve"),
        depth_units=_same(shard_states, "depth_units"),
    )


__all__ = [
    "SHARDABLE_BENCHMARKS",
    "accumulator_state",
    "merge_accumulator_states",
    "merged_metrics",
    "shard_range",
]
