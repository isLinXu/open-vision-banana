"""Writers for the paper-benchmark result JSON / CSV (eval script and shard merge share them)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Convert non-finite metric values to JSON null without hiding them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def sidecar(output_json: Path, suffix: str) -> Path:
    """A file next to ``output_json`` named after all of it (``gen.shard0.json`` -> ``gen.shard0<suffix>``)."""
    return output_json.with_name(output_json.name.removesuffix(".json") + suffix)


def write_results(path: Path, results: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(results), indent=2, allow_nan=False), encoding="utf-8")


def write_metrics_csv(path: Path, results: dict[str, Any]) -> None:
    """All numeric diagnostics, including nested protocol variants and per-class lists."""
    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from flatten(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from flatten(child, f"{prefix}.{index}")
        elif isinstance(value, (int, float)) or value is None:
            yield prefix, value
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "dataset", "metric", "value"])
        for key, record in results["benchmarks"].items():
            benchmark, dataset = key.split("/", 1)
            for metric, value in flatten(record.get("decoded_metrics") or {}):
                writer.writerow([benchmark, dataset, metric, json_safe(value)])


__all__ = ["json_safe", "sidecar", "write_metrics_csv", "write_results"]
