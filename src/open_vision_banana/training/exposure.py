"""Per-rank, per-invocation consumed-sample audits (including dropped updates)."""

from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from pathlib import Path


class ExposureAudit:
    def __init__(self, output: Path, rank: int, start_step: int):
        self.path = output / f"exposure-rank{rank}-{uuid.uuid4().hex}.jsonl"
        self.rank, self.start_step = rank, start_step
        self.counts = Counter()
        self.scenes = Counter()
        self.palettes = Counter()
        self.unique = defaultdict(set)
        self.dropped_updates = 0

    def observe(self, batch: dict) -> None:
        if not isinstance(batch, dict):
            return
        for record in batch.get("experiment_audit", []):
            if not record:
                continue
            source, task = record["source"], record["task"]
            key = f"{source}/{task}"
            self.counts[key] += 1
            self.unique[key].add(record["id"])
            if record.get("scene"):
                self.scenes[f"{source}/{record['scene']}"] += 1
            if record.get("palette"):
                self.palettes[record["palette"]] += 1

    def flush(self, step: int, *, complete: bool = False) -> None:
        if not self.counts:
            return  # Unmodified datasets retain their existing logging behavior.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(
            rank=self.rank,
            invocation_start_step=self.start_step,
            optimizer_step=step,
            consumed_counts=dict(self.counts),
            unique_counts={k: len(v) for k, v in self.unique.items()},
            scene_counts=dict(self.scenes),
            palette_counts=dict(self.palettes),
            dropped_updates=self.dropped_updates,
            complete=complete,
            scope="cumulative within this invocation/rank; includes dropped-gradient batches; not a checkpoint state",
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
