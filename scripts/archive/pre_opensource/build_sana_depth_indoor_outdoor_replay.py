#!/usr/bin/env python3
"""Build Hypersim-heavy + light KITTI replay manifest (no pixel copy).

Writes ``data/sana_depth_indoor_outdoor_replay_512/{pairs.jsonl,manifest.json}``
with portable paths relative to the new manifest dir:

- all 1000 Hypersim metric pairs
- 200 KITTI train pairs (seeded subsample)

    PYTHONPATH=src python scripts/build_sana_depth_indoor_outdoor_replay.py
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from open_vision_banana.data.manifest import fingerprint_records, read_jsonl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = PROJECT_ROOT / "data" / "sana_depth_indoor_outdoor_replay_512"
DEFAULT_HYPERSIM = PROJECT_ROOT / "data" / "hypersim_depth_512" / "pairs.jsonl"
DEFAULT_KITTI = PROJECT_ROOT / "data" / "kitti_metric_512" / "pairs.jsonl"


def _rewrite_relative(record: dict, src_manifest: Path, out_dir: Path) -> dict:
    """Point input/target at paths relative to *out_dir* (may use ``../``)."""
    row = dict(record)
    out_resolved = out_dir.resolve()
    for key in ("input", "target"):
        raw = Path(row[key])
        abs_path = raw if raw.is_absolute() else (src_manifest.parent / raw).resolve()
        row[key] = abs_path.resolve().relative_to(out_resolved, walk_up=True).as_posix()
    return row


def build(
    *,
    hypersim_pairs: Path,
    kitti_pairs: Path,
    output: Path,
    kitti_count: int,
    seed: int,
) -> dict:
    hypersim = read_jsonl(hypersim_pairs)
    kitti = read_jsonl(kitti_pairs)
    if len(hypersim) < 1:
        raise SystemExit(f"no hypersim rows in {hypersim_pairs}")
    if len(kitti) < kitti_count:
        raise SystemExit(f"need {kitti_count} KITTI rows, found {len(kitti)}")

    rng = random.Random(seed)
    kitti_indices = sorted(rng.sample(range(len(kitti)), kitti_count))
    selected_kitti = [kitti[i] for i in kitti_indices]

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for rec in hypersim:
        rows.append(_rewrite_relative(rec, hypersim_pairs, output))
    for rec in selected_kitti:
        rows.append(_rewrite_relative(rec, kitti_pairs, output))

    pairs_path = output / "pairs.jsonl"
    pairs_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    # Verify resolve
    missing = []
    for r in rows[:3] + rows[-3:]:
        for key in ("input", "target"):
            p = (output / r[key]).resolve()
            if not p.is_file():
                missing.append(str(p))
    if missing:
        raise SystemExit(f"resolved paths missing: {missing[:6]}")

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pair_manifest": "pairs.jsonl",
        "path_policy": "manifest-relative",
        "hypersim_rows": len(hypersim),
        "kitti_rows": kitti_count,
        "kitti_source_rows": len(kitti),
        "kitti_sample_seed": seed,
        "kitti_sample_indices": kitti_indices,
        "pairs_written": len(rows),
        "hypersim_source": str(hypersim_pairs.resolve()),
        "kitti_source": str(kitti_pairs.resolve()),
        "note": (
            "Indoor-heavy metric replay for depth-only strengthen from Hypersim-500. "
            "All Hypersim + seeded KITTI subsample. Images stay in original dirs."
        ),
        "dataset_fingerprint": fingerprint_records(rows),
        "fingerprint_algorithm": "sha256:canonical-pairs-v1",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hypersim-pairs", type=Path, default=DEFAULT_HYPERSIM)
    p.add_argument("--kitti-pairs", type=Path, default=DEFAULT_KITTI)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--kitti-count", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    # Prefer pathlib walk_up relative_to (3.12+); fall back handled in rewrite
    try:
        Path("a").relative_to(Path("b"), walk_up=True)
    except TypeError as exc:
        raise SystemExit("Python 3.12+ required for relative_to(walk_up=True)") from exc

    manifest = build(
        hypersim_pairs=args.hypersim_pairs,
        kitti_pairs=args.kitti_pairs,
        output=args.output,
        kitti_count=args.kitti_count,
        seed=args.seed,
    )
    print(
        f"wrote {manifest['pairs_written']} pairs "
        f"(hypersim={manifest['hypersim_rows']}, kitti={manifest['kitti_rows']}) "
        f"→ {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
