#!/usr/bin/env python3
"""Write post_sft_gate.json from an evaluate_paper_benchmarks NYU JSON.

    python scripts/write_sft_gate.py output/eval/paper_metrics.json \
        --output output/sft_gate/post_sft_gate.json

colormap_ok requires decoded_metrics.oracle_colormap_hist (re-run eval with
--depth-oracle-colormaps). GRPO still refuses unless raw δ1.25 is below aligned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.training.post_training import (  # noqa: E402
    grpo_sft_gate_status,
    sft_gate_from_eval,
    write_sft_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_json", type=Path, help="evaluate_paper_benchmarks.py --output-json")
    parser.add_argument("--output", type=Path, required=True, help="path for post_sft_gate.json")
    parser.add_argument(
        "--trained-colormap",
        default=None,
        help="defaults to metadata.depth_colormap, then cube_edge",
    )
    args = parser.parse_args()
    results = json.loads(args.eval_json.read_text(encoding="utf-8"))
    metadata = results.get("metadata") if isinstance(results, dict) else {}
    trained = args.trained_colormap or (metadata or {}).get("depth_colormap") or "cube_edge"
    try:
        gate = sft_gate_from_eval(results, trained_colormap=trained, source=str(args.eval_json))
    except ValueError as exc:
        print(f"SFT gate not written: {exc}", file=sys.stderr)
        return 3
    write_sft_gate(args.output, gate)
    ok, detail = grpo_sft_gate_status(gate)
    print(json.dumps(gate, indent=2))
    print(f"\nwritten: {args.output}")
    print(f"grpo_sft_gate: {'PASS' if ok else 'FAIL'} ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
