#!/usr/bin/env python3
"""Apply the pre-registered depth gates to a model evaluation.

    python scripts/check_depth_gate.py \
        --model-json output/eval/kontext_step500_nyu.json --benchmark depth/nyu \
        --baseline-json output/eval/nyu_trivial_baselines.json --margin 0.10

Exit code 0 = both gates passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.evaluation.gates import evaluate_depth_gate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-json", type=Path, required=True, help="evaluate_paper_benchmarks.py output")
    p.add_argument("--benchmark", default="depth/nyu", help="key under 'benchmarks' in --model-json")
    p.add_argument("--baseline-json", type=Path, required=True, help="depth_trivial_baselines.py output")
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    try:
        metrics = model["benchmarks"][args.benchmark]["decoded_metrics"]
    except KeyError as exc:
        raise SystemExit(f"{args.model_json}: no benchmark {args.benchmark!r} ({exc})") from exc
    baselines = json.loads(args.baseline_json.read_text(encoding="utf-8"))["baselines"]

    result = evaluate_depth_gate(metrics, baselines, margin=args.margin)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))
    print(
        f"\nmetric gate:    {'PASS' if result.metric_gate else 'FAIL'} "
        f"(delta1 {result.model_delta1:.3f} vs required {result.required_delta1:.3f} "
        f"= {result.reference_baseline} {result.reference_delta1:.3f} + {result.margin})"
    )
    print(
        f"structure gate: {'PASS' if result.structure_gate else 'FAIL'} "
        f"(aligned delta1 {result.model_aligned_delta1:.3f} vs oracle_image_median "
        f"{result.structure_reference_delta1:.3f})"
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
