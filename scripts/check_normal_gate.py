#!/usr/bin/env python3
"""Apply the pre-registered surface-normal gates.

    python scripts/check_normal_gate.py \
        --model-json output/eval/nyu_normal.json --benchmark normal/nyu \
        --baseline-json output/eval/nyu_normal_trivial.json --margin-deg 5

Exit code 0 = both gates passed, 1 = at least one failed.

Metric gate: mean angular error <= best non-oracle baseline − margin (default
5°). Structure gate: mean angular error < oracle_image_mean. Thresholds are
fixed before looking at a model run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.evaluation.gates import evaluate_normal_gate  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-json", type=Path, required=True)
    p.add_argument("--benchmark", default="normal/nyu")
    p.add_argument("--baseline-json", type=Path, required=True)
    p.add_argument("--margin-deg", type=float, default=5.0)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    try:
        metrics = model["benchmarks"][args.benchmark]["decoded_metrics"]
    except KeyError as exc:
        raise SystemExit(f"{args.model_json}: no benchmark {args.benchmark!r} ({exc})") from exc
    baselines = json.loads(args.baseline_json.read_text(encoding="utf-8"))["baselines"]

    result = evaluate_normal_gate(metrics, baselines, margin_deg=args.margin_deg)
    payload = result.to_dict()
    print(json.dumps(payload, indent=2))
    print(
        f"\nmetric gate:    {'PASS' if result.metric_gate else 'FAIL'} "
        f"(mean {result.model_mean_deg:.3f}° vs required {result.required_mean_deg:.3f}° "
        f"= {result.reference_baseline} {result.reference_mean_deg:.3f}° − {result.margin_deg})"
    )
    print(
        f"structure gate: {'PASS' if result.structure_gate else 'FAIL'} "
        f"(mean {result.model_mean_deg:.3f}° vs oracle_image_mean {result.structure_reference_deg:.3f}°)"
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
