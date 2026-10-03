"""CLI for explicit plans and benchmark runs; no implicit model downloads."""
import argparse
import json
import sys

from .config import load_config
from .runner import plan_benchmark, run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark: models x tasks x sizes")
    parser.add_argument("--config", required=True, help="JSON profile; relative paths resolve beside it")
    parser.add_argument("--dataset-root", help="Override image dataset root")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="MODEL_ID=PATH")
    parser.add_argument("--device", help="cpu, cuda or cuda:N; no implicit fallback")
    parser.add_argument("--output", help="New empty directory for results and artifacts")
    parser.add_argument("--plan", action="store_true", help="Print matrix without importing models/torch")
    parser.add_argument("--allow-download", action="store_true", help="Explicit opt-in to model downloads")
    parser.add_argument("--allow-legacy", action="store_true", help="Unsupported old runtime: qualitative components only")
    args = parser.parse_args(argv)
    try:
        overrides = {}
        for item in args.checkpoint:
            key, separator, value = item.partition("=")
            if not separator or not key or not value or key in overrides:
                raise ValueError("--checkpoint requires a unique MODEL_ID=PATH")
            overrides[key] = value
        config = load_config(args.config, args.dataset_root, args.device, overrides)
        if args.plan:
            print(json.dumps(plan_benchmark(config), indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        if not args.output:
            parser.error("--output is required for an actual run")
        report = run_benchmark(config, args.output, args.allow_download, args.allow_legacy)
        print("Completed {} jobs. Results: {}/results.json".format(len(report["jobs"]), args.output))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print("Benchmark failed: {}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
