#!/usr/bin/env python3
"""Compose immutable clean-depth training manifests; never change source datasets."""

import argparse
import importlib.util
import json
from pathlib import Path

# Avoid data.__init__ importing torch in this CPU-only preparation command.
_PATH = Path(__file__).resolve().parents[1] / "src/open_vision_banana/data/clean_depth.py"
_SPEC = importlib.util.spec_from_file_location("clean_depth", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        action="append",
        help="One root per manifest; defaults to manifest directory",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; no overwrite/resume"
    )
    parser.add_argument("--max-invalid-fraction", default="0.02")
    parser.add_argument(
        "--expected-depth-rejections",
        type=int,
        help="Assert measured count, never select a fixed number",
    )
    args = parser.parse_args()
    report = _MODULE.build_clean_manifest(
        args.manifest,
        args.output,
        roots=args.data_root,
        max_invalid_fraction=args.max_invalid_fraction,
        expected_depth_rejections=args.expected_depth_rejections,
    )
    print(json.dumps(report["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
