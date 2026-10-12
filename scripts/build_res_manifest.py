#!/usr/bin/env python3
"""Build a portable RES corpus from official local RefCOCO annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_vision_banana.res_manifest import build_res_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-mode", choices=("copy",), default="copy")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Pilot limit after full global eval exclusion; official eval remains complete",
    )
    args = parser.parse_args()
    report = build_res_manifest(**vars(args))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "counts",
                    "unique_images",
                    "unique_targets",
                    "dataset_fingerprint",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
