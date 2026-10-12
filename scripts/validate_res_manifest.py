#!/usr/bin/env python3
"""Fully verify every image, mask, record, source split, and artifact hash."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_vision_banana.res_manifest import validate_res_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_res_manifest(args.dataset_root), indent=2))


if __name__ == "__main__":
    main()
