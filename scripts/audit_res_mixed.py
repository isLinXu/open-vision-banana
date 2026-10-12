#!/usr/bin/env python3
"""Audit a mixed-seg JSON against official refs, COCO images and a RES release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_vision_banana.res_mixed import audit_mixed_res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-json", type=Path, required=True)
    parser.add_argument("--annotations-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        parser.error("--output-json must be a new file")
    report = audit_mixed_res(
        args.annotations_json, args.annotations_root, args.image_root, args.dataset_root
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
