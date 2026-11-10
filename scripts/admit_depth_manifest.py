#!/usr/bin/env python3
"""Run depth admission once over a training manifest and record the result next to it.

Training loaders decode every train depth row in every rank. For a large pool that is
hours per rank before the first step. This admits all rows once with a process pool and
writes <manifest>.depth-admission.json bound to the manifest SHA256, the data root and the
admission version; loaders that find a matching record skip the per-row decode. Any row
failure writes no record.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.data.clean_depth import write_admission_record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--data-root", type=Path, help="Defaults to the manifest directory, as in training"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    root = args.data_root or args.manifest.parent
    print(json.dumps(write_admission_record(args.manifest, root, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
