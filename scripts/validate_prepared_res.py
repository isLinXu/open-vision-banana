#!/usr/bin/env python3
"""Validate generic source-adapter pairs before packaging with package_res_dataset.py."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.res_preparation import validate_prepared_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    print(json.dumps(validate_prepared_pairs(parser.parse_args().dataset_root), indent=2))
