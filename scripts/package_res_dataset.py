#!/usr/bin/env python3
"""Package validated RES data into portable Hub upload shards."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.res_bundle import package_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size-mb", type=int, default=256)
    args = parser.parse_args()
    result = package_dataset(args.data_dir, args.output_dir, args.shard_size_mb * 1024**2)
    print(json.dumps({"assets": result["asset_files"], "payloads": len(result["files"])}))


if __name__ == "__main__":
    main()
