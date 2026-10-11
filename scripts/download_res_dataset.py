#!/usr/bin/env python3
"""Download a pinned RES release and restore its training manifest layout."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.res_bundle import FORMAT, restore_bundle, safe_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id")
    parser.add_argument("--revision", help="Immutable Hub commit SHA (required for Hub downloads)")
    parser.add_argument(
        "--bundle-dir", type=Path, help="Alternatively restore an existing local bundle"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.bundle_dir is not None:
        if args.repo_id or args.revision:
            parser.error("Use either --bundle-dir or --repo-id/--revision")
        bundle = args.bundle_dir
    else:
        if not args.repo_id or not args.revision:
            parser.error("Hub downloads require --repo-id and --revision")
        if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
            parser.error("--revision must be a 40-character immutable commit SHA")
        from huggingface_hub import hf_hub_download, snapshot_download

        index_path = hf_hub_download(
            args.repo_id, "bundle.json", repo_type="dataset", revision=args.revision
        )
        index = json.loads(Path(index_path).read_text(encoding="utf-8"))
        if index.get("format") != FORMAT:
            raise ValueError("Unsupported RES bundle format")
        for item in index["files"]:
            safe_path(Path.cwd(), item["path"])
        bundle = Path(
            snapshot_download(
                args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=["bundle.json", *[e["path"] for e in index["files"]]],
                local_dir=args.output_dir.parent
                / (args.output_dir.name + ".bundle-" + args.revision[:8]),
            )
        )
    print(json.dumps(restore_bundle(bundle, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
