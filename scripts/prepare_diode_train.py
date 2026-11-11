#!/usr/bin/env python3
"""Build clean official DIODE indoor train with native coverage and codec gates.

Use separate output directories per source. Never overwrite an existing build;
--resume requires identical inputs/options and verifies all existing asset bytes.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from open_vision_banana.data.diode_train import (
    VERSION,
    DiodeSource,
    EncodingQualityError,
    convert_pair,
    diode_ids,
    excluded_ids,
    file_sha,
    finalize,
    immutable_write,
    json_bytes,
    prepare_arrays,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("diode",), required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="DIODE train directory or uncompressed TAR",
    )
    parser.add_argument("--official-metadata", type=Path, help="DIODE official diode_meta.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-manifest", action="append", type=Path, default=[])
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly permit incomplete DIODE train source; report is marked partial",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.98,
        help="Native coverage gate, at least 0.98",
    )
    parser.add_argument(
        "--max-encoding-error-m",
        type=float,
        default=0.05,
        help="Maximum valid-pixel RGB roundtrip error in metres, at most 0.05",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 0.98 <= args.min_valid_fraction <= 1:
        parser.error("--min-valid-fraction must be between 0.98 and 1")
    if args.max_encoding_error_m is not None and (
        not math.isfinite(args.max_encoding_error_m) or not 0 <= args.max_encoding_error_m <= 0.05
    ):
        parser.error("--max-encoding-error-m must be finite and between 0 and 0.05")
    if args.source == "diode" and not args.official_metadata:
        parser.error("--official-metadata is required for DIODE")
    import torch

    torch.set_num_threads(1)
    root = args.output_dir
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError("Output directory is nonempty; choose a new directory or --resume")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".build.lock"
    # A hard interruption can leave this lock. Verify no builder is running before
    # manually removing only this lock and resuming. Existing payload is never deleted.
    with lock.open("x") as handle:
        handle.write("Exclusive builder lock. Do not remove while a builder is running.\n")
    source = None
    try:
        config = dict(
            version=VERSION,
            source=args.source,
            input=str(args.input.resolve()),
            input_sha256=file_sha(args.input) if args.input.is_file() else None,
            allow_partial=args.allow_partial,
            min_valid_fraction=args.min_valid_fraction,
            max_encoding_error_m=args.max_encoding_error_m,
            exclude_manifests=[
                dict(path=str(p.resolve()), sha256=file_sha(p)) for p in args.exclude_manifest
            ],
        )
        if args.official_metadata:
            config["official_metadata_sha256"] = file_sha(args.official_metadata)
        immutable_write(root / "build_config.json", json_bytes(config), args.resume)
        excluded = excluded_ids(args.exclude_manifest)
        metadata = json.loads(args.official_metadata.read_text(encoding="utf-8-sig"))
        allowed, heldout = diode_ids(metadata)
        expected = 8574
        if len(allowed) != expected:
            raise ValueError(f"Official indoor metadata must contain {expected} train IDs")
        source = DiodeSource(args.input)
        jobs = source.jobs(allowed, heldout)
        available = len(jobs)
        if available != expected and not args.allow_partial:
            raise ValueError(
                f"Incomplete DIODE source: {available}/{expected}; use --allow-partial explicitly"
            )
        iterator = source.samples(jobs, config["official_metadata_sha256"])
        records, seen, skipped, quality_rejected = [], set(), [], []
        for identity, rgb, depth, mask, provenance in iterator:
            if identity in seen:
                raise ValueError("Duplicate source sample ID")
            seen.add(identity)
            if identity in excluded:
                skipped.append(dict(sample_id=identity, reason="existing_exclusion_manifest"))
                continue
            _, _, stats = prepare_arrays(rgb, depth, mask)
            if stats["valid_fraction"] < args.min_valid_fraction:
                quality_rejected.append(
                    dict(
                        sample_id=identity,
                        reason="explicit_min_valid_fraction",
                        **provenance,
                        **stats,
                    )
                )
                continue
            try:
                records.append(
                    convert_pair(
                        root,
                        identity,
                        rgb,
                        depth,
                        mask,
                        provenance,
                        args.resume,
                        max_encoding_error_m=args.max_encoding_error_m,
                    )
                )
            except EncodingQualityError as exc:
                quality_rejected.append(
                    dict(
                        sample_id=identity,
                        reason="explicit_max_encoding_error",
                        max_encoding_error_m=exc.threshold,
                        **provenance,
                        **exc.stats,
                    )
                )
        if len(seen) != available:
            raise ValueError("Source iteration count mismatch")
        # A changed directory source cannot silently leave old records in a resumed build.
        expected_records = {record["row"]["sample_id"] + ".json" for record in records}
        actual_records = {p.name for p in (root / "records").glob("*.json")}
        if expected_records != actual_records:
            raise ValueError("Unexpected or missing output records")
        immutable_write(root / "exclusions.jsonl", b"".join(map(json_bytes, skipped)), args.resume)
        immutable_write(
            root / "quality_rejections.jsonl",
            b"".join(map(json_bytes, quality_rejected)),
            args.resume,
        )
        report = dict(
            version=VERSION,
            source=args.source,
            expected_official_train_pairs=expected,
            min_valid_fraction=args.min_valid_fraction,
            max_encoding_error_m=args.max_encoding_error_m,
            quality_rejected_pairs=len(quality_rejected),
            quality_rejected_by_reason=dict(
                sorted(Counter(row["reason"] for row in quality_rejected).items())
            ),
            source_available_pairs=available,
            source_complete=available == expected,
            excluded_existing_pairs=len(skipped),
            excluded_sample_ids=[r["sample_id"] for r in skipped],
            unmatched_exclusion_ids=sorted(excluded - seen),
            official_test_or_val_pairs_included=0,
            heldout_exclusion_verification="verified",
            split_verification="official_metadata",
            build_config_sha256=file_sha(root / "build_config.json"),
        )
        print(json.dumps(finalize(root, records, report, args.resume), sort_keys=True))
    finally:
        if source:
            source.close()
        lock.unlink()


if __name__ == "__main__":
    main()
